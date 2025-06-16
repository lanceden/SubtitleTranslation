# subtitle_semantic_translator.py
# 多線程 + 上下文翻譯 + 語意對位裁切（避免重複字幕）

import re
from tqdm import tqdm
from opencc import OpenCC
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_NAME = "facebook/nllb-200-distilled-600M"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INPUT_VTT = "subtitles-en.vtt"
OUTPUT_VTT = "translated_bilingual_semantic.vtt"
MAX_SEGMENTS = 15
THREADS = 4

cc = OpenCC("s2twp")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="eng_Latn",
    tgt_lang="zho_Hant",
    max_length=512,
)
embedder = SentenceTransformer(EMBED_MODEL)


def parse_vtt(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            index = lines[0].strip()
            start, end = lines[1].split(" --> ")
            text = " ".join(l.strip() for l in lines[2:] if l.strip())
            entries.append({"index": index, "start": start, "end": end, "text": text})
    return entries[:MAX_SEGMENTS]


def get_context(i, entries):
    prev = entries[i - 1]["text"] if i > 0 else ""
    curr = entries[i]["text"]
    next = entries[i + 1]["text"] if i < len(entries) - 1 else ""
    return f"{prev} {curr} {next}", curr


def align_semantic(en_anchor, zh_block):
    en_embedding = embedder.encode(en_anchor, convert_to_tensor=True)
    zh_splits = re.split(r"[。！？!?]", zh_block)
    zh_embeddings = embedder.encode(zh_splits, convert_to_tensor=True)
    sim_scores = util.cos_sim(en_embedding, zh_embeddings)[0]
    best = int(sim_scores.argmax())
    return cc.convert(zh_splits[best].strip())


def translate_entry(i, entries):
    try:
        context, anchor = get_context(i, entries)
        zh_block = translator(context)[0]["translation_text"]
        aligned = align_semantic(anchor, zh_block)
        return i, aligned
    except Exception as e:
        return i, f"[翻譯失敗] {e}"


def translate_all(entries):
    results = [None] * len(entries)
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {
            executor.submit(translate_entry, i, entries): i for i in range(len(entries))
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="語意翻譯中"
        ):
            i, zh = future.result()
            results[i] = {**entries[i], "zh": zh}
    return results


def write_vtt(entries, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for e in entries:
            f.write(f"{e['index']}\n{e['start']} --> {e['end']}\n")
            f.write(f"{e['zh']}\n{e['text']}\n\n")


if __name__ == "__main__":
    entries = parse_vtt(INPUT_VTT)
    translated = translate_all(entries)
    write_vtt(translated, OUTPUT_VTT)
    print(f"✅ 完成翻譯輸出：{OUTPUT_VTT}")
