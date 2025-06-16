# merge_vtt_to_bilingual_srt.py
# 新策略：先將字幕片段按句子合併，再進行翻譯，從根源上避免重複。

import re
from tqdm import tqdm
from opencc import OpenCC
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local

# --- 全域設定 ---
MODEL_NAME = "facebook/nllb-200-distilled-600M"  # 翻譯模型
INPUT_VTT = "subtitles-en.vtt"  # 輸入的 VTT 檔案
OUTPUT_SRT = "translated_bilingual_semantic.srt"  # 輸出的雙語 SRT 檔案
MAX_SEGMENTS = None  # 限制處理的字幕數量，None 表示不限制
THREADS = 4  # 使用的線程數

# --- 初始化工具 ---
cc = OpenCC("s2twp")  # 簡轉繁（台灣正體）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 為每個線程創建獨立的 pipeline 實例以確保線程安全
_thread_ctx = local()


def get_translator():
    """為當前線程獲取或創建一個獨立的翻譯器 pipeline。"""
    if not hasattr(_thread_ctx, "translator"):
        _thread_ctx.translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang="eng_Latn",  # 來源語言：英文
            tgt_lang="zho_Hant",  # 目標語言：繁體中文
            max_length=1024,  # 增加長度以處理合併後的長句
        )
    return _thread_ctx.translator


# --- 核心功能函數 ---


def parse_vtt(path):
    """解析 VTT 格式的字幕檔。"""
    entries = []
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # VTT 檔案以雙換行符分隔每個字幕塊，先移除 WEBTT 標頭
    content = re.sub(r"WEBVTT.*\n", "", content).strip()
    blocks = re.split(r"\n\n+", content)

    for i, block in enumerate(blocks):
        lines = block.strip().splitlines()
        if not lines:
            continue

        # 處理時間戳行
        if "-->" in lines[0]:
            time_line_index = 0
        elif len(lines) > 1 and "-->" in lines[1]:
            time_line_index = 1
        else:
            continue  # 無效的塊

        start, end = lines[time_line_index].split(" --> ")
        text = " ".join(l.strip() for l in lines[time_line_index + 1 :] if l.strip())

        if text:
            entries.append(
                {"index": str(i + 1), "start": start, "end": end, "text": text}
            )

    return entries[:MAX_SEGMENTS]


def group_entries_by_sentence(entries):
    """
    【新核心邏輯】將字幕片段按完整句子分組。
    這是解決重複翻譯問題的關鍵。
    """
    grouped_entries = []
    current_group = []

    if not entries:
        return []

    for entry in entries:
        current_group.append(entry)
        # 判斷句尾：如果文字以句號、問號、驚嘆號結尾，或包含換行符（通常意味著段落結束）
        # 則認為這是一個句子的結束。
        if entry["text"].strip().endswith((".", "?", "!", "。", "？", "！")):
            # 合併當前組
            combined_text = " ".join([e["text"] for e in current_group])
            new_entry = {
                "start": current_group[0]["start"],
                "end": current_group[-1]["end"],
                "text": combined_text,
                "zh": "",  # 預留給翻譯結果
            }
            grouped_entries.append(new_entry)
            current_group = []  # 清空，準備下一組

    # 處理最後一組（如果文件結尾不是完整句子）
    if current_group:
        combined_text = " ".join([e["text"] for e in current_group])
        new_entry = {
            "start": current_group[0]["start"],
            "end": current_group[-1]["end"],
            "text": combined_text,
            "zh": "",
        }
        grouped_entries.append(new_entry)

    return grouped_entries


def translate_entry(entry):
    """單一（已分組的）字幕條目的翻譯流程。"""
    try:
        # 直接翻譯合併後的文本
        translated_text = get_translator()(entry["text"])[0]["translation_text"]
        entry["zh"] = cc.convert(translated_text.strip())
        return entry
    except Exception as e:
        print(f"Error translating entry: {entry['text']}. Error: {e}")
        entry["zh"] = f"[翻譯失敗] {entry['text']}"
        return entry


def translate_all(grouped_entries):
    """使用多線程並行翻譯所有分組後的字幕條目。"""
    results = []
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        # 直接提交分組後的條目
        futures = [executor.submit(translate_entry, entry) for entry in grouped_entries]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="句子分組翻譯中"
        ):
            results.append(future.result())

    # 保持原始順序
    results.sort(key=lambda e: e["start"])
    return results


def write_srt(entries, path):
    """將處理好的字幕條目寫入 SRT 檔案。"""
    with open(path, "w", encoding="utf-8") as f:
        for i, e in enumerate(entries):
            # SRT 格式要求時間戳中的小數點用逗號
            start = e["start"].replace(".", ",")
            end = e["end"].replace(".", ",")

            # 寫入序號、時間戳、中文翻譯和原始英文
            f.write(f"{i + 1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{e['zh']}\n")
            f.write(f"{e['text']}\n\n")


# --- 主執行流程 ---
if __name__ == "__main__":
    print("1. 開始解析 VTT 檔案...")
    entries = parse_vtt(INPUT_VTT)
    print(f"   解析完成，共 {len(entries)} 條原始字幕片段。")

    print("2. 開始按句子合併字幕片段...")
    grouped_entries = group_entries_by_sentence(entries)
    print(f"   合併完成，共形成 {len(grouped_entries)} 個句子組進行翻譯。")

    print("3. 開始進行多線程翻譯...")
    translated_entries = translate_all(grouped_entries)

    print(f"4. 開始寫入雙語 SRT 檔案至 {OUTPUT_SRT}...")
    write_srt(translated_entries, OUTPUT_SRT)

    print(f"✅ 全部完成！輸出檔案：{OUTPUT_SRT}")
