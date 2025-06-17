import deepl

# === 替換為你的 DeepL API 金鑰 ===
auth_key = ""  # 例如：'abc123-xyz...'
translator = deepl.Translator(auth_key)

# === 檔案設定 ===
input_path = "什么是机器学习？.vtt"
output_path = "什么是机器学习？_中英雙語.srt"

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

output_lines = []
counter = 1
buffer = []
time_line = ""

for i, line in enumerate(lines):
    stripped = line.strip()

    if stripped.startswith("WEBVTT") or stripped == "":
        continue
    elif "-->" in stripped:
        # .vtt 使用 '.' 表示毫秒，要改為 ','
        time_line = stripped.replace(".", ",")
    else:
        buffer.append(stripped)
        # 若下一行是時間或是檔案結尾，表示這段字幕結束
        next_line_is_time = i + 1 < len(lines) and "-->" in lines[i + 1]
        end_of_file = i == len(lines) - 1
        if next_line_is_time or end_of_file:
            english_text = " ".join(buffer)
            try:
                zh = translator.translate_text(english_text, target_lang="ZH")
                output_lines.append(f"{counter}\n")
                output_lines.append(f"{time_line}\n")
                output_lines.append(f"{english_text}\n{zh.text}\n\n")
            except Exception as e:
                output_lines.append(f"{counter}\n")
                output_lines.append(f"{time_line}\n")
                output_lines.append(f"{english_text}\n[翻譯失敗]\n\n")
            counter += 1
            buffer = []

with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(output_lines)

print(f"✅ 已產出：{output_path}")
