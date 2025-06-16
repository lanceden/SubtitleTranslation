# 🎬 VTT to Bilingual SRT Translator

本專案為一個 Python 腳本，用於將英文 `.vtt` 字幕檔轉換為 **中英雙語 `.srt` 檔案**，並透過 **NLLB 多語翻譯模型**將英文句子翻譯為繁體中文。具備**多線程翻譯加速、句子合併消除重複翻譯**等特色，適合用於字幕翻譯、教學資源本地化等場景。

---

## 📦 功能特點

- ✅ 自動解析 VTT 字幕並重組為句子單位
- 🌐 使用 [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) 進行英文 ➜ 繁體中文翻譯
- 🔁 支援多線程翻譯，提升處理速度
- 🔧 自動轉換時間戳與 SRT 格式輸出
- ✨ 輸出結果為：中英雙語對照 `.srt` 檔案

---

## 🧰 安裝需求

請先安裝以下 Python 套件：

```bash
pip install opencc-python-reimplemented transformers tqdm
