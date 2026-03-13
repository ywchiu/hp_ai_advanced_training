# Enterprise AI Development: Advanced Training

## Day1

### Demo 20260312
- https://colab.research.google.com/drive/180InGwBBjtoFGBS64fKoouOxRu8-ryxx?usp=sharing

### Demo 20250626
- [https://colab.research.google.com/github/ywchiu/hp_ai_advanced_training/blob/main/Demo20250626.ipynb](https://colab.research.google.com/github/ywchiu/hp_ai_advanced_training/blob/main/Demo20250626.ipynb)

### Name Prediction With ML
- https://colab.research.google.com/github/ywchiu/hp_ai_advanced_training/blob/main/day1_name_prediction.ipynb

### NLP - Generative AI 
- https://colab.research.google.com/github/ywchiu/hp_ai_advanced_training/blob/main/day1_generative_ai_nlp.ipynb

### NLP - Discriminative AI 
- https://colab.research.google.com/github/ywchiu/hp_ai_advanced_training/blob/main/day1_discriminative_ai_nlp.ipynb


## Day2

### Demo 20260313
- https://colab.research.google.com/drive/1gFNreZ4rrkl6iwE1h9T3w1YemH3kX7Mt?usp=sharing

### Demo 20250627
- https://colab.research.google.com/drive/1f5wrenIqrinkUKf0obfGcLAxoqMZA12N?usp=sharing

### Gemma 3 Finetune
- https://colab.research.google.com/github/ywchiu/hp_ai_advanced_training/blob/main/day2_gemma3_sft.ipynb

### Opal Translation Demo Video
- https://youtu.be/cSEgf-M51Z8

### Make Translation Demo Video
- https://youtu.be/DwE6z7IbpMg

### AI Agent
- https://colab.research.google.com/github/ywchiu/hp_ai_advanced_training/blob/main/day2_ai_agent.ipynb
- https://colab.research.google.com/github/ywchiu/hp_ai_advanced_training/blob/main/day2_hp_multi_agent.ipynb

### RAG
- https://colab.research.google.com/github/ywchiu/hp_ai_advanced_training/blob/main/day2_rag.ipynb

### 練習題
以下是一個以文字方式完整描述的練習題指引，最後明確使用 `Update a Row` 模組將分析結果寫回 B 欄：

---

## 📘 練習題：Opal + Google Sheets + OpenAI 情緒分析流程

**目標**

* 從 Google Sheets 讀取 hp\_comment\_make.xlsx 留言
* 使用 OpenAI 進行情緒判斷（正向 / 中立 / 負向）
* 將結果寫回同一Sheets的 B 欄

---

### 1. 前置設定

1. 將 `hp_comment_make.xlsx` 上傳至 Google Drive，並開啟成 Google Sheets，確保 A 欄為留言內容，B 欄空白。
2. 在 Opal 建立新 Scenario

---

### 2. 模組串接步驟

#### (1) **Google Sheets – Search Rows**

* 選擇剛才的試算表與工作表（sheet）。
* 設定篩選條件：`A != ""`，確保只抓取有留言的行數。

#### (2) **OpenAI – Create Chat Completion**

* 模型選 `gpt-5`（或可用型號）。
* Prompt 範例 ↓

  ```
  你是一個情緒分析機器人，請單純回傳：  
  正向 / 中立 / 負向  
  文字內容：
  ```

#### (3) **Google Sheets – Update a Row**

* 使用上一步的 `Row Number`定義要更新哪一行。
* 設定僅更新 B 欄的情緒結果。

