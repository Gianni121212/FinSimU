# sentiment_engine.py (修正版)
import pandas as pd
from datetime import datetime, timezone, timedelta
import feedparser
import google.generativeai as genai
import os
import re
import numpy as np
import urllib.parse
import time
import random
import pymysql

# ==============================================================================
# --- 全局設定 ---
# ==============================================================================
MAX_TOTAL_HEADLINES = 150
MAX_HEADLINES_PER_TOPIC = 8
CSV_FILEPATH = '2021-2025每週新聞及情緒分析.csv'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TARGET_COMPANIES_AND_TOPICS = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Google": "GOOGL",
    "Amazon": "AMZN", "Meta": "META", "Tesla": "TSLA", "S&P 500": None,
    "Nasdaq": None, "Dow Jones": None, "Federal Reserve": "Fed", "inflation": "CPI",
    "jobs report": "nonfarm payrolls", "interest rates": None, "crude oil": "WTI",
    "US election": None, "trade war": "tariffs", "Trump": "tariffs",
}

# ==============================================================================
# --- 資料庫連線 ---
# ==============================================================================
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME", "finsimu_db"),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}
def execute_db_query(query, args=None, commit=False, executemany=False):
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            if executemany: rowcount = cursor.executemany(query, args)
            else: rowcount = cursor.execute(query, args)
            if commit: conn.commit(); return rowcount
            else: return cursor.fetchall()
    except Exception as e:
        print(f"[DB_ERROR] in sentiment_engine: {e}")
        if conn and commit: conn.rollback()
        return None
    finally:
        if conn and conn.open: conn.close()

# ==============================================================================
# --- 模型初始化 ---
# ==============================================================================
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

sentiment_model_gemini = None
if GEMINI_API_KEY:
    try:
        # ## 修正 ##: 使用新的 Gemini 初始化方式
        # genai.configure(api_key=GEMINI_API_KEY) # 舊的方式
        safety_settings = [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        sentiment_model_gemini = genai.GenerativeModel("models/gemini-2.0-flash-latest", safety_settings=safety_settings)
        print("[sentiment_engine.py] Gemini sentiment_model_gemini configured successfully.")
    except Exception as gemini_err:
        print(f"[sentiment_engine.py] Failed to configure Gemini sentiment_model_gemini: {gemini_err}")
else:
    print("[sentiment_engine.py] GEMINI_API_KEY not found in environment variables for sentiment engine.")

finbert_tokenizer = None
finbert_model = None

# ==============================================================================
# --- 核心功能函式 ---
# ==============================================================================
def load_finbert_model():
    global finbert_tokenizer, finbert_model
    if finbert_model is None and FINBERT_AVAILABLE:
        try:
            model_name = "ProsusAI/finbert"
            finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as e: return False
    return finbert_model is not None

def analyze_titles_with_finbert(titles: list):
    if not load_finbert_model(): return []
    analyzed_results = []
    for title in titles:
        try:
            inputs = finbert_tokenizer(title, padding=True, truncation=True, return_tensors='pt')
            outputs = finbert_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction_idx = torch.argmax(probabilities).item()
            label = finbert_model.config.id2label[prediction_idx]
            score = probabilities[0][0].item() - probabilities[0][1].item()
            analyzed_results.append({"title": title, "label": label, "score": score, "formatted_string": f"[{label.upper()}, Score: {score:+.2f}] {title}"})
        except Exception: pass
    return analyzed_results

def get_daily_english_news(target_topics: dict):
    real_today = datetime.now(timezone.utc)
    real_start_date = real_today - timedelta(hours=120) 
    seen_titles, total_headlines = set(), 0
    for company_or_topic, ticker_or_keyword in target_topics.items():
        if total_headlines >= MAX_TOTAL_HEADLINES: break
        query = f'"{company_or_topic}" {ticker_or_keyword or ""} stock'
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(query)}&hl=en-US&gl=US&ceid=US:en&tbs=qdr:d2"
        try:
            feed = feedparser.parse(url)
            if feed.entries:
                for entry in feed.entries:
                    if total_headlines >= MAX_TOTAL_HEADLINES: break
                    try:
                        published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                        if real_start_date <= published_dt <= real_today and entry.title.strip() not in seen_titles:
                            seen_titles.add(entry.title.strip())
                            total_headlines += 1
                    except Exception: continue
        except Exception: pass
        time.sleep(random.uniform(0.3, 0.8))
    return list(seen_titles), f"{real_start_date.strftime('%Y-%m-%d')} to {real_today.strftime('%Y-%m-%d')}"

def get_macro_sentiment_from_gemini(analyzed_news_strings: list, week_key_for_csv: str, real_news_date_range: str):
    if not sentiment_model_gemini or not analyzed_news_strings: return None, "Model or news not available"
    prompt = f"""Target week: {week_key_for_csv}. Based on real news from {real_news_date_range}, analyze the following headlines and provide a sentiment score and summary.
    News: {"; ".join(analyzed_news_strings)}
    Required Output:
    Sentiment Score: [integer 0-100]
    News Summary (Traditional Chinese): [concise summary]
    """
    try:
        response = sentiment_model_gemini.generate_content(prompt, request_options={'timeout': 120})
        content = response.text
        score_match = re.search(r"Sentiment Score:\s*(\d+)", content, re.I)
        summary_match = re.search(r"News Summary \(Traditional Chinese\):\s*(.+)", content, re.I | re.DOTALL)
        score = int(score_match.group(1)) if score_match else 50
        summary = summary_match.group(1).strip().replace('\n', ' ') if summary_match else "未能生成摘要"
        return score, summary
    except Exception as e:
        return None, f"Gemini API Error: {e}"

def update_weekly_csv(week_key, score, summary):
    if score is None: return
    try:
        df = pd.read_csv(CSV_FILEPATH, encoding='utf-8-sig') if os.path.exists(CSV_FILEPATH) else pd.DataFrame(columns=['年/週', '情緒分數', '重大新聞摘要'])
        df['年/週'] = df['年/週'].astype(str).str.strip()
        if week_key in df['年/週'].values:
            df.loc[df['年/週'] == week_key, ['情緒分數', '重大新聞摘要']] = [score, summary]
        else:
            df = pd.concat([df, pd.DataFrame([{'年/週': week_key, '情緒分數': score, '重大新聞摘要': summary}])], ignore_index=True)
        df.to_csv(CSV_FILEPATH, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"CSV Update Error: {e}")

# ==============================================================================
# --- 主流程函式 (供外部調用) ---
# ==============================================================================
def run_sentiment_pipeline():
    if not FINBERT_AVAILABLE or not sentiment_model_gemini: return
    raw_titles, date_range = get_daily_english_news(TARGET_COMPANIES_AND_TOPICS)
    if not raw_titles: return
    
    finbert_results = analyze_titles_with_finbert(raw_titles)
    if not finbert_results: return
    
    today = datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    csv_week_key = f"{start_of_week.strftime('%Y/%m/%d')}-{end_of_week.strftime('%Y/%m/%d')}"
    
    macro_score, macro_summary = get_macro_sentiment_from_gemini([res["formatted_string"] for res in finbert_results], csv_week_key, date_range)
    update_weekly_csv(csv_week_key, macro_score, macro_summary)
    
    execute_db_query("DELETE FROM daily_news_sentiment", commit=True)
    db_data = [(res["title"], res["label"], res["score"], datetime.now(timezone.utc)) for res in finbert_results]
    if db_data:
        execute_db_query("INSERT INTO daily_news_sentiment (title, sentiment_label, sentiment_score, fetched_at) VALUES (%s, %s, %s, %s)", db_data, commit=True, executemany=True)

if __name__ == "__main__":
    run_sentiment_pipeline()