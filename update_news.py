# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timezone, timedelta
import feedparser
import google.generativeai as genai
import os
import re
import numpy as np

CSV_FILEPATH = '2021-2025每週新聞及情緒分析.csv'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- 在這裡定義你感興趣的新聞標題關鍵字 ---
# 如果列表為空，則抓取所有新聞標題 (與之前行為一致)
TARGET_NEWS_KEYWORDS = [
    # 台股相關
    "台股", "加權指數", "櫃買指數", "台積電", "鴻海", "聯發科", "大盤",
    "電子", "金融", "傳產", "半導體", "AI", "法人", "外資", "投信",
    # 美股相關
    "美股", "道瓊", "納斯達克", "標普500", "費半", "科技", "蘋果", "微軟",
    "輝達", "Nvidia", "特斯拉", "Google", "Amazon", "Meta", "聯準會", "Fed",
    "升息", "降息", "通膨", "CPI", "PPI", "非農", "就業", "財報",
    # 川普相關
    "川普", "特朗普", "美國大選", "貿易戰", "關稅",
    # 宏觀經濟與政策 (可能影響股市)
    "經濟成長", "衰退", "GDP", "油價", "金價", "匯率", "地緣政治","股市"
]

# --- Gemini 設定 ---
safety_settings_gemini = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        sentiment_model = genai.GenerativeModel("models/gemini-1.5-flash-latest", safety_settings=safety_settings_gemini)
        print("Gemini sentiment_model configured successfully.")
    else:
        print("GEMINI_API_KEY not found. Gemini model will not be configured.")
        sentiment_model = None
except Exception as gemini_err:
    print(f"Failed to configure Gemini sentiment_model: {gemini_err}")
    sentiment_model = None

def get_this_weeks_news_titles_so_far(target_keywords=None):
    """
    嘗試獲取本週從週一到今天為止，Yahoo RSS Feed 中包含的新聞標題。
    可以根據 target_keywords 進行篩選。
    注意：RSS Feed 的內容有時效性，可能無法獲取到本週早期的所有新聞。
    """
    if target_keywords is None:
        target_keywords = []

    all_week_titles = []
    seen_titles = set() # 用於去重，因為同一新聞可能在feed中停留多天

    today_local_date = datetime.now(timezone.utc).astimezone().date()
    start_of_week_date = today_local_date - timedelta(days=today_local_date.weekday())  # 本週一

    print(f"正在嘗試抓取從 {start_of_week_date.strftime('%Y/%m/%d')} 到 {today_local_date.strftime('%Y/%m/%d')} 的新聞...")
    if target_keywords:
        print(f"使用關鍵字: {', '.join(target_keywords)}")
    else:
        print("不使用特定關鍵字，抓取所有新聞。")


    feed = feedparser.parse("https://tw.news.yahoo.com/rss/")
    if not feed.entries:
        print("  無法從 Yahoo RSS Feed 獲取任何新聞條目。")
        return []

    print(f"  已從 RSS Feed 獲取 {len(feed.entries)} 條新聞，開始篩選本週新聞...")

    for entry in feed.entries:
        try:
            published_dt_utc = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')
            published_dt_local = published_dt_utc.astimezone()
            published_date_local = published_dt_local.date()

            if start_of_week_date <= published_date_local <= today_local_date:
                title = entry.title.strip()
                if title in seen_titles:
                    continue
                
                if target_keywords:
                    if any(keyword.lower() in title.lower() for keyword in target_keywords):
                        all_week_titles.append(f"({published_date_local.strftime('%m/%d')}) {title}")
                        seen_titles.add(title)
                else:
                    all_week_titles.append(f"({published_date_local.strftime('%m/%d')}) {title}")
                    seen_titles.add(title)
        except Exception as e:
            # print(f"  處理新聞條目時出錯: {e} (條目: {entry.get('title', 'N/A')})")
            continue
    
    if all_week_titles:
        print(f"  已收集到本週內 {len(all_week_titles)} 條相關新聞標題。")
    else:
        print(f"  未能在 RSS Feed 中找到符合條件的本週新聞標題 (關鍵字: {target_keywords if target_keywords else '無'})。")
    
    def sort_key(item_title):
        match = re.match(r'\((\d{2}/\d{2})\)', item_title)
        if match:
            return match.group(1)
        return "99/99"

    all_week_titles.sort(key=sort_key)
    return all_week_titles


def get_sentiment_from_gemini(news_titles: list, week_key_str: str, few_shot_examples=None, keywords_used=None):
    if not sentiment_model:
        print("Gemini sentiment_model is not configured. Cannot get sentiment.")
        return None, "Gemini模型未配置"
    if not news_titles:
        return None, "新聞標題列表為空"
    
    example_prompt_part = ""
    if few_shot_examples:
        example_prompt_part = "以下是一些歷史評分範例供您參考：\n"
        for ex_date, ex_score, ex_summary in few_shot_examples:
            ex_score_display = ex_score if pd.notna(ex_score) else "N/A"
            ex_summary_display = ex_summary if pd.notna(ex_summary) else "N/A"
            example_prompt_part += f"- 週期間：{ex_date}；情緒分數：{ex_score_display}；摘要：{ex_summary_display}\n"
        example_prompt_part += "\n"

    news_titles_str = "\n".join([f"- {title}" for title in news_titles])
    
    keyword_info = ""
    if keywords_used and len(keywords_used) > 0 : # 確保 keywords_used 非空
        keyword_info = f"新聞標題已根據關鍵字 '{', '.join(keywords_used)}' 進行篩選。\n"

    prompt = f"""
    請根據以下 '{week_key_str}' 這一週內（到目前為止）的台灣Yahoo新聞標題，綜合評估當週的金融與社會情緒。
    {keyword_info}
    請給出一個介於0到100之間的情緒分數（0代表極度悲觀/恐慌，50代表中性，100代表極度樂觀/狂熱）。
    並請總結出1-3條最重要的當週焦點作為「重大新聞摘要」。
    {example_prompt_part}
    本週新聞標題列表（可能包含日期標記）：
    {news_titles_str}

    輸出格式：
    1. 情緒分數：[一個0到100的整數]
    2. 重大新聞摘要：[1-3條簡潔摘要，用分號隔開]

    僅回覆以上格式。
    """
    try:
        print(f"  發送 {len(news_titles)} 條新聞到 Gemini，代表週：{week_key_str}...")
        response = sentiment_model.generate_content(prompt)
        if not response.candidates:
            print("  Gemini API 無回應。")
            return None, "Gemini API無返回內容"
        
        content = response.text
        score_match = re.search(r"情緒分數：\s*(\d+)", content)
        summary_match = re.search(r"重大新聞摘要：\s*(.+)", content, re.DOTALL)
        
        sentiment_score_val = int(score_match.group(1)) if score_match and score_match.group(1).isdigit() else None
        news_summary_val = summary_match.group(1).strip() if summary_match else "未能生成摘要"
        
        news_summary_val = re.sub(r'\n\s*-\s*', '；', news_summary_val)
        news_summary_val = re.sub(r'^\s*-\s*', '', news_summary_val)
        news_summary_val = news_summary_val.replace('\n', ' ').replace(';', '；').strip()
        news_summary_val = re.sub(r'\s*；\s*', '；', news_summary_val)
        news_summary_val = re.sub(r'；$', '', news_summary_val)

        print(f"  已解析 ({week_key_str}): 分數={sentiment_score_val}, 摘要='{news_summary_val[:100]}...'")
        return sentiment_score_val, news_summary_val
    except Exception as e:
        print(f"  Gemini API 調用或解析時出錯 ({week_key_str}): {e}")
        print(f"  Gemini raw response: {response.text if 'response' in locals() and hasattr(response, 'text') else 'N/A'}")
        return None, f"Gemini API調用或解析錯誤: {str(e)}"

def get_few_shot_examples(csv_filepath, num_examples=3):
    try:
        df = pd.read_csv(csv_filepath, encoding='utf-8-sig')
        if df.empty or '情緒分數' not in df.columns or '重大新聞摘要' not in df.columns:
            return []
        df_valid_examples = df.dropna(subset=['情緒分數', '重大新聞摘要'])
        df_to_sample = df_valid_examples.tail(num_examples) if len(df_valid_examples) >= num_examples else df.tail(num_examples)
        examples = []
        for _, row in df_to_sample.iterrows():
            date_val = str(row.get('年/週', 'N/A')).strip()
            score_val = row.get('情緒分數', np.nan)
            summary_val = str(row.get('重大新聞摘要', 'N/A')).strip()
            examples.append((date_val, score_val, summary_val))
        return examples
    except Exception as e:
        print(f"讀取 few-shot 範例時出錯: {e}")
        return []

def get_current_week_key():
    today = datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return f"{start_of_week.strftime('%Y/%m/%d')}-{end_of_week.strftime('%Y/%m/%d')}"

def update_sentiment_csv(csv_filepath, target_keywords=None):
    if not sentiment_model:
        print("Gemini sentiment model not loaded. Exiting update_sentiment_csv.")
        return

    current_week_key = get_current_week_key()
    print(f"本週的鍵值為: {current_week_key}")

    news_titles = get_this_weeks_news_titles_so_far(target_keywords)
    
    if not news_titles:
        print(f"本週 ({current_week_key}) 到目前為止，未能從 RSS Feed 獲取到符合條件的新聞標題。")
        return

    few_shot_examples = get_few_shot_examples(csv_filepath, num_examples=5)
    score, summary = get_sentiment_from_gemini(news_titles, current_week_key, few_shot_examples, keywords_used=target_keywords)

    if score is not None:
        try:
            if os.path.exists(csv_filepath):
                df = pd.read_csv(csv_filepath, encoding='utf-8-sig')
            else:
                df = pd.DataFrame(columns=['年/週', '情緒分數', '重大新聞摘要'])

            week_exists = df['年/週'].astype(str).str.strip() == current_week_key.strip()
            
            if week_exists.any():
                print(f"更新本週 ({current_week_key}) 的情緒分數與摘要...")
                idx = df[week_exists].index[0]
                df.loc[idx, '情緒分數'] = score
                df.loc[idx, '重大新聞摘要'] = summary
            else:
                print(f"新增本週 ({current_week_key}) 的情緒分數與摘要...")
                new_row = pd.DataFrame([{'年/週': current_week_key, '情緒分數': score, '重大新聞摘要': summary}])
                df = pd.concat([df, new_row], ignore_index=True)
            
            df.drop_duplicates(subset=['年/週'], keep='last', inplace=True)
            df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
            print(f"已成功將 {current_week_key} 的資料寫入/更新到 CSV！")

        except Exception as e:
            print(f"寫入 CSV 時出錯: {e}")
    else:
        print(f"本週 ({current_week_key}) 未能取得分數：{summary}")

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("錯誤：GEMINI_API_KEY 環境變數未設置。腳本無法執行 Gemini API 調用。")
    else:
        if not os.path.exists(CSV_FILEPATH):
            print(f"'{CSV_FILEPATH}' 不存在，創建一個空的範例檔案...")
            pd.DataFrame(columns=['年/週', '情緒分數', '重大新聞摘要']).to_csv(CSV_FILEPATH, index=False, encoding='utf-8-sig')
            print(f"範例 CSV '{CSV_FILEPATH}' 已創建。")
        
        update_sentiment_csv(CSV_FILEPATH, target_keywords=TARGET_NEWS_KEYWORDS)