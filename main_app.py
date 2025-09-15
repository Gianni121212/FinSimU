# program_merged.py - 整合版 AI 策略分析與市場分析平台
# 整合了：
# 1. 來自 program.py 的市場分析儀表板、Gemini AI 新聞分析、個股深度報告和自動化排程任務
# 2. 來自 stock_ga_web.py 的使用者認證系統、策略訓練器、手動回測和策略清單功能

import os
import logging
import re
import json
from datetime import datetime, timedelta, date
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv
# === 從 stock_ga_web.py 移植：使用者認證與資料庫相關模組 ===
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import traceback

# Google Gemini AI
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# 資料庫連接
import pymysql

# 🆕 新增：排程、時區、時間與追蹤模組
from apscheduler.schedulers.background import BackgroundScheduler
import pytz
import atexit
import time
import traceback

# ==============================================================================
# >>> (新整合) 新聞情緒分析所需模組 <<<
# ==============================================================================
import feedparser
import urllib.parse
import random
import queue
import threading
import uuid

# FinBERT 相關函式庫 (軟性依賴)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 建立Flask應用
app = Flask(__name__)
CORS(app)

# 【新增程式碼 START】
# --- 簡單的內建任務佇列系統 ---
# 1. 任務佇列: 存放待處理的訓練任務
task_queue = queue.Queue()

# 2. 結果字典: 用於儲存任務的狀態和最終結果
#    鍵是 task_id，值是包含 status 和 result 的字典
task_results = {}

# 3. 執行緒鎖: 保護 task_results 在多執行緒環境下的讀寫安全
results_lock = threading.Lock()
# 【新增程式碼 END】

# === 從 stock_ga_web.py 移植：Flask-Login 設定 ===
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'
login_manager.login_message = "請先登入以訪問此頁面。"
login_manager.login_message_category = "info"

# 🆕 新增：導入回測引擎模組 (安全檢查)
try:
    from ga_engine import (
        ga_load_data, ga_precompute_indicators, run_strategy_numba_core,
        format_ga_gene_parameters_to_text, STRATEGY_CONFIG_SHARED_GA,
        GA_PARAMS_CONFIG, GENE_MAP, check_ga_buy_signal_at_latest_point
    )
    from ga_engine_b import (
        load_stock_data_b, precompute_indicators_b, run_strategy_numba_core_b,
        format_gene_parameters_to_text_b, STRATEGY_CONFIG_B,GA_PARAMS_CONFIG_B
    )
    # === 從 stock_ga_web.py 移植：GA 引擎額外功能 ===
    from ga_engine import genetic_algorithm_unified
    from ga_engine_b import genetic_algorithm_unified_b, run_strategy_b
    from utils import execute_db_query, calculate_performance_metrics, calc_trade_extremes
    
    ENGINES_IMPORTED = True
    logger.info("✅ 成功導入所有必要的回測引擎模組。")
except ImportError as e:
    logger.error(f"❌ 導入回測引擎模組失敗: {e}。排程回測功能將被禁用。")
    ENGINES_IMPORTED = False

TARGET_SCAN_DATE = None


# 資料庫設定
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME", "finsimu_db"),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'connect_timeout': 15
}

# Gemini AI 設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None

# 安全設定
safety_settings_gemini = []
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        safety_settings_gemini = [
            genai_types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            ),
            genai_types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_MEDIUM_AND_ABOVE"
            )
        ]
        logger.info("Gemini AI 客戶端已成功配置")
    except Exception as e:
        logger.error(f"配置 Gemini AI 失敗: {e}")

WEIGHTS = {
    '積極型': {'annualized_return': 0.55, 'sharpe_ratio': 0.25, 'max_drawdown': 0.10, 'win_rate': 0.10},
    '均衡型': {'annualized_return': 0.20, 'sharpe_ratio': 0.50, 'max_drawdown': 0.20, 'win_rate': 0.10},
    '保守型': {'annualized_return': 0.10, 'sharpe_ratio': 0.40, 'max_drawdown': 0.40, 'win_rate': 0.10}
}

AI_ADJUSTMENT_FACTORS = {
    'Bullish': 1.10,
    'Neutral': 1.0,
    'Bearish': 0.90
}

# 【新增程式碼 START】
def training_worker_function():
    """
    這是在背景執行的執行緒函式，它會永遠循環，
    從 task_queue 中依序取出任務並執行。
    """
    logger.info("✅ [Worker Thread] 背景訓練工人已啟動，等待任務...")
    while True:
        task_id, task_data = task_queue.get()
        
        logger.info(f"🚚 [Worker Thread] 接收到新任務: {task_id} ({task_data['ticker']})")

        try:
            with results_lock:
                task_results[task_id] = {'status': 'STARTED', 'start_time': time.time()}

            local_trainer = SingleStockTrainer()

            ticker = task_data['ticker']
            start_date = task_data['start_date']
            end_date = task_data['end_date']
            custom_weights = task_data['custom_weights']
            basic_params = task_data['basic_params']
            fixed_num_runs = 3

            logger.info(f"--- [Worker Thread] 開始訓練系統 A for {ticker} ---")
            result_A = local_trainer.run_training(
                ticker=ticker, start_date=start_date, end_date=end_date, system_type='A',
                custom_weights=custom_weights, basic_params=basic_params, num_runs=fixed_num_runs
            )
            
            logger.info(f"--- [Worker Thread] 開始訓練系統 B for {ticker} ---")
            result_B = local_trainer.run_training(
                ticker=ticker, start_date=start_date, end_date=end_date, system_type='B',
                custom_weights=custom_weights, basic_params=basic_params, num_runs=fixed_num_runs
            )
            
            # ▼▼▼▼▼【需求修改】對調策略 1 和策略 2 的順序 ▼▼▼▼▼
            combined_results = []
            
            # 將系統 B (原策略2) 作為策略 1
            if result_B.get('success') and result_B.get('results'):
                strategy_B = result_B['results'][0]
                strategy_B['rank'] = 1
                strategy_B['strategy_type_name'] = '策略 1 '
                combined_results.append(strategy_B)
            
            # 將系統 A (原策略1) 作為策略 2
            if result_A.get('success') and result_A.get('results'):
                strategy_A = result_A['results'][0]
                strategy_A['rank'] = 2
                strategy_A['strategy_type_name'] = '策略 2 '
                combined_results.append(strategy_A)
            # ▲▲▲▲▲ 修改結束 ▲▲▲▲▲

            if not combined_results:
                raise Exception("訓練成功，但未能產生任何有效策略。")
            
            base_result = result_A if result_A.get('success') else result_B
            final_response = {
                'success': True, 'ticker': ticker,
                'training_period': base_result.get('training_period'),
                'results': combined_results
            }

            with results_lock:
                task_results[task_id].update({
                    'status': 'SUCCESS',
                    'result': final_response,
                    'end_time': time.time()
                })
            logger.info(f"✅ [Worker Thread] 任務 {task_id} 成功完成。")

        except Exception as e:
            logger.error(f"❌ [Worker Thread] 任務 {task_id} 執行失敗: {e}", exc_info=True)
            with results_lock:
                task_results[task_id].update({
                    'status': 'FAILURE',
                    'result': f'背景任務執行失敗: {str(e)}',
                    'end_time': time.time()
                })
        finally:
            task_queue.task_done()

def log_new_ticker_to_csv(ticker: str, market: str):
    """
    檢查並記錄新的股票代號到對應的 CSV 檔案中。(穩健版)
    - ticker: 經過驗證的完整股票代號 (例如 'AAPL', '2330.TW')
    - market: 'US' 或 'TW'
    """
    try:
        if market == 'TW':
            filepath = 'tw_stock.csv'
            header_name = '股票代號'
        elif market == 'US':
            filepath = 'usa_stock.csv'
            header_name = 'Symbol'
        else:
            logger.warning(f"[Ticker Logging] 未知的市場類型 '{market}'，跳過記錄 '{ticker}'。")
            return

        file_exists = os.path.exists(filepath)
        existing_tickers = set()

        # 讀取現有代號以避免重複
        if file_exists and os.path.getsize(filepath) > 0:
            try:
                # 使用 utf-8-sig 來處理可能存在的 BOM
                df = pd.read_csv(filepath, encoding='utf-8-sig')
                if header_name in df.columns:
                    # 清理可能的前後空白
                    existing_tickers = set(df[header_name].astype(str).str.strip())
            except pd.errors.EmptyDataError:
                logger.warning(f"[Ticker Logging] CSV 檔案 '{filepath}' 為空。")
            except Exception as e:
                logger.error(f"[Ticker Logging] 讀取 CSV '{filepath}' 失敗: {e}")
                return

        # 如果代號不存在，則使用 pandas 將其附加到檔案中
        if ticker not in existing_tickers:
            logger.info(f"[Ticker Logging] 發現新代號 '{ticker}'，寫入 '{filepath}'...")
            try:
                # 創建一個只包含新代號的 DataFrame
                new_ticker_df = pd.DataFrame([{header_name: ticker}])
                
                # 使用 to_csv 的附加模式 ('a') 進行寫入
                # header=not file_exists: 只有在檔案不存在時才寫入標頭
                # index=False: 不寫入 DataFrame 的索引
                new_ticker_df.to_csv(
                    filepath, 
                    mode='a', 
                    header=not file_exists or os.path.getsize(filepath) == 0,
                    index=False, 
                    encoding='utf-8-sig'
                )
                logger.info(f"✅ [Ticker Logging] 已成功將 '{ticker}' 添加到 '{filepath}'。")
            except Exception as e:
                logger.error(f"❌ [Ticker Logging] 寫入檔案 '{filepath}' 失敗: {e}")

    except Exception as e:
        logger.error(f"❌ [Ticker Logging] 記錄新股票代號時發生未預期錯誤: {e}")

# === 從 stock_ga_web.py 移植：User 類別和認證系統 ===
class User(UserMixin):
    """一個符合 Flask-Login 要求的使用者類別"""
    def __init__(self, user_data):
        self.id = user_data['id']
        self.username = user_data['username']
        self.email = user_data['email']
        self.password_hash = user_data['password_hash']

@login_manager.user_loader
def load_user(user_id):
    """Flask-Login 需要這個函式來從 session 中重新載入使用者物件"""
    user_data = execute_db_query("SELECT * FROM users WHERE id = %s", (user_id,), fetch_one=True)
    if user_data:
        return User(user_data)
    return None

# ==============================================================================
# >>> 以下為原始 program.py 的函式 (完全未修改) <<<
# ==============================================================================

def format_market_cap(market_cap, currency='USD'):
    """格式化市值顯示"""
    if not market_cap:
        return '未提供'
    currency_symbol = {'USD': '$', 'TWD': 'NT$', 'EUR': '€', 'JPY': '¥'}.get(currency, currency)
    if market_cap >= 1e12:
        return f"{currency_symbol}{market_cap/1e12:.1f}兆"
    elif market_cap >= 1e9:
        return f"{currency_symbol}{market_cap/1e9:.1f}B"
    elif market_cap >= 1e6:
        return f"{currency_symbol}{market_cap/1e6:.0f}M"
    else:
        return f"{currency_symbol}{market_cap:,.0f}"

def get_latest_vix():
    """使用 yfinance 獲取最新的 VIX 指數"""
    try:
        vix_ticker = yf.Ticker("^VIX")
        hist = vix_ticker.history(period="5d")
        if not hist.empty:
            latest_vix = hist['Close'].iloc[-1]
            logger.info(f"成功獲取最新VIX指數: {latest_vix:.2f}")
            return round(latest_vix, 2)
        else:
            logger.warning("無法獲取VIX歷史數據。")
            return None
    except Exception as e:
        logger.error(f"獲取VIX指數失敗: {e}")
        return None

def get_latest_sentiment_from_csv(csv_path='2021-2025每週新聞及情緒分析.csv'):
    """從CSV檔案讀取最新的市場情緒分數和摘要"""
    try:
        if not os.path.exists(csv_path):
            logger.warning(f"情緒分析CSV檔案不存在: {csv_path}")
            return None, None
        df = pd.read_csv(csv_path)
        if not df.empty:
            if '情緒分數' in df.columns and '重大新聞摘要' in df.columns:
                latest_sentiment = df.iloc[-1]
                score = latest_sentiment['情緒分數']
                summary = latest_sentiment['重大新聞摘要']
                logger.info(f"成功從CSV獲取最新情緒分數: {score}")
                return score, summary
            else:
                logger.warning("CSV檔案中缺少 '情緒分數' 或 '重大新聞摘要' 欄位。")
                return None, None
        else:
            logger.warning("情緒分析CSV檔案為空。")
            return None, None
    except Exception as e:
        logger.error(f"讀取情緒分析CSV失敗: {e}")
        return None, None

class EnhancedStockAnalyzer:
    """增強版股票分析器 - 整合基本面、技術面與AI策略"""
    def __init__(self, ticker: str):
        self.ticker_input = ticker.strip().upper()
        # 我們不再強制預設後綴，讓 get_basic_stock_data 函式去智慧判斷
        self.ticker = self.ticker_input 
        self.market = "US" # 預設為美股
        
        # 僅做初步的市場類型判斷，實際有效的 ticker 將在獲取數據時確認
        if self.ticker.endswith(".TW") or self.ticker.endswith(".TWO"):
            self.market = "TW"
        elif re.fullmatch(r'\d{4,6}', self.ticker):
            # 如果是數字代碼，暫時標記為台股，等待後續確認
            self.market = "TW"
        
        logger.info(f"初始化增強分析器：{self.ticker_input} (市場：{self.market})")

    # 找到並替換 EnhancedStockAnalyzer 的 get_basic_stock_data 函式
    def get_basic_stock_data(self):
        """獲取基本股票數據 - 增強版 (整合.TW/.TWO自動重試)"""
        try:
            # <<<<<<< 這裡是新的智慧重試邏輯 >>>>>>>
            is_tw_stock_code = re.fullmatch(r'\d{4,6}[A-Z]?', self.ticker_input)
            stock = None
            hist_data = pd.DataFrame() # 初始化一個空的 DataFrame

            if is_tw_stock_code:
                logger.info(f"偵測到台股數字代號 {self.ticker_input}，將依序嘗試 .TW 和 .TWO 後綴。")
                for suffix in ['.TW', '.TWO']:
                    potential_ticker = f"{self.ticker_input}{suffix}"
                    logger.info(f"正在嘗試使用 {potential_ticker}...")
                    try:
                        temp_stock = yf.Ticker(potential_ticker)
                        temp_hist = temp_stock.history(period="1y")
                        if not temp_hist.empty:
                            logger.info(f"成功使用 {potential_ticker} 獲取數據。")
                            self.ticker = potential_ticker  # 重要：更新類別實例中有效的 ticker
                            self.market = "TW"
                            stock = temp_stock
                            hist_data = temp_hist
                            break  # 成功找到數據，跳出迴圈
                    except Exception:
                        logger.warning(f"嘗試 {potential_ticker} 失敗，繼續下一個。")
                        continue
            
            # 如果不是台股代號，或所有嘗試都失敗，則執行原始邏輯
            if stock is None:
                logger.info(f"執行標準查詢：{self.ticker}")
                stock = yf.Ticker(self.ticker)
                hist_data = stock.history(period="1y")

            # 在所有嘗試後，最終檢查數據是否為空
            if hist_data.empty:
                raise ValueError(f"無法獲取 {self.ticker_input} 的歷史數據 (已嘗試 .TW 和 .TWO)")
            # <<<<<<< 智慧重試邏輯結束 >>>>>>>

            info = stock.info
            current_price = hist_data['Close'].iloc[-1]
            prev_close = hist_data['Close'].iloc[-2] if len(hist_data) >= 2 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close != 0 else 0
            
            change_str = f"{change:+.2f}"
            change_pct_str = f"{change_pct:+.2f}%"
            price_change_str = f"{change_str} ({change_pct_str})"
            
            return {
                "success": True, "ticker": self.ticker, 
                "company_name": info.get('longName', info.get('shortName', self.ticker)),
                "market": self.market, "current_price": round(float(current_price), 2), 
                "price_change": round(float(change), 2),
                "price_change_pct": round(float(change_pct), 2), 
                "price_change_str": price_change_str,
                "currency": info.get('currency', 'TWD' if self.market == 'TW' else 'USD'), 
                "market_cap": info.get('marketCap'),
                "pe_ratio": round(info.get('trailingPE'), 2) if info.get('trailingPE') else None,
                "forward_pe": round(info.get('forwardPE'), 2) if info.get('forwardPE') else None,
                "eps": round(info.get('trailingEps'), 2) if info.get('trailingEps') else None,
                "roe": round(info.get('returnOnEquity'), 4) if info.get('returnOnEquity') else None,
                "dividend_yield": round(info.get('dividendYield'), 4) if info.get('dividendYield') else None,
                "beta": round(info.get('beta'), 2) if info.get('beta') else None,
                "price_to_book": round(info.get('priceToBook'), 2) if info.get('priceToBook') else None,
                "debt_to_equity": round(info.get('debtToEquity'), 2) if info.get('debtToEquity') else None,
                "volume": int(hist_data['Volume'].iloc[-1]) if not pd.isna(hist_data['Volume'].iloc[-1]) else 0,
                "day_high": round(float(hist_data['High'].iloc[-1]), 2), 
                "day_low": round(float(hist_data['Low'].iloc[-1]), 2),
                "year_high": round(float(hist_data['High'].max()), 2), 
                "year_low": round(float(hist_data['Low'].min()), 2),
                "historical_data": hist_data, 
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logger.error(f"獲取基本股票數據失敗：{e}")
            # 提供更明確的錯誤訊息
            error_message = f"無法獲取股票 {self.ticker_input} 的數據。請檢查代號是否正確。對於台股，我們已自動嘗試 .TW 和 .TWO 後綴。錯誤詳情: {e}"
            return {"success": False, "error": error_message}

    def get_technical_indicators(self, hist_data):
        """計算技術指標"""
        try:
            close_prices = hist_data['Close']
            ma_5 = close_prices.rolling(window=5).mean().iloc[-1] if len(close_prices) >= 5 else None
            ma_10 = close_prices.rolling(window=10).mean().iloc[-1] if len(close_prices) >= 10 else None
            ma_20 = close_prices.rolling(window=20).mean().iloc[-1] if len(close_prices) >= 20 else None
            ma_60 = close_prices.rolling(window=60).mean().iloc[-1] if len(close_prices) >= 60 else None
            ma_120 = close_prices.rolling(window=120).mean().iloc[-1] if len(close_prices) >= 120 else None
            
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if len(rsi) >= 14 else None
            
            bb_period = 20
            bb_std = 2
            if len(close_prices) >= bb_period:
                bb_middle = close_prices.rolling(window=bb_period).mean()
                bb_std_dev = close_prices.rolling(window=bb_period).std()
                bb_upper = bb_middle + (bb_std_dev * bb_std)
                bb_lower = bb_middle - (bb_std_dev * bb_std)
                bb_upper_val = bb_upper.iloc[-1]
                bb_lower_val = bb_lower.iloc[-1]
                bb_middle_val = bb_middle.iloc[-1]
            else:
                bb_upper_val = bb_lower_val = bb_middle_val = None
                
            def format_to_2_decimal(value):
                return round(float(value), 2) if value and not pd.isna(value) else None
                
            return {
                "ma_5": format_to_2_decimal(ma_5), "ma_10": format_to_2_decimal(ma_10), 
                "ma_20": format_to_2_decimal(ma_20),
                "ma_60": format_to_2_decimal(ma_60), "ma_120": format_to_2_decimal(ma_120), 
                "rsi": format_to_2_decimal(current_rsi),
                "bb_upper": format_to_2_decimal(bb_upper_val), 
                "bb_middle": format_to_2_decimal(bb_middle_val), 
                "bb_lower": format_to_2_decimal(bb_lower_val)
            }
        except Exception as e:
            logger.error(f"計算技術指標失敗: {e}")
            return {}

    def get_ai_strategies_data(self):
        """獲取AI策略數據（從資料庫讀取真實回測日期）- (修改版：只獲取 Rank 1)"""
        try:
            market_type = "TW" if self.market == "TW" else "US"
            common_fields = "ai_strategy_gene, strategy_rank, strategy_details, period_return_pct, max_drawdown_pct, win_rate_pct, total_trades, average_trade_return_pct, max_trade_drop_pct, max_trade_gain_pct, game_start_date, game_end_date"
            
            # <<<< 修改點：將 LIMIT 3 改為 LIMIT 1，只抓取每個系統的最佳策略 >>>>
            system_a_query = f"SELECT {common_fields} FROM ai_vs_user_games WHERE user_id = 2 AND market_type = %s AND stock_ticker = %s ORDER BY strategy_rank ASC LIMIT 1"
            system_b_query = f"SELECT {common_fields} FROM ai_vs_user_games WHERE user_id = 3 AND market_type = %s AND stock_ticker = %s ORDER BY strategy_rank ASC LIMIT 1"
            
            system_a_strategies = execute_db_query(system_a_query, (market_type, self.ticker), fetch_all=True)
            system_b_strategies = execute_db_query(system_b_query, (market_type, self.ticker), fetch_all=True)
            
            start_date_str, end_date_str = "N/A", "N/A"
            first_strategy = (system_a_strategies or system_b_strategies or [None])[0]
            if first_strategy and first_strategy.get('game_start_date'):
                start_date_obj, end_date_obj = first_strategy['game_start_date'], first_strategy['game_end_date']
                if isinstance(start_date_obj, date): start_date_str = start_date_obj.strftime('%Y-%m-%d')
                if isinstance(end_date_obj, date): end_date_str = end_date_obj.strftime('%Y-%m-%d')
                
            return { 
                "system_a": system_a_strategies or [], "system_b": system_b_strategies or [], 
                "market_type": market_type,
                "backtest_start_date": start_date_str, "backtest_end_date": end_date_str,
                "backtest_period_description": f"回測期間：{start_date_str} 至 {end_date_str}"
            }
        except Exception as e:
            logger.error(f"獲取AI策略數據失敗：{e}")
            return { 
                "system_a": [], "system_b": [], "market_type": "", 
                "backtest_start_date": "N/A", "backtest_end_date": "N/A", 
                "backtest_period_description": "回測期間：無法獲取" 
            }
# 在 main_app.py 中新增這個函式

# 在 main_app.py 中，找到並用此【最終完美版】函式完整替換

def create_backtest_chart_assets(ticker, system_type, rank, portfolio, prices, dates, buys, sells):
    """為回測結果創建靜態PNG和互動HTML，並返回URL - (原版 + 隱藏工具列 + 懸停/座標軸日期格式化)"""
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                          row_heights=[0.7, 0.3],
                          subplot_titles=(f'{ticker} 價格走勢與交易信號', '價值變化'))
        
        # 股價與買賣點 (已加入懸停格式)
        fig.add_trace(go.Scatter(
            x=dates, y=prices, mode='lines', name='收盤價', 
            line=dict(color='rgba(102, 126, 234, 0.7)'),
            hovertemplate='日期: %{x:%Y/%m/%d}<br>收盤價: %{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        if buys:
            fig.add_trace(go.Scatter(
                x=[s['date'] for s in buys], y=[s['price'] for s in buys], 
                mode='markers', name='買入信號', 
                marker=dict(symbol='triangle-up', size=10, color='#27AE60', line=dict(width=1, color='white')),
                hovertemplate='買入信號<br>日期: %{x:%Y/%m/%d}<br>價格: %{y:.2f}<extra></extra>'
            ), row=1, col=1)
        
        if sells:
            fig.add_trace(go.Scatter(
                x=[s['date'] for s in sells], y=[s['price'] for s in sells], 
                mode='markers', name='賣出信號', 
                marker=dict(symbol='triangle-down', size=10, color='#E74C3C', line=dict(width=1, color='white')),
                hovertemplate='賣出信號<br>日期: %{x:%Y/%m/%d}<br>價格: %{y:.2f}<extra></extra>'
            ), row=1, col=1)

        # 投資組合價值 (已加入懸停格式)
        if portfolio is not None and len(portfolio) > 0:
             fig.add_trace(go.Scatter(
                x=dates, y=portfolio, mode='lines', name='組合價值', 
                line=dict(color='purple'),
                hovertemplate='日期: %{x:%Y/%m/%d}<br>組合價值: %{y:.4f}<extra></extra>'
            ), row=2, col=1)

        # 整體排版
        fig.update_layout(
            template='plotly_white', 
            height=500, 
            margin=dict(l=40, r=20, t=50, b=30), 
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1
            ),
            hovermode='x unified'
        )

        fig.update_xaxes(tickformat='%Y/%m/%d')

        
        base_filename = f"{ticker.replace('.', '_')}_{system_type}_Rank{rank}_backtest"
        
        # 儲存靜態圖片
        img_filename = f"{base_filename}.png"
        img_path = os.path.join('static/charts', img_filename)
        fig.write_image(img_path, scale=2)
        
        # 儲存互動HTML (已隱藏工具列)
        html_filename = f"{base_filename}.html"
        html_path = os.path.join('charts', html_filename)
        fig.write_html(
            html_path, 
            include_plotlyjs='cdn', 
            config={'displayModeBar': False}
        )
        
        logger.info(f"✅ (最終完美版) 回測圖表已生成：{img_filename} 和 {html_filename}")
        return f"/static/charts/{img_filename}", f"/charts/{html_filename}"
        
    except Exception as e:
        logger.error(f"創建回測圖表(最終完美版)失敗: {e}", exc_info=True)
        return None, None
    
def create_enhanced_stock_chart(ticker, company_name, hist_data):
    """創建股價圖表，同時生成靜態PNG和互動HTML"""
    try:
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.06)
        
        fig.add_trace(go.Candlestick(
            x=hist_data.index, open=hist_data['Open'], high=hist_data['High'], 
            low=hist_data['Low'], close=hist_data['Close'], name=f'{ticker} 股價', 
            increasing_line_color='#26C6DA', decreasing_line_color='#EF5350'
        ), row=1, col=1)
        
        fig.update_layout(
            # 移除圖表內的大標題，讓圖更乾淨
            template='plotly_white', height=500, 
            font=dict(family="Arial", size=12),
            margin=dict(l=40, r=20, t=20, b=30), # 縮小邊距
            xaxis_rangeslider_visible=False,
            showlegend=False # 手機上通常不顯示圖例
        )
        
        base_filename = f"{ticker.replace('.', '_')}_stock_chart"
        
        # 儲存靜態圖片
        img_filename = f"{base_filename}.png"
        img_path = os.path.join('static/charts', img_filename)
        fig.write_image(img_path, scale=2) # scale=2 讓圖片更清晰
        
        # 儲存互動HTML (移除Plotly控制欄)
        html_filename = f"{base_filename}.html"
        html_path = os.path.join('charts', html_filename)
        fig.write_html(html_path, include_plotlyjs='cdn', config={'displayModeBar': True}) # 在全螢幕模式顯示控制欄
        
        logger.info(f"圖表已生成：{img_filename} 和 {html_filename}")
        return f"/static/charts/{img_filename}", f"/charts/{html_filename}"
    
    except Exception as e:
        logger.error(f"創建圖表失敗：{e}")
        return None, None

def generate_enhanced_news_analysis(stock_data, tech_indicators, strategies_data, latest_vix, latest_sentiment):
    """使用 Gemini AI 生成包含最新新聞的深度分析報告 - 🔥 完整指標版本"""
    if not gemini_client:
        return "Gemini AI 服務暫時無法使用"
    
    try:
        # 🔥 修改：準備完整策略數據摘要
        system_a_summary = ""
        if strategies_data.get('system_a'):
            system_a_summary = "System A (28基因多策略):\n"
            for strategy in strategies_data['system_a'][:3]:
                system_a_summary += f" Rank {strategy['strategy_rank']}:\n"
                system_a_summary += f" 📈 總報酬率: {strategy.get('period_return_pct', 0):.2f}%\n"
                system_a_summary += f" 💰 平均交易報酬: {strategy.get('average_trade_return_pct', 0):.3f}%\n"
                system_a_summary += f" 🎯 勝率: {strategy.get('win_rate_pct', 0):.1f}%\n"
                system_a_summary += f" 🔢 交易次數: {strategy.get('total_trades', 0)}\n"
                system_a_summary += f" 📉 最大回撤: {strategy.get('max_drawdown_pct', 0):.2f}%\n"
                system_a_summary += f" 📉 最大跌幅: {strategy.get('max_trade_drop_pct', 0):.2f}%\n"
                system_a_summary += f" 📈 最大漲幅: {strategy.get('max_trade_gain_pct', 0):.2f}%\n"
        
        system_b_summary = ""
        if strategies_data.get('system_b'):
            system_b_summary = "System B (10基因RSI策略):\n"
            for strategy in strategies_data['system_b'][:3]:
                system_b_summary += f" Rank {strategy['strategy_rank']}:\n"
                system_b_summary += f" 📈 總報酬率: {strategy.get('period_return_pct', 0):.2f}%\n"
                system_b_summary += f" 💰 平均交易報酬: {strategy.get('average_trade_return_pct', 0):.3f}%\n"
                system_b_summary += f" 🎯 勝率: {strategy.get('win_rate_pct', 0):.1f}%\n"
                system_b_summary += f" 🔢 交易次數: {strategy.get('total_trades', 0)}\n"
                system_b_summary += f" 📉 最大回撤: {strategy.get('max_drawdown_pct', 0):.2f}%\n"
                system_b_summary += f" 📉 最大跌幅: {strategy.get('max_trade_drop_pct', 0):.2f}%\n"
                system_b_summary += f" 📈 最大漲幅: {strategy.get('max_trade_gain_pct', 0):.2f}%\n"
        
        system_a_details = ""
        if strategies_data.get('system_a'):
            system_a_details = "System A (28基因多策略):\n"
            for strategy in strategies_data['system_a'][:1]:
                system_a_details += f" Rank {strategy['strategy_rank']}:\n"
                system_a_details += f" 策略詳情: {strategy.get('strategy_details')}\n"
        
        system_b_details = ""
        if strategies_data.get('system_b'):
            system_b_details = "System B (10基因RSI策略):\n"
            for strategy in strategies_data['system_b'][:1]:
                system_b_details += f" Rank {strategy['strategy_rank']}:\n"
                system_b_details += f" 策略詳情: {strategy.get('strategy_details')}\n"
        
        # 🆕 新增：回測時間資訊
        backtest_info = f"回測期間: {strategies_data.get('backtest_start_date', 'N/A')} 至 {strategies_data.get('backtest_end_date', 'N/A')}"
        
        # 構建包含新聞查詢的提示詞
        prompt = f"""你是頂尖的量化投資分析師，請為股票 {stock_data['company_name']} ({stock_data['ticker']}) 撰寫一份包含最新新聞的投資分析報告。(不要加入自我介紹及問候)

請先搜尋以下關鍵字的最新新聞：
- "{stock_data['company_name']} stock news"
- "{stock_data['ticker']} adx、 macd、kdj今日技術指標"
- "{stock_data['ticker']} earnings"
- "{stock_data['company_name']} financial results"
- "market news today"
- "{stock_data['ticker']} 同業競爭"

基本面數據：
- 當前股價: {stock_data['current_price']:.2f} {stock_data['currency']}
- 市值: {format_market_cap(stock_data.get('market_cap'), stock_data['currency'])}
- 本益比: {stock_data.get('pe_ratio', '未提供')}
- 每股盈餘: {stock_data.get('eps', '未提供')}
- ROE: {f"{stock_data.get('roe', 0)*100:.2f}%" if stock_data.get('roe') else '未提供'}
- Beta值: {stock_data.get('beta', '未提供')}

技術指標：
- RSI: {tech_indicators.get('rsi', '未提供')}
- 5日均線: {tech_indicators.get('ma_5', '未提供')}
- 10日均線: {tech_indicators.get('ma_10', '未提供')}
- 20日均線: {tech_indicators.get('ma_20', '未提供')}
- 60日均線: {tech_indicators.get('ma_60', '未提供')}
- 120日均線: {tech_indicators.get('ma_120', '未提供')}
- 布林帶上軌: {tech_indicators.get('bb_upper', '未提供')}
- 布林帶下軌: {tech_indicators.get('bb_lower', '未提供')}

量化策略回測結果（{backtest_info}）：

策略1詳情:
{system_a_summary}
{system_a_details}

策略2詳情:
{system_b_summary}
{system_b_details}

請嚴格按以下格式撰寫分析報告，每個部分都使用獨立的 `##` 標題：

##  最新新聞分析
[根據搜尋到的最新新聞分析對股價的影響，100-120字]

##  基本面分析
[搜尋並分析公司財務體質、獲利能力、估值水準等還有同業比較，120-150字]

##  近期趨勢
[搜尋網路及基於技術指標分析股價趨勢等，60-100字]

##  策略解讀 (每個小段落都要換行，小標題不要加**)(!!如果該股票沒有提供策略，請直接回覆"此股票目前尚無訓練好策略，系統將自動將其納入下一批次的訓練清單中。"!!)
[請基於上方提供的 **策略 1** 和 **策略 2** 的數據，撰寫一份專業分析 (100-200字)。請務必遵循以下要點：
- **禁止使用 'System A' 或 'System B' 等內部術語**，只能直接使用 "策略 1" 和 "策略 2 " 來稱呼它們，不需加入(28基因多策略或10基因RSI策略作解釋)。
- **比較表現差異**: 分析兩套策略的風險收益特徵。哪一個看起來更穩健？哪個比較好？為什麼？
- **解讀關鍵指標**: 根據提供的「策略詳情」，解讀它們分別依賴哪些核心技術指標。
- **預測近期信號**: 結合當前技術指標，提醒近期是否可能出現買入或賣出信號。(不要提到有數據缺失的部分，謹說明有數據支持的部分即可)]

VIX 恐慌指數:{latest_vix if latest_vix is not None else '未能獲取'} (註：指數越高，市場恐慌程度越高)
市場情緒分數:{latest_sentiment[0] if latest_sentiment and latest_sentiment[0] is not None else '未能獲取'}

##  投資機會 (每個小段落都要換行，小標題不要加**)
[基於新聞、基本面、技術面和策略分析，列出2-3點主要投資機會。](每個小段落都要換行，小標題不要加**)

 風險提醒 (每個小段落都要換行，小標題不要加**)
[基於新聞、基本面、技術面和策略分析，列出2-3點主要投資風險。](每個小段落都要換行，小標題不要加**)

請確保分析專業、客觀，並重點關注最新新聞對投資決策的影響。"""

        # 🔥 關鍵：配置 Gemini 使用 Google Search 工具
        config = genai_types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4500,
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            safety_settings=safety_settings_gemini
        )
        
        response = gemini_client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=prompt,
            config=config
        )
        
        # 🔥 修復：添加空值檢查
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            logger.warning("Gemini API 返回空響應")
            return "AI分析暫時無法生成，但股票基本資料和技術指標分析正常運行"
            
    except Exception as e:
        logger.error(f"Gemini 新聞分析生成失敗: {e}")
        return f"AI新聞分析暫時無法使用：{str(e)}"

# === 從 stock_ga_web.py 移植：SingleStockTrainer 類別 ===
class SingleStockTrainer:
    """單支股票訓練器類別 - K線圖整合版"""
    def __init__(self):
        # 固定的GA參數
        self.fixed_params = {
            'mutation_rate': 0.25,
            'crossover_rate': 0.7,
            'no_trade_penalty': 0.1,
            'nsga2_enabled': True
        }
        
        # 預設的自定義權重
        self.default_custom_weights = {
            'total_return_weight': 0.5,
            'avg_trade_return_weight': 0.40,
            'win_rate_weight': 0.05,
            'trade_count_weight': 0,
            'drawdown_weight': 0.05
        }
        
        self.system_a_config = GA_PARAMS_CONFIG.copy()
        self.system_b_config = GA_PARAMS_CONFIG_B.copy()

    def validate_inputs(self, ticker, start_date, end_date, system_type):
        """驗證輸入參數"""
        errors = []
        
        if not ticker or len(ticker.strip()) == 0:
            errors.append("股票代號不能為空")
        
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            if start_dt >= end_dt:
                errors.append("開始日期必須早於結束日期")
            if end_dt > datetime.now():
                errors.append("結束日期不能超過今天")
        except ValueError:
            errors.append("日期格式錯誤，請使用 YYYY-MM-DD 格式")
        
        if system_type not in ['A', 'B']:
            errors.append("系統類型必須是 A 或 B")
        
        return errors

    def validate_custom_weights(self, custom_weights):
        """驗證自定義權重"""
        errors = []
        required_weights = ['total_return_weight', 'avg_trade_return_weight',
                          'win_rate_weight', 'trade_count_weight', 'drawdown_weight']
        
        total_weight = 0
        for weight_name in required_weights:
            if weight_name not in custom_weights:
                errors.append(f"缺少權重參數: {weight_name}")
                continue
            
            try:
                value = float(custom_weights[weight_name])
                if value < 0 or value > 1:
                    errors.append(f"{weight_name} 必須在 0 到 1 之間")
                total_weight += value
            except ValueError:
                errors.append(f"{weight_name} 必須是有效的數字")
        
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"所有權重總和應該等於1.0，目前總和為: {total_weight:.3f}")
        
        return errors

    # 找到 SingleStockTrainer 類別並完整替換 load_stock_data 函式
    # 檔案: main_app.py
# 在 SingleStockTrainer 類別中...

    # 找到 SingleStockTrainer 類別並完整替換 load_stock_data 函式 (最終修正版)
    def load_stock_data(self, ticker, start_date, end_date, system_type):
        """載入股票數據 - (V2.0 預熱期分離版)"""
        try:
            # <<< NEW LOGIC START >>>
            # 1. 將使用者輸入的日期字串轉換為 datetime 物件
            user_start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            user_end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

            # 2. 計算用於數據獲取的真正起始日期（往前推120天）
            data_fetch_start_date_obj = user_start_date_obj - timedelta(days=120)
            data_fetch_start_date_str = data_fetch_start_date_obj.strftime("%Y-%m-%d")

            # 3. 為了確保 yfinance 包含結束日期，將其加一天
            inclusive_end_date_for_yf = user_end_date_obj + timedelta(days=1)
            end_date_for_yf_str = inclusive_end_date_for_yf.strftime("%Y-%m-%d")

            logger.info(f"[Trainer Data] 使用者區間: {start_date} ~ {end_date}")
            logger.info(f"[Trainer Data] 預熱數據獲取區間: {data_fetch_start_date_str} ~ {end_date_for_yf_str}")
            
            # 4. 使用新的、更早的起始日期來定義載入函式
            load_func_a = lambda t: ga_load_data(
                t, start_date=data_fetch_start_date_str, end_date=end_date_for_yf_str,
                sentiment_csv_path='2021-2025每週新聞及情緒分析.csv' if os.path.exists('2021-2025每週新聞及情緒分析.csv') else None,
                verbose=False
            )
            load_func_b = lambda t: load_stock_data_b(t, start_date=data_fetch_start_date_str, end_date=end_date_for_yf_str, verbose=False)
            # <<< NEW LOGIC END >>>

            is_tw_stock_code = re.fullmatch(r'\d{4,6}[A-Z]?', ticker)
            loaded_data = None
            
            # (台股 .TW/.TWO 智慧重試邏輯不需變更)
            if is_tw_stock_code:
                logger.info(f"偵測到台股數字代號 {ticker}，將依序嘗試 .TW 和 .TWO 後綴。")
                for suffix in ['.TW', '.TWO']:
                    potential_ticker = f"{ticker}{suffix}"
                    prices_check = None
                    if system_type == 'A':
                        loaded_data = load_func_a(potential_ticker)
                        prices_check = loaded_data[0]
                    else:
                        loaded_data = load_func_b(potential_ticker)
                        prices_check = loaded_data[0]
                    
                    if prices_check and len(prices_check) > 0:
                        logger.info(f"成功使用 {potential_ticker} 載入數據。")
                        break
                    else:
                        loaded_data = None
            
            if not loaded_data:
                logger.info(f"執行標準查詢：{ticker}")
                if system_type == 'A':
                    loaded_data = load_func_a(ticker)
                else:
                    loaded_data = load_func_b(ticker)
            
            prices_check = loaded_data[0]
            if not prices_check or len(prices_check) == 0:
                # <<< MODIFICATION: 返回 None 給 iloc >>>
                return None, f"數據不足或載入失敗 (已嘗試 .TW/.TWO)", None

            # <<< NEW LOGIC START >>>
            # 5. 找到使用者原始 start_date 在擴展數據中的索引位置
            all_dates = loaded_data[1]
            dates_pd = pd.to_datetime([d.date() for d in all_dates])
            
            try:
                # 使用 searchsorted 快速定位
                user_start_date_iloc = dates_pd.searchsorted(user_start_date_obj, side='left')

                # 驗證找到的索引是否在範圍內且有效
                if user_start_date_iloc >= len(dates_pd):
                    raise IndexError("找不到開始日期") # 如果日期超出範圍，觸發except
            except (IndexError, TypeError):
                 # 如果找不到，返回一個明確的錯誤訊息
                logger.warning(f"在獲取的數據中找不到使用者指定的開始日期 {start_date}，回測中止。")
                return None, f"在獲取的數據中找不到使用者指定的開始日期 {start_date}，請確認該日期為交易日或選擇其他日期。", None
            # <<< NEW LOGIC END >>>
            
            if system_type == 'A':
                prices, dates, stock_df, vix_series, sentiment_series = loaded_data
                precalculated, ready = ga_precompute_indicators(
                    stock_df, vix_series, STRATEGY_CONFIG_SHARED_GA,
                    sentiment_series=sentiment_series, verbose=False
                )
                if not ready: return None, "系統A技術指標計算失敗", None
                return {
                    'prices': prices, 'dates': dates, 'stock_df': stock_df, 
                    'precalculated': precalculated, 'data_points': len(prices)
                }, None, user_start_date_iloc
            else: # 系統B
                prices, dates, stock_df, vix_series = loaded_data
                precalculated, ready = precompute_indicators_b(
                    stock_df, vix_series, STRATEGY_CONFIG_B, verbose=False
                )
                if not ready: return None, "系統B技術指標計算失敗", None
                return {
                    'prices': prices, 'dates': dates, 'stock_df': stock_df, 
                    'precalculated': precalculated, 'data_points': len(prices)
                }, None, user_start_date_iloc
                
        except Exception as e:
            logger.error(f"載入數據時發生錯誤: {e}", exc_info=True)
            return None, f"載入數據失敗: {str(e)}", None

    def apply_fixed_and_custom_params(self, system_type, custom_weights, basic_params):
        """應用固定參數和自定義權重到GA參數 - (修正版：同步最小交易次數)"""
        if system_type == 'A':
            config = self.system_a_config.copy()
        else:
            config = self.system_b_config.copy()
        
        config.update(self.fixed_params)
        
        # 處理使用者可調整的基礎參數
        if 'generations' in basic_params:
            config['generations'] = max(5, min(100, int(basic_params['generations'])))
        if 'population_size' in basic_params:
            config['population_size'] = max(20, min(200, int(basic_params['population_size'])))
            
        # --- ✨ 修正點 START ---
        # 確保使用者設定的 'min_trades' 同時更新軟性懲罰和 NSGA-II 的硬性約束
        if 'min_trades' in basic_params:
            # 1. 從前端獲取並驗證交易次數值，確保其在合理範圍內
            user_min_trades = max(1, min(20, int(basic_params['min_trades'])))
            
            # 2. 更新用於適應度函數（Fitness Function）的「軟性懲罰」參數
            #    這個參數決定了交易次數不足時，總回報的懲罰力度。
            config['min_trades_for_full_score'] = user_min_trades
            
            # 3. 更新用於 NSGA-II 演算法的「硬性約束」參數
            #    這個參數告訴演算法，交易次數少於此值的解是「無效」的，應盡力避免。
            config['min_required_trades'] = user_min_trades
            
            # 為了方便偵錯，可以加上日誌輸出
            logger.info(f"[Config Update] 最小交易次數已嚴格設定為: {user_min_trades} (同步更新懲罰與約束)")
        # --- ✨ 修正點 END ---
        
        # 設定 NSGA-II 的選擇方法和自定義權重
        config['nsga2_selection_method'] = 'custom_balance'
        config['custom_weights'] = custom_weights
        
        return config

    def generate_trading_signals(self, gene, data_result, ga_config, system_type):
        """生成交易信號數據"""
        try:
            logger.info(f"開始生成交易信號 - 系統{system_type}")
            
            if system_type == 'A':
                def get_indicator_list(name, gene_indices, opt_keys):
                    params = [ga_config[k][gene[g_idx]] for g_idx, k in zip(gene_indices, opt_keys)]
                    key = tuple(params) if len(params) > 1 else params[0]
                    indicator_data = data_result['precalculated'].get(name, {}).get(key, [np.nan] * len(data_result['prices']))
                    return np.array(indicator_data, dtype=np.float64)
                
                result = run_strategy_numba_core(
                    np.array(gene, dtype=np.float64),
                    np.array(data_result['prices'], dtype=np.float64),
                    get_indicator_list('vix_ma', [GENE_MAP['vix_ma_p']], ['vix_ma_period_options']),
                    get_indicator_list('sentiment_ma', [GENE_MAP['sentiment_ma_p']], ['sentiment_ma_period_options']),
                    get_indicator_list('rsi', [GENE_MAP['rsi_p']], ['rsi_period_options']),
                    get_indicator_list('adx', [GENE_MAP['adx_p']], ['adx_period_options']),
                    get_indicator_list('bbl', [GENE_MAP['bb_l_p'], GENE_MAP['bb_s_p']], ['bb_length_options', 'bb_std_options']),
                    get_indicator_list('bbm', [GENE_MAP['bb_l_p'], GENE_MAP['bb_s_p']], ['bb_length_options', 'bb_std_options']),
                    get_indicator_list('bbu', [GENE_MAP['bb_l_p'], GENE_MAP['bb_s_p']], ['bb_length_options', 'bb_std_options']),
                    get_indicator_list('ma', [GENE_MAP['ma_s_p']], ['ma_period_options']),
                    get_indicator_list('ma', [GENE_MAP['ma_l_p']], ['ma_period_options']),
                    get_indicator_list('ema_s', [GENE_MAP['ema_s_p']], ['ema_s_period_options']),
                    get_indicator_list('ema_m', [GENE_MAP['ema_m_p']], ['ema_m_period_options']),
                    get_indicator_list('ema_l', [GENE_MAP['ema_l_p']], ['ema_l_period_options']),
                    get_indicator_list('atr', [GENE_MAP['atr_p']], ['atr_period_options']),
                    get_indicator_list('atr_ma', [GENE_MAP['atr_p']], ['atr_period_options']),
                    get_indicator_list('kd_k', [GENE_MAP['kd_k_p'], GENE_MAP['kd_d_p'], GENE_MAP['kd_s_p']],
                                     ['kd_k_period_options', 'kd_d_period_options', 'kd_smooth_period_options']),
                    get_indicator_list('kd_d', [GENE_MAP['kd_k_p'], GENE_MAP['kd_d_p'], GENE_MAP['kd_s_p']],
                                     ['kd_k_period_options', 'kd_d_period_options', 'kd_smooth_period_options']),
                    get_indicator_list('macd_line', [GENE_MAP['macd_f_p'], GENE_MAP['macd_s_p'], GENE_MAP['macd_sig_p']],
                                     ['macd_fast_period_options', 'macd_slow_period_options', 'macd_signal_period_options']),
                    get_indicator_list('macd_signal', [GENE_MAP['macd_f_p'], GENE_MAP['macd_s_p'], GENE_MAP['macd_sig_p']],
                                     ['macd_fast_period_options', 'macd_slow_period_options', 'macd_signal_period_options']),
                    ga_config.get('commission_rate', 0.003),
                    61
                )
                
                if result is None or len(result) < 6:
                    logger.warning("系統A策略執行失敗")
                    return None, None, None
                
                portfolio_values, buy_indices, buy_prices, sell_indices, sell_prices, _ = result
                
                buy_signals = []
                sell_signals = []
                
                if buy_indices is not None and len(buy_indices) > 0:
                    buy_signals = [{'date': data_result['dates'][i], 'price': buy_prices[idx] if buy_prices is not None and idx < len(buy_prices) else data_result['prices'][i]}
                                 for idx, i in enumerate(buy_indices) if i < len(data_result['dates'])]
                
                if sell_indices is not None and len(sell_indices) > 0:
                    sell_signals = [{'date': data_result['dates'][i], 'price': sell_prices[idx] if sell_prices is not None and idx < len(sell_prices) else data_result['prices'][i]}
                                  for idx, i in enumerate(sell_indices) if i < len(data_result['dates'])]
                
                logger.info(f"系統A信號生成完成: {len(buy_signals)}買入, {len(sell_signals)}賣出")
                return portfolio_values.tolist(), buy_signals, sell_signals
                
            else:  # 系統B
                # 解析基因參數
                rsi_buy_entry_threshold = float(gene[0])
                rsi_exit_threshold = float(gene[1])
                vix_threshold = float(gene[2])
                low_vol_exit_strategy = int(gene[3])
                rsi_p_choice = int(gene[4])
                vix_ma_p_choice = int(gene[5])
                bb_len_choice = int(gene[6])
                bb_std_choice = int(gene[7])
                adx_threshold = float(gene[8])
                high_vol_entry_choice = int(gene[9])
                
                # 獲取指標數據
                rsi_period = STRATEGY_CONFIG_B['rsi_period_options'][rsi_p_choice]
                vix_ma_period = STRATEGY_CONFIG_B['vix_ma_period_options'][vix_ma_p_choice]
                bb_len = STRATEGY_CONFIG_B['bb_length_options'][bb_len_choice]
                bb_std = STRATEGY_CONFIG_B['bb_std_options'][bb_std_choice]
                
                rsi_list = data_result['precalculated']['rsi'][rsi_period]
                vix_ma_list = data_result['precalculated']['vix_ma'][vix_ma_period]
                bbl_list = data_result['precalculated']['bbl'][(bb_len, bb_std)]
                bbm_list = data_result['precalculated']['bbm'][(bb_len, bb_std)]
                adx_list = data_result['precalculated']['fixed']['adx_list']
                ma_short_list = data_result['precalculated']['fixed']['ma_short_list']
                ma_long_list = data_result['precalculated']['fixed']['ma_long_list']
                
                # 執行策略
                portfolio_values, buy_signals, sell_signals = run_strategy_b(
                    rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold, vix_threshold,
                    low_vol_exit_strategy, high_vol_entry_choice,
                    ga_config['commission_rate'],
                    data_result['prices'], data_result['dates'],
                    rsi_list, bbl_list, bbm_list, adx_list,
                    vix_ma_list, ma_short_list, ma_long_list
                )
                
                # 轉換信號格式
                buy_signals_formatted = [{'date': s[0], 'price': s[1]} for s in buy_signals]
                sell_signals_formatted = [{'date': s[0], 'price': s[1]} for s in sell_signals]
                
                logger.info(f"系統B信號生成完成: {len(buy_signals_formatted)}買入, {len(sell_signals_formatted)}賣出")
                return portfolio_values, buy_signals_formatted, sell_signals_formatted
                
        except Exception as e:
            logger.error(f"生成交易信號時發生錯誤: {e}")
            logger.error(traceback.format_exc())
            return None, None, None

    def create_line_chart_with_signals(self, ticker, data_result, portfolio_values, buy_signals, sell_signals):
        """用收盤價線圖和交易信號創建圖表"""
        try:
            df = data_result['stock_df'].copy()
            if df.index.name != 'Date':
                df = df.reset_index()
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                df['Date'] = pd.to_datetime(df.index)
            df = df.reset_index(drop=True)
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{ticker} 價格走勢與交易信號', '投資組合價值變化')
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=data_result['prices'],
                    mode='lines',
                    name='收盤價',
                    line=dict(color='rgba(102, 126, 234, 0.7)', width=2)
                ),
                row=1, col=1
            )
            
            if buy_signals:
                buy_dates = [pd.to_datetime(signal['date']) for signal in buy_signals]
                buy_prices = [signal['price'] for signal in buy_signals]
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=12, color='red', line=dict(width=2, color='darkred')),
                        name=f'買入 ({len(buy_signals)}次)',
                        hovertemplate='買入信號<br>日期: %{x}<br>價格: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if sell_signals:
                sell_dates = [pd.to_datetime(signal['date']) for signal in sell_signals]
                sell_prices = [signal['price'] for signal in sell_signals]
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates,
                        y=sell_prices,
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=12, color='blue', line=dict(width=2, color='darkblue')),
                        name=f'賣出 ({len(sell_signals)}次)',
                        hovertemplate='賣出信號<br>日期: %{x}<br>價格: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if portfolio_values and len(portfolio_values) >= len(df):
                portfolio_data = portfolio_values[:len(df)]
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=portfolio_data,
                        mode='lines',
                        line=dict(color='purple', width=2),
                        name='投資組合價值',
                        hovertemplate='組合價值<br>日期: %{x}<br>價值: %{y:.4f}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title={'text': f'{ticker} 策略回測結果', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
                xaxis_title='日期', yaxis_title='股價',
                xaxis2_title='', yaxis2_title='組合價值',
                height=600, showlegend=True,
                xaxis_rangeslider_visible=False,
                hovermode='x unified', template='plotly_white'
            )
            
            fig.update_xaxes(tickformat='%Y-%m-%d', tickangle=45)
            
            chart_json = fig.to_json()
            logger.info(f"成功生成 {ticker} 的Plotly線圖JSON")
            return chart_json
            
        except Exception as e:
            logger.error(f"創建線圖時發生錯誤: {e}")
            logger.error(traceback.format_exc())
            return None

    def calculate_detailed_metrics(self, gene, data_result, ga_config, system_type):
        """計算詳細績效指標 (已重構為使用 utils)"""
        try:
            portfolio_values, buy_signals, sell_signals = self.generate_trading_signals(gene, data_result, ga_config, system_type)
            if not portfolio_values: 
                return {}
            
            # 🔥 直接呼叫 utils 中的函式
            metrics = calculate_performance_metrics(
                portfolio_values,
                data_result['dates'],
                buy_signals,
                sell_signals,
                data_result['prices']
            )
            
            return metrics
        except Exception as e:
            logger.error(f"計算詳細指標時發生錯誤: {e}", exc_info=True)
            return {}

    def analyze_signal_status(self, buy_signals, sell_signals):
        """分析最新的信號狀態，返回一個提示字串"""
        if not buy_signals and not sell_signals: 
            return "目前無任何訊號"
        
        today = datetime.now().date()
        seven_days_ago = today - timedelta(days=7)
        
        last_buy_date = pd.to_datetime(buy_signals[-1]['date']).date() if buy_signals else None
        last_sell_date = pd.to_datetime(sell_signals[-1]['date']).date() if sell_signals else None
        
        # 1. 檢查近期信號
        recent_buy_signal = last_buy_date if last_buy_date and last_buy_date >= seven_days_ago else None
        recent_sell_signal = last_sell_date if last_sell_date and last_sell_date >= seven_days_ago else None
        
        if recent_buy_signal and (not recent_sell_signal or recent_buy_signal >= recent_sell_signal):
            return f"注意：{recent_buy_signal.strftime('%Y/%m/%d')} 有買入訊號！"
        
        if recent_sell_signal and (not recent_buy_signal or recent_sell_signal > recent_buy_signal):
            return f"注意：{recent_sell_signal.strftime('%Y/%m/%d')} 有賣出訊號！"
        
        # 2. 如果沒有近期信號，判斷長期持有狀態
        if last_buy_date and (not last_sell_date or last_buy_date > last_sell_date):
            return "目前策略狀態為「持有中」"
        else:
            return "目前策略狀態為「無倉位」"

    # 在 main_app.py 中，找到 SingleStockTrainer 類別並替換以下兩個函式

    def run_training(self, ticker, start_date, end_date, system_type, custom_weights, basic_params, num_runs=10):
        """執行訓練 - (V2.5 僅為Top1生成圖表)"""
        try:
            data_result, error_msg, user_start_date_iloc = self.load_stock_data(ticker, start_date, end_date, system_type)
            if error_msg: 
                return {'success': False, 'errors': [error_msg]}
            
            errors = self.validate_inputs(ticker, start_date, end_date, system_type)
            if errors: return {'success': False, 'errors': errors}
            weight_errors = self.validate_custom_weights(custom_weights)
            if weight_errors: return {'success': False, 'errors': weight_errors}
            
            ga_config = self.apply_fixed_and_custom_params(system_type, custom_weights, basic_params)
            
            strategy_pool = []
            logger.info(f"開始為 {ticker} 執行 {num_runs} 次系統{system_type} NSGA-II優化 (高效模式)...")
            
            for run_idx in range(num_runs):
                try:
                    runner_func = genetic_algorithm_unified if system_type == 'A' else genetic_algorithm_unified_b
                    result = runner_func(data_result['prices'], data_result['dates'], data_result['precalculated'], ga_config)
                    if result and result[0] is not None:
                        gene, _ = result
                        metrics = self.calculate_detailed_metrics(gene, data_result, ga_config, system_type)
                        if not metrics: continue
                        strategy_pool.append({
                            'gene': gene, 'fitness': metrics.get('total_return', 0),
                            'metrics': metrics, 'run': run_idx + 1,
                        })
                except Exception as e:
                    logger.warning(f"第 {run_idx + 1} 次運行失敗: {e}")
                    continue
            
            if not strategy_pool: 
                return {'success': False, 'errors': ['所有訓練運行都失敗了']}
            
            strategy_pool.sort(key=lambda x: x['fitness'], reverse=True)
            top_3 = strategy_pool[:3]
            
            results = []
            logger.info(f"訓練完成，開始為 Top {len(top_3)} 策略生成圖表與最終績效...")

            for i, strategy in enumerate(top_3):
                portfolio_values, buy_signals, sell_signals = self.generate_trading_signals(
                    strategy['gene'], data_result, ga_config, system_type
                )
                
                # 將URL初始化為None
                chart_image_url, chart_interactive_url = None, None
                
                if portfolio_values is not None:
                    sliced_portfolio_raw = portfolio_values[user_start_date_iloc:]
                    sliced_dates = data_result['dates'][user_start_date_iloc:]
                    sliced_prices = data_result['prices'][user_start_date_iloc:]
                    
                    period_buy_signals, period_sell_signals = self._filter_signals_for_period(buy_signals, sell_signals, start_date)

                    final_portfolio_for_metrics = []
                    
                    if not period_buy_signals and not period_sell_signals:
                        final_portfolio_for_metrics = [1.0] * len(sliced_dates)
                    else:
                        first_trade_date = period_buy_signals[0]['date']
                        first_trade_iloc = next((i for i, d in enumerate(sliced_dates) if d >= first_trade_date), 0)
                        metrics_prefix = [1.0] * first_trade_iloc
                        trade_period_portfolio = sliced_portfolio_raw[first_trade_iloc:]
                        initial_trade_value = trade_period_portfolio[0] if len(trade_period_portfolio) > 0 else 1.0
                        normalized_trade_curve = [p / initial_trade_value for p in trade_period_portfolio] if initial_trade_value > 0 else trade_period_portfolio
                        final_portfolio_for_metrics.extend(metrics_prefix)
                        final_portfolio_for_metrics.extend(normalized_trade_curve)
                    
                    # 指標計算對所有Top3策略都執行
                    final_display_metrics = calculate_performance_metrics(
                        final_portfolio_for_metrics,
                        sliced_dates, 
                        period_buy_signals, period_sell_signals, sliced_prices,
                        risk_free_rate=ga_config.get('risk_free_rate', 0.025),
                        commission_rate=ga_config.get('commission_rate', 0.0035)
                    )
                    
                    # ▼▼▼▼▼ 【核心修改點】 ▼▼▼▼▼
                    # 只有當策略是Top 1 (i == 0) 時，才生成圖表
                    if i == 0:
                        logger.info(f"  -> 為 Top 1 策略生成圖表...")
                        chart_image_url, chart_interactive_url = create_backtest_chart_assets(
                            ticker, f"System{system_type}", strategy['run'],
                            final_portfolio_for_metrics, 
                            sliced_prices, sliced_dates,
                            period_buy_signals, period_sell_signals
                        )
                    # ▲▲▲▲▲ 【修改結束】 ▲▲▲▲▲
                else:
                    final_display_metrics = strategy['metrics']

                formatter_func = format_ga_gene_parameters_to_text if system_type == 'A' else format_gene_parameters_to_text_b
                description = formatter_func(strategy['gene'])
                cache_buster = f"?v={int(time.time())}"
                
                results.append({
                    'rank': i + 1, 'gene': strategy['gene'],
                    'metrics': final_display_metrics,
                    'description': description,
                    'chart_image_url': f"{chart_image_url}{cache_buster}" if chart_image_url else None,
                    'chart_interactive_url': f"{chart_interactive_url}{cache_buster}" if chart_interactive_url else None
                })
            
            logger.info("Top 3 策略績效計算完成 (僅 Top 1 生成圖表)。")
            return {
                'success': True, 'ticker': ticker, 'system_type': system_type,
                'training_period': f"{start_date} ~ {end_date}",
                'results': results
            }
            
        except Exception as e:
            logger.error(f"訓練過程發生錯誤: {e}", exc_info=True)
            return {'success': False, 'errors': [f'訓練失敗: {str(e)}']}


    def _filter_signals_for_period(self, buy_signals, sell_signals, start_date):
        """
        (V2.2 新增) 智慧過濾並修正交易訊號，解決"孤兒賣出"問題。
        """
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")

        # 1. 先過濾出在回測區間內的所有訊號
        period_buys = [s for s in buy_signals if s['date'] >= start_date_obj]
        period_sells = [s for s in sell_signals if s['date'] >= start_date_obj]

        if not period_buys and not period_sells:
            return [], []

        # 2. 處理"孤兒賣出"問題
        first_buy_date = period_buys[0]['date'] if period_buys else None
        first_sell_date = period_sells[0]['date'] if period_sells else None

        # 如果區間內有賣出訊號，但沒有買入訊號，或者第一個賣出訊號早於第一個買入訊號
        # 這意味著第一個賣出是對應預熱期的買入，我們必須將其移除
        if first_sell_date and (first_buy_date is None or first_sell_date < first_buy_date):
            # 持續移除開頭的賣出訊號，直到第一個訊號是買入為止
            while period_sells and (not period_buys or period_sells[0]['date'] < period_buys[0]['date']):
                logger.info(f"移除孤兒賣出訊號: {period_sells[0]['date'].strftime('%Y-%m-%d')}")
                period_sells.pop(0)

        return period_buys, period_sells
    
    # =================== 【修改此方法】 ===================
# 在 SingleStockTrainer 類別中...
    # 檔案: main_app.py -> class SingleStockTrainer

    # 檔案: main_app.py -> class SingleStockTrainer

    def run_manual_backtest(self, ticker, gene, start_date, end_date):
        """執行手動回測 - (V2.4 圖表忠實版)"""
        try:
            system_type = 'A' if len(gene) in range(27, 29) else 'B' if len(gene) in range(9, 11) else None
            if not system_type: 
                return {'success': False, 'error': f"無法識別的基因長度: {len(gene)}"}
            
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
            today = datetime.now().date()
            if end_date_obj > today:
                 return {'success': False, 'error': '結束日期不能晚於今天'}

            data_result, error_msg, user_start_date_iloc = self.load_stock_data(ticker, start_date, end_date, system_type)
            if error_msg: 
                return {'success': False, 'error': error_msg}
            
            ga_config = self.system_a_config if system_type == 'A' else self.system_b_config
            
            portfolio_values, buy_signals, sell_signals = self.generate_trading_signals(gene, data_result, ga_config, system_type)
            if not portfolio_values: 
                return {'success': False, 'error': '無法生成回測結果，可能是此基因在此期間無交易。'}
            
            sliced_portfolio_raw = portfolio_values[user_start_date_iloc:]
            sliced_dates = data_result['dates'][user_start_date_iloc:]
            sliced_prices = data_result['prices'][user_start_date_iloc:]
            
            period_buy_signals, period_sell_signals = self._filter_signals_for_period(buy_signals, sell_signals, start_date)

            final_portfolio_for_metrics = []

            if not period_buy_signals and not period_sell_signals:
                final_portfolio_for_metrics = [1.0] * len(sliced_dates)
            else:
                first_trade_date = period_buy_signals[0]['date']
                first_trade_iloc = next((i for i, d in enumerate(sliced_dates) if d >= first_trade_date), 0)
                metrics_prefix = [1.0] * first_trade_iloc
                trade_period_portfolio = sliced_portfolio_raw[first_trade_iloc:]
                initial_trade_value = trade_period_portfolio[0] if len(trade_period_portfolio) > 0 else 1.0
                normalized_trade_curve = [p / initial_trade_value for p in trade_period_portfolio] if initial_trade_value > 0 else trade_period_portfolio
                final_portfolio_for_metrics.extend(metrics_prefix)
                final_portfolio_for_metrics.extend(normalized_trade_curve)

            metrics = calculate_performance_metrics(
                final_portfolio_for_metrics,
                sliced_dates,
                period_buy_signals, period_sell_signals,
                sliced_prices,
                risk_free_rate=ga_config.get('risk_free_rate', 0.025),
                commission_rate=ga_config.get('commission_rate', 0.0035)
            )

            # ▼▼▼▼▼ 【唯一的修改點】 ▼▼▼▼▼
            # 將 B&H 填充的 display_portfolio_values 替換為真實的 final_portfolio_for_metrics
            chart_image_url, chart_interactive_url = create_backtest_chart_assets(
                ticker, f"System{system_type}", "Manual",
                final_portfolio_for_metrics, # <--- 傳遞真實的績效曲線
                sliced_prices, sliced_dates,
                period_buy_signals, period_sell_signals
            )
            # ▲▲▲▲▲ 【修改結束】 ▲▲▲▲▲

            signal_status = None
            if end_date_obj == today:
                signal_status = self.analyze_signal_status(period_buy_signals, period_sell_signals)
            
            cache_buster = f"?v={int(time.time())}"

            return {
                'success': True, 'ticker': ticker, 'system_type_detected': f'系統 {system_type}',
                'backtest_period': f"{start_date} ~ {end_date}",
                'metrics': metrics, 
                'chart_image_url': f"{chart_image_url}{cache_buster}" if chart_image_url else None, 
                'chart_interactive_url': f"{chart_interactive_url}{cache_buster}" if chart_interactive_url else None, 
                'signal_status': signal_status
            }
            
        except Exception as e:
            logger.error(f"手動回測過程發生嚴重錯誤: {e}", exc_info=True)
            return {'success': False, 'error': f'回測失敗: {str(e)}'}

# ==============================================================================
#           >>> 【步驟 1: 新增使用者策略監控的核心邏輯】 <<<
# ==============================================================================
class UserStrategyMonitor:
    """
    專門用於每日掃描和更新使用者儲存策略的最新信號。
    """
    def __init__(self):
        # 使用一個較短的回測週期以大幅提高效能
        self.scan_period_days = 365 # 只回測最近365天的數據
        self.signal_check_days = 7   # 判斷最近7天內的信號
        self.start_date, self.end_date = self._get_date_range()
        self.trainer = SingleStockTrainer() # 借用其內部方法
        logger.info(f"👤 [使用者策略監控] 監控器初始化。掃描期間: {self.start_date} to {self.end_date}")

    def _get_date_range(self):
        # ▼▼▼▼▼【需求修改】▼▼▼▼▼
        # 檢查是否存在指定的目標掃描日期
        if TARGET_SCAN_DATE:
            logger.info(f"👤 [使用者策略監控] *** 偵測到目標掃描日期: {TARGET_SCAN_DATE} ***")
            end_date_obj = datetime.strptime(TARGET_SCAN_DATE, "%Y-%m-%d").date()
        else:
            # 恢復正常邏輯
            end_date_obj = datetime.now(pytz.timezone('Asia/Taipei')).date()
        # ▲▲▲▲▲ 修改結束 ▲▲▲▲▲
        
        start_date_obj = end_date_obj - timedelta(days=self.scan_period_days)
        inclusive_end_date_for_yf = end_date_obj + timedelta(days=1)
        
        return start_date_obj.strftime("%Y-%m-%d"), inclusive_end_date_for_yf.strftime("%Y-%m-%d")


    def get_all_user_strategies(self):
        """從資料庫獲取所有使用者儲存的策略。"""
        query = "SELECT id, ticker, gene FROM saved_strategies"
        strategies = execute_db_query(query, fetch_all=True)
        logger.info(f"👤 [使用者策略監控] 從資料庫找到 {len(strategies)} 條使用者策略需要掃描。")
        return strategies

# 檔案: main_app備分.py
# 在 class UserStrategyMonitor 中...

    def scan_strategy_for_recent_signal(self, ticker, gene_str):
                    
            try:
                gene = json.loads(gene_str)
                system_type = 'A' if len(gene) in range(27, 29) else 'B' if len(gene) in range(9, 11) else None
                if not system_type: return {'signal_type': 'NONE', 'signal_date': None}

                data_result, error_msg, _ = self.trainer.load_stock_data(ticker, self.start_date, self.end_date, system_type)
                
                if not data_result: 
                    if error_msg:
                        logger.warning(f"  -> 數據載入失敗 for {ticker}: {error_msg}")
                    return {'signal_type': 'NONE', 'signal_date': None}

                ga_config = self.trainer.system_a_config if system_type == 'A' else self.trainer.system_b_config
                _, buy_signals, sell_signals = self.trainer.generate_trading_signals(gene, data_result, ga_config, system_type)

                last_buy_date = pd.to_datetime(buy_signals[-1]['date']).date() if buy_signals else None
                last_sell_date = pd.to_datetime(sell_signals[-1]['date']).date() if sell_signals else None
                
                # ▼▼▼▼▼【需求修改】▼▼▼▼▼
                # 再次檢查是否存在指定的目標掃描日期，以確保日期比較基準一致
                if TARGET_SCAN_DATE:
                    scan_base_date = datetime.strptime(TARGET_SCAN_DATE, "%Y-%m-%d").date()
                else:
                    scan_base_date = datetime.now().date()
                # ▲▲▲▲▲ 修改結束 ▲▲▲▲▲
                
                # 找出在監測期內的近期買賣信號
                # 使用 scan_base_date 來取代原本的 today
                recent_buy = last_buy_date if last_buy_date and (scan_base_date - last_buy_date).days < self.signal_check_days else None
                recent_sell = last_sell_date if last_sell_date and (scan_base_date - last_sell_date).days < self.signal_check_days else None
                
                final_signal_type = 'NONE'
                final_signal_date = None

                # 判斷最新的 "近期" 信號
                if recent_buy and recent_sell:
                    if recent_buy >= recent_sell:
                        final_signal_type = 'BUY'
                        final_signal_date = recent_buy
                    else:
                        final_signal_type = 'SELL'
                        final_signal_date = recent_sell
                elif recent_buy:
                    final_signal_type = 'BUY'
                    final_signal_date = recent_buy
                elif recent_sell:
                    final_signal_type = 'SELL'
                    final_signal_date = recent_sell
                
                # 如果近期沒有任何新信號，則判斷長期持倉狀態
                if final_signal_type == 'NONE':
                    if last_buy_date and (not last_sell_date or last_buy_date > last_sell_date):
                        final_signal_type = 'HOLD'
                        final_signal_date = last_buy_date
                    else:
                        final_signal_type = 'NOP'
                        final_signal_date = last_sell_date
                
                return {'signal_type': final_signal_type, 'signal_date': final_signal_date}

            except Exception as e:
                logger.warning(f"  -> 掃描策略 {ticker} 時出錯: {e}")
                return {'signal_type': 'NONE', 'signal_date': None}
    
    def run_scan_and_update_db(self):
        """
        執行完整流程：獲取策略 -> 掃描 -> 更新資料庫
        """
        all_strategies = self.get_all_user_strategies()
        if not all_strategies:
            logger.info("👤 [使用者策略監控] 沒有找到任何使用者策略，任務結束。")
            return

        update_payloads = []
        for i, strategy in enumerate(all_strategies):
            logger.info(f"  - ({i+1}/{len(all_strategies)}) 正在掃描策略 ID: {strategy['id']}, Ticker: {strategy['ticker']}...")
            signal_result = self.scan_strategy_for_recent_signal(strategy['ticker'], strategy['gene'])
            
            update_payloads.append({
                'id': strategy['id'],
                'last_signal_type': signal_result['signal_type'],
                'last_signal_date': signal_result['signal_date'],
                'last_checked_at': datetime.now()
            })

        # 批次更新資料庫
        if update_payloads:
            try:
                conn = pymysql.connect(**DB_CONFIG)
                with conn.cursor() as cursor:
                    update_query = """
                    UPDATE saved_strategies 
                    SET last_signal_type = %s, last_signal_date = %s, last_checked_at = %s
                    WHERE id = %s
                    """
                    # 將字典列表轉換為元組列表
                    update_tuples = [
                        (p['last_signal_type'], p['last_signal_date'], p['last_checked_at'], p['id'])
                        for p in update_payloads
                    ]
                    cursor.executemany(update_query, update_tuples)
                    conn.commit()
                logger.info(f"💾 [使用者策略監控] 成功批次更新了 {len(update_payloads)} 條策略的信號狀態。")
            except Exception as e:
                logger.error(f"❌ [使用者策略監控] 批次更新資料庫失敗: {e}", exc_info=True)
            finally:
                if conn: conn.close()
        
        logger.info("✅ [使用者策略監控] 所有使用者策略掃描與更新任務完成。")

# ==============================================================================
#           >>> 【步驟 2: 新增排程任務的主函式】 <<<
# ==============================================================================
def run_user_strategies_scan():
    """每日自動執行的使用者策略監控任務"""
    with app.app_context():
        logger.info("="*50 + f"\n👤 [排程任務] 啟動使用者策略每日掃描... (台灣時間: {datetime.now(pytz.timezone('Asia/Taipei'))})\n" + "="*50)
        try:
            if not ENGINES_IMPORTED:
                logger.error("❌ [排程任務] 回測引擎模組未成功導入。使用者策略掃描任務中止。")
                return
            
            monitor = UserStrategyMonitor()
            monitor.run_scan_and_update_db()

        except Exception as e:
            logger.error(f"\n❌ [排程任務] 使用者策略掃描執行期間發生嚴重錯誤: {e}\n{traceback.format_exc()}")
        finally:
            logger.info("=" * 50)

# 建立訓練器實例
trainer = SingleStockTrainer()

# ==============================================================================
# >>> 以下為原始 program.py 的其他功能模組 (省略以節省篇幅，但實際使用時需要包含) <<<
# ==============================================================================

# 新聞情緒分析相關常數和函式
MOCK_TODAY = None
MAX_TOTAL_HEADLINES = 100
MAX_HEADLINES_PER_TOPIC = 5
CSV_FILEPATH = '2021-2025每週新聞及情緒分析.csv'

TARGET_COMPANIES_AND_TOPICS = {
    "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Google": "GOOGL",
    "Amazon": "AMZN", "Meta": "META", "Tesla": "TSLA",
    "S&P 500": None, "Nasdaq": None, "Dow Jones": None, "Federal Reserve": "Fed",
    "inflation": "CPI", "jobs report": "nonfarm payrolls", "interest rates": None,
    "crude oil": "WTI", "US election": None, "trade war": "tariffs","war": "war",
    "Trump": "tariffs",
}

# FinBERT 模型相關
finbert_tokenizer = None
finbert_model = None

if not FINBERT_AVAILABLE:
    logger.warning("PyTorch 或 Transformers 未安裝，FinBERT 情緒分析功能將被跳過。")


# === 來自 program.py 的原始首頁路由 ===
@app.route('/')
def home():
    """原始市場分析儀表板首頁"""
    return redirect(url_for('trainer_page'))

# === 從 stock_ga_web.py 移植：策略訓練平台路由 ===
@app.route('/trainer')
@login_required
def trainer_page():
    """策略訓練平台主頁面"""
    return render_template('index_page.html')

@app.route('/login')
def login_page():
    """提供登入頁面"""
    if current_user.is_authenticated:
        return redirect(url_for('trainer_page'))
    return render_template('login.html')

# === 從 stock_ga_web.py 移植：使用者認證 API ===
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    password_confirm = data.get('password_confirm')
    
    if not all([username, email, password, password_confirm]):
        return jsonify({'success': False, 'message': '所有欄位都不能為空'}), 400
    
    if password != password_confirm:
        return jsonify({'success': False, 'message': '兩次輸入的密碼不一致'}), 400
    
    if execute_db_query("SELECT id FROM users WHERE email = %s", (email,), fetch_one=True):
        return jsonify({'success': False, 'message': '此 Email 已被註冊'}), 409
    
    password_hash = generate_password_hash(password)
    sql = "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)"
    execute_db_query(sql, (username, email, password_hash))
    
    return jsonify({'success': True, 'message': '註冊成功！請使用 Email 登入。'}), 201

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    user_data = execute_db_query("SELECT * FROM users WHERE email = %s", (email,), fetch_one=True)
    
    if user_data and check_password_hash(user_data['password_hash'], password):
        user = User(user_data)
        login_user(user, remember=True)
        
        # 關鍵：獲取 next 參數，如果沒有，就預設跳轉到 /trainer
        next_url = request.args.get('next')
        # 增加安全性檢查，防止開放重導向漏洞
        if not next_url or not next_url.startswith('/'):
            next_url = url_for('trainer_page')
            
        return jsonify({'success': True, 'message': '登入成功', 'redirect_url': next_url}) # <--- 返回重導向 URL
    
    return jsonify({'success': False, 'message': 'Email 或密碼錯誤'}), 401

@app.route('/api/logout')
@login_required
def api_logout():
    logout_user()
    return jsonify({'success': True, 'message': '已成功登出'})

@app.route('/api/user/status')
def user_status():
    if current_user.is_authenticated:
        return jsonify({'logged_in': True, 'username': current_user.username})
    return jsonify({'logged_in': False})

# === 從 stock_ga_web.py 移植：核心功能 API (已受保護) ===
@app.route('/api/train', methods=['POST'])
@login_required
def api_train():
    """
    (新版) 訓練API端點 - 接收請求，產生任務ID，並將任務放入佇列。
    """
    if not ENGINES_IMPORTED:
        return jsonify({'success': False, 'errors': ['遺傳算法引擎未正確載入']}), 500
    
    try:
        data = request.json
        ticker = data.get('ticker', '').strip().upper()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        custom_weights = data.get('custom_weights', trainer.default_custom_weights)
        
        logger.info(f"Validating ticker '{ticker}' and checking if it needs to be added to lists...")
        analyzer = EnhancedStockAnalyzer(ticker)
        stock_data = analyzer.get_basic_stock_data()
        
        if not stock_data.get("success"):
            return jsonify({'success': False, 'errors': [stock_data.get('error', 'Invalid ticker')]}), 400
        
        validated_ticker = stock_data['ticker']
        market = stock_data['market']
        log_new_ticker_to_csv(validated_ticker, market)
        
        basic_params_from_user = data.get('basic_params', {})
        
        # ▼▼▼▼▼【需求修改】暗改最低交易次數 ▼▼▼▼▼
        user_min_trades = int(basic_params_from_user.get('min_trades', 4))
        # 如果使用者設定的交易次數低於 3，則在後端強制修改為 3
        effective_min_trades = 3 if user_min_trades < 3 else user_min_trades
        if effective_min_trades != user_min_trades:
            logger.info(f"[Trainer] 使用者設定 min_trades={user_min_trades}，已自動修正為 {effective_min_trades}。")
        
        fixed_basic_params = {
            'min_trades': effective_min_trades # 使用修正後的值
        }
        # ▲▲▲▲▲ 修改結束 ▲▲▲▲▲

        task_id = str(uuid.uuid4())

        task_data = {
            'ticker': validated_ticker,
            'start_date': start_date,
            'end_date': end_date,
            'custom_weights': custom_weights,
            'basic_params': fixed_basic_params
        }
        
        task_queue.put((task_id, task_data))
        
        with results_lock:
            task_results[task_id] = {'status': 'QUEUED'}
        
        logger.info(f"📥 訓練任務已加入佇列，ID: {task_id}。目前佇列大小: {task_queue.qsize()}")
        return jsonify({
            'success': True,
            'message': '訓練任務已成功提交，正在排隊等候執行。',
            'task_id': task_id,
        }), 202

    except Exception as e:
        logger.error(f"API錯誤 /api/train: {e}", exc_info=True)
        return jsonify({'success': False, 'errors': [f'API伺服器錯誤: {str(e)}']}), 500

# 【新增程式碼 START】
# 在 /api/train 之後，新增這個用於狀態查詢的 API
@app.route('/api/task_status/<string:task_id>')
@login_required
def get_task_status(task_id):
    """查詢內建任務系統的狀態和結果。"""
    with results_lock:
        # 從結果字典中安全地獲取任務資訊
        task = task_results.get(task_id, {})
    
    # 如果任務完成(成功或失敗)，我們才返回結果，否則 result 為 null
    result_payload = None
    if task.get('status') in ['SUCCESS', 'FAILURE']:
        result_payload = task.get('result')

    response = {
        'task_id': task_id,
        'status': task.get('status', 'NOT_FOUND'), # 如果 task_id 不存在，返回 NOT_FOUND
        'result': result_payload
    }
    return jsonify(response)
# 【新增程式碼 END】

# =================== 【修改此函式】 ===================
@app.route('/api/manual-backtest', methods=['POST'])
@login_required
def api_manual_backtest():
    if not ENGINES_IMPORTED:
        return jsonify({'success': False, 'error': '遺傳算法引擎未正確載入'}), 500
    
    try:
        data = request.json
        ticker = data.get('ticker', '').strip().upper()
        gene = data.get('gene')
        # 【修改點】接收 start_date 和 end_date，不再使用 duration_months
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # 【修改點】更新驗證邏輯
        if not all([ticker, gene, start_date, end_date]) or not isinstance(gene, list):
            return jsonify({'success': False, 'error': '無效的輸入參數'}), 400
        
        # 【修改點】將新參數傳遞給核心方法
        result = trainer.run_manual_backtest(ticker, gene, start_date, end_date)
        return jsonify(result)
    except Exception as e:
        logger.error(f"手動回測API錯誤: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'API 伺服器錯誤: {str(e)}'}), 500

# === 從 stock_ga_web.py 移植：策略管理 API ===
@app.route('/api/strategies', methods=['POST'])
@login_required
def save_strategy():
    """儲存一個新策略到使用者的清單中"""
    try:
        data = request.get_json()
        required_fields = ['ticker', 'train_start_date', 'train_end_date', 'gene', 'metrics', 'strategy_details']
        if not all(field in data for field in required_fields):
            return jsonify({'success': False, 'message': '缺少必要參數'}), 400
        
        metrics = data['metrics']
        
        sql = """
        INSERT INTO saved_strategies (
            user_id, ticker, train_start_date, train_end_date, gene,
            win_rate, total_return, trade_count, avg_trade_return, max_drawdown, sharpe_ratio,
            max_trade_extremes, strategy_details
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        args = (
            current_user.id,
            data['ticker'],
            data['train_start_date'],
            data['train_end_date'],
            json.dumps(data.get('gene', [])),
            metrics.get('win_rate_pct', 0.0),
            metrics.get('total_return', 0.0),
            metrics.get('trade_count', 0),
            metrics.get('average_trade_return', 0.0),
            metrics.get('max_drawdown', 0.0),
            metrics.get('sharpe_ratio', 0.0),  # <--- 新增這一行
            f"{metrics.get('max_trade_gain_pct', 0.0):.2f}% / {metrics.get('max_trade_drop_pct', 0.0):.2f}%", 
            data.get('strategy_details', '')
        )
        
        execute_db_query(sql, args)
        return jsonify({'success': True, 'message': '策略已成功儲存！'}), 201
    except Exception as e:
        logger.error(f"儲存策略時發生錯誤: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'伺服器錯誤: {str(e)}'}), 500

@app.route('/api/strategies', methods=['GET'])
@login_required
def get_strategies():
    """獲取目前使用者儲存的所有策略"""
    try:
        sql = "SELECT * FROM saved_strategies WHERE user_id = %s ORDER BY saved_at DESC"
        strategies_from_db = execute_db_query(sql, (current_user.id,), fetch_all=True)
        
        strategies_serializable = []
        if strategies_from_db:
            for s in strategies_from_db:
                for key in ['win_rate', 'total_return', 'avg_trade_return', 'max_drawdown']:
                    if s.get(key) is not None:
                        s[key] = float(s[key])
                
                if isinstance(s.get('gene'), str):
                    try:
                        s['gene'] = json.loads(s['gene'])
                    except json.JSONDecodeError:
                        s['gene'] = []
                
                for dt_key in ['saved_at', 'train_start_date', 'train_end_date']:
                    if isinstance(s.get(dt_key), (datetime, date)):
                        s[dt_key] = s[dt_key].isoformat().split('T')[0]
                
                strategies_serializable.append(s)
        
        return jsonify({'success': True, 'strategies': strategies_serializable})
    except Exception as e:
        logger.error(f"獲取策略時發生錯誤: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'伺服器錯誤: {str(e)}'}), 500

@app.route('/api/strategies/<int:strategy_id>', methods=['DELETE'])
@login_required
def delete_strategy(strategy_id):
    """刪除一個指定的策略"""
    try:
        sql = "DELETE FROM saved_strategies WHERE id = %s AND user_id = %s"
        rowcount = execute_db_query(sql, (strategy_id, current_user.id))
        
        if rowcount > 0:
            return jsonify({'success': True, 'message': '策略已刪除'})
        else:
            return jsonify({'success': False, 'message': '找不到策略或權限不足'}), 404
            
    except Exception as e:
        logger.error(f"刪除策略時發生錯誤: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'伺服器錯誤: {e}'}), 500

@app.route('/api/strategies/batch-delete', methods=['DELETE'])
@login_required
def batch_delete_strategies():
    """批次刪除多個指定的策略"""
    try:
        data = request.get_json()
        strategy_ids = data.get('strategy_ids')

        # 驗證輸入
        if not strategy_ids or not isinstance(strategy_ids, list):
            return jsonify({'success': False, 'message': '無效的請求，未提供策略 ID 清單'}), 400
        
        # 確保所有 ID 都是數字，增加安全性
        if not all(isinstance(sid, int) for sid in strategy_ids):
            return jsonify({'success': False, 'message': '無效的策略 ID 格式'}), 400

        # 創建對應數量的佔位符
        placeholders = ', '.join(['%s'] * len(strategy_ids))
        
        # 建立 SQL 查詢，確保只刪除屬於當前使用者的策略
        sql = f"DELETE FROM saved_strategies WHERE id IN ({placeholders}) AND user_id = %s"
        
        # 準備參數，將 user_id 放在最後
        params = tuple(strategy_ids) + (current_user.id,)
        
        rowcount = execute_db_query(sql, params)
        
        return jsonify({'success': True, 'message': f'已成功刪除 {rowcount} 個策略'})

    except Exception as e:
        logger.error(f"批次刪除策略時發生錯誤: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'伺服器錯誤: {str(e)}'}), 500
    
# ==============================================================================
# >>> 新增：資金配置 API 端點 <<<
# ==============================================================================

def _build_allocation_prompt(risk_profile, strategies_data):
    """一個輔助函式，用於動態生成給 Gemini 的 Prompt - 加強版"""
    
    # 1. 構建策略資產的文字描述
    assets_description = ""
    search_queries = []
    
    for i, strategy in enumerate(strategies_data, 1):
        # --- START: 修正後的程式碼區塊 ---
        # 處理最大漲跌幅，使其更穩健
        extremes_str = strategy.get('max_trade_extremes', '0% / 0%')
        parts = extremes_str.split('/')

        if len(parts) == 2:
            # 這是預期的格式，例如: "-12.3% / +25.4%"
            max_drop = parts[0].strip()
            max_gain = parts[1].strip()
        else:
            # 處理邊界情況，例如 "N/A" 或其他沒有 '/' 的格式
            max_drop = parts[0].strip() if parts else 'N/A'
            max_gain = 'N/A' # 給一個安全的預設值
        # --- END: 修正後的程式碼區塊 ---
        
        # 為每個股票準備搜尋關鍵字
        ticker = strategy['ticker']
        search_queries.append(f"{ticker} stock news today")
        search_queries.append(f"{ticker} earnings financial results")
        
        assets_description += f"""
資產 {i}: {ticker}
- 總報酬率: {float(strategy['total_return'])*100:+.2f}%
- 平均交易報酬率: {float(strategy['avg_trade_return'])*100:+.3f}%
- 勝率: {float(strategy['win_rate']):.1f}%
- 最大回撤: -{float(strategy['max_drawdown'])*100:.2f}%
- 最大漲跌幅: {max_gain} / {max_drop}
"""

    # 2. 構建完整的 Prompt（優化版）
    prompt = f"""你是專業的投資組合經理，需要為客戶分配投資資金。

**第一步：搜尋最新資訊**
請使用 Google Search 工具搜尋以下每個股票的最新資訊：

{chr(10).join([f"- {query}" for query in search_queries])}

重點搜尋內容：
- 最新財報和業績表現
- 重大新聞事件和公司動態
- 分析師評級和目標價
- 行業趨勢和市場情緒

**第二步：投資組合分析**

客戶風險偏好: {risk_profile}
- 保守型：重視穩定性，優先低回撤、高勝率的策略
- 均衡型：平衡收益與風險，尋求最佳風險調整後報酬
- 積極型：追求高報酬，可承受較高波動，但仍需稍微回撤風險及勝率

策略資產績效數據:
{assets_description}

**步驟 3：輸出結果**
基於以上所有資訊，**嚴格按照以下JSON格式回應**，不要有任何額外文字或markdown。理由(justification)必須非常簡潔，限於30-50字。

{{
  "allocations": [
    {{"ticker": "股票代號", "percentage": 數字}},
    {{"ticker": "股票代號", "percentage": 數字}}
  ],
  "reasoning": {{
    "overall_summary": "對整體配置策略的簡短總結(50-70字)。",
    "per_stock_analysis": [
      {{
        "ticker": "股票代號",
        "role_in_portfolio": "用 '核心增長'、'衛星配置' 或 '穩定基石' 來定義其角色。",
        "justification": "一句話解釋為何如此配置，以及它在組合中的作用。"
      }}
    ]
  }}
}}

**輸出要求：**
- 所有百分比總和必須是100。
- 分析理由必須精煉、專業、直指核心。
- **最終的輸出內容中，絕對不能包含任何方括號 `[]` 加上數字的引文標記、基因序列。**
"""
    
    return prompt

def _build_new_gemini_prompt(tickers_list):
    """
    為我們的量化模型，生成一個專注於市場分析的 Gemini Prompt。
    """
    unique_tickers = sorted(list(set(tickers_list)))

    prompt = f"""你是頂尖的金融市場分析師。請基於最新的市場資訊，為以下股票清單提供簡潔的質化分析。

**分析目標股票:**

{json.dumps(unique_tickers, indent=2)}

**任務:**

1. 使用 Google Search 搜尋每支股票最近一個月的重大新聞、財報表現、分析師評級變化。

2. 判斷每支股票當前的市場情緒和短期（未來1-3個月）的潛在催化劑或風險。

**輸出格式:**

請嚴格按照以下 JSON 格式回覆，不要有任何額外文字或 markdown。

{{

"analysis": [

{{

"ticker": "股票代號",

"sentiment": "用 'Bullish', 'Neutral', 'Bearish' 三個詞之一來描述",

"summary": "一句話總結其當前的市場地位和短期展望 (30-50字、使用繁體中文)。"

}}

],

"overall_summary": "對這幾支股票所在的市場板塊或整體市場氛圍的簡短總結 (50-70字、使用繁體中文)。"

}}

**重要提醒：**

- `summary` 內容必須簡潔、精準。

- 最終的輸出內容中，絕對不能包含任何方括號 `[]` 加上數字的引文標記。

"""

    return prompt


def calculate_annualized_return(total_return, start_date_str, end_date_str):
    """根據總報酬率和起訖日期，計算年化報酬率 (CAGR)。"""
    try:
        # 確保日期是字串格式
        start_date = datetime.strptime(str(start_date_str), '%Y-%m-%d')
        end_date = datetime.strptime(str(end_date_str), '%Y-%m-%d')
        
        days = (end_date - start_date).days
        if days <= 30: # 如果訓練期太短，年化意義不大，直接返回0或總報酬
            return total_return 

        number_of_years = days / 365.25
        if number_of_years <= 0: return 0.0

        ending_value = 1 + float(total_return)
        # 處理 total_return 是負數的情況
        if ending_value < 0:
            return -1.0 # 如果虧到本金都沒了，年化是負無窮，返回-100%

        annualized_rate = (ending_value ** (1 / number_of_years)) - 1
        return annualized_rate
    except (ValueError, TypeError, AttributeError):
        # 如果日期格式錯誤或無效，返回一個安全的0.0
        return 0.0

def assign_portfolio_roles(strategies_data):
    """
    根據策略的分數，分配「核心增長」、「穩定基石」、「衛星配置」的角色。
    
    Args:
        strategies_data: 包含每個策略所有數據的列表，每個元素是一個字典，
                         必須包含 'ticker', 'final_adjusted_score', 'stability_score'。
                                     
    Returns:
        一個字典，鍵是 ticker，值是分配的角色字串。
    """
    if not strategies_data:
        return {}

    # 處理只有一個策略的情況
    if len(strategies_data) == 1:
        return {strategies_data[0]['ticker']: '核心增長'}

    # 按 final_adjusted_score 降序排序
    strategies_sorted = sorted(strategies_data, key=lambda x: x['final_adjusted_score'], reverse=True)
    
    # 1. 指定「核心增長」
    core_growth_strategy = strategies_sorted[0]
    roles = {core_growth_strategy['ticker']: '核心增長'}
    
    # 2. 在剩餘策略中，找出「穩定基石」
    remaining_strategies = [s for s in strategies_data if s['ticker'] != core_growth_strategy['ticker']]
    
    if remaining_strategies:
        stable_cornerstone_strategy = max(remaining_strategies, key=lambda x: x['stability_score'])
        roles[stable_cornerstone_strategy['ticker']] = '穩定基石'

    # 3. 其餘的都是「衛星配置」
    for strategy in strategies_data:
        if strategy['ticker'] not in roles:
            roles[strategy['ticker']] = '衛星配置'
            
    return roles

def _allocate_percentages_largest_remainder(strategies):
    """
    使用最大餘額法來分配整數百分比，確保總和為100且無負數。
    Args:
        strategies: 一個字典列表，每個字典必須包含 'ticker' 和 'final_adjusted_score'。
    Returns:
        一個字典列表，包含 'ticker' 和 'percentage'。
    """
    total_score = sum(s['final_adjusted_score'] for s in strategies)
    if total_score <= 0:
        # 如果總分為0或負數，則平均分配
        equal_share = 100 // len(strategies)
        remainder = 100 % len(strategies)
        allocations = [{'ticker': s['ticker'], 'percentage': equal_share} for s in strategies]
        for i in range(remainder):
            allocations[i]['percentage'] += 1
        return allocations

    # 1. 計算每個策略的精確百分比和餘額
    for s in strategies:
        exact_percentage = (s['final_adjusted_score'] / total_score) * 100
        s['exact_percentage'] = exact_percentage
        s['floor_percentage'] = int(exact_percentage)
        s['remainder'] = exact_percentage - s['floor_percentage']

    # 2. 分配基礎百分比 (整數部分)
    allocated_sum = sum(s['floor_percentage'] for s in strategies)
    
    # 3. 計算還需分配多少個 1%
    remainder_to_distribute = 100 - allocated_sum

    # 4. 根據餘額大小排序，來決定誰能獲得額外的 1%
    strategies.sort(key=lambda x: x['remainder'], reverse=True)

    # 5. 分配剩餘的百分比
    for i in range(remainder_to_distribute):
        strategies[i]['floor_percentage'] += 1

    # 6. 整理並返回最終結果
    final_allocations = [
        {'ticker': s['ticker'], 'percentage': s['floor_percentage']}
        for s in strategies
    ]
    
    # 按百分比降序返回，讓前端顯示更好看
    final_allocations.sort(key=lambda x: x['percentage'], reverse=True)
    
    return final_allocations

@app.route('/api/capital-allocation', methods=['POST'])
@login_required
def api_capital_allocation():
    try:
        data = request.get_json()
        strategy_ids = data.get('strategy_ids')
        risk_profile = data.get('risk_profile')

        if not all([strategy_ids, risk_profile]):
            return jsonify({'success': False, 'message': '無效的請求參數'}), 400

        placeholders = ', '.join(['%s'] * len(strategy_ids))
        sql = f"""
        SELECT id, ticker, total_return, sharpe_ratio, max_drawdown, win_rate,
               train_start_date, train_end_date
        FROM saved_strategies
        WHERE id IN ({placeholders}) AND user_id = %s
        """
        params = tuple(strategy_ids) + (current_user.id,)
        strategies_from_db = execute_db_query(sql, params, fetch_all=True)

        if not strategies_from_db:
            return jsonify({'success': False, 'message': '找不到策略'}), 404

        processed_strategies = [{
            'id': s['id'], 'ticker': s['ticker'],
            'annualized_return': calculate_annualized_return(s['total_return'], s['train_start_date'], s['train_end_date']),
            'sharpe_ratio': float(s.get('sharpe_ratio', 0.0)),
            'max_drawdown': float(s.get('max_drawdown', 0.0)),
            'win_rate': float(s.get('win_rate', 0.0))
        } for s in strategies_from_db]

        metrics_to_normalize = ['annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        for metric in metrics_to_normalize:
            values = [s[metric] for s in processed_strategies if s.get(metric) is not None]
            if not values: continue
            min_val, max_val = min(values), max(values)
            for s in processed_strategies:
                value = s.get(metric)
                norm_key = f'norm_{metric}'
                if value is None or max_val == min_val:
                    s[norm_key] = 50.0; continue
                if metric == 'max_drawdown':
                    s[norm_key] = 100 * (max_val - value) / (max_val - min_val)
                else:
                    s[norm_key] = 100 * (value - min_val) / (max_val - min_val)

        weights = WEIGHTS.get(risk_profile, WEIGHTS['均衡型'])
        tickers_list = list(set([s['ticker'] for s in processed_strategies]))
        
        gemini_analysis = {"analysis": [], "overall_summary": "AI市場總結生成中..."}
        if gemini_client and tickers_list:
            try:
                prompt_text = _build_new_gemini_prompt(tickers_list)
                config = genai_types.GenerateContentConfig(
                    temperature=0.3, 
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
                )
                response = gemini_client.models.generate_content(
                    model='models/gemini-2.5-flash', contents=prompt_text, config=config
                )
                if response and hasattr(response, 'text') and response.text:
                    cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
                    if cleaned_text:
                        gemini_analysis = json.loads(cleaned_text)
            except Exception as gemini_err:
                logger.error(f"Gemini API 調用失敗: {gemini_err}")
        
        # --- (計算分數的邏輯) ---
        for s in processed_strategies:
            s['quant_score'] = (s.get('norm_annualized_return', 50) * weights['annualized_return'] +
                              s.get('norm_sharpe_ratio', 50) * weights['sharpe_ratio'] +
                              s.get('norm_max_drawdown', 50) * weights['max_drawdown'] +
                              s.get('norm_win_rate', 50) * weights['win_rate'])
            s['stability_score'] = (s.get('norm_max_drawdown', 50) * 0.6) + (s.get('norm_win_rate', 50) * 0.4)
            
            ticker_analysis = next((item for item in gemini_analysis.get('analysis', []) if item['ticker'] == s['ticker']), None)
            sentiment = ticker_analysis['sentiment'] if ticker_analysis else 'Neutral'
            
            # ▼▼▼▼▼【修改點 1】將 sentiment 標籤直接存入策略字典中 ▼▼▼▼▼
            s['ai_sentiment'] = sentiment
            # ▲▲▲▲▲ 修改結束 ▲▲▲▲▲

            ai_factor = AI_ADJUSTMENT_FACTORS.get(sentiment, 1.0)
            s['final_adjusted_score'] = s['quant_score'] * ai_factor
            s['ai_summary'] = ticker_analysis['summary'] if ticker_analysis else "無即時市場分析。"

        final_allocations = _allocate_percentages_largest_remainder(processed_strategies)
        
        portfolio_roles = assign_portfolio_roles(processed_strategies)
        
        reasoning = {
            "overall_summary": gemini_analysis.get("overall_summary", "AI市場總結生成失敗。"),
            "per_stock_analysis": [{
                "ticker": s['ticker'],
                "role_in_portfolio": portfolio_roles.get(s['ticker']),
                "justification": s['ai_summary'],
                # ▼▼▼▼▼【修改點 2】將 sentiment 標籤加入到回傳給前端的 reasoning 物件中 ▼▼▼▼▼
                "ai_sentiment": s['ai_sentiment']
                # ▲▲▲▲▲ 修改結束 ▲▲▲▲▲
            } for s in processed_strategies]
        }

        final_data = {"allocations": final_allocations, "reasoning": reasoning}
        return jsonify({"success": True, "data": final_data})

    except Exception as e:
        logger.error(f"資金配置 API 發生嚴重錯誤: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'伺服器內部錯誤: {str(e)}'}), 500
    

@app.route('/api/lookup-strategy', methods=['GET'])
@login_required
def api_lookup_strategy():
    """
    查詢資料庫中已存在的、針對特定股票的最佳策略 (系統A和系統B的Rank 1)。
    """
    try:
        ticker_query = request.args.get('ticker', '').strip().upper()
        if not ticker_query:
            return jsonify({'success': False, 'message': '請提供股票代號'}), 400

        # --- 步驟 1: 使用 EnhancedStockAnalyzer 驗證並獲取標準化的股票代號 ---
        # 這樣可以自動處理例如 "2330" -> "2330.TW" 的情況
        analyzer = EnhancedStockAnalyzer(ticker_query)
        stock_data = analyzer.get_basic_stock_data()
        
        if not stock_data.get("success"):
            return jsonify({'success': False, 'message': f"無效的股票代號: {stock_data.get('error', '未知錯誤')}"}), 404

        validated_ticker = stock_data['ticker']
        logger.info(f"策略查詢: 使用者查詢 '{ticker_query}', 標準化為 '{validated_ticker}'")

        # --- 步驟 2: 查詢資料庫 ---
        sql_query = """
            SELECT 
                user_id, stock_ticker, strategy_rank, 
                ai_strategy_gene AS gene, 
                strategy_details, 
                game_start_date AS train_start_date, 
                game_end_date AS train_end_date,
                period_return_pct,
                win_rate_pct,
                average_trade_return_pct,
                max_drawdown_pct,
                sharpe_ratio,              
                total_trades,
                max_trade_drop_pct,
                max_trade_gain_pct
            FROM ai_vs_user_games 
            WHERE 
                stock_ticker = %s AND 
                strategy_rank = 1 AND 
                user_id IN (2, 3)
            ORDER BY 
                user_id;
        """
        
        found_strategies = execute_db_query(sql_query, (validated_ticker,), fetch_all=True)

        if not found_strategies:
            return jsonify({'success': True, 'found': False, 'message': f'資料庫中尚無 {validated_ticker} 的最佳策略，請至訓練器自行訓練。'})

        # --- 步驟 3: 格式化返回的數據，使其與前端的數據結構一致 ---
        results = []
        for strategy in found_strategies:
            # 將Decimal類型轉換為float，以確保JSON序列化正常
            for key, value in strategy.items():
                if isinstance(value, (datetime, date)):
                    strategy[key] = value.isoformat().split('T')[0]
                elif hasattr(value, 'to_eng_string'): # 處理Decimal
                    strategy[key] = float(value.to_eng_string())
            
            # 將基因字串解析為JSON陣列
            try:
                strategy['gene'] = json.loads(strategy.get('gene', '[]'))
            except (json.JSONDecodeError, TypeError):
                strategy['gene'] = []

            # 創建與前端 'metrics' 對應的嵌套對象
            metrics = {
                'total_return': strategy.get('period_return_pct', 0) / 100.0,
                'win_rate_pct': strategy.get('win_rate_pct', 0),
                'average_trade_return': strategy.get('average_trade_return_pct', 0) / 100.0,
                'max_drawdown': strategy.get('max_drawdown_pct', 0) / 100.0,
                'sharpe_ratio': strategy.get('sharpe_ratio', 0.0),  
                'trade_count': strategy.get('total_trades', 0),
                'max_trade_drop_pct': strategy.get('max_trade_drop_pct', 0),
                'max_trade_gain_pct': strategy.get('max_trade_gain_pct', 0)
            }
            
            # 構建與前端訓練結果卡片一致的數據結構
            formatted_strategy = {
                'strategy_type_name': '策略 1' if strategy['user_id'] == 2 else '策略 2',
                'ticker': strategy['stock_ticker'],
                'train_start_date': strategy['train_start_date'],
                'train_end_date': strategy['train_end_date'],
                'gene': strategy['gene'],
                'strategy_details': strategy.get('strategy_details', ''), 
                'metrics': metrics
            }
            results.append(formatted_strategy)

        logger.info(f"成功為 {validated_ticker} 找到 {len(results)} 個最佳策略。")
        return jsonify({'success': True, 'found': True, 'strategies': results})

    except Exception as e:
        logger.error(f"查詢策略API時發生錯誤: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'伺服器內部錯誤: {str(e)}'}), 500


    
# === 來自 program.py 的原始 API 端點 ===
# 這是 main_app.py 中的一個函式，請完整替換
@app.route('/api/enhanced-analyze', methods=['POST'])
def enhanced_analyze_stock():
    """增強版股票分析API - 包含完整指標、回測時間，並同時生成靜態圖片與互動HTML圖表"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()
        if not ticker: 
            return jsonify({"success": False, "message": "請提供股票代號"})
        
        logger.info(f"開始增強分析：{ticker}")
        analyzer = EnhancedStockAnalyzer(ticker)
        
        stock_data = analyzer.get_basic_stock_data()
        if not stock_data["success"]: 
            return jsonify({"success": False, "message": f"無法獲取股票資料：{stock_data.get('error', '未知錯誤')}"})
        
         # ==================== 在這裡加入新的程式碼 ====================
        # 成功獲取股票資料後，呼叫函式記錄新的代號
        log_new_ticker_to_csv(stock_data['ticker'], stock_data['market'])
        # ============================================================
        tech_indicators = analyzer.get_technical_indicators(stock_data['historical_data'])
        strategies_data = analyzer.get_ai_strategies_data()
        
        # <<<<<<< 這是修改後的圖表生成呼叫 >>>>>>>
        chart_image_url, chart_interactive_url = create_enhanced_stock_chart(
            stock_data['ticker'], stock_data['company_name'], 
            stock_data['historical_data']
        )
        # <<<<<<< 修改結束 >>>>>>>
        
        logger.info("正在獲取最新的VIX指數和市場情緒分數...")
        latest_vix = get_latest_vix()
        latest_sentiment = get_latest_sentiment_from_csv()
        
        ai_analysis = generate_enhanced_news_analysis(
            stock_data, tech_indicators, strategies_data, latest_vix, latest_sentiment
        )
        
        if stock_data.get('market_cap'):
            stock_data['market_cap_formatted'] = format_market_cap(
                stock_data['market_cap'], stock_data['currency']
            )
        
        # 格式化策略數據
        if strategies_data.get('system_a'):
            for s in strategies_data['system_a']:
                s['formatted_metrics'] = {
                    'average_trade_return_formatted': f"{s.get('average_trade_return_pct',0):.3f}%",
                    'max_drawdown_formatted': f"{s.get('max_drawdown_pct',0):.2f}%",
                    'max_drop_formatted': f"{s.get('max_trade_drop_pct',0):.2f}%",
                    'max_gain_formatted': f"{s.get('max_trade_gain_pct',0):.2f}%"
                }
        
        if strategies_data.get('system_b'):
            for s in strategies_data['system_b']:
                s['formatted_metrics'] = {
                    'average_trade_return_formatted': f"{s.get('average_trade_return_pct',0):.3f}%",
                    'max_drawdown_formatted': f"{s.get('max_drawdown_pct',0):.2f}%",
                    'max_drop_formatted': f"{s.get('max_trade_drop_pct',0):.2f}%",
                    'max_gain_formatted': f"{s.get('max_trade_gain_pct',0):.2f}%"
                }
        
        del stock_data['historical_data']
        
        logger.info(f"增強分析完成：{ticker}")
        
        # <<<<<<< 這是修改後的API回傳內容 >>>>>>>
        return jsonify({
            "success": True,
            "data": {
                **stock_data,
                "technical_indicators": tech_indicators,
                "ai_strategies": strategies_data,
                "gemini_analysis": ai_analysis,
                "chart_image_url": chart_image_url,           # 靜態圖片URL
                "chart_interactive_url": chart_interactive_url # 互動HTML的URL
            }
        })
        # <<<<<<< 修改結束 >>>>>>>
        
    except Exception as e:
        logger.error(f"增強股票分析API錯誤：{e}")
        return jsonify({"success": False, "message": f"分析失敗：{str(e)}"})

@app.route('/api/news/search', methods=['POST'])
def api_news_search():
    """使用 Gemini Tools 搜尋最新新聞"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query: 
            return jsonify({"success": False, "message": "請提供搜尋關鍵字"})
        
        if not gemini_client: 
            return jsonify({"success": False, "message": "Gemini AI 服務不可用"})
        
        news_prompt = f"""請搜尋關於 "{query}" 的最新新聞和事件，並提供以下資訊：

1. 搜尋最近7天內的相關新聞
2. 重點關注財經、科技、政策等影響投資的新聞
3. 整理成結構化報告

請按以下格式回應：

##  最新新聞摘要
[列出3-5則最重要的新聞，每則包含標題、時間、重點內容](每個小段落都要換行，小標題不要加**)(150-200字)

##  市場影響分析
[分析這些新聞對相關股票或市場的潛在影響](每個小段落都要換行，小標題不要加**)(100-150字)

##  機會與風險(每個小段落都要換行，小標題不要加**)
[基於新聞內容指出可能的投資機會和風險點](100-150字)

請確保資訊準確且具有時效性。"""

        config = genai_types.GenerateContentConfig(
            temperature=0.6,
            max_output_tokens=4000,
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            safety_settings=safety_settings_gemini
        )
        
        response = gemini_client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=news_prompt,
            config=config
        )
        
        if response and hasattr(response, 'text') and response.text:
            return jsonify({
                "success": True,
                "data": {
                    "query": query,
                    "news_analysis": response.text.strip(),
                    "search_time": datetime.now().isoformat()
                }
            })
        else:
            return jsonify({"success": False, "message": "無法獲取新聞搜尋結果"})
            
    except Exception as e:
        logger.error(f"新聞搜尋API錯誤: {e}")
        return jsonify({"success": False, "message": f"新聞搜尋失敗：{str(e)}"})

# 檔案: main_app.py
# 請用此函式完整替換原有的 api_strategy_signals 函式

@app.route('/api/strategy-signals', methods=['GET'])
def api_strategy_signals():
    """
    (V3.3 目標一致版) AI策略信號 API - 顯示原始訓練績效，但信號來自近期回測
    """
    try:
        market = request.args.get('market', 'TW')
        signal_type_filter = request.args.get('type', 'buy').upper()
        filter_by_win_rate = request.args.get('min_win_rate_50', 'false').lower() == 'true'
        
        # 篩選條件現在會應用於 'a' 表 (ai_vs_user_games) 中的原始勝率
        win_rate_filter_sql = "AND a.win_rate_pct >= 50" if filter_by_win_rate else ""
        
        query = f"""
        SELECT 
            -- 1. 從近期回測中獲取信號本身的資訊
            bs.stock_ticker, bs.market_type, bs.system_type, bs.strategy_rank,
            bs.signal_type, bs.signal_date, bs.buy_price, bs.sell_price,

            -- 2. 從原始訓練數據庫中獲取一致的績效指標
            a.period_return_pct AS return_pct,
            a.win_rate_pct AS win_rate,
            a.ai_strategy_gene, a.strategy_details, a.game_start_date, a.game_end_date,
            a.total_trades, a.average_trade_return_pct, a.max_drawdown_pct,
            a.sharpe_ratio, a.max_trade_drop_pct, a.max_trade_gain_pct
        FROM 
            backtest_signals bs
        JOIN 
            ai_vs_user_games a ON bs.stock_ticker = a.stock_ticker COLLATE utf8mb4_unicode_ci
                               AND bs.strategy_rank = a.strategy_rank
                               AND bs.system_type = (CASE WHEN a.user_id = 2 THEN 'SystemA' ELSE 'SystemB' END) COLLATE utf8mb4_unicode_ci
        WHERE 
            bs.market_type = %s AND bs.signal_type = %s
            {win_rate_filter_sql}
        ORDER BY bs.signal_date DESC, a.win_rate_pct DESC;
        """
        
        signals = execute_db_query(query, (market, signal_type_filter), fetch_all=True)
        
        if signals:
            for signal in signals:
                # 這部分的格式化邏輯不需改變
                if isinstance(signal.get('signal_date'), date):
                    signal['signal_date_only'] = signal['signal_date'].strftime('%Y-%m-%d')
                if isinstance(signal.get('game_start_date'), date):
                    signal['game_start_date'] = signal['game_start_date'].isoformat()
                if isinstance(signal.get('game_end_date'), date):
                    signal['game_end_date'] = signal['game_end_date'].isoformat()
                if signal.get('win_rate') is not None: 
                    signal['win_rate'] = round(signal['win_rate'], 2)
                if signal.get('return_pct') is not None: 
                    signal['return_pct'] = round(signal['return_pct'], 2)

        return jsonify({"success": True, "data": signals or []})
        
    except Exception as e:
        logger.error(f"AI策略信號API錯誤: {e}", exc_info=True)
        return jsonify({"success": False, "message": "內部伺服器錯誤，策略信號查詢失敗"})

# --- 全局設定與開關 --

# ==============================================================================
#           >>> 以下為新加入的排程回測功能 (獨立區塊) <<<
# ==============================================================================

class StrategyBacktesterWithSignals:
    """策略回測器 - (從 backtest.py 遷移並整合，使用 logger)"""
    
    def __init__(self):
        self.backtest_months = 12
        self.signal_check_days = 7
        self.start_date, self.end_date = self._get_date_range()
        self.charts_dir = "charts"
        self.data_cache_a = {}
        self.data_cache_b = {}
        os.makedirs(self.charts_dir, exist_ok=True)
        logger.info(f"🎯 [排程回測] 回測器初始化完成")
        logger.info(f"📅 [排程回測] 回測期間: {self.start_date} ~ {self.end_date}")
        logger.info(f"📁 [排程回測] 圖表目錄: {self.charts_dir}")


    def _get_date_range(self):
        end_date_obj = datetime.now(pytz.timezone('Asia/Taipei')).date()
        start_date_obj = end_date_obj - timedelta(days=self.backtest_months * 30)
        inclusive_end_date_for_yf = end_date_obj + timedelta(days=1)
        
        return start_date_obj.strftime("%Y-%m-%d"), inclusive_end_date_for_yf.strftime("%Y-%m-%d")

    
    # 檔案: main_app.py
# 在 StrategyBacktesterWithSignals 類別中...

    def create_signals_table(self):
        """檢查並創建 backtest_signals 資料庫表 - (修正版：新增 signal_date 欄位)"""
        query = """
        CREATE TABLE IF NOT EXISTS `backtest_signals` (
          `id` INT AUTO_INCREMENT PRIMARY KEY, `stock_ticker` VARCHAR(20) NOT NULL,
          `stock_name` VARCHAR(100), `market_type` VARCHAR(10) NOT NULL,
          `system_type` VARCHAR(20) NOT NULL, `strategy_rank` INT NOT NULL,
          `signal_type` ENUM('BUY', 'SELL', 'BUY_SELL') NOT NULL, `signal_reason` TEXT,
          `signal_date` DATE NULL,
          `buy_price` FLOAT NULL, `sell_price` FLOAT NULL, `return_pct` FLOAT,
          `win_rate` FLOAT NULL, `chart_path` VARCHAR(255), `processed_at` DATETIME NOT NULL,
          INDEX `idx_market_signal` (`market_type`, `signal_type`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""
        try:
            execute_db_query(query)
            logger.info("✅ [排程回測] `backtest_signals` 表已確認存在 (含 signal_date 欄位)")
        except Exception as e:
            logger.error(f"❌ [排程回測] 創建 `backtest_signals` 表失敗: {e}")

    def save_results_to_db(self, results):
        """將有信號的結果儲存到資料庫 - (修正版：寫入 signal_date 欄位)"""
        conn = None
        try:
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE backtest_signals")
                logger.info("🗑️ [排程回測] 已清空舊的信號資料")
                query = """INSERT INTO backtest_signals (stock_ticker, stock_name, market_type, system_type, strategy_rank, 
                    signal_type, signal_reason, signal_date, buy_price, sell_price, return_pct, win_rate, chart_path, processed_at) 
                    VALUES (%(ticker)s, NULL, %(market_type)s, %(system)s, %(rank)s, %(signal_type)s, %(signal_reason)s, 
                    %(signal_date)s, %(buy_price)s, %(sell_price)s, %(return_pct)s, %(win_rate)s, %(chart_path)s, %(processed_at)s)"""
                
                to_save = [res for res in results if res.get('signal_type')]
                
                if to_save:
                    cursor.executemany(query, to_save)
                    conn.commit()
                    logger.info(f"💾 [排程回測] 成功將 {len(to_save)} 筆新信號儲存到資料庫")
                else:
                    logger.info("ℹ️ [排程回測] 本次運行沒有發現任何新信號可儲存")
        except Exception as e:
            if conn: conn.rollback()
            logger.error(f"❌ [排程回測] 儲存信號到資料庫失敗: {e}\n{traceback.format_exc()}")
        finally:
            if conn: conn.close()

    def get_all_strategies(self):
        """從資料庫獲取所有待回測的策略"""
        query = """SELECT user_id, market_type, stock_ticker, ai_strategy_gene, strategy_details, strategy_rank
                   FROM ai_vs_user_games WHERE strategy_rank = 1 AND ai_strategy_gene IS NOT NULL 
                   AND (user_id = 2 OR user_id = 3) ORDER BY stock_ticker, user_id, strategy_rank"""
        strategies = execute_db_query(query, fetch_all=True)
        if strategies:
            logger.info(f"📊 [排程回測] 從資料庫獲取到 {len(strategies)} 個策略")
            return strategies
        logger.warning("❌ [排程回測] 資料庫中沒有找到任何策略")
        return []
    
    def parse_strategy_gene(self, gene_json_str):
        try:
            return json.loads(gene_json_str) if isinstance(gene_json_str, str) else None
        except (json.JSONDecodeError, TypeError):
            return None
    
    def backtest_system_a_with_signals(self, gene, ticker):
        try:
            if ticker in self.data_cache_a:
                prices, dates, stock_df, vix_series, sentiment_series = self.data_cache_a[ticker]
            else:
                prices, dates, stock_df, vix_series, sentiment_series = ga_load_data(ticker, start_date=self.start_date, end_date=self.end_date, verbose=False)
                if prices is not None and len(prices) > 0: self.data_cache_a[ticker] = (prices, dates, stock_df, vix_series, sentiment_series)
            if not prices or len(prices) < 30: return None, None, None, None, None
            precalculated, ready = ga_precompute_indicators(stock_df, vix_series, STRATEGY_CONFIG_SHARED_GA, sentiment_series=sentiment_series, verbose=False)
            if not ready: return None, None, None, None, None
            def get_ind(name, g_indices, opt_keys):
                params = [GA_PARAMS_CONFIG[k][gene[g_idx]] for g_idx, k in zip(g_indices, opt_keys)]
                key = tuple(params) if len(params) > 1 else params[0]
                return np.array(precalculated.get(name, {}).get(key, [np.nan]*len(prices)), dtype=np.float64)
            result = run_strategy_numba_core(np.array(gene, dtype=np.float64), np.array(prices, dtype=np.float64),
                get_ind('vix_ma', [GENE_MAP['vix_ma_p']], ['vix_ma_period_options']), get_ind('sentiment_ma', [GENE_MAP['sentiment_ma_p']], ['sentiment_ma_period_options']),
                get_ind('rsi', [GENE_MAP['rsi_p']], ['rsi_period_options']), get_ind('adx', [GENE_MAP['adx_p']], ['adx_period_options']),
                get_ind('bbl', [GENE_MAP['bb_l_p'], GENE_MAP['bb_s_p']], ['bb_length_options', 'bb_std_options']), get_ind('bbm', [GENE_MAP['bb_l_p'], GENE_MAP['bb_s_p']], ['bb_length_options', 'bb_std_options']),
                get_ind('bbu', [GENE_MAP['bb_l_p'], GENE_MAP['bb_s_p']], ['bb_length_options', 'bb_std_options']), get_ind('ma', [GENE_MAP['ma_s_p']], ['ma_period_options']),
                get_ind('ma', [GENE_MAP['ma_l_p']], ['ma_period_options']), get_ind('ema_s', [GENE_MAP['ema_s_p']], ['ema_s_period_options']),
                get_ind('ema_m', [GENE_MAP['ema_m_p']], ['ema_m_period_options']), get_ind('ema_l', [GENE_MAP['ema_l_p']], ['ema_l_period_options']),
                get_ind('atr', [GENE_MAP['atr_p']], ['atr_period_options']), get_ind('atr_ma', [GENE_MAP['atr_p']], ['atr_period_options']),
                get_ind('kd_k', [GENE_MAP['kd_k_p'], GENE_MAP['kd_d_p'], GENE_MAP['kd_s_p']], ['kd_k_period_options', 'kd_d_period_options', 'kd_smooth_period_options']),
                get_ind('kd_d', [GENE_MAP['kd_k_p'], GENE_MAP['kd_d_p'], GENE_MAP['kd_s_p']], ['kd_k_period_options', 'kd_d_period_options', 'kd_smooth_period_options']),
                get_ind('macd_line', [GENE_MAP['macd_f_p'], GENE_MAP['macd_s_p'], GENE_MAP['macd_sig_p']], ['macd_fast_period_options', 'macd_slow_period_options', 'macd_signal_period_options']),
                get_ind('macd_signal', [GENE_MAP['macd_f_p'], GENE_MAP['macd_s_p'], GENE_MAP['macd_sig_p']], ['macd_fast_period_options', 'macd_slow_period_options', 'macd_signal_period_options']),
                GA_PARAMS_CONFIG.get('commission_rate', 0.003), 61)
            if result is None or len(result) < 6: return None, None, None, None, None
            portfolio_values, buy_indices, buy_prices, sell_indices, sell_prices, _ = result
            buy_signals = [{'date': dates[i], 'price': buy_prices[idx], 'index': i} for idx, i in enumerate(buy_indices) if i < len(dates) and idx < len(buy_prices)]
            sell_signals = [{'date': dates[i], 'price': sell_prices[idx], 'index': i} for idx, i in enumerate(sell_indices) if i < len(dates) and idx < len(sell_prices)]
            return portfolio_values, dates, prices, buy_signals, sell_signals
        except Exception: return None, None, None, None, None

    def backtest_system_b_with_signals(self, gene, ticker):
        try:
            if ticker in self.data_cache_b: prices, dates, stock_df, vix_series = self.data_cache_b[ticker]
            else:
                prices, dates, stock_df, vix_series = load_stock_data_b(ticker, start_date=self.start_date, end_date=self.end_date, verbose=False)
                if prices is not None and len(prices) > 0: self.data_cache_b[ticker] = (prices, dates, stock_df, vix_series)
            if not prices or len(prices) < 30: return None, None, None, None, None
            precalculated, ready = precompute_indicators_b(stock_df, vix_series, STRATEGY_CONFIG_B, verbose=False)
            if not ready: return None, None, None, None, None
            rsi_buy_entry, rsi_exit, vix_threshold = gene[0], gene[1], gene[2]
            low_vol_exit, rsi_period_choice, vix_ma_choice = int(gene[3]), int(gene[4]), int(gene[5])
            bb_len_choice, bb_std_choice, adx_thresh, high_vol_entry_choice = int(gene[6]), int(gene[7]), gene[8], int(gene[9])
            rsi_p, vix_ma_p = STRATEGY_CONFIG_B['rsi_period_options'][rsi_period_choice], STRATEGY_CONFIG_B['vix_ma_period_options'][vix_ma_choice]
            bb_l, bb_s = STRATEGY_CONFIG_B['bb_length_options'][bb_len_choice], STRATEGY_CONFIG_B['bb_std_options'][bb_std_choice]
            rsi_arr, vix_ma_arr = np.array(precalculated['rsi'][rsi_p], dtype=np.float64), np.array(precalculated['vix_ma'][vix_ma_p], dtype=np.float64)
            bbl_arr, bbm_arr = np.array(precalculated['bbl'][(bb_l, bb_s)], dtype=np.float64), np.array(precalculated['bbm'][(bb_l, bb_s)], dtype=np.float64)
            adx_arr, ma_s_arr, ma_l_arr = np.array(precalculated['fixed']['adx_list'], dtype=np.float64), np.array(precalculated['fixed']['ma_short_list'], dtype=np.float64), np.array(precalculated['fixed']['ma_long_list'], dtype=np.float64)
            def get_valid_iloc(arr):
                valid_indices = np.where(np.isfinite(arr))[0]
                return valid_indices[0] if len(valid_indices) > 0 else len(prices)
            start_iloc = max(get_valid_iloc(arr) for arr in [rsi_arr, vix_ma_arr, bbl_arr, bbm_arr, adx_arr, ma_s_arr, ma_l_arr]) + 1
            if start_iloc >= len(prices): return None, None, None, None, None
            result = run_strategy_numba_core_b(float(rsi_buy_entry), float(rsi_exit), float(adx_thresh), float(vix_threshold), low_vol_exit, high_vol_entry_choice,
                float(STRATEGY_CONFIG_B['commission_pct']), np.array(prices, dtype=np.float64), rsi_arr, bbl_arr, bbm_arr, adx_arr, vix_ma_arr, ma_s_arr, ma_l_arr, start_iloc)
            if result is None or len(result) < 7: return None, None, None, None, None
            portfolio_values, buy_indices, _, _, sell_indices, _, _ = result
            buy_signals = [{'date': dates[i], 'price': prices[i], 'index': i} for i in buy_indices if i < len(dates)]
            sell_signals = [{'date': dates[i], 'price': prices[i], 'index': i} for i in sell_indices if i < len(dates)]
            return portfolio_values, dates, prices, buy_signals, sell_signals
        except Exception: return None, None, None, None, None
    
    def check_recent_signals(self, signals, signal_type_text):
        if not signals: return False, f"無{signal_type_text}信號", None, None
        recent_signals_info = []
        today = datetime.now().date()
        latest_signal_date = None
        
        for signal in signals: # 檢查所有信號以找到最近的
            s_date = pd.to_datetime(signal['date']).date()
            if 0 <= (today - s_date).days < self.signal_check_days:
                if latest_signal_date is None or s_date > latest_signal_date:
                    latest_signal_date = s_date
                recent_signals_info.append(signal)

        if not recent_signals_info:
            return False, f"近期無{signal_type_text}信號", None, None

        latest_signal = max(recent_signals_info, key=lambda x: pd.to_datetime(x['date']))
        latest_price = latest_signal['price']
        reason = f"在 ({latest_signal_date.strftime('%Y-%m-%d')}) 檢測到{signal_type_text}信號"
        
        return True, reason, latest_price, latest_signal_date

    def create_strategy_backtest_chart(self, ticker, system_type, rank, portfolio, prices, dates, buys, sells, details, final_return):
        try:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3],
                                subplot_titles=[f'{ticker} {system_type} Rank{rank} - 股價與買賣點', '投資組合價值曲線'])
            if prices is not None and len(prices) > 0 and dates is not None: fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='股價', line=dict(color='white', width=1)), row=1, col=1)
            if buys: fig.add_trace(go.Scatter(x=[s['date'] for s in buys], y=[s['price'] for s in buys], mode='markers', name='買入點', marker=dict(symbol='triangle-up', size=12, color='lime')), row=1, col=1)
            if sells: fig.add_trace(go.Scatter(x=[s['date'] for s in sells], y=[s['price'] for s in sells], mode='markers', name='賣出點', marker=dict(symbol='triangle-down', size=12, color='red')), row=1, col=1)
            if portfolio is not None and portfolio.size > 0 and dates is not None:
                ret_pct = (portfolio - 1.0) * 100
                line_color = 'lime' if final_return >= 0 else 'red'
                fig.add_trace(go.Scatter(x=dates, y=ret_pct, mode='lines', name=f'報酬率 ({final_return:.2f}%)', line=dict(color=line_color, width=2), fill='tozeroy' if final_return >= 0 else None), row=2, col=1)
            fig.update_layout(title=f"{ticker} {system_type} Rank{rank} (報酬率: {final_return:.2f}%)", template='plotly_dark', height=800)
            filename = f"{ticker.replace('.', '_')}_{system_type}_Rank{rank}_backtest.html"
            path = os.path.join(self.charts_dir, filename)
            fig.write_html(path)
            return path
        except Exception as e:
            logger.error(f"    ❌ [排程回測] 為 {ticker} 創建圖表失敗: {e}")
            return None

    def _calculate_win_rate(self, buy_signals, sell_signals):
        if not buy_signals or not sell_signals: return 0.0
        buys, sells = sorted(buy_signals, key=lambda x: x['index']), sorted(sell_signals, key=lambda x: x['index'])
        total, wins = 0, 0
        num_trades = min(len(buys), len(sells))
        for i in range(num_trades):
            if sells[i]['index'] > buys[i]['index']:
                total += 1
                if sells[i]['price'] > buys[i]['price']: wins += 1
        return (wins / total) * 100 if total > 0 else 0.0

    def process_single_strategy(self, strategy):
        """處理單一策略 - (修正版：捕捉並儲存真實信號日期)"""
        try:
            ticker, sys_type, rank = strategy['stock_ticker'], "SystemA" if strategy['user_id'] == 2 else "SystemB", strategy['strategy_rank']
            logger.info(f"\n[{self.current_strategy_index}/{self.total_strategies}] 📊 [排程回測] 正在回測 {ticker} {sys_type} rank{rank}...")
            gene = self.parse_strategy_gene(strategy['ai_strategy_gene'])
            if not gene:
                logger.warning(f"    ⚠️ 無法解析 {ticker} 的策略基因，已跳過。")
                return None
            
            portfolio, dates, prices, buys, sells = (None,)*5
            if sys_type == "SystemA": portfolio, dates, prices, buys, sells = self.backtest_system_a_with_signals(gene, ticker)
            else: portfolio, dates, prices, buys, sells = self.backtest_system_b_with_signals(gene, ticker)
            
            if portfolio is None or buys is None or sells is None:
                logger.warning(f"    ⚠️ {ticker} 回測數據不足或失敗，已跳過。")
                return None
            
            final_return = (portfolio[-1] - 1.0) * 100 if portfolio.size > 0 else 0.0
            win_rate = self._calculate_win_rate(buys, sells)
            
            has_buy, buy_reason, buy_price, last_buy_date = self.check_recent_signals(buys, '買入')
            has_sell, sell_reason, sell_price, last_sell_date = self.check_recent_signals(sells, '賣出')

            has_recent_signal = has_buy or has_sell
            final_signal_type = None
            signal_reason = "無"
            actual_signal_date = None

            if has_recent_signal:
                if has_buy and (not has_sell or last_buy_date >= last_sell_date):
                    final_signal_type = 'BUY'
                    signal_reason = buy_reason
                    actual_signal_date = last_buy_date
                elif has_sell:
                    final_signal_type = 'SELL'
                    signal_reason = sell_reason
                    actual_signal_date = last_sell_date
            
            result = {
                'ticker': ticker, 'system': sys_type, 'rank': rank, 
                'market_type': strategy['market_type'], 'return_pct': final_return, 
                'win_rate': win_rate, 'signal_type': final_signal_type, 
                'signal_reason': signal_reason,
                'signal_date': actual_signal_date,
                'buy_price': buy_price, 
                'sell_price': sell_price, 'processed_at': datetime.now()
            }
            
            if final_signal_type:
                chart_path = self.create_strategy_backtest_chart(ticker, sys_type, rank, portfolio, prices, dates, buys, sells, strategy.get('strategy_details', ''), final_return)
                if chart_path: result['chart_path'] = os.path.basename(chart_path)
            
            return result
        except Exception as e:
            logger.error(f"    ❌ [排程回測] 處理策略 {strategy.get('stock_ticker')} 失敗: {e}\n{traceback.format_exc()}")
            return None

    def run_full_backtest_with_signals(self):
        logger.info("🚀 [排程回測] 開始執行策略批量回測...")
        start_time = time.time()
        strategies = self.get_all_strategies()
        if not strategies: return []
        
        self.total_strategies = len(strategies)
        logger.info(f"📋 [排程回測] 找到 {self.total_strategies} 個策略待回測")
        
        results = []
        for i, strategy in enumerate(strategies, 1):
            self.current_strategy_index = i
            result = self.process_single_strategy(strategy)
            if result: results.append(result)
        
        elapsed = time.time() - start_time
        logger.info("\n" + "=" * 70 + "\n📊 [排程回測] 回測總結\n" + "=" * 70)
        
        # =================== 【核心修正點】 ===================
        # 使用 res.get('signal_type') 來判斷是否有信號，這比 res['has_recent_signal'] 更安全且符合新邏輯
        signals_found = [res for res in results if res.get('signal_type')]
        # =======================================================

        logger.info(f"⏱️ [排程回測] 總耗時: {elapsed:.2f} 秒")
        logger.info(f"🎯 [排程回測] 發現信號: {len(signals_found)}")
        
        if signals_found:
            logger.info("\n🎯 [排程回測] 【近期有買賣信號的策略】")
            for res in signals_found:
                # =================== 【核心修正點】 ===================
                # 重寫日誌記錄邏輯，以適應新的 'signal_type' 欄位
                is_buy = res['signal_type'] == 'BUY'
                signal_icon = "🟢" if is_buy else "🔴"
                signal_text = "買入" if is_buy else "賣出"
                price_key = 'buy_price' if is_buy else 'sell_price'
                price = res.get(price_key)
                price_info = f"@ {price:.2f}" if price is not None else ""
                
                logger.info(f"  - {res['ticker']} | {res['system']} R{res['rank']} | 勝率: {res['win_rate']:.2f}% | {signal_icon} {signal_text} {price_info}")
                # =======================================================
        
        return results

def run_scheduled_backtest():
    """每日自動執行的主任務 (在 App Context 中運行)"""
    with app.app_context():
        logger.info("="*50 + f"\n⏰ [排程任務] 啟動每日自動回測... (台灣時間: {datetime.now(pytz.timezone('Asia/Taipei'))})\n" + "="*50)
        try:
            if not os.getenv("DB_PASSWORD"):
                logger.error("❌ [排程任務] 錯誤: DB_PASSWORD 環境變數未設定。任務中止。")
                return
            if not ENGINES_IMPORTED:
                logger.error("❌ [排程任務] 回測引擎模組未成功導入。任務中止。")
                return
            
            backtester = StrategyBacktesterWithSignals()
            backtester.create_signals_table()
            results = backtester.run_full_backtest_with_signals()
            if results:
                backtester.save_results_to_db(results)
            
            logger.info("✅ [排程任務] 每日自動回測任務執行完畢。")
        except Exception as e:
            logger.error(f"\n❌ [排程任務] 執行期間發生嚴重錯誤: {e}\n{traceback.format_exc()}")
        finally:
            logger.info("=" * 50)

# ==============================================================================
#           >>> 【新增】圖表檔案服務路由 <<<
# ==============================================================================

@app.route('/charts/<path:filename>')
def serve_chart(filename):

    return send_from_directory('charts', filename)

# ==============================================================================
# >>> Flask App 啟動區塊 (整合版) <<<
# ==============================================================================

if __name__ == '__main__':
    # 確保必要目錄存在
    os.makedirs('templates', exist_ok=True)
    os.makedirs('charts', exist_ok=True)
    os.makedirs('static/charts', exist_ok=True)
    
    # 建立簡單的模板文件以免 Flask 報錯
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head><title>Market Analysis Platform</title></head>
<body><h1>市場分析平台運行中</h1></body>
</html>""")
    
       # 【新增程式碼 START】
    # --- 啟動我們的背景工作執行緒 ---
    # 將執行緒設定為 daemon=True，這樣當主程式 (Flask app) 結束時，
    # 這個背景執行緒也會自動跟著關閉，不會卡住。
    logger.info("⚙️ 正在啟動背景訓練工作執行緒...")
    worker_thread = threading.Thread(target=training_worker_function, daemon=True)
    worker_thread.start()
    # 【新增程式碼 END】

    # 設定並啟動排程器
    logger.info("⚙️ 正在設定排程器...")
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Taipei'))
    
    if ENGINES_IMPORTED:
        # 新增任務：每日回測
        scheduler.add_job(
            func=run_scheduled_backtest,
            trigger='cron',
            hour=21,
            minute=35,
            id='daily_backtest_job',
            name='每日台灣時間 17:30 執行策略回測',
            replace_existing=True
        )
        logger.info("✅ 已設定每日策略回測排程 (17:30)。")
        scheduler.add_job(
            func=run_scheduled_backtest,
            trigger='cron',
            hour=12,
            minute=0,
            id='daily_backtest_job',
            name='每日台灣時間 09:30 執行策略回測',
            replace_existing=True
        )
        logger.info("✅ 已設定每日策略回測排程 (17:30)。")
        scheduler.add_job(
            func=run_scheduled_backtest,
            trigger='cron',
            hour=22,
            minute=30,
            id='daily_backtest_job',
            name='每日台灣時間 22:00 執行策略回測',
            replace_existing=True
        )
        logger.info("✅ 已設定每日策略回測排程 (17:30)。")
    else:
        logger.warning("⚠️ 由於模組導入失敗，每日自動回測功能已停用。")
        

  # ==============================================================================
        #           >>> 【步驟 3: 註冊新的排程任務】 <<<
        # ==============================================================================
    scheduler.add_job(
        func=run_user_strategies_scan, # <--- 呼叫我們的新函式
        trigger='cron', hour=11, minute=0, # <--- 錯開時間執行
        id='daily_user_strategy_scan_job', # <--- 給它一個新的唯一 ID
        name='每日台灣時間 11:00 掃描使用者策略',
        replace_existing=True
        )
    logger.info("✅ 已設定每日使用者策略掃描排程 (11:00)。")
    scheduler.add_job(
        func=run_user_strategies_scan, # <--- 呼叫我們的新函式
        trigger='cron', hour=17, minute=30, # <--- 錯開時間執行
        id='daily_user_strategy_scan_job', # <--- 給它一個新的唯一 ID
        name='每日台灣時間 17:30 掃描使用者策略',
        replace_existing=True
        )
    logger.info("✅ 已設定每日使用者策略掃描排程 (17:30)。")
    scheduler.add_job(
        func=run_user_strategies_scan, # <--- 呼叫我們的新函式
        trigger='cron', hour=22, minute=0, # <--- 錯開時間執行
        id='daily_user_strategy_scan_job', # <--- 給它一個新的唯一 ID
        name='每日台灣時間 22:00 掃描使用者策略',
        replace_existing=True
        )
    logger.info("✅ 已設定每日使用者策略掃描排程 (22:00)。")
        # ==============================================================================
   
    # 啟動排程器
    scheduler.start()
    logger.info("🚀 排程器已啟動。")
    
    
    atexit.register(lambda: scheduler.shutdown())
    
    logger.info("🚀 啟動整合版 AI 策略分析與市場分析平台...")
    logger.info("📊 策略訓練平台訪問: http://localhost:5001/trainer")
    logger.info("📈 市場分析平台訪問: http://localhost:5001/")

    app.run(debug=False, host='0.0.0.0', port=5001)
