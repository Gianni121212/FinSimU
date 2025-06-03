import os
import json
import datetime
import logging
import secrets
import pymysql
import yfinance as yf
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import google.generativeai as genai
from functools import wraps
from apscheduler.schedulers.background import BackgroundScheduler # APScheduler


# --- 從 ga_engine 導入 GA 相關功能 ---
try:
    from ga_engine import (
        ga_load_stock_data,
        ga_precompute_indicators,
        check_ga_buy_signal_at_latest_point,
        generate_buy_reason, 
        format_ga_gene_parameters_to_text,
        genetic_algorithm_with_elitism,  # <--- 把它的名字加到這裡！
        STRATEGY_CONFIG_SHARED_GA,
        GA_PARAMS_CONFIG
    )
    GA_ENGINE_IMPORTED = True
except ImportError as e:
    logging.error(f"無法從 ga_engine.py 導入: {e}. GA 功能可能受限。")
    GA_ENGINE_IMPORTED = False
    # 為了完整性，如果導入失敗，這裡也應該為 genetic_algorithm_with_elitism 提供一個“替身”（dummy function）
    # 這樣即使 ga_engine.py 出了問題，app.py 在嘗試調用它時也不會直接崩潰，而是調用這個替身。
    def ga_load_stock_data(*args, **kwargs): return None, None, None, None
    def ga_precompute_indicators(*args, **kwargs): return {}, False
    def check_ga_buy_signal_at_latest_point(*args, **kwargs): return False
    def generate_buy_reason(*args, **kwargs): return "GA買入信號（原因生成組件缺失）。"
    def format_ga_gene_parameters_to_text(*args, **kwargs): return "GA策略參數描述生成組件缺失。"
    def genetic_algorithm_with_elitism(*args, **kwargs): # <--- 為它也準備一個替身
        app.logger.error("genetic_algorithm_with_elitism (dummy) 被調用！GA引擎未正確導入。")
        return None, -float('inf') # 替身函數的返回值應該和真實函數的返回值類型一致
    STRATEGY_CONFIG_SHARED_GA = {}
    GA_PARAMS_CONFIG = {}


# --- StockAnalyzer Class 的導入 ---
import pandas_ta as ta
import feedparser
import urllib.parse
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 基本設定 (Flask App) ---
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

charts_dir = os.path.join(app.static_folder, 'charts')
if not os.path.exists(charts_dir):
    os.makedirs(charts_dir)

# --- 資料庫設定 ---
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD", "your_db_password"), # 強烈建議從環境變數讀取
    'database': os.getenv("DB_NAME", "finsimu_db"),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'connect_timeout': 10
}
if DB_CONFIG['password'] == "your_db_password": # 提醒用戶修改預設密碼
    app.logger.warning("請在 .env 文件或環境變數中設置一個安全的 DB_PASSWORD！")


# --- Gemini API 設定 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
report_generation_model = None
stock_analysis_model_gemini = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        safety_settings_gemini = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
    except Exception as gemini_err:
        app.logger.error(f"配置 Gemini API 金鑰失敗: {gemini_err}", exc_info=True)
        pass # 即使配置失敗，也允許應用繼續運行，只是AI功能會受限
else:
    app.logger.warning("未找到 GEMINI_API_KEY。AI 報告和股票分析功能將被禁用。")

# --- APScheduler 設定 ---
scheduler = BackgroundScheduler(daemon=True, timezone='UTC')

# --- GA 相關配置 (用於即時訓練和預計算) ---
SYSTEM_AI_USER_ID = 2 
AI_WATCHLIST_SIZE = 150 # 從 GA_PARAMS_CONFIG 獲取，預設100

GA_PARAMS_FOR_ONDEMAND_TRAIN = GA_PARAMS_CONFIG.copy()
GA_PARAMS_FOR_ONDEMAND_TRAIN['generations'] = GA_PARAMS_CONFIG.get('ondemand_train_generations', 20) 
GA_PARAMS_FOR_ONDEMAND_TRAIN['population_size'] = 80
GA_PARAMS_FOR_ONDEMAND_TRAIN['show_process'] = True 
NUM_GA_RUNS_ON_DEMAND = GA_PARAMS_CONFIG.get('ondemand_train_runs', 50)

ON_DEMAND_TRAIN_START_DATE = (datetime.date.today() - datetime.timedelta(days=3*365)).strftime("%Y-%m-%d")
ON_DEMAND_TRAIN_END_DATE = datetime.date.today().strftime("%Y-%m-%d")


# --- 輔助函數：設置 no-cache headers ---
def set_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# --- 資料庫輔助函數 ---
def get_db_connection():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except pymysql.Error as e:
        app.logger.error(f"資料庫連接錯誤: {e}")
        return None

def execute_db_query(query, args=None, fetch_one=False, fetch_all=False, commit=False, conn_param=None):
    conn_to_use = conn_param if conn_param else get_db_connection()
    if not conn_to_use:
        app.logger.error(f"資料庫查詢失敗: 無連接. 查詢: {query}")
        return None
    result = None
    try:
        with conn_to_use.cursor() as cursor:
            cursor.execute(query, args)
            if commit:
                if not conn_param: conn_to_use.commit()
                result = cursor.lastrowid if cursor.lastrowid else cursor.rowcount
            elif fetch_one: result = cursor.fetchone()
            elif fetch_all: result = cursor.fetchall()
    except pymysql.Error as e:
        app.logger.error(f"資料庫查詢錯誤: {e}\n查詢: {query}\n參數: {args}", exc_info=True)
        if conn_to_use and commit and not conn_param: conn_to_use.rollback()
        return None
    finally:
        if conn_to_use and not conn_param: conn_to_use.close()
    return result

# --- 資料庫初始化 ---
def init_finsimu_database():
    execute_db_query("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY, nickname VARCHAR(100) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL, investment_style VARCHAR(50),
        market_type ENUM('TW', 'US') NOT NULL DEFAULT 'TW',
        initial_capital_tw DECIMAL(15, 2) DEFAULT 0.00,
        cash_balance_tw DECIMAL(15, 2) DEFAULT 0.00,
        initial_capital_us DECIMAL(15, 2) DEFAULT 0.00,
        cash_balance_us DECIMAL(15, 2) DEFAULT 0.00,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_login_at TIMESTAMP NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    app.logger.info("Users table checked/created.")

    execute_db_query("""
    CREATE TABLE IF NOT EXISTS holdings (
        id INT AUTO_INCREMENT PRIMARY KEY, user_id INT NOT NULL,
        market_type ENUM('TW', 'US') NOT NULL,
        ticker VARCHAR(20) NOT NULL, stock_name VARCHAR(255), shares INT NOT NULL,
        average_cost DECIMAL(15, 4) NOT NULL, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, market_type, ticker)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    app.logger.info("Holdings table checked/created.")

    execute_db_query("""
    CREATE TABLE IF NOT EXISTS trade_history (
        id INT AUTO_INCREMENT PRIMARY KEY, user_id INT NOT NULL,
        market_type ENUM('TW', 'US') NOT NULL,
        timestamp DATETIME NOT NULL, trade_type ENUM('buy', 'sell') NOT NULL,
        ticker VARCHAR(20) NOT NULL, stock_name VARCHAR(255), shares INT NOT NULL,
        price_per_share DECIMAL(15, 4) NOT NULL, total_value DECIMAL(15, 2) NOT NULL,
        mood VARCHAR(50), reason TEXT, commission DECIMAL(10, 2) DEFAULT 0.00,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE, INDEX(user_id, market_type, timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    app.logger.info("Trade history table checked/created.")

    execute_db_query("""
    CREATE TABLE IF NOT EXISTS portfolio_history (
        id INT AUTO_INCREMENT PRIMARY KEY, user_id INT NOT NULL,
        market_type ENUM('TW', 'US') NOT NULL,
        timestamp DATETIME NOT NULL, total_value DECIMAL(15, 2) NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE, INDEX(user_id, market_type, timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    app.logger.info("Portfolio history table checked/created.")

    execute_db_query("""
    CREATE TABLE IF NOT EXISTS stock_data_cache (
        ticker VARCHAR(20) PRIMARY KEY, name VARCHAR(255), current_price DECIMAL(15, 4),
        previous_close DECIMAL(15, 4), daily_change DECIMAL(10, 4), daily_change_percent DECIMAL(8, 4),
        last_fetched TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    app.logger.info("Stock data cache table checked/created.")
    execute_db_query("""
    CREATE TABLE IF NOT EXISTS ai_vs_user_games (
        id INT AUTO_INCREMENT PRIMARY KEY, 
        user_id INT NOT NULL, 
        market_type ENUM('TW', 'US') NOT NULL,
        stock_ticker VARCHAR(20) NOT NULL, 
        game_start_date DATE NOT NULL, 
        game_end_date DATE NOT NULL,
        ai_strategy_gene JSON, 
        ai_initial_cash DECIMAL(15,2), 
        user_initial_cash DECIMAL(15,2),
        ai_final_portfolio_value DECIMAL(15, 2), 
        user_final_portfolio_value DECIMAL(15, 2),
        game_completed_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY `uq_user_market_stock` (`user_id`, `market_type`, `stock_ticker`), -- 確保這個唯一鍵存在
        INDEX idx_user_market_stock_perf (user_id, market_type, stock_ticker, ai_final_portfolio_value) 
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    app.logger.info("AI vs User Games table checked/created with unique key.")

    execute_db_query("""
    CREATE TABLE IF NOT EXISTS ai_vs_user_trades (
        id INT AUTO_INCREMENT PRIMARY KEY, 
        game_id INT NOT NULL,                 -- 這是外鍵列
        trader_type ENUM('user', 'ai') NOT NULL,
        timestamp DATETIME NOT NULL, 
        trade_type ENUM('buy', 'sell') NOT NULL, 
        ticker VARCHAR(20) NOT NULL,
        shares INT NOT NULL, 
        price_per_share DECIMAL(15, 4) NOT NULL,
        user_mood VARCHAR(50), 
        user_reason TEXT,
        FOREIGN KEY (game_id) REFERENCES ai_vs_user_games(id) ON DELETE CASCADE 
        -- ON DELETE CASCADE 表示如果 ai_vs_user_games 中的記錄被刪除，這裡相關的交易也會被刪除
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    app.logger.info("AI vs User Trades table checked/created.")
    
    execute_db_query("""
    CREATE TABLE IF NOT EXISTS ga_current_potential_buys (
        id INT AUTO_INCREMENT PRIMARY KEY,
        market_type ENUM('TW', 'US') NOT NULL,
        stock_ticker VARCHAR(20) NOT NULL,
        stock_name VARCHAR(255),
        current_price DECIMAL(15, 4),
        ai_strategy_gene JSON, 
        buy_reason VARCHAR(500), 
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX (market_type, calculated_at),
        UNIQUE (market_type, stock_ticker)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    app.logger.info("GA Current Potential Buys table checked/created.")

    app.logger.info("FinSimU 資料庫初始化完成。")


# --- 股票數據服務 ---
STOCK_CACHE_DURATION = datetime.timedelta(minutes=1) 
def get_stock_data_from_yf(ticker_symbol):
    try:
        app.logger.info(f"YFINANCE FETCH: 嘗試獲取 {ticker_symbol} 的最新數據")
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        current_price = info.get('currentPrice', info.get('regularMarketPrice', info.get('bid')))
        hist_2d = stock.history(period="2d") 

        previous_close = info.get('previousClose')

        if not hist_2d.empty:
            if current_price is None and len(hist_2d) >= 1:
                current_price = hist_2d['Close'].iloc[-1]
            if previous_close is None and len(hist_2d) >= 2: 
                previous_close = hist_2d['Close'].iloc[-2]
            elif previous_close is None and len(hist_2d) == 1 and current_price is not None: 
                 previous_close = current_price 

        if current_price is None or previous_close is None:
             app.logger.warning(f"無法從 yfinance 可靠地確定 {ticker_symbol} 的當前價格或前收盤價。")
             name = info.get('longName', info.get('shortName', ticker_symbol)) if info else ticker_symbol
             if name and name != ticker_symbol:
                 execute_db_query(""" REPLACE INTO stock_data_cache (ticker, name, last_fetched) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE name=VALUES(name), last_fetched=VALUES(last_fetched) """, (ticker_symbol.upper(), name, datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)), commit=True)
                 return {'ticker': ticker_symbol.upper(), 'name': name, 'current_price': None, 'previous_close': None, 'daily_change': None, 'daily_change_percent': None, 'last_fetched': datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)}
             return None

        current_price_f = float(current_price)
        previous_close_f = float(previous_close)
        daily_change = current_price_f - previous_close_f
        daily_change_percent = (daily_change / previous_close_f) * 100 if previous_close_f != 0 else 0.0

        data_to_cache = {
            'ticker': ticker_symbol.upper(),
            'name': info.get('longName', info.get('shortName', ticker_symbol)) if info else ticker_symbol,
            'current_price': current_price_f,
            'previous_close': previous_close_f,
            'daily_change': daily_change,
            'daily_change_percent': daily_change_percent,
            'last_fetched': datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) 
        }
        execute_db_query(""" REPLACE INTO stock_data_cache (ticker, name, current_price, previous_close, daily_change, daily_change_percent, last_fetched) VALUES (%(ticker)s, %(name)s, %(current_price)s, %(previous_close)s, %(daily_change)s, %(daily_change_percent)s, %(last_fetched)s) """, data_to_cache, commit=True)
        app.logger.info(f"YFINANCE FETCH: 成功獲取並快取 {ticker_symbol} 的數據")
        return data_to_cache
    except Exception as e:
        app.logger.error(f"從 yfinance 獲取 {ticker_symbol} 數據時發生錯誤: {e}", exc_info=True)
        return None

def get_stock_info(ticker_symbol):
    ticker_symbol = ticker_symbol.upper()
    cached_data = execute_db_query("SELECT * FROM stock_data_cache WHERE ticker = %s", (ticker_symbol,), fetch_one=True)
    if cached_data and cached_data['last_fetched']:
        last_fetched_dt = cached_data['last_fetched']
        if isinstance(last_fetched_dt, str): 
            try: last_fetched_dt = datetime.datetime.fromisoformat(last_fetched_dt)
            except ValueError: last_fetched_dt = datetime.datetime.min 

        if last_fetched_dt.tzinfo is not None: 
            last_fetched_dt = last_fetched_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        
        if (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - last_fetched_dt) < STOCK_CACHE_DURATION:
            if cached_data['current_price'] is not None: 
                app.logger.info(f"{ticker_symbol} 的快取命中。")
                for key in ['current_price', 'previous_close', 'daily_change', 'daily_change_percent']:
                    if cached_data.get(key) is not None: cached_data[key] = float(cached_data[key])
                return cached_data
    app.logger.info(f"{ticker_symbol} 的快取未命中，從 yfinance 獲取。")
    return get_stock_data_from_yf(ticker_symbol)

# --- StockAnalyzer Class Definition ---
class StockAnalyzer:
    def __init__(self, ticker: str, period: str = "3y", market: str = "AUTO", temperature: float = 0.6):
        self.logger = app.logger 
        self.ticker_input = ticker.strip()
        self.ticker = self.ticker_input
        self.period = period
        self.market = market.upper()
        self.temperature = max(0.0, min(1.0, temperature))

        if self.market == "AUTO":
            if re.fullmatch(r'\d{4,6}', self.ticker) and not self.ticker.endswith(".TW"):
                self.ticker = f"{self.ticker}.TW"
                self.market = "TW"
            elif self.ticker.endswith(".TW"):
                self.market = "TW"
            else:
                self.market = "US"
        elif self.market == "TW" and not self.ticker.endswith(".TW") and re.fullmatch(r'\d{4,6}', self.ticker):
            self.ticker = f"{self.ticker}.TW"

        self.stock = yf.Ticker(self.ticker)
        self.data = pd.DataFrame() 
        self.stock_df_for_ga = None 
        self.vix_series_for_ga = None 

        self.company_name = self.ticker 
        self.currency = 'USD' 
        self.pe_ratio, self.market_cap, self.eps, self.roe = None, None, None, None
        self.net_profit_margin_str, self.current_ratio_str = "N/A", "N/A"
        self.model = stock_analysis_model_gemini 

        try:
            self._get_data() 
            if not self.data.empty:
                self._get_financial_data()
                self._calculate_indicators() 
            else:
                self.logger.warning(f"[{self.ticker}] StockAnalyzer 的數據為空。跳過財務數據和指標計算。")
            self.logger.info(f"StockAnalyzer 為 {self.ticker} 初始化完成。市場: {self.market}")
        except Exception as e:
             self.logger.error(f"StockAnalyzer 為 ({self.ticker_input} -> {self.ticker}) 初始化失敗: {e}", exc_info=True)
             if not isinstance(self.data, pd.DataFrame) or self.data.empty: self.data = pd.DataFrame() 
             if self.company_name is None: self.company_name = self.ticker_input 

    def _get_data(self): 
        try:
            self.logger.info(f"StockAnalyzer ({self.ticker}): 獲取 '{self.period}' 的歷史數據")
            self.data = self.stock.history(period=self.period, timeout=20)
            if self.data.empty:
                self.logger.warning(f"StockAnalyzer ({self.ticker}): '{self.period}' 無歷史數據。嘗試 '1y'。")
                self.data = self.stock.history(period="1y", timeout=15)
                if self.data.empty:
                    temp_data = self.stock.history(period="6mo", timeout=10)
                    if not temp_data.empty:
                        self.data = temp_data
                        self.logger.warning(f"StockAnalyzer ({self.ticker}): 回退到 6 個月歷史數據。")
                    else:
                         raise ValueError(f"無法取得 {self.ticker} 任何有效歷史資料")

            info = {}
            try: info = self.stock.info 
            except Exception: self.logger.warning(f"StockAnalyzer ({self.ticker}): 獲取 .info 失敗。嘗試 .fast_info。")
            if not info: 
                try: info = self.stock.fast_info 
                except Exception: self.logger.error(f"StockAnalyzer ({self.ticker}): 獲取 .fast_info 也失敗。")

            self.company_name = info.get('longName', info.get('shortName', self.ticker)) if info else self.ticker
            self.currency = info.get('currency', 'TWD' if self.market == 'TW' else 'USD') if info else ('TWD' if self.market == 'TW' else 'USD')
            self.logger.info(f"StockAnalyzer ({self.ticker}): 數據/資訊獲取完成。公司: {self.company_name}")
        except ValueError as ve: 
            self.logger.error(f"StockAnalyzer ({self.ticker}): 獲取數據時發生 ValueError: {ve}")
            self.data = pd.DataFrame() 
            raise 
        except Exception as e: 
            self.logger.error(f"StockAnalyzer ({self.ticker}): 獲取數據時發生一般錯誤: {e}", exc_info=True)
            self.data = pd.DataFrame() 
    
    def _get_financial_data(self, retries=1, delay=0.5):
        financials, balance_sheet, info = pd.DataFrame(), pd.DataFrame(), {}
        try:
            self.logger.info(f"StockAnalyzer ({self.ticker}): 獲取財務數據...")
            try: info = self.stock.info or {} 
            except: self.logger.warning(f"StockAnalyzer ({self.ticker}): 在 _get_financial_data 中 stock.info 再次失敗")

            self.pe_ratio = info.get('trailingPE')
            self.market_cap = info.get('marketCap')
            self.eps = info.get('trailingEps')
            self.roe = info.get('returnOnEquity') 

            self.current_ratio_str = "N/A"
            if info.get('currentRatio') is not None:
                try: self.current_ratio_str = f"{float(info['currentRatio']):.2f}" if pd.notna(info['currentRatio']) else "N/A"
                except: self.logger.warning(f"StockAnalyzer ({self.ticker}): info 中的 currentRatio 無效: {info['currentRatio']}")
            
            for attempt in range(retries):
                try:
                    if financials.empty: financials = self.stock.financials
                    if balance_sheet.empty: balance_sheet = self.stock.balance_sheet
                    if not financials.empty and not balance_sheet.empty: break 
                except Exception as stmt_err:
                    self.logger.warning(f"StockAnalyzer ({self.ticker}): 獲取財報失敗 (嘗試 {attempt+1}/{retries}): {stmt_err}")
                if attempt < retries - 1: time.sleep(delay)

            if self.roe is None and not financials.empty and not balance_sheet.empty:
                net_income_key = next((k for k in ["Net Income","NetIncome","Net Income Common Stockholders"] if k in financials.index),None)
                equity_key = next((k for k in ["Total Stockholder Equity","Stockholders Equity","TotalEquityGrossMinorityInterest","Stockholders Equity"] if k in balance_sheet.index),None)
                if net_income_key and equity_key and not financials.columns.empty and not balance_sheet.columns.empty:
                    try:
                        ni = financials.loc[net_income_key].iloc[0]
                        eq = balance_sheet.loc[equity_key].iloc[0]
                        if pd.notna(ni) and pd.notna(eq) and eq != 0: self.roe = ni / eq
                    except IndexError: self.logger.warning(f"StockAnalyzer ({self.ticker}): 從財報計算 ROE 時發生 IndexError。")
                    except Exception as roe_calc_err: self.logger.warning(f"StockAnalyzer ({self.ticker}): 計算 ROE 時發生例外: {roe_calc_err}")
            
            self.net_profit_margin_str = "N/A"
            if not financials.empty:
                revenue_key = next((k for k in ["Total Revenue","Operating Revenue","TotalRevenue"] if k in financials.index),None)
                net_income_key_npm = next((k for k in ["Net Income","NetIncome","Net Income Common Stockholders"] if k in financials.index),None)
                if revenue_key and net_income_key_npm and not financials.columns.empty:
                    try:
                        ni_npm = financials.loc[net_income_key_npm].iloc[0]
                        rev = financials.loc[revenue_key].iloc[0]
                        if pd.notna(rev) and pd.notna(ni_npm) and rev != 0:
                            npm_val = (ni_npm / rev) * 100
                            self.net_profit_margin_str = f"{npm_val:.2f}%"
                    except IndexError: self.logger.warning(f"StockAnalyzer ({self.ticker}): 計算淨利率時發生 IndexError。")
                    except Exception as npm_calc_err: self.logger.warning(f"StockAnalyzer ({self.ticker}): 計算淨利率時發生例外: {npm_calc_err}")

            if self.current_ratio_str == "N/A" and not balance_sheet.empty:
                current_assets_key = next((k for k in ["Current Assets","Total Current Assets"] if k in balance_sheet.index),None)
                current_liab_key = next((k for k in ["Current Liabilities","Total Current Liabilities"] if k in balance_sheet.index),None)
                if current_assets_key and current_liab_key and not balance_sheet.columns.empty:
                    try:
                        ca = balance_sheet.loc[current_assets_key].iloc[0]
                        cl = balance_sheet.loc[current_liab_key].iloc[0]
                        if pd.notna(ca) and pd.notna(cl) and cl != 0:
                            cr_val = ca / cl
                            self.current_ratio_str = f"{cr_val:.2f}"
                    except IndexError: self.logger.warning(f"StockAnalyzer ({self.ticker}): 從資產負債表計算流動比率時發生 IndexError。")
                    except Exception as cr_calc_err: self.logger.warning(f"StockAnalyzer ({self.ticker}): 計算流動比率時發生例外: {cr_calc_err}")

            self.logger.info(f"StockAnalyzer ({self.ticker}): 財務數據處理完成。ROE: {self.roe}, NPM: {self.net_profit_margin_str}, CR: {self.current_ratio_str}")
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker}): 處理財務數據時發生錯誤: {e}", exc_info=True)
            for attr in ['pe_ratio','market_cap','eps','roe']: setattr(self, attr, getattr(self, attr, None))
            for attr_str in ['net_profit_margin_str','current_ratio_str']: setattr(self, attr_str, getattr(self, attr_str, "N/A"))

    def _calculate_indicators(self): 
        try:
            if self.data.empty:
                self.logger.warning(f"StockAnalyzer ({self.ticker}): 無可用數據計算指標。"); return
            df = self.data.copy() 
            required_cols_for_ta = ['Open', 'High', 'Low', 'Close'] 
            if not all(col in df.columns and df[col].notna().any() for col in required_cols_for_ta):
                self.logger.warning(f"StockAnalyzer ({self.ticker}): 缺少核心K線數據或全為NaN。無法計算多數指標。"); return

            min_len_ma = 50 
            min_len_rsi_macd = 26 

            if len(df.dropna(subset=['Close'])) >= 5: df.ta.sma(length=5, append=True, col_names='MA5', fillna=np.nan)
            else: df['MA5'] = np.nan
            if len(df.dropna(subset=['Close'])) >= 20: df.ta.sma(length=20, append=True, col_names='MA20', fillna=np.nan)
            else: df['MA20'] = np.nan
            if len(df.dropna(subset=['Close'])) >= min_len_ma : df.ta.sma(length=50, append=True, col_names='MA50', fillna=np.nan)
            else: df['MA50'] = np.nan
            if len(df.dropna(subset=['Close'])) >= 120: df.ta.sma(length=120, append=True, col_names='MA120', fillna=np.nan)
            else: df['MA120'] = np.nan

            if len(df.dropna(subset=['Close'])) >= 12 : df.ta.rsi(length=12, append=True, col_names='RSI', fillna=np.nan) 
            else: df['RSI'] = np.nan

            if len(df.dropna(subset=['Close'])) >= min_len_rsi_macd:
                macd = df.ta.macd(fast=12, slow=26, signal=9, append=True, fillna=np.nan)
                if macd is not None and not macd.empty:
                    df.rename(columns={'MACD_12_26_9':'MACD', 'MACDs_12_26_9':'MACD_signal', 'MACDh_12_26_9':'MACD_hist'}, inplace=True, errors='ignore')
            else: df['MACD'], df['MACD_signal'], df['MACD_hist'] = np.nan, np.nan, np.nan

            if len(df.dropna(subset=['High', 'Low', 'Close'])) >= 9: 
                stoch = df.ta.stoch(k=9, d=3, smooth_k=3, append=True, fillna=np.nan)
                if stoch is not None and not stoch.empty:
                    df.rename(columns={'STOCHk_9_3_3':'K', 'STOCHd_9_3_3':'D'}, inplace=True, errors='ignore')
                    if 'K' in df.columns and 'D' in df.columns and df['K'].notna().any() and df['D'].notna().any():
                        df['J'] = 3 * df['K'] - 2 * df['D']
                    else: df['J'] = np.nan
                else: df['K'], df['D'], df['J'] = np.nan, np.nan, np.nan
            else: df['K'], df['D'], df['J'] = np.nan, np.nan, np.nan

            if len(df.dropna(subset=['Close'])) >= 20: 
                bbands = df.ta.bbands(length=20, std=2, append=True, fillna=np.nan)
                if bbands is not None and not bbands.empty:
                    df.rename(columns={'BBL_20_2.0':'BB_lower', 'BBM_20_2.0':'BB_middle', 'BBU_20_2.0':'BB_upper'}, inplace=True, errors='ignore')
            else: df['BB_lower'], df['BB_middle'], df['BB_upper'] = np.nan, np.nan, np.nan
            
            if len(df.dropna(subset=['High', 'Low', 'Close'])) >= 14: df.ta.willr(length=14, append=True, col_names='WMSR', fillna=np.nan)
            else: df['WMSR'] = np.nan

            if 'Close' in df.columns and df['Close'].notna().any():
                if len(df) >= 12: df['PSY'] = df['Close'].diff().apply(lambda x:1 if x>0 else 0).rolling(12).sum()/12*100
                else: df['PSY'] = np.nan
                if len(df) >= 6:
                    bm = df['Close'].rolling(window=6).mean()
                    df['BIAS6'] = ((df['Close']-bm)/bm.replace(0,np.nan)*100).replace([np.inf,-np.inf],np.nan) 
                else: df['BIAS6'] = np.nan
            else: df['PSY'], df['BIAS6'] = np.nan, np.nan

            self.data = df 
            self.logger.info(f"StockAnalyzer ({self.ticker}): 指標計算完成。")
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker}): 計算指標時發生錯誤: {e}", exc_info=True)
            if not isinstance(self.data, pd.DataFrame): self.data = pd.DataFrame() 

    def _identify_patterns(self, days=30): 
        try:
            if self.data.empty or len(self.data) < 2: return ["數據不足"]
            df = self.data.tail(days).copy() 
            patterns = []; cols = df.columns
            if all(fld in cols for fld in ['MA5','MA20']) and len(df['MA5'].dropna()) >=2 and len(df['MA20'].dropna()) >=2 : 
                if df['MA5'].iloc[-2]<=df['MA20'].iloc[-2] and df['MA5'].iloc[-1]>df['MA20'].iloc[-1]: patterns.append("黃金交叉 (MA5>MA20)")
                if df['MA5'].iloc[-2]>=df['MA20'].iloc[-2] and df['MA5'].iloc[-1]<df['MA20'].iloc[-1]: patterns.append("死亡交叉 (MA5<MA20)")
            
            if all(fld in cols for fld in ['Close','BB_upper']) and pd.notna(df['Close'].iloc[-1]) and pd.notna(df['BB_upper'].iloc[-1]):
                if df['Close'].iloc[-1]>df['BB_upper'].iloc[-1]: patterns.append("突破布林帶上軌")
            if all(fld in cols for fld in ['Close','BB_lower']) and pd.notna(df['Close'].iloc[-1]) and pd.notna(df['BB_lower'].iloc[-1]):
                if df['Close'].iloc[-1]<df['BB_lower'].iloc[-1]: patterns.append("跌破布林帶下軌")

            if 'RSI' in cols and pd.notna(df['RSI'].iloc[-1]):
                if df['RSI'].iloc[-1] > 75: patterns.append("RSI 超買 (>75)") 
                if df['RSI'].iloc[-1] < 25: patterns.append("RSI 超賣 (<25)") 
            
            return patterns if patterns else ["近期無明顯技術形態"]
        except Exception as e: self.logger.error(f"StockAnalyzer ({self.ticker}): 識別形態時發生錯誤: {e}", exc_info=True); return ["無法識別形態"]

    def _generate_chart(self, days=180): 
        global charts_dir 
        try:
            if self.data.empty or len(self.data) < 2: raise ValueError(f"數據不足以生成圖表 ({self.ticker})")

            df = self.data.tail(days).copy() 
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
            cols = df.columns

            if all(c in cols for c in ['Open','High','Low','Close']) and df['Close'].notna().any():
                fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],name='K線'),row=1,col=1)
            elif 'Close' in cols and df['Close'].notna().any(): 
                fig.add_trace(go.Scatter(x=df.index,y=df['Close'],name='收盤價',line=dict(color='lightblue',width=1.5)),row=1,col=1)
            else:
                self.logger.warning(f"StockAnalyzer ({self.ticker}): 圖表無足夠數據繪製K線或收盤價線。")

            ma_map = {'MA5':'orange', 'MA20':'blue', 'MA50':'purple', 'MA120':'green'}
            for ma,color in ma_map.items():
                if ma in cols and df[ma].notna().any():
                    fig.add_trace(go.Scatter(x=df.index,y=df[ma],name=ma,line=dict(color=color,width=1)),row=1,col=1)
            
            if all(c in cols for c in ['BB_upper','BB_middle','BB_lower']) and df['BB_upper'].notna().any(): 
                fig.add_trace(go.Scatter(x=df.index,y=df['BB_upper'],name='布林上軌',line=dict(color='rgba(173,204,255,0.5)',width=1),fill=None),row=1,col=1)
                fig.add_trace(go.Scatter(x=df.index,y=df['BB_lower'],name='布林下軌',line=dict(color='rgba(173,204,255,0.5)',width=1,dash='dash'),fill='tonexty',fillcolor='rgba(173,204,255,0.05)'),row=1,col=1)
                fig.add_trace(go.Scatter(x=df.index,y=df['BB_middle'],name='布林中軌',line=dict(color='rgba(220,220,220,0.6)',width=1,dash='dot')),row=1,col=1)

            if 'Volume' in cols and df['Volume'].notna().any() and df['Volume'].sum() > 0:
                volume_colors = ['#2ca02c' if row['Close'] >= row['Open'] else '#d62728' for index, row in df.iterrows()] if all(c in cols for c in ['Open','Close']) and df['Open'].notna().any() and df['Close'].notna().any() else 'grey'
                fig.add_trace(go.Bar(x=df.index,y=df['Volume'],name='成交量',marker_color=volume_colors,marker_line_width=0),row=2,col=1)
            
            fig.update_layout(
                title=f'{self.company_name or self.ticker} ({self.ticker}) 技術分析 ({days}天)',
                xaxis_rangeslider_visible=False, template='plotly_dark', height=600,
                margin=dict(l=50,r=120,t=80,b=50), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(30,30,30,0.2)',
                font=dict(color='white',size=11),
                legend=dict(orientation="v",yanchor="top",y=1,xanchor="left",x=1.02,bgcolor="rgba(30,30,30,0.7)",bordercolor="rgba(200,200,200,0.5)",borderwidth=1)
            )
            fig.update_yaxes(title_text='價格 ('+self.currency+')',row=1,col=1,title_font_size=10,tickfont_size=9,side='left', gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(title_text='成交量',row=2,col=1,title_font_size=10,tickfont_size=9,side='left', gridcolor='rgba(255,255,255,0.1)')
            fig.update_xaxes(showgrid=True,gridwidth=0.5,gridcolor='rgba(255,255,255,0.1)')

            if not os.path.exists(charts_dir): os.makedirs(charts_dir) 
            chart_filename_base = f"{self.ticker.replace('.','_')}_simplified_chart_{secrets.token_hex(4)}.html" 
            chart_path_disk = os.path.join(charts_dir, chart_filename_base)
            fig.write_html(chart_path_disk,full_html=False,include_plotlyjs='cdn') 

            chart_url_path = f"charts/{chart_filename_base}" 
            self.logger.info(f"StockAnalyzer ({self.ticker}): 簡化版圖表已生成: {chart_url_path}")
            return chart_url_path
        except ValueError as ve: self.logger.error(f"StockAnalyzer ({self.ticker}): 生成簡化版圖表時發生 ValueError: {ve}"); return None
        except Exception as e: self.logger.error(f"StockAnalyzer ({self.ticker}): 生成簡化版圖表時發生未預期錯誤: {e}",exc_info=True); return None

    def _get_stock_news(self, days=7, num_news=5): 
        try:
            query_base = self.company_name if self.market!="TW" else self.ticker.replace('.TW','') 
            lang, geo, ceid_sfx = ('en-US','US','US:en') if self.market != "TW" else ('zh-TW','TW','TW:zh-Hant') 

            query_suffix = "" 
            if self.market == "TW" and not re.fullmatch(r'\d{4,6}', query_base): query_suffix = " 股票" 
            elif self.market != "TW" and not any(kw in query_base.lower() for kw in [" stock", " inc", " corp", " ltd"]): query_suffix = " stock" 
            final_query = query_base + query_suffix

            url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(final_query)}&hl={lang}&gl={geo}&ceid={ceid_sfx}"
            self.logger.info(f"StockAnalyzer ({self.ticker}): 使用 URL 獲取新聞: {url}")

            feed_data = feedparser.parse(url) 
            relevant_news, now_utc = [], datetime.datetime.now(datetime.timezone.utc)

            if not feed_data.entries: self.logger.warning(f"StockAnalyzer ({self.ticker}): Google News 未找到 '{final_query}' 的條目"); return []
            for entry in feed_data.entries:
                try:
                    pub_time_utc = now_utc 
                    if hasattr(entry,'published_parsed') and entry.published_parsed:
                        try: pub_time_utc = datetime.datetime(*entry.published_parsed[:6],tzinfo=datetime.timezone.utc)
                        except : pass 
                    if (now_utc - pub_time_utc).days <= days: 
                        relevant_news.append({'title':entry.get('title',"無標題").strip(),'link':entry.get('link',"#"),'date':pub_time_utc.strftime('%Y-%m-%d'), 'source':entry.get('source',{}).get('title','Google News')})
                    if len(relevant_news) >= num_news: break 
                except Exception as item_err: self.logger.error(f"StockAnalyzer ({self.ticker}): 處理新聞條目 '{entry.get('title','N/A')}' 時發生錯誤: {item_err}")
            self.logger.info(f"StockAnalyzer ({self.ticker}): 找到 {len(relevant_news)} 條新聞。"); return relevant_news
        except Exception as e: self.logger.error(f"StockAnalyzer ({self.ticker}): 獲取新聞時發生錯誤: {e}",exc_info=True); return []

    def _get_ai_analysis(self, news_list: list = None): 
        if not self.model: return "AI 分析模型未能成功載入或初始化。" 
        try:
            if self.data.empty or len(self.data) < 2: return f"股票 ({self.ticker}) 歷史數據不足，無法進行AI分析。"

            last_close = self.data['Close'].iloc[-1] if not self.data.empty and 'Close' in self.data.columns and not self.data['Close'].empty and pd.notna(self.data['Close'].iloc[-1]) else None
            daily_change_pct, weekly_change_pct, monthly_change_pct = None, None, None

            if last_close is not None and 'Close' in self.data.columns and len(self.data['Close'].dropna()) >= 23 : 
                if len(self.data['Close'].dropna())>=2: daily_change_pct = (self.data['Close'].iloc[-1]/self.data['Close'].iloc[-2]-1)*100 if pd.notna(self.data['Close'].iloc[-2]) and self.data['Close'].iloc[-2]!=0 else None
                if len(self.data['Close'].dropna())>=6: weekly_change_pct = (self.data['Close'].iloc[-1]/self.data['Close'].iloc[-6]-1)*100 if pd.notna(self.data['Close'].iloc[-6]) and self.data['Close'].iloc[-6]!=0 else None
                if len(self.data['Close'].dropna())>=23: monthly_change_pct = (self.data['Close'].iloc[-1]/self.data['Close'].iloc[-23]-1)*100 if pd.notna(self.data['Close'].iloc[-23]) and self.data['Close'].iloc[-23]!=0 else None
            
            temp_change_list = [daily_change_pct, weekly_change_pct, monthly_change_pct]
            for i in range(len(temp_change_list)):
                if temp_change_list[i] is not None and (temp_change_list[i] > 200 or temp_change_list[i] < -90): 
                    temp_change_list[i] = None 
            daily_change_pct, weekly_change_pct, monthly_change_pct = temp_change_list

            technical_patterns = ", ".join(self._identify_patterns()) if self._identify_patterns() else "近期無明顯技術形態"
            
            def format_value(value, precision=2, is_currency=False, is_percentage=False, currency_symbol=''):
                if value is None or pd.isna(value) or str(value).upper()=='N/A': return "N/A"
                try:
                    num = float(value)
                    if is_percentage: return f"{num:.{precision}f}%"
                    symbol_to_use = currency_symbol if is_currency else ''
                    if abs(num) >= 1e12 and is_currency: return f"{symbol_to_use}{num/1e12:.1f}兆"
                    if abs(num) >= 1e8 and is_currency: return f"{symbol_to_use}{num/1e8:.1f}億"
                    if abs(num) >= 1e4 and is_currency and self.currency=='TWD': return f"{symbol_to_use}{num/1e4:.1f}萬"
                    return f"{symbol_to_use}{num:,.{precision}f}" if (abs(num) < 1e6 or not is_currency) else \
                           (f"{symbol_to_use}{num/1e9:.1f}B" if abs(num) >= 1e9 else f"{symbol_to_use}{num/1e6:.1f}M") 
                except (ValueError, TypeError): return str(value)

            currency_sym = '$' if self.currency == 'USD' else ('NT$' if self.currency == 'TWD' else self.currency)
            latest_data_row = self.data.iloc[-1] if not self.data.empty else pd.Series(dtype='object')

            news_summary_str = "近期無相關新聞。"
            if news_list:
                titles = [f"{idx+1}. {news_item.get('title','N/A')} (來源: {news_item.get('source','N/A')})" for idx, news_item in enumerate(news_list[:3])] 
                if titles: news_summary_str = "\n".join(titles)

            prompt_text = f"""
請您扮演一位專業的股票市場分析師。針對以下股票提供一份全面、客觀、數據驅動的繁體中文分析報告。
報告應結構清晰，條列分明，並包含明確的投資觀點總結及必要的免責聲明。

**股票基本資訊:**
- 公司名稱: {self.company_name or 'N/A'}
- 股票代號: {self.ticker}
- 交易市場: {'台灣股市(TWSE/TPEX)' if self.market == 'TW' else '美國股市(US Market)'}
- 目前股價: {format_value(last_close, is_currency=True, currency_symbol=currency_sym)} ({self.currency})
- 近期價格變動:
  - 日漲跌幅: {format_value(daily_change_pct, is_percentage=True)}
  - 週漲跌幅: {format_value(weekly_change_pct, is_percentage=True)}
  - 月漲跌幅: {format_value(monthly_change_pct, is_percentage=True)}

**技術面分析數據:**
- RSI (12日): {format_value(latest_data_row.get('RSI'))}
- 主要技術形態: {technical_patterns}

**基本面數據:**
- 本益比 (P/E Ratio): {format_value(self.pe_ratio)}
- 市值 (Market Cap): {format_value(self.market_cap, is_currency=True, currency_symbol=currency_sym, precision=0)}
- 每股盈餘 (EPS): {format_value(self.eps, is_currency=True, currency_symbol=currency_sym)}
- 股東權益報酬率 (ROE): {format_value(self.roe*100 if isinstance(self.roe, (int,float)) else self.roe, is_percentage=True) if self.roe is not None else "N/A"}
- 淨利率 (Net Profit Margin): {self.net_profit_margin_str or 'N/A'}
- 流動比率 (Current Ratio): {self.current_ratio_str or 'N/A'}

**近期相關新聞摘要 (最多三條，請解讀其潛在影響):**
{news_summary_str}

**請根據以上資訊，提供如下結構的分析報告：**

1.  **公司業務概要:** (簡述公司核心業務及市場定位，1-2句話)
2.  **投資亮點與機會 (Bullish Points):** (列舉2-4點支撐看多觀點的理由，結合基本面、技術面或新聞事件)
3.  **潛在風險與挑戰 (Bearish Points):** (列舉2-4點潛在的風險或看空理由)
4.  **技術面綜合評述:** (總結當前股價趨勢、關鍵支撐/壓力位、市場動能等)
5.  **基本面綜合評述:** (評估公司盈利能力、財務健康狀況、估值水平。若數據為N/A請註明)
6.  **新聞事件解讀:** (分析提供的新聞摘要可能對股價產生的短期或長期影響，市場情緒反應)
7.  **總結與投資展望:** (綜合評價，提出短期 (1週-1個月) 和中期 (3-6個月) 的展望，並給出明確的操作建議。**務必包含免責聲明：此分析僅為模擬AI提供，基於歷史數據和公開信息，不構成任何真實投資建議，用戶應獨立判斷並自負風險。**)
"""
            generation_config_settings = genai.types.GenerationConfig(temperature=self.temperature)
            self.logger.info(f"StockAnalyzer ({self.ticker}): 使用 Gemini 模型生成 AI 分析 (溫度: {self.temperature}).")

            current_safety_settings = safety_settings_gemini if 'safety_settings_gemini' in globals() and safety_settings_gemini is not None else None

            response = self.model.generate_content(prompt_text, generation_config=generation_config_settings, safety_settings=current_safety_settings)
            
            analysis_text = ""
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text'): analysis_text += part.text

            if hasattr(response,'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                 feedback_reason = response.prompt_feedback.block_reason
                 self.logger.warning(f"StockAnalyzer ({self.ticker}): AI 分析內容生成被阻擋。原因: {feedback_reason}")
                 return f"AI 分析請求因安全或內容政策被阻擋。原因: {feedback_reason}"
            
            if not analysis_text:
                 self.logger.warning(f"StockAnalyzer ({self.ticker}): AI 分析生成了空文本。完整回應: {response}")
                 return "AI 分析未能生成有效內容，請稍後再試。"
            return analysis_text.strip()
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker}): 生成 AI 分析時發生錯誤: {e}",exc_info=True)
            return f"生成AI股票分析報告時發生未預期錯誤: {str(e)}"

    def get_stock_summary(self): 
        try:
            if self.data.empty or len(self.data) < 2:
                self.logger.warning(f"StockAnalyzer ({self.ticker_input}): 數據為空或不足以生成摘要。")
                return {"ticker": self.ticker_input,"company_name":self.company_name or self.ticker_input,"error":f"數據不足 ({self.ticker_input})"}

            latest_row = self.data.iloc[-1] if not self.data.empty else pd.Series(dtype='object')
            price_change_val_pct = None
            price_change_abs_val = None
            if not self.data.empty and len(self.data) >= 2 and 'Close' in self.data.columns and self.data['Close'].notna().iloc[-2:].size == 2: 
                 prev_close_val_gs = self.data['Close'].iloc[-2]
                 latest_close_val_gs = latest_row.get('Close')
                 if pd.notna(latest_close_val_gs) and pd.notna(prev_close_val_gs) and prev_close_val_gs != 0:
                     price_change_val_pct = (latest_close_val_gs / prev_close_val_gs - 1) * 100
                     price_change_abs_val = latest_close_val_gs - prev_close_val_gs

            chart_relative_path = self._generate_chart() 
            news_items = self._get_stock_news() 
            ai_report_text = self._get_ai_analysis(news_list=news_items) 

            summary_map = {
                'ticker': self.ticker,
                'company_name': self.company_name or 'N/A',
                'currency': self.currency,
                'current_price': latest_row.get('Close') if pd.notna(latest_row.get('Close')) else None,
                'price_change': f"{price_change_abs_val:+.2f} ({price_change_val_pct:+.2f}%)" if price_change_val_pct is not None and price_change_abs_val is not None else "N/A",
                'price_change_value': price_change_abs_val,
                'volume': int(latest_row.get('Volume',0)) if pd.notna(latest_row.get('Volume')) else 0,
                'pe_ratio': self.pe_ratio if pd.notna(self.pe_ratio) else None,
                'market_cap': self.market_cap if pd.notna(self.market_cap) else None,
                'eps': self.eps if pd.notna(self.eps) else None,
                'roe': self.roe if pd.notna(self.roe) else None,
                'net_profit_margin': self.net_profit_margin_str or 'N/A',
                'current_ratio': self.current_ratio_str or 'N/A',
                'rsi': latest_row.get('RSI') if pd.notna(latest_row.get('RSI')) else None, 
                'patterns': self._identify_patterns(),
                'chart_path': chart_relative_path, 
                'chart_url': None, 
                'news': news_items,
                'analysis': ai_report_text
            }
            summary_map["roe_display"] = f"{summary_map['roe']*100:.2f}%" if isinstance(summary_map["roe"],(int,float)) else "N/A"
            return summary_map
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker_input}): get_stock_summary 時發生錯誤: {e}",exc_info=True)
            return {"ticker": self.ticker_input,"company_name":self.company_name or self.ticker_input,"error":f"獲取股票綜合分析時發生錯誤: {str(e)}"}


# --- FinSimU Core User Auth & Helper Functions (單一市場模型) ---
def get_current_user_id(): return session.get('user_id')

def get_user_data(user_id):
    user = execute_db_query("""SELECT id, nickname, investment_style, market_type,
                               initial_capital_tw, cash_balance_tw,
                               initial_capital_us, cash_balance_us, created_at
                               FROM users WHERE id = %s""", (user_id,), fetch_one=True)
    if user:
        for cap_key in ['initial_capital_tw', 'cash_balance_tw', 'initial_capital_us', 'cash_balance_us']:
            if user.get(cap_key) is not None:
                 user[cap_key] = float(user[cap_key])
    return user

def login_required(f):
    @wraps(f)
    def decorated_function(*args,**kwargs):
        if 'user_id' not in session or session.get('user_id') is None:
            is_api = request.endpoint and (request.endpoint.startswith('api_') or \
                                           (hasattr(request,'blueprint') and request.blueprint=='api')) 
            if is_api:
                error_response_data = {'success': False, 'message': "需要身份驗證。"} 
                error_response = make_response(jsonify(error_response_data), 401)
                return set_no_cache_headers(error_response)
            return redirect(url_for('finsimu_login_route',next=request.url))
        return f(*args,**kwargs)
    return decorated_function

def _record_portfolio_value_with_cursor(cursor,user_id,market_type,total_value):
    try:
        cursor.execute("INSERT INTO portfolio_history(user_id,market_type,timestamp,total_value)VALUES(%s,%s,%s,%s)",(user_id,market_type,datetime.datetime.now(),float(total_value)))
        return True
    except pymysql.Error as e:
        app.logger.error(f"Error recording portfolio value for user {user_id}, market {market_type}: {e}")
        return False

def update_holdings_in_db(cursor, user_id, market_type, ticker, stock_name, shares_change, current_price_for_trade):
    try:
        cursor.execute("SELECT shares, average_cost FROM holdings WHERE user_id = %s AND market_type = %s AND ticker = %s FOR UPDATE",
                       (user_id, market_type, ticker))
        current_holding = cursor.fetchone()

        if current_holding: 
            new_shares = current_holding['shares'] + shares_change
            if new_shares < 0: 
                app.logger.error(f"錯誤：使用者 {user_id}/{market_type} 試圖賣出超過持有的 {ticker} 股票。持有: {current_holding['shares']}, 試圖變動: {shares_change}")
                return False 

            if new_shares == 0: 
                cursor.execute("DELETE FROM holdings WHERE user_id = %s AND market_type = %s AND ticker = %s", (user_id, market_type, ticker))
            else: 
                final_avg_cost = float(current_holding['average_cost'])
                if shares_change > 0: 
                    existing_value = float(current_holding['average_cost']) * current_holding['shares']
                    new_purchase_value = current_price_for_trade * shares_change
                    final_avg_cost = (existing_value + new_purchase_value) / new_shares
                cursor.execute("UPDATE holdings SET shares = %s, average_cost = %s, stock_name = %s WHERE user_id = %s AND market_type = %s AND ticker = %s",
                               (new_shares, final_avg_cost, stock_name, user_id, market_type, ticker))
        elif shares_change > 0: 
            cursor.execute("INSERT INTO holdings (user_id, market_type, ticker, stock_name, shares, average_cost) VALUES (%s, %s, %s, %s, %s, %s)",
                           (user_id, market_type, ticker, stock_name, shares_change, float(current_price_for_trade)))
        elif shares_change < 0: 
            app.logger.error(f"錯誤：更新持股時，使用者 {user_id}/{market_type} 試圖賣出不存在的持股 {ticker}。")
            return False 
        else: 
            app.logger.info(f"股票 {ticker}（使用者 {user_id}/{market_type}）持股數量無變動。")

        return True
    except pymysql.Error as e:
        app.logger.error(f"資料庫錯誤 (update_holdings_in_db)，使用者 {user_id}/{market_type}，股票 {ticker}: {e}", exc_info=True)
        return False


# --- Flask Routes ---
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    app.logger.info(f"註冊嘗試: {data.get('nickname')}")
    nickname = data.get('nickname', '').strip()
    password = data.get('password', '')
    investment_style = data.get('investmentStyle', '保守型') 
    market_type = data.get('marketType', '').upper()

    if not nickname or not password:
        resp_data = {'success': False, 'message': '暱稱和密碼為必填項。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))
    if market_type not in ['TW', 'US']:
        resp_data = {'success': False, 'message': '請選擇有效的市場（TW 或 US）。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))
    if len(password) < 6:
        resp_data = {'success': False, 'message': '密碼長度至少需6位。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))

    fixed_initial_capital = 10000000.00
    initial_capital_tw = fixed_initial_capital if market_type == 'TW' else 0.00
    cash_balance_tw = initial_capital_tw
    initial_capital_us = fixed_initial_capital if market_type == 'US' else 0.00
    cash_balance_us = initial_capital_us

    conn = get_db_connection()
    if not conn:
        resp_data = {'success': False, 'message': '資料庫連接錯誤。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))

    user_id = None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE nickname = %s", (nickname,))
            if cursor.fetchone():
                resp_data = {'success': False, 'message': '此暱稱已被註冊，請選擇其他暱稱。'}
                return set_no_cache_headers(make_response(jsonify(resp_data), 409))

            password_hash = generate_password_hash(password)
            cursor.execute("""INSERT INTO users (nickname, password_hash, investment_style, market_type,
                                                initial_capital_tw, cash_balance_tw,
                                                initial_capital_us, cash_balance_us)
                              VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                           (nickname, password_hash, investment_style, market_type,
                            initial_capital_tw, cash_balance_tw,
                            initial_capital_us, cash_balance_us))
            user_id = cursor.lastrowid
            if not user_id: raise pymysql.Error("無法在用戶插入後獲取 lastrowid。")

            if not _record_portfolio_value_with_cursor(cursor, user_id, market_type, fixed_initial_capital):
                raise pymysql.Error(f"註冊期間為市場 {market_type} 記錄初始投資組合歷史失敗。")

        conn.commit()
        app.logger.info(f"新用戶 '{nickname}' (ID: {user_id}, 市場: {market_type}) 註冊成功。")
        resp_data = {'success': True, 'message': '註冊成功！請登錄。', 'username': nickname}
        return set_no_cache_headers(make_response(jsonify(resp_data)))
    except pymysql.Error as e:
        app.logger.error(f"用戶 {nickname} 註冊期間資料庫錯誤: {e}", exc_info=True)
        if conn: conn.rollback()
        em = e.args[1] if len(e.args)>1 and isinstance(e.args[1],str) else "未知資料庫錯誤"
        resp_data = {'success': False, 'message': f'資料庫錯誤: {em}'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))
    except Exception as e:
        app.logger.error(f"註冊期間發生未預期錯誤: {e}", exc_info=True)
        if conn: conn.rollback()
        resp_data = {'success': False, 'message': '註冊期間發生未預期錯誤。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))
    finally:
        if conn and conn.open: conn.close()

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    nickname = data.get('nickname', '').strip()
    password = data.get('password', '')
    app.logger.info(f"登錄嘗試，暱稱: {nickname}")

    if not nickname or not password:
        resp_data = {'success': False, 'message': '暱稱和密碼為必填項。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))

    user = execute_db_query("SELECT id, nickname, password_hash, market_type FROM users WHERE nickname = %s", (nickname,), fetch_one=True)

    if user and check_password_hash(user['password_hash'], password):
        session.clear()
        session['user_id'] = user['id']
        session['username'] = user['nickname']
        session['market_type'] = user['market_type'] 

        execute_db_query("UPDATE users SET last_login_at = %s WHERE id = %s", (datetime.datetime.now(), user['id']), commit=True)

        app.logger.info(f"用戶 '{nickname}' (ID: {user['id']}, 市場: {user['market_type']}) 登錄成功。 Session: {dict(session)}")
        resp_data = {'success': True, 'message': '登錄成功。', 'username': user['nickname'], 'marketType': user['market_type']}
        return set_no_cache_headers(make_response(jsonify(resp_data)))
    else:
        app.logger.warning(f"暱稱 {nickname} 登錄失敗。")
        resp_data = {'success': False, 'message': '暱稱或密碼無效。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 401))

@app.route('/api/user_session', methods=['GET'])
@login_required
def get_user_session_api():
    user_id = get_current_user_id()
    app.logger.info(f"API /user_session: 獲取用戶 ID {user_id} 的 session")

    user = get_user_data(user_id)
    if not user:
        session.clear() 
        resp_data = {'loggedIn': False, 'message': '用戶數據不一致，Session 已清除。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))

    market_type = user['market_type'] 
    cash_balance_key = f'cash_balance_{market_type.lower()}'
    initial_capital_key = f'initial_capital_{market_type.lower()}'

    market_specific_data = {
        'cashBalance': user[cash_balance_key],
        'initialCapital': user[initial_capital_key],
        'portfolio': {}, 
        'tradeHistory': [], 
        'portfolioHistory': [] 
    }

    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT ticker, stock_name, shares, average_cost FROM holdings WHERE user_id = %s AND market_type = %s AND shares > 0", (user_id, market_type))
                for row_h in cursor.fetchall():
                    market_specific_data['portfolio'][row_h['ticker']] = {
                        'shares': row_h['shares'], 'avgCost': float(row_h['average_cost']), 'name': row_h['stock_name']
                    }

                cursor.execute("SELECT timestamp, total_value FROM portfolio_history WHERE user_id = %s AND market_type = %s ORDER BY timestamp ASC", (user_id, market_type)) 
                for row_ph in cursor.fetchall():
                    market_specific_data['portfolioHistory'].append({'timestamp': row_ph['timestamp'].isoformat(), 'value': float(row_ph['total_value'])})
                if not market_specific_data['portfolioHistory']:
                    created_at_ts = user.get('created_at', datetime.datetime.now()) 
                    market_specific_data['portfolioHistory'].append({'timestamp': created_at_ts.isoformat(), 'value': float(user[initial_capital_key])})
                
                cursor.execute("SELECT timestamp, trade_type, ticker, stock_name, shares, price_per_share, mood, reason FROM trade_history WHERE user_id = %s AND market_type = %s ORDER BY timestamp DESC LIMIT 20", (user_id, market_type))
                for row_th in cursor.fetchall():
                    market_specific_data['tradeHistory'].append({
                        'timestamp': row_th['timestamp'].isoformat(), 'type': row_th['trade_type'], 'ticker': row_th['ticker'],
                        'name': row_th['stock_name'], 'shares': row_th['shares'], 'price': float(row_th['price_per_share']),
                        'mood': row_th['mood'], 'reason': row_th['reason']
                    })
        except pymysql.Error as db_err:
            app.logger.error(f"資料庫錯誤，獲取用戶 {user_id} 市場 {market_type} 的擴展 session 數據失敗: {db_err}")
        finally:
            if conn: conn.close()
    
    resp_data = {
        'loggedIn': True,
        'currentUser': {
            'id': user['id'], 'name': user['nickname'],
            'investmentStyle': user['investment_style'],
            'marketType': market_type, 
            'account': market_specific_data 
        }
    }
    return set_no_cache_headers(make_response(jsonify(resp_data)))

@app.route('/api/stock_quote/<path:ticker>', methods=['GET'])
@login_required
def stock_quote_api(ticker):
    app.logger.info(f"API 請求股票報價: {ticker}")
    data = get_stock_info(ticker)
    if data:
        resp_data = {'success': True, 'data': data}
        return set_no_cache_headers(make_response(jsonify(resp_data)))
    resp_data = {'success': False, 'message': f'無法獲取股票 {ticker} 的數據。'}
    return set_no_cache_headers(make_response(jsonify(resp_data), 404))

@app.route('/api/search_stocks', methods=['GET'])
@login_required
def search_stocks_api():
    query = request.args.get('q', '').strip().lower()
    user_id = get_current_user_id()
    user = get_user_data(user_id)
    if not user:
        resp_data = {'success': False, 'message': '找不到用戶。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 401))

    market_type = user['market_type']
    app.logger.info(f"用戶 {user_id} (市場: {market_type}) 搜索股票，查詢: '{query}'")

    if not query or len(query) < 1: 
        resp_data = {'success': True, 'stocks': []}
        return set_no_cache_headers(make_response(jsonify(resp_data)))
    results = []
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                if market_type == 'TW':
                    sql = """
                        (SELECT ticker, name, current_price, daily_change_percent FROM stock_data_cache
                         WHERE ticker LIKE %s AND ticker LIKE '%%.TW' ORDER BY ticker LIMIT 5)
                        UNION ALL
                        (SELECT ticker, name, current_price, daily_change_percent FROM stock_data_cache
                         WHERE name LIKE %s AND ticker LIKE '%%.TW' AND ticker NOT LIKE %s ORDER BY name LIMIT 5)
                        LIMIT 10; 
                    """ 
                    cursor.execute(sql, (f"{query}%", f"%{query}%", f"{query}%"))
                else: 
                    sql = """
                        (SELECT ticker, name, current_price, daily_change_percent FROM stock_data_cache
                         WHERE ticker LIKE %s AND ticker NOT LIKE '%%.TW' ORDER BY ticker LIMIT 5)
                        UNION ALL
                        (SELECT ticker, name, current_price, daily_change_percent FROM stock_data_cache
                         WHERE name LIKE %s AND ticker NOT LIKE '%%.TW' AND ticker NOT LIKE %s ORDER BY name LIMIT 5)
                        LIMIT 10;
                    """
                    cursor.execute(sql, (f"{query}%", f"%{query}%", f"{query}%"))

                for row in cursor.fetchall():
                    results.append({
                        'ticker': row['ticker'], 'name': row['name'],
                        'price': float(row['current_price']) if row['current_price'] is not None else 'N/A',
                        'change': float(row['daily_change_percent']) if row['daily_change_percent'] is not None else 0.0,
                        'change_type': "positive" if row['daily_change_percent'] is not None and float(row['daily_change_percent']) >= 0 else "negative"
                    })
        except pymysql.Error as e:
            app.logger.error(f"從快取搜索股票錯誤，查詢 '{query}'，市場 {market_type}: {e}")
        finally:
            conn.close()
    
    if len(results) < 2 and len(query) <= 7 : 
        potential_ticker = query.upper()
        if market_type == 'TW' and not potential_ticker.endswith('.TW'):
            if not any(char.isdigit() for char in potential_ticker.split('.')[0]) or \
               any(potential_ticker.endswith(s) for s in ['.O', '.N', '.K', '.L']): 
                 pass 
            else:
                potential_ticker_tw = f"{potential_ticker}.TW"
                stock_data_yf_tw = get_stock_data_from_yf(potential_ticker_tw) 
                if stock_data_yf_tw and stock_data_yf_tw.get('current_price') is not None:
                    if not any(r['ticker'] == stock_data_yf_tw['ticker'] for r in results):
                        results.append({'ticker': stock_data_yf_tw['ticker'], 'name': stock_data_yf_tw['name'], 'price': stock_data_yf_tw['current_price'],
                                        'change': stock_data_yf_tw.get('daily_change_percent', 0.0),
                                        'change_type': "positive" if stock_data_yf_tw.get('daily_change_percent', 0) >= 0 else "negative"})
                        resp_data = {'success': True, 'stocks': results}
                        return set_no_cache_headers(make_response(jsonify(resp_data))) 

        if not (market_type == 'TW' and potential_ticker.endswith('.TW')): 
            stock_data_yf = get_stock_data_from_yf(potential_ticker)
            if stock_data_yf and stock_data_yf.get('current_price') is not None:
                if market_type == 'US' and stock_data_yf['ticker'].endswith('.TW'):
                    pass 
                elif not any(r['ticker'] == stock_data_yf['ticker'] for r in results): 
                    results.append({'ticker': stock_data_yf['ticker'], 'name': stock_data_yf['name'], 'price': stock_data_yf['current_price'],
                                    'change': stock_data_yf.get('daily_change_percent', 0.0),
                                    'change_type': "positive" if stock_data_yf.get('daily_change_percent', 0) >= 0 else "negative"})
    
    app.logger.info(f"搜索 '{query}' (市場: {market_type}) 返回 {len(results)} 個結果。")
    resp_data = {'success': True, 'stocks': results}
    return set_no_cache_headers(make_response(jsonify(resp_data)))

@app.route('/api/trade', methods=['POST'])
@login_required
def trade_api():
    user_id = get_current_user_id()
    data = request.get_json()
    app.logger.info(f"用戶 {user_id} 交易 API 請求: {data}")

    trade_type = data.get('type')
    ticker = data.get('ticker', '').upper()
    try: shares = int(data.get('shares', 0))
    except (ValueError, TypeError):
        resp_data = {'success': False, 'message': '無效的股數格式。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))
    mood, reason = data.get('mood'), data.get('reason')

    user_account = get_user_data(user_id)
    if not user_account:
        resp_data = {'success': False, 'message': '無法獲取用戶帳戶詳細資料。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))

    market_type = user_account['market_type']

    if not all([trade_type, ticker, shares > 0]):
        resp_data = {'success': False, 'message': '缺少交易信息或股數無效。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))
    if trade_type not in ['buy', 'sell']:
        resp_data = {'success': False, 'message': '無效的交易類型。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))

    if market_type == 'TW' and not ticker.endswith('.TW'):
        resp_data = {'success': False, 'message': f'您的台股帳戶 ({ticker}) 股票代號無效。台股股票必須以 .TW 結尾。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))
    if market_type == 'US' and ticker.endswith('.TW'):
        resp_data = {'success': False, 'message': f'您的美股帳戶 ({ticker}) 股票代號無效。您不能交易 .TW 股票。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))

    stock_live_data = get_stock_info(ticker)
    if not stock_live_data or stock_live_data.get('current_price') is None:
        resp_data = {'success': False, 'message': f'無法獲取 {ticker} 的當前價格。交易已取消。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 404))

    price_at_trade = float(stock_live_data['current_price'])
    stock_name_at_trade = stock_live_data.get('name', ticker) 
    total_transaction_value = shares * price_at_trade
    commission = round(total_transaction_value * 0.001425, 2) if market_type == 'TW' else round(max(1.0, total_transaction_value * 0.005), 2)


    cash_balance_key = f'cash_balance_{market_type.lower()}'
    current_cash = user_account[cash_balance_key]

    conn = get_db_connection()
    if not conn:
        resp_data = {'success': False, 'message': '資料庫連接錯誤。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))

    try:
        with conn.cursor() as cursor:
            if trade_type == 'buy':
                if total_transaction_value + commission > current_cash:
                    resp_data = {'success': False, 'message': '現金餘額不足。'}
                    return set_no_cache_headers(make_response(jsonify(resp_data), 400))
                cursor.execute(f"UPDATE users SET {cash_balance_key} = {cash_balance_key} - %s WHERE id = %s", (total_transaction_value + commission, user_id))

            elif trade_type == 'sell':
                cursor.execute("SELECT shares FROM holdings WHERE user_id = %s AND market_type = %s AND ticker = %s FOR UPDATE", (user_id, market_type, ticker))
                current_holding = cursor.fetchone()
                owned_shares = current_holding['shares'] if current_holding else 0
                if shares > owned_shares:
                    resp_data = {'success': False, 'message': f'持股不足以賣出。您持有 {owned_shares} 股。'}
                    return set_no_cache_headers(make_response(jsonify(resp_data), 400))
                cursor.execute(f"UPDATE users SET {cash_balance_key} = {cash_balance_key} + %s WHERE id = %s", (total_transaction_value - commission, user_id))

            sql_trade_hist = """INSERT INTO trade_history (user_id, market_type, timestamp, trade_type, ticker, stock_name, shares, price_per_share, total_value, mood, reason, commission)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql_trade_hist, (user_id, market_type, datetime.datetime.now(), trade_type, ticker, stock_name_at_trade, shares, price_at_trade, total_transaction_value, mood, reason, commission))

            shares_change = shares if trade_type == 'buy' else -shares
            holdings_update_success = update_holdings_in_db(cursor, user_id, market_type, ticker, stock_name_at_trade, shares_change, price_at_trade)

            if not holdings_update_success:
                conn.rollback() 
                app.logger.critical(f"用戶 {user_id}/{market_type} 持股更新失敗，交易: {data}。事務已回滾。")
                resp_data = {'success': False, 'message': '持股更新期間交易處理失敗。交易已取消。'}
                return set_no_cache_headers(make_response(jsonify(resp_data), 500))

            cursor.execute(f"SELECT {cash_balance_key} FROM users WHERE id = %s", (user_id,)) 
            updated_cash_balance_from_db = cursor.fetchone()[cash_balance_key]

            new_total_portfolio_value = float(updated_cash_balance_from_db)
            cursor.execute("SELECT ticker, shares FROM holdings WHERE user_id = %s AND market_type = %s AND shares > 0", (user_id, market_type))
            for holding_val in cursor.fetchall():
                h_stock_data = get_stock_info(holding_val['ticker']) 
                h_price = float(h_stock_data['current_price']) if h_stock_data and h_stock_data.get('current_price') is not None else price_at_trade 
                new_total_portfolio_value += holding_val['shares'] * h_price
            
            if not _record_portfolio_value_with_cursor(cursor, user_id, market_type, new_total_portfolio_value):
                conn.rollback() 
                app.logger.error(f"為用戶 {user_id}/{market_type} 記錄交易後投資組合價值失敗。事務已回滾。")
                resp_data = {'success': False, 'message': '交易已處理但更新投資組合歷史失敗。交易已取消。'}
                return set_no_cache_headers(make_response(jsonify(resp_data), 500))

            conn.commit() 
            app.logger.info(f"用戶 {user_id} 交易 {trade_type} {shares} 股 {ticker} @ {price_at_trade} (市場: {market_type}) 成功。")
            
            final_user_data = get_user_data(user_id) 
            if not final_user_data:
                app.logger.error(f"交易成功後無法獲取用戶 {user_id} 的最終數據。")
                resp_data = {'success': True, 'message': f'交易成功: {trade_type.capitalize()} {shares} 股 {ticker}。請查看您的投資組合。', 
                             'newCashBalance': float(updated_cash_balance_from_db), 'marketType': market_type}
                return set_no_cache_headers(make_response(jsonify(resp_data)))

            resp_data = {'success': True, 'message': f'交易成功: {trade_type.capitalize()} {shares} 股 {ticker}。', 
                         'newCashBalance': float(final_user_data[cash_balance_key]), 'marketType': market_type}
            return set_no_cache_headers(make_response(jsonify(resp_data)))

    except pymysql.Error as e:
        app.logger.error(f"用戶 {user_id}，股票 {ticker} 交易期間資料庫錯誤: {e}", exc_info=True)
        if conn and conn.open: conn.rollback() 
        resp_data = {'success': False, 'message': f'交易期間資料庫錯誤: {str(e)}'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))
    except Exception as e:
        app.logger.error(f"用戶 {user_id} 交易期間發生未預期錯誤: {e}", exc_info=True)
        if conn and conn.open: conn.rollback() 
        resp_data = {'success': False, 'message': f'交易期間發生未預期錯誤: {str(e)}'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))
    finally:
        if conn and conn.open: conn.close()

@app.route('/api/portfolio_data', methods=['GET'])
@login_required
def portfolio_data_api():
    user_id = get_current_user_id()
    user_account_full = get_user_data(user_id)
    if not user_account_full:
        resp_data = {'success': False, 'message': '找不到用戶帳戶。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 404))

    market_type = user_account_full['market_type']
    app.logger.info(f"用戶 {user_id}，市場 {market_type} 投資組合數據請求")

    cash_balance = user_account_full[f'cash_balance_{market_type.lower()}']
    initial_capital = user_account_full[f'initial_capital_{market_type.lower()}']

    conn = get_db_connection()
    if not conn:
        resp_data = {'success': False, 'message': '資料庫連接錯誤。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))
    try:
        with conn.cursor() as cursor:
            cursor.execute("""SELECT h.ticker, h.stock_name, h.shares, h.average_cost,
                                     sc.current_price, sc.previous_close
                              FROM holdings h LEFT JOIN stock_data_cache sc ON h.ticker = sc.ticker
                              WHERE h.user_id = %s AND h.market_type = %s AND h.shares > 0""", (user_id, market_type))
            holdings_db = cursor.fetchall()

            holdings_frontend, total_stock_market_value, total_today_pl_value_for_stocks = [], 0, 0
            for h_db in holdings_db:
                live_stock_data = None
                if h_db['current_price'] is None or h_db['previous_close'] is None:
                    app.logger.info(f"Portfolio: 快取中 {h_db['ticker']} 數據不完整，嘗試即時獲取。")
                    live_stock_data = get_stock_info(h_db['ticker'])

                current_price = float(h_db['current_price']) if h_db['current_price'] is not None else \
                                (float(live_stock_data['current_price']) if live_stock_data and live_stock_data.get('current_price') is not None else float(h_db['average_cost']))
                
                previous_close = float(h_db['previous_close']) if h_db['previous_close'] is not None else \
                                 (float(live_stock_data['previous_close']) if live_stock_data and live_stock_data.get('previous_close') is not None else current_price)


                market_value = h_db['shares'] * current_price
                total_stock_market_value += market_value
                cost_basis = h_db['shares'] * float(h_db['average_cost'])
                total_pl_holding = market_value - cost_basis
                total_pl_percent_holding = (total_pl_holding / cost_basis) * 100 if cost_basis != 0 else 0
                
                holding_today_pl_value = (current_price - previous_close) * h_db['shares']
                yesterday_holding_market_value = previous_close * h_db['shares']
                holding_today_pl_percent = (holding_today_pl_value / yesterday_holding_market_value) * 100 if yesterday_holding_market_value != 0 else 0.0
                total_today_pl_value_for_stocks += holding_today_pl_value
                
                holdings_frontend.append({
                    'ticker': h_db['ticker'], 'name': h_db['stock_name'], 'shares': h_db['shares'],
                    'avgCost': float(h_db['average_cost']), 'currentPrice': current_price, 'marketValue': market_value,
                    'todayChangePercent': holding_today_pl_percent, 'totalPL': total_pl_holding,
                    'totalPLPercent': total_pl_percent_holding})
            
            total_portfolio_value = cash_balance + total_stock_market_value
            overall_total_pl = total_portfolio_value - initial_capital
            overall_total_pl_percent = (overall_total_pl / initial_capital) * 100 if initial_capital != 0 else 0
            
            yesterday_portfolio_value = total_portfolio_value - total_today_pl_value_for_stocks
            
            today_pl_portfolio_percent = 0.0
            if yesterday_portfolio_value != 0 : 
                today_pl_portfolio_percent = (total_today_pl_value_for_stocks / yesterday_portfolio_value) * 100
            elif total_today_pl_value_for_stocks > 0: 
                 today_pl_portfolio_percent = float('inf') 
            elif total_today_pl_value_for_stocks < 0:
                 today_pl_portfolio_percent = float('-inf') 


            overview = {'totalPortfolioValue': total_portfolio_value, 'totalOverallPL': overall_total_pl,
                        'totalOverallPLPercent': overall_total_pl_percent, 'todayTotalPL': total_today_pl_value_for_stocks,
                        'todayTotalPLPercent': today_pl_portfolio_percent, 'cashBalance': cash_balance}

            resp_data = {'success': True, 'overview': overview, 'holdings': holdings_frontend, 'marketType': market_type}
            return set_no_cache_headers(make_response(jsonify(resp_data)))

    except pymysql.Error as e:
        app.logger.error(f"用戶 {user_id}，市場 {market_type} 投資組合數據獲取資料庫錯誤: {e}", exc_info=True)
        resp_data = {'success': False, 'message': f'資料庫錯誤: {str(e)}'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))
    except Exception as e:
        app.logger.error(f"用戶 {user_id}，市場 {market_type} 投資組合數據獲取未預期錯誤: {e}", exc_info=True)
        resp_data = {'success': False, 'message': '發生未預期錯誤。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))
    finally:
        if conn: conn.close()

@app.route('/api/generate_ai_report', methods=['POST'])
@login_required
def generate_ai_report_api():
    user_id = get_current_user_id()
    if not report_generation_model:
        resp_data = {'success': False, 'message': 'AI 模型不可用。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 503))

    data = request.get_json()
    report_type = data.get('reportType')

    user_data_db = get_user_data(user_id)
    if not user_data_db:
        resp_data = {'success': False, 'message': '無法獲取用戶數據。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))

    market_type = user_data_db['market_type']
    app.logger.info(f"用戶 {user_id}，市場 {market_type}，報告類型 {report_type} 的 AI 報告請求")

    if report_type not in ['investment', 'behavioral']:
        resp_data = {'success': False, 'message': '無效的報告類型。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))

    conn = get_db_connection()
    if not conn:
        resp_data = {'success': False, 'message': '資料庫連接錯誤。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))

    cash_balance_for_report = user_data_db[f'cash_balance_{market_type.lower()}']
    initial_capital_for_report = user_data_db[f'initial_capital_{market_type.lower()}']
    holdings_summary, trade_history_summary = "無", "尚無交易紀錄。"
    portfolio_total_value = cash_balance_for_report 

    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT ticker, stock_name, shares, average_cost FROM holdings WHERE user_id = %s AND market_type = %s AND shares > 0", (user_id, market_type))
            holdings_db = cursor.fetchall()
            if holdings_db:
                h_list = []
                for h_db in holdings_db:
                    s_info = get_stock_info(h_db['ticker']) 
                    c_price = float(s_info['current_price']) if s_info and s_info.get('current_price') is not None else float(h_db['average_cost'])
                    portfolio_total_value += h_db['shares'] * c_price
                    h_list.append(f"{h_db['shares']} 股 {h_db['stock_name'] or h_db['ticker']} (平均成本 ${float(h_db['average_cost']):.2f}, 目前總值 ${h_db['shares'] * c_price:.2f})")
                if h_list: holdings_summary = "； ".join(h_list)

            limit = 10 if report_type == 'behavioral' else 5 
            cursor.execute("SELECT trade_type, ticker, stock_name, shares, price_per_share, mood, reason, timestamp FROM trade_history WHERE user_id = %s AND market_type = %s ORDER BY timestamp DESC LIMIT %s", (user_id, market_type, limit))
            trades_db = cursor.fetchall()
            if trades_db:
                t_list = []
                for t_db in trades_db:
                    r_str = t_db['reason']; r_str = (r_str[:47] + "...") if r_str and len(r_str) > 50 else r_str
                    trade_action_chinese = "買入" if t_db['trade_type'] == 'buy' else "賣出"
                    t_str = f"{trade_action_chinese} {t_db['shares']} 股 {t_db['stock_name'] or t_db['ticker']} @ ${float(t_db['price_per_share']):.2f} (於 {t_db['timestamp'].strftime('%y-%m-%d')})"
                    if t_db['mood']: t_str += f" (心情: {t_db['mood']})"
                    if r_str: t_str += f" (原因: {r_str})"
                    t_list.append(t_str)
                if t_list: trade_history_summary = "； ".join(t_list)
    
    except pymysql.Error as e:
        app.logger.error(f"用戶 {user_id}/{market_type} AI 報告資料庫錯誤: {e}")
        resp_data = {'success': False, 'message': '獲取報告數據時發生錯誤。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))
    finally:
        if conn: conn.close()

    prompt = f"您是 FinSimU AI，一位為學生 '{user_data_db['nickname']}'（投資風格：{user_data_db['investment_style']}）提供股市模擬（{market_type} 市場）教育輔助的助理。\n"
    prompt += f"初始資金 ({market_type}): ${initial_capital_for_report:.2f}。 目前現金 ({market_type}): ${cash_balance_for_report:.2f}。 總投資組合價值 ({market_type}): ${portfolio_total_value:.2f}。\n"
    prompt += f"目前持股 ({market_type}): {holdings_summary}\n近期交易 ({market_type}): {trade_history_summary}\n\n"
    if report_type == 'investment':
        prompt += "請生成一份簡潔的繁體中文「投資報告」：1. 整體表現快照（與初始資金比較）。2. 投資組合構成洞察（若有持股）。3. 交易模式觀察（若有交易紀錄）。4. 提出一個符合其投資風格的反思性問題或溫和建議。語氣：鼓勵性、教育性。重點：反思、學習。格式：清晰，第2、3、4點請使用項目符號。"
    elif report_type == 'behavioral':
        prompt += "請生成一份簡潔的繁體中文「行為分析報告」：1. 交易中主要的情緒/原因（若有）。2. 一個潛在的行為偏差（例如：錯失恐懼症 FOMO、損失規避、過度自信）並簡單解釋。3. 一個關於投資心理的自我覺察反思性問題。語氣：同理心、支持性。目標：培養自我覺察。格式：清晰。"
    prompt += "\n重要提示：請輸出適合顯示的良好格式文本。可以使用 Markdown 結構（如項目符號），但不要使用三個反引號的代碼塊。"

    app.logger.info(f"為用戶 {user_id}，市場 {market_type}，類型 {report_type} 生成 AI 報告")
    try:
        response = report_generation_model.generate_content(prompt, safety_settings=safety_settings_gemini if 'safety_settings_gemini' in globals() and safety_settings_gemini is not None else None)
        
        analysis_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'): analysis_text += part.text

        if hasattr(response,'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            app.logger.warning(f"用戶 {user_id} 的 Gemini 內容被阻擋。原因:{response.prompt_feedback.block_reason}")
            resp_data = {'success':False, 'message':f'AI 內容被阻擋: {response.prompt_feedback.block_reason_message or "安全考量"}'}
            return set_no_cache_headers(make_response(jsonify(resp_data), 400))

        if not analysis_text:
            app.logger.warning(f"用戶 {user_id} 的 Gemini 回應無可用文本。回應:{response}")
            resp_data = {'success':False, 'message':'AI 服務回應為空。'}
            return set_no_cache_headers(make_response(jsonify(resp_data), 500))

        resp_data = {'success': True, 'report_type': report_type, 'report_content': analysis_text.strip(), 'marketType': market_type}
        return set_no_cache_headers(make_response(jsonify(resp_data)))
    except Exception as e:
        app.logger.error(f"用戶 {user_id} 使用 Gemini 時發生錯誤: {e}", exc_info=True)
        resp_data = {'success': False, 'message': 'AI 服務發生錯誤。'}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))

@app.route('/api/ga/potential_buys', methods=['GET'])
@login_required
def get_ga_potential_buys():
    market_type_req = request.args.get('marketType', '').upper()

    if not GA_ENGINE_IMPORTED:
        resp_data = {'success':False, 'message':"GA 引擎組件不可用。"}
        return set_no_cache_headers(make_response(jsonify(resp_data), 503))

    if not market_type_req or market_type_req not in ['TW', 'US']:
        resp_data = {'success':False, 'message':"需要有效的市場類型 (TW 或 US)。"}
        return set_no_cache_headers(make_response(jsonify(resp_data), 400))

    app.logger.info(f"請求預計算的 GA 潛力買入股，市場 {market_type_req}。")
    
    query = """
        SELECT stock_ticker, stock_name, current_price, buy_reason 
        FROM ga_current_potential_buys
        WHERE market_type = %s
        ORDER BY calculated_at DESC, stock_ticker ASC 
    """ 
    
    potential_buys_db = execute_db_query(query, (market_type_req,), fetch_all=True)

    if potential_buys_db is None:
        resp_data = {'success':False, 'message':"從預計算數據中獲取 GA 潛力買入股時發生錯誤。"}
        return set_no_cache_headers(make_response(jsonify(resp_data), 500))

    potential_buys_frontend = []
    if potential_buys_db:
        for row in potential_buys_db:
            potential_buys_frontend.append({
                'ticker': row['stock_ticker'],
                'name': row['stock_name'],
                'current_price': float(row['current_price']) if row['current_price'] is not None else None,
                'reason': row['buy_reason'] or "GA買入信號已觸發", 
                'market_type': market_type_req
            })
    
    if not potential_buys_frontend:
         app.logger.info(f"市場 {market_type_req} 目前無預計算的 GA 潛力買入股。")
    
    resp_data = {'success':True, 'stocks':potential_buys_frontend, 'marketType':market_type_req}
    return set_no_cache_headers(make_response(jsonify(resp_data)))


@app.route('/api/analyze_stock_on_demand', methods=['POST'])
@login_required
def api_analyze_stock_on_demand():
    user_id = get_current_user_id()
    data = request.get_json()
    ticker_input = data.get('ticker','').strip().upper()
    include_ga_check = data.get('include_ga', False) 

    if not ticker_input:
        return make_response(jsonify({'success': False, 'message': '請提供股票代碼。'}), 400)

    app.logger.info(f"個股即時分析請求: {ticker_input} (用戶: {user_id}, 包含GA檢查: {include_ga_check})")

    if not stock_analysis_model_gemini: 
        return make_response(jsonify({'success': False, 'message': 'AI分析服務暫時無法使用 (AI模型未加載)。'}), 503)

    try:
        analyzer = StockAnalyzer(ticker=ticker_input, period="3y", temperature=0.5)
        summary_data = analyzer.get_stock_summary()

        if summary_data.get('error'):
            app.logger.warning(f"個股即時分析 {ticker_input} 失敗: {summary_data['error']}")
            if "無法取得" in summary_data['error'] and ("歷史資料" in summary_data['error'] or "基本資訊" in summary_data['error']):
                return make_response(jsonify({'success': False, 'message': f"無法獲取股票 {analyzer.ticker} 的足夠歷史數據或基本資訊進行分析。請確認股票代碼是否正確或稍後再試。"}), 404)
            return make_response(jsonify({'success': False, 'message': f"分析股票 {ticker_input} 時發生錯誤: {summary_data['error']}"}), 500)


        if summary_data.get("chart_path"):
            summary_data["chart_url"] = url_for('static', filename=summary_data["chart_path"])
        else:
            summary_data["chart_url"] = None
        
        summary_data['ga_signal_info'] = {'signal': 'N/A', 'reason': '未執行GA信號檢查或無可用策略。'} 

        if include_ga_check and GA_ENGINE_IMPORTED:
            app.logger.info(f"為 {analyzer.ticker} (市場: {analyzer.market}) 執行 GA 信號檢查/即時訓練...")
            ga_signal_info_payload = {'signal': 'ERROR', 'reason': 'GA信號檢查時發生未知內部錯誤。'} 
            
            gene_to_use = None
            newly_trained_gene = False
            gene_source_for_frontend = "N/A"


            best_gene_record = execute_db_query(
                """SELECT ai_strategy_gene 
                   FROM ai_vs_user_games 
                   WHERE stock_ticker = %s AND market_type = %s AND user_id = %s AND ai_strategy_gene IS NOT NULL
                   ORDER BY ai_final_portfolio_value DESC LIMIT 1""",
                (analyzer.ticker, analyzer.market, SYSTEM_AI_USER_ID), fetch_one=True
            )

            if best_gene_record and best_gene_record.get('ai_strategy_gene'):
                try:
                    gene_to_use = json.loads(best_gene_record['ai_strategy_gene'])
                    if not isinstance(gene_to_use, list) or len(gene_to_use) != 10:
                        app.logger.warning(f"股票 {analyzer.ticker} 的已存儲基因格式無效 ({gene_to_use})，將嘗試重新訓練。")
                        gene_to_use = None 
                    else:
                        gene_source_for_frontend = "已有的預訓練基因"
                except json.JSONDecodeError:
                    app.logger.warning(f"股票 {analyzer.ticker} 的已存儲基因解析失敗，將嘗試重新訓練。")
                    gene_to_use = None
            
            if not gene_to_use: 
                app.logger.info(f"股票 {analyzer.ticker} (市場: {analyzer.market}) 無預訓練GA策略或基因無效，開始即時GA優化...")
                
                train_prices, train_dates, train_stock_df, train_vix_series = ga_load_stock_data(
                    analyzer.ticker, 
                    start_date=ON_DEMAND_TRAIN_START_DATE, 
                    end_date=ON_DEMAND_TRAIN_END_DATE, 
                    verbose=False
                )
                if not train_prices or train_stock_df is None or train_stock_df.empty:
                    ga_signal_info_payload = {'signal': 'ERROR', 'reason': '即時GA訓練失敗：無法為此股票加載足夠的訓練數據。', 'gene_source': '即時訓練嘗試'}
                else:
                    train_indicators, train_ready = ga_precompute_indicators(
                        train_stock_df, train_vix_series, STRATEGY_CONFIG_SHARED_GA, verbose=False
                    )
                    if not train_ready:
                        ga_signal_info_payload = {'signal': 'ERROR', 'reason': '即時GA訓練失敗：無法計算訓練數據的指標。', 'gene_source': '即時訓練嘗試'}
                    else:
                        app.logger.info(f"開始為 {analyzer.ticker} 運行 {NUM_GA_RUNS_ON_DEMAND} 次即時 GA...")
                        overall_best_gene_run = None
                        overall_best_fitness_run = -float('inf')
                        for _ in range(NUM_GA_RUNS_ON_DEMAND): 
                            current_gene, current_fitness = genetic_algorithm_with_elitism(
                                train_prices, train_dates,
                                train_indicators.get('rsi',{}), train_indicators.get('vix_ma',{}),
                                train_indicators.get('bbl',{}), train_indicators.get('bbm',{}),
                                train_indicators.get('fixed',{}).get('adx_list',[]),
                                train_indicators.get('fixed',{}).get('ma_short_list',[]),
                                train_indicators.get('fixed',{}).get('ma_long_list',[]),
                                ga_params=GA_PARAMS_FOR_ONDEMAND_TRAIN 
                            )
                            if current_gene and current_fitness > overall_best_fitness_run:
                                overall_best_fitness_run = current_fitness
                                overall_best_gene_run = current_gene
                        
                        if overall_best_gene_run and overall_best_fitness_run > -float('inf'):
                            gene_to_use = overall_best_gene_run
                            newly_trained_gene = True
                            gene_source_for_frontend = "即時優化生成的新基因"
                            app.logger.info(f"股票 {analyzer.ticker} 即時GA訓練完成，最佳適應度: {overall_best_fitness_run:.4f}，基因: {gene_to_use}")
                            
                            game_data_on_demand = {
                                "user_id": SYSTEM_AI_USER_ID, "market_type": analyzer.market,
                                "stock_ticker": analyzer.ticker, 
                                "game_start_date": datetime.datetime.strptime(ON_DEMAND_TRAIN_START_DATE, "%Y-%m-%d").date(),
                                "game_end_date": datetime.datetime.strptime(ON_DEMAND_TRAIN_END_DATE, "%Y-%m-%d").date(),
                                "ai_strategy_gene": json.dumps(gene_to_use),
                                "ai_initial_cash": 1.0, "user_initial_cash": 1.0,
                                "ai_final_portfolio_value": float(overall_best_fitness_run),
                                "user_final_portfolio_value": None, "game_completed_at": datetime.datetime.now()
                            }
                            insert_query_on_demand = """
                                INSERT INTO ai_vs_user_games
                                (user_id, market_type, stock_ticker, game_start_date, game_end_date, ai_strategy_gene, ai_initial_cash, user_initial_cash, ai_final_portfolio_value, user_final_portfolio_value, game_completed_at)
                                VALUES (%(user_id)s, %(market_type)s, %(stock_ticker)s, %(game_start_date)s, %(game_end_date)s, %(ai_strategy_gene)s, %(ai_initial_cash)s, %(user_initial_cash)s, %(ai_final_portfolio_value)s, %(user_final_portfolio_value)s, %(game_completed_at)s)
                                ON DUPLICATE KEY UPDATE
                                    game_start_date = VALUES(game_start_date), game_end_date = VALUES(game_end_date),
                                    ai_strategy_gene = VALUES(ai_strategy_gene), ai_final_portfolio_value = VALUES(ai_final_portfolio_value),
                                    game_completed_at = VALUES(game_completed_at);
                            """
                            db_save_result = execute_db_query(insert_query_on_demand, game_data_on_demand, commit=True)
                            if db_save_result: app.logger.info(f"已成功將 {analyzer.ticker} 的即時訓練GA結果保存到資料庫。")
                            else: app.logger.error(f"保存 {analyzer.ticker} 的即時訓練GA結果到資料庫失敗。")
                        else:
                            ga_signal_info_payload = {'signal': 'N/A', 'reason': f'股票 {analyzer.ticker} 即時GA訓練未能找到有效策略。', 'gene_source': '即時訓練嘗試失敗'}
            
            if gene_to_use:
                try:
                    current_signal_start_date = (datetime.date.today() - datetime.timedelta(days=max(STRATEGY_CONFIG_SHARED_GA.get('ma_long_period',20), GA_PARAMS_CONFIG.get('adx_period',14)) + 90)).strftime("%Y-%m-%d") # 增加緩衝
                    current_signal_end_date = datetime.date.today().strftime("%Y-%m-%d")
                    
                    prices_sig, dates_sig, stock_df_sig, vix_series_sig = ga_load_stock_data(
                        analyzer.ticker, start_date=current_signal_start_date, end_date=current_signal_end_date, verbose=False
                    )
                    if not prices_sig or stock_df_sig is None or stock_df_sig.empty:
                        raise ValueError("為GA當前信號判斷加載數據失敗。")

                    indicators_sig, ready_sig = ga_precompute_indicators(stock_df_sig, vix_series_sig, STRATEGY_CONFIG_SHARED_GA, verbose=False)
                    if not ready_sig:
                        raise ValueError("為GA當前信號判斷計算指標失敗。")

                    param_rsi_buy_entry = gene_to_use[0]; param_vix_threshold = gene_to_use[2]
                    rsi_period_idx = gene_to_use[4]; vix_ma_period_idx = gene_to_use[5]
                    bb_len_idx = gene_to_use[6]; bb_std_idx = gene_to_use[7]
                    param_adx_threshold = gene_to_use[8]; param_high_vol_entry_choice = gene_to_use[9]
                    
                    chosen_rsi_p = STRATEGY_CONFIG_SHARED_GA['rsi_period_options'][rsi_period_idx]
                    
                    def get_latest_and_prev_safe(indicator_list_val_func):
                        if indicator_list_val_func and isinstance(indicator_list_val_func, list) and len(indicator_list_val_func) >= 2: return indicator_list_val_func[-1], indicator_list_val_func[-2]
                        elif indicator_list_val_func and isinstance(indicator_list_val_func, list) and len(indicator_list_val_func) == 1: return indicator_list_val_func[-1], indicator_list_val_func[-1]
                        return np.nan, np.nan

                    latest_price_val_sig = prices_sig[-1] if prices_sig else np.nan
                    rsi_latest_val_sig, rsi_prev_val_sig = get_latest_and_prev_safe(indicators_sig.get('rsi', {}).get(chosen_rsi_p, []))
                    bbl_latest_val_sig, _ = get_latest_and_prev_safe(indicators_sig.get('bbl', {}).get((STRATEGY_CONFIG_SHARED_GA['bb_length_options'][bb_len_idx], STRATEGY_CONFIG_SHARED_GA['bb_std_options'][bb_std_idx]), []))
                    adx_latest_val_sig, _ = get_latest_and_prev_safe(indicators_sig.get('fixed', {}).get('adx_list', []))
                    vix_ma_latest_val_sig, _ = get_latest_and_prev_safe(indicators_sig.get('vix_ma', {}).get(STRATEGY_CONFIG_SHARED_GA['vix_ma_period_options'][vix_ma_period_idx], []))
                    ma_short_latest_val_sig, ma_short_prev_val_sig = get_latest_and_prev_safe(indicators_sig.get('fixed', {}).get('ma_short_list', []))
                    ma_long_latest_val_sig, ma_long_prev_val_sig = get_latest_and_prev_safe(indicators_sig.get('fixed', {}).get('ma_long_list', []))


                    is_buy_signal_final = check_ga_buy_signal_at_latest_point(
                        param_rsi_buy_entry, param_adx_threshold, param_vix_threshold, param_high_vol_entry_choice,
                        latest_price_val_sig, rsi_latest_val_sig, rsi_prev_val_sig, bbl_latest_val_sig,
                        adx_latest_val_sig, vix_ma_latest_val_sig,
                        ma_short_latest_val_sig, ma_long_latest_val_sig, ma_short_prev_val_sig, ma_long_prev_val_sig
                    )
                    
                    reason_final = generate_buy_reason(gene_to_use, latest_price_val_sig, rsi_latest_val_sig, rsi_prev_val_sig, bbl_latest_val_sig, 
                                                     adx_latest_val_sig, vix_ma_latest_val_sig, ma_short_latest_val_sig, ma_long_latest_val_sig, 
                                                     ma_short_prev_val_sig, ma_long_prev_val_sig, STRATEGY_CONFIG_SHARED_GA)
                    
                    ga_signal_info_payload = {
                        'signal': 'BUY' if is_buy_signal_final else 'NEUTRAL',
                        'reason': reason_final,
                        'parameters_desc': format_ga_gene_parameters_to_text(gene_to_use, STRATEGY_CONFIG_SHARED_GA),
                        'gene_source': gene_source_for_frontend
                    }
                except ValueError as ve_sig:
                    ga_signal_info_payload = {'signal': 'ERROR', 'reason': f'GA當前信號判斷數據準備失敗: {str(ve_sig)}', 'gene_source': gene_source_for_frontend}
                except Exception as e_sig:
                    app.logger.error(f"使用基因 {gene_to_use} 為 {analyzer.ticker} 判斷GA信號時發生錯誤: {e_sig}", exc_info=True)
                    ga_signal_info_payload = {'signal': 'ERROR', 'reason': f'GA當前信號判斷時發生未知錯誤。', 'gene_source': gene_source_for_frontend}
            
            elif not newly_trained_gene: 
                 ga_signal_info_payload = {'signal': 'N/A', 'reason': f'股票 {analyzer.ticker} 無預訓練GA策略，且即時訓練未能生成有效策略。', 'gene_source': '無可用基因'}


            summary_data['ga_signal_info'] = ga_signal_info_payload
        

        app.logger.info(f"成功生成 {ticker_input} 的即時分析。")
        return make_response(jsonify({'success': True, 'data': summary_data}))

    except ValueError as ve_main: 
        app.logger.error(f"個股即時分析 {ticker_input} 時發生 ValueError: {ve_main}", exc_info=True)
        return make_response(jsonify({'success': False, 'message': f"股票代碼 '{ticker_input}' 無效或無法獲取數據: {str(ve_main)}"}), 400)
    except Exception as e_main:
        app.logger.error(f"個股即時分析 {ticker_input} 時發生未預期錯誤: {e_main}", exc_info=True)
        return make_response(jsonify({'success': False, 'message': "執行股票分析時發生未預期的內部錯誤，請稍後再試。"}), 500)

# --- Root and Static Routes ---
@app.route('/')
def finsimu_landing_route():
    if 'user_id' in session and session.get('user_id') is not None:
        return redirect(url_for('finsimu_app_route', _anchor='dashboard'))
    return redirect(url_for('finsimu_login_route'))

@app.route('/login')
def finsimu_login_route():
    if 'user_id' in session and session.get('user_id') is not None:
        return redirect(url_for('finsimu_app_route', _anchor='dashboard'))
    response = make_response(render_template('login.html'))
    return set_no_cache_headers(response)


@app.route('/register')
def finsimu_register_route():
    if 'user_id' in session and session.get('user_id') is not None:
        return redirect(url_for('finsimu_app_route', _anchor='dashboard'))
    response = make_response(render_template('register.html'))
    return set_no_cache_headers(response)


@app.route('/app')
@login_required
def finsimu_app_route():
    app.logger.info(f"用戶 {session.get('user_id')} 正在訪問 /app。 Session: {dict(session)}")
    response = make_response(render_template('app_main.html'))
    return set_no_cache_headers(response)

@app.route('/logout_api', methods=['POST'])
@login_required
def logout_api():
    user_id_before = session.get('user_id', 'N/A')
    app.logger.info(f"用戶 {user_id_before} 嘗試登出。登出前 Session: {dict(session)}")
    session.clear()
    app.logger.info(f"Session 清除後 (原用戶 {user_id_before}): {dict(session)}")
    resp_data = {'success': True, 'message': '已成功登出。'}
    return set_no_cache_headers(make_response(jsonify(resp_data)))

# --- 後台預計算任務 ---
def perform_ga_potential_buys_precalculation(market_type):
    with app.app_context(): 
        app.logger.info(f"APScheduler: 開始為市場 {market_type} 預計算 GA 潛力買入股...")
        if not GA_ENGINE_IMPORTED:
            app.logger.error(f"APScheduler: GA 引擎未導入，無法為市場 {market_type} 執行預計算。")
            return

        ai_watchlist = execute_db_query( 
            """SELECT stock_ticker, ai_strategy_gene FROM ai_vs_user_games
               WHERE user_id = %s AND market_type = %s AND ai_strategy_gene IS NOT NULL
               ORDER BY ai_final_portfolio_value DESC LIMIT %s""",
            (SYSTEM_AI_USER_ID, market_type, AI_WATCHLIST_SIZE), fetch_all=True
        )

        if ai_watchlist is None: 
            app.logger.error(f"APScheduler: 無法為市場 {market_type} 獲取 AI 觀察列表 (資料庫查詢失敗)。")
            return
        
        app.logger.info(f"APScheduler: 市場 {market_type} 的 AI 觀察列表包含 {len(ai_watchlist)} 支股票準備進行回測。")

        if not ai_watchlist:
            app.logger.info(f"APScheduler: 市場 {market_type} 的 AI 觀察列表為空或無已訓練策略。")
            conn_temp = get_db_connection()
            if conn_temp:
                try:
                    with conn_temp.cursor() as cursor_temp:
                        cursor_temp.execute("DELETE FROM ga_current_potential_buys WHERE market_type = %s", (market_type,))
                    conn_temp.commit()
                    app.logger.info(f"APScheduler: 已清除市場 {market_type} 的舊 GA 潛力股數據 (因觀察列表為空)。")
                except pymysql.Error as db_err_clear:
                    app.logger.error(f"APScheduler: 清除市場 {market_type} 舊 GA 潛力股時資料庫錯誤: {db_err_clear}")
                    if conn_temp: conn_temp.rollback()
                finally:
                    if conn_temp: conn_temp.close()
            return
            
        potential_buys_to_store = []
        required_days_for_ga = max(STRATEGY_CONFIG_SHARED_GA.get('ma_long_period',20), GA_PARAMS_CONFIG.get('adx_period',14)) + 60
        start_date_str = (datetime.date.today() - datetime.timedelta(days=required_days_for_ga * 2)).strftime("%Y-%m-%d") 
        end_date_str = datetime.date.today().strftime("%Y-%m-%d")

        for item in ai_watchlist:
            ticker = item['stock_ticker']
            try:
                gene_str = item.get('ai_strategy_gene', '[]')
                gene = json.loads(gene_str)
                if not isinstance(gene, list) or len(gene) != 10: 
                    app.logger.warning(f"APScheduler: 股票 {ticker} 的基因結構無效: {gene_str}")
                    continue

                app.logger.debug(f"APScheduler: 為 {ticker} (基因: {gene}) 檢查 GA 買入信號")
                prices, dates, stock_df, vix_series = ga_load_stock_data(
                    ticker, start_date=start_date_str, end_date=end_date_str, verbose=False
                )
                if not prices or stock_df is None or stock_df.empty: 
                    app.logger.warning(f"APScheduler: 為 {ticker} 加載數據失敗，跳過。")
                    continue

                indicators, ready = ga_precompute_indicators(stock_df, vix_series, STRATEGY_CONFIG_SHARED_GA, verbose=False)
                if not ready:
                    app.logger.warning(f"APScheduler: 為 {ticker} 計算指標失敗，跳過。")
                    continue
                
                param_rsi_buy_entry = gene[0]; param_vix_threshold = gene[2]
                rsi_period_idx = gene[4]; vix_ma_period_idx = gene[5]
                bb_len_idx = gene[6]; bb_std_idx = gene[7]
                param_adx_threshold = gene[8]; param_high_vol_entry_choice = gene[9]
                
                chosen_rsi_p = STRATEGY_CONFIG_SHARED_GA['rsi_period_options'][rsi_period_idx]
                
                def get_latest_and_prev_safe_sched(indicator_list_val_func_s): 
                    if indicator_list_val_func_s and isinstance(indicator_list_val_func_s, list) and len(indicator_list_val_func_s) >= 2:
                        return indicator_list_val_func_s[-1], indicator_list_val_func_s[-2]
                    elif indicator_list_val_func_s and isinstance(indicator_list_val_func_s, list) and len(indicator_list_val_func_s) == 1:
                        return indicator_list_val_func_s[-1], indicator_list_val_func_s[-1]
                    return np.nan, np.nan
                
                latest_price_val = prices[-1] if prices else np.nan
                rsi_latest_val, rsi_prev_val = get_latest_and_prev_safe_sched(indicators.get('rsi', {}).get(chosen_rsi_p, []))
                bbl_latest_val, _ = get_latest_and_prev_safe_sched(indicators.get('bbl', {}).get((STRATEGY_CONFIG_SHARED_GA['bb_length_options'][bb_len_idx], STRATEGY_CONFIG_SHARED_GA['bb_std_options'][bb_std_idx]), []))
                adx_latest_val, _ = get_latest_and_prev_safe_sched(indicators.get('fixed', {}).get('adx_list', []))
                vix_ma_latest_val, _ = get_latest_and_prev_safe_sched(indicators.get('vix_ma', {}).get(STRATEGY_CONFIG_SHARED_GA['vix_ma_period_options'][vix_ma_period_idx], []))
                ma_short_latest_val, ma_short_prev_val = get_latest_and_prev_safe_sched(indicators.get('fixed', {}).get('ma_short_list', []))
                ma_long_latest_val, ma_long_prev_val = get_latest_and_prev_safe_sched(indicators.get('fixed', {}).get('ma_long_list', []))


                is_buy_signal = check_ga_buy_signal_at_latest_point(
                    param_rsi_buy_entry, param_adx_threshold, param_vix_threshold, param_high_vol_entry_choice,
                    latest_price_val, rsi_latest_val, rsi_prev_val, bbl_latest_val,
                    adx_latest_val, vix_ma_latest_val,
                    ma_short_latest_val, ma_long_latest_val, ma_short_prev_val, ma_long_prev_val
                )

                if is_buy_signal:
                    buy_reason_text = generate_buy_reason(gene, latest_price_val, rsi_latest_val, rsi_prev_val, bbl_latest_val, 
                                                         adx_latest_val, vix_ma_latest_val, ma_short_latest_val, ma_long_latest_val, 
                                                         ma_short_prev_val, ma_long_prev_val, STRATEGY_CONFIG_SHARED_GA)
                    stock_data_cache = get_stock_info(ticker) 
                    stock_name = stock_data_cache.get('name', ticker) if stock_data_cache else ticker
                    
                    potential_buys_to_store.append({
                        'market_type': market_type,
                        'stock_ticker': ticker,
                        'stock_name': stock_name,
                        'current_price': latest_price_val if pd.notna(latest_price_val) else None,
                        'ai_strategy_gene': json.dumps(gene), 
                        'buy_reason': buy_reason_text,
                        'calculated_at': datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) 
                    })
                    app.logger.info(f"APScheduler: 市場 {market_type} 發現 GA 潛力買入股: {ticker}")

            except json.JSONDecodeError:
                app.logger.error(f"APScheduler: 解析股票 {ticker} 的 GA 策略基因失敗")
            except Exception as e:
                app.logger.error(f"APScheduler: 處理股票 {ticker} 時發生錯誤: {e}", exc_info=True)
        
        conn_main_task = get_db_connection()
        if not conn_main_task:
            app.logger.error(f"APScheduler: 無法為市場 {market_type} 連接資料庫以存儲 GA 潛力股。")
            return

        try:
            with conn_main_task.cursor() as cursor:
                cursor.execute("DELETE FROM ga_current_potential_buys WHERE market_type = %s", (market_type,))
                if potential_buys_to_store:
                    insert_sql = """
                        INSERT INTO ga_current_potential_buys 
                        (market_type, stock_ticker, stock_name, current_price, ai_strategy_gene, buy_reason, calculated_at)
                        VALUES (%(market_type)s, %(stock_ticker)s, %(stock_name)s, %(current_price)s, %(ai_strategy_gene)s, %(buy_reason)s, %(calculated_at)s)
                    """
                    rows_affected = cursor.executemany(insert_sql, potential_buys_to_store)
                    app.logger.info(f"APScheduler: 為市場 {market_type} 成功存儲 {rows_affected} 條 GA 潛力買入股。")
                else:
                    app.logger.info(f"APScheduler: 市場 {market_type} 本次未發現新的 GA 潛力買入股，已清除舊數據。")
            conn_main_task.commit()
        except pymysql.Error as db_err:
            app.logger.error(f"APScheduler: 為市場 {market_type} 存儲 GA 潛力股時資料庫錯誤: {db_err}")
            if conn_main_task: conn_main_task.rollback()
        finally:
            if conn_main_task: conn_main_task.close()
        app.logger.info(f"APScheduler: 市場 {market_type} 的 GA 潛力買入股預計算完成。")


# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context(): 
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
            app.logger.info(f"已創建目錄 (從主程序): {charts_dir}")

        if GEMINI_API_KEY:
            if not report_generation_model or not stock_analysis_model_gemini:
                try:
                    safety_settings = [ 
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    ]
                    report_generation_model = genai.GenerativeModel("models/gemini-1.5-flash-latest", safety_settings=safety_settings)
                    stock_analysis_model_gemini = genai.GenerativeModel("models/gemini-1.5-flash-latest", safety_settings=safety_settings)
                    app.logger.info("Gemini 模型為 FinSimU (從主程序) 配置成功。")
                except Exception as gem_err:
                    app.logger.error(f"為 FinSimU (從主程序) 配置 Gemini 模型失敗: {gem_err}", exc_info=True)
                    report_generation_model = None 
                    stock_analysis_model_gemini = None
        else:
            app.logger.warning("未找到 GEMINI_API_KEY 環境變數。AI 功能將被禁用。")

        init_finsimu_database() 
        app.logger.info(f"AI_WATCHLIST_SIZE is set to: {AI_WATCHLIST_SIZE}") 

        # 確保 APScheduler 只在主進程中啟動，避免重載器導致重複執行
        # 或者在生產環境中 (非 debug 模式)
        # 注意: 這裡的 os.environ.get('WERKZEUG_RUN_MAIN') == 'true' 是針對 Werkzeug 開發服務器的
        # 如果你使用其他 WSGI 服務器如 Gunicorn, uWSGI 部署，你可能需要不同的方式來確保任務只被初始化一次。
        # 對於生產環境，通常建議將調度器作為一個獨立的進程運行，或者使用更成熟的任務隊列系統。
        if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
            if not scheduler.running:
                # 台灣市場任務 (UTC 時間)
                # 台股 UTC+8, 收盤 13:30 (05:30 UTC), 開盤 09:00 (01:00 UTC)
                # 例如: 開盤後30分鐘(UTC 01:30), 收盤後30分鐘(UTC 06:00)
                scheduler.add_job(perform_ga_potential_buys_precalculation, args=['TW'], trigger='cron', hour='1', minute='35', id='ga_tw_open_job_main_v3', misfire_grace_time=3600, replace_existing=True) 
                scheduler.add_job(perform_ga_potential_buys_precalculation, args=['TW'], trigger='cron', hour='6', minute='5', id='ga_tw_close_job_main_v3', misfire_grace_time=3600, replace_existing=True) 
                
                # 美股市場任務 (UTC 時間)
                # 假設 EDT UTC-4, 收盤 16:00 (20:00 UTC), 開盤 09:30 (13:30 UTC)
                # 例如: 開盤後30分鐘(UTC 14:00), 收盤後30分鐘(UTC 20:30)
                scheduler.add_job(perform_ga_potential_buys_precalculation, args=['US'], trigger='cron', hour='13', minute='45', id='ga_us_open_job_main_v3', misfire_grace_time=3600, replace_existing=True) 
                scheduler.add_job(perform_ga_potential_buys_precalculation, args=['US'], trigger='cron', hour='20', minute='35', id='ga_us_close_job_main_v3', misfire_grace_time=3600, replace_existing=True) 
                
                try:
                    scheduler.start()
                    app.logger.info("APScheduler 已在主進程中啟動，用於 GA 潛力買入股預計算。")
                except Exception as e:
                    app.logger.error(f"啟動 APScheduler 失敗: {e}", exc_info=True)
        elif app.debug: # 如果是 debug 模式下的 Werkzeug 子進程
             app.logger.info("APScheduler 未在 Werkzeug 子進程中啟動。")
        
        app.logger.info("啟動 FinSimU Flask 伺服器...")

    # 在開發時，如果 APScheduler 仍然有問題，可以明確禁用重載器
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
