import os
import json
import datetime
import logging
import secrets
import pymysql
import yfinance as yf
import pandas as pd
import pandas_ta as ta # <--- 新增 (來自舊版 StockAnalyzer)
import feedparser      # <--- 新增 (來自舊版 StockAnalyzer)
import urllib.parse    # <--- 新增 (來自舊版 StockAnalyzer)
import re              # <--- 新增 (來自舊版 StockAnalyzer)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory # send_from_directory 可能不需要，看 chart_url 如何處理
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import google.generativeai as genai
from functools import wraps
import plotly.graph_objects as go # <--- 新增 (來自舊版 StockAnalyzer)
from plotly.subplots import make_subplots # <--- 新增 (來自舊版 StockAnalyzer)
import numpy as np               # <--- 新增 (來自舊版 StockAnalyzer)
import time                      # <--- 新增 (來自舊版 StockAnalyzer)


# --- 基本設定 ---
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# --- 圖表文件夾設定 ---
charts_dir = os.path.join(app.static_folder, 'charts')
if not os.path.exists(charts_dir):
    os.makedirs(charts_dir)
    logging.info(f"Created directory: {charts_dir}")


# --- 日誌設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
# Flask app.logger 會在 app context 之後可用，但我們可以在此之前使用標準 logging
# logger = logging.getLogger(__name__) # 如果想用 app.logger, 需在 app context 內

# --- 資料庫設定 ---
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD", "0912559910"), # <<<--- !!! 請務必修改這裡的密碼 !!!
    'database': os.getenv("DB_NAME", "finsimu_db"),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'connect_timeout': 10
}

# --- Gemini API 設定 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
report_generation_model = None
stock_analysis_model_gemini = None # 新增 stock_analysis_model_gemini

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        safety_settings_gemini = [ # 來自舊版的安全設定
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        report_generation_model = genai.GenerativeModel("models/gemini-1.5-flash-latest", safety_settings=safety_settings_gemini)
        stock_analysis_model_gemini = genai.GenerativeModel("models/gemini-1.5-flash-latest", safety_settings=safety_settings_gemini) # 初始化
        logging.info("Gemini Models configured successfully for FinSimU.")
    except Exception as gemini_err:
        logging.error(f"Failed to configure Gemini models for FinSimU: {gemini_err}", exc_info=True)
else:
    logging.warning("GEMINI_API_KEY not found. AI report and stock analysis will be disabled.")


# --- 資料庫輔助函數 ---
def get_db_connection():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except pymysql.Error as e:
        # 使用 app.logger 如果在 app context 內，否則用標準 logging
        logger = app.logger if app else logging
        logger.error(f"Database connection error: {e}")
        return None

def execute_db_query(query, args=None, fetch_one=False, fetch_all=False, commit=False, conn_param=None):
    logger_instance = app.logger if app else logging # 確保 logger 可用
    conn_to_use = conn_param if conn_param else get_db_connection()
    if not conn_to_use:
        logger_instance.error(f"DB query failed: No connection. Query: {query}")
        return None

    result = None
    try:
        with conn_to_use.cursor() as cursor:
            cursor.execute(query, args)
            if commit:
                if not conn_param: conn_to_use.commit()
                result = cursor.lastrowid if cursor.lastrowid else cursor.rowcount
            elif fetch_one:
                result = cursor.fetchone()
            elif fetch_all:
                result = cursor.fetchall()
    except pymysql.Error as e:
        logger_instance.error(f"Database query error: {e}\nQuery: {query}\nArgs: {args}", exc_info=True)
        if conn_to_use and commit and not conn_param: conn_to_use.rollback()
        return None
    finally:
        if conn_to_use and not conn_param:
            conn_to_use.close()
    return result


def init_finsimu_database(): # 單一市場模型的資料庫結構
    logger_instance = app.logger if app else logging
    execute_db_query("""
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        nickname VARCHAR(100) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        investment_style VARCHAR(50),
        market_type ENUM('TW', 'US') NOT NULL DEFAULT 'TW',
        initial_capital_tw DECIMAL(15, 2) DEFAULT 0.00,
        cash_balance_tw DECIMAL(15, 2) DEFAULT 0.00,
        initial_capital_us DECIMAL(15, 2) DEFAULT 0.00,
        cash_balance_us DECIMAL(15, 2) DEFAULT 0.00,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login_at TIMESTAMP NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    logger_instance.info("Users table checked/created.")

    execute_db_query("""
    CREATE TABLE IF NOT EXISTS holdings (
        id INT AUTO_INCREMENT PRIMARY KEY, user_id INT NOT NULL,
        market_type ENUM('TW', 'US') NOT NULL,
        ticker VARCHAR(20) NOT NULL, stock_name VARCHAR(255), shares INT NOT NULL,
        average_cost DECIMAL(15, 4) NOT NULL, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(user_id, market_type, ticker)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    logger_instance.info("Holdings table checked/created.")

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
    logger_instance.info("Trade history table checked/created.")

    execute_db_query("""
    CREATE TABLE IF NOT EXISTS portfolio_history (
        id INT AUTO_INCREMENT PRIMARY KEY, user_id INT NOT NULL,
        market_type ENUM('TW', 'US') NOT NULL,
        timestamp DATETIME NOT NULL, total_value DECIMAL(15, 2) NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE, INDEX(user_id, market_type, timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    logger_instance.info("Portfolio history table checked/created.")

    execute_db_query("""
    CREATE TABLE IF NOT EXISTS stock_data_cache (
        ticker VARCHAR(20) PRIMARY KEY, name VARCHAR(255), current_price DECIMAL(15, 4),
        previous_close DECIMAL(15, 4), daily_change DECIMAL(10, 4), daily_change_percent DECIMAL(8, 4),
        last_fetched TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    logger_instance.info("Stock data cache table checked/created.")

    # ai_vs_user_games and ai_vs_user_trades 保持 market_type
    execute_db_query("""
    CREATE TABLE IF NOT EXISTS ai_vs_user_games (
        id INT AUTO_INCREMENT PRIMARY KEY, user_id INT NOT NULL, market_type ENUM('TW', 'US') NOT NULL,
        stock_ticker VARCHAR(20) NOT NULL, game_start_date DATE NOT NULL, game_end_date DATE NOT NULL,
        ai_strategy_gene JSON, ai_initial_cash DECIMAL(15,2) NOT NULL, user_initial_cash DECIMAL(15,2) NOT NULL,
        ai_final_portfolio_value DECIMAL(15, 2), user_final_portfolio_value DECIMAL(15, 2),
        game_completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    logger_instance.info("AI vs User Games table checked/created.")

    execute_db_query("""
    CREATE TABLE IF NOT EXISTS ai_vs_user_trades (
        id INT AUTO_INCREMENT PRIMARY KEY, game_id INT NOT NULL, trader_type ENUM('user', 'ai') NOT NULL,
        timestamp DATETIME NOT NULL, trade_type ENUM('buy', 'sell') NOT NULL, ticker VARCHAR(20) NOT NULL,
        shares INT NOT NULL, price_per_share DECIMAL(15, 4) NOT NULL,
        user_mood VARCHAR(50), user_reason TEXT,
        FOREIGN KEY (game_id) REFERENCES ai_vs_user_games(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;""")
    logger_instance.info("AI vs User Trades table checked/created.")
    logger_instance.info("FinSimU Database initialization complete (Single Market Model).")


# --- 股票數據服務 (沿用單一市場版的，因為 yfinance 本身不區分帳戶) ---
STOCK_CACHE_DURATION = datetime.timedelta(minutes=1)
def get_stock_data_from_yf(ticker_symbol):
    logger_instance = app.logger if app else logging
    try:
        logger_instance.info(f"YFINANCE FETCH: Attempting to fetch fresh data for {ticker_symbol}")
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        current_price = info.get('currentPrice', info.get('regularMarketPrice', info.get('bid')))
        hist_2d = stock.history(period="2d") # Get last 2 days to ensure previous close

        previous_close = info.get('previousClose')

        # More robust price fetching from history if info is incomplete
        if not hist_2d.empty:
            if current_price is None and len(hist_2d) >= 1:
                current_price = hist_2d['Close'].iloc[-1]
            if len(hist_2d) >= 2: # We need at least two rows to get previous close reliably
                previous_close = hist_2d['Close'].iloc[-2]
            elif len(hist_2d) == 1 and previous_close is None: # If only 1 day, prev_close might be missing
                 previous_close = current_price # Fallback, though not ideal

        if current_price is None or previous_close is None:
             logger_instance.warning(f"Could not reliably determine current or previous price for {ticker_symbol} from yfinance.")
             name = info.get('longName', info.get('shortName', ticker_symbol)) if info else ticker_symbol
             # If only name is available, still cache it
             if name and name != ticker_symbol:
                 execute_db_query(""" REPLACE INTO stock_data_cache (ticker, name, last_fetched) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE name=VALUES(name), last_fetched=VALUES(last_fetched) """, (ticker_symbol.upper(), name, datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)), commit=True)
                 return {'ticker': ticker_symbol.upper(), 'name': name, 'current_price': None, 'previous_close': None, 'daily_change': None, 'daily_change_percent': None, 'last_fetched': datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)}
             return None # Not enough data to proceed

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
        logger_instance.info(f"YFINANCE FETCH: Successfully fetched and cached data for {ticker_symbol}")
        return data_to_cache
    except Exception as e:
        logger_instance.error(f"Error fetching data for {ticker_symbol} from yfinance: {e}", exc_info=True)
        return None

def get_stock_info(ticker_symbol):
    logger_instance = app.logger if app else logging
    ticker_symbol = ticker_symbol.upper()
    cached_data = execute_db_query("SELECT * FROM stock_data_cache WHERE ticker = %s", (ticker_symbol,), fetch_one=True)
    if cached_data and cached_data['last_fetched']:
        last_fetched_dt = cached_data['last_fetched']
        if isinstance(last_fetched_dt, str): # Handle if somehow stored as string
            try: last_fetched_dt = datetime.datetime.fromisoformat(last_fetched_dt)
            except ValueError: last_fetched_dt = datetime.datetime.min # Invalid string, treat as very old

        if last_fetched_dt.tzinfo is not None: # Ensure tz-naive for comparison
            last_fetched_dt = last_fetched_dt.replace(tzinfo=None)

        if (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - last_fetched_dt) < STOCK_CACHE_DURATION:
            if cached_data['current_price'] is not None: # Only return if price data is somewhat complete
                logger_instance.info(f"CACHE HIT for {ticker_symbol}.")
                for key in ['current_price', 'previous_close', 'daily_change', 'daily_change_percent']:
                    if cached_data.get(key) is not None: cached_data[key] = float(cached_data[key])
                return cached_data
    logger_instance.info(f"CACHE MISS for {ticker_symbol}, fetching from yfinance.")
    return get_stock_data_from_yf(ticker_symbol)


# --- StockAnalyzer Class (來自舊版，稍作調整) ---
class StockAnalyzer:
    def __init__(self, ticker: str, period: str = "3y", market: str = "AUTO", temperature: float = 0.6):
        self.logger = app.logger if app else logging
        self.ticker_input = ticker.strip()
        self.ticker = self.ticker_input
        self.period = period
        self.market = market.upper() # TW, US, or AUTO
        self.temperature = max(0.0, min(1.0, temperature))

        # 自動偵測市場 (如果 market == "AUTO") 或修正 ticker
        if self.market == "AUTO":
            if re.fullmatch(r'\d{4,6}', self.ticker): # 纯数字，可能是台股
                self.ticker = f"{self.ticker}.TW"
                self.market = "TW"
            elif self.ticker.endswith(".TW"):
                self.market = "TW"
            else: # 其他情況默認為美股
                self.market = "US"
        elif self.market == "TW" and not self.ticker.endswith(".TW") and re.fullmatch(r'\d{4,6}', self.ticker):
            self.ticker = f"{self.ticker}.TW"
        # (可以加入更多 .US 後綴的檢查，例如 .O, .N 等，但此處簡化)

        self.stock = yf.Ticker(self.ticker)
        self.data = pd.DataFrame()
        self.company_name = self.ticker # Default
        self.currency = 'USD' # Default
        self.pe_ratio, self.market_cap, self.eps, self.roe = None, None, None, None
        self.net_profit_margin_str, self.current_ratio_str = "N/A", "N/A"
        self.model = stock_analysis_model_gemini # 使用已配置的 Gemini 模型

        try:
            self._get_data() # 獲取股價歷史和基本公司資訊
            if not self.data.empty:
                self._get_financial_data() # 獲取詳細財務數據
                self._calculate_indicators() # 計算技術指標
            else:
                self.logger.warning(f"[{self.ticker}] Data is empty. Skipping financials/indicators.")
            self.logger.info(f"StockAnalyzer initialized for {self.ticker}. Market: {self.market}, ROE: {self.roe}, Net Margin: {self.net_profit_margin_str}")
        except Exception as e:
             self.logger.error(f"StockAnalyzer init failed for ({self.ticker_input} -> {self.ticker}): {e}", exc_info=True)
             if not isinstance(self.data, pd.DataFrame) or self.data.empty: self.data = pd.DataFrame() #確保 self.data 是 DataFrame
             if self.company_name is None: self.company_name = self.ticker_input # 使用輸入的 ticker 作為備用

    def _get_data(self):
        try:
            self.logger.info(f"StockAnalyzer ({self.ticker}): Fetching history for '{self.period}'")
            self.data = self.stock.history(period=self.period, timeout=20)
            if self.data.empty:
                self.logger.warning(f"StockAnalyzer ({self.ticker}): No history for '{self.period}'. Trying '1y'.")
                self.data = self.stock.history(period="1y", timeout=15)
                if self.data.empty:
                    # 尝试用 get_stock_info 获取一次，它有更强的缓存和回退
                    fallback_data = get_stock_info(self.ticker)
                    if fallback_data and fallback_data.get('current_price') is not None: # 至少要有當前價格
                         # 嘗試用更短的期間獲取歷史
                        temp_data = self.stock.history(period="6mo", timeout=10)
                        if not temp_data.empty:
                            self.data = temp_data
                            self.logger.warning(f"StockAnalyzer ({self.ticker}): Fallback to 6mo history.")
                        else:
                             raise ValueError(f"無法取得 {self.ticker} 任何有效歷史資料")
                    else:
                        raise ValueError(f"無法取得 {self.ticker} 歷史資料或基本資訊")

            info = {}
            try: info = self.stock.info
            except Exception: self.logger.warning(f"StockAnalyzer ({self.ticker}): Error fetching .info. Trying .fast_info.")
            if not info:
                try: info = self.stock.fast_info
                except Exception: self.logger.error(f"StockAnalyzer ({self.ticker}): Error fetching .fast_info.")

            self.company_name = info.get('longName', info.get('shortName', self.ticker)) if info else self.ticker
            self.currency = info.get('currency', 'TWD' if self.market == 'TW' else 'USD') if info else ('TWD' if self.market == 'TW' else 'USD')
            self.logger.info(f"StockAnalyzer ({self.ticker}): Data/info fetched. Company: {self.company_name}")
        except ValueError as ve:
            self.logger.error(f"StockAnalyzer ({self.ticker}): ValueError fetching data: {ve}")
            self.data = pd.DataFrame() # 確保 self.data 是 DataFrame
            raise # 重新拋出錯誤，讓上層處理
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker}): Generic error fetching data: {e}", exc_info=True)
            self.data = pd.DataFrame() # 確保 self.data 是 DataFrame

    def _get_financial_data(self, retries=1, delay=0.5):
        financials, balance_sheet, info = pd.DataFrame(), pd.DataFrame(), {}
        try:
            self.logger.info(f"StockAnalyzer ({self.ticker}): Fetching financials...")
            try: info = self.stock.info or {}
            except: self.logger.warning(f"StockAnalyzer ({self.ticker}): stock.info failed again in _get_financial_data")

            self.pe_ratio = info.get('trailingPE')
            self.market_cap = info.get('marketCap')
            self.eps = info.get('trailingEps')
            self.roe = info.get('returnOnEquity') # 優先使用 info 的 ROE

            self.current_ratio_str = "N/A"
            if info.get('currentRatio') is not None:
                try: self.current_ratio_str = f"{float(info['currentRatio']):.2f}" if pd.notna(info['currentRatio']) else "N/A"
                except: self.logger.warning(f"StockAnalyzer ({self.ticker}): Invalid currentRatio from info: {info['currentRatio']}")

            for attempt in range(retries): # 嘗試獲取財報和資產負債表
                try:
                    if financials.empty: financials = self.stock.financials
                    if balance_sheet.empty: balance_sheet = self.stock.balance_sheet
                    if not financials.empty and not balance_sheet.empty: break
                except Exception as stmt_err:
                    self.logger.warning(f"StockAnalyzer ({self.ticker}): Error fetching statements (Attempt {attempt+1}/{retries}): {stmt_err}")
                if attempt < retries - 1: time.sleep(delay)

            # 如果 ROE 仍為 None 且財報數據存在，嘗試計算
            if self.roe is None and not financials.empty and not balance_sheet.empty:
                net_income_key = next((k for k in ["Net Income","NetIncome","Net Income Common Stockholders"] if k in financials.index),None)
                equity_key = next((k for k in ["Total Stockholder Equity","Stockholders Equity","TotalEquityGrossMinorityInterest","Stockholders Equity"] if k in balance_sheet.index),None)
                if net_income_key and equity_key and not financials.columns.empty and not balance_sheet.columns.empty:
                    try:
                        ni = financials.loc[net_income_key].iloc[0]
                        eq = balance_sheet.loc[equity_key].iloc[0]
                        if pd.notna(ni) and pd.notna(eq) and eq != 0: self.roe = ni / eq
                    except IndexError: self.logger.warning(f"StockAnalyzer ({self.ticker}): IndexError calculating ROE from statements.")
                    except Exception as roe_calc_err: self.logger.warning(f"StockAnalyzer ({self.ticker}): Exception calculating ROE: {roe_calc_err}")

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
                    except IndexError: self.logger.warning(f"StockAnalyzer ({self.ticker}): IndexError calculating Net Profit Margin.")
                    except Exception as npm_calc_err: self.logger.warning(f"StockAnalyzer ({self.ticker}): Exception calculating Net Profit Margin: {npm_calc_err}")

            if self.current_ratio_str == "N/A" and not balance_sheet.empty: # 如果 info 中沒有，從 balance_sheet 計算
                current_assets_key = next((k for k in ["Current Assets","Total Current Assets"] if k in balance_sheet.index),None)
                current_liab_key = next((k for k in ["Current Liabilities","Total Current Liabilities"] if k in balance_sheet.index),None)
                if current_assets_key and current_liab_key and not balance_sheet.columns.empty:
                    try:
                        ca = balance_sheet.loc[current_assets_key].iloc[0]
                        cl = balance_sheet.loc[current_liab_key].iloc[0]
                        if pd.notna(ca) and pd.notna(cl) and cl != 0:
                            cr_val = ca / cl
                            self.current_ratio_str = f"{cr_val:.2f}"
                    except IndexError: self.logger.warning(f"StockAnalyzer ({self.ticker}): IndexError calculating Current Ratio from balance sheet.")
                    except Exception as cr_calc_err: self.logger.warning(f"StockAnalyzer ({self.ticker}): Exception calculating Current Ratio: {cr_calc_err}")

            self.logger.info(f"StockAnalyzer ({self.ticker}): Financials processed. ROE: {self.roe}, NPM: {self.net_profit_margin_str}, CR: {self.current_ratio_str}")
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker}): Error processing financials: {e}", exc_info=True)
            # Ensure attributes exist even if they fail to be set
            for attr in ['pe_ratio','market_cap','eps','roe']: setattr(self, attr, getattr(self, attr, None))
            for attr_str in ['net_profit_margin_str','current_ratio_str']: setattr(self, attr_str, getattr(self, attr_str, "N/A"))


    def _calculate_indicators(self):
        try:
            if self.data.empty:
                self.logger.warning(f"StockAnalyzer ({self.ticker}): No data available for calculating indicators."); return
            df = self.data.copy()
            # Ensure required columns exist and have enough data points
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] # Volume might not always be needed by all ta functions
            if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']): # Core K-line data
                self.logger.warning(f"StockAnalyzer ({self.ticker}): Missing core K-line columns (Open, High, Low, Close). Cannot calculate many indicators."); return

            min_len_ma = 50 # For MA50
            min_len_rsi_macd = 26 # For MACD slow

            if len(df) >= 5: df.ta.sma(length=5, append=True, col_names='MA5')
            if len(df) >= 20: df.ta.sma(length=20, append=True, col_names='MA20')
            if len(df) >= min_len_ma : df.ta.sma(length=50, append=True, col_names='MA50')
            if len(df) >= 120: df.ta.sma(length=120, append=True, col_names='MA120')

            if len(df) >= 12 : df.ta.rsi(length=12, append=True, col_names='RSI')

            if len(df) >= min_len_rsi_macd:
                macd = df.ta.macd(fast=12, slow=26, signal=9, append=True)
                if macd is not None and not macd.empty:
                    df.rename(columns={'MACD_12_26_9':'MACD', 'MACDs_12_26_9':'MACD_signal', 'MACDh_12_26_9':'MACD_hist'}, inplace=True, errors='ignore')

            if len(df) >= 9: # STOCH k=9
                stoch = df.ta.stoch(k=9, d=3, smooth_k=3, append=True)
                if stoch is not None and not stoch.empty:
                    df.rename(columns={'STOCHk_9_3_3':'K', 'STOCHd_9_3_3':'D'}, inplace=True, errors='ignore')
                    if 'K' in df.columns and 'D' in df.columns: df['J'] = 3 * df['K'] - 2 * df['D']

            if len(df) >= 20: # BBANDS length=20
                bbands = df.ta.bbands(length=20, std=2, append=True)
                if bbands is not None and not bbands.empty:
                    df.rename(columns={'BBL_20_2.0':'BB_lower', 'BBM_20_2.0':'BB_middle', 'BBU_20_2.0':'BB_upper'}, inplace=True, errors='ignore')

            if len(df) >= 14: df.ta.willr(length=14, append=True, col_names='WMSR')

            if 'Close' in df.columns:
                if len(df) >= 12: df['PSY'] = df['Close'].diff().apply(lambda x:1 if x>0 else 0).rolling(12).sum()/12*100
                if len(df) >= 6:
                    bm = df['Close'].rolling(window=6).mean()
                    df['BIAS6'] = ((df['Close']-bm)/bm.replace(0,np.nan)*100).replace([np.inf,-np.inf],np.nan)

            self.data = df
            self.logger.info(f"StockAnalyzer ({self.ticker}): Indicators calculated.")
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker}): Error calculating indicators: {e}", exc_info=True)

    def _identify_patterns(self, days=30):
        try:
            if self.data.empty or len(self.data) < 2: return ["數據不足"]
            df = self.data.tail(days).copy()
            patterns = []
            cols = df.columns

            if 'MA5' in cols and 'MA20' in cols and len(df) >= 2 and df['MA5'].notna().any() and df['MA20'].notna().any():
                # Ensure iloc[-2] is valid
                if len(df['MA5'].dropna()) >= 2 and len(df['MA20'].dropna()) >= 2:
                    if df['MA5'].iloc[-2] <= df['MA20'].iloc[-2] and df['MA5'].iloc[-1] > df['MA20'].iloc[-1]:
                        patterns.append("黃金交叉 (MA5>MA20)")
                    if df['MA5'].iloc[-2] >= df['MA20'].iloc[-2] and df['MA5'].iloc[-1] < df['MA20'].iloc[-1]:
                        patterns.append("死亡交叉 (MA5<MA20)")

            if 'Close' in cols and 'BB_upper' in cols and df['Close'].notna().any() and df['BB_upper'].notna().any():
                 if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1]: patterns.append("突破布林帶上軌")
            if 'Close' in cols and 'BB_lower' in cols and df['Close'].notna().any() and df['BB_lower'].notna().any():
                 if df['Close'].iloc[-1] < df['BB_lower'].iloc[-1]: patterns.append("跌破布林帶下軌")

            if 'RSI' in cols and df['RSI'].notna().any():
                if df['RSI'].iloc[-1] > 75: patterns.append("RSI 超買 (>75)")
                if df['RSI'].iloc[-1] < 25: patterns.append("RSI 超賣 (<25)")

            return patterns if patterns else ["近期無明顯技術形態"]
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker}): Error identifying patterns: {e}", exc_info=True)
            return ["無法識別形態"]

    def _generate_chart(self, days=180):
        try:
            if self.data.empty or len(self.data) < 2:
                raise ValueError(f"Data insufficient for chart generation ({self.ticker})")

            df = self.data.tail(days).copy()
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
            cols = df.columns

            # K線圖或收盤價線圖
            if all(c in cols for c in ['Open','High','Low','Close']) and df['Close'].notna().any():
                fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],name='K線'),row=1,col=1)
            elif 'Close' in cols and df['Close'].notna().any(): # 備用：如果沒有OHL，只畫收盤價
                fig.add_trace(go.Scatter(x=df.index,y=df['Close'],name='收盤價',line=dict(color='lightblue',width=1.5)),row=1,col=1)
            else:
                self.logger.warning(f"StockAnalyzer ({self.ticker}): Not enough data for Candlestick or Close line in chart.")


            # 移動平均線
            ma_map = {'MA5':'orange', 'MA20':'blue', 'MA50':'purple', 'MA120':'green'}
            for ma,color in ma_map.items():
                if ma in cols and df[ma].notna().any():
                    fig.add_trace(go.Scatter(x=df.index,y=df[ma],name=ma,line=dict(color=color,width=1)),row=1,col=1)

            # 布林帶
            if all(c in cols for c in ['BB_upper','BB_middle','BB_lower']) and df['BB_upper'].notna().any():
                fig.add_trace(go.Scatter(x=df.index,y=df['BB_upper'],name='布林上軌',line=dict(color='rgba(173,204,255,0.5)',width=1),fill=None),row=1,col=1)
                fig.add_trace(go.Scatter(x=df.index,y=df['BB_lower'],name='布林下軌',line=dict(color='rgba(173,204,255,0.5)',width=1,dash='dash'),fill='tonexty',fillcolor='rgba(173,204,255,0.05)'),row=1,col=1)
                fig.add_trace(go.Scatter(x=df.index,y=df['BB_middle'],name='布林中軌',line=dict(color='rgba(220,220,220,0.6)',width=1,dash='dot')),row=1,col=1) # BB_middle often same as MA20

            # 成交量
            if 'Volume' in cols and df['Volume'].notna().any() and df['Volume'].sum() > 0:
                volume_colors = ['#2ca02c' if row['Close'] >= row['Open'] else '#d62728' for index, row in df.iterrows()] if all(c in cols for c in ['Open','Close']) else 'grey'
                fig.add_trace(go.Bar(x=df.index,y=df['Volume'],name='成交量',marker_color=volume_colors,marker_line_width=0),row=2,col=1)

            fig.update_layout(
                title=f'{self.company_name or self.ticker} ({self.ticker}) 技術分析 ({days}天)',
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                height=600,
                margin=dict(l=50,r=120,t=80,b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,30,0.2)', # Slightly darker plot area for better contrast
                font=dict(color='white',size=11),
                legend=dict(orientation="v",yanchor="top",y=1,xanchor="left",x=1.02,bgcolor="rgba(30,30,30,0.7)",bordercolor="rgba(200,200,200,0.5)",borderwidth=1)
            )
            fig.update_yaxes(title_text='價格 ('+self.currency+')',row=1,col=1,title_font_size=10,tickfont_size=9,side='left', gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(title_text='成交量',row=2,col=1,title_font_size=10,tickfont_size=9,side='left', gridcolor='rgba(255,255,255,0.1)')
            fig.update_xaxes(showgrid=True,gridwidth=0.5,gridcolor='rgba(255,255,255,0.1)')

            chart_filename_base = f"{self.ticker.replace('.','_')}_simplified_chart_{secrets.token_hex(4)}.html"
            chart_path_disk = os.path.join(charts_dir, chart_filename_base)
            fig.write_html(chart_path_disk,full_html=False,include_plotlyjs='cdn')

            chart_url_path = f"charts/{chart_filename_base}" # Relative path for url_for
            self.logger.info(f"StockAnalyzer ({self.ticker}): SIMPLIFIED Chart generated: {chart_url_path}")
            return chart_url_path
        except ValueError as ve:
            self.logger.error(f"StockAnalyzer ({self.ticker}): ValueError generating SIMPLIFIED chart: {ve}")
            return None
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker}): Unexpected error generating SIMPLIFIED chart: {e}",exc_info=True)
            return None


    def _get_stock_news(self, days=7, num_news=5):
        try:
            # 根據市場決定搜尋語言和地區
            query_base = self.company_name if self.market != "TW" else self.ticker.replace('.TW','') # 台股用代號搜尋可能更準
            lang, geo, ceid_sfx = ('en-US','US','US:en') if self.market != "TW" else ('zh-TW','TW','TW:zh-Hant')

            # 為非代號的台股查詢加上 "股票"
            query_suffix = ""
            if self.market == "TW" and not re.fullmatch(r'\d{4,6}', query_base):
                query_suffix = " 股票"
            elif self.market != "TW" and not any(kw in query_base.lower() for kw in [" stock", " inc", " corp", " ltd"]):
                 query_suffix = " stock"

            final_query = query_base + query_suffix

            url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(final_query)}&hl={lang}&gl={geo}&ceid={ceid_sfx}"
            self.logger.info(f"StockAnalyzer ({self.ticker}): Fetching news with URL: {url}")

            feed_data = feedparser.parse(url)
            relevant_news, now_utc = [], datetime.datetime.now(datetime.timezone.utc)

            if not feed_data.entries:
                self.logger.warning(f"StockAnalyzer ({self.ticker}): No Google News entries found for query '{final_query}'")
                return []

            for entry in feed_data.entries:
                try:
                    pub_time_utc = now_utc # Default to now if no parsed time
                    if hasattr(entry,'published_parsed') and entry.published_parsed:
                        try: pub_time_utc = datetime.datetime(*entry.published_parsed[:6],tzinfo=datetime.timezone.utc)
                        except : pass # Keep default if parsing fails

                    if (now_utc - pub_time_utc).days <= days:
                        relevant_news.append({
                            'title':entry.get('title',"無標題").strip(),
                            'link':entry.get('link',"#"),
                            'date':pub_time_utc.strftime('%Y-%m-%d'), # Use UTC date
                            'source':entry.get('source',{}).get('title','Google News')
                        })
                    if len(relevant_news) >= num_news: break
                except Exception as item_err:
                    self.logger.error(f"StockAnalyzer ({self.ticker}): Error processing news item '{entry.get('title','N/A')}': {item_err}")

            self.logger.info(f"StockAnalyzer ({self.ticker}): Found {len(relevant_news)} relevant news items.")
            return relevant_news
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker}): Error fetching or parsing stock news: {e}",exc_info=True)
            return []

    def _get_ai_analysis(self, news_list: list = None):
        if not self.model: return "AI 分析模型未能成功載入或初始化。"
        try:
            if self.data.empty or len(self.data) < 2:
                return f"股票 ({self.ticker}) 歷史數據不足，無法進行AI分析。"

            last_close = self.data['Close'].iloc[-1] if not self.data.empty else None
            daily_change_pct, weekly_change_pct, monthly_change_pct = None, None, None

            if last_close is not None:
                if len(self.data) >= 2: daily_change_pct = (self.data['Close'].iloc[-1]/self.data['Close'].iloc[-2]-1)*100 if self.data['Close'].iloc[-2]!=0 else None
                if len(self.data) >= 6: weekly_change_pct = (self.data['Close'].iloc[-1]/self.data['Close'].iloc[-6]-1)*100 if self.data['Close'].iloc[-6]!=0 else None # Approx 1 week
                if len(self.data) >= 23: monthly_change_pct = (self.data['Close'].iloc[-1]/self.data['Close'].iloc[-23]-1)*100 if self.data['Close'].iloc[-23]!=0 else None # Approx 1 month

            # 數據合理性檢查
            for val_ptr in [daily_change_pct, weekly_change_pct, monthly_change_pct]:
                if val_ptr is not None and (val_ptr > 200 or val_ptr < -90): val_ptr = None # 過濾極端不合理值

            technical_patterns = ", ".join(self._identify_patterns()) if self._identify_patterns() else "近期無明顯技術形態"

            def format_value(value, precision=2, is_currency=False, is_percentage=False, currency_symbol=''):
                if value is None or pd.isna(value) or str(value).upper()=='N/A': return "N/A"
                try:
                    num = float(value)
                    if is_percentage: return f"{num:.{precision}f}%"

                    symbol_to_use = currency_symbol if is_currency else ''
                    if abs(num) >= 1e12 and is_currency: return f"{symbol_to_use}{num/1e12:.1f}兆"
                    if abs(num) >= 1e8 and is_currency: return f"{symbol_to_use}{num/1e8:.1f}億"
                    if abs(num) >= 1e4 and is_currency and self.currency=='TWD': return f"{symbol_to_use}{num/1e4:.1f}萬" # 台幣適用萬

                    # 一般數字或小額貨幣
                    return f"{symbol_to_use}{num:,.{precision}f}" if (abs(num) < 1e6 or not is_currency) else \
                           (f"{symbol_to_use}{num/1e9:.1f}B" if abs(num) >= 1e9 else f"{symbol_to_use}{num/1e6:.1f}M") # 處理百萬(M)和十億(B)
                except (ValueError, TypeError): return str(value) # 無法轉換則返回原樣

            currency_sym = '$' if self.currency == 'USD' else ('NT$' if self.currency == 'TWD' else self.currency)
            latest_data_row = self.data.iloc[-1] if not self.data.empty else pd.Series(dtype='object')

            news_summary_str = "近期無相關新聞。"
            if news_list:
                titles = [f"{idx+1}. {news_item.get('title','N/A')} (來源: {news_item.get('source','N/A')})" for idx, news_item in enumerate(news_list[:3])] # 最多取三條
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
(其他指標如MA, MACD, KDJ, 布林帶等已反映在圖表中，此處不贅述)

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
7.  **總結與投資展望:** (綜合評價，提出短期 (1週-1個月) 和中期 (3-6個月) 的展望，並給出明確的操作建議，例如：積極買入、逢低吸納、中性持有、謹慎避開、考慮賣出等。**務必包含免責聲明：此分析僅為模擬AI提供，基於歷史數據和公開信息，不構成任何真實投資建議，用戶應獨立判斷並自負風險。**)
"""
            generation_config_settings = genai.types.GenerationConfig(temperature=self.temperature)
            self.logger.info(f"StockAnalyzer ({self.ticker}): Generating AI analysis with Gemini model (Temperature: {self.temperature}).")

            response = self.model.generate_content(prompt_text, generation_config=generation_config_settings, safety_settings=safety_settings_gemini)

            if not response.candidates or (hasattr(response,'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason):
                 feedback_reason = response.prompt_feedback.block_reason if hasattr(response,'prompt_feedback') and response.prompt_feedback else 'N/A'
                 self.logger.warning(f"StockAnalyzer ({self.ticker}): AI analysis content generation blocked. Reason: {feedback_reason}")
                 return f"AI 分析請求因安全或內容政策被阻擋。原因: {feedback_reason}"

            # 處理 Gemini 可能的多部分回應
            analysis_text = ""
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text'):
                        analysis_text += part.text

            if not analysis_text: # 如果還是空的
                 self.logger.warning(f"StockAnalyzer ({self.ticker}): AI analysis generated empty text. Full response: {response}")
                 return "AI 分析未能生成有效內容，請稍後再試。"

            return analysis_text.strip()
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker}): Error generating AI analysis: {e}",exc_info=True)
            return f"生成AI股票分析報告時發生未預期錯誤: {str(e)}"

    def get_stock_summary(self):
        try:
            if self.data.empty or len(self.data) < 2:
                self.logger.warning(f"StockAnalyzer ({self.ticker_input}): Data is empty or insufficient for summary.")
                return {
                    "ticker": self.ticker_input,
                    "company_name": self.company_name or self.ticker_input,
                    "error": f"數據不足 ({self.ticker_input})"
                }

            latest_row = self.data.iloc[-1] if not self.data.empty else pd.Series(dtype='object')
            price_change_val_pct = None
            if not self.data.empty and len(self.data) >= 2 and 'Close' in self.data.columns:
                 prev_close = self.data['Close'].iloc[-2]
                 if pd.notna(latest_row.get('Close')) and pd.notna(prev_close) and prev_close != 0:
                     price_change_val_pct = (latest_row.get('Close') / prev_close - 1) * 100

            chart_relative_path = self._generate_chart() # 這會返回 'charts/filename.html' 或 None
            news_items = self._get_stock_news()
            ai_report_text = self._get_ai_analysis(news_list=news_items)

            summary_map = {
                'ticker': self.ticker,
                'company_name': self.company_name or 'N/A',
                'currency': self.currency,
                'current_price': latest_row.get('Close') if pd.notna(latest_row.get('Close')) else None,
                'price_change': f"{price_change_val_pct:+.2f}%" if price_change_val_pct is not None else "N/A",
                'price_change_value': (latest_row.get('Close') - self.data['Close'].iloc[-2]) if price_change_val_pct is not None and pd.notna(latest_row.get('Close')) and len(self.data) >=2 and pd.notna(self.data['Close'].iloc[-2]) else None,
                'volume': int(latest_row.get('Volume',0)) if pd.notna(latest_row.get('Volume')) else 0,
                'pe_ratio': self.pe_ratio if pd.notna(self.pe_ratio) else None,
                'market_cap': self.market_cap if pd.notna(self.market_cap) else None,
                'eps': self.eps if pd.notna(self.eps) else None,
                'roe': self.roe if pd.notna(self.roe) else None, # Store raw ROE (decimal)
                'net_profit_margin': self.net_profit_margin_str or 'N/A', # Already a string with % or N/A
                'current_ratio': self.current_ratio_str or 'N/A', # Already a string or N/A
                'rsi': latest_row.get('RSI') if pd.notna(latest_row.get('RSI')) else None,
                'patterns': self._identify_patterns(),
                'chart_path': chart_relative_path, # Storing the relative path
                'news': news_items,
                'analysis': ai_report_text
            }
            # Add formatted ROE for display if raw ROE is available
            summary_map["roe_display"] = f"{summary_map['roe']*100:.2f}%" if isinstance(summary_map["roe"],(int,float)) else "N/A"

            return summary_map
        except Exception as e:
            self.logger.error(f"StockAnalyzer ({self.ticker_input}): Error in get_stock_summary: {e}",exc_info=True)
            return {
                "ticker": self.ticker_input,
                "company_name": self.company_name or self.ticker_input,
                "error":f"獲取股票綜合分析時發生錯誤: {str(e)}"
            }


# --- FinSimU Core User Auth & Helper Functions (單一市場模型) ---
def get_current_user_id(): return session.get('user_id')

def get_user_data(user_id):
    logger_instance = app.logger if app else logging
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
            is_api = request.endpoint and (request.endpoint.startswith('api_') or (hasattr(request,'blueprint') and request.blueprint=='api'))
            if is_api: return jsonify(success=False,message="Authentication required."),401
            return redirect(url_for('finsimu_login_route',next=request.url))
        return f(*args,**kwargs)
    return decorated_function

def _record_portfolio_value_with_cursor(cursor,user_id,market_type,total_value):
    logger_instance = app.logger if app else logging
    try:
        cursor.execute("INSERT INTO portfolio_history(user_id,market_type,timestamp,total_value)VALUES(%s,%s,%s,%s)",(user_id,market_type,datetime.datetime.now(),float(total_value)))
        return True
    except pymysql.Error as e:
        logger_instance.error(f"Error recording portfolio value for user {user_id}, market {market_type}: {e}")
        return False

def update_holdings_in_db(cursor, user_id, market_type, ticker, stock_name, shares_change, current_price_for_trade):
    logger_instance = app.logger if app else logging
    try:
        cursor.execute("SELECT shares, average_cost FROM holdings WHERE user_id = %s AND market_type = %s AND ticker = %s FOR UPDATE",
                       (user_id, market_type, ticker))
        current_holding = cursor.fetchone()

        if current_holding:
            new_shares = current_holding['shares'] + shares_change
            if new_shares < 0:
                logger_instance.error(f"Error: Selling more {ticker} than owned by user {user_id}/{market_type}. Owned: {current_holding['shares']}, Trying to change by: {shares_change}")
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
            logger_instance.error(f"Error updating holdings: Attempt to sell non-existent holding {ticker} for user {user_id}/{market_type}.")
            return False
        else:
            logger_instance.info(f"No change in shares for {ticker}, user {user_id}/{market_type}.")

        return True
    except pymysql.Error as e:
        logger_instance.error(f"DB error in update_holdings_in_db for user {user_id}/{market_type}, ticker {ticker}: {e}", exc_info=True)
        return False

# --- FinSimU Core API Routes (單一市場模型) ---
@app.route('/api/register', methods=['POST'])
def api_register():
    logger_instance = app.logger if app else logging
    data = request.get_json()
    logger_instance.info(f"Registration attempt: {data.get('nickname')}")
    nickname = data.get('nickname', '').strip()
    password = data.get('password', '')
    investment_style = data.get('investmentStyle', 'Conservative')
    market_type = data.get('marketType', '').upper()

    if not nickname or not password:
        return jsonify({'success': False, 'message': 'Nickname and password are required.'}), 400
    if market_type not in ['TW', 'US']:
        return jsonify({'success': False, 'message': 'Please select a valid market (TW or US).'}), 400
    if len(password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters.'}), 400

    fixed_initial_capital = 10000000.00
    initial_capital_tw = fixed_initial_capital if market_type == 'TW' else 0.00
    cash_balance_tw = initial_capital_tw
    initial_capital_us = fixed_initial_capital if market_type == 'US' else 0.00
    cash_balance_us = initial_capital_us

    conn = get_db_connection()
    if not conn: return jsonify({'success': False, 'message': 'Database connection error.'}), 500

    user_id = None
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE nickname = %s", (nickname,))
            if cursor.fetchone():
                return jsonify({'success': False, 'message': 'Nickname already exists. Please choose another.'}), 409

            password_hash = generate_password_hash(password)
            cursor.execute("""INSERT INTO users (nickname, password_hash, investment_style, market_type,
                                                initial_capital_tw, cash_balance_tw,
                                                initial_capital_us, cash_balance_us)
                              VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                           (nickname, password_hash, investment_style, market_type,
                            initial_capital_tw, cash_balance_tw,
                            initial_capital_us, cash_balance_us))
            user_id = cursor.lastrowid
            if not user_id: raise pymysql.Error("Failed to get lastrowid after user insert.")

            if not _record_portfolio_value_with_cursor(cursor, user_id, market_type, fixed_initial_capital):
                raise pymysql.Error(f"Failed to record initial portfolio history for market {market_type} during registration.")

        conn.commit()
        logger_instance.info(f"New user '{nickname}' (ID: {user_id}, Market: {market_type}) registered successfully.")
        return jsonify({'success': True, 'message': 'Registration successful! Please log in.', 'username': nickname})
    except pymysql.Error as e:
        logger_instance.error(f"Database error during registration for {nickname}: {e}", exc_info=True)
        if conn: conn.rollback()
        em = e.args[1] if len(e.args)>1 and isinstance(e.args[1],str) else "Unknown DB error"
        return jsonify({'success': False, 'message': f'Database error: {em}'}), 500
    except Exception as e:
        logger_instance.error(f"Unexpected error during registration: {e}", exc_info=True)
        if conn: conn.rollback()
        return jsonify({'success': False, 'message': 'An unexpected error occurred during registration.'}), 500
    finally:
        if conn and conn.open: conn.close()

@app.route('/api/login', methods=['POST'])
def api_login():
    logger_instance = app.logger if app else logging
    data = request.get_json()
    nickname = data.get('nickname', '').strip()
    password = data.get('password', '')
    logger_instance.info(f"Login attempt for nickname: {nickname}")

    if not nickname or not password:
        return jsonify({'success': False, 'message': 'Nickname and password are required.'}), 400

    user = execute_db_query("SELECT id, nickname, password_hash, market_type FROM users WHERE nickname = %s", (nickname,), fetch_one=True)

    if user and check_password_hash(user['password_hash'], password):
        session.clear()
        session['user_id'] = user['id']
        session['username'] = user['nickname']
        session['market_type'] = user['market_type']

        execute_db_query("UPDATE users SET last_login_at = %s WHERE id = %s", (datetime.datetime.now(), user['id']), commit=True)

        logger_instance.info(f"User '{nickname}' (ID: {user['id']}, Market: {user['market_type']}) logged in. Session: {dict(session)}")
        return jsonify({'success': True, 'message': 'Login successful.', 'username': user['nickname'], 'marketType': user['market_type']})
    else:
        logger_instance.warning(f"Failed login attempt for nickname: {nickname}")
        return jsonify({'success': False, 'message': 'Invalid nickname or password.'}), 401

@app.route('/api/user_session', methods=['GET'])
@login_required
def get_user_session_api():
    logger_instance = app.logger if app else logging
    user_id = get_current_user_id()
    logger_instance.info(f"API /user_session: Fetching session for user_id: {user_id}")

    user = get_user_data(user_id)
    if not user:
        session.clear()
        return jsonify({'loggedIn': False, 'message': 'User data inconsistency. Session cleared.'}), 500

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
            logger_instance.error(f"DB error fetching extended user session data for user {user_id}, market {market_type}: {db_err}")
        finally:
            if conn: conn.close()

    return jsonify({
        'loggedIn': True,
        'currentUser': {
            'id': user['id'], 'name': user['nickname'],
            'investmentStyle': user['investment_style'],
            'marketType': market_type,
            'account': market_specific_data
        }
    })

@app.route('/api/stock_quote/<path:ticker>', methods=['GET'])
@login_required
def stock_quote_api(ticker):
    logger_instance = app.logger if app else logging
    logger_instance.info(f"API call for stock_quote: {ticker}")
    data = get_stock_info(ticker)
    if data: return jsonify({'success': True, 'data': data})
    return jsonify({'success': False, 'message': f'Could not retrieve data for {ticker}.'}), 404


@app.route('/api/search_stocks', methods=['GET'])
@login_required
def search_stocks_api():
    logger_instance = app.logger if app else logging
    query = request.args.get('q', '').strip().lower()
    user_id = get_current_user_id()
    user = get_user_data(user_id)
    if not user: return jsonify({'success': False, 'message': 'User not found.'}), 401

    market_type = user['market_type']
    logger_instance.info(f"Stock search for user {user_id} (Market: {market_type}), query: '{query}'")

    if not query or len(query) < 1: return jsonify({'success': True, 'stocks': []})
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
            logger_instance.error(f"Error searching stocks in cache for query '{query}', market {market_type}: {e}")
        finally:
            conn.close()

    if not results and len(query) <= 7:
        potential_ticker = query.upper()
        if market_type == 'TW' and not potential_ticker.endswith('.TW'):
            if not any(char.isdigit() for char in potential_ticker.split('.')[0]) or \
               any(potential_ticker.endswith(s) for s in ['.O', '.N', '.K', '.L']):
                 pass
            else:
                potential_ticker_tw = f"{potential_ticker}.TW"
                stock_data_yf_tw = get_stock_data_from_yf(potential_ticker_tw)
                if stock_data_yf_tw and stock_data_yf_tw.get('current_price') is not None:
                    results.append({'ticker': stock_data_yf_tw['ticker'], 'name': stock_data_yf_tw['name'], 'price': stock_data_yf_tw['current_price'],
                                    'change': stock_data_yf_tw['daily_change_percent'],
                                    'change_type': "positive" if stock_data_yf_tw.get('daily_change_percent', 0) >= 0 else "negative"}) # Added .get default for safety
                    return jsonify({'success': True, 'stocks': results})

        if not (market_type == 'TW' and potential_ticker.endswith('.TW')):
            stock_data_yf = get_stock_data_from_yf(potential_ticker)
            if stock_data_yf and stock_data_yf.get('current_price') is not None:
                if market_type == 'US' and stock_data_yf['ticker'].endswith('.TW'):
                    pass
                else:
                    results.append({'ticker': stock_data_yf['ticker'], 'name': stock_data_yf['name'], 'price': stock_data_yf['current_price'],
                                    'change': stock_data_yf['daily_change_percent'],
                                    'change_type': "positive" if stock_data_yf.get('daily_change_percent', 0) >= 0 else "negative"})

    logger_instance.info(f"Search for '{query}' (Market: {market_type}) returned {len(results)} results.")
    return jsonify({'success': True, 'stocks': results})


@app.route('/api/trade', methods=['POST'])
@login_required
def trade_api():
    logger_instance = app.logger if app else logging
    user_id = get_current_user_id()
    data = request.get_json()
    logger_instance.info(f"Trade API request for user {user_id}: {data}")

    trade_type = data.get('type')
    ticker = data.get('ticker', '').upper()
    try: shares = int(data.get('shares', 0))
    except (ValueError, TypeError): return jsonify({'success': False, 'message': 'Invalid shares format.'}), 400
    mood, reason = data.get('mood'), data.get('reason')

    user_account = get_user_data(user_id)
    if not user_account: return jsonify({'success': False, 'message': 'Could not retrieve user account details.'}), 500

    market_type = user_account['market_type']

    if not all([trade_type, ticker, shares > 0]): return jsonify({'success': False, 'message': 'Missing trade information or invalid shares.'}), 400
    if trade_type not in ['buy', 'sell']: return jsonify({'success': False, 'message': 'Invalid trade type.'}), 400

    if market_type == 'TW' and not ticker.endswith('.TW'):
        return jsonify({'success': False, 'message': f'Invalid ticker ({ticker}) for your TW market account. TW stocks must end with .TW'}), 400
    if market_type == 'US' and ticker.endswith('.TW'):
        return jsonify({'success': False, 'message': f'Invalid ticker ({ticker}) for your US market account. You cannot trade .TW stocks.'}), 400

    stock_live_data = get_stock_info(ticker)
    if not stock_live_data or stock_live_data.get('current_price') is None:
        return jsonify({'success': False, 'message': f'Could not get current price for {ticker}. Trade cancelled.'}), 404

    price_at_trade = float(stock_live_data['current_price'])
    stock_name_at_trade = stock_live_data.get('name', ticker)
    total_transaction_value = shares * price_at_trade
    commission = round(total_transaction_value * 0.001425, 2) if market_type == 'TW' else round(max(1.0, total_transaction_value * 0.005), 2)

    cash_balance_key = f'cash_balance_{market_type.lower()}'
    current_cash = user_account[cash_balance_key]

    conn = get_db_connection()
    if not conn: return jsonify({'success': False, 'message': 'Database connection error.'}), 500

    try:
        with conn.cursor() as cursor:
            if trade_type == 'buy':
                if total_transaction_value + commission > current_cash:
                    return jsonify({'success': False, 'message': 'Not enough cash balance.'}), 400
                cursor.execute(f"UPDATE users SET {cash_balance_key} = {cash_balance_key} - %s WHERE id = %s", (total_transaction_value + commission, user_id))

            elif trade_type == 'sell':
                cursor.execute("SELECT shares FROM holdings WHERE user_id = %s AND market_type = %s AND ticker = %s FOR UPDATE", (user_id, market_type, ticker))
                current_holding = cursor.fetchone()
                owned_shares = current_holding['shares'] if current_holding else 0
                if shares > owned_shares:
                    return jsonify({'success': False, 'message': f'Not enough shares to sell. You own {owned_shares}.'}), 400
                cursor.execute(f"UPDATE users SET {cash_balance_key} = {cash_balance_key} + %s WHERE id = %s", (total_transaction_value - commission, user_id))

            sql_trade_hist = """INSERT INTO trade_history (user_id, market_type, timestamp, trade_type, ticker, stock_name, shares, price_per_share, total_value, mood, reason, commission)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql_trade_hist, (user_id, market_type, datetime.datetime.now(), trade_type, ticker, stock_name_at_trade, shares, price_at_trade, total_transaction_value, mood, reason, commission))

            shares_change = shares if trade_type == 'buy' else -shares
            holdings_update_success = update_holdings_in_db(cursor, user_id, market_type, ticker, stock_name_at_trade, shares_change, price_at_trade)

            if not holdings_update_success:
                conn.rollback()
                logger_instance.critical(f"Holdings update failed for user {user_id}/{market_type}, trade: {data}. Transaction rolled back.")
                return jsonify({'success': False, 'message': 'Trade processing failed during holdings update. Transaction cancelled.'}), 500

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
                logger_instance.error(f"Failed to record portfolio value post-trade for {user_id}/{market_type}. Transaction rolled back.")
                return jsonify({'success': False, 'message': 'Trade processed but failed to update portfolio history. Transaction cancelled.'}), 500

            conn.commit()

            final_user_data = get_user_data(user_id)
            if not final_user_data:
                logger_instance.error(f"Could not fetch final user data for {user_id} after successful trade.")
                return jsonify({'success': True, 'message': f'Trade successful: {trade_type.capitalize()} {shares} of {ticker}. Check your portfolio.',
                                'newCashBalance': float(updated_cash_balance_from_db), 'marketType': market_type})

            return jsonify({'success': True, 'message': f'Trade successful: {trade_type.capitalize()} {shares} of {ticker}.',
                            'newCashBalance': float(final_user_data[cash_balance_key]), 'marketType': market_type})

    except pymysql.Error as e:
        logger_instance.error(f"Database error during trade for user {user_id}, ticker {ticker}: {e}", exc_info=True)
        if conn and conn.open: conn.rollback()
        return jsonify({'success': False, 'message': f'Database error during trade: {str(e)}'}), 500
    except Exception as e:
        logger_instance.error(f"Unexpected error during trade for user {user_id}: {e}", exc_info=True)
        if conn and conn.open: conn.rollback()
        return jsonify({'success': False, 'message': f'An unexpected error occurred during trade: {str(e)}'}), 500
    finally:
        if conn and conn.open: conn.close()


@app.route('/api/portfolio_data', methods=['GET'])
@login_required
def portfolio_data_api():
    logger_instance = app.logger if app else logging
    user_id = get_current_user_id()
    user_account_full = get_user_data(user_id)
    if not user_account_full: return jsonify({'success': False, 'message': 'User account not found.'}), 404

    market_type = user_account_full['market_type']
    logger_instance.info(f"Portfolio data request for user {user_id}, market {market_type}")

    cash_balance = user_account_full[f'cash_balance_{market_type.lower()}']
    initial_capital = user_account_full[f'initial_capital_{market_type.lower()}']

    conn = get_db_connection()
    if not conn: return jsonify({'success': False, 'message': 'Database connection error.'}), 500
    try:
        with conn.cursor() as cursor:
            cursor.execute("""SELECT h.ticker, h.stock_name, h.shares, h.average_cost,
                                     sc.current_price, sc.previous_close
                              FROM holdings h LEFT JOIN stock_data_cache sc ON h.ticker = sc.ticker
                              WHERE h.user_id = %s AND h.market_type = %s AND h.shares > 0""", (user_id, market_type))
            holdings_db = cursor.fetchall()

            holdings_frontend, total_stock_market_value, total_today_pl_value_for_stocks = [], 0, 0
            for h_db in holdings_db:
                current_price = float(h_db['current_price']) if h_db['current_price'] is not None else float(h_db['average_cost'])
                previous_close = float(h_db['previous_close']) if h_db['previous_close'] is not None else current_price
                market_value = h_db['shares'] * current_price
                total_stock_market_value += market_value
                cost_basis = h_db['shares'] * float(h_db['average_cost'])
                total_pl_holding = market_value - cost_basis
                total_pl_percent_holding = (total_pl_holding / cost_basis) * 100 if cost_basis != 0 else 0
                holding_today_pl_value = (current_price - previous_close) * h_db['shares']
                holding_today_pl_percent = (holding_today_pl_value / (previous_close * h_db['shares'])) * 100 if previous_close != 0 and h_db['shares'] > 0 else 0.0 # Avoid division by zero if previous_close or shares is zero
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
            elif total_today_pl_value_for_stocks > 0: # e.g. initial capital was 0, now value is >0 due to profit
                 today_pl_portfolio_percent = float('inf')
            elif total_today_pl_value_for_stocks < 0:
                 today_pl_portfolio_percent = float('-inf')

            overview = {'totalPortfolioValue': total_portfolio_value, 'totalOverallPL': overall_total_pl,
                        'totalOverallPLPercent': overall_total_pl_percent, 'todayTotalPL': total_today_pl_value_for_stocks,
                        'todayTotalPLPercent': today_pl_portfolio_percent, 'cashBalance': cash_balance}

            return jsonify({'success': True, 'overview': overview, 'holdings': holdings_frontend, 'marketType': market_type})

    except pymysql.Error as e:
        logger_instance.error(f"DB error fetching portfolio data for user {user_id}, market {market_type}: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'}), 500
    except Exception as e:
        logger_instance.error(f"Unexpected error fetching portfolio data for user {user_id}, market {market_type}: {e}", exc_info=True)
        return jsonify({'success': False, 'message': 'An unexpected error occurred.'}), 500
    finally:
        if conn: conn.close()


@app.route('/api/generate_ai_report', methods=['POST'])
@login_required
def generate_ai_report_api():
    logger_instance = app.logger if app else logging
    user_id = get_current_user_id()
    if not report_generation_model: return jsonify({'success': False, 'message': 'AI model not available.'}), 503

    data = request.get_json()
    report_type = data.get('reportType')

    user_data_db = get_user_data(user_id)
    if not user_data_db: return jsonify({'success': False, 'message': 'Could not retrieve user data.'}), 500

    market_type = user_data_db['market_type']
    logger_instance.info(f"AI report request for user {user_id}, market {market_type}, type {report_type}")

    if report_type not in ['investment', 'behavioral']: return jsonify({'success': False, 'message': 'Invalid report type.'}), 400

    conn = get_db_connection()
    if not conn: return jsonify({'success': False, 'message': 'Database connection error.'}), 500

    cash_balance_for_report = user_data_db[f'cash_balance_{market_type.lower()}']
    initial_capital_for_report = user_data_db[f'initial_capital_{market_type.lower()}']
    holdings_summary, trade_history_summary = "None", "No trades yet."
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
                    h_list.append(f"{h_db['shares']} {h_db['stock_name'] or h_db['ticker']} (avg cost ${float(h_db['average_cost']):.2f}, val ${h_db['shares'] * c_price:.2f})")
                if h_list: holdings_summary = "; ".join(h_list)

            limit = 10 if report_type == 'behavioral' else 5
            cursor.execute("SELECT trade_type, ticker, stock_name, shares, price_per_share, mood, reason, timestamp FROM trade_history WHERE user_id = %s AND market_type = %s ORDER BY timestamp DESC LIMIT %s", (user_id, market_type, limit))
            trades_db = cursor.fetchall()
            if trades_db:
                t_list = []
                for t_db in trades_db:
                    r_str = t_db['reason']; r_str = (r_str[:47] + "...") if r_str and len(r_str) > 50 else r_str
                    t_str = f"{t_db['trade_type'].upper()} {t_db['shares']} {t_db['stock_name'] or t_db['ticker']} @ ${float(t_db['price_per_share']):.2f} on {t_db['timestamp'].strftime('%y-%m-%d')}"
                    if t_db['mood']: t_str += f" (Mood: {t_db['mood']})"
                    if r_str: t_str += f" (Reason: {r_str})"
                    t_list.append(t_str)
                if t_list: trade_history_summary = "; ".join(t_list)

    except pymysql.Error as e:
        logger_instance.error(f"DB error for AI report (user {user_id}/{market_type}): {e}")
        return jsonify({'success': False, 'message': 'Error fetching data for report.'}), 500
    finally:
        if conn: conn.close()

    prompt = f"You are FinSimU AI, an educational assistant for student '{user_data_db['nickname']}' (Style: {user_data_db['investment_style']}) in a stock sim for the {market_type} market.\n"
    prompt += f"Initial Capital ({market_type}): ${initial_capital_for_report:.2f}. Current Cash ({market_type}): ${cash_balance_for_report:.2f}. Total Portfolio Value ({market_type}): ${portfolio_total_value:.2f}.\n"
    prompt += f"Holdings ({market_type}): {holdings_summary}\nRecent Trades ({market_type}): {trade_history_summary}\n\n"
    if report_type == 'investment':
        prompt += "Generate a concise Investment Report: 1. Overall Performance Snapshot (vs initial capital). 2. Portfolio Composition Insight (if holdings exist). 3. Trading Pattern Observation (if trades exist). 4. One Reflective Question or Gentle Suggestion aligned with their style. Tone: encouraging, educational. Focus: reflection, learning. Format: clear, use bullet points for 2,3,4."
    elif report_type == 'behavioral':
        prompt += "Generate a concise Behavioral Analysis: 1. Dominant Moods/Reasons in trades (if any). 2. ONE Potential Behavioral Bias (e.g., FOMO, loss aversion, overconfidence) with simple explanation. 3. One Reflective Question for self-awareness on psychology. Tone: empathetic, supportive. Goal: foster self-awareness. Format: clear."
    prompt += "\nIMPORTANT: Output well-formatted text for display. Use Markdown for structure (bullets), but NO triple backticks for code blocks."

    logger_instance.info(f"Generating AI report for user {user_id}, market {market_type}, type: {report_type}")
    try:
        response = report_generation_model.generate_content(prompt, safety_settings=safety_settings_gemini)

        analysis_text = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text'): analysis_text += part.text

        if hasattr(response,'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            logger_instance.warning(f"Gemini content blocked for {user_id}. Reason:{response.prompt_feedback.block_reason}")
            return jsonify(success=False,message=f'AI content blocked: {response.prompt_feedback.block_reason_message or "Safety"}'),400

        if not analysis_text:
            logger_instance.warning(f"Gemini response for {user_id} no usable text. Resp:{response}")
            return jsonify(success=False,message='AI service empty response.'),500

        return jsonify({'success': True, 'report_type': report_type, 'report_content': analysis_text.strip(), 'marketType': market_type})
    except Exception as e:
        logger_instance.error(f"Error with Gemini for user {user_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'message': 'Error from AI service.'}), 500

# --- AI 即時個股分析路由 (使用 StockAnalyzer) ---
@app.route('/api/analyze_stock_on_demand', methods=['POST'])
@login_required
def api_analyze_stock_on_demand():
    logger_instance = app.logger if app else logging
    user_id = get_current_user_id() # 雖然此功能不綁定用戶市場，但仍需登入
    data = request.get_json()
    ticker_input = data.get('ticker','').strip().upper()

    if not ticker_input:
        return jsonify(success=False, message='請提供股票代碼 (Ticker cannot be empty)'), 400

    logger_instance.info(f"On-demand stock analysis request for: {ticker_input} (User: {user_id})")

    if not stock_analysis_model_gemini: # 檢查 Gemini 模型是否可用
        logger_instance.error("Gemini model (stock_analysis_model_gemini) is not available for on-demand analysis.")
        return jsonify(success=False, message='AI分析服務暫時無法使用 (AI model not loaded)'), 503

    try:
        # StockAnalyzer 會自動判斷市場或根據 ticker 後綴修正
        analyzer = StockAnalyzer(ticker=ticker_input, period="3y", temperature=0.5) # period 和 temperature 可以調整

        summary_data = analyzer.get_stock_summary()

        if summary_data.get('error'):
            logger_instance.warning(f"On-demand analysis for {ticker_input} failed with error: {summary_data['error']}")
            if "無法取得" in summary_data['error'] and ("歷史資料" in summary_data['error'] or "基本資訊" in summary_data['error']):
                 # yfinance 有時對不存在或下市的股票會快速失敗
                return jsonify(success=False, message=f"無法獲取股票 {analyzer.ticker} 的足夠歷史數據或基本資訊進行分析。請確認股票代碼是否正確或稍後再試。"), 404
            return jsonify(success=False, message=f"分析股票 {ticker_input} 時發生錯誤: {summary_data['error']}"), 500

        # 如果圖表生成成功，轉換為可訪問的 URL
        if summary_data.get("chart_path"):
            # url_for('static', filename='charts/THE_FILENAME.html')
            summary_data["chart_url"] = url_for('static', filename=summary_data["chart_path"])
        else:
            summary_data["chart_url"] = None # 確保即使沒有圖表也有這個鍵

        logger_instance.info(f"Successfully generated on-demand analysis for {ticker_input}")
        return jsonify(success=True, data=summary_data)

    except ValueError as ve: # StockAnalyzer 初始化時可能拋出 ValueError
        logger_instance.error(f"ValueError during on-demand analysis for {ticker_input}: {ve}", exc_info=True)
        return jsonify(success=False, message=f"股票代碼 '{ticker_input}' 無效或無法獲取數據: {str(ve)}"), 400
    except Exception as e:
        logger_instance.error(f"Unexpected error during on-demand analysis for {ticker_input}: {e}", exc_info=True)
        return jsonify(success=False, message="執行股票分析時發生未預期的內部錯誤，請稍後再試。"), 500


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
    return render_template('login.html')

@app.route('/register')
def finsimu_register_route():
    if 'user_id' in session and session.get('user_id') is not None:
        return redirect(url_for('finsimu_app_route', _anchor='dashboard'))
    return render_template('register.html')


@app.route('/app')
@login_required
def finsimu_app_route():
    logger_instance = app.logger if app else logging
    logger_instance.info(f"User {session.get('user_id')} accessing /app. Session: {dict(session)}")
    return render_template('app_main.html')

@app.route('/logout_api', methods=['POST'])
@login_required
def logout_api():
    logger_instance = app.logger if app else logging
    user_id_before = session.get('user_id', 'N/A')
    logger_instance.info(f"User {user_id_before} attempting logout. Session before: {dict(session)}")
    session.clear()
    logger_instance.info(f"Session after clear (formerly user {user_id_before}): {dict(session)}")
    return jsonify({'success': True, 'message': 'Logged out successfully.'})

# --- Main Execution ---
if __name__ == '__main__':
    # 確保在 app context 之外也能用 logging
    if not app.logger.handlers: # 如果 Flask logger 尚未配置 handler
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
        (app.logger if app else logging).info(f"Created directory (from main): {charts_dir}")

    init_finsimu_database() # 使用單一市場模型的資料庫初始化
    (app.logger if app else logging).info("Starting FinSimU Flask server (Single Market Model with StockAnalyzer)...")
    app.run(debug=True, host='0.0.0.0', port=5001)