# --- START OF FILE v4.py (Modified v5 - Full Code & Syntax Fixes) ---

import os
import re
import secrets
import logging
import warnings
import datetime as dt
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import feedparser
import urllib.parse
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime, timedelta, timezone # Import timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mysql.connector
import bcrypt
import json
import plotly.express as px
# import chart_studio.plotly as py # 可能不再需要
# import chart_studio.tools as tls # 可能不再需要
import time # For retry delay

# 載入環境變數
load_dotenv()

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# 建立需要的資料夾
for path in ['static/charts', 'static/data']:
    os.makedirs(path, exist_ok=True)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# --- Gemini API 設定 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
try:
    genai.configure(api_key=GEMINI_API_KEY)
    general_model = genai.GenerativeModel("models/gemini-1.5-flash-latest", safety_settings=safety_settings)
    portfolio_model = genai.GenerativeModel("models/gemini-1.5-flash-latest", safety_settings=safety_settings)
    stock_analysis_model = genai.GenerativeModel("models/gemini-1.5-flash-latest", safety_settings=safety_settings)
    logger.info("Gemini Models configured successfully.")
except Exception as gemini_err:
    logger.error(f"Failed to configure Gemini models: {gemini_err}", exc_info=True)
    general_model = None
    portfolio_model = None
    stock_analysis_model = None


# --- Database Connection ---
def get_db_connection():
    try:
        # 增加連接超時和讀寫超時設置
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "0912559910"),
            database=os.getenv("DB_NAME", "testdb"),
            pool_name = "stock_pool",
            pool_size = 5,
            connect_timeout=10, # 連接超時10秒
        )
        # 設置讀寫超時 (需要連接後設置)
        # conn.config(option_files=None, option_groups=None, use_pure=False, **{'option_strings': 'MYSQL_OPT_READ_TIMEOUT=30;MYSQL_OPT_WRITE_TIMEOUT=30'})
        if conn.is_connected():
            return conn
        else:
            logger.error("Failed to establish database connection.")
            return None
    except mysql.connector.Error as err:
        logger.error(f"Database connection error: {err}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during database connection: {e}")
        return None

# --- Database Initialization (Added portfolio_name to watchlist) ---
def init_database():
    conn = get_db_connection()
    if conn:
        cursor = None
        try:
            cursor = conn.cursor()
            db_name = os.getenv("DB_NAME", "testdb")

            # Users Table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY, username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL, email VARCHAR(100) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            ''')
            # Watchlist Table (Added portfolio_name)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                ticker VARCHAR(20) NOT NULL,
                name VARCHAR(100) NOT NULL,
                portfolio_name VARCHAR(100) DEFAULT NULL, -- NEW: Source portfolio name
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, ticker),
                INDEX (user_id, portfolio_name) -- Add index for grouping
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            ''')
            # Check and Add portfolio_name column if missing (for existing tables)
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'watchlist' AND COLUMN_NAME = 'portfolio_name'
            """, (db_name,))
            result_wl_pn = cursor.fetchone()
            if result_wl_pn and result_wl_pn[0] == 0:
                try:
                    cursor.execute("ALTER TABLE watchlist ADD COLUMN portfolio_name VARCHAR(100) DEFAULT NULL AFTER name, ADD INDEX (user_id, portfolio_name)")
                    logger.info("Added 'portfolio_name' column and index to watchlist table.")
                    conn.commit()
                except mysql.connector.Error as alter_err_wl:
                    logger.error(f"Failed to add 'portfolio_name' column to watchlist table: {alter_err_wl}")

            # Settings Table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                id INT AUTO_INCREMENT PRIMARY KEY, user_id INT, dark_mode BOOLEAN DEFAULT TRUE,
                font_size VARCHAR(10) DEFAULT 'medium', price_alert BOOLEAN DEFAULT FALSE,
                market_summary BOOLEAN DEFAULT TRUE, data_source VARCHAR(20) DEFAULT 'default',
                temperature FLOAT DEFAULT 0.7, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            ''')
            # Check and Add temperature column
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'settings' AND COLUMN_NAME = 'temperature'
            """, (db_name,))
            result = cursor.fetchone()
            if result and result[0] == 0:
                try:
                    cursor.execute("ALTER TABLE settings ADD COLUMN temperature FLOAT DEFAULT 0.7")
                    logger.info("Added 'temperature' column to settings table.")
                    conn.commit() # Commit after alter
                except mysql.connector.Error as alter_err: logger.error(f"Failed to add 'temperature' column: {alter_err}") # Don't raise

            # Stocks Table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                id INT AUTO_INCREMENT PRIMARY KEY, symbol VARCHAR(20) UNIQUE NOT NULL, name VARCHAR(100) NOT NULL,
                market VARCHAR(10) NOT NULL,
                ma5 FLOAT DEFAULT NULL, ma20 FLOAT DEFAULT NULL, ma50 FLOAT DEFAULT NULL, ma120 FLOAT DEFAULT NULL, ma200 FLOAT DEFAULT NULL,
                bb_upper FLOAT DEFAULT NULL, bb_middle FLOAT DEFAULT NULL, bb_lower FLOAT DEFAULT NULL,
                rsi FLOAT DEFAULT NULL, wmsr FLOAT DEFAULT NULL, psy FLOAT DEFAULT NULL, bias6 FLOAT DEFAULT NULL,
                macd FLOAT DEFAULT NULL, macd_signal FLOAT DEFAULT NULL, macd_hist FLOAT DEFAULT NULL,
                k FLOAT DEFAULT NULL, d FLOAT DEFAULT NULL, j FLOAT DEFAULT NULL, pe_ratio FLOAT DEFAULT NULL,
                market_cap BIGINT DEFAULT NULL, open_price FLOAT DEFAULT NULL, close_price FLOAT DEFAULT NULL,
                high_price FLOAT DEFAULT NULL, low_price FLOAT DEFAULT NULL, volume BIGINT DEFAULT NULL, INDEX(symbol)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            ''')
            # Check and ADD last_updated column if missing (safer approach)
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'stocks' AND COLUMN_NAME = 'last_updated'
            """, (db_name,))
            result_stocks_lu = cursor.fetchone()
            if result_stocks_lu and result_stocks_lu[0] == 0:
                try:
                    cursor.execute("ALTER TABLE stocks ADD COLUMN last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")
                    logger.info("Added 'last_updated' column to stocks table.")
                    conn.commit() # Commit after alter
                except mysql.connector.Error as alter_err_stocks:
                    logger.error(f"Failed to add 'last_updated' column to stocks table: {alter_err_stocks}")

            # stock_strategy Table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_strategy (
                ticker VARCHAR(20) PRIMARY KEY,
                gene JSON,
                fitness FLOAT,
                last_backtest_date DATE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            ''')
            # stock_signal Table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_signal (
                ticker VARCHAR(20) PRIMARY KEY,
                signal_date DATE,
                status ENUM('buy','sell','hold') NOT NULL,
                current_price FLOAT,
                INDEX (signal_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            ''')

            conn.commit() # Commit all changes at the end
            logger.info("資料庫初始化成功 (包含 strategy/signal 表, watchlist.portfolio_name)")
        except mysql.connector.Error as err:
            logger.error(f"資料庫初始化錯誤: {err}")
            if conn: conn.rollback()
        except Exception as e:
             logger.error(f"資料庫初始化期間發生未知錯誤: {e}")
             if conn: conn.rollback()
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()

# --- Helper function to get user settings ---
def get_user_settings(user_id=1):
    default_settings = {
        'dark_mode': True, 'font_size': 'medium', 'price_alert': False,
        'market_summary': True, 'data_source': 'default', 'temperature': 0.7
    }
    conn = get_db_connection()
    if not conn: return default_settings
    settings = None
    cursor = None
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM settings WHERE user_id = %s', (user_id,))
        settings = cursor.fetchone()
    except mysql.connector.Error as err:
        logger.error(f"獲取用戶 {user_id} 設定時資料庫錯誤: {err}")
    except Exception as e:
        logger.error(f"獲取用戶 {user_id} 設定時發生未知錯誤: {e}")
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

    if settings:
        final_settings = default_settings.copy()
        final_settings.update(settings)
        try:
            temp_val = final_settings.get('temperature')
            if temp_val is None:
                final_settings['temperature'] = default_settings['temperature']
            else:
                final_settings['temperature'] = max(0.0, min(1.0, float(temp_val)))
        except (ValueError, TypeError):
            logger.warning(f"Invalid temperature value '{final_settings.get('temperature')}' for user {user_id}, using default.")
            final_settings['temperature'] = default_settings['temperature']
        return final_settings
    else:
        # Insert default settings if not found
        conn_insert = get_db_connection()
        if conn_insert:
            cursor_insert = None
            try:
                cursor_insert = conn_insert.cursor()
                cursor_insert.execute('''
                    INSERT INTO settings (user_id, dark_mode, font_size, price_alert, market_summary, data_source, temperature)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (user_id, default_settings['dark_mode'], default_settings['font_size'], default_settings['price_alert'],
                      default_settings['market_summary'], default_settings['data_source'], default_settings['temperature']))
                conn_insert.commit()
                logger.info(f"為用戶 {user_id} 插入了預設設定。")
            except mysql.connector.Error as insert_err:
                if insert_err.errno == 1062: # Duplicate entry
                    logger.warning(f"嘗試為用戶 {user_id} 插入預設設定時發現重複條目。")
                else:
                    logger.error(f"為用戶 {user_id} 插入預設設定時出錯: {insert_err}")
                    conn_insert.rollback()
            except Exception as insert_e:
                logger.error(f"為用戶 {user_id} 插入預設設定時發生未知錯誤: {insert_e}")
                conn_insert.rollback()
            finally:
                 if cursor_insert: cursor_insert.close()
                 if conn_insert and conn_insert.is_connected(): conn_insert.close()
        return default_settings

# ------------------ 股票分析功能 ------------------
class StockAnalyzer:
    def __init__(self, ticker: str, api_key: str, period: str = "10y", market: str = "TW", temperature: float = 0.7):
        self.ticker = ticker.strip()
        # --- .TW Auto Append Logic ---
        if market == "TW" and not ticker.endswith(".TW") and re.fullmatch(r'\d{4,6}', ticker):
            logger.info(f"Appending .TW to potential Taiwan stock code: {ticker}")
            self.ticker = f"{ticker}.TW"
        elif market == "TW" and "." not in ticker: # Handle cases like '0050' if needed, though previous line covers most
             self.ticker = f"{ticker}.TW"
        else:
             self.ticker = ticker # Use original if already has .TW or not TW market
        # --- End .TW Logic ---
        self.period = period
        self.market = market
        self.temperature = max(0.0, min(1.0, temperature))
        self.stock = yf.Ticker(self.ticker)
        self.data = None
        self.company_name = None
        self.currency = None
        self.pe_ratio = None
        self.market_cap = None
        self.eps = None
        self.roe = None
        self.net_profit_margin_str = "N/A"
        self.current_ratio_str = "N/A"
        self.model = stock_analysis_model
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')
        try:
            self._get_data()
            self._get_financial_data()
            self._calculate_indicators()
            self._update_db_data()
            logger.debug(f"[{self.ticker}] Initialized Financials - ROE: {self.roe}, Net Margin: {self.net_profit_margin_str}, Current Ratio: {self.current_ratio_str}")
        except Exception as e:
             logger.error(f"StockAnalyzer 初始化失敗 ({self.ticker}): {e}", exc_info=True)

    def _get_data(self):
        try:
            self.data = self.stock.history(period=self.period, timeout=15)
            if self.data.empty:
                raise ValueError(f"無法取得 {self.ticker} 的歷史資料 (資料為空)")
            info = self.stock.info
            if not info:
                # Try fetching basic info again if full info fails
                try:
                    basic_info = self.stock.fast_info
                    self.company_name = basic_info.get('longName', basic_info.get('shortName', self.ticker))
                    self.currency = basic_info.get('currency', 'TWD' if self.market == 'TW' else 'USD')
                    logger.warning(f"無法取得 {self.ticker} 的完整公司資訊，使用 fast_info。")
                except Exception:
                     raise ValueError(f"無法取得 {self.ticker} 的公司資訊 (info 和 fast_info 均為空或錯誤)")
            else:
                self.company_name = info.get('longName', info.get('shortName', self.ticker))
                self.currency = info.get('currency', 'TWD' if self.market == 'TW' else 'USD')
            logger.info("成功取得 %s 的股票資料和資訊", self.ticker)
        except Exception as e:
            logger.error("取得股票資料時發生錯誤 (%s): %s", self.ticker, e)
            self.data = pd.DataFrame() # Ensure data is an empty DataFrame on error

    def _get_financial_data(self, retries=2, delay=1):
        financials = pd.DataFrame()
        balance_sheet = pd.DataFrame()
        info = {}
        try:
            logger.info(f"開始獲取 {self.ticker} 的財務數據...")
            info = self.stock.info or {} # Ensure info is a dict

            # --- 1. Get data directly from info if available ---
            self.pe_ratio = info.get('trailingPE')
            self.market_cap = info.get('marketCap')
            self.eps = info.get('trailingEps')
            self.roe = info.get('returnOnEquity') # Get raw ROE from info
            self.current_ratio_str = "N/A" # Initialize
            if info.get('currentRatio'):
                try:
                    ratio_val = float(info['currentRatio'])
                    # Use ternary operator correctly
                    self.current_ratio_str = f"{ratio_val:.2f}" if pd.notna(ratio_val) else "N/A"
                    if pd.notna(ratio_val):
                        logger.info(f"從 info 取得 {self.ticker} 的流動比率: {self.current_ratio_str}")
                except (ValueError, TypeError):
                     logger.warning(f"無法將 info 中的 currentRatio '{info['currentRatio']}' 轉換為浮點數。")

            logger.debug(f"[{self.ticker}] Info fetched. ROE: {self.roe}, Current Ratio: {self.current_ratio_str}")

            # --- 2. Fetch statements (with retry) ---
            logger.debug(f"[{self.ticker}] 嘗試獲取財務報表...")
            for attempt in range(retries):
                try:
                    financials = self.stock.financials
                    # Use ternary operator correctly
                    logger.info(f"[{self.ticker}] 成功獲取 financials (嘗試 {attempt+1})") if not financials.empty else logger.warning(f"[{self.ticker}] financials 為空 (嘗試 {attempt+1})")
                    if not financials.empty:
                        break
                except Exception as fin_err:
                    logger.warning(f"獲取 {self.ticker} 的 financials 時出錯 (嘗試 {attempt+1}): {fin_err}")
                # Use ternary operator correctly
                time.sleep(delay) if attempt < retries - 1 else logger.error(f"[{self.ticker}] 多次嘗試後仍無法獲取 financials。")

            for attempt in range(retries):
                try:
                    balance_sheet = self.stock.balance_sheet
                    # Use ternary operator correctly
                    logger.info(f"[{self.ticker}] 成功獲取 balance_sheet (嘗試 {attempt+1})") if not balance_sheet.empty else logger.warning(f"[{self.ticker}] balance_sheet 為空 (嘗試 {attempt+1})")
                    if not balance_sheet.empty:
                        break
                except Exception as bs_err:
                    logger.warning(f"獲取 {self.ticker} 的 balance_sheet 時出錯 (嘗試 {attempt+1}): {bs_err}")
                # Use ternary operator correctly
                time.sleep(delay) if attempt < retries - 1 else logger.error(f"[{self.ticker}] 多次嘗試後仍無法獲取 balance_sheet。")
            # --- End Fetch ---


            # --- 3. Calculate ROE ---
            if self.roe is None:
                if not financials.empty and not balance_sheet.empty:
                    logger.debug(f"[{self.ticker}] 嘗試從報表計算 ROE...")
                    net_income_keys = ["Net Income", "Net Income Common Stockholders"]
                    equity_keys = ["Total Stockholder Equity", "Stockholders Equity"]
                    net_income_key = next((key for key in net_income_keys if key in financials.index), None)
                    equity_key = next((key for key in equity_keys if key in balance_sheet.index), None)
                    if net_income_key and equity_key:
                        try:
                            if len(financials.columns) > 0 and len(balance_sheet.columns) > 0:
                                net_income = financials.loc[net_income_key].iloc[0]
                                equity = balance_sheet.loc[equity_key].iloc[0]
                                # Use ternary operator correctly
                                self.roe = net_income / equity if pd.notna(net_income) and pd.notna(equity) and equity != 0 else None
                                if self.roe is not None:
                                     logger.info(f"從報表計算得到 {self.ticker} 的 ROE: {self.roe:.4f}")

                            if self.roe is None and len(financials.columns) > 1 and len(balance_sheet.columns) > 1:
                                net_income_prev = financials.loc[net_income_key].iloc[1]
                                equity_prev = balance_sheet.loc[equity_key].iloc[1]
                                # Use ternary operator correctly
                                self.roe = net_income_prev / equity_prev if pd.notna(net_income_prev) and pd.notna(equity_prev) and equity_prev != 0 else None
                                if self.roe is not None:
                                    logger.info(f"從報表(前一年)計算得到 {self.ticker} 的 ROE: {self.roe:.4f}")

                            if self.roe is None:
                                logger.warning(f"無法從報表計算 {self.ticker} 的 ROE (數據不足或無效)")
                        except Exception as roe_calc_e:
                            logger.warning(f"從報表計算 ROE 時出錯 ({self.ticker}): {roe_calc_e}")
                    else:
                        logger.warning(f"無法計算 ROE，缺少必要的鍵名 (NetIncome: {net_income_key is None}, Equity: {equity_key is None}) ({self.ticker})")
                else:
                    logger.warning(f"無法計算 ROE，因為 financials 或 balance_sheet 為空 ({self.ticker})")


            # --- 4. Calculate Net Profit Margin ---
            self.net_profit_margin_str = "N/A"
            if not financials.empty:
                logger.debug(f"[{self.ticker}] 嘗試計算淨利率...")
                revenue_keys = ["Total Revenue", "Operating Revenue"]
                net_income_keys = ["Net Income", "Net Income Common Stockholders"]
                revenue_key = next((key for key in revenue_keys if key in financials.index), None)
                net_income_key = next((key for key in net_income_keys if key in financials.index), None)
                if revenue_key and net_income_key:
                    if len(financials.columns) > 0:
                        revenue = financials.loc[revenue_key].iloc[0]
                        net_income = financials.loc[net_income_key].iloc[0]
                        # Use ternary operator correctly
                        self.net_profit_margin_str = f"{(net_income / revenue) * 100:.2f}%" if pd.notna(revenue) and pd.notna(net_income) and revenue != 0 else "N/A"
                        if self.net_profit_margin_str != "N/A":
                             logger.info(f"計算得到 {self.ticker} 的淨利率: {self.net_profit_margin_str}")
                        else:
                             logger.warning(f"無法計算淨利率，Revenue ({revenue}) 或 Net Income ({net_income}) 數值無效或 Revenue 為零 ({self.ticker})")
                    else:
                        logger.warning(f"無法計算淨利率，financials 沒有數據列 ({self.ticker})")
                else:
                    logger.warning(f"無法計算淨利率，缺少必要的鍵名 (Revenue Key Found: {revenue_key is not None}, NetIncome Key Found: {net_income_key is not None}) ({self.ticker})")
            else:
                logger.warning(f"無法計算淨利率，因為 financials 數據未能獲取 ({self.ticker})")


            # --- 5. Calculate Current Ratio ---
            if self.current_ratio_str == "N/A" and not balance_sheet.empty:
                logger.debug(f"[{self.ticker}] 嘗試從報表計算流動比率...")
                assets_keys = ["Current Assets", "Total Current Assets"]
                liabilities_keys = ["Current Liabilities", "Total Current Liabilities"]
                assets_key = next((key for key in assets_keys if key in balance_sheet.index), None)
                liabilities_key = next((key for key in liabilities_keys if key in balance_sheet.index), None)
                logger.debug(f"[{self.ticker}] Balance Sheet Index for Current Ratio: {balance_sheet.index.tolist()}")
                if assets_key and liabilities_key:
                    if len(balance_sheet.columns) > 0:
                        latest_col = balance_sheet.columns[0]
                        current_assets = balance_sheet.loc[assets_key, latest_col]
                        current_liabilities = balance_sheet.loc[liabilities_key, latest_col]
                        logger.info(f"[{self.ticker}] Attempting Current Ratio Calc - Assets ({assets_key}): {current_assets}, Liabilities ({liabilities_key}): {current_liabilities}")
                        # Use ternary operator correctly
                        self.current_ratio_str = f"{current_assets / current_liabilities:.2f}" if pd.notna(current_assets) and pd.notna(current_liabilities) and current_liabilities != 0 else "N/A"
                        if self.current_ratio_str != "N/A":
                            logger.info(f"從報表計算得到 {self.ticker} 的流動比率: {self.current_ratio_str}")
                        else:
                            logger.warning(f"無法計算流動比率，資產 ({current_assets}) / 負債 ({current_liabilities}) 數值無效或負債為零 ({self.ticker})")
                    else:
                        logger.warning(f"無法計算流動比率，balance_sheet 沒有數據列 ({self.ticker})")
                else:
                    logger.warning(f"無法計算流動比率，缺少必要的鍵名 (Assets Key Found: {assets_key is not None}, Liabilities Key Found: {liabilities_key is not None}) ({self.ticker})")
            elif self.current_ratio_str == "N/A":
                 logger.warning(f"最終未能計算流動比率，可能 info 和 balance_sheet 均缺少有效數據 ({self.ticker})")


            logger.info(f"完成處理 {self.ticker} 的財務資料")

        except Exception as e:
            logger.error(f"處理 {self.ticker} 財務資料時發生未預期錯誤: {e}", exc_info=True)
            # Ensure attributes exist even on major error
            self.pe_ratio = getattr(self, 'pe_ratio', None)
            self.market_cap = getattr(self, 'market_cap', None)
            self.eps = getattr(self, 'eps', None)
            self.roe = getattr(self, 'roe', None)
            self.net_profit_margin_str = getattr(self, 'net_profit_margin_str', "N/A")
            self.current_ratio_str = getattr(self, 'current_ratio_str', "N/A")

    def _calculate_indicators(self):
        try:
            if self.data is None or self.data.empty:
                logger.warning(f"[{self.ticker}] No data available for indicator calculation.")
                return
            df = self.data.copy()
            # Define lengths needed for each indicator
            lengths = {
                'MA5': 5, 'MA20': 20, 'MA50': 50, 'MA120': 120, 'MA200': 200,
                'RSI': 13, # RSI period 12 needs 13 data points
                'MACD': 35, # MACD(12,26,9) needs roughly 26+9 = 35 points for signal line
                'KDJ': 14, # STOCH(9,3,3) needs 9+3-1 = 11, plus smoothing needs more, 14 is safe
                'BBands': 20,
                'WMSR': 14,
                'OBV': 2, # Needs previous close
                'ADX': 28, # ADX(14) needs 14*2 = 28 points
                'VolChg': 2,
                'Momentum': 11, # Momentum(10) needs 11 points
                'Vol': 20,
                'PSY': 13, # PSY(12) needs 13 points
                'BIAS6': 7 # BIAS(6) needs 7 points
            }
            df_len = len(df)

            # Calculate indicators only if enough data exists
            if df_len >= lengths['MA5']: df['MA5'] = ta.sma(df['Close'], length=5)
            if df_len >= lengths['MA20']: df['MA20'] = ta.sma(df['Close'], length=20)
            if df_len >= lengths['MA50']: df['MA50'] = ta.sma(df['Close'], length=50)
            if df_len >= lengths['MA120']: df['MA120'] = ta.sma(df['Close'], length=120)
            if df_len >= lengths['MA200']: df['MA200'] = ta.sma(df['Close'], length=200)
            if df_len >= lengths['RSI']: df['RSI'] = ta.rsi(df['Close'], length=12)

            if df_len >= lengths['MACD']:
                macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
                # Use ternary operator correctly
                macd_df.rename(columns={'MACD_12_26_9': 'MACD', 'MACDs_12_26_9': 'MACD_signal', 'MACDh_12_26_9': 'MACD_hist'}, inplace=True) if macd_df is not None else None
                # Use ternary operator correctly
                df = pd.concat([df, macd_df[['MACD', 'MACD_signal', 'MACD_hist']]], axis=1) if macd_df is not None else df

            if df_len >= lengths['KDJ']:
                stoch_df = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3, smooth_k=3)
                # Use ternary operator correctly
                df['K'] = stoch_df['STOCHk_9_3_3'] if stoch_df is not None else np.nan
                # Use ternary operator correctly
                df['D'] = stoch_df['STOCHd_9_3_3'] if stoch_df is not None else np.nan
                # Use ternary operator correctly
                df['J'] = 3 * df['K'] - 2 * df['D'] if 'K' in df.columns and 'D' in df.columns and df['K'].notna().any() and df['D'].notna().any() else np.nan

            if df_len >= lengths['BBands']:
                bbands = ta.bbands(df['Close'], length=20, std=2)
                # Use ternary operator correctly
                df['BB_lower'] = bbands['BBL_20_2.0'] if bbands is not None else np.nan
                # Use ternary operator correctly
                df['BB_middle'] = bbands['BBM_20_2.0'] if bbands is not None else np.nan
                # Use ternary operator correctly
                df['BB_upper'] = bbands['BBU_20_2.0'] if bbands is not None else np.nan
                # Use ternary operator correctly
                df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'].replace(0, np.nan) if all(c in df.columns for c in ['BB_upper', 'BB_lower', 'BB_middle']) and df['BB_middle'].notna().any() else np.nan

            if df_len >= lengths['WMSR']: df['WMSR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
            if df_len >= lengths['OBV']: df['OBV'] = ta.obv(df['Close'], df['Volume'])

            if df_len >= lengths['ADX']:
                 adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
                 # Use ternary operator correctly
                 df['ADX'] = adx_df['ADX_14'] if adx_df is not None else np.nan

            if df_len >= lengths['VolChg']: df['Volume_Change'] = df['Volume'].pct_change() * 100
            if df_len >= lengths['Momentum']: df['Momentum'] = df['Close'] - df['Close'].shift(10)
            if df_len >= lengths['Vol']: df['Volatility'] = df['Close'].rolling(window=20).std()
            if df_len >= lengths['PSY']:
                 df['PSY'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else 0).rolling(12).sum() / 12 * 100
            if df_len >= lengths['BIAS6']:
                 bias_mean = df['Close'].rolling(window=6).mean()
                 # Handle potential division by zero or NaN in bias_mean
                 df['BIAS6'] = ((df['Close'] - bias_mean) / bias_mean.replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan)

            self.data = df
            logger.info("成功計算 %s 的技術指標", self.ticker)
        except Exception as e:
            logger.error("計算技術指標時發生錯誤 (%s): %s", self.ticker, e)

    def _update_db_data(self):
        conn = None
        cursor = None
        try:
            if self.data is None or self.data.empty:
                return
            conn = get_db_connection()
            if not conn:
                return

            cursor = conn.cursor()
            cursor.execute("SELECT id FROM stocks WHERE symbol = %s", (self.ticker,))
            result = cursor.fetchone()
            latest_data = self.data.iloc[-1]

            def get_float(series, key):
                val = series.get(key)
                # Use ternary operator correctly
                return float(val) if pd.notna(val) else None

            def get_int(series, key):
                val = series.get(key)
                # Use ternary operator correctly
                return int(float(val)) if pd.notna(val) else None

            stock_data = {
                'symbol': self.ticker,
                'name': self.company_name or self.ticker,
                'market': self.market,
                'ma5': get_float(latest_data, 'MA5'), 'ma20': get_float(latest_data, 'MA20'),
                'ma50': get_float(latest_data, 'MA50'), 'ma120': get_float(latest_data, 'MA120'),
                'ma200': get_float(latest_data, 'MA200'), 'bb_upper': get_float(latest_data, 'BB_upper'),
                'bb_middle': get_float(latest_data, 'BB_middle'), 'bb_lower': get_float(latest_data, 'BB_lower'),
                'rsi': get_float(latest_data, 'RSI'), 'wmsr': get_float(latest_data, 'WMSR'),
                'psy': get_float(latest_data, 'PSY'), 'bias6': get_float(latest_data, 'BIAS6'),
                'macd': get_float(latest_data, 'MACD'), 'macd_signal': get_float(latest_data, 'MACD_signal'),
                'macd_hist': get_float(latest_data, 'MACD_hist'), 'k': get_float(latest_data, 'K'),
                'd': get_float(latest_data, 'D'), 'j': get_float(latest_data, 'J'),
                # Use ternary operator correctly
                'pe_ratio': float(self.pe_ratio) if isinstance(self.pe_ratio, (int, float)) else None,
                # Use ternary operator correctly
                'market_cap': int(self.market_cap) if isinstance(self.market_cap, (int, float)) else None,
                'open_price': get_float(latest_data, 'Open'), 'close_price': get_float(latest_data, 'Close'),
                'high_price': get_float(latest_data, 'High'), 'low_price': get_float(latest_data, 'Low'),
                'volume': get_int(latest_data, 'Volume')
            }

            if result:
                # Update existing record, including last_updated
                set_clauses = ', '.join([f"{k} = %({k})s" for k in stock_data if k != 'symbol'])
                update_query = f"UPDATE stocks SET {set_clauses}, last_updated = CURRENT_TIMESTAMP WHERE symbol = %(symbol)s"
                cursor.execute(update_query, stock_data)
            else:
                # Insert new record, including last_updated
                cols_to_insert_dict = {k: v for k, v in stock_data.items() if k != 'id'}
                cols = ', '.join(cols_to_insert_dict.keys())
                placeholders = ', '.join([f'%({k})s' for k in cols_to_insert_dict.keys()])
                insert_query = f"INSERT INTO stocks ({cols}, last_updated) VALUES ({placeholders}, CURRENT_TIMESTAMP)"
                cursor.execute(insert_query, cols_to_insert_dict)

            conn.commit()
        except mysql.connector.Error as db_err:
             logger.error(f"更新資料庫數據時發生 DB 錯誤 ({self.ticker}): {db_err}")
             # Use ternary operator correctly
             conn.rollback() if conn else None
        except Exception as e:
            logger.error(f"更新資料庫數據時發生一般錯誤 ({self.ticker}): {e}")
            # Use ternary operator correctly
            conn.rollback() if conn else None
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()

    def _identify_patterns(self, days=30):
        try:
            if self.data is None or self.data.empty or len(self.data) < 2:
                return ["數據不足"]
            df = self.data.tail(days).copy()
            patterns = []
            required_cols = ['MA5', 'MA20', 'Close', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal', 'K', 'D', 'RSI']
            valid_cols = {col for col in required_cols if col in df.columns and df[col].notna().sum() >= 2}

            def check_condition(col1, op, col2):
                if col1 in valid_cols and col2 in valid_cols:
                    try:
                        v1_l, v1_p = df[col1].iloc[-1], df[col1].iloc[-2]
                        v2_l, v2_p = df[col2].iloc[-1], df[col2].iloc[-2]
                        if pd.isna(v1_l) or pd.isna(v1_p) or pd.isna(v2_l) or pd.isna(v2_p): return False
                        if op == 'cross_above': return v1_p <= v2_p and v1_l > v2_l
                        if op == 'cross_below': return v1_p >= v2_p and v1_l < v2_l
                        if op == 'break_above': return v1_p <= v2_p and v1_l > v2_l
                        if op == 'break_below': return v1_p >= v2_p and v1_l < v2_l
                    except IndexError: return False
                return False

            def check_level(col, op, level):
                 if col in valid_cols:
                     try:
                         val = df[col].iloc[-1]
                         if pd.isna(val): return False
                         if op == '>': return val > level
                         if op == '<': return val < level
                     except IndexError: return False
                 return False

            if check_condition('MA5', 'cross_above', 'MA20'): patterns.append("黃金交叉 (MA5>MA20)")
            if check_condition('MA5', 'cross_below', 'MA20'): patterns.append("死亡交叉 (MA5<MA20)")
            if check_condition('Close', 'break_above', 'BB_upper'): patterns.append("突破布林帶上軌")
            if check_condition('Close', 'break_below', 'BB_lower'): patterns.append("跌破布林帶下軌")
            if check_condition('MACD', 'cross_above', 'MACD_signal'): patterns.append("MACD 金叉")
            if check_condition('MACD', 'cross_below', 'MACD_signal'): patterns.append("MACD 死叉")
            if check_condition('K', 'cross_above', 'D'): patterns.append("KDJ 金叉")
            if check_condition('K', 'cross_below', 'D'): patterns.append("KDJ 死叉")
            if check_level('RSI', '>', 75): patterns.append("RSI 超買 (>75)")
            if check_level('RSI', '<', 25): patterns.append("RSI 超賣 (<25)")

            # Use ternary operator correctly
            return patterns if patterns else ["近期無明顯技術形態"]
        except Exception as e:
            logger.error(f"識別技術形態時發生錯誤 ({self.ticker}): {e}")
            return ["無法識別技術形態"]

    def _generate_chart(self, days=180):
        try:
            if self.data is None or self.data.empty or len(self.data) < 2:
                raise ValueError(f"數據不足 ({len(self.data) if self.data is not None else 0}點)，無法生成圖表")
            df = self.data.tail(days).copy()
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.15, 0.15, 0.2])

            # Candlestick or Line
            if all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']) and df['Close'].notna().any():
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            elif 'Close' in df.columns and df['Close'].notna().any():
                 fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='收盤價', line=dict(color='lightblue', width=1.5)), row=1, col=1)

            # Moving Averages
            for ma in ['MA5', 'MA20', 'MA120']:
                if ma in df.columns and df[ma].notna().any():
                    color = {'MA5':'orange', 'MA20':'blue', 'MA120':'green'}[ma]
                    fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=color, width=1)), row=1, col=1)

            # Bollinger Bands
            if all(c in df.columns for c in ['BB_upper', 'BB_middle', 'BB_lower']) and df['BB_upper'].notna().any():
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='布林上軌', line=dict(color='rgba(173, 204, 255, 0.7)', width=1), fill=None), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='布林下軌', line=dict(color='rgba(173, 204, 255, 0.7)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(173, 204, 255, 0.1)'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='布林中軌', line=dict(color='rgba(255, 255, 255, 0.5)', width=1, dash='dot')), row=1, col=1)

            # Volume
            if 'Volume' in df.columns and df['Volume'].notna().any():
                if all(c in df.columns for c in ['Open', 'Close']):
                     valid_rows = df.dropna(subset=['Open', 'Close'])
                     # Use ternary operator correctly
                     colors = ['#2ca02c' if row['Close'] >= row['Open'] else '#d62728' for _, row in valid_rows.iterrows()]
                     fig.add_trace(go.Bar(x=valid_rows.index, y=valid_rows['Volume'], name='成交量', marker_color=colors, marker_line_width=0), row=2, col=1)
                else:
                     fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color='grey', marker_line_width=0), row=2, col=1)

            # MACD
            if all(c in df.columns for c in ['MACD', 'MACD_signal', 'MACD_hist']) and df['MACD'].notna().any():
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD信號', line=dict(color='red', width=1)), row=3, col=1)
                # Use ternary operator correctly
                colors_macd = ['#2ca02c' if val >= 0 else '#d62728' for val in df['MACD_hist'].fillna(0)]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD柱', marker_color=colors_macd, marker_line_width=0), row=3, col=1)

            # KDJ
            if all(c in df.columns for c in ['K', 'D', 'J']) and df['K'].notna().any():
                fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K值', line=dict(color='blue', width=1)), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D值', line=dict(color='red', width=1)), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['J'], name='J值', line=dict(color='green', width=1)), row=4, col=1)
                fig.add_hline(y=80, line=dict(dash="dot", color="grey", width=1), row=4, col=1)
                fig.add_hline(y=20, line=dict(dash="dot", color="grey", width=1), row=4, col=1)

            # Layout
            fig.update_layout(
                title=f'{self.company_name} ({self.ticker}) 技術分析圖 ({days}天)',
                xaxis_rangeslider_visible=False, template='plotly_dark', height=750,
                margin=dict(l=40, r=120, t=80, b=40), paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.2)', font=dict(color='white', size=11),
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            fig.update_yaxes(title_text='價格', row=1, col=1, title_font_size=10, tickfont_size=9)
            fig.update_yaxes(title_text='成交量', row=2, col=1, title_font_size=10, tickfont_size=9)
            fig.update_yaxes(title_text='MACD', row=3, col=1, title_font_size=10, tickfont_size=9, zeroline=True, zerolinewidth=1, zerolinecolor='grey')
            fig.update_yaxes(title_text='KDJ', row=4, col=1, title_font_size=10, tickfont_size=9, range=[0, 100])
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')

            # Save chart
            chart_filename = f"{self.ticker.replace('.', '_')}_chart_{secrets.token_hex(4)}.html"
            chart_path = os.path.join('static', 'charts', chart_filename)
            fig.write_html(chart_path, full_html=False, include_plotlyjs='cdn')
            logger.info(f"圖表已生成: {chart_path}")
            return os.path.join('static', 'charts', chart_filename).replace("\\", "/")
        except ValueError as ve:
             logger.error(f"生成圖表時發生錯誤 ({self.ticker}): {ve}")
             return None
        except Exception as e:
            logger.error(f"生成圖表時發生未知錯誤 ({self.ticker}): {e}", exc_info=True)
            return None

    def _get_fallback_news(self):
        logger.warning(f"無法從主要來源獲取 {self.ticker} 的新聞，返回空列表。")
        return []

    def _get_stock_news(self, days=7, num_news=10):
        try:
            if self.market == "TW":
                query = self.ticker.replace('.TW', '')
                lang, gl, ceid = 'zh-TW', 'TW', 'TW:zh-Hant'
                encoded_query = urllib.parse.quote_plus(query)
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl={lang}&gl={gl}&ceid={ceid}"
            else:
                query = self.company_name if self.company_name else self.ticker
                lang, gl, ceid = 'en-US', 'US', 'US:en'
                encoded_query = urllib.parse.quote_plus(f"{query} stock")
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl={lang}&gl={gl}&ceid={ceid}"

            logger.info(f"Fetching news from Google News RSS ({self.market}): {rss_url}")
            feed = feedparser.parse(rss_url)
            recent_news = []
            now = dt.datetime.now(dt.timezone.utc)

            if not feed.entries:
                logger.warning(f"找不到 '{query}' 的 Google 新聞")
                return self._get_fallback_news()

            for entry in feed.entries:
                try:
                    published_time = dt.datetime.now(dt.timezone.utc) # Default to now
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            time_tuple = entry.published_parsed[:6]
                            # Use ternary operator correctly
                            published_time = dt.datetime(*time_tuple).replace(tzinfo=dt.timezone.utc) if len(time_tuple) == 6 else published_time
                        except (ValueError, TypeError) as time_err:
                            logger.warning(f"無法解析新聞日期: {entry.get('published_parsed')} - Error: {time_err}")

                    if (now - published_time).days <= days:
                        title = entry.get('title', "無標題")
                        link = entry.get('link', "#")
                        source = entry.get('source', {}).get('title', 'Google News')

                        news_entry = {
                            'title': title, 'link': link,
                            'date': published_time.strftime('%Y-%m-%d'),
                            'source': source,
                            'sentiment': 'N/A', 'sentiment_score': None
                        }

                        if title != "無標題":
                            try:
                                sentiment = self.sentiment_analyzer(title[:512])[0]
                                news_entry['sentiment'] = sentiment['label']
                                news_entry['sentiment_score'] = sentiment['score']
                            except Exception as e:
                                logger.warning(f"情緒分析失敗 for '{title[:50]}...': {e}")

                        recent_news.append(news_entry)
                        if len(recent_news) >= num_news:
                            break

                except Exception as inner_e:
                     logger.error(f"處理新聞條目 '{entry.get('title', 'N/A')}' 時發生錯誤: {inner_e}")
                     continue

            # Use ternary operator correctly
            logger.info(f"--- 為 {self.ticker} 找到的近期新聞 ({len(recent_news)} 則, {days}天內) ---") if recent_news else logger.info(f"--- 未找到 {self.ticker} 的相關近期新聞 ({days}天內) ---")
            if recent_news:
                logger.info("-------------------------------------------------")

            return recent_news

        except Exception as e:
            logger.error(f"獲取近期新聞時發生錯誤 ({self.ticker}): {e}")
            return self._get_fallback_news()

    def _get_ai_analysis(self, news_list: list = None):
        if not stock_analysis_model:
             return "AI 分析模型未能成功載入，無法生成報告。"
        try:
            min_data_points_daily = 2
            min_data_points_weekly = 6
            min_data_points_monthly = 23

            if self.data is None or len(self.data) < min_data_points_daily:
                 return f"無法為 {self.company_name} ({self.ticker}) 生成 AI 分析報告，數據不足 (需要至少 {min_data_points_daily} 天)。"

            # Change Calculation
            daily_change_val = None
            weekly_change_val = None
            monthly_change_val = None
            latest_close = self.data['Close'].iloc[-1] if not self.data.empty else None

            if latest_close is not None and len(self.data) >= min_data_points_daily:
                daily_change_series = self.data['Close'].pct_change(periods=1) * 100
                # Use ternary operator correctly
                daily_change_val = daily_change_series.iloc[-1] if not daily_change_series.empty and pd.notna(daily_change_series.iloc[-1]) else None

            if latest_close is not None and len(self.data) >= min_data_points_weekly:
                 weekly_change_series = self.data['Close'].pct_change(periods=5) * 100
                 # Use ternary operator correctly
                 weekly_change_val = weekly_change_series.iloc[-1] if not weekly_change_series.empty and pd.notna(weekly_change_series.iloc[-1]) else None

            if latest_close is not None and len(self.data) >= min_data_points_monthly:
                 monthly_change_series = self.data['Close'].pct_change(periods=22) * 100
                 # Use ternary operator correctly
                 monthly_change_val = monthly_change_series.iloc[-1] if not monthly_change_series.empty and pd.notna(monthly_change_series.iloc[-1]) else None

            # Apply limits
            change_limit = 200
            if daily_change_val is not None and not (-change_limit <= daily_change_val <= change_limit): logger.warning(f"[{self.ticker}] 日漲跌幅異常 ({daily_change_val:.2f}%)，將顯示為 N/A。"); daily_change_val = None
            if weekly_change_val is not None and not (-change_limit <= weekly_change_val <= change_limit): logger.warning(f"[{self.ticker}] 週漲跌幅異常 ({weekly_change_val:.2f}%)，將顯示為 N/A。"); weekly_change_val = None
            if monthly_change_val is not None and not (-change_limit <= monthly_change_val <= change_limit): logger.warning(f"[{self.ticker}] 月漲跌幅異常 ({monthly_change_val:.2f}%)，將顯示為 N/A。"); monthly_change_val = None

            patterns = self._identify_patterns()
            # Use ternary operator correctly
            patterns_str = ", ".join(patterns) if patterns else "近期無明顯技術形態"

            def format_val(val, precision=2, is_currency=False, is_percent=False, currency_symbol=''):
                if val is None or str(val).upper() == 'N/A' or pd.isna(val): return "N/A"
                try:
                    num = float(val)
                    if is_percent: return f"{num:.{precision}f}%"
                    if is_currency:
                        if abs(num) >= 1e12: formatted_num = f"{num/1e12:.1f}兆"
                        elif abs(num) >= 1e8: formatted_num = f"{num/1e8:.1f}億"
                        elif abs(num) >= 1e4: formatted_num = f"{num/1e4:.1f}萬"
                        else: formatted_num = f"{num:,.{precision}f}"
                        return f"{currency_symbol}{formatted_num}"
                    else:
                        if abs(num) >= 1e9: return f"{num/1e9:.1f}B"
                        if abs(num) >= 1e6: return f"{num/1e6:.1f}M"
                        return f"{num:,.{precision}f}"
                except (ValueError, TypeError): return str(val)

            # Use ternary operator correctly
            currency_symbol = '' if self.currency == 'TWD' else '$'
            latest_rsi = self.data.iloc[-1].get('RSI')
            latest_macd = self.data.iloc[-1].get('MACD')
            latest_k = self.data.iloc[-1].get('K')
            latest_d = self.data.iloc[-1].get('D')
            latest_bb_upper = self.data.iloc[-1].get('BB_upper')
            latest_bb_middle = self.data.iloc[-1].get('BB_middle')
            latest_bb_lower = self.data.iloc[-1].get('BB_lower')

            # Format news
            news_section_prompt = "近期無相關新聞可供參考。"
            if news_list:
                formatted_news = []
                for i, item in enumerate(news_list[:7]):
                    title = item.get('title', 'N/A').strip()
                    source = item.get('source', 'N/A')
                    formatted_news.append(f"{i+1}. {title} ({source})")
                # Use ternary operator correctly
                news_section_prompt = "\n".join(formatted_news) if formatted_news else news_section_prompt

            # Prompt (Keep the detailed prompt structure)
            prompt = f"""
            你是一位頂尖的股票分析師，請針對以下股票數據和**近期新聞標題**，生成一份專業、精簡、結構化的分析報告。請使用繁體中文，並盡可能以 **條列式 (bullet points)** 或 **表格** 的方式呈現各個分析要點。重點是提供洞見，而不僅是重複數據。請勿在報告中提及你的身份或報告日期。

            **股票基本資料：**
            *   股票名稱: {self.company_name or 'N/A'}
            *   股票代碼: {self.ticker}
            *   市場: {'台股' if self.market == 'TW' else '美股'}
            *   當前價格: {format_val(latest_close, is_currency=True, currency_symbol=currency_symbol)} {self.currency}
            *   日漲跌幅: {format_val(daily_change_val, is_percent=True) if daily_change_val is not None else 'N/A'}
            *   週漲跌幅: {format_val(weekly_change_val, is_percent=True) if weekly_change_val is not None else 'N/A'}
            *   月漲跌幅: {format_val(monthly_change_val, is_percent=True) if monthly_change_val is not None else 'N/A'}

            **關鍵數據：**
            *   技術指標: RSI={format_val(latest_rsi)}, MACD={format_val(latest_macd, precision=4)}, K={format_val(latest_k)}, D={format_val(latest_d)}
            *   布林帶: 上軌={format_val(latest_bb_upper)}, 中軌={format_val(latest_bb_middle)}, 下軌={format_val(latest_bb_lower)}
            *   近期技術形態: {patterns_str}
            *   基本面: P/E={format_val(self.pe_ratio)},
                      市值={format_val(self.market_cap, is_currency=True, precision=0, currency_symbol=currency_symbol)},
                      EPS={format_val(self.eps, is_currency=True, currency_symbol=currency_symbol)},
                      ROE={format_val(self.roe * 100, is_percent=True) if isinstance(self.roe, (int, float)) else 'N/A'},
                      淨利率={self.net_profit_margin_str or 'N/A'},
                      流動比率={self.current_ratio_str or 'N/A'}

            **近期相關新聞標題 (供參考):**
            {news_section_prompt}

            **請根據以上所有數據 (包含基本面、技術面、近期形態和新聞標題)，生成包含以下部分的分析報告 (使用條列式或表格)：**

            **1. 公司簡介與業務重點:** (1-2句話總結)
            **2. 主要優勢 / 看漲理由 (Strengths / Bull Case):**
               *   (分析基本面、技術面或市場趨勢的正面因素)
            **3. 主要風險 / 看跌理由 (Weaknesses / Risks / Bear Case):**
               *   (分析基本面、技術面或市場趨勢的負面因素)
            **4. 技術面分析摘要:**
               *   短期趨勢: (例如：多頭排列、盤整、空頭)
               *   動能與超買/賣: (基於 RSI, MACD, KDJ 判斷)
               *   支撐/壓力: (基於均線、布林帶或近期高低點，可選)
            **5. 基本面分析摘要:**
               *   盈利能力: (評估 ROE 和 淨利率 水平。如果數據為 'N/A'，請說明數據缺失。)
               *   估值水平: (評估 P/E 相對歷史或同業。如果數據為 'N/A'，請說明。)
               *   財務健康: (評估 流動比率。如果數據為 'N/A' 或 '數據不足'，請說明數據缺失或無法評估。)
            **6. 近期新聞分析:**
               *   (請**解讀**上面提供的「近期相關新聞標題」列表，分析這些新聞可能對股價產生的短期或長期影響、反映的市場情緒或重要的公司/產業動態。簡述即可，1-3點。)
            **7. 總結與展望:**
               *   綜合評價與未來潛力 (**結合技術、基本面與新聞分析**)。
               *   短期操作建議參考 (例如：逢低佈局、突破追買、保持觀望、風險控制)。
               *   **免責聲明:** 此分析僅供參考，不構成投資建議。

            **要求：**
            *   分析需客觀、基於數據，提出明確觀點。**新聞分析部分需簡潔地點出新聞重點及其潛在影響。**
            *   語言精煉，格式清晰。
            *   不要只羅列數據，要解釋數據代表的意義。
            """

            generation_config = genai.types.GenerationConfig(temperature=self.temperature)
            logger.info(f"為 {self.ticker} 生成 AI 分析 (含新聞分析，內部溫度: {self.temperature})")
            response = self.model.generate_content(prompt, generation_config=generation_config)

            if not response.candidates:
                 # Use ternary operator correctly
                 prompt_feedback_info = f" Prompt Feedback: {response.prompt_feedback}" if hasattr(response, 'prompt_feedback') and response.prompt_feedback else ""
                 logger.warning(f"AI 分析請求可能因安全設定被阻擋 ({self.ticker})。{prompt_feedback_info}")
                 return f"無法為 {self.company_name} ({self.ticker}) 生成 AI 分析報告，請求可能因安全設定被阻擋。{prompt_feedback_info}"

            try:
                candidate = response.candidates[0]
                if candidate.finish_reason.name != "STOP":
                    # Use ternary operator correctly
                    safety_info = f" Safety Ratings: {candidate.safety_ratings}" if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings else ""
                    logger.warning(f"AI 分析回應未正常結束 ({self.ticker})。原因: {candidate.finish_reason.name}{safety_info}")
                    return f"無法為 {self.company_name} ({self.ticker}) 生成 AI 分析報告 (原因: {candidate.finish_reason.name})。{safety_info}"
                analysis = response.text
            except ValueError:
                 finish_reason = response.candidates[0].finish_reason.name
                 # Use ternary operator correctly
                 safety_info = f" Safety Ratings: {response.candidates[0].safety_ratings}" if hasattr(response.candidates[0], 'safety_ratings') and response.candidates[0].safety_ratings else ""
                 logger.warning(f"AI 分析回應被阻擋或無有效文本 ({self.ticker})。 Candidate: {response.candidates[0]} Reason: {finish_reason}{safety_info}")
                 return f"無法為 {self.company_name} ({self.ticker}) 生成 AI 分析報告 (原因: {finish_reason}，可能內容被過濾)。{safety_info}"
            except Exception as text_err:
                 logger.error(f"提取 AI 分析文本時出錯 ({self.ticker}): {text_err}")
                 return f"無法為 {self.company_name} ({self.ticker}) 生成 AI 分析報告 (文本提取錯誤)。"

            return analysis
        except Exception as e:
            logger.error(f"生成 AI 分析 ({self.ticker}) 時發生錯誤: {e}", exc_info=True)
            return f"無法為 {self.company_name} ({self.ticker}) 生成 AI 分析報告。\n錯誤詳情: {str(e)}"

    def get_stock_summary(self):
        try:
            if self.data is None or self.data.empty or len(self.data) < 2:
                return { "ticker": self.ticker, "company_name": self.company_name or self.ticker,
                         "error": f"數據不足，無法生成綜合分析 (需要至少 2 天數據)" }

            # Change Calculation
            price_change_val = None
            if len(self.data) >= 2:
                 change_series = self.data['Close'].pct_change(periods=1) * 100
                 # Use ternary operator correctly
                 price_change_val = change_series.iloc[-1] if not change_series.empty and pd.notna(change_series.iloc[-1]) else None

            # Apply limit
            change_limit = 200
            if price_change_val is not None and not (-change_limit <= price_change_val <= change_limit):
                logger.warning(f"[{self.ticker}] Summary 日漲跌幅異常 ({price_change_val:.2f}%)，將顯示為 N/A。")
                price_change_val = None

            # Use ternary operator correctly
            price_change_str = f"{price_change_val:+.2f}%" if price_change_val is not None else "N/A"

            latest = self.data.iloc[-1]

            patterns = self._identify_patterns()
            chart_path = self._generate_chart()
            news = self._get_stock_news(days=7, num_news=10)
            analysis = self._get_ai_analysis(news_list=news)

            summary = {
                "ticker": self.ticker,
                "company_name": self.company_name or 'N/A',
                # Use ternary operator correctly
                "currency": self.currency or ('TWD' if self.market == 'TW' else 'USD'),
                "current_price": latest.get('Close'),
                "price_change": price_change_str,
                "price_change_value": price_change_val,
                # Use ternary operator correctly
                "volume": int(latest.get('Volume', 0)) if pd.notna(latest.get('Volume')) else 0,
                # Use ternary operator correctly
                "pe_ratio": self.pe_ratio if self.pe_ratio is not None else None,
                # Use ternary operator correctly
                "market_cap": self.market_cap if self.market_cap is not None else None,
                # Use ternary operator correctly
                "eps": self.eps if self.eps is not None else None,
                # Use ternary operator correctly
                "roe": self.roe if isinstance(self.roe, (int, float)) else None,
                "net_profit_margin": self.net_profit_margin_str or 'N/A',
                "current_ratio": self.current_ratio_str or 'N/A',
                "rsi": latest.get('RSI'), "macd": latest.get('MACD'),
                "macd_signal": latest.get('MACD_signal'), "k": latest.get('K'),
                "d": latest.get('D'), "j": latest.get('J'),
                "patterns": patterns, "chart_path": chart_path,
                "news": news,
                "analysis": analysis
            }
            # Create display ROE separately
            # Use ternary operator correctly
            summary["roe_display"] = f"{summary['roe'] * 100:.2f}%" if isinstance(summary["roe"], (int, float)) else "N/A"

            return summary
        except Exception as e:
            logger.error(f"獲取股票綜合分析 ({self.ticker}) 時發生錯誤: {e}", exc_info=True)
            return { "ticker": self.ticker, "company_name": self.company_name or self.ticker,
                     "error": f"獲取綜合分析失敗: {str(e)}" }


# ------------------ API 路由 ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis/<ticker>')
def analysis(ticker):
    market = request.args.get('market', 'TW')
    # --- .TW Auto Append Logic for URL ---
    if market == "TW" and not ticker.endswith(".TW") and re.fullmatch(r'\d{4,6}', ticker):
        ticker = f"{ticker}.TW"
    # --- End .TW Logic ---
    return render_template('analysis.html', ticker=ticker, market=market)

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    if not stock_analysis_model: return jsonify({'error': 'AI 分析模型未載入'}), 503
    try:
        data = request.get_json()
        ticker_input = data.get('ticker', '').strip()
        market = data.get('market', 'TW')
        if not ticker_input: return jsonify({'error': '請提供股票代碼'}), 400

        # --- .TW Auto Append Logic ---
        ticker = ticker_input
        if market == "TW" and not ticker.endswith(".TW") and re.fullmatch(r'\d{4,6}', ticker_input):
            logger.info(f"API /analyze: Appending .TW to potential Taiwan stock code: {ticker_input}")
            ticker = f"{ticker_input}.TW"
        # --- End .TW Logic ---

        user_id = 1
        settings = get_user_settings(user_id)
        user_temperature = settings.get('temperature', 0.7)

        analyzer = StockAnalyzer(ticker, GEMINI_API_KEY, period="5y", market=market, temperature=user_temperature)
        summary = analyzer.get_stock_summary()

        if 'error' in summary: return jsonify({'error': summary['error']}), 500
        return jsonify({'success': True, 'data': summary})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"分析股票時發生錯誤 ({ticker_input}): {e}", exc_info=True)
        return jsonify({'error': f"分析股票時發生意外錯誤: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    if not general_model or not stock_analysis_model or not portfolio_model:
         return jsonify({'error': 'AI 模型未完全載入，部分功能可能無法使用'}), 503
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        market = data.get('market', 'TW')
        if not message: return jsonify({'error': '請輸入訊息'}), 400

        user_id = 1
        settings = get_user_settings(user_id)
        user_temperature = settings.get('temperature', 0.7)
        generation_config = genai.types.GenerationConfig(temperature=user_temperature)

        stock_match = re.search(r'[#＃]([0-9A-Za-z\.]+)', message)
        if stock_match:
            ticker_input = stock_match.group(1)
            # --- .TW Auto Append Logic ---
            ticker = ticker_input
            if market == "TW" and not ticker.endswith(".TW") and re.fullmatch(r'\d{4,6}', ticker_input):
                logger.info(f"API /chat: Appending .TW to potential Taiwan stock code: {ticker_input}")
                ticker = f"{ticker_input}.TW"
            # --- End .TW Logic ---
            try:
                analyzer = StockAnalyzer(ticker, GEMINI_API_KEY, period="5y", market=market, temperature=user_temperature)
                summary = analyzer.get_stock_summary()
                if 'error' in summary:
                    return jsonify({'success': True, 'type': 'text', 'data': f"分析股票 {ticker} 時發生錯誤: {summary['error']}"})
                # Add to watchlist (no portfolio name here)
                conn = get_db_connection()
                cursor = None
                if conn:
                    try:
                        cursor = conn.cursor()
                        # Add NULL for portfolio_name when adding via chat
                        cursor.execute('INSERT IGNORE INTO watchlist (user_id, ticker, name, portfolio_name) VALUES (%s, %s, %s, %s)',
                                       (user_id, analyzer.ticker, analyzer.company_name or analyzer.ticker, None))
                        conn.commit()
                    except Exception as db_err:
                        logger.error(f"更新追蹤清單時出錯 ({ticker}): {db_err}")
                    finally:
                        if cursor: cursor.close()
                        if conn and conn.is_connected(): conn.close()
                return jsonify({'success': True, 'type': 'stock', 'data': summary})
            except ValueError as e:
                 return jsonify({'success': True, 'type': 'text', 'data': str(e)})
            except Exception as e:
                logger.error(f"處理股票查詢時發生錯誤 ({ticker_input}): {e}", exc_info=True)
                return jsonify({'success': True, 'type': 'text', 'data': f"無法查詢股票 {ticker_input}。錯誤: {str(e)}"})

        # General chat query
        try:
            market_str = "台股" if market == "TW" else "美股"
            prompt = f"""你是一位專業的股票分析師和投資顧問，專精於{market_str}市場分析。請以專業、友善且樂於助人的語氣回答。\n\n使用者的訊息: {message}\n\n請用繁體中文回覆。如果使用者詢問特定股票，提醒他們可以使用 `#股票代碼` 的格式來查詢詳細分析。如果訊息模糊不清，可以禮貌地請使用者提供更多細節。如果使用者詢問投資組合建議，請引導他們使用「投資組合」頁面的「取得 AI 建議」功能。"""
            response = general_model.generate_content(prompt, generation_config=generation_config)
            if not response.candidates:
                return jsonify({'success': True, 'type': 'text', 'data': "抱歉，我目前無法處理您的請求 (內容安全)。"})
            try:
                response_text = response.text
            except ValueError:
                finish_reason = response.candidates[0].finish_reason.name
                return jsonify({'success': True, 'type': 'text', 'data': f"抱歉，我目前無法處理您的請求 (原因: {finish_reason})。"})
            except Exception as text_err:
                logger.error(f"提取一般聊天文本時出錯: {text_err}")
                return jsonify({'success': True, 'type': 'text', 'data': "抱歉，我目前無法處理您的請求 (文本提取錯誤)。"})
            return jsonify({'success': True, 'type': 'text', 'data': response_text})
        except Exception as e:
            logger.error(f"使用Gemini處理一般訊息時發生錯誤: {e}", exc_info=True)
            return jsonify({'success': True, 'type': 'text', 'data': "抱歉，我目前無法處理您的請求，請稍後再試或換個方式提問。"})
    except Exception as e:
        logger.error(f"處理聊天請求時發生意外錯誤: {e}", exc_info=True)
        return jsonify({'error': f"處理聊天請求時發生意外錯誤: {str(e)}"}), 500

# --- Watchlist Routes (FIXED SQL, Added Clear Route, Grouping Logic) ---
@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    conn = None
    cursor = None
    try:
        user_id = 1
        conn = get_db_connection()
        if not conn: return jsonify({'error': '無法連接資料庫'}), 500
        cursor = conn.cursor(dictionary=True)
        # Fetch portfolio_name and order by it, then by added_at
        cursor.execute('''
            SELECT w.ticker, w.name, w.portfolio_name, s.close_price, s.open_price
            FROM watchlist w
            LEFT JOIN stocks s ON w.ticker = s.symbol
            WHERE w.user_id = %s
            ORDER BY w.portfolio_name ASC, w.added_at DESC
        ''', (user_id,))
        watchlist_items_raw = cursor.fetchall()

        # Group items by portfolio_name
        grouped_watchlist = {}
        for item in watchlist_items_raw:
            ticker, name = item['ticker'], item['name']
            portfolio_name = item.get('portfolio_name') # Can be None
            # Use ternary operator correctly
            group_key = portfolio_name if portfolio_name else "__manual__" # Use special key for manual adds

            if group_key not in grouped_watchlist:
                grouped_watchlist[group_key] = {
                    # Use ternary operator correctly
                    "name": portfolio_name if portfolio_name else "手動添加", # Display name for the group
                    "is_manual": portfolio_name is None,
                    "items": []
                }

            # Process price/change
            price, change, source, error_flag = None, 0.0, "N/A", False
            db_price, db_open = item.get('close_price'), item.get('open_price')
            if db_price is not None:
                 price = db_price
                 # Use ternary operator correctly
                 change = ((db_price - db_open) / db_open) * 100 if db_open is not None and db_open != 0 else 0.0
                 source = "DB"
            else:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="2d")
                    if not hist.empty and len(hist) >= 2:
                        current, prev_close = hist.iloc[-1], hist.iloc[-2]['Close']
                        price = current['Close']
                        # Use ternary operator correctly
                        change = ((price - prev_close) / prev_close) * 100 if pd.notna(price) and pd.notna(prev_close) and prev_close != 0 else 0.0
                        error_flag = pd.isna(price)
                        source = "Yahoo"
                    elif not hist.empty:
                        price = hist.iloc[-1]['Close']
                        error_flag = pd.isna(price)
                        source = "Yahoo (1d)"
                    else:
                        error_flag = True
                        source = "Fetch Error"
                except Exception as yf_e:
                    error_flag = True
                    source = "Fetch Error"
                    logger.error(f"yfinance error for {ticker}: {yf_e}")

            grouped_watchlist[group_key]["items"].append({
                'ticker': ticker, 'name': name, 'price': price, 'change': change,
                'source': source, 'error': error_flag or (price is None)
            })

        # Convert dict to list for easier frontend iteration, manual group first
        final_watchlist_groups = []
        if "__manual__" in grouped_watchlist:
            final_watchlist_groups.append(grouped_watchlist.pop("__manual__"))
        # Add remaining portfolio groups
        final_watchlist_groups.extend(grouped_watchlist.values())

        return jsonify({'grouped_watchlist': final_watchlist_groups}) # Return grouped data

    except mysql.connector.Error as db_err:
        logger.error(f"獲取追蹤清單時發生 DB 錯誤: {db_err}", exc_info=True)
        return jsonify({'error': f"獲取追蹤清單時資料庫錯誤: {db_err.msg}"}), 500
    except Exception as e:
        logger.error(f"獲取追蹤清單時發生錯誤: {e}", exc_info=True)
        return jsonify({'error': f"獲取追蹤清單時發生錯誤: {str(e)}"}), 500
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


@app.route('/api/watchlist/add', methods=['POST'])
def add_to_watchlist():
    conn = None
    cursor = None
    try:
        data = request.get_json()
        ticker_input = data.get('ticker', '').strip().upper()
        portfolio_name = data.get('portfolio_name', None) # Get optional portfolio name
        user_id = 1
        if not ticker_input: return jsonify({'error': '請提供股票代碼'}), 400

        # --- .TW Auto Append Logic ---
        ticker = ticker_input
        if not ticker.endswith(".TW") and "." not in ticker and re.fullmatch(r'\d{4,6}', ticker_input):
            logger.info(f"API /watchlist/add: Appending .TW to potential Taiwan stock code: {ticker_input}")
            ticker = f"{ticker_input}.TW"
        # --- End .TW Logic ---

        fetched_name = ticker
        try:
            stock_info = yf.Ticker(ticker).get_info(timeout=5)
            fetched_name = stock_info.get('longName', stock_info.get('shortName', ticker))
        except Exception as e: logger.warning(f"無法從 yfinance 獲取 {ticker} 的名稱: {e}")

        conn = get_db_connection()
        if not conn: return jsonify({'error': '無法連接資料庫'}), 500
        cursor = conn.cursor()
        # Use INSERT ... ON DUPLICATE KEY UPDATE to handle existing tickers
        cursor.execute(
            'INSERT INTO watchlist (user_id, ticker, name, portfolio_name) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE name=VALUES(name), portfolio_name=VALUES(portfolio_name)',
            (user_id, ticker, fetched_name, portfolio_name)
        )
        affected_rows = cursor.rowcount # 1 for insert, 2 for update, 0 for no change
        conn.commit()

        # Determine message based on affected_rows
        if affected_rows == 1:
            # Use ternary operator correctly
            message = f'已將 {fetched_name} ({ticker}) 加入追蹤清單' + (f' (來自 {portfolio_name})' if portfolio_name else '')
        elif affected_rows == 2:
            # Use ternary operator correctly
            message = f'已更新 {fetched_name} ({ticker}) 在追蹤清單中的資訊' + (f' (歸類於 {portfolio_name})' if portfolio_name else '')
        else: # affected_rows == 0
            message = f'{fetched_name} ({ticker}) 已在追蹤清單中，資訊未變更'

        return jsonify({'success': True, 'message': message, 'added': affected_rows > 0}) # 'added' is true for insert or update

    except mysql.connector.Error as db_err:
         logger.error(f"添加到追蹤清單時發生 DB 錯誤 ({ticker_input}): {db_err}", exc_info=True)
         # Use ternary operator correctly
         conn.rollback() if conn else None
         return jsonify({'error': f"添加到追蹤清單時資料庫錯誤: {db_err.msg}"}), 500
    except Exception as e:
        logger.error(f"添加到追蹤清單時發生一般錯誤 ({ticker_input}): {e}", exc_info=True)
        # Use ternary operator correctly
        conn.rollback() if conn else None
        return jsonify({'error': f"添加到追蹤清單時發生錯誤: {str(e)}"}), 500
    finally:
         if cursor: cursor.close()
         if conn and conn.is_connected(): conn.close()

@app.route('/api/watchlist/remove', methods=['POST'])
def remove_from_watchlist():
    conn = None
    cursor = None
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()
        user_id = 1
        if not ticker: return jsonify({'error': '請提供股票代碼'}), 400
        conn = get_db_connection()
        if not conn: return jsonify({'error': '無法連接資料庫'}), 500
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM watchlist WHERE user_id = %s AND ticker = %s', (user_id, ticker))
        result = cursor.fetchone()
        # Use ternary operator correctly
        name = result[0] if result else ticker
        cursor.execute('DELETE FROM watchlist WHERE user_id = %s AND ticker = %s', (user_id, ticker))
        affected_rows = cursor.rowcount
        conn.commit()
        # Use ternary operator correctly
        return jsonify({'success': True, 'message': f'已從追蹤清單中移除 {name}'}) if affected_rows > 0 else jsonify({'success': False, 'error': f'{ticker} 不在追蹤清單中'}), 404
    except mysql.connector.Error as db_err:
         logger.error(f"從追蹤清單移除時發生 DB 錯誤 ({ticker}): {db_err}", exc_info=True)
         # Use ternary operator correctly
         conn.rollback() if conn else None
         return jsonify({'error': f"從追蹤清單移除時資料庫錯誤: {db_err.msg}"}), 500
    except Exception as e:
        logger.error(f"從追蹤清單移除時發生一般錯誤 ({ticker}): {e}", exc_info=True)
        # Use ternary operator correctly
        conn.rollback() if conn else None
        return jsonify({'error': f"從追蹤清單移除時發生錯誤: {str(e)}"}), 500
    finally:
         if cursor: cursor.close()
         if conn and conn.is_connected(): conn.close()

# --- Watchlist Clear Route ---
@app.route('/api/watchlist/clear', methods=['POST'])
def clear_watchlist():
    conn = None
    cursor = None
    try:
        user_id = 1 # Assuming single user
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '無法連接資料庫'}), 500

        cursor = conn.cursor()
        cursor.execute('DELETE FROM watchlist WHERE user_id = %s', (user_id,))
        affected_rows = cursor.rowcount
        conn.commit()
        logger.info(f"User {user_id} cleared watchlist, removed {affected_rows} items.")
        return jsonify({'success': True, 'message': f'已清空追蹤清單 (移除 {affected_rows} 項)'})
    except mysql.connector.Error as db_err:
        logger.error(f"清空追蹤清單時發生 DB 錯誤 (User {user_id}): {db_err}", exc_info=True)
        # Use ternary operator correctly
        conn.rollback() if conn else None
        return jsonify({'error': f"清空追蹤清單時資料庫錯誤: {db_err.msg}"}), 500
    except Exception as e:
        logger.error(f"清空追蹤清單時發生一般錯誤 (User {user_id}): {e}", exc_info=True)
        # Use ternary operator correctly
        conn.rollback() if conn else None
        return jsonify({'error': f"清空追蹤清單時發生錯誤: {str(e)}"}), 500
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

# --- Portfolio Suggestion Route (MODIFIED v4 - Allocation & Parsing) ---
def parse_suggested_portfolios(text):
    """嘗試從 Gemini 的文本回應中解析出建議的投資組合列表 (含分配比例)"""
    portfolios = []
    # 正則表達式尋找組合標題
    portfolio_pattern = re.compile(
        r"^\s*(?P<title>\*\*?\s*(?:組合|建議|Portfolio)\s*[一二三四五六七八九十\dABC]+\s*[:：]?\s*[^*]*?\*?\*?)\s*$",
        re.MULTILINE | re.IGNORECASE
    )
    # 正則表達式尋找股票列表行
    tickers_pattern = re.compile(r"^\s*(?:包含股票|股票列表)\s*[:：]?\s*(?P<tickers>.*?)$", re.MULTILINE | re.IGNORECASE)
    # 正則表達式尋找資金分配行
    allocation_pattern = re.compile(r"^\s*(?:建議分配|資金分配|配置比例)\s*[:：]?\s*(?P<allocations>.*?)$", re.MULTILINE | re.IGNORECASE)
    # 正則表達式從分配行中提取 Ticker: XX%
    ticker_alloc_pattern = re.compile(r'([A-Z0-9]+(?:\.TW)?)\s*[:：]?\s*(\d{1,3})\s*%')
    # 股票代碼的模式 (用於備案)
    ticker_pattern_fallback = re.compile(r'\b([A-Z]{1,5}(?:\.[A-Z]{1,2})?|\d{4,6}\.TW)\b')

    portfolio_matches = list(portfolio_pattern.finditer(text))
    parsed_count = 0

    for i, match in enumerate(portfolio_matches):
        start_pos = match.end()
        # 確定當前組合文本的結束位置
        # Use ternary operator correctly
        end_pos = portfolio_matches[i+1].start() if i + 1 < len(portfolio_matches) else len(text)
        portfolio_block = text[start_pos:end_pos]

        portfolio_name = match.group('title').strip().replace('*','').replace(':','').replace('：','')

        tickers = []
        allocations = {}

        # 查找股票列表
        tickers_match = tickers_pattern.search(portfolio_block)
        if tickers_match:
            tickers_str = tickers_match.group('tickers')
            potential_tickers = ticker_pattern_fallback.findall(tickers_str)
            tickers = sorted(list(set(t.upper() for t in potential_tickers if len(t)>1)))
        else:
            # 如果沒有明確的 "包含股票" 行，嘗試直接在區塊內找
            potential_tickers = ticker_pattern_fallback.findall(portfolio_block)
            tickers = sorted(list(set(t.upper() for t in potential_tickers if len(t)>1)))[:5] # 最多取5個
            logger.warning(f"組合 '{portfolio_name}' 未找到明確的 '包含股票' 行，嘗試直接提取。")


        # 查找資金分配
        allocation_match = allocation_pattern.search(portfolio_block)
        if allocation_match:
            allocations_str = allocation_match.group('allocations')
            alloc_pairs = ticker_alloc_pattern.findall(allocations_str)
            # 轉換為字典，確保鍵是大寫
            allocations = {pair[0].upper(): int(pair[1]) for pair in alloc_pairs}
            # 驗證分配總和是否接近100% (可選)
            total_alloc = sum(allocations.values())
            if not (95 <= total_alloc <= 105) and total_alloc != 0 :
                logger.warning(f"組合 '{portfolio_name}' 的解析分配比例總和 ({total_alloc}%) 不在合理範圍 (95-105)。")
            # 確保分配字典中的 ticker 也在提取的 tickers 列表中
            allocations = {tk: pc for tk, pc in allocations.items() if tk in tickers}


        if tickers and len(tickers) >= 2: # 至少需要2個才算一個組合
            portfolios.append({
                "name": portfolio_name,
                "tickers": tickers,
                "allocations": allocations # 添加解析出的分配比例
            })
            parsed_count += 1
            logger.info(f"成功解析組合 '{portfolio_name}'，股票: {', '.join(tickers)}, 分配: {allocations}")
        else:
            logger.warning(f"在組合 '{portfolio_name}' 區塊中未能解析出足夠的有效股票代碼。")


    if not portfolios: # 備案邏輯
        logger.warning("未能從 Gemini 回應中解析出結構化的投資組合建議。嘗試備案提取。")
        all_tickers_in_text = ticker_pattern_fallback.findall(text)
        unique_tickers = sorted(list(set(t.upper() for t in all_tickers_in_text if len(t)>1)))
        if len(unique_tickers) >= 3:
             logger.info(f"備案提取：找到 {len(unique_tickers)} 個可能的股票代碼，將它們作為單一組合返回。")
             portfolios.append({
                 "name": "建議提及的股票 (自動提取)",
                 "tickers": unique_tickers[:5],
                 "allocations": {} # 備案無法解析分配
             })
             parsed_count = 1 # 標記為解析出一個

    logger.info(f"最終解析出 {parsed_count} 個投資組合。")
    return portfolios


@app.route('/api/portfolio_suggestion', methods=['GET'])
def get_portfolio_suggestion():
    if not portfolio_model:
        return jsonify({'success': False, 'error': '投資組合 AI 模型未載入'}), 503

    conn = None
    cursor = None
    candidates = []
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'success': False, 'error': '無法連接資料庫'}), 500

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT ticker, current_price FROM stock_signal WHERE status = 'buy' ORDER BY ticker ASC")
        candidates = cursor.fetchall()
        logger.info(f"找到 {len(candidates)} 個 'buy' 狀態的股票候選。")

    except mysql.connector.Error as db_err:
        logger.error(f"從 stock_signal 獲取 'buy' 狀態股票時出錯: {db_err}")
        return jsonify({'success': False, 'error': f'資料庫查詢錯誤: {db_err.msg}'}), 500
    except Exception as e:
        logger.error(f"獲取 'buy' 狀態股票時發生未知錯誤: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'獲取候選股票時發生錯誤: {str(e)}'}), 500
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

    user_id = 1
    settings = get_user_settings(user_id)
    user_temperature = settings.get('temperature', 0.7)
    portfolio_generation_config = genai.types.GenerationConfig(temperature=user_temperature)

    if not candidates:
        suggestion_text = "目前沒有偵測到符合 '買入' 信號的股票，無法提供投資組合建議。請稍後再試或等待策略產生新的信號。"
        return jsonify({'success': True, 'suggestion_text': suggestion_text, 'suggested_portfolios': []})

    candidate_list_str = ", ".join([c['ticker'] for c in candidates])
    # --- MODIFIED PROMPT v3 (Added Allocation Request) ---
    prompt = f"""
    你是一位投資組合顧問。目前根據一套特定的交易策略，以下股票列表 ({len(candidates)} 檔) 顯示 '買入' 信號：
    {candidate_list_str}

    請從這份列表中，**提出 1 到 3 個不同風格或側重的範例投資組合**。
    每個組合請包含 **3 到 5 檔** 從上述列表中選出的股票。

    對於**每一個**你提出的組合，請**嚴格按照**以下格式呈現：

    **組合 [數字或字母]: [組合名稱]**
    包含股票: [股票代碼列表，用逗號分隔，例如: AAPL, MSFT, GOOGL.TW]
    建議分配: [股票代碼: 百分比%，用逗號分隔，例如: AAPL: 30%, MSFT: 40%, NVDA: 30%]
    選股理由: [簡要說明選擇這些股票的原因]

    (重複以上格式呈現 1 到 3 個組合)

    **整體風險提示:**
    [在此處加上風險提示，強調範例性質、個人風險考量和諮詢專家建議]

    **重要要求:**
    - 請務必使用 "**組合 [數字/字母]: [名稱]**" 作為每個組合的標題行。
    - 請務必使用 "**包含股票:**" 作為列出股票代碼行的開頭。
    - 請務必使用 "**建議分配:**" 作為列出資金分配行的開頭，且百分比總和應接近 100%。
    - 請務必使用 "**選股理由:**" 作為解釋理由行的開頭。
    - 股票代碼之間、分配比例之間請用**逗號**分隔。
    - 請使用繁體中文回答。
    """
    # --- END MODIFIED PROMPT v3 ---

    try:
        logger.info(f"向 Gemini 發送投資組合建議請求 (要求多組合+分配，指定格式)，候選股票: {candidate_list_str}")
        response = portfolio_model.generate_content(prompt, generation_config=portfolio_generation_config)

        if not response.candidates:
            logger.warning("Gemini 投資組合建議請求可能因安全設定被阻擋。")
            return jsonify({'success': False, 'error': "無法生成投資組合建議 (內容安全)。"})

        suggestion_text = ""
        try:
            candidate_response = response.candidates[0]
            if candidate_response.finish_reason.name != "STOP":
                 # Use ternary operator correctly
                 safety_info = f" Safety Ratings: {candidate_response.safety_ratings}" if hasattr(candidate_response, 'safety_ratings') and candidate_response.safety_ratings else ""
                 logger.warning(f"Gemini 投資組合建議回應未正常結束。原因: {candidate_response.finish_reason.name}{safety_info}")
                 return jsonify({'success': False, 'error': f"無法生成投資組合建議 (原因: {candidate_response.finish_reason.name})。"})
            suggestion_text = response.text
        except ValueError:
            finish_reason = response.candidates[0].finish_reason.name
            # Use ternary operator correctly
            safety_info = f" Safety Ratings: {response.candidates[0].safety_ratings}" if hasattr(response.candidates[0], 'safety_ratings') and response.candidates[0].safety_ratings else ""
            logger.warning(f"Gemini 投資組合建議回應被阻擋或無有效文本。原因: {finish_reason}{safety_info}")
            return jsonify({'success': False, 'error': f"無法生成投資組合建議 (原因: {finish_reason}，可能內容被過濾)。"})
        except Exception as text_err:
            logger.error(f"提取 Gemini 投資組合建議文本時出錯: {text_err}")
            return jsonify({'success': False, 'error': "無法生成投資組合建議 (文本提取錯誤)。"})

        # Parse the suggestion text using the improved parser
        suggested_portfolios = parse_suggested_portfolios(suggestion_text)

        logger.info(f"成功從 Gemini 獲取建議文本，並解析出 {len(suggested_portfolios)} 個組合。")
        return jsonify({'success': True, 'suggestion_text': suggestion_text, 'suggested_portfolios': suggested_portfolios})

    except Exception as e:
        logger.error(f"調用 Gemini API 或解析建議時發生錯誤: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f"獲取 AI 建議時發生錯誤: {str(e)}"}), 500

# --- Settings Routes ---
@app.route('/api/settings', methods=['GET'])
def get_settings():
    try:
        user_id = 1
        settings = get_user_settings(user_id)
        # Exclude temperature from the response sent to the frontend
        filtered_settings = {k: v for k, v in settings.items() if k != 'temperature'}
        return jsonify({'settings': filtered_settings})
    except Exception as e:
        logger.error(f"獲取設定時發生錯誤: {e}", exc_info=True)
        return jsonify({'error': f"獲取設定時發生錯誤: {str(e)}"}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    conn = None
    cursor = None
    try:
        data = request.get_json()
        user_id = 1
        errors = {}
        # Validate incoming data (excluding temperature)
        if not isinstance(data.get('dark_mode'), bool): errors['dark_mode'] = '必須是布爾值'
        if data.get('font_size') not in ['small', 'medium', 'large']: errors['font_size'] = '無效的字體大小'
        if not isinstance(data.get('price_alert'), bool): errors['price_alert'] = '必須是布爾值'
        if not isinstance(data.get('market_summary'), bool): errors['market_summary'] = '必須是布爾值'
        if data.get('data_source') not in ['default']: errors['data_source'] = '無效的資料來源'
        if errors: return jsonify({'error': '無效的設定值', 'details': errors}), 400

        conn = get_db_connection()
        if not conn: return jsonify({'error': '無法連接資料庫'}), 500
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM settings WHERE user_id = %s', (user_id,))
        exists = cursor.fetchone()
        # Prepare data for update/insert (excluding temperature)
        settings_to_update = {k: data[k] for k in ['dark_mode', 'font_size', 'price_alert', 'market_summary', 'data_source']}
        settings_to_update['user_id'] = user_id

        if exists:
            set_clause = ', '.join([f"{k} = %({k})s" for k in settings_to_update if k != 'user_id'])
            sql = f'UPDATE settings SET {set_clause} WHERE user_id = %(user_id)s'
        else:
            cols = ', '.join(settings_to_update.keys())
            placeholders = ', '.join([f'%({k})s' for k in settings_to_update.keys()])
            # Include temperature with default value on insert if it doesn't exist
            default_temp = get_user_settings(user_id)['temperature'] # Get default/current temp
            settings_to_update['temperature'] = default_temp
            cols += ', temperature'
            placeholders += ', %(temperature)s'
            sql = f'INSERT INTO settings ({cols}) VALUES ({placeholders})'

        cursor.execute(sql, settings_to_update)
        conn.commit()
        return jsonify({'success': True, 'message': '設定已更新'})
    except mysql.connector.Error as db_err:
         logger.error(f"更新設定時資料庫操作錯誤: {db_err}", exc_info=True)
         # Use ternary operator correctly
         conn.rollback() if conn else None
         return jsonify({'error': f"更新設定時發生資料庫錯誤: {db_err.msg}"}), 500
    except Exception as e:
        logger.error(f"更新設定時資料庫操作錯誤: {e}", exc_info=True)
        # Use ternary operator correctly
        conn.rollback() if conn else None
        return jsonify({'error': f"更新設定時發生錯誤: {str(e)}"}), 500
    finally:
         if cursor: cursor.close()
         if conn and conn.is_connected(): conn.close()

# --- Market News & Summary Routes ---
@app.route('/api/market_news', methods=['GET'])
def get_market_news():
    try:
        market = request.args.get('market', 'TW')
        category = request.args.get('category', 'general')
        news_list = fetch_market_news(market, category)
        return jsonify({'news': news_list})
    except Exception as e:
        logger.error(f"獲取市場新聞時發生錯誤: {e}", exc_info=True)
        return jsonify({'error': f"獲取市場新聞時發生錯誤: {str(e)}"}), 500

def fetch_market_news(market, category):
    try:
        # Use ternary operator correctly
        base_term, lang, loc, ceid = ("台股", 'zh-TW', 'TW', 'TW:zh-Hant') if market == 'TW' else ("US stock market", 'en-US', 'US', 'US:en')
        # Use ternary operator correctly
        category_terms = {'tech': " 科技" if market == 'TW' else " tech", 'finance': " 金融" if market == 'TW' else " finance", 'industry': " 產業" if market == 'TW' else " economy", 'general': ""}
        search_term = base_term + category_terms.get(category, "")
        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(search_term)}&hl={lang}&gl={loc}&ceid={ceid}"
        logger.info(f"Fetching market news from: {rss_url}")
        feed = feedparser.parse(rss_url)
        news_list = []
        if not feed.entries: return []
        sentiment_analyzer = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')
        for entry in feed.entries[:15]:
            try:
                published_time = dt.datetime.now() # Default
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        time_tuple = entry.published_parsed[:6]
                        # Use ternary operator correctly
                        published_time = dt.datetime(*time_tuple) if len(time_tuple) == 6 else published_time
                    except (ValueError, TypeError): pass # Ignore parsing errors
                sentiment_label, sentiment_score = 'Neutral', 0.5
                title = entry.get('title', "無標題")
                if title != "無標題":
                    try:
                        result = sentiment_analyzer(title[:512])[0]
                        sentiment_label = result['label']
                        sentiment_score = result['score']
                    except Exception: pass # Ignore sentiment errors
                summary = ""
                if hasattr(entry, 'summary'):
                    summary = re.sub(r'<[^>]+>', '', entry.summary)
                    # Use ternary operator correctly
                    summary = summary[:150].strip() + ('...' if len(summary) > 150 else '')
                news_list.append({'title': title, 'link': entry.get('link', "#"), 'date': published_time.strftime('%Y-%m-%d %H:%M'), 'source': entry.get('source', {}).get('title', 'Google News'), 'summary': summary, 'sentiment': sentiment_label, 'sentiment_score': sentiment_score})
            except Exception as inner_e:
                logger.error(f"處理市場新聞條目時發生錯誤: {inner_e}", exc_info=False)
        return news_list
    except Exception as e:
        logger.error(f"獲取市場新聞 (market={market}, category={category}) 時發生錯誤: {e}", exc_info=True)
        return []

@app.route('/api/market_summary', methods=['GET'])
def get_market_summary():
    try:
        market = request.args.get('market', 'TW')
        # Use ternary operator correctly
        indices = {'^TWII': '加權指數', '0050.TW': '台灣50 ETF', '^TWOII': '櫃買指數', '0056.TW': '高股息 ETF'} if market == 'TW' else {'^GSPC': 'S&P 500', '^DJI': '道瓊工業', '^IXIC': '納斯達克', '^VIX': '恐慌指數 (VIX)'}
        market_data = []
        for symbol, name in indices.items():
            price, change, error_flag = 0.0, 0.0, False
            try:
                hist = yf.Ticker(symbol).history(period="2d")
                if not hist.empty and len(hist) >= 2:
                    current_close, prev_close = hist['Close'].iloc[-1], hist['Close'].iloc[-2]
                    # Use ternary operator correctly
                    price = current_close if pd.notna(current_close) else 0.0
                    # Use ternary operator correctly
                    change = ((current_close - prev_close) / prev_close) * 100 if pd.notna(current_close) and pd.notna(prev_close) and prev_close != 0 else 0.0
                    error_flag = pd.isna(current_close)
                elif not hist.empty:
                    price = hist.iloc[-1]['Close']
                    error_flag = pd.isna(price)
                else:
                    error_flag = True
                    logger.warning(f"無法獲取指數 {symbol} 的數據")
            except Exception as inner_e:
                error_flag = True
                logger.error(f"獲取指數 {symbol} 數據時出錯: {inner_e}")
            # Use ternary operator correctly
            market_data.append({'symbol': symbol, 'name': name, 'price': price if not error_flag else 0, 'change': change if not error_flag else 0, 'error': error_flag})

        news = fetch_market_news(market, 'general')[:10]
        market_sentiment = "未知"
        if news:
             scores = {'Positive': 1.0, 'Negative': 0.0, 'Neutral': 0.5}
             valid_scores = [scores.get(item['sentiment'], 0.5) for item in news]
             if valid_scores:
                 average_sentiment = sum(valid_scores) / len(valid_scores)
                 # Use nested ternary operator correctly
                 market_sentiment = "樂觀" if average_sentiment > 0.65 else ("謹慎" if average_sentiment < 0.40 else "中性")
        return jsonify({'market': market, 'indices': market_data, 'news_summary': news[:5], 'sentiment': market_sentiment, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    except Exception as e:
        logger.error(f"獲取市場摘要時發生錯誤: {e}", exc_info=True)
        return jsonify({'error': f"獲取市場摘要時發生錯誤: {str(e)}"}), 500

# --- Static file serving ---
@app.route('/static/charts/<path:filename>')
def serve_chart(filename):
    if '..' in filename or filename.startswith('/'): return "Forbidden", 403
    charts_dir = os.path.join(app.root_path, 'static', 'charts')
    return send_from_directory(charts_dir, filename)

@app.route('/static/data/<path:filename>')
def serve_data(filename):
     if '..' in filename or filename.startswith('/'): return "Forbidden", 403
     data_dir = os.path.join(app.root_path, 'static', 'data')
     return send_from_directory(data_dir, filename)

# ------------------ 主程式 ------------------
if __name__ == '__main__':
    init_database()
    # Consider using Waitress or Gunicorn for production
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=5000)
    app.run(debug=True, host='0.0.0.0', port=5000) # Keep debug for development

# --- END OF FILE v4.py (Modified v5) ---