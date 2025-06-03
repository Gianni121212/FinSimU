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
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mysql.connector
import bcrypt
import json
import plotly.express as px
import chart_studio.plotly as py
import chart_studio.tools as tls

# 載入環境變數
load_dotenv()

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# 建立需要的資料夾
for path in ['static/charts', 'static/data']:
    os.makedirs(path, exist_ok=True)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# --- Gemini API 設定 (No changes needed here) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
# It's generally better to create models per request or manage them if needed,
# but global instances are okay for this example.
# Consider potential concurrency issues in a larger app.
general_model = genai.GenerativeModel("models/gemini-2.5-flash-preview-04-17")
portfolio_model = genai.GenerativeModel("models/gemini-2.5-flash-preview-04-17")
stock_analysis_model = genai.GenerativeModel("models/gemini-2.5-flash-preview-04-17") # Specific model for stock analysis

# --- Database Connection (No changes needed here) ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", "0912559910"),
            database=os.getenv("DB_NAME", "testdb")
        )
        return conn
    except Exception as e:
        logging.error(f"資料庫連接錯誤: {e}")
        return None

# --- Database Initialization (Add 'temperature' column) ---
def init_database():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()

        # 創建用戶表 (No changes needed here)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            email VARCHAR(100) UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # 創建追蹤清單表 (No changes needed here)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            ticker VARCHAR(20) NOT NULL,
            name VARCHAR(100) NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, ticker)
        )
        ''')

        # 創建設定表 (Add temperature column)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            dark_mode BOOLEAN DEFAULT TRUE,
            font_size VARCHAR(10) DEFAULT 'medium',
            price_alert BOOLEAN DEFAULT FALSE,
            market_summary BOOLEAN DEFAULT TRUE,
            data_source VARCHAR(20) DEFAULT 'default',
            temperature FLOAT DEFAULT 0.7,  # Added temperature setting
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id)
        )
        ''')
        # ---- Add column if table exists but column doesn't (Optional, but safer) ----
        try:
            cursor.execute("ALTER TABLE settings ADD COLUMN temperature FLOAT DEFAULT 0.7")
            logging.info("Added 'temperature' column to settings table.")
        except mysql.connector.Error as err:
            # Error 1060: Duplicate column name - means it already exists, safe to ignore
            if err.errno == 1060:
                logging.info("'temperature' column already exists in settings table.")
            else:
                raise # Re-raise other errors
        # ---------------------------------------------------------------------------

        # 創建股票數據表 (No changes needed here)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(20) UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            market VARCHAR(10) NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ma5 FLOAT DEFAULT NULL,
            ma20 FLOAT DEFAULT NULL,
            ma50 FLOAT DEFAULT NULL,
            ma120 FLOAT DEFAULT NULL,
            ma200 FLOAT DEFAULT NULL,
            bb_upper FLOAT DEFAULT NULL,
            bb_middle FLOAT DEFAULT NULL,
            bb_lower FLOAT DEFAULT NULL,
            rsi FLOAT DEFAULT NULL,
            wmsr FLOAT DEFAULT NULL,
            psy FLOAT DEFAULT NULL,
            bias6 FLOAT DEFAULT NULL,
            macd FLOAT DEFAULT NULL,
            macd_signal FLOAT DEFAULT NULL,
            macd_hist FLOAT DEFAULT NULL,
            k FLOAT DEFAULT NULL,
            d FLOAT DEFAULT NULL,
            j FLOAT DEFAULT NULL,
            pe_ratio FLOAT DEFAULT NULL,
            market_cap BIGINT DEFAULT NULL,
            open_price FLOAT DEFAULT NULL,
            close_price FLOAT DEFAULT NULL,
            high_price FLOAT DEFAULT NULL,
            low_price FLOAT DEFAULT NULL,
            volume BIGINT DEFAULT NULL
        )
        ''')

        conn.commit()
        cursor.close()
        conn.close()
        logging.info("資料庫初始化成功 (含 temperature 設定)")

# --- Helper function to get user settings ---
def get_user_settings(user_id=1):
    """Fetches settings for a given user_id, returns defaults if not found."""
    default_settings = {
        'dark_mode': True,
        'font_size': 'medium',
        'price_alert': False,
        'market_summary': True,
        'data_source': 'default',
        'temperature': 0.7 # Default temperature
    }
    conn = get_db_connection()
    if not conn:
        logging.warning(f"無法連接資料庫獲取用戶 {user_id} 的設定，使用預設值。")
        return default_settings

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT dark_mode, font_size, price_alert, market_summary, data_source, temperature
            FROM settings WHERE user_id = %s
        ''', (user_id,))
        settings = cursor.fetchone()
        cursor.close()
        conn.close()

        if settings:
            # Ensure all keys exist, using defaults for missing ones (like temperature if old DB)
            for key, default_value in default_settings.items():
                if key not in settings or settings[key] is None:
                    settings[key] = default_value
            return settings
        else:
            return default_settings
    except Exception as e:
        logging.error(f"獲取用戶 {user_id} 設定時發生錯誤: {e}")
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
        return default_settings


# ------------------ 股票分析功能 (Use temperature) ------------------
class StockAnalyzer:
    # Add temperature to __init__
    def __init__(self, ticker: str, api_key: str, period: str = "10y", market: str = "TW", temperature: float = 0.7):
        self.ticker = ticker.strip()
        if market == "TW" and "." not in self.ticker:
            self.ticker = f"{self.ticker}.TW"
        self.period = period
        self.market = market
        self.temperature = temperature # Store temperature
        self.stock = yf.Ticker(self.ticker)
        self.data = None
        self.company_name = None
        self.currency = None
        self.pe_ratio = None
        self.market_cap = None
        self.forward_pe = None
        self.profit_margins = None
        self.eps = None
        self.roe = None
        self.financials_head = None
        self.balance_sheet_head = None
        self.cashflow_head = None
        self.net_profit_margin_str = None
        self.current_ratio_str = None

        # Removed genai configure from here, assuming it's configured globally
        self.model = stock_analysis_model # Use the specific model instance
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')

        self._get_data()
        self._get_financial_data()
        self._calculate_indicators()
        self._update_db_data()

    # --- _get_data, _get_financial_data, _calculate_indicators, _update_db_data, _identify_patterns, _generate_chart, _get_stock_news ---
    # (No changes needed in these methods themselves for temperature)
    def _get_data(self):
        try:
            self.data = self.stock.history(period=self.period)
            if self.data.empty:
                raise ValueError(f"無法取得 {self.ticker} 的資料，請確認股票代碼是否正確")
            company_info = self.stock.info
            self.company_name = company_info.get('longName', self.ticker)
            self.currency = company_info.get('currency', 'TWD' if self.market == 'TW' else 'USD')
            logging.info("成功取得 %s 的股票資料", self.ticker)
        except Exception as e:
            logging.error("取得股票資料時發生錯誤: %s", e)
            raise

    def _get_financial_data(self):
        try:
            info = self.stock.info
            self.pe_ratio = info.get('trailingPE', 'N/A')
            self.market_cap = info.get('marketCap', 'N/A')
            self.forward_pe = info.get('forwardPE', 'N/A')
            self.profit_margins = info.get('profitMargins', 'N/A')
            self.eps = info.get('trailingEps', 'N/A')

            annual_financials = self.stock.financials
            annual_balance_sheet = self.stock.balance_sheet
            annual_cashflow = self.stock.cashflow

            self.financials_head = annual_financials.head().to_string()
            self.balance_sheet_head = annual_balance_sheet.head().to_string()
            self.cashflow_head = annual_cashflow.head().to_string()

            try:
                financials = self.stock.financials
                balance_sheet = self.stock.balance_sheet

                net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
                self.roe = net_income / equity if equity != 0 else 'N/A'

                if "Total Revenue" in annual_financials.index and "Net Income" in annual_financials.index:
                    revenue = annual_financials.loc["Total Revenue"]
                    net_income = annual_financials.loc["Net Income"]
                    net_profit_margin = (net_income / revenue) * 100
                    net_profit_margin_value = net_profit_margin.iloc[0]
                    self.net_profit_margin_str = f"{net_profit_margin_value:.2f}%"
                else:
                    self.net_profit_margin_str = "無法計算（缺少 Total Revenue 或 Net Income 數據）"

                if ("Total Current Assets" in annual_balance_sheet.index and
                    "Total Current Liabilities" in annual_balance_sheet.index):
                    current_assets = annual_balance_sheet.loc["Total Current Assets"]
                    current_liabilities = annual_balance_sheet.loc["Total Current Liabilities"]
                    current_ratio = current_assets / current_liabilities
                    current_ratio_value = current_ratio.iloc[0]
                    self.current_ratio_str = f"{current_ratio_value:.2f}"
                else:
                    self.current_ratio_str = "無法計算（缺少 Total Current Assets 或 Total Current Liabilities 數據）"

            except Exception as inner_e:
                logging.error("計算財務指標時發生錯誤: %s", inner_e)
                self.roe = 'N/A'
                self.net_profit_margin_str = 'N/A'
                self.current_ratio_str = 'N/A'

            logging.info("成功取得 %s 的財務資料", self.ticker)

        except Exception as e:
            logging.error("取得財務資料時發生錯誤: %s", e)
            raise

    def _calculate_indicators(self):
        try:
            df = self.data.copy()
            df['MA5'] = ta.sma(df['Close'], length=5)
            df['MA20'] = ta.sma(df['Close'], length=20)
            df['MA50'] = ta.sma(df['Close'], length=50)
            df['MA120'] = ta.sma(df['Close'], length=120)
            df['MA200'] = ta.sma(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=12)
            macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            df['MACD'] = macd_df['MACD_12_26_9']
            df['MACD_signal'] = macd_df['MACDs_12_26_9']
            df['MACD_hist'] = macd_df['MACDh_12_26_9']
            stoch_df = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3, smooth_k=3)
            df['K'] = stoch_df['STOCHk_9_3_3']
            df['D'] = stoch_df['STOCHd_9_3_3']
            df['J'] = 3 * df['K'] - 2 * df['D']
            bbands = ta.bbands(df['Close'], length=20, std=2)
            df['BB_lower'] = bbands['BBL_20_2.0']
            df['BB_middle'] = bbands['BBM_20_2.0']
            df['BB_upper'] = bbands['BBU_20_2.0']
            df['WMSR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']

            # 計算布林帶寬度
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

            # 計算成交量變化率
            df['Volume_Change'] = df['Volume'].pct_change() * 100

            # 計算價格動量
            df['Momentum'] = df['Close'] - df['Close'].shift(10)

            # 計算波動率 (20日標準差)
            df['Volatility'] = df['Close'].rolling(window=20).std()

            # 計算心理線指標 (PSY)
            df['PSY'] = df['Close'].diff().apply(lambda x: 1 if x > 0 else 0).rolling(12).sum() / 12 * 100

            # 計算乖離率 (BIAS6)
            df['BIAS6'] = (df['Close'] - df['Close'].rolling(window=6).mean()) / df['Close'].rolling(window=6).mean() * 100

            self.data = df
            logging.info("成功計算 %s 的技術指標", self.ticker)
        except Exception as e:
            logging.error("計算技術指標時發生錯誤: %s", e)
            raise

    def _update_db_data(self):
        """更新資料庫中的股票數據"""
        try:
            conn = get_db_connection()
            if not conn:
                logging.error("無法連接資料庫，跳過數據更新")
                return

            cursor = conn.cursor()

            # 檢查股票是否已存在
            cursor.execute("SELECT id FROM stocks WHERE symbol = %s", (self.ticker,))
            result = cursor.fetchone()

            if self.data is None or self.data.empty:
                 logging.warning(f"沒有數據可更新資料庫 {self.ticker}")
                 return

            latest_data = self.data.iloc[-1]

            # 準備數據
            stock_data = {
                'symbol': self.ticker,
                'name': self.company_name,
                'market': self.market,
                'ma5': float(latest_data['MA5']) if not pd.isna(latest_data['MA5']) else None,
                'ma20': float(latest_data['MA20']) if not pd.isna(latest_data['MA20']) else None,
                'ma50': float(latest_data['MA50']) if not pd.isna(latest_data['MA50']) else None,
                'ma120': float(latest_data['MA120']) if not pd.isna(latest_data['MA120']) else None,
                'ma200': float(latest_data['MA200']) if not pd.isna(latest_data['MA200']) else None,
                'bb_upper': float(latest_data['BB_upper']) if not pd.isna(latest_data['BB_upper']) else None,
                'bb_middle': float(latest_data['BB_middle']) if not pd.isna(latest_data['BB_middle']) else None,
                'bb_lower': float(latest_data['BB_lower']) if not pd.isna(latest_data['BB_lower']) else None,
                'rsi': float(latest_data['RSI']) if not pd.isna(latest_data['RSI']) else None,
                'wmsr': float(latest_data['WMSR']) if not pd.isna(latest_data['WMSR']) else None,
                'psy': float(latest_data['PSY']) if not pd.isna(latest_data['PSY']) else None,
                'bias6': float(latest_data['BIAS6']) if not pd.isna(latest_data['BIAS6']) else None,
                'macd': float(latest_data['MACD']) if not pd.isna(latest_data['MACD']) else None,
                'macd_signal': float(latest_data['MACD_signal']) if not pd.isna(latest_data['MACD_signal']) else None,
                'macd_hist': float(latest_data['MACD_hist']) if not pd.isna(latest_data['MACD_hist']) else None,
                'k': float(latest_data['K']) if not pd.isna(latest_data['K']) else None,
                'd': float(latest_data['D']) if not pd.isna(latest_data['D']) else None,
                'j': float(latest_data['J']) if not pd.isna(latest_data['J']) else None,
                'pe_ratio': float(self.pe_ratio) if isinstance(self.pe_ratio, (int, float)) else None,
                'market_cap': int(self.market_cap) if isinstance(self.market_cap, (int, float)) else None,
                'open_price': float(latest_data['Open']) if not pd.isna(latest_data['Open']) else None,
                'close_price': float(latest_data['Close']) if not pd.isna(latest_data['Close']) else None,
                'high_price': float(latest_data['High']) if not pd.isna(latest_data['High']) else None,
                'low_price': float(latest_data['Low']) if not pd.isna(latest_data['Low']) else None,
                'volume': int(latest_data['Volume']) if not pd.isna(latest_data['Volume']) else None
            }

            if result:
                # 更新現有記錄
                update_query = """
                UPDATE stocks SET
                    name = %s, market = %s, last_updated = NOW(),
                    ma5 = %s, ma20 = %s, ma50 = %s, ma120 = %s, ma200 = %s,
                    bb_upper = %s, bb_middle = %s, bb_lower = %s,
                    rsi = %s, wmsr = %s, psy = %s, bias6 = %s,
                    macd = %s, macd_signal = %s, macd_hist = %s,
                    k = %s, d = %s, j = %s,
                    pe_ratio = %s, market_cap = %s,
                    open_price = %s, close_price = %s, high_price = %s, low_price = %s, volume = %s
                WHERE symbol = %s
                """
                cursor.execute(update_query, (
                    stock_data['name'], stock_data['market'],
                    stock_data['ma5'], stock_data['ma20'], stock_data['ma50'], stock_data['ma120'], stock_data['ma200'],
                    stock_data['bb_upper'], stock_data['bb_middle'], stock_data['bb_lower'],
                    stock_data['rsi'], stock_data['wmsr'], stock_data['psy'], stock_data['bias6'],
                    stock_data['macd'], stock_data['macd_signal'], stock_data['macd_hist'],
                    stock_data['k'], stock_data['d'], stock_data['j'],
                    stock_data['pe_ratio'], stock_data['market_cap'],
                    stock_data['open_price'], stock_data['close_price'], stock_data['high_price'], stock_data['low_price'], stock_data['volume'],
                    self.ticker
                ))
            else:
                # 插入新記錄
                insert_query = """
                INSERT INTO stocks (
                    symbol, name, market, last_updated,
                    ma5, ma20, ma50, ma120, ma200,
                    bb_upper, bb_middle, bb_lower,
                    rsi, wmsr, psy, bias6,
                    macd, macd_signal, macd_hist,
                    k, d, j,
                    pe_ratio, market_cap,
                    open_price, close_price, high_price, low_price, volume
                ) VALUES (
                    %s, %s, %s, NOW(),
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s, %s
                )
                """
                cursor.execute(insert_query, (
                    self.ticker, stock_data['name'], stock_data['market'],
                    stock_data['ma5'], stock_data['ma20'], stock_data['ma50'], stock_data['ma120'], stock_data['ma200'],
                    stock_data['bb_upper'], stock_data['bb_middle'], stock_data['bb_lower'],
                    stock_data['rsi'], stock_data['wmsr'], stock_data['psy'], stock_data['bias6'],
                    stock_data['macd'], stock_data['macd_signal'], stock_data['macd_hist'],
                    stock_data['k'], stock_data['d'], stock_data['j'],
                    stock_data['pe_ratio'], stock_data['market_cap'],
                    stock_data['open_price'], stock_data['close_price'], stock_data['high_price'], stock_data['low_price'], stock_data['volume']
                ))

            conn.commit()
            cursor.close()
            conn.close()
            logging.info("成功更新 %s 的資料庫數據", self.ticker)
        except Exception as e:
            logging.error("更新資料庫數據時發生錯誤: %s", e)
            if conn and conn.is_connected():
                 cursor.close()
                 conn.close()

    def _identify_patterns(self, days=30):
        """識別最近的技術形態"""
        try:
            if self.data is None or self.data.empty:
                return ["無法識別形態 (缺少數據)"]
            df = self.data.tail(days).copy()
            if len(df) < 2: return ["數據不足無法識別形態"] # Need at least 2 days for comparison
            patterns = []

            # Ensure required columns exist and have enough data
            required_cols = ['MA5', 'MA20', 'Close', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal', 'K', 'D', 'RSI', 'High', 'Low']
            if not all(col in df.columns for col in required_cols):
                 return ["缺少必要指標無法識別形態"]

            # Check for NaN values in the last two rows for comparisons
            if df[required_cols].iloc[-2:].isnull().values.any():
                return ["近期數據不完整無法識別形態"]


            # 黃金交叉 (MA5 上穿 MA20)
            if (df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]) and (df['MA5'].iloc[-1] > df['MA20'].iloc[-1]):
                patterns.append("黃金交叉 (MA5>MA20)")

            # 死亡交叉 (MA5 下穿 MA20)
            if (df['MA5'].iloc[-2] >= df['MA20'].iloc[-2]) and (df['MA5'].iloc[-1] < df['MA20'].iloc[-1]):
                patterns.append("死亡交叉 (MA5<MA20)")

            # 突破布林帶上軌
            if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1] and df['Close'].iloc[-2] <= df['BB_upper'].iloc[-2]:
                patterns.append("突破布林帶上軌")

            # 跌破布林帶下軌
            if df['Close'].iloc[-1] < df['BB_lower'].iloc[-1] and df['Close'].iloc[-2] >= df['BB_lower'].iloc[-2]:
                patterns.append("跌破布林帶下軌")

            # MACD 金叉
            if (df['MACD'].iloc[-2] <= df['MACD_signal'].iloc[-2]) and (df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1]):
                patterns.append("MACD 金叉")

            # MACD 死叉
            if (df['MACD'].iloc[-2] >= df['MACD_signal'].iloc[-2]) and (df['MACD'].iloc[-1] < df['MACD_signal'].iloc[-1]):
                patterns.append("MACD 死叉")

            # KDJ 金叉
            if (df['K'].iloc[-2] <= df['D'].iloc[-2]) and (df['K'].iloc[-1] > df['D'].iloc[-1]):
                patterns.append("KDJ 金叉")

            # KDJ 死叉
            if (df['K'].iloc[-2] >= df['D'].iloc[-2]) and (df['K'].iloc[-1] < df['D'].iloc[-1]):
                patterns.append("KDJ 死叉")

            # RSI 超買/超賣
            if df['RSI'].iloc[-1] > 75: # Slightly higher threshold
                patterns.append("RSI 超買 (>75)")
            elif df['RSI'].iloc[-1] < 25: # Slightly lower threshold
                patterns.append("RSI 超賣 (<25)")

            # --- Simplified Pattern Checks (Requires more data points) ---
            if len(df) >= 20:
                 # Head & Shoulders Top (Very Simplified)
                 recent_highs = df['High'].rolling(5).max().dropna()
                 if len(recent_highs) >= 4:
                     if (recent_highs.iloc[-4] < recent_highs.iloc[-3] > recent_highs.iloc[-2] < recent_highs.iloc[-1]): # Basic shape check
                         patterns.append("疑似頭肩頂 (簡)")

                 # Head & Shoulders Bottom (Very Simplified)
                 recent_lows = df['Low'].rolling(5).min().dropna()
                 if len(recent_lows) >= 4:
                     if (recent_lows.iloc[-4] > recent_lows.iloc[-3] < recent_lows.iloc[-2] > recent_lows.iloc[-1]): # Basic shape check
                          patterns.append("疑似頭肩底 (簡)")

            if len(df) >= 15:
                 # Double Top (Very Simplified)
                 recent_highs = df['High'].rolling(3).max().dropna()
                 if len(recent_highs) >= 4: # Need enough points for comparison
                     # Check if recent highs are close and there's a dip in between
                     if abs(recent_highs.iloc[-1] - recent_highs.iloc[-3]) / recent_highs.iloc[-3] < 0.03 and recent_highs.iloc[-2] < recent_highs.iloc[-3]:
                         patterns.append("疑似雙頂 (簡)")

                 # Double Bottom (Very Simplified)
                 recent_lows = df['Low'].rolling(3).min().dropna()
                 if len(recent_lows) >= 4:
                     if abs(recent_lows.iloc[-1] - recent_lows.iloc[-3]) / recent_lows.iloc[-3] < 0.03 and recent_lows.iloc[-2] > recent_lows.iloc[-3]:
                         patterns.append("疑似雙底 (簡)")

            return patterns if patterns else ["近期無明顯技術形態"]
        except Exception as e:
            logging.error("識別技術形態時發生錯誤: %s", e)
            return ["無法識別技術形態"]

    def _generate_chart(self, days=180):
        """生成股票走勢圖"""
        try:
            if self.data is None or self.data.empty:
                raise ValueError("無法生成圖表，缺少數據")

            df = self.data.tail(days).copy()
            if df.empty:
                 raise ValueError(f"過去 {days} 天內沒有數據，無法生成圖表")


            # 創建子圖
            fig = make_subplots(rows=4, cols=1,
                               shared_xaxes=True,
                               vertical_spacing=0.05,
                               row_heights=[0.5, 0.15, 0.15, 0.2]) # Adjusted row heights slightly

            # 添加K線圖
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='K線'
            ), row=1, col=1)

            # 添加移動平均線 (Check if column exists before adding)
            if 'MA5' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5', line=dict(color='orange', width=1)), row=1, col=1)
            if 'MA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue', width=1)), row=1, col=1)
            if 'MA120' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA120'], name='MA120', line=dict(color='green', width=1)), row=1, col=1)

            # 添加布林帶 (Check if columns exist)
            if all(c in df.columns for c in ['BB_upper', 'BB_middle', 'BB_lower']):
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='布林上軌', line=dict(color='rgba(173, 204, 255, 0.7)', width=1), fill=None), row=1, col=1) # Changed fill to None for clarity
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='布林下軌', line=dict(color='rgba(173, 204, 255, 0.7)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(173, 204, 255, 0.1)'), row=1, col=1) # Fill between upper and lower
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='布林中軌', line=dict(color='rgba(255, 255, 255, 0.5)', width=1, dash='dot')), row=1, col=1) # Make middle line distinct

            # 添加成交量 (Check if Volume exists)
            if 'Volume' in df.columns and 'Close' in df.columns and 'Open' in df.columns:
                 colors = ['#2ca02c' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#d62728' for i in range(len(df))] # More standard red/green
                 fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors, marker_line_width=0), row=2, col=1)

            # 添加MACD (Check if columns exist)
            if all(c in df.columns for c in ['MACD', 'MACD_signal', 'MACD_hist']):
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='MACD信號', line=dict(color='red', width=1)), row=3, col=1)
                colors_macd = ['#2ca02c' if val >= 0 else '#d62728' for val in df['MACD_hist']]
                fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD柱', marker_color=colors_macd, marker_line_width=0), row=3, col=1)

            # 添加KDJ (Check if columns exist)
            if all(c in df.columns for c in ['K', 'D', 'J']):
                fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K值', line=dict(color='blue', width=1)), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D值', line=dict(color='red', width=1)), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['J'], name='J值', line=dict(color='green', width=1)), row=4, col=1)
                # Add horizontal lines for KDJ thresholds
                fig.add_hline(y=80, line_dash="dot", line_color="grey", line_width=1, row=4, col=1)
                fig.add_hline(y=20, line_dash="dot", line_color="grey", line_width=1, row=4, col=1)


            # 更新佈局
            fig.update_layout(
                title=f'{self.company_name} ({self.ticker}) 技術分析圖 ({days}天)',
                xaxis_rangeslider_visible=False,
                template='plotly_dark',
                height=750, # Reduced height slightly
                # width=1000, # Let it be responsive if possible
                margin=dict(l=40, r=40, t=80, b=40), # Reduced margins
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.2)',
                font=dict(color='white', size=11), # Slightly smaller font
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Move legend to top
            )

            # 更新Y軸格式 and range
            fig.update_yaxes(title_text='價格', row=1, col=1, title_font_size=10, tickfont_size=9)
            fig.update_yaxes(title_text='成交量', row=2, col=1, title_font_size=10, tickfont_size=9)
            fig.update_yaxes(title_text='MACD', row=3, col=1, title_font_size=10, tickfont_size=9, zeroline=True, zerolinewidth=1, zerolinecolor='grey')
            fig.update_yaxes(title_text='KDJ', row=4, col=1, title_font_size=10, tickfont_size=9, range=[0, 100]) # Set KDJ range

            # Update X axis format
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')


            # 保存圖表
            chart_filename = f"{self.ticker.replace('.', '_')}_chart_{secrets.token_hex(4)}.html" # Add random hex to avoid caching issues
            chart_path = f"static/charts/{chart_filename}"
            fig.write_html(chart_path, full_html=False, include_plotlyjs='cdn') # Use CDN for smaller file size
            logging.info(f"圖表已生成: {chart_path}")

            return chart_path # Return the relative path for use in HTML src
        except Exception as e:
            logging.error("生成圖表時發生錯誤: %s", e)
            # Don't re-raise, return None or an error indicator
            return None


    def _get_stock_news(self, max_news=5):
        """獲取相關股票新聞"""
        try:
            # Use company name for better relevance, fall back to ticker if needed
            search_term = f"{self.company_name or self.ticker} stock" if self.market == "US" else f"{self.company_name or self.ticker} 股票"
            rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_term)}&hl={'en-US' if self.market == 'US' else 'zh-TW'}&gl={'US' if self.market == 'US' else 'TW'}&ceid={'US:en' if self.market == 'US' else 'TW:zh-Hant'}"

            feed = feedparser.parse(rss_url)
            news_list = []

            if not feed.entries:
                logging.warning(f"找不到 '{search_term}' 的 Google 新聞")
                return []

            for entry in feed.entries[:max_news]:
                try:
                    # Safely parse published time
                    published_time = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                       try:
                           published_time = dt.datetime(*entry.published_parsed[:6])
                       except ValueError:
                            logging.warning(f"無法解析新聞日期: {entry.published_parsed}")
                            published_time = dt.datetime.now() # Fallback
                    else:
                       published_time = dt.datetime.now() # Fallback


                    # Sentiment analysis
                    sentiment_label = 'Neutral'
                    sentiment_score = 0.5
                    if entry.title: # Only analyze if title exists
                        try:
                            result = self.sentiment_analyzer(entry.title)[0]
                            sentiment_label = result['label']
                            sentiment_score = result['score']
                        except Exception as sentiment_err:
                             logging.warning(f"Finbert sentiment analysis failed for '{entry.title}': {sentiment_err}")


                    news_entry = {
                        'title': entry.title or "無標題",
                        'link': entry.link or "#",
                        'date': published_time.strftime('%Y-%m-%d'),
                        'source': entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else 'Google News',
                        'sentiment': sentiment_label,
                        'sentiment_score': sentiment_score
                    }

                    news_list.append(news_entry)
                except Exception as inner_e:
                    logging.error(f"處理新聞條目 '{entry.title if hasattr(entry, 'title') else 'N/A'}' 時發生錯誤: {inner_e}")
                    continue

            return news_list
        except Exception as e:
            logging.error(f"獲取股票新聞時發生錯誤 ({self.ticker}): {e}")
            return []

    # --- _get_ai_analysis (Use temperature) ---
    def _get_ai_analysis(self):
        """使用Gemini生成AI分析，並使用設定的 temperature"""
        try:
            if self.data is None or len(self.data) < 23: # Need enough data for comparison
                 return f"無法為 {self.company_name} ({self.ticker}) 生成 AI 分析報告，數據不足。"

            latest = self.data.iloc[-1]
            prev_day = self.data.iloc[-2]
            prev_week = self.data.iloc[-6]
            prev_month = self.data.iloc[-23]

            # 計算各種漲跌幅
            daily_change = ((latest['Close'] - prev_day['Close']) / prev_day['Close']) * 100 if prev_day['Close'] else 0
            weekly_change = ((latest['Close'] - prev_week['Close']) / prev_week['Close']) * 100 if prev_week['Close'] else 0
            monthly_change = ((latest['Close'] - prev_month['Close']) / prev_month['Close']) * 100 if prev_month['Close'] else 0

            # 獲取技術形態
            patterns = self._identify_patterns()
            patterns_str = ", ".join(patterns) if patterns else "近期無明顯技術形態"

            # Format numbers carefully to avoid errors with N/A
            def format_val(val, precision=2, is_currency=False, is_percent=False):
                if val is None or val == 'N/A' or pd.isna(val): return "N/A"
                try:
                    num = float(val)
                    if is_percent: return f"{num:.{precision}f}%"
                    if is_currency: return f"{num:,.{precision}f}"
                    # Basic number formatting for large numbers (optional)
                    if abs(num) >= 1e9: return f"{num/1e9:.{precision}f}B"
                    if abs(num) >= 1e6: return f"{num/1e6:.{precision}f}M"
                    return f"{num:.{precision}f}"
                except (ValueError, TypeError):
                    return str(val) # Return original string if conversion fails

            prompt = f"""
            你是一位頂尖的股票分析師，請針對以下股票數據，生成一份專業、精簡、結構化的分析報告。請使用繁體中文，並盡可能以 **條列式 (bullet points)** 或 **表格** 的方式呈現各個分析要點。重點是提供洞見，而不僅是重複數據。

            **股票基本資料：**
            *   股票名稱: {self.company_name or 'N/A'}
            *   股票代碼: {self.ticker}
            *   市場: {'台股' if self.market == 'TW' else '美股'}
            *   當前價格: {format_val(latest['Close'], is_currency=True)} {self.currency}
            *   日漲跌幅: {format_val(daily_change, is_percent=True)}
            *   週漲跌幅: {format_val(weekly_change, is_percent=True)}
            *   月漲跌幅: {format_val(monthly_change, is_percent=True)}

            **關鍵數據：**
            *   技術指標: RSI={format_val(latest.get('RSI'))}, MACD={format_val(latest.get('MACD'), precision=4)}, K={format_val(latest.get('K'))}, D={format_val(latest.get('D'))}
            *   布林帶: 上軌={format_val(latest.get('BB_upper'))}, 中軌={format_val(latest.get('BB_middle'))}, 下軌={format_val(latest.get('BB_lower'))}
            *   近期技術形態: {patterns_str}
            *   基本面: P/E={format_val(self.pe_ratio)},
                      市值={format_val(self.market_cap, is_currency=True, precision=0)},
                      EPS={format_val(self.eps, is_currency=True)},
                      ROE={format_val(self.roe, is_percent=True) if not isinstance(self.roe, str) else self.roe},
                      淨利率={self.net_profit_margin_str or 'N/A'},
                      流動比率={self.current_ratio_str or 'N/A'}

            **請根據以上數據，生成包含以下部分的分析報告 (使用條列式或表格)：**

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
               *   盈利能力: (評估 ROE, 淨利率水平)
               *   估值水平: (評估 P/E 相對歷史或同業)
               *   財務健康: (評估流動比率等，可選)

            **6. 總結與展望:**
               *   綜合評價與未來潛力 (結合技術與基本面)
               *   短期操作建議參考 (例如：逢低佈局、突破追買、保持觀望、風險控制)
               *   **免責聲明:** 此分析僅供參考，不構成投資建議。

            **要求：**
            *   分析需客觀、基於數據，提出明確觀點。
            *   語言精煉，格式清晰。
            *   不要只羅列數據，要解釋數據代表的意義。
            """

            # 設定生成參數，使用 self.temperature
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature
            )

            logging.info(f"為 {self.ticker} 生成 AI 分析，溫度: {self.temperature}")
            response = self.model.generate_content(prompt, generation_config=generation_config)

            analysis = response.text

            # Basic Markdown cleanup (optional)
            # analysis = analysis.replace('\n\n', '\n').replace('*   ', '* ')

            return analysis
        except Exception as e:
            logging.error(f"生成 AI 分析 ({self.ticker}) 時發生錯誤: {e}")
            return f"無法為 {self.company_name} ({self.ticker}) 生成 AI 分析報告。\n錯誤詳情: {str(e)}"

    # --- get_stock_summary (Ensure chart_path handling is robust) ---
    def get_stock_summary(self):
        """獲取股票綜合分析"""
        try:
            if self.data is None or self.data.empty or len(self.data) < 2:
                raise ValueError("數據不足，無法生成綜合分析")

            latest = self.data.iloc[-1]
            prev = self.data.iloc[-2]

            # 計算漲跌幅
            price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100 if prev['Close'] else 0
            price_change_str = f"{price_change:+.2f}%" # Add sign explicitly

            patterns = self._identify_patterns()
            chart_path = self._generate_chart() # This now returns None on error
            news = self._get_stock_news()
            analysis = self._get_ai_analysis()

            # Prepare summary dictionary carefully, handling potential None/NaN values
            summary = {
                "ticker": self.ticker,
                "company_name": self.company_name or 'N/A',
                "currency": self.currency or 'N/A',
                "current_price": latest.get('Close', 0),
                "price_change": price_change_str,
                "price_change_value": price_change,
                "volume": int(latest.get('Volume', 0)),
                "pe_ratio": self.pe_ratio if self.pe_ratio != 'N/A' else None,
                "market_cap": self.market_cap if self.market_cap != 'N/A' else None,
                "eps": self.eps if self.eps != 'N/A' else None,
                "roe": self.roe if isinstance(self.roe, (int, float)) else None, # Store raw value if possible
                "net_profit_margin": self.net_profit_margin_str or 'N/A',
                "current_ratio": self.current_ratio_str or 'N/A',
                "rsi": latest.get('RSI'),
                "macd": latest.get('MACD'),
                "macd_signal": latest.get('MACD_signal'),
                "k": latest.get('K'),
                "d": latest.get('D'),
                "j": latest.get('J'),
                "patterns": patterns,
                "chart_path": chart_path, # This might be None now
                "news": news,
                "analysis": analysis
            }
             # Format ROE for display later if needed, but store raw if possible
            if isinstance(summary["roe"], (int, float)):
                 summary["roe_display"] = f"{summary['roe']:.2%}"
            elif isinstance(self.roe, str):
                 summary["roe_display"] = self.roe # Keep original string if calculation failed
            else:
                 summary["roe_display"] = "N/A"


            return summary
        except Exception as e:
            logging.error("獲取股票綜合分析 (%s) 時發生錯誤: %s", self.ticker, e)
            # Return a structure indicating failure
            return {
                "ticker": self.ticker,
                "company_name": self.company_name or self.ticker,
                "error": f"獲取綜合分析失敗: {str(e)}"
            }


# ------------------ API 路由 ------------------

@app.route('/')
def index():
    return render_template('index.html')

# --- /analysis/<ticker> (No changes needed) ---
@app.route('/analysis/<ticker>')
def analysis(ticker):
    market = request.args.get('market', 'TW')
    return render_template('analysis.html', ticker=ticker, market=market)

# --- /api/analyze (Use user's temperature setting) ---
@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()
        market = data.get('market', 'TW')

        if not ticker:
            return jsonify({'error': '請提供股票代碼'}), 400

        # --- Get user settings (including temperature) ---
        user_id = 1 # Assume user_id 1 for now
        settings = get_user_settings(user_id)
        user_temperature = settings.get('temperature', 0.7) # Default if missing
        # -------------------------------------------------

        # Pass temperature to StockAnalyzer
        analyzer = StockAnalyzer(
            ticker,
            GEMINI_API_KEY,
            period="5y",
            market=market,
            temperature=user_temperature # Pass the fetched temperature
        )
        summary = analyzer.get_stock_summary()

        # Check if summary generation failed
        if 'error' in summary:
             return jsonify({'error': summary['error']}), 500


        return jsonify({
            'success': True,
            'data': summary
        })
    except ValueError as e: # Catch specific errors from StockAnalyzer
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error("分析股票時發生錯誤: %s", e, exc_info=True) # Log traceback
        return jsonify({'error': f"分析股票時發生意外錯誤: {str(e)}"}), 500

# --- /api/chat (Use user's temperature setting) ---
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        market = data.get('market', 'TW')

        if not message:
            return jsonify({'error': '請輸入訊息'}), 400

        # --- Get user settings (including temperature) ---
        user_id = 1 # Assume user_id 1 for now
        settings = get_user_settings(user_id)
        user_temperature = settings.get('temperature', 0.7)
        # -------------------------------------------------

        # Prepare generation config for Gemini calls
        generation_config = genai.types.GenerationConfig(
             temperature=user_temperature
        )

        # Check for stock query
        stock_match = re.search(r'[#＃]([0-9A-Za-z\.]+)', message)
        if stock_match:
            ticker = stock_match.group(1)
            try:
                # Pass temperature to StockAnalyzer
                analyzer = StockAnalyzer(
                    ticker,
                    GEMINI_API_KEY,
                    period="5y",
                    market=market,
                    temperature=user_temperature # Pass fetched temperature
                )
                summary = analyzer.get_stock_summary()

                # Check for errors during analysis
                if 'error' in summary:
                    return jsonify({
                        'success': True,
                        'type': 'text',
                        'data': f"分析股票 {ticker} 時發生錯誤: {summary['error']}"
                    })

                # Add to watchlist logic (remains the same)
                conn = get_db_connection()
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute('SELECT id FROM watchlist WHERE user_id = %s AND ticker = %s', (user_id, analyzer.ticker))
                        if not cursor.fetchone():
                            cursor.execute('INSERT INTO watchlist (user_id, ticker, name) VALUES (%s, %s, %s)', (user_id, analyzer.ticker, analyzer.company_name))
                            conn.commit()
                        cursor.close()
                        conn.close()
                    except Exception as db_err:
                         logging.error(f"更新追蹤清單時出錯 ({ticker}): {db_err}")
                         if conn and conn.is_connected():
                            cursor.close()
                            conn.close()
                         # Continue returning analysis even if watchlist fails


                return jsonify({
                    'success': True,
                    'type': 'stock',
                    'data': summary
                })
            except ValueError as e: # Specific errors like invalid ticker
                 logging.warning(f"股票查詢錯誤 ({ticker}): {e}")
                 return jsonify({'success': True, 'type': 'text', 'data': str(e)})
            except Exception as e:
                logging.error(f"處理股票查詢時發生錯誤 ({ticker}): {e}", exc_info=True)
                return jsonify({
                    'success': True,
                    'type': 'text',
                    'data': f"無法查詢股票 {ticker}。錯誤: {str(e)}"
                })

        # Check for portfolio optimization request
        if re.search(r'(投資|投組|資產|配置|組合).*(優化|建議|推薦|分配)', message):
            try:
                # Apply temperature to portfolio model too? Maybe keep this one more stable?
                # Let's apply it for consistency for now.
                portfolio_generation_config = genai.types.GenerationConfig(temperature=user_temperature)

                prompt = f"""
                使用者想要投資組合優化建議。請先簡短詢問他們的風險承受度 (例如：保守、穩健、積極)、投資期限 (例如：1-3年、3-5年、5年以上) 和投資目標 (例如：資本增值、穩定收益、兩者平衡)，然後提供一個 **範例** 資產配置建議。強調這只是範例，實際建議需要更多資訊。

                使用者的原始訊息: {message}

                請用繁體中文回覆，提供一個包含股票/債券/現金等比例的範例配置。
                """
                response = portfolio_model.generate_content(prompt, generation_config=portfolio_generation_config)

                return jsonify({
                    'success': True,
                    'type': 'text',
                    'data': response.text
                })
            except Exception as e:
                logging.error(f"處理投資組合優化請求時發生錯誤: {e}", exc_info=True)
                return jsonify({
                    'success': True,
                    'type': 'text',
                    'data': "抱歉，處理投資組合建議時遇到問題，請稍後再試。"
                })

        # General chat query
        try:
            market_str = "台股" if market == "TW" else "美股"
            prompt = f"""
            你是一位專業的股票分析師和投資顧問，專精於{market_str}市場分析。請以專業、友善且樂於助人的語氣回答。

            使用者的訊息: {message}

            請用繁體中文回覆。如果使用者詢問特定股票，提醒他們可以使用 `#股票代碼` 的格式來查詢詳細分析。
            如果訊息模糊不清，可以禮貌地請使用者提供更多細節。
            """
            # Use the main generation_config with user's temperature
            response = general_model.generate_content(prompt, generation_config=generation_config)

            return jsonify({
                'success': True,
                'type': 'text',
                'data': response.text
            })
        except Exception as e:
            logging.error(f"使用Gemini處理一般訊息時發生錯誤: {e}", exc_info=True)
            return jsonify({
                'success': True,
                'type': 'text',
                'data': "抱歉，我目前無法處理您的請求，請稍後再試或換個方式提問。"
            })
    except Exception as e:
        logging.error("處理聊天請求時發生意外錯誤: %s", e, exc_info=True)
        return jsonify({'error': f"處理聊天請求時發生意外錯誤: {str(e)}"}), 500


# --- /api/watchlist GET/POST/REMOVE (No changes needed for temperature) ---
@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    try:
        user_id = 1 # Assume user_id 1
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '無法連接資料庫'}), 500

        cursor = conn.cursor(dictionary=True)
        cursor.execute('''
            SELECT w.ticker, w.name, s.close_price, s.open_price, s.last_updated
            FROM watchlist w
            LEFT JOIN stocks s ON w.ticker = s.symbol
            WHERE w.user_id = %s
            ORDER BY w.added_at DESC
        ''', (user_id,))
        watchlist_items = cursor.fetchall()
        cursor.close()
        conn.close()

        watchlist_processed = []
        max_update_age = timedelta(days=1) # How old can DB data be?

        for item in watchlist_items:
            ticker = item['ticker']
            name = item['name']
            price = 0
            change = 0
            source = "N/A"

            # Try using DB data if recent and valid
            if item['close_price'] is not None and item['open_price'] is not None and item['last_updated'] is not None and (datetime.now() - item['last_updated']) < max_update_age :
                 price = item['close_price']
                 # Use open price for daily change if available, otherwise fallback needed
                 change = ((item['close_price'] - item['open_price']) / item['open_price']) * 100 if item['open_price'] else 0
                 source = "DB"
                 watchlist_processed.append({
                    'ticker': ticker, 'name': name, 'price': price, 'change': change, 'source': source
                 })
                 continue # Skip yfinance fetch if DB data is good

            # Fallback to yfinance if DB data is old, missing, or invalid
            try:
                stock = yf.Ticker(ticker)
                # Fetch last 2 days to calculate change
                hist = stock.history(period="2d")
                if not hist.empty and len(hist) >= 2:
                    current = hist.iloc[-1]
                    prev = hist.iloc[-2] # Use previous day's close for change calc
                    price = current['Close']
                    change = ((current['Close'] - prev['Close']) / prev['Close']) * 100 if prev['Close'] else 0
                    source = "Yahoo"
                elif not hist.empty: # Only 1 day data?
                    price = hist.iloc[-1]['Close']
                    change = 0 # Cannot calculate change
                    source = "Yahoo (1d)"
                else:
                    logging.warning(f"無法從 yfinance 取得 {ticker} 的數據")
                    price = 0
                    change = 0
                    source = "Error"

                watchlist_processed.append({
                    'ticker': ticker, 'name': name, 'price': price, 'change': change, 'source': source
                })

            except Exception as e:
                logging.error(f"獲取追蹤清單股票 {ticker} 數據時出錯: {e}")
                watchlist_processed.append({
                    'ticker': ticker, 'name': name, 'price': 0, 'change': 0, 'error': True, 'source': "Fetch Error"
                })

        return jsonify({'watchlist': watchlist_processed})
    except Exception as e:
        logging.error(f"獲取追蹤清單時發生錯誤: {e}", exc_info=True)
        return jsonify({'error': f"獲取追蹤清單時發生錯誤: {str(e)}"}), 500


@app.route('/api/watchlist/add', methods=['POST'])
def add_to_watchlist():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip().upper() # Standardize
        name = data.get('name', '').strip() # Optional name from frontend
        user_id = 1 # Assume user_id 1

        if not ticker:
            return jsonify({'error': '請提供股票代碼'}), 400

        # Attempt to fetch name from yfinance if not provided
        fetched_name = name
        if not fetched_name:
            try:
                stock_info = yf.Ticker(ticker).info
                fetched_name = stock_info.get('longName', stock_info.get('shortName', ticker))
                logging.info(f"Fetched name for {ticker}: {fetched_name}")
            except Exception as e:
                 logging.warning(f"無法從 yfinance 獲取 {ticker} 的名稱: {e}. 使用代碼作為名稱。")
                 fetched_name = ticker # Fallback to ticker

        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '無法連接資料庫'}), 500

        try:
            cursor = conn.cursor()
            # Use INSERT IGNORE or ON DUPLICATE KEY UPDATE for atomicity
            # Using INSERT IGNORE here for simplicity
            cursor.execute('''
                INSERT IGNORE INTO watchlist (user_id, ticker, name)
                VALUES (%s, %s, %s)
            ''', (user_id, ticker, fetched_name))

            affected_rows = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()

            if affected_rows > 0:
                return jsonify({'success': True, 'message': f'已將 {fetched_name} ({ticker}) 加入追蹤清單'})
            else:
                return jsonify({'success': True, 'message': f'{fetched_name} ({ticker}) 已在追蹤清單中'})

        except Exception as e:
            logging.error(f"添加到追蹤清單時發生錯誤 ({ticker}): {e}", exc_info=True)
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
            return jsonify({'error': f"添加到追蹤清單時發生錯誤: {str(e)}"}), 500

    except Exception as e:
        logging.error(f"處理加入追蹤清單請求時發生意外錯誤: {e}", exc_info=True)
        return jsonify({'error': f"處理請求時發生意外錯誤: {str(e)}"}), 500

@app.route('/api/watchlist/remove', methods=['POST'])
def remove_from_watchlist():
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').strip()
        user_id = 1 # Assume user_id 1

        if not ticker:
            return jsonify({'error': '請提供股票代碼'}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '無法連接資料庫'}), 500

        try:
            cursor = conn.cursor()
            # Get name first for message, though not strictly necessary
            cursor.execute('SELECT name FROM watchlist WHERE user_id = %s AND ticker = %s', (user_id, ticker))
            result = cursor.fetchone()
            name = result[0] if result else ticker

            # Delete the item
            cursor.execute('DELETE FROM watchlist WHERE user_id = %s AND ticker = %s', (user_id, ticker))
            affected_rows = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()

            if affected_rows > 0:
                 return jsonify({'success': True, 'message': f'已從追蹤清單中移除 {name}'})
            else:
                 return jsonify({'success': False, 'error': f'{ticker} 不在追蹤清單中'}), 404 # Return 404 if not found


        except Exception as e:
            logging.error(f"從追蹤清單移除時發生錯誤 ({ticker}): {e}", exc_info=True)
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
            return jsonify({'error': f"從追蹤清單移除時發生錯誤: {str(e)}"}), 500

    except Exception as e:
        logging.error(f"處理移除追蹤清單請求時發生意外錯誤: {e}", exc_info=True)
        return jsonify({'error': f"處理請求時發生意外錯誤: {str(e)}"}), 500


# --- /api/portfolio/optimize (Apply temperature, update prompt slightly) ---
@app.route('/api/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    try:
        data = request.get_json()
        risk_level = data.get('risk_level', 'moderate')
        investment_amount = data.get('investment_amount', 100000) # Corrected default
        investment_period = data.get('investment_period', '1-5')
        investment_goal = data.get('investment_goal', 'growth')
        user_id = 1 # Assume user_id 1
        settings = get_user_settings(user_id)
        user_temperature = settings.get('temperature', 0.7) # Use user's temp

        portfolio_generation_config = genai.types.GenerationConfig(temperature=user_temperature)


        # Simplified prompt focusing on asset allocation example
        prompt = f"""
        請根據以下投資者信息提供一個 **範例** 的投資組合配置建議：

        風險承受度：{risk_level}（conservative/moderate/aggressive）
        投資金額：{investment_amount} TWD
        投資期限：{investment_period} 年
        投資目標：{investment_goal}（income/growth/balanced）

        請提供：
        1.  一個清晰的資產類別配置 **比例** (例如：股票 X%, 債券 Y%, 現金 Z%)。
        2.  簡單說明此配置適合此類投資者的 **原因**。
        3.  **提醒** 這僅為範例，實際投資應考慮更多因素並諮詢專家。

        回答需簡潔明瞭，使用繁體中文。
        """

        response = portfolio_model.generate_content(prompt, generation_config=portfolio_generation_config)
        portfolio_suggestion = response.text

        # --- Chart Generation (Keep as is, using regex) ---
        try:
            stocks_match = re.search(r'(?:股票|權益)[：:\s]*(\d+)%', portfolio_suggestion)
            bonds_match = re.search(r'債券[：:\s]*(\d+)%', portfolio_suggestion)
            cash_match = re.search(r'(?:現金|貨幣市場)[：:\s]*(\d+)%', portfolio_suggestion)
            # More flexible regex for "Other"
            other_match = re.search(r'(?:其他|另類|黃金|房地產)[資產：:\s]*(\d+)%', portfolio_suggestion)


            stocks = int(stocks_match.group(1)) if stocks_match else 60
            bonds = int(bonds_match.group(1)) if bonds_match else 30
            cash = int(cash_match.group(1)) if cash_match else 10
            other = int(other_match.group(1)) if other_match else 0


            # Normalize to 100%
            allocation = {'股票': stocks, '債券': bonds, '現金': cash}
            if other > 0: allocation['其他'] = other

            total = sum(allocation.values())
            if total == 0: # Avoid division by zero if regex fails completely
                 allocation = {'股票': 60, '債券': 30, '現金': 10} # Fallback
                 total = 100

            if total != 100:
                 factor = 100.0 / total
                 allocation = {k: int(round(v * factor)) for k, v in allocation.items()}
                 # Adjust rounding errors to ensure sum is exactly 100
                 current_sum = sum(allocation.values())
                 diff = 100 - current_sum
                 # Add difference to the largest category (usually stocks)
                 if diff != 0:
                    largest_cat = max(allocation, key=allocation.get)
                    allocation[largest_cat] += diff


            labels = list(allocation.keys())
            values = list(allocation.values())

            # Define consistent colors
            color_map = {'股票': '#0066ff', '債券': '#00cc88', '現金': '#ffcc00', '其他': '#ff6b6b'}
            colors = [color_map.get(label, '#cccccc') for label in labels] # Use grey for unknown


            fig = px.pie(
                values=values,
                names=labels,
                color_discrete_sequence=colors,
                title=f"範例投資組合配置 - {risk_level.capitalize()}"
            )

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=11),
                margin=dict(l=20, r=20, t=50, b=20), # Reduced margin
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5) # Legend below
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', insidetextorientation='radial')


            chart_id = secrets.token_hex(8)
            chart_filename = f"portfolio_{chart_id}.html"
            chart_path = f"static/charts/{chart_filename}"
            fig.write_html(chart_path, full_html=False, include_plotlyjs='cdn')

            # Return allocation percentages for potential frontend use
            final_allocation_percent = {k.lower(): v for k, v in allocation.items()}


            return jsonify({
                'success': True,
                'suggestion': portfolio_suggestion,
                'chart_path': chart_path,
                'allocation': final_allocation_percent # Return processed allocation
            })
        except Exception as chart_error:
            logging.error(f"生成投資組合圖表時發生錯誤: {chart_error}", exc_info=True)
            return jsonify({
                'success': True,
                'suggestion': portfolio_suggestion,
                'chart_error': f"無法生成圖表: {str(chart_error)}"
            })
    except Exception as e:
        logging.error(f"優化投資組合時發生錯誤: {e}", exc_info=True)
        return jsonify({'error': f"優化投資組合時發生錯誤: {str(e)}"}), 500

# --- /api/settings GET (Include temperature) ---
@app.route('/api/settings', methods=['GET'])
def get_settings():
    try:
        user_id = 1 # Assume user_id 1
        settings = get_user_settings(user_id) # Use the helper function
        return jsonify({'settings': settings})
    except Exception as e:
        logging.error(f"獲取設定時發生錯誤: {e}", exc_info=True)
        return jsonify({'error': f"獲取設定時發生錯誤: {str(e)}"}), 500

# --- /api/settings POST (Include temperature) ---
@app.route('/api/settings', methods=['POST'])
def update_settings():
    try:
        data = request.get_json()
        user_id = 1 # Assume user_id 1

        # --- Validate incoming data ---
        errors = {}
        if not isinstance(data.get('dark_mode'), bool): errors['dark_mode'] = '必須是布爾值'
        if data.get('font_size') not in ['small', 'medium', 'large']: errors['font_size'] = '無效的字體大小'
        if not isinstance(data.get('price_alert'), bool): errors['price_alert'] = '必須是布爾值'
        if not isinstance(data.get('market_summary'), bool): errors['market_summary'] = '必須是布爾值'
        if data.get('data_source') not in ['default', 'yahoo', 'alpha']: errors['data_source'] = '無效的資料來源'
        try:
            temp = float(data.get('temperature', 0.7)) # Get temperature
            if not (0.0 <= temp <= 1.0):
                errors['temperature'] = '溫度必須介於 0.0 和 1.0 之間'
            else:
                # Store the validated float value
                data['temperature'] = temp
        except (ValueError, TypeError):
             errors['temperature'] = '溫度必須是有效的數字'


        if errors:
            return jsonify({'error': '無效的設定值', 'details': errors}), 400
        # -----------------------------

        conn = get_db_connection()
        if not conn:
            return jsonify({'error': '無法連接資料庫'}), 500

        try:
            cursor = conn.cursor()

            # Check if settings exist, then INSERT or UPDATE
            cursor.execute('SELECT id FROM settings WHERE user_id = %s', (user_id,))
            exists = cursor.fetchone()

            if exists:
                sql = '''
                    UPDATE settings
                    SET dark_mode = %s, font_size = %s, price_alert = %s,
                        market_summary = %s, data_source = %s, temperature = %s
                    WHERE user_id = %s
                '''
                params = (
                    data['dark_mode'], data['font_size'], data['price_alert'],
                    data['market_summary'], data['data_source'], data['temperature'],
                    user_id
                )
            else:
                sql = '''
                    INSERT INTO settings
                    (user_id, dark_mode, font_size, price_alert, market_summary, data_source, temperature)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                '''
                params = (
                    user_id,
                    data['dark_mode'], data['font_size'], data['price_alert'],
                    data['market_summary'], data['data_source'], data['temperature']
                )

            cursor.execute(sql, params)
            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({'success': True, 'message': '設定已更新'})
        except Exception as e:
            logging.error(f"更新設定時資料庫操作錯誤: {e}", exc_info=True)
            if conn and conn.is_connected():
                cursor.close()
                conn.close()
            return jsonify({'error': f"更新設定時發生錯誤: {str(e)}"}), 500

    except Exception as e:
        logging.error(f"處理更新設定請求時發生意外錯誤: {e}", exc_info=True)
        return jsonify({'error': f"處理請求時發生意外錯誤: {str(e)}"}), 500


# --- /api/market_news & fetch_market_news (No changes needed for temperature) ---
@app.route('/api/market_news', methods=['GET'])
def get_market_news():
    try:
        market = request.args.get('market', 'TW')
        category = request.args.get('category', 'general') # Keep category option
        news_list = fetch_market_news(market, category) # Pass category
        return jsonify({'news': news_list})
    except Exception as e:
        logging.error(f"獲取市場新聞時發生錯誤: {e}", exc_info=True)
        return jsonify({'error': f"獲取市場新聞時發生錯誤: {str(e)}"}), 500

def fetch_market_news(market, category):
    """獲取市場新聞"""
    try:
        # Base search term
        if market == 'TW':
            base_term = "台股"
            lang = 'zh-TW'
            loc = 'TW'
            ceid = 'TW:zh-Hant'
        else:
            base_term = "US stock market"
            lang = 'en-US'
            loc = 'US'
            ceid = 'US:en'

        # Add category keywords
        category_terms = {
            'tech': " 科技" if market == 'TW' else " tech",
            'finance': " 金融" if market == 'TW' else " finance",
            'industry': " 產業" if market == 'TW' else " economy", # Broaden US term
            'general': "" # No extra term for general
        }
        search_term = base_term + category_terms.get(category, "")

        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_term)}&hl={lang}&gl={loc}&ceid={ceid}"
        logging.info(f"Fetching market news from: {rss_url}")

        feed = feedparser.parse(rss_url)
        news_list = []

        if not feed.entries:
             logging.warning(f"找不到 '{search_term}' (market={market}, category={category}) 的 Google 新聞")
             return []


        # Initialize sentiment analyzer once if needed frequently
        sentiment_analyzer = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')

        for entry in feed.entries[:15]: # Limit to 15 news items
            try:
                # --- Safely parse time ---
                published_time = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        published_time = dt.datetime(*entry.published_parsed[:6])
                    except ValueError:
                        logging.warning(f"無法解析新聞日期: {entry.published_parsed}")
                        published_time = dt.datetime.now()
                else:
                    published_time = dt.datetime.now()
                # -------------------------

                # --- Sentiment Analysis ---
                sentiment_label = 'Neutral'
                sentiment_score = 0.5
                title = entry.title if hasattr(entry, 'title') else ""
                if title:
                    try:
                        result = sentiment_analyzer(title)[0]
                        sentiment_label = result['label']
                        sentiment_score = result['score']
                    except Exception as sentiment_err:
                        logging.warning(f"Finbert sentiment analysis failed for '{title}': {sentiment_err}")
                # -------------------------

                # --- Summary Extraction ---
                summary = ""
                if hasattr(entry, 'summary'):
                    summary = entry.summary
                    # Basic HTML tag removal
                    summary = re.sub(r'<[^>]+>', '', summary)
                    # Truncate and add ellipsis
                    summary = summary[:150].strip() + ('...' if len(summary) > 150 else '')
                # ------------------------

                news_entry = {
                    'title': title or "無標題",
                    'link': entry.link or "#",
                    'date': published_time.strftime('%Y-%m-%d %H:%M'), # Add time
                    'source': entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else 'Google News',
                    'summary': summary,
                    'sentiment': sentiment_label,
                    'sentiment_score': sentiment_score
                }
                news_list.append(news_entry)
            except Exception as inner_e:
                logging.error(f"處理市場新聞條目 '{getattr(entry, 'title', 'N/A')}' 時發生錯誤: {inner_e}", exc_info=False) # Avoid flooding logs
                continue

        return news_list
    except Exception as e:
        logging.error(f"獲取市場新聞 (market={market}, category={category}) 時發生錯誤: {e}", exc_info=True)
        return []

# --- /api/market_summary (No changes needed for temperature) ---
@app.route('/api/market_summary', methods=['GET'])
def get_market_summary():
    try:
        market = request.args.get('market', 'TW')

        if market == 'TW':
            # Using common Taiwan indices/ETFs
            indices = {'^TWII': '加權指數', '0050.TW': '台灣50 ETF', '^TWOII': '櫃買指數', '0056.TW': '高股息 ETF'}
        else:
            # Common US indices
            indices = {'^GSPC': 'S&P 500', '^DJI': '道瓊工業', '^IXIC': '納斯達克', '^VIX': '恐慌指數 (VIX)'}

        market_data = []
        for symbol, name in indices.items():
            try:
                data = yf.Ticker(symbol)
                # Use '1d' for current day's data or last close, '2d' for change calculation
                hist = data.history(period="2d") # Get last 2 days

                if not hist.empty and len(hist) >= 2:
                    current_close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    change = ((current_close - prev_close) / prev_close) * 100 if prev_close else 0
                    price = current_close
                elif not hist.empty: # Only one day data available
                    price = hist['Close'].iloc[-1]
                    change = 0 # Cannot calculate change
                else:
                     logging.warning(f"無法獲取指數 {symbol} 的數據 from Yahoo Finance.")
                     price = 0
                     change = 0

                market_data.append({
                    'symbol': symbol,
                    'name': name,
                    'price': price,
                    'change': change
                })
            except Exception as inner_e:
                logging.error(f"獲取指數 {symbol} 數據時出錯: {inner_e}")
                market_data.append({
                    'symbol': symbol, 'name': name, 'price': 0, 'change': 0, 'error': True
                })

        # Fetch recent general news for sentiment calculation
        news = fetch_market_news(market, 'general')[:10] # Use more news for better sentiment avg

        # Calculate Market Sentiment (Improved logic)
        market_sentiment = "中性"
        sentiment_score_sum = 0
        valid_news_count = 0
        if news:
             for item in news:
                 # Map label to score: Positive=1, Neutral=0.5, Negative=0
                 score = 0.5
                 if item['sentiment'] == 'Positive': score = 1.0
                 elif item['sentiment'] == 'Negative': score = 0.0
                 sentiment_score_sum += score
                 valid_news_count += 1

             if valid_news_count > 0:
                 average_sentiment = sentiment_score_sum / valid_news_count
                 if average_sentiment > 0.65: market_sentiment = "樂觀"
                 elif average_sentiment < 0.40: market_sentiment = "謹慎"
                 else: market_sentiment = "中性"
             else:
                  market_sentiment = "未知" # If no news processed


        return jsonify({
            'market': market,
            'indices': market_data,
            'news_summary': news[:5], # Return only top 5 news headlines in summary
            'sentiment': market_sentiment,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        logging.error(f"獲取市場摘要時發生錯誤: {e}", exc_info=True)
        return jsonify({'error': f"獲取市場摘要時發生錯誤: {str(e)}"}), 500


# --- Static file serving (No changes needed) ---
@app.route('/static/charts/<path:filename>')
def serve_chart(filename):
    return send_from_directory('static/charts', filename)

@app.route('/static/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('static/data', filename)

# ------------------ 主程式 ------------------

if __name__ == '__main__':
    init_database() # Ensure DB schema is updated
    # Set debug=False for production deployment
    app.run(debug=True, host='0.0.0.0', port=5000)