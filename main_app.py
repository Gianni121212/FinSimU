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
                df = pd.read_csv(filepath, encoding='utf-8-sig')
                if header_name in df.columns:
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
# 在 main_app.py 中，找到並完整替換這個函式

def create_backtest_chart_assets(ticker, system_type, rank, portfolio, prices, dates, buys, sells):
    """為回測結果創建靜態PNG和互動HTML，並返回URL - (新增清晰圖例)"""
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                          row_heights=[0.7, 0.3],
                          subplot_titles=(f'{ticker} 價格走勢與交易信號', '投資組合價值變化'))
        
        # 股價與買賣點
        fig.add_trace(go.Scatter(
            x=dates, y=prices, mode='lines', name='收盤價', 
            line=dict(color='rgba(102, 126, 234, 0.7)')
        ), row=1, col=1)
        
        if buys:
            fig.add_trace(go.Scatter(
                x=[s['date'] for s in buys], y=[s['price'] for s in buys], 
                mode='markers', 
                # <<<<<<< 變更點 1: 為買入信號命名 >>>>>>>
                name='買入信號', 
                marker=dict(symbol='triangle-up', size=10, color='#27AE60', line=dict(width=1, color='white'))
            ), row=1, col=1)
        
        if sells:
            fig.add_trace(go.Scatter(
                x=[s['date'] for s in sells], y=[s['price'] for s in sells], 
                mode='markers', 
                # <<<<<<< 變更點 2: 為賣出信號命名 >>>>>>>
                name='賣出信號', 
                marker=dict(symbol='triangle-down', size=10, color='#E74C3C', line=dict(width=1, color='white'))
            ), row=1, col=1)

        # 投資組合價值
        if portfolio is not None and len(portfolio) > 0:
             fig.add_trace(go.Scatter(
                x=dates, y=portfolio, mode='lines', name='組合價值', 
                line=dict(color='purple')
            ), row=2, col=1)

        # <<<<<<< 變更點 3: 啟用並設定圖例樣式 >>>>>>>
        fig.update_layout(
            template='plotly_white', 
            height=500, 
            margin=dict(l=40, r=20, t=50, b=30), 
            showlegend=True,  # <-- 啟用圖例
            legend=dict(
                orientation="h",  # 水平排列
                yanchor="bottom",
                y=1.03,           # 放在圖表頂部之上
                xanchor="right",
                x=1
            )
        )
        # <<<<<<< 變更點結束 >>>>>>>
        
        base_filename = f"{ticker.replace('.', '_')}_{system_type}_Rank{rank}_backtest"
        
        # 儲存靜態圖片
        img_filename = f"{base_filename}.png"
        img_path = os.path.join('static/charts', img_filename)
        fig.write_image(img_path, scale=2)
        
        # 儲存互動HTML
        html_filename = f"{base_filename}.html"
        html_path = os.path.join('charts', html_filename)
        fig.write_html(html_path, include_plotlyjs='cdn', config={'displayModeBar': True})
        
        logger.info(f"回測圖表已生成（帶圖例）：{img_filename} 和 {html_filename}")
        return f"/static/charts/{img_filename}", f"/charts/{html_filename}"
        
    except Exception as e:
        logger.error(f"創建回測圖表失敗: {e}")
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
            'total_return_weight': 0.35,
            'avg_trade_return_weight': 0.30,
            'win_rate_weight': 0.30,
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
            if (end_dt - start_dt).days < 100:
                errors.append("訓練期間至少需要100天")
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
    def load_stock_data(self, ticker, start_date, end_date, system_type):
        """載入股票數據 - (整合.TW/.TWO自動重試)"""
        try:
            # <<<<<<< 這裡是新的智慧重試邏輯 >>>>>>>
            is_tw_stock_code = re.fullmatch(r'\d{4,6}[A-Z]?', ticker)
            loaded_data = None
            final_ticker = ticker
            
            load_func_a = lambda t: ga_load_data(
                t, start_date=start_date, end_date=end_date,
                sentiment_csv_path='2021-2025每週新聞及情緒分析.csv' if os.path.exists('2021-2025每週新聞及情緒分析.csv') else None,
                verbose=False
            )
            load_func_b = lambda t: load_stock_data_b(t, start_date=start_date, end_date=end_date, verbose=False)

            if is_tw_stock_code:
                logger.info(f"偵測到台股數字代號 {ticker}，將依序嘗試 .TW 和 .TWO 後綴。")
                for suffix in ['.TW', '.TWO']:
                    potential_ticker = f"{ticker}{suffix}"
                    logger.info(f"正在為系統 {system_type} 嘗試使用 {potential_ticker} 載入數據...")
                    
                    prices = None
                    if system_type == 'A':
                        loaded_data = load_func_a(potential_ticker)
                        prices = loaded_data[0] # prices is the first element
                    else:
                        loaded_data = load_func_b(potential_ticker)
                        prices = loaded_data[0] # prices is the first element
                    
                    if prices and len(prices) > 0:
                        logger.info(f"成功使用 {potential_ticker} 載入數據。")
                        final_ticker = potential_ticker
                        break # 成功，跳出迴圈
                    else:
                        loaded_data = None # 重置以進行下一次嘗試
            
            # 如果不是台股代號，或所有嘗試都失敗，則執行原始邏輯
            if not loaded_data:
                logger.info(f"執行標準查詢：{ticker}")
                if system_type == 'A':
                    loaded_data = load_func_a(ticker)
                else:
                    loaded_data = load_func_b(ticker)
            
            # 最終檢查數據
            prices = loaded_data[0]
            if not prices or len(prices) == 0:
                return None, f"數據不足，請檢查股票代號 {ticker} 或調整日期範圍 (已嘗試 .TW 和 .TWO)。"
            # <<<<<<< 智慧重試邏輯結束 >>>>>>>

            # 根據系統類型解包並預計算指標
            if system_type == 'A':
                prices, dates, stock_df, vix_series, sentiment_series = loaded_data
                precalculated, ready = ga_precompute_indicators(
                    stock_df, vix_series, STRATEGY_CONFIG_SHARED_GA,
                    sentiment_series=sentiment_series, verbose=False
                )
                if not ready: return None, "系統A技術指標計算失敗"
                return {'prices': prices, 'dates': dates, 'stock_df': stock_df, 'vix_series': vix_series,
                        'sentiment_series': sentiment_series, 'precalculated': precalculated, 'data_points': len(prices)}, None
            else: # 系統B
                prices, dates, stock_df, vix_series = loaded_data
                precalculated, ready = precompute_indicators_b(
                    stock_df, vix_series, STRATEGY_CONFIG_B, verbose=False
                )
                if not ready: return None, "系統B技術指標計算失敗"
                return {'prices': prices, 'dates': dates, 'stock_df': stock_df, 'vix_series': vix_series,
                        'precalculated': precalculated, 'data_points': len(prices)}, None
                
        except Exception as e:
            logger.error(f"載入數據時發生錯誤: {e}")
            return None, f"載入數據失敗: {str(e)}"

    def apply_fixed_and_custom_params(self, system_type, custom_weights, basic_params):
        """應用固定參數和自定義權重到GA參數"""
        if system_type == 'A':
            config = self.system_a_config.copy()
        else:
            config = self.system_b_config.copy()
        
        config.update(self.fixed_params)
        
        if 'generations' in basic_params:
            config['generations'] = max(5, min(100, int(basic_params['generations'])))
        if 'population_size' in basic_params:
            config['population_size'] = max(20, min(200, int(basic_params['population_size'])))
        if 'min_trades' in basic_params:
            config['min_trades_for_full_score'] = max(1, min(20, int(basic_params['min_trades'])))
        
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
            return f"注意：{recent_buy_signal.strftime('%Y/%m/%d')} 有近期買入訊號！"
        
        if recent_sell_signal and (not recent_buy_signal or recent_sell_signal > recent_buy_signal):
            return f"注意：{recent_sell_signal.strftime('%Y/%m/%d')} 有近期賣出訊號！"
        
        # 2. 如果沒有近期信號，判斷長期持有狀態
        if last_buy_date and (not last_sell_date or last_buy_date > last_sell_date):
            return "目前策略狀態為「持有中」"
        else:
            return "目前策略狀態為「無倉位」"

    # 在 main_app.py 中，找到 SingleStockTrainer 類別並替換以下兩個函式

# --- 1. 完整替換 run_training 函式 ---
    # 在 main_app.py 中，找到 SingleStockTrainer 類別並替換 run_training 函式

    def run_training(self, ticker, start_date, end_date, system_type, custom_weights, basic_params, num_runs=10):
        """執行訓練 - (效能優化版：只為Top 3生成圖表)"""
        try:
            
            
            errors = self.validate_inputs(ticker, start_date, end_date, system_type)
            if errors: 
                return {'success': False, 'errors': errors}
            
            weight_errors = self.validate_custom_weights(custom_weights)
            if weight_errors: 
                return {'success': False, 'errors': weight_errors}
            
            data_result, error_msg = self.load_stock_data(ticker, start_date, end_date, system_type)
            if error_msg: 
                return {'success': False, 'errors': [error_msg]}
            
            ga_config = self.apply_fixed_and_custom_params(system_type, custom_weights, basic_params)
            
            strategy_pool = []
            logger.info(f"開始為 {ticker} 執行 {num_runs} 次系統{system_type} NSGA-II優化 (高效模式)...")
            
            # <<<<<<< 效能優化點 1：迴圈內不再生成圖表 >>>>>>>
            for run_idx in range(num_runs):
                try:
                    runner_func = genetic_algorithm_unified if system_type == 'A' else genetic_algorithm_unified_b
                    result = runner_func(data_result['prices'], data_result['dates'], data_result['precalculated'], ga_config)
                    
                    if result and result[0] is not None:
                        gene, performance = result
                        metrics = self.calculate_detailed_metrics(gene, data_result, ga_config, system_type)
                        if not metrics: 
                            continue
                        
                        # 只儲存核心數據，不進行耗時的圖表生成
                        strategy_pool.append({
                            'gene': gene,
                            'fitness': metrics.get('total_return', 0),
                            'metrics': metrics,
                            'run': run_idx + 1,
                        })
                except Exception as e:
                    logger.warning(f"第 {run_idx + 1} 次運行失敗: {e}")
                    continue
            
            if not strategy_pool: 
                return {'success': False, 'errors': ['所有訓練運行都失敗了']}
            
            # 排序並選出最佳策略
            strategy_pool.sort(key=lambda x: x['fitness'], reverse=True)
            top_3 = strategy_pool[:3]
            
            results = []
            logger.info(f"訓練完成，開始為 Top {len(top_3)} 策略生成圖表...")

            # <<<<<<< 效能優化點 2：只為 Top 3 生成圖表 >>>>>>>
            for i, strategy in enumerate(top_3):
                # 在這裡才生成圖表，大大減少了 I/O 操作
                portfolio_values, buy_signals, sell_signals = self.generate_trading_signals(
                    strategy['gene'], data_result, ga_config, system_type
                )
                
                chart_image_url, chart_interactive_url = None, None
                if portfolio_values is not None:
                    chart_image_url, chart_interactive_url = create_backtest_chart_assets(
                        ticker, f"System{system_type}", strategy['run'],
                        portfolio_values, data_result['prices'], data_result['dates'],
                        buy_signals, sell_signals
                    )
                
                formatter_func = format_ga_gene_parameters_to_text if system_type == 'A' else format_gene_parameters_to_text_b
                description = formatter_func(strategy['gene'])
                
                results.append({
                    'rank': i + 1,
                    'gene': strategy['gene'],
                    'fitness': strategy['fitness'],
                    'metrics': strategy['metrics'],
                    'description': description,
                    'run_number': strategy['run'],
                    'chart_image_url': chart_image_url,
                    'chart_interactive_url': chart_interactive_url,
                    'buy_signals_count': len(buy_signals or []),
                    'sell_signals_count': len(sell_signals or [])
                })
            
            logger.info("Top 3 策略圖表生成完畢。")
            return {
                'success': True, 'ticker': ticker, 'system_type': system_type,
                'training_period': f"{start_date} ~ {end_date}",
                'data_points': data_result['data_points'], 'total_runs': num_runs,
                'successful_runs': len(strategy_pool), 'results': results,
                'config_used': {k: v for k, v in ga_config.items() if k not in ['custom_weights']},
                'custom_weights_used': custom_weights,
                'chart_engine': 'ImagePreview'
            }
            
        except Exception as e:
            logger.error(f"訓練過程發生錯誤: {e}", exc_info=True)
            return {'success': False, 'errors': [f'訓練失敗: {str(e)}']}


    # --- 2. 完整替換 run_manual_backtest 函式 ---
    def run_manual_backtest(self, ticker, gene, duration_months):
        """執行手動回測 - (修改版：生成圖片和HTML URL)"""
        try:
            
            
            system_type = 'A' if len(gene) in range(27, 29) else 'B' if len(gene) in range(9, 11) else None
            if not system_type: 
                return {'success': False, 'error': f"無法識別的基因長度: {len(gene)}"}
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=duration_months * 30.4)
            start_date_str, end_date_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            
            errors = self.validate_inputs(ticker, start_date_str, end_date_str, system_type)
            if errors: 
                return {'success': False, 'error': ", ".join(errors)}
            
            data_result, error_msg = self.load_stock_data(ticker, start_date_str, end_date_str, system_type)
            if error_msg: 
                return {'success': False, 'error': error_msg}
            
            ga_config = self.system_a_config if system_type == 'A' else self.system_b_config
            
            metrics = self.calculate_detailed_metrics(gene, data_result, ga_config, system_type)
            if not metrics: 
                return {'success': False, 'error': '無法生成回測結果，可能是此基因在此期間無交易。'}
            
            portfolio_values, buy_signals, sell_signals = self.generate_trading_signals(gene, data_result, ga_config, system_type)
            
            # <<<<<<< 變更點：呼叫新的圖表生成函式 >>>>>>>
            chart_image_url, chart_interactive_url = create_backtest_chart_assets(
                ticker, f"System{system_type}", "Manual",
                portfolio_values, data_result['prices'], data_result['dates'],
                buy_signals, sell_signals
            )
            
            signal_status = self.analyze_signal_status(buy_signals, sell_signals)
            
            # <<<<<<< 變更點：在回傳的字典中包含 URL >>>>>>>
            return {
                'success': True, 'ticker': ticker, 'system_type_detected': f'系統 {system_type}',
                'backtest_period': f"{start_date_str} ~ {end_date_str}",
                'metrics': metrics, 
                'chart_image_url': chart_image_url, 
                'chart_interactive_url': chart_interactive_url, 
                'signal_status': signal_status
            }
            
        except Exception as e:
            logger.error(f"手動回測過程發生嚴重錯誤: {e}", exc_info=True)
            return {'success': False, 'error': f'回測失敗: {str(e)}'}

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
    return render_template('index.html')

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
    (新版) 訓練API端點 - 自動化訓練系統A和系統B，並回傳各自的最佳策略
    """
    if not ENGINES_IMPORTED:
        return jsonify({'success': False, 'errors': ['遺傳算法引擎未正確載入']}), 500
    
    try:
        data = request.json
        ticker = data.get('ticker', '').strip().upper()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        custom_weights = data.get('custom_weights', trainer.default_custom_weights)
        
        # <<<< 修改點 1: 後端定義固定的訓練參數 >>>>
        # 從前端獲取用戶唯一能設定的 min_trades
        basic_params_from_user = data.get('basic_params', {})
        
        # 將固定的參數與用戶設定的參數合併
        fixed_basic_params = {
            'generations': 15,       # 固定世代數
            'population_size': 50,   # 固定族群大小
            'min_trades': int(basic_params_from_user.get('min_trades', 4)) # 保留用戶設定
        }
        fixed_num_runs = 15 # 固定訓練次數

        logger.info(f"收到簡化版訓練請求: {ticker} ({start_date} to {end_date})")
        logger.info(f"將使用固定參數: {fixed_basic_params}，訓練 {fixed_num_runs} 次")

        # <<<< 修改點 2: 依序執行系統 A 和 B 的訓練 >>>>
        # 訓練系統 A
        logger.info("--- 開始訓練系統 A ---")
        result_A = trainer.run_training(
            ticker=ticker, start_date=start_date, end_date=end_date,
            system_type='A',
            custom_weights=custom_weights,
            basic_params=fixed_basic_params,
            num_runs=fixed_num_runs
        )
        
        # 訓練系統 B
        logger.info("--- 開始訓練系統 B ---")
        result_B = trainer.run_training(
            ticker=ticker, start_date=start_date, end_date=end_date,
            system_type='B',
            custom_weights=custom_weights,
            basic_params=fixed_basic_params,
            num_runs=fixed_num_runs
        )

        # <<<< 修改點 3: 合併兩個系統的最佳結果 >>>>
        combined_results = []
        
        # 提取系統 A 的最佳策略 (Rank 1)
        if result_A.get('success') and result_A.get('results'):
            strategy_A = result_A['results'][0]
            strategy_A['rank'] = 1 # 重新排名為 1
            # 新增一個欄位來標示策略類型，方便前端未來客製化顯示
            strategy_A['strategy_type_name'] = '策略 1 ' 
            combined_results.append(strategy_A)
            logger.info("系統 A 訓練成功，已提取最佳策略。")
        else:
            logger.warning("系統 A 訓練失敗或無結果。")

        # 提取系統 B 的最佳策略 (Rank 1)
        if result_B.get('success') and result_B.get('results'):
            strategy_B = result_B['results'][0]
            strategy_B['rank'] = 2 # 重新排名為 2
            strategy_B['strategy_type_name'] = '策略 2 '
            combined_results.append(strategy_B)
            logger.info("系統 B 訓練成功，已提取最佳策略。")
        else:
            logger.warning("系統 B 訓練失敗或無結果。")
            
        # <<<< 修改點 4: 處理訓練失敗的情況並回傳合併後的結果 >>>>
        if not combined_results:
            # 如果兩個系統都失敗，回傳一個綜合的錯誤訊息
            error_A = result_A.get('errors', ['未知錯誤'])[0] if not result_A.get('success') else '無有效策略'
            error_B = result_B.get('errors', ['未知錯誤'])[0] if not result_B.get('success') else '無有效策略'
            return jsonify({'success': False, 'errors': [f"所有訓練均失敗。系統A: {error_A} | 系統B: {error_B}"]})

        # 使用任一成功結果的元數據來建立最終的回傳物件
        base_result = result_A if result_A.get('success') else result_B
        
        final_response = {
            'success': True,
            'ticker': base_result.get('ticker'),
            'training_period': base_result.get('training_period'),
            'results': combined_results
        }
        
        logger.info(f"訓練完成，將回傳 {len(combined_results)} 個最佳策略。")
        return jsonify(final_response)

    except Exception as e:
        logger.error(f"API錯誤 /api/train: {e}", exc_info=True)
        return jsonify({'success': False, 'errors': [f'API伺服器錯誤: {str(e)}']}), 500

@app.route('/api/manual-backtest', methods=['POST'])
@login_required
def api_manual_backtest():
    if not ENGINES_IMPORTED:
        return jsonify({'success': False, 'error': '遺傳算法引擎未正確載入'}), 500
    
    try:
        data = request.json
        ticker = data.get('ticker', '').strip().upper()
        gene = data.get('gene')
        duration_months = data.get('duration_months', 36)
        
        if not ticker or not gene or not isinstance(gene, list):
            return jsonify({'success': False, 'error': '無效的輸入參數'}), 400
        
        result = trainer.run_manual_backtest(ticker, gene, duration_months)
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
            win_rate, total_return, trade_count, avg_trade_return, max_drawdown,
            max_trade_extremes, strategy_details
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            f"{metrics.get('max_trade_drop_pct', 0.0):.2f}% / {metrics.get('max_trade_gain_pct', 0.0):.2f}%",
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


@app.route('/api/capital-allocation', methods=['POST'])
@login_required
def api_capital_allocation():
    """處理資金配置請求的 API 端點 - 經過兩輪除錯的穩健版本"""
    try:
        # 1. 接收並驗證前端傳來的數據
        data = request.get_json()
        strategy_ids = data.get('strategy_ids')
        risk_profile = data.get('risk_profile')

        if not isinstance(strategy_ids, list) or not strategy_ids or not risk_profile:
            return jsonify({'success': False, 'message': '無效的請求參數'}), 400
        
        if not gemini_client:
            return jsonify({'success': False, 'message': 'Gemini AI 服務未配置'}), 503

        # 2. 從資料庫查詢被選中策略的詳細數據
        placeholders = ', '.join(['%s'] * len(strategy_ids))
        sql = f"""
            SELECT id, ticker, total_return, avg_trade_return, win_rate, max_drawdown, max_trade_extremes 
            FROM saved_strategies 
            WHERE id IN ({placeholders}) AND user_id = %s
        """
        params = tuple(strategy_ids) + (current_user.id,)
        strategies_from_db = execute_db_query(sql, params, fetch_all=True)

        if not strategies_from_db or len(strategies_from_db) != len(strategy_ids):
             return jsonify({'success': False, 'message': '找不到部分或全部策略，請刷新後再試'}), 404

        # 3. 構建給 Gemini 的 Prompt (依賴已修正的 _build_allocation_prompt)
        prompt_text = _build_allocation_prompt(risk_profile, strategies_from_db)
        logger.info(f"為使用者 {current_user.id} 生成的資金配置 Prompt 已建立。")

        # 4. 調用 Gemini API
        config = genai_types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=20000,
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            safety_settings=safety_settings_gemini
        )
        
        response = gemini_client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=prompt_text,
            config=config
        )

        # 5. 強化的回應處理 (核心修正)
        logger.info(f"Gemini API 回應類型: {type(response)}")
        response_text = None

        # 檢查是否有因 Prompt 本身的問題而被阻擋
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason_str = str(response.prompt_feedback.block_reason)
            logger.error(f"❌ 請求被阻擋！原因: {block_reason_str}")
            return jsonify({
                'success': False,
                'message': f'AI 分析請求被安全策略阻擋，原因: {block_reason_str}。請嘗試調整策略描述或風險偏好。'
            }), 400

        # 嘗試從 response.text 直接獲取 (最簡單的情況)
        if hasattr(response, 'text') and response.text:
            response_text = response.text.strip()
            logger.info("✅ 使用 response.text 成功獲取回應")
        # 如果不行，深入挖掘 candidates
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0] # 通常只關心第一個候選項
            
            # 檢查完成原因，這是判斷是否被安全攔截的關鍵
            finish_reason_str = str(candidate.finish_reason) if hasattr(candidate, 'finish_reason') else 'UNKNOWN'
            if finish_reason_str.upper() != 'STOP':
                 logger.warning(f"⚠️ Gemini 回應的完成原因並非 'STOP', 而是 '{finish_reason_str}'。這通常表示內容因安全或其他原因被攔截。")
                 if finish_reason_str.upper() == 'SAFETY':
                     return jsonify({
                         'success': False, 
                         'message': 'AI 生成的內容因觸發安全策略而被攔截。請稍後重試或調整請求。'
                     }), 500

            # 如果完成原因是正常的，再嘗試解析內容
            if hasattr(candidate, 'content') and candidate.content.parts:
                response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text')).strip()
                if response_text:
                    logger.info("✅ 從 candidate.content.parts 成功組合回應")

        if not response_text:
            logger.error("❌ 無法從 Gemini API 獲取任何文本內容。可能是因為安全攔截或空回應。")
            logger.error(f"原始回應物件詳情: {response}") # 記錄下整個物件以供分析
            return jsonify({
                'success': False, 
                'message': 'AI 服務返回了空的回應，可能因內容審核被攔截，請稍後再試。'
            }), 500

        logger.info(f"獲取到的回應長度: {len(response_text)}")
        logger.info(f"回應前200字符: {response_text[:200]}...")

        # 6. 智能 JSON 解析
        try:
            cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = cleaned_text[json_start:json_end]
                parsed_response = json.loads(json_text)
                
                # 驗證必要欄位
                if 'allocations' not in parsed_response or not isinstance(parsed_response['allocations'], list):
                    raise ValueError("回應中缺少 'allocations' 陣列")
                
                total_percentage = sum(item.get('percentage', 0) for item in parsed_response['allocations'])
                if not (95 <= total_percentage <= 105):
                    logger.warning(f"AI 配置總和為 {total_percentage}%，偏離100%較多")
                
                if 'reasoning' not in parsed_response:
                    parsed_response['reasoning'] = "AI已完成分析，但未提供詳細說明"
                
                logger.info("✅ JSON 解析成功")
                return jsonify({'success': True, 'data': parsed_response})
            else:
                raise ValueError("在回應中找不到有效的JSON結構")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON 解析失敗: {e}")
            logger.error(f"原始回應: {response_text}")
            return jsonify({
                'success': False, 
                'message': f'AI 回應格式異常，原始回應: {response_text[:300]}...',
                'raw_response': response_text
            }), 500

    except Exception as e:
        logger.error(f"資金配置 API 發生嚴重錯誤: {e}", exc_info=True)
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

@app.route('/api/strategy-signals', methods=['GET'])
def api_strategy_signals():
    """
    (修正版) AI策略信號 API - 修正了 Collation 錯誤並 JOIN ai_vs_user_games 表以包含完整策略基因供儲存
    """
    try:
        market = request.args.get('market', 'TW')
        signal_type_filter = request.args.get('type', 'buy')
        
        signal_conditions = "('BUY', 'BUY_SELL')" if signal_type_filter == 'buy' else "('SELL', 'BUY_SELL')"
        
        # <<<< 修正點：在 JOIN ON 條件中加入 COLLATE utf8mb4_unicode_ci 來統一比較規則 >>>>
        query = f"""
        WITH RankedSignals AS (
            SELECT 
                bs.stock_ticker, bs.system_type, bs.strategy_rank, bs.signal_type, 
                bs.signal_reason, bs.buy_price, bs.sell_price, bs.return_pct, 
                bs.win_rate, bs.chart_path, bs.processed_at,
                a.ai_strategy_gene,
                a.strategy_details,
                a.game_start_date,
                a.game_end_date,
                a.total_trades,
                a.average_trade_return_pct,
                a.max_drawdown_pct,
                a.max_trade_drop_pct,
                a.max_trade_gain_pct,
                ROW_NUMBER() OVER(PARTITION BY bs.stock_ticker, bs.system_type ORDER BY bs.win_rate DESC, bs.return_pct DESC) as rn
            FROM 
                backtest_signals bs
            JOIN 
                ai_vs_user_games a ON bs.stock_ticker = a.stock_ticker COLLATE utf8mb4_unicode_ci
                                   AND bs.strategy_rank = a.strategy_rank
                                   AND bs.system_type = (CASE WHEN a.user_id = 2 THEN 'SystemA' ELSE 'SystemB' END) COLLATE utf8mb4_unicode_ci
            WHERE 
                bs.market_type = %s AND bs.signal_type IN {signal_conditions}
        )
        SELECT * FROM RankedSignals WHERE rn = 1 ORDER BY stock_ticker ASC, system_type ASC;
        """
        
        signals = execute_db_query(query, (market,), fetch_all=True)
        
        if signals:
            for signal in signals:
                if isinstance(signal.get('processed_at'), datetime):
                    signal['processed_at_str'] = signal['processed_at'].strftime('%Y-%m-%d %H:%M')
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

@app.route('/charts/<filename>')
def serve_chart(filename):
    """提供圖表檔案"""
    try:
        return send_from_directory('charts', filename)
    except:
        return jsonify({"error": "Chart not found"}), 404

# ==============================================================================
# >>> 以下為原始 program.py 的新聞情緒分析功能 (完整保留) <<<
# ==============================================================================
# ==============================================================================
#      >>> (新整合) 以下為新聞情緒分析功能 (從 update_news.py 移植) <<<
# ==============================================================================

# --- 全局設定與開關 ---
MOCK_TODAY = None # 正常執行時為 None, 可設定為 datetime(YYYY, M, D).date() 進行測試
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

# FinBERT 模型快取
finbert_tokenizer = None
finbert_model = None

if not FINBERT_AVAILABLE:
    logger.warning("PyTorch 或 Transformers 未安裝，FinBERT 情緒分析功能將被跳過。")

def load_finbert_model():
    """載入 FinBERT 模型和 Tokenizer，並進行快取。"""
    global finbert_tokenizer, finbert_model
    if finbert_model is None and FINBERT_AVAILABLE:
        try:
            logger.info("  [新聞分析] 首次載入 FinBERT 模型 (ProsusAI/finbert)...")
            model_name = "ProsusAI/finbert"
            finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info("  [新聞分析] FinBERT 模型載入成功！")
        except Exception as e:
            logger.error(f"  [新聞分析] 載入 FinBERT 模型時發生錯誤: {e}")
            return False
    return finbert_model is not None

def analyze_titles_with_finbert(titles: list):
    """使用 FinBERT 分析新聞標題的情緒。"""
    if not load_finbert_model():
        logger.warning("  [新聞分析] FinBERT 模型不可用，跳過情緒分析。")
        return [f"[ANALYSIS_SKIPPED] {title}" for title in titles]
    
    logger.info(f"  [新聞分析] 正在使用 FinBERT 分析 {len(titles)} 條英文新聞標題...")
    analyzed_titles = []
    
    for title in titles:
        try:
            inputs = finbert_tokenizer(title, padding=True, truncation=True, return_tensors='pt')
            outputs = finbert_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction_idx = torch.argmax(probabilities).item()
            label = finbert_model.config.id2label[prediction_idx]
            score = probabilities[0][0].item() - probabilities[0][1].item() # Positive - Negative
            analyzed_titles.append(f"[{label.upper()}, Score: {score:+.2f}] {title}")
        except Exception as e:
            logger.error(f"  [新聞分析] FinBERT 分析標題 '{title}' 時出錯: {e}")
            analyzed_titles.append(f"[ANALYSIS_FAILED] {title}")
    
    logger.info("  [新聞分析] FinBERT 分析完成。")
    return analyzed_titles

def get_this_weeks_english_news(target_topics: dict):
    """抓取新聞並返回標題列表及真實新聞的日期範圍。"""
    real_today = datetime.now(pytz.utc)
    real_start_date = real_today - timedelta(days=7)
    real_date_range_str = f"{real_start_date.strftime('%Y-%m-%d')} to {real_today.strftime('%Y-%m-%d')}"
    
    if MOCK_TODAY:
        logger.info(f"\n--- [新聞分析] 模擬測試模式已啟動 (模擬日期: {MOCK_TODAY.strftime('%Y-%m-%d')}) ---")
        logger.info(f"  [新聞分析] 將抓取真實世界近期 ({real_date_range_str}) 的新聞作為分析材料。")
    else:
        logger.info(f"\n--- [新聞分析] 正常模式已啟動 ---")
        logger.info(f"  [新聞分析] 將抓取真實世界近期 ({real_date_range_str}) 的新聞作為分析材料。")
    
    seen_titles = set()
    topic_items = list(target_topics.items())
    total_headlines_collected = 0
    
    for i, (company_or_topic, ticker_or_keyword) in enumerate(topic_items):
        if total_headlines_collected >= MAX_TOTAL_HEADLINES:
            logger.info(f"\n  [新聞分析] 已達到全局新聞上限 ({MAX_TOTAL_HEADLINES}條)，停止抓取。")
            break
        
        logger.info(f"  [新聞分析] [進度 {i+1}/{len(topic_items)}] 查詢: '{company_or_topic}' (已收集: {total_headlines_collected}條)...")
        
        primary_query = f'"{company_or_topic}" {ticker_or_keyword} stock' if ticker_or_keyword else f'"{company_or_topic}" stock market'
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(primary_query)}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(url)
            if feed.entries:
                headlines_from_this_topic = 0
                for entry in feed.entries:
                    if headlines_from_this_topic >= MAX_HEADLINES_PER_TOPIC or total_headlines_collected >= MAX_TOTAL_HEADLINES:
                        break
                    
                    try:
                        published_dt = datetime(*entry.published_parsed[:6])
                        published_dt_utc = pytz.utc.localize(published_dt)
                        if real_start_date <= published_dt_utc <= real_today:
                            title = entry.title.strip()
                            if title not in seen_titles:
                                seen_titles.add(title)
                                headlines_from_this_topic += 1
                                total_headlines_collected += 1
                    except Exception:
                        continue
                
                if headlines_from_this_topic > 0:
                    logger.info(f"  ✔️ [新聞分析] 查詢成功，為此主題新增 {headlines_from_this_topic} 條新聞。")
        except Exception as e:
            logger.error(f"  -> [新聞分析] 抓取查詢時發生錯誤: {e}")
        
        if i < len(topic_items) - 1:
            time.sleep(random.uniform(0.5, 1.2))
    
    if seen_titles:
        logger.info(f"\n【新聞分析抓取完成】總共收集到 {len(seen_titles)} 條不重複的相關英文新聞標題。")
    else:
        logger.warning("\n【新聞分析抓取完成】未能找到任何符合條件的新聞。")
    
    return list(seen_titles), real_date_range_str

def get_sentiment_and_translate_summary(analyzed_titles: list, simulated_week_key: str, real_news_date_range: str, few_shot_examples=None):
    """使用"時空橋接提示"讓 Gemini 進行模擬分析。"""
    if not gemini_client:
        return None, "Gemini client未配置"
    
    if not analyzed_titles:
        return None, "分析後的新聞標題列表為空"
    
    example_prompt_part = ""
    if few_shot_examples:
        example_prompt_part = "Here are some historical rating examples for your reference (in Traditional Chinese):\n"
        for ex_date, ex_score, ex_summary in few_shot_examples:
            example_prompt_part += f"- Week: {ex_date}; Sentiment Score: {ex_score}; Summary: {ex_summary}\n"
        example_prompt_part += "\n"
    
    news_titles_str = "\n".join([f"- {title}" for title in analyzed_titles])
    
    prompt = f"""
You are an expert financial analyst participating in a market simulation.
**CONTEXT:**
- The **simulated week** you are analyzing is: **{simulated_week_key}**
- To perform your analysis, you have been provided with **real-world news headlines** from the recent period of: **{real_news_date_range}**
**YOUR TASK:**
You must **interpret these real-world events as if they were happening during the simulated week**. Synthesize the key themes (e.g., inflation, Fed policy, tech trends, geopolitical events) and generate a market analysis *for the simulated week*.
The headlines below are pre-analyzed by FinBERT. The format is [SENTIMENT_LABEL, FinBERT_Score] Original Headline. Use this as a key reference for sentiment.
**REQUIRED OUTPUT FORMAT:**
1. **Sentiment Score**: A single integer from 0 to 100 representing the market sentiment for the **simulated week**. (0=fear, 50=neutral, 100=greed).
2. **News Summary (Traditional Chinese)**: A translated summary of the key events, written *as if* they occurred in the simulated week. Separate items with a semicolon.
---
**Provided Real-World News Headlines:**
{news_titles_str}
---
**Now, provide the analysis in the required format for the simulated week of {simulated_week_key}:**
Sentiment Score: [A single integer between 0 and 100]
News Summary (Traditional Chinese): [Your translated summary for the simulated week]
"""
    
    try:
        logger.info(f"\n  [新聞分析] 發送 {len(analyzed_titles)} 條新聞到 Gemini (模擬週: {simulated_week_key}, 真實新聞源: {real_news_date_range})...")
        
        response = gemini_client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.3, max_output_tokens=1000)
        )
        
        content = response.text
        
        score_match = re.search(r"Sentiment Score:\s*(\d+)", content, re.IGNORECASE)
        summary_match = re.search(r"News Summary \(Traditional Chinese\):\s*(.+)", content, re.IGNORECASE | re.DOTALL)
        
        sentiment_score_val = int(score_match.group(1)) if score_match else None
        
        if summary_match:
            news_summary_val = summary_match.group(1).strip()
            news_summary_val = re.sub(r'^\s*-\s*', '', news_summary_val)
            news_summary_val = news_summary_val.replace('\n', ' ').replace(';', '；').strip()
            news_summary_val = re.sub(r'\s*；\s*', '；', news_summary_val)
            news_summary_val = re.sub(r'；$', '', news_summary_val)
        else:
            news_summary_val = "未能生成摘要"
        
        logger.info(f"  [新聞分析] 已解析 ({simulated_week_key}): 分數={sentiment_score_val}, 摘要='{news_summary_val[:100]}...'")
        
        return sentiment_score_val, news_summary_val
        
    except Exception as e:
        logger.error(f"  [新聞分析] Gemini API 調用或解析時出錯: {e}")
        return None, f"Gemini API調用或解析錯誤: {str(e)}"

def get_few_shot_examples(csv_filepath, num_examples=5):
    """從 CSV 讀取 few-shot 學習的範例。"""
    try:
        df = pd.read_csv(csv_filepath, encoding='utf-8-sig')
        if df.empty: return []
        df_valid = df.dropna(subset=['情緒分數', '重大新聞摘要']).tail(num_examples)
        return [(str(r['年/週']), r['情緒分數'], str(r['重大新聞摘要'])) for _, r in df_valid.iterrows()]
    except Exception as e:
        logger.warning(f"[新聞分析] 讀取 few-shot 範例時出錯: {e}")
        return []

def get_current_week_key():
    """根據是否在模擬模式，獲取本週的日期鍵值。"""
    today = MOCK_TODAY if MOCK_TODAY else datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return f"{start_of_week.strftime('%Y/%m/%d')}-{end_of_week.strftime('%Y/%m/%d')}"

def update_sentiment_csv(csv_filepath, target_topics):
    """主流程函式：整合所有步驟來更新 CSV。"""
    if not gemini_client:
        logger.error("[新聞分析] Gemini client 未載入，任務終止。")
        return
    
    simulated_week_key = get_current_week_key()
    logger.info(f"[新聞分析] 目標模擬週的鍵值為: {simulated_week_key}")
    
    raw_english_titles, real_date_range = get_this_weeks_english_news(target_topics)
    if not raw_english_titles:
        logger.warning(f"[新聞分析] 無法獲取近期真實新聞，流程終止。")
        return
    
    analyzed_titles = analyze_titles_with_finbert(raw_english_titles)
    few_shot_examples = get_few_shot_examples(csv_filepath, num_examples=5)
    
    score, summary_chinese = get_sentiment_and_translate_summary(analyzed_titles, simulated_week_key, real_date_range, few_shot_examples)
    
    if score is not None and summary_chinese and "未能生成摘要" not in summary_chinese:
        try:
            df = pd.read_csv(csv_filepath, encoding='utf-8-sig') if os.path.exists(csv_filepath) else pd.DataFrame(columns=['年/週', '情緒分數', '重大新聞摘要'])
            df['年/週'] = df['年/週'].astype(str).str.strip()
            week_key_stripped = simulated_week_key.strip()
            
            week_exists_mask = df['年/週'] == week_key_stripped
            
            if week_exists_mask.any():
                logger.info(f"\n[新聞分析] 更新模擬週 ({week_key_stripped}) 的情緒分數與摘要...")
                df.loc[week_exists_mask, '情緒分數'] = score
                df.loc[week_exists_mask, '重大新聞摘要'] = summary_chinese
            else:
                logger.info(f"\n[新聞分析] 新增模擬週 ({week_key_stripped}) 的情緒分數與摘要...")
                new_row = pd.DataFrame([{'年/週': week_key_stripped, '情緒分數': score, '重大新聞摘要': summary_chinese}])
                df = pd.concat([df, new_row], ignore_index=True)
            
            df.drop_duplicates(subset=['年/週'], keep='last', inplace=True)
            df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
            logger.info(f"[新聞分析] 已成功將 {week_key_stripped} 的資料寫入/更新到 CSV！")
            
        except Exception as e:
            logger.error(f"[新聞分析] 寫入 CSV 時出錯: {e}")
    else:
        logger.error(f"\n[新聞分析] 未能從 Gemini 取得有效的模擬分析結果：{summary_chinese}")


# ==============================================================================
#           >>> 以下為新加入的排程回測功能 (獨立區塊) <<<
# ==============================================================================

class StrategyBacktesterWithSignals:
    """策略回測器 - (從 backtest.py 遷移並整合，使用 logger)"""
    
    def __init__(self):
        self.backtest_months = 36
        self.signal_check_days = 5
        self.start_date, self.end_date = self._get_date_range()
        self.charts_dir = "charts"
        self.data_cache_a = {}
        self.data_cache_b = {}
        os.makedirs(self.charts_dir, exist_ok=True)
        logger.info(f"🎯 [排程回測] 回測器初始化完成")
        logger.info(f"📅 [排程回測] 回測期間: {self.start_date} ~ {self.end_date}")
        logger.info(f"📁 [排程回測] 圖表目錄: {self.charts_dir}")
    
    def _get_date_range(self):
        end_date = datetime.now(pytz.timezone('Asia/Taipei')).date()
        start_date = end_date - timedelta(days=self.backtest_months * 30)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    
    def create_signals_table(self):
        """檢查並創建 backtest_signals 資料庫表"""
        query = """
        CREATE TABLE IF NOT EXISTS `backtest_signals` (
          `id` INT AUTO_INCREMENT PRIMARY KEY, `stock_ticker` VARCHAR(20) NOT NULL,
          `stock_name` VARCHAR(100), `market_type` VARCHAR(10) NOT NULL,
          `system_type` VARCHAR(20) NOT NULL, `strategy_rank` INT NOT NULL,
          `signal_type` ENUM('BUY', 'SELL', 'BUY_SELL') NOT NULL, `signal_reason` TEXT,
          `buy_price` FLOAT NULL, `sell_price` FLOAT NULL, `return_pct` FLOAT,
          `win_rate` FLOAT NULL, `chart_path` VARCHAR(255), `processed_at` DATETIME NOT NULL,
          INDEX `idx_market_signal` (`market_type`, `signal_type`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""
        try:
            execute_db_query(query)
            logger.info("✅ [排程回測] `backtest_signals` 表已確認存在")
        except Exception as e:
            logger.error(f"❌ [排程回測] 創建 `backtest_signals` 表失敗: {e}")

    def save_results_to_db(self, results):
        """將有信號的結果儲存到資料庫"""
        conn = None
        try:
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE backtest_signals")
                logger.info("🗑️ [排程回測] 已清空舊的信號資料")
                query = """INSERT INTO backtest_signals (stock_ticker, stock_name, market_type, system_type, strategy_rank, 
                    signal_type, signal_reason, buy_price, sell_price, return_pct, win_rate, chart_path, processed_at) 
                    VALUES (%(ticker)s, NULL, %(market_type)s, %(system)s, %(rank)s, %(signal_type)s, %(signal_reason)s, 
                    %(buy_price)s, %(sell_price)s, %(return_pct)s, %(win_rate)s, %(chart_path)s, %(processed_at)s)"""
                
                to_save = []
                for res in results:
                    if res.get('has_recent_signal'):
                        signal_type = 'BUY_SELL' if res['has_buy_signal'] and res['has_sell_signal'] else 'BUY' if res['has_buy_signal'] else 'SELL'
                        res_copy = res.copy()
                        res_copy['signal_type'] = signal_type
                        res_copy['chart_path'] = res_copy.get('chart_path', None)
                        to_save.append(res_copy)
                
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
                   FROM ai_vs_user_games WHERE strategy_rank > 0 AND ai_strategy_gene IS NOT NULL 
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
        if not signals: return False, f"無{signal_type_text}信號", None
        recent, latest_price, today = [], None, datetime.now().date()
        for signal in signals[-self.signal_check_days:]:
            s_date = pd.to_datetime(signal['date']).date()
            days_diff = (today - s_date).days
            if 0 <= days_diff < self.signal_check_days:
                day_str = {0: "今天", 1: "昨天"}.get(days_diff, f"{days_diff}天前")
                recent.append(f"{day_str}({s_date})")
                latest_price = signal['price']
        return (True, f"在 {', '.join(recent)} 檢測到{signal_type_text}信號", latest_price) if recent else (False, f"近期無{signal_type_text}信號", None)

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
            has_buy, buy_reason, buy_price = self.check_recent_signals(buys, '買入')
            has_sell, sell_reason, sell_price = self.check_recent_signals(sells, '賣出')
            has_signal = has_buy or has_sell
            signal_reason = " | ".join(filter(None, [buy_reason if has_buy else None, sell_reason if has_sell else None]))
            
            result = {'ticker': ticker, 'system': sys_type, 'rank': rank, 'market_type': strategy['market_type'], 'return_pct': final_return, 'win_rate': win_rate,
                      'has_recent_signal': has_signal, 'signal_reason': signal_reason, 'has_buy_signal': has_buy, 'buy_reason': buy_reason,
                      'has_sell_signal': has_sell, 'sell_reason': sell_reason, 'buy_price': buy_price, 'sell_price': sell_price,
                      'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
            if has_signal:
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
        signals_found = [res for res in results if res['has_recent_signal']]
        logger.info(f"⏱️ [排程回測] 總耗時: {elapsed:.2f} 秒")
        logger.info(f"🎯 [排程回測] 發現信號: {len(signals_found)}")
        
        if signals_found:
            logger.info("\n🎯 [排程回測] 【近期有買賣信號的策略】")
            for res in signals_found:
                buy_info = f" @ {res['buy_price']:.2f}" if res['has_buy_signal'] and res['buy_price'] is not None else ""
                sell_info = f" @ {res['sell_price']:.2f}" if res['has_sell_signal'] and res['sell_price'] is not None else ""
                logger.info(f"  - {res['ticker']} | {res['system']} R{res['rank']} | 勝率: {res['win_rate']:.2f}% | 🟢:{res['has_buy_signal']}{buy_info} | 🔴:{res['has_sell_signal']}{sell_info}")
        
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

def run_scheduled_news_update():
    """(新整合) 每日自動執行的市場情緒分析任務"""
    with app.app_context():
        logger.info("="*50 + f"\n⏰ [排程任務] 啟動每日市場情緒分析... (台灣時間: {datetime.now(pytz.timezone('Asia/Taipei'))})\n" + "="*50)
        try:
            if not GEMINI_API_KEY:
                logger.error("❌ [排程任務] 錯誤: GEMINI_API_KEY 環境變數未設定。市場情緒分析任務中止。")
                return

            # 確保 CSV 檔案存在
            if not os.path.exists(CSV_FILEPATH):
                logger.warning(f"'{CSV_FILEPATH}' 不存在，創建一個空的範例檔案...")
                pd.DataFrame(columns=['年/週', '情緒分數', '重大新聞摘要']).to_csv(CSV_FILEPATH, index=False, encoding='utf-8-sig')

            # 執行主流程
            update_sentiment_csv(CSV_FILEPATH, target_topics=TARGET_COMPANIES_AND_TOPICS)

            logger.info("✅ [排程任務] 每日市場情緒分析任務執行完畢。")
        except Exception as e:
            logger.error(f"\n❌ [排程任務] 市場情緒分析執行期間發生嚴重錯誤: {e}\n{traceback.format_exc()}")
        finally:
            logger.info("=" * 50)


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
    
    # 設定並啟動排程器
    logger.info("⚙️ 正在設定排程器...")
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Taipei'))
    
    # 新增任務：每日市場情緒分析
    scheduler.add_job(
        func=run_scheduled_news_update,
        trigger='cron',
        hour=8,
        minute=30,
        id='daily_news_update_job',
        name='每日台灣時間 8:30 執行市場情緒分析',
        replace_existing=True
    )
    logger.info("✅ 已設定每日市場情緒分析排程 (08:30)。")
    
    if ENGINES_IMPORTED:
        # 新增任務：每日回測
        scheduler.add_job(
            func=run_scheduled_backtest,
            trigger='cron',
            hour=17,
            minute=30,
            id='daily_backtest_job',
            name='每日台灣時間 17:30 執行策略回測',
            replace_existing=True
        )
        logger.info("✅ 已設定每日策略回測排程 (17:30)。")
    else:
        logger.warning("⚠️ 由於模組導入失敗，每日自動回測功能已停用。")
    
    # 啟動排程器
    scheduler.start()
    logger.info("🚀 排程器已啟動。")
    
    # 應用程式結束時優雅地關閉排程器
    atexit.register(lambda: scheduler.shutdown())
    
    logger.info("🚀 啟動整合版 AI 策略分析與市場分析平台...")
    logger.info("📊 策略訓練平台訪問: http://localhost:5001/trainer")
    logger.info("📈 市場分析平台訪問: http://localhost:5001/")
    
    # 在生產環境中應使用 WSGI 伺服器如 Gunicorn
    # debug 設為 False 是很重要的，因為 Flask 的自動重載器會導致排程任務被初始化兩次
    app.run(debug=False, host='0.0.0.0', port=5001)
