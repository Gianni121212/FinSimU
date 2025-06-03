# v2.py
import os
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
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mysql.connector

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# 建立需要的資料夾
for path in ['static/charts', 'static/data']:
    os.makedirs(path, exist_ok=True)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# ------------------ 股票分析功能 (原 v2.py) ------------------
class StockAnalyzer:
    def __init__(self, ticker: str, api_key: str, period: str = "10y", market: str = "TW"):
        self.ticker = ticker.strip()
        if market == "TW" and "." not in self.ticker:
            self.ticker = f"{self.ticker}.TW"
        self.period = period
        self.market = market
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

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.0-flash-thinking-exp")
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')

        self._get_data()
        self._get_financial_data()
        self._calculate_indicators()

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
            df['MA120'] = ta.sma(df['Close'], length=120)
            df['MA240'] = ta.sma(df['Close'], length=240)
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

            # PSY
            period_psy = 12
            up_day = df['Close'] > df['Close'].shift(1)
            df['PSY'] = up_day.rolling(period_psy).mean() * 100

            # VR
            period_vr = 26
            df['UpVol'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], 0)
            df['DownVol'] = np.where(df['Close'] < df['Close'].shift(1), df['Volume'], 0)
            df['FlatVol'] = np.where(df['Close'] == df['Close'].shift(1), df['Volume'], 0)
            sum_up = df['UpVol'].rolling(period_vr).sum()
            sum_down = df['DownVol'].rolling(period_vr).sum()
            sum_flat = df['FlatVol'].rolling(period_vr).sum()
            df['VR'] = np.where(
                (sum_down + 0.5 * sum_flat) == 0,
                np.nan,
                (sum_up + 0.5 * sum_flat) / (sum_down + 0.5 * sum_flat) * 100
            )

            # BIAS6
            ma_bias6 = ta.sma(df['Close'], length=6)
            df['BIAS6'] = (df['Close'] - ma_bias6) / ma_bias6 * 100

            # AR, BR
            period_arbr = 26
            df['HO'] = df['High'] - df['Open']
            df['OL'] = df['Open'] - df['Low']
            sum_HO = df['HO'].rolling(period_arbr).sum()
            sum_OL = df['OL'].rolling(period_arbr).sum()
            df['AR'] = np.where(sum_OL == 0, np.nan, sum_HO / sum_OL * 100)
            df['HPC'] = df['High'] - df['Close'].shift(1)
            df['PCL'] = df['Close'].shift(1) - df['Low']
            sum_HPC = df['HPC'].rolling(period_arbr).sum()
            sum_PCL = df['PCL'].rolling(period_arbr).sum()
            df['BR'] = np.where(sum_PCL == 0, np.nan, sum_HPC / sum_PCL * 100)

            self.data = df
            logging.info("技術指標計算完成: %s", self.ticker)
        except Exception as e:
            logging.error("計算技術指標時發生錯誤: %s", e)
            raise

    def _calculate_sentiment_score(self):
        try:
            last = self.data.iloc[-1]
            score = 0
            # RSI
            if last['RSI'] < 30:
                score += 1
            elif last['RSI'] > 70:
                score -= 1
            # MACD
            if last['MACD'] > last['MACD_signal']:
                score += 1
            else:
                score -= 1
            # K/D
            if last['K'] > last['D']:
                score += 1
            else:
                score -= 1
            # VR
            if pd.notnull(last['VR']):
                if last['VR'] > 150:
                    score += 1
                elif last['VR'] < 50:
                    score -= 1
            # BIAS6
            if pd.notnull(last['BIAS6']):
                if last['BIAS6'] > 5:
                    score -= 1
                elif last['BIAS6'] < -5:
                    score += 1

            if score <= -2:
                label = "看淡"
            elif -1 <= score <= 1:
                label = "中立"
            else:
                label = "看好"
            return score, label
        except Exception as e:
            logging.error("計算情緒分數時發生錯誤: %s", e)
            return 0, "中立"

    def _plot_gauge(self, score, label):
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"指標情緒: {label}"},
                gauge={
                    'axis': {'range': [-5, 5]},
                    'bar': {
                        'color': "green" if score > 0 else ("red" if score < 0 else "gray")
                    },
                    'steps': [
                        {'range': [-5, -2], 'color': 'rgba(255,0,0,0.3)'},
                        {'range': [-2, -1], 'color': 'rgba(255,0,0,0.15)'},
                        {'range': [-1, 1],  'color': 'rgba(128,128,128,0.15)'},
                        {'range': [1, 2],   'color': 'rgba(0,255,0,0.15)'},
                        {'range': [2, 5],   'color': 'rgba(0,255,0,0.3)'}
                    ]
                }
            ))
            fig.update_layout(width=400, height=400, margin=dict(l=50, r=50, t=50, b=50))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            gauge_file = f"gauge_{self.ticker}_{timestamp}.html"
            gauge_path = os.path.join('static', 'charts', gauge_file)
            fig.write_html(gauge_path)
            return gauge_path
        except Exception as e:
            logging.error("生成儀表板時發生錯誤: %s", e)
            return None

    def calculate_fibonacci_levels(self, window=60):
        try:
            recent_data = self.data.tail(window)
            max_price = recent_data['High'].max()
            min_price = recent_data['Low'].min()
            diff = max_price - min_price
            levels = {
                '0.0': min_price,
                '0.236': min_price + 0.236 * diff,
                '0.382': min_price + 0.382 * diff,
                '0.5': min_price + 0.5 * diff,
                '0.618': min_price + 0.618 * diff,
                '1.0': max_price
            }
            return levels
        except Exception as e:
            logging.error("計算 Fibonacci 水平時發生錯誤: %s", e)
            return {}

    def identify_patterns(self):
        patterns = []
        try:
            if len(self.data) >= 2:
                last = self.data.iloc[-1]
                prev = self.data.iloc[-2]
                if last['MA5'] > last['MA20'] and prev['MA5'] <= prev['MA20']:
                    patterns.append("MA5向上穿越MA20 (黃金交叉)")
                elif last['MA5'] < last['MA20'] and prev['MA5'] >= prev['MA20']:
                    patterns.append("MA5向下穿越MA20 (死亡交叉)")

            if len(self.data) >= 5:
                recent_highs = self.data['High'].tail(5)
                recent_lows = self.data['Low'].tail(5)
                # 頭肩頂示範
                if (recent_highs.iloc[1] < recent_highs.iloc[2] > recent_highs.iloc[3] and
                    recent_highs.iloc[0] < recent_highs.iloc[2] and
                    recent_highs.iloc[4] < recent_highs.iloc[2] and
                    recent_lows.iloc[1] > recent_lows.iloc[3]):
                    patterns.append("潛在頭肩頂形態")

                # W底示範
                closes = self.data['Close'].tail(5)
                if (closes.iloc[0] > closes.iloc[1] < closes.iloc[2] and
                    closes.iloc[2] > closes.iloc[3] < closes.iloc[4] and
                    closes.iloc[1] < closes.iloc[3] and closes.iloc[4] > closes.iloc[2]):
                    patterns.append("潛在雙底 (W 底)")

                # M頂示範
                if (closes.iloc[0] < closes.iloc[1] > closes.iloc[2] and
                    closes.iloc[2] < closes.iloc[3] > closes.iloc[4] and
                    closes.iloc[1] > closes.iloc[3] and closes.iloc[4] < closes.iloc[2]):
                    patterns.append("潛在雙頂 (M 頂)")

            if len(self.data) >= 10:
                recent_highs = self.data['High'].tail(10)
                recent_lows = self.data['Low'].tail(10)
                high_mean = recent_highs.mean()
                high_std = recent_highs.std()
                if (high_std < 0.02 * high_mean and
                    recent_lows.iloc[-1] > recent_lows.iloc[-3] > recent_lows.iloc[-5]):
                    patterns.append("潛在上升三角形")

                low_mean = recent_lows.mean()
                low_std = recent_lows.std()
                if (low_std < 0.02 * low_mean and
                    recent_highs.iloc[-1] < recent_highs.iloc[-3] < recent_highs.iloc[-5]):
                    patterns.append("潛在下降三角形")

        except Exception as e:
            logging.error("識別技術形態時發生錯誤: %s", e)
        return patterns

    def get_recent_news(self, days=30, num_news=10):
        try:
            if self.market == "TW":
                query = self.ticker.replace('.TW', '')
            else:
                query = self.company_name if self.company_name else self.ticker

            encoded_query = urllib.parse.quote(query)
            if self.market == "TW":
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}+stock&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            else:
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}+stock&hl=en-US&gl=US&ceid=US:en"

            feed = feedparser.parse(rss_url)
            recent_news = []
            now = dt.datetime.now()

            for entry in feed.entries:
                try:
                    published_time = dt.datetime(*entry.published_parsed[:6])
                except Exception:
                    published_time = now

                if (now - published_time).days <= days:
                    news_entry = {
                        'title': entry.title,
                        'link': entry.link,
                        'date': published_time.strftime('%Y-%m-%d'),
                        'source': entry.get('source', {}).get('title', 'Google News')
                    }
                    try:
                        sentiment = self.sentiment_analyzer(entry.title)[0]
                        news_entry['sentiment'] = sentiment['label']
                        news_entry['sentiment_score'] = sentiment['score']
                    except Exception as e:
                        logging.error("情緒分析失敗: %s", e)
                        news_entry['sentiment'] = 'N/A'
                        news_entry['sentiment_score'] = 0

                    recent_news.append(news_entry)
                    if len(recent_news) >= num_news:
                        break

            if not recent_news:
                return self._get_fallback_news()
            return recent_news

        except Exception as e:
            logging.error("取得近期新聞時發生錯誤: %s", e)
            return self._get_fallback_news()

    def _get_fallback_news(self):
        try:
            now_str = dt.datetime.now().strftime('%Y-%m-%d')
            if self.market == "TW":
                return [{
                    'title': f'請訪問 Yahoo 股市或工商時報查看 {self.ticker} 的最新新聞',
                    'date': now_str,
                    'source': '系統訊息',
                    'link': f'https://tw.stock.yahoo.com/quote/{self.ticker.replace(".TW", "")}/news',
                    'sentiment': 'N/A',
                    'sentiment_score': 0
                }]
            else:
                return [{
                    'title': f'請訪問 Yahoo Finance 或 MarketWatch 查看 {self.ticker} 的最新新聞',
                    'date': now_str,
                    'source': '系統訊息',
                    'link': f'https://finance.yahoo.com/quote/{self.ticker}/news',
                    'sentiment': 'N/A',
                    'sentiment_score': 0
                }]
        except Exception as e:
            logging.error("備用新聞方法發生錯誤: %s", e)
            return [{
                'title': '暫時無法獲取相關新聞',
                'date': dt.datetime.now().strftime('%Y-%m-%d'),
                'source': '系統訊息',
                'link': '#',
                'sentiment': 'N/A',
                'sentiment_score': 0
            }]

    def generate_strategy(self):
        try:
            last_row = self.data.iloc[-1]
            sentiment_summary = [news['sentiment'] for news in self.get_recent_news()]
            positive_count = sentiment_summary.count('Positive')
            negative_count = sentiment_summary.count('Negative')
            total_news = len(sentiment_summary)
            sentiment_ratio = (positive_count - negative_count) / total_news if total_news > 0 else 0

            if last_row['RSI'] < 30 and last_row['MACD'] > last_row['MACD_signal'] and sentiment_ratio > 0:
                return "Buy"
            elif last_row['RSI'] > 70 and sentiment_ratio < 0:
                return "Sell"
            else:
                return "Hold"

        except Exception as e:
            logging.error("生成策略時發生錯誤: %s", e)
            return "Hold"

    def plot_analysis(self, days=180, ma_lines=['MA5', 'MA20', 'MA120', 'MA240']):
        try:
            plot_data = self.data.tail(days).copy()

            specs = [
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
            ]

            fig = make_subplots(
                rows=8,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    "價格與均線", "RSI", "MACD", "KDJ",
                    "Williams %R", "PSY & VR", "BIAS", "AR & BR"
                ),
                row_heights=[0.35, 0.10, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
                specs=specs
            )

            # 蠟燭圖
            fig.add_trace(
                go.Candlestick(
                    x=plot_data.index,
                    open=plot_data['Open'],
                    high=plot_data['High'],
                    low=plot_data['Low'],
                    close=plot_data['Close'],
                    name='蠟燭圖',
                    showlegend=False
                ),
                row=1,
                col=1
            )

            # MA 顏色參考
            ma_colors = {'MA5': 'blue', 'MA20': 'orange', 'MA120': 'purple', 'MA240': 'brown'}

            for ma in ma_lines:
                if ma in plot_data.columns:
                    color = ma_colors.get(ma, 'black')
                    label = ma if ma not in ['MA120', 'MA240'] else (
                        f"{ma} (半年線)" if ma == 'MA120' else f"{ma} (年線)"
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data[ma],
                            mode='lines',
                            name=label,
                            line=dict(color=color, width=1),
                            showlegend=True
                        ),
                        row=1,
                        col=1
                    )

            # BBands
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['BB_upper'],
                    mode='lines',
                    name='BB 上軌',
                    line=dict(color='grey', width=1),
                    showlegend=True
                ),
                row=1,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['BB_lower'],
                    mode='lines',
                    name='BB 下軌',
                    line=dict(color='grey', width=1),
                    showlegend=True
                ),
                row=1,
                col=1
            )

            # RSI
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1),
                    showlegend=False
                ),
                row=2,
                col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超買", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超賣", row=2, col=1)

            # MACD
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['MACD'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=1),
                    showlegend=False
                ),
                row=3,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['MACD_signal'],
                    mode='lines',
                    name='MACD Signal',
                    line=dict(color='orange', width=1),
                    showlegend=False
                ),
                row=3,
                col=1
            )
            macd_hist = plot_data['MACD'] - plot_data['MACD_signal']
            fig.add_trace(
                go.Bar(
                    x=plot_data.index,
                    y=macd_hist,
                    name='MACD Histogram',
                    marker_color='grey',
                    opacity=0.5,
                    showlegend=False
                ),
                row=3,
                col=1
            )

            # KDJ
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['K'],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name='K',
                    showlegend=False
                ),
                row=4,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['D'],
                    mode='lines',
                    line=dict(color='orange', width=1),
                    name='D',
                    showlegend=False
                ),
                row=4,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['J'],
                    mode='lines',
                    line=dict(color='green', width=1),
                    name='J',
                    showlegend=False
                ),
                row=4,
                col=1
            )

            # WMSR
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['WMSR'],
                    mode='lines',
                    line=dict(color='purple', width=1),
                    name='WMSR',
                    showlegend=False
                ),
                row=5,
                col=1
            )
            fig.add_hline(y=-20, line_dash="dash", line_color="red", annotation_text="超買", row=5, col=1)
            fig.add_hline(y=-80, line_dash="dash", line_color="green", annotation_text="超賣", row=5, col=1)

            # PSY & VR (同一個 row，雙 Y 軸)
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['PSY'],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name='PSY',
                    showlegend=False
                ),
                row=6,
                col=1,
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['VR'],
                    mode='lines',
                    line=dict(color='orange', width=1),
                    name='VR',
                    showlegend=False
                ),
                row=6,
                col=1,
                secondary_y=True
            )
            fig.update_yaxes(title_text="PSY", row=6, col=1, secondary_y=False)
            fig.update_yaxes(title_text="VR", row=6, col=1, secondary_y=True)

            # BIAS6
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['BIAS6'],
                    mode='lines',
                    line=dict(color='green', width=1),
                    name='BIAS6',
                    showlegend=False
                ),
                row=7,
                col=1
            )
            fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="超買", row=7, col=1)
            fig.add_hline(y=-5, line_dash="dash", line_color="green", annotation_text="超賣", row=7, col=1)

            # AR & BR
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['AR'],
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name='AR',
                    showlegend=False
                ),
                row=8,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['BR'],
                    mode='lines',
                    line=dict(color='orange', width=1),
                    name='BR',
                    showlegend=False
                ),
                row=8,
                col=1
            )

            fig.update_layout(
                title={'text': f"{self.company_name} ({self.ticker}) 技術分析圖表", 'x':0.5, 'y':0.95, 'xanchor': 'center'},
                height=1800,
                width=1000,
                margin=dict(l=50, r=50, t=80, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template="plotly_white",
                showlegend=True
            )
            fig.update_xaxes(rangeslider_visible=False)

            fig.update_yaxes(title_text=f"價格 ({self.currency})", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)
            fig.update_yaxes(title_text="KDJ", row=4, col=1)
            fig.update_yaxes(title_text="WMSR", row=5, col=1)
            fig.update_yaxes(title_text="BIAS", row=7, col=1)
            fig.update_yaxes(title_text="AR & BR", row=8, col=1)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{self.ticker}_full_analysis_{timestamp}.html"
            html_path = os.path.join('static', 'charts', file_name)
            fig.write_html(html_path)

            logging.info("完整指標圖表生成並儲存至 %s", html_path)
            return html_path

        except Exception as e:
            logging.error("生成完整指標圖表時發生錯誤: %s", e)
            raise

    def format_market_cap(self):
        if self.market_cap == 'N/A' or not isinstance(self.market_cap, (int, float)):
            return 'N/A'
        value = float(self.market_cap)
        units_tw = ['元', '萬', '億', '兆']
        units_us = ['USD', '萬', '億', '兆']
        units = units_tw if self.currency == 'TWD' else units_us
        divisor = 1
        unit_idx = 0
        while value >= 10000 and unit_idx < len(units) - 1:
            value /= 10000
            unit_idx += 1
            divisor *= 10000
        return f"{value:.2f} {units[unit_idx]} {self.currency}"

    def generate_ai_analysis(self, days=180):
        try:
            if len(self.data) < 2:
                raise ValueError("數據不足，無法計算價格變動。")
            last_price = self.data['Close'].iloc[-1]
            prev_price = self.data['Close'].iloc[-2]
            price_change = ((last_price - prev_price) / prev_price * 100) if prev_price else 0
            recent_news = self.get_recent_news()
            news_summary = "\n".join([f"- [{news['date']}] {news['title']} (來源: {news['source']}, 情緒: {news['sentiment']})"
                                      for news in recent_news])
            technical_status = {
                'last_price': last_price,
                'price_change': price_change,
                'rsi': self.data['RSI'].iloc[-1],
                'ma5': self.data['MA5'].iloc[-1],
                'ma20': self.data['MA20'].iloc[-1],
                'ma120': self.data['MA120'].iloc[-1],
                'ma240': self.data['MA240'].iloc[-1],
                'bb_upper': self.data['BB_upper'].iloc[-1],
                'bb_lower': self.data['BB_lower'].iloc[-1],
                'macd': self.data['MACD'].iloc[-1],
                'macd_signal': self.data['MACD_signal'].iloc[-1],
                'kdj_k': self.data['K'].iloc[-1],
                'kdj_d': self.data['D'].iloc[-1],
                'kdj_j': self.data['J'].iloc[-1],
                'patterns': self.identify_patterns(),
                'fib_levels': self.calculate_fibonacci_levels(),
                'financial_head': self.financials_head
            }
            prompt = f"""
你是一位擁有三十年經驗的專業股市分析師，請根據以下 {self.company_name} ({self.ticker}) 的數據和技術指標進行深入的股市分析，並使用專業術語（如動能、趨勢、波動性）解釋其影響不要透漏自己得身分，開頭直接:以下是針對(公司名稱，如果是台灣公司請盡量用中文名稱)的分析，然後幫我排版排好一點，盡量簡潔有力一點，不要說出今日日期):
【基本資訊】
- 公司名稱: {self.company_name}
- 股票代碼: {self.ticker}
- 最新收盤價: {last_price:.2f} {self.currency}
- 日漲跌: {price_change:.2f}%

【財務數據】
- 市盈率 (Trailing P/E): {self.pe_ratio}
- 市值: {self.format_market_cap()}
- 預期市盈率 (Forward P/E): {self.forward_pe}
- 利潤率: {self.profit_margins}
- EPS: {self.eps}
- ROE: {self.roe}

【技術指標】
- RSI(14): {technical_status['rsi']:.2f}
- MA5: {technical_status['ma5']:.2f}
- MA20: {technical_status['ma20']:.2f}
- MA120 (半年線): {technical_status['ma120']:.2f}
- MA240 (年線): {technical_status['ma240']:.2f}
- 布林帶上軌: {technical_status['bb_upper']:.2f}
- 布林帶下軌: {technical_status['bb_lower']:.2f}
- MACD: {technical_status['macd']:.2f}
- MACD Signal: {technical_status['macd_signal']:.2f}
- KDJ (K/D/J): {technical_status['kdj_k']:.2f}/{technical_status['kdj_d']:.2f}/{technical_status['kdj_j']:.2f}
- WMSR (Williams %R): {self.data['WMSR'].iloc[-1]:.2f}
- PSY: {self.data['PSY'].iloc[-1]:.2f}
- VR: {self.data['VR'].iloc[-1]:.2f}
- BIAS6: {self.data['BIAS6'].iloc[-1]:.2f}
- AR: {self.data['AR'].iloc[-1]:.2f}
- BR: {self.data['BR'].iloc[-1]:.2f}

【技術形態】
- {', '.join(technical_status['patterns']) if technical_status['patterns'] else '無明顯技術形態'}

【Fibonacci 回檔水平】
- {', '.join([f"{k}: {v:.2f}" for k, v in technical_status['fib_levels'].items()])}

【近期新聞】
{news_summary}

請按以下結構提供分析：
1. 近期價格走勢評估（分析動能與趨勢）
2. 支撐與壓力位分析（可根據Fibonacci、布林帶還有均線）
3. 短期走勢預測（基於 MACD、RSI 和 KDJ）
4. 策略建議與風險分析（提供買入、賣出或持有建議，並說明理由）
5. 近期新聞與情緒影響分析
6. 綜合結論

最後請在分析結尾加上免責聲明："本分析報告僅供參考，不構成投資建議。投資者應自行承擔投資風險。"
"""
            response = self.model.generate_content(prompt)
            response_text = response.text.replace('\n', '<br>')
            return response_text
        except Exception as e:
            logging.error("生成 AI 分析時發生錯誤: %s", e)
            return f"生成 AI 分析時發生錯誤: {str(e)}"

    def generate_financial_analysis(self, days=180):
        try:
            prompt = f"""
你是一位擁有三十年經驗的專業財金分析師，請根據以下 {self.company_name} ({self.ticker}) 的數據和財務數據進行深入的分析，
並使用專業術語解釋(不要透漏自己得身分，開頭直接:以下是針對(公司名稱，如果是台灣公司請盡量用中文名稱)的分析，然後幫我排版排好一點，盡量簡潔有力一點，不要說出今日日期):

【基本資訊】
- 公司名稱: {self.company_name}
- 股票代碼: {self.ticker}

【財務指標】
- 淨利率: {self.net_profit_margin_str}
- 流動比率: {self.current_ratio_str}
- 市盈率 (Trailing P/E): {self.pe_ratio}
- 預期市盈率 (Forward P/E): {self.forward_pe}
- 利潤率: {self.profit_margins}
- EPS: {self.eps}
- ROE: {self.roe}
- 年度損益表:\n{self.financials_head}
- 年度資產負債表:\n{self.balance_sheet_head}
- 年度現金流量表:\n{self.cashflow_head}

請按以下結構提供分析：
1. 財務健康狀況評估（盈利能力、流動性和負債情況）
2. 成長性分析 (歷史成長與未來展望)
3. 估值分析（基於市盈率、EPS等）
4. 財務風險提示
5. 綜合結論

最後在分析末尾加上免責聲明："本分析報告僅供參考，不構成投資建議。投資者應自行承擔投資風險。"
"""
            response = self.model.generate_content(prompt)
            response_text = response.text.replace('\n', '<br>')
            return response_text
        except Exception as e:
            logging.error("生成財務分析時發生錯誤: %s", e)
            return f"生成財務分析時發生錯誤: {str(e)}"

    def run_full_analysis(self, days_to_analyze=180, ma_lines=['MA5', 'MA20', 'MA120', 'MA240']):
        try:
            ai_analysis = self.generate_ai_analysis(days_to_analyze)
            financial_analysis = self.generate_financial_analysis(days_to_analyze)
            chart_path = self.plot_analysis(days_to_analyze, ma_lines)
            strategy = self.generate_strategy()
            score, label = self._calculate_sentiment_score()
            gauge_path = self._plot_gauge(score, label)

            last_row = self.data.iloc[-1]
            summary = {
                "company_name": self.company_name,
                "ticker": self.ticker,
                "close_price": float(last_row['Close']),
                "open_price": float(last_row['Open']),
                "high_price": float(last_row['High']),
                "low_price": float(last_row['Low']),
                "currency": self.currency,
                "rsi": float(last_row['RSI']),
                "ma5": float(last_row['MA5']),
                "ma20": float(last_row['MA20']),
                "ma120": float(last_row['MA120']),
                "ma240": float(last_row['MA240']),
                "bb_upper": float(last_row['BB_upper']),
                "bb_lower": float(last_row['BB_lower']),
                "macd": float(last_row['MACD']),
                "macd_signal": float(last_row['MACD_signal']),
                "kdj_k": float(last_row['K']),
                "kdj_d": float(last_row['D']),
                "kdj_j": float(last_row['J']),
                "wmsr": float(last_row['WMSR']),
                "psy": float(last_row['PSY']),
                "vr": float(last_row['VR']),
                "bias6": float(last_row['BIAS6']),
                "ar": float(last_row['AR']),
                "br": float(last_row['BR']),
                "patterns": self.identify_patterns(),
                "fib_levels": {k: float(v) for k, v in self.calculate_fibonacci_levels().items()},
                "strategy": strategy,
                "pe_ratio": self.pe_ratio,
                "market_cap": self.format_market_cap(),
                "forward_pe": self.forward_pe,
                "eps": self.eps,
                "sentiment_score": score,
                "sentiment_label": label
            }
            timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

            # 如果要儲存 JSON 結果，可加上以下:
            # result_file = os.path.join('static', 'data', f"{self.ticker}_analysis_{timestamp}.json")
            # with open(result_file, 'w', encoding='utf-8') as f:
            #     json.dump({
            #         'a_analysis': ai_analysis,
            #         'f_analysis': financial_analysis,
            #         'summary': summary,
            #         'chart_path': chart_path,
            #         'gauge_path': gauge_path,
            #         'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #     }, f, ensure_ascii=False, indent=4)

            return {
                'a_analysis': ai_analysis,
                'f_analysis': financial_analysis,
                'summary': summary,
                'chart_path': chart_path,
                'gauge_path': gauge_path,
                'timestamp': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logging.error("完整分析流程發生錯誤: %s", e)
            raise

# ------------------ 股票推薦功能 (合併自 stock_recommend.py) ------------------

# 設定 Gemini API 金鑰與模型 (全域用於推薦功能)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("請先在環境變數中設定 GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
portfolio_model = genai.GenerativeModel("models/gemini-2.0-flash-thinking-exp")

db_config = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASS", "0912559910"),
    'database': os.getenv("DB_NAME", "testdb")
}

@app.route("/api/portfolio", methods=["POST"])
def portfolio_api():
    try:
        data = request.get_json()
        # 固定 RSI 門檻，由後端設定
        fixed_rsi_threshold = 100
        category_pref = data.get("category", "").strip()
    except Exception as e:
        logging.error("輸入資料格式錯誤: %s", e)
        return jsonify({"error": "請確認輸入格式正確"}), 400

    query = """
        SELECT symbol, name, rsi, ma5, ma20, macd, macd_signal
        FROM stocks
        WHERE rsi < %s AND ma5 > ma20 AND macd > macd_signal
    """
    params = [fixed_rsi_threshold]

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
    except Exception as e:
        logging.error("資料庫查詢錯誤: %s", e)
        return jsonify({"error": "資料庫錯誤"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    if not rows:
        return jsonify({"error": "找不到符合條件的股票"}), 404

    candidate_stocks = []
    for row in rows:
        candidate_stocks.append({
            "ticker": row[0],
            "company_name": row[1],
            "rsi": row[2],
            "ma5": row[3],
            "ma20": row[4],
            "macd": row[5],
            "macd_signal": row[6]
        })

    prompt_lines = [
        "你是一位專業的投資組合管理專家，請根據以下候選股票的數據及用戶偏好，選出最適合投資的股票並進行資金分配。",
        f"用戶偏好: {category_pref}" if category_pref else "用戶未提供特定股票類別偏好。",
        "請考慮每檔股票的技術指標與基本面數據、風險與報酬，並給出具體的投資比例建議。候選股票如下："
    ]
    for stock in candidate_stocks:
        prompt_lines.append(
            f"股票代碼: {stock['ticker']}, 公司名稱: {stock['company_name']}, RSI: {stock['rsi']:.2f}, MA5: {stock['ma5']:.2f}, MA20: {stock['ma20']:.2f}, MACD: {stock['macd']:.2f}, MACD Signal: {stock['macd_signal']:.2f}"
        )
    prompt_lines.append(
        "請根據上述數據選出 3 至 5 檔股票組成投資組合，並依據合理的資金分配原則提出具體的比例建議，"
        "說明每檔股票的優點與風險。最後，請在報告結尾加入以下免責聲明："
        "本分析報告僅供參考，不構成投資建議。投資者應自行承擔投資風險。"
    )
    prompt = "\n".join(prompt_lines)

    try:
        response = portfolio_model.generate_content(prompt)
        portfolio_analysis = response.text
    except Exception as e:
        logging.error("Gemini AI 呼叫錯誤: %s", e)
        portfolio_analysis = f"呼叫 Gemini AI 時發生錯誤: {e}"

    result = {
        "portfolio_analysis": portfolio_analysis,
        "candidate_stocks": candidate_stocks
    }
    return jsonify(result)

# ------------------ Flask 路由 ---------------------

# 預設封面頁
@app.route('/')
def cover():
    return render_template('cover.html')

# 把原本 testindex.html 改成使用 example.html
@app.route('/index')
def index():
    return render_template('example.html')

load_dotenv()

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.form
        ticker = data.get('ticker', '').strip()
        market = data.get('market', 'TW')
        period = '10y'
        days = int(data.get('days', 180))
        ma_lines = data.getlist('ma_lines')
        api_key = os.getenv("GEMINI_API_KEY")

        if not ticker:
            return jsonify({'error': '請輸入股票代碼'}), 400
        if not api_key:
            return jsonify({'error': 'API 金鑰無效，請檢查 .env 設定'}), 500

        analyzer = StockAnalyzer(ticker=ticker, api_key=api_key, period=period, market=market)
        results = analyzer.run_full_analysis(days_to_analyze=days, ma_lines=ma_lines or ['MA5', 'MA20'])

        return jsonify({'status': 'success', 'analysis_data': results})

    except Exception as e:
        logging.error("分析流程發生錯誤: %s", e)
        return jsonify({'error': f"分析過程發生錯誤: {str(e)}"}), 500

@app.route('/get_stock_change/<ticker>', methods=['GET'])
def get_stock_change(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        change_percent = info.get('regularMarketChangePercent', 0)
        return jsonify({'change': change_percent})
    except Exception as e:
        logging.error("取得股票漲跌幅錯誤: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
