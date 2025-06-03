import os
import json # Not strictly needed here anymore but good to have
import datetime
import logging
import secrets
# import pymysql # Not needed if StockAnalyzer doesn't write to DB
import yfinance as yf
import pandas as pd
from flask import Flask, render_template, request # Removed jsonify as we render template directly
from dotenv import load_dotenv
import google.generativeai as genai
# Conditional import for transformers
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    hf_pipeline = None # Define it as None if not available
    print("WARNING: Hugging Face Transformers library not found. Sentiment analysis will be disabled.")

import pandas_ta as ta
import feedparser
import urllib.parse
import re # For cleaning up AI response if needed

# --- 基本設定 ---
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY_V5", secrets.token_hex(16))

# --- 日誌設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# --- Gemini API 設定 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
stock_analyzer_gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        stock_analyzer_gemini_model = genai.GenerativeModel("models/gemini-1.5-flash-latest") # Or your preferred model
        logger.info("Gemini Model configured successfully for StockAnalyzer (V5).")
    except Exception as gemini_err:
        logger.error(f"Failed to configure Gemini models for StockAnalyzer (V5): {gemini_err}", exc_info=True)
else:
    logger.warning("GEMINI_API_KEY not found. AI analysis in StockAnalyzer (V5) will be disabled.")


# --- StockAnalyzer Class (Copied and adapted from v4.py) ---
class StockAnalyzer:
    def __init__(self, ticker: str, api_key: str, period: str = "5y", market: str = "US", temperature: float = 0.7, gemini_model_instance=None):
        self.ticker_input = ticker.strip()
        self.ticker = self.ticker_input
        self.market = market.upper()

        if self.market == "TW" and not self.ticker.endswith(".TW") and self.ticker.isdigit() and 4 <= len(self.ticker) <= 6:
            self.ticker = f"{self.ticker}.TW"
        elif self.market == "US" and self.ticker.endswith(".TW"):
            self.ticker = self.ticker.replace(".TW", "")
        
        self.api_key = api_key # Retained if model is created internally
        self.period = period
        self.temperature = max(0.0, min(1.0, temperature))
        self.stock_yf_object = yf.Ticker(self.ticker)
        self.data = pd.DataFrame()
        self.company_name = self.ticker 
        self.currency = 'USD' if self.market == 'US' else 'TWD'
        self.pe_ratio, self.market_cap, self.eps, self.roe = None, None, None, None
        self.net_profit_margin_str, self.current_ratio_str = "N/A", "N/A"
        self.roe_display = "N/A" # For template
        
        self.gemini_model = gemini_model_instance
        if not self.gemini_model and GEMINI_API_KEY: # Fallback to create one if not passed
             try:
                self.gemini_model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
                logger.info(f"StockAnalyzer for {self.ticker} created its own Gemini model instance.")
             except Exception as e:
                logger.error(f"StockAnalyzer for {self.ticker} failed to create Gemini model: {e}")
        
        self.sentiment_analyzer = None
        
        logger.info(f"StockAnalyzer V5 initialized for {self.ticker} (Market: {self.market})")
        try:
            self._get_data()
            if not self.data.empty:
                self._get_financial_data()
                self._calculate_indicators()
            # No _update_db_data call
        except Exception as e:
             logger.error(f"StockAnalyzer V5 initialization failed for ({self.ticker}): {e}", exc_info=True)
             self.company_name = self.company_name or self.ticker

    def _load_sentiment_analyzer_if_needed(self):
        if self.sentiment_analyzer is None and TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_analyzer = hf_pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')
                logger.info("FinBERT sentiment analyzer loaded for StockAnalyzer V5.")
            except Exception as e:
                logger.error(f"Failed to load FinBERT sentiment analyzer: {e}")
                self.sentiment_analyzer = "UNAVAILABLE"
        elif not TRANSFORMERS_AVAILABLE:
            self.sentiment_analyzer = "UNAVAILABLE"


    def _get_data(self):
        # Copied from v4.py's StockAnalyzer, ensure it's the version you prefer
        try:
            self.data = self.stock_yf_object.history(period=self.period, timeout=15)
            if self.data.empty: raise ValueError(f"No historical data for {self.ticker}.")
            info = self.stock_yf_object.info
            if not info:
                logger.warning(f"Primary .info failed for {self.ticker}. Trying .fast_info.")
                try: info = self.stock_yf_object.fast_info
                except Exception as fast_info_err: logger.error(f"Both .info and .fast_info failed for {self.ticker}: {fast_info_err}")
            
            self.company_name = info.get('longName', info.get('shortName', self.ticker))
            self.currency = info.get('currency', 'USD' if self.market == 'US' else 'TWD')
            logger.info(f"Successfully fetched data and info for {self.ticker}")
        except Exception as e:
            logger.error(f"Error fetching stock data for {self.ticker}: {e}")
            self.data = pd.DataFrame()

    def _get_financial_data(self):
        # Copied from v4.py's StockAnalyzer, ensure it's the version you prefer
        logger.info(f"[{self.ticker}] Attempting to get financial data...")
        try:
            info = self.stock_yf_object.info or {}
            self.pe_ratio = info.get('trailingPE')
            self.market_cap = info.get('marketCap')
            self.eps = info.get('trailingEps')
            self.roe = info.get('returnOnEquity')
            self.roe_display = f"{(self.roe * 100):.2f}%" if self.roe is not None else "N/A"
            self.current_ratio_str = f"{info.get('currentRatio'):.2f}" if info.get('currentRatio') is not None else "N/A"
            
            if info.get('profitMargins') is not None:
                 self.net_profit_margin_str = f"{info.get('profitMargins') * 100:.2f}%"
            else: # Fallback if profitMargins not in info
                try:
                    financials = self.stock_yf_object.financials
                    if not financials.empty:
                        net_income_series = financials.loc['Net Income'] if 'Net Income' in financials.index else (financials.loc['NetIncome'] if 'NetIncome' in financials.index else None)
                        total_revenue_series = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None
                        
                        if net_income_series is not None and total_revenue_series is not None and \
                           not net_income_series.empty and not total_revenue_series.empty:
                            net_income = net_income_series.iloc[0]
                            total_revenue = total_revenue_series.iloc[0]
                            if pd.notna(net_income) and pd.notna(total_revenue) and total_revenue != 0:
                                self.net_profit_margin_str = f"{(net_income / total_revenue) * 100:.2f}%"
                except Exception as fin_e:
                    logger.warning(f"[{self.ticker}] Could not fetch financials for net profit margin: {fin_e}")
            logger.info(f"[{self.ticker}] Financial data processed.")
        except Exception as e:
            logger.warning(f"[{self.ticker}] Error processing financial data: {e}.")
            self.pe_ratio = getattr(self, 'pe_ratio', None)
            self.market_cap = getattr(self, 'market_cap', None)
            self.eps = getattr(self, 'eps', None)
            self.roe = getattr(self, 'roe', None)
            self.roe_display = getattr(self, 'roe_display', "N/A")
            self.net_profit_margin_str = getattr(self, 'net_profit_margin_str', "N/A")
            self.current_ratio_str = getattr(self, 'current_ratio_str', "N/A")


    def _calculate_indicators(self):
        # Copied from v4.py's StockAnalyzer, ensure it's the version you prefer
        if self.data.empty: logger.warning(f"[{self.ticker}] No data for indicator calculation."); return
        df = self.data.copy()
        try:
            min_lengths = {'MA5': 5, 'MA20': 20, 'RSI': 13, 'MACD': 35, 'KDJ': 14, 'BBANDS': 20}
            if len(df) >= min_lengths['MA5']: df['MA5'] = ta.sma(df['Close'], length=5)
            if len(df) >= min_lengths['MA20']: df['MA20'] = ta.sma(df['Close'], length=20)
            if len(df) >= min_lengths['RSI']: df['RSI'] = ta.rsi(df['Close'], length=12)
            if len(df) >= min_lengths['MACD']:
                macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
                if macd_df is not None and not macd_df.empty:
                    df['MACD'] = macd_df.iloc[:,0]; df['MACD_signal'] = macd_df.iloc[:,1]
            if len(df) >= min_lengths['KDJ']:
                stoch_df = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3, smooth_k=3)
                if stoch_df is not None and not stoch_df.empty:
                    df['K'] = stoch_df.iloc[:,0]; df['D'] = stoch_df.iloc[:,1]; df['J'] = 3 * df['K'] - 2 * df['D']
            if len(df) >= min_lengths['BBANDS']:
                bbands = ta.bbands(df['Close'], length=20, std=2)
                if bbands is not None and not bbands.empty:
                    df['BB_Lower'] = bbands.iloc[:,0]; df['BB_Middle'] = bbands.iloc[:,1]; df['BB_Upper'] = bbands.iloc[:,2]
            self.data = df
            logger.info(f"[{self.ticker}] Technical indicators calculated.")
        except Exception as e:
            logger.error(f"[{self.ticker}] Error calculating indicators: {e}", exc_info=True)

    def _identify_patterns(self, days=30):
        # Copied from v4.py's StockAnalyzer, ensure it's the version you prefer
        if self.data.empty or len(self.data) < 2: return ["數據不足無法判斷形態"]
        patterns = []
        df_c = self.data.tail(max(2,days)).copy() # Use copy to avoid SettingWithCopyWarning
        
        required_cols = ['MA5', 'MA20', 'Close', 'BB_Upper', 'BB_Lower', 'MACD', 'MACD_signal', 'K', 'D', 'RSI']
        for col in required_cols: # Ensure columns exist before trying to access
            if col not in df_c.columns: df_c[col] = pd.NA 

        # MA Crossover
        if pd.notna(df_c['MA5'].iloc[-1]) and pd.notna(df_c['MA20'].iloc[-1]) and \
           pd.notna(df_c['MA5'].iloc[-2]) and pd.notna(df_c['MA20'].iloc[-2]):
            if df_c['MA5'].iloc[-2] < df_c['MA20'].iloc[-2] and df_c['MA5'].iloc[-1] > df_c['MA20'].iloc[-1]:
                patterns.append("均線黃金交叉 (MA5穿越MA20)")
            elif df_c['MA5'].iloc[-2] > df_c['MA20'].iloc[-2] and df_c['MA5'].iloc[-1] < df_c['MA20'].iloc[-1]:
                patterns.append("均線死亡交叉 (MA5跌破MA20)")
        # RSI
        if pd.notna(df_c['RSI'].iloc[-1]):
            rsi = df_c['RSI'].iloc[-1]
            if rsi > 70: patterns.append(f"RSI超買 ({rsi:.1f})")
            elif rsi < 30: patterns.append(f"RSI超賣 ({rsi:.1f})")
        # MACD
        if pd.notna(df_c['MACD'].iloc[-1]) and pd.notna(df_c['MACD_signal'].iloc[-1]) and \
           pd.notna(df_c['MACD'].iloc[-2]) and pd.notna(df_c['MACD_signal'].iloc[-2]):
            if df_c['MACD'].iloc[-2] < df_c['MACD_signal'].iloc[-2] and df_c['MACD'].iloc[-1] > df_c['MACD_signal'].iloc[-1]:
                patterns.append("MACD黃金交叉")
            elif df_c['MACD'].iloc[-2] > df_c['MACD_signal'].iloc[-2] and df_c['MACD'].iloc[-1] < df_c['MACD_signal'].iloc[-1]:
                patterns.append("MACD死亡交叉")
        return patterns if patterns else ["近期無顯著技術形態"]


    def _generate_chart(self, days=180):
        # Copied from v4.py's StockAnalyzer, ensure it's the version you prefer
        if self.data.empty or len(self.data) < 2: logger.warning(f"[{self.ticker}] Not enough data for chart."); return None
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            df_chart = self.data.tail(days).copy()
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3],
                                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]) # Add secondary_y for price chart

            # Price Candlestick
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                                         low=df_chart['Low'], close=df_chart['Close'], name='股價'), row=1, col=1)
            # MAs
            if 'MA5' in df_chart: fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA5'], name='MA5', line=dict(color='rgba(255,165,0,0.7)', width=1)), row=1, col=1)
            if 'MA20' in df_chart: fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], name='MA20', line=dict(color='rgba(0,0,255,0.7)', width=1)), row=1, col=1)
            # Volume
            vol_colors = ['#2ca02c' if row['Close'] >= row['Open'] else '#d62728' for _, row in df_chart.iterrows()]
            fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name='成交量', marker_color=vol_colors), row=2, col=1)
            
            fig.update_layout(title_text=f'{self.company_name or self.ticker} 技術分析圖 ({days}日)',
                              xaxis_rangeslider_visible=False, template="plotly_white", height=650,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig.update_yaxes(title_text="價格 ("+self.currency+")", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="成交量", row=2, col=1)

            chart_filename = f"{self.ticker.replace('.', '_').replace('^', '')}_v5_chart_{secrets.token_hex(3)}.html"
            static_charts_dir = os.path.join(app.static_folder, 'charts')
            os.makedirs(static_charts_dir, exist_ok=True)
            chart_path_on_disk = os.path.join(static_charts_dir, chart_filename)
            fig.write_html(chart_path_on_disk, full_html=False, include_plotlyjs='cdn')
            logger.info(f"Chart generated for {self.ticker} at {chart_path_on_disk}")
            return os.path.join('charts', chart_filename)
        except Exception as e:
            logger.error(f"[{self.ticker}] Error generating chart: {e}", exc_info=True)
            return None

    def _get_stock_news(self, days=7, num_news=5):
        # Copied from v4.py's StockAnalyzer, ensure it's the version you prefer
        self._load_sentiment_analyzer_if_needed()
        news_list = []
        logger.info(f"[{self.ticker}] Fetching news (max {num_news}, last {days} days)...")
        try:
            query_term = self.company_name or self.ticker
            if self.market == "TW": query_term = self.ticker.replace(".TW", "")
            lang, gl, ceid = ('zh-TW', 'TW', 'TW:zh-Hant') if self.market == 'TW' else ('en-US', 'US', 'US:en')
            search_query = f"{query_term} 股票 新聞" if self.market == 'TW' else f"{query_term} stock news"
            rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote_plus(search_query)}&hl={lang}&gl={gl}&ceid={ceid}"
            feed = feedparser.parse(rss_url)
            if not feed.entries: logger.warning(f"No news entries for {query_term} via Google News RSS.")
            for entry in feed.entries[:num_news]:
                title = entry.get('title', "N/A")
                sentiment_label = "N/A"
                if self.sentiment_analyzer and self.sentiment_analyzer != "UNAVAILABLE" and TRANSFORMERS_AVAILABLE:
                    try: sentiment_label = self.sentiment_analyzer(title[:512])[0]['label']
                    except Exception as se: logger.debug(f"Sentiment analysis failed for '{title[:30]}...': {se}")
                published_date_str = 'N/A'
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try: published_date_str = datetime.datetime(*(entry.published_parsed[:6])).strftime('%Y-%m-%d')
                    except: pass
                news_list.append({'title': title, 'link': entry.get('link', "#"), 'date': published_date_str,
                                  'source': entry.get('source', {}).get('title', 'Google News'), 'sentiment': sentiment_label })
        except Exception as e: logger.error(f"Error fetching news for {self.ticker}: {e}")
        return news_list


    def _get_ai_analysis(self, news_list: list = None):
        # Copied from v4.py's StockAnalyzer, ensure it's the version you prefer
        # AND ensure it uses self.gemini_model and requests繁體中文
        if not self.gemini_model: return "AI 分析模型目前無法使用。"
        if self.data.empty or len(self.data) < 2: return f"數據不足，無法為 {self.company_name} 生成 AI 分析。"
        
        latest_dp = self.data.iloc[-1]
        latest_close = latest_dp.get('Close', 'N/A')
        pe_d = f"{self.pe_ratio:.2f}" if self.pe_ratio is not None and isinstance(self.pe_ratio, (int, float)) else "N/A"
        roe_d = self.roe_display
        rsi_d = f"{latest_dp.get('RSI'):.2f}" if pd.notna(latest_dp.get('RSI')) else "N/A"
        macd_d = f"{latest_dp.get('MACD'):.2f}" if pd.notna(latest_dp.get('MACD')) else "N/A"
        
        # Prompt for Chinese user
        prompt = f"請為股票 {self.company_name} ({self.ticker}) 提供一份簡潔、專業且易於理解的個股分析報告，目標讀者為正在學習股票投資的中文使用者。請使用繁體中文。\n"
        prompt += f"市場: {'台股' if self.market == 'TW' else '美股'}. 幣別: {self.currency}.\n"
        prompt += f"關鍵數據參考: 目前股價約 {latest_close:.2f}, 本益比(P/E)={pe_d}, 股東權益報酬率(ROE)={roe_d}, RSI={rsi_d}, MACD={macd_d}.\n"
        if news_list:
            prompt += "近期相關新聞標題 (情緒分析結果):\n"
            for item in news_list[:3]: # Limit to 3 news items for prompt brevity
                prompt += f"- {item['title'][:70]}... ({item['sentiment']})\n" # Truncate title
        
        prompt += "\n請依照以下結構進行分析 (使用 Markdown 項目符號):\n"
        prompt += "- **公司簡介:** (1-2句話說明公司業務)\n"
        prompt += "- **近期表現與關鍵指標解讀:** 簡要評論上述提供的數據點所反映的情況。\n"
        prompt += "- **潛在利多因素:** 提及一個可能的正面因素或優勢。\n"
        prompt += "- **潛在風險與注意事項:** 提及一個可能的風險或需要注意的地方。\n"
        prompt += "- **總結與學習點:** 提供一個簡短的總結，或一個與分析此類股票相關的學習要點。\n"
        prompt += "分析應客觀，避免直接的買賣建議，重點在於教育和提供洞見。"

        logger.info(f"Generating AI analysis for {self.ticker} (繁體中文).")
        try:
            response = self.gemini_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=self.temperature))
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"AI content blocked for {self.ticker}. Reason: {response.prompt_feedback.block_reason}")
                return f"AI 分析因內容安全限制被阻擋。(原因: {response.prompt_feedback.block_reason_message or '未知'})"
            
            # Clean up potential markdown code block fences if Gemini adds them
            text_response = response.text.strip()
            if text_response.startswith("```markdown"): text_response = text_response.split("\n",1)[1] if "\n" in text_response else ""
            if text_response.endswith("```"): text_response = text_response.rsplit("\n",1)[0] if "\n" in text_response else ""
            return text_response.strip()
        except Exception as e:
            logger.error(f"Error in _get_ai_analysis for {self.ticker}: {e}", exc_info=True)
            return "生成 AI 分析時發生錯誤。"

    def get_stock_summary(self):
        # Copied from v4.py's StockAnalyzer, ensure it's the version you prefer
        if self.data.empty: return {"error": f"無法載入 {self.ticker} 的數據，無法生成摘要。"}
        required_indicators = ['RSI', 'MACD', 'K']
        if not all(indicator in self.data.columns and self.data[indicator].notna().any() for indicator in required_indicators):
            logger.info(f"[{self.ticker}] Some indicators missing or all NaN, recalculating...")
            self._calculate_indicators() # Ensure indicators are there

        patterns = self._identify_patterns()
        chart_rel_path = self._generate_chart()
        news = self._get_stock_news()
        ai_text_analysis = self._get_ai_analysis(news_list=news)
        latest_dp = self.data.iloc[-1]
        price_change_val = None
        if len(self.data) >= 2:
            prev_close = self.data['Close'].iloc[-2]
            if pd.notna(latest_dp['Close']) and pd.notna(prev_close) and prev_close != 0:
                price_change_val = float(((latest_dp['Close'] - prev_close) / prev_close) * 100)
        
        summary = {
            "ticker": self.ticker, "company_name": self.company_name or 'N/A',
            "market": self.market, # Added market for context in template
            "currency": self.currency or ('TWD' if self.market == 'TW' else 'USD'),
            "current_price": float(latest_dp['Close']) if pd.notna(latest_dp['Close']) else None,
            "price_change_value": price_change_val,
            "pe_ratio": float(self.pe_ratio) if self.pe_ratio is not None and self.pe_ratio != "Infinity" else None, # Handle "Infinity"
            "market_cap": int(self.market_cap) if self.market_cap is not None else None,
            "eps": float(self.eps) if self.eps is not None else None,
            "roe": float(self.roe) if self.roe is not None else None,
            "roe_display": self.roe_display,
            "net_profit_margin": self.net_profit_margin_str, "current_ratio": self.current_ratio_str,
            "rsi": float(latest_dp.get('RSI')) if pd.notna(latest_dp.get('RSI')) else None,
            "macd": float(latest_dp.get('MACD')) if pd.notna(latest_dp.get('MACD')) else None,
            "macd_signal": float(latest_dp.get('MACD_signal')) if pd.notna(latest_dp.get('MACD_signal')) else None,
            "k": float(latest_dp.get('K')) if pd.notna(latest_dp.get('K')) else None,
            "d": float(latest_dp.get('D')) if pd.notna(latest_dp.get('D')) else None,
            "j": float(latest_dp.get('J')) if pd.notna(latest_dp.get('J')) else None,
            "patterns": patterns, "chart_path": chart_rel_path, # Already relative to static
            "news": news, "analysis": ai_text_analysis
        }
        return summary

# --- Flask Routes for V5 ---
@app.route('/', methods=['GET', 'POST'])
def analyze_stock_page_v5(): # Renamed route function
    analysis_data = None
    error_message = None
    loading = False # This will be set true before calling analyzer
    form_ticker = request.form.get('ticker', '').strip()
    form_market = request.form.get('market', 'US').upper()

    if request.method == 'POST':
        if not form_ticker:
            error_message = "請輸入股票代碼。"
        else:
            loading = True # Indicate loading process starts
            # Render template immediately to show loading message if desired,
            # but for this simple version, we'll do synchronous analysis.
            logger.info(f"V5 Page: Analyzing {form_ticker} for market {form_market}")
            
            # Ensure static/charts directory exists
            static_charts_dir = os.path.join(app.static_folder, 'charts')
            if not os.path.exists(static_charts_dir):
                try:
                    os.makedirs(static_charts_dir)
                    logger.info(f"Created directory: {static_charts_dir}")
                except OSError as e:
                    logger.error(f"Could not create static/charts directory: {e}")
                    error_message = "伺服器設定錯誤，無法儲存圖表。"
                    return render_template('stock_analyzer_v5.html', error_message=error_message, loading=False)


            try:
                analyzer = StockAnalyzer(
                    ticker=form_ticker,
                    api_key=GEMINI_API_KEY, # Pass API key
                    market=form_market,
                    temperature=0.6,
                    gemini_model_instance=stock_analyzer_gemini_model # Pass the model instance
                )
                summary = analyzer.get_stock_summary()
                if summary and not summary.get('error'):
                    analysis_data = summary
                    # The chart_path from summary should be like 'charts/filename.html'
                    # url_for('static', filename=analysis_data.chart_path) will generate /static/charts/filename.html
                else:
                    error_message = summary.get('error', f"無法獲取 {form_ticker} 的分析報告。請檢查代碼或市場選擇。")
            except Exception as e:
                logger.error(f"Error in V5 analyze_stock_page for {form_ticker}: {e}", exc_info=True)
                error_message = f"分析過程中發生意外錯誤，請稍後再試。"
            loading = False # Analysis complete

    return render_template('stock_analyzer_v5.html', 
                           analysis_data=analysis_data, 
                           error_message=error_message,
                           loading=loading,
                           request_form=request.form) # Pass form back to repopulate


# --- Main Execution for V5 ---
if __name__ == '__main__':
    logger.info("Starting Stock Analyzer V5 Flask server...")
    # Ensure static/charts directory exists at startup as well
    static_charts_dir_startup = os.path.join(app.static_folder, 'charts')
    if not os.path.exists(static_charts_dir_startup):
        try:
            os.makedirs(static_charts_dir_startup)
            logger.info(f"Created directory at startup: {static_charts_dir_startup}")
        except OSError as e:
             logger.error(f"Could not create static/charts directory at startup: {e}")
             # Decide if this is a fatal error for the app

    app.run(debug=True, host='0.0.0.0', port=5002)