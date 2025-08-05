# utils.py
# ==============================================================================
# ---                       FinSimU 專案 - 共用工具模組 v2.0                   ---
# ==============================================================================
import os
import pymysql
import yfinance as yf
import datetime
import logging
from datetime import timedelta
import numpy as np
import pandas as pd

# --- 全域設定 ---
logger = logging.getLogger("FinSimU.utils")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Utils] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- 資料庫設定 ---
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME", "finsimu_db"),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'connect_timeout': 10
}
if not DB_CONFIG['password']:
    logger.critical("致命錯誤: 未在環境變數中設定 DB_PASSWORD。")

# --- 股票數據快取設定 ---
STOCK_CACHE_DURATION = timedelta(minutes=5)

# ==============================================================================
# ---                         1. 資料庫輔助函式                            ---
# ==============================================================================
def execute_db_query(query, args=None, fetch_one=False, fetch_all=False, executemany=False):
    conn = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        with conn.cursor() as cursor:
            if executemany:
                rowcount = cursor.executemany(query, args)
            else:
                rowcount = cursor.execute(query, args)
            if fetch_one:
                return cursor.fetchone()
            if fetch_all:
                return cursor.fetchall()
            conn.commit()
            return rowcount
    except pymysql.Error as e:
        logger.error(f"資料庫查詢錯誤: {e}\n查詢: {query}", exc_info=True)
        if conn: conn.rollback()
        return None
    finally:
        if conn and conn.open:
            conn.close()

# ==============================================================================
# ---                         2. 股票數據服務函式                          ---
# ==============================================================================
def get_stock_data_from_yf(ticker_symbol):
    try:
        logger.info(f"YFINANCE FETCH: 嘗試獲取 {ticker_symbol} 的最新數據")
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice', info.get('bid')))
        previous_close = info.get('previousClose')
        if current_price is None or previous_close is None:
            hist_2d = stock.history(period="2d", auto_adjust=False)
            if not hist_2d.empty:
                if current_price is None and len(hist_2d) >= 1:
                    current_price = hist_2d['Close'].iloc[-1]
                if previous_close is None and len(hist_2d) >= 2:
                    previous_close = hist_2d['Close'].iloc[-2]
                elif previous_close is None and len(hist_2d) == 1 and current_price is not None:
                     previous_close = current_price
        if current_price is None or previous_close is None:
             logger.warning(f"無法從 yfinance 可靠地確定 {ticker_symbol} 的當前價格或前收盤價。")
             return None

        current_price_f = float(current_price)
        previous_close_f = float(previous_close)
        daily_change = current_price_f - previous_close_f
        daily_change_percent = (daily_change / previous_close_f) * 100 if previous_close_f != 0 else 0.0

        data_to_cache = {
            'ticker': ticker_symbol.upper(),
            'name': info.get('longName', info.get('shortName', ticker_symbol)),
            'current_price': current_price_f,
            'previous_close': previous_close_f,
            'daily_change': daily_change,
            'daily_change_percent': daily_change_percent,
            'last_fetched': datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        }

        sql = """
            INSERT INTO stock_data_cache
                (ticker, name, current_price, previous_close, daily_change, daily_change_percent, last_fetched)
            VALUES
                (%(ticker)s, %(name)s, %(current_price)s, %(previous_close)s, %(daily_change)s, %(daily_change_percent)s, %(last_fetched)s)
            ON DUPLICATE KEY UPDATE
                name = VALUES(name), current_price = VALUES(current_price), previous_close = VALUES(previous_close),
                daily_change = VALUES(daily_change), daily_change_percent = VALUES(daily_change_percent),
                last_fetched = VALUES(last_fetched)
        """
        execute_db_query(sql, data_to_cache)
        logger.info(f"YFINANCE FETCH: 成功獲取並快取 {ticker_symbol} 的數據")
        return data_to_cache
    except Exception as e:
        logger.warning(f"從 yfinance 獲取 {ticker_symbol} 數據時發生警告: {e}", exc_info=False)
        return None

def get_stock_info(ticker_symbol):
    ticker_symbol = ticker_symbol.upper()
    cached_data = execute_db_query("SELECT * FROM stock_data_cache WHERE ticker = %s", (ticker_symbol,), fetch_one=True)
    if cached_data and cached_data['last_fetched']:
        last_fetched_dt = cached_data['last_fetched']
        if last_fetched_dt.tzinfo is not None:
            last_fetched_dt = last_fetched_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        if (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - last_fetched_dt) < STOCK_CACHE_DURATION:
            if cached_data.get('current_price') is not None:
                logger.info(f"{ticker_symbol} 的快取命中。")
                for key in ['current_price', 'previous_close', 'daily_change', 'daily_change_percent']:
                    if cached_data.get(key) is not None:
                        cached_data[key] = float(cached_data[key])
                return cached_data
    logger.info(f"{ticker_symbol} 的快取未命中或已過期，從 yfinance 獲取新數據。")
    return get_stock_data_from_yf(ticker_symbol)

# ==============================================================================
# ---                  3. 統一績效與交易分析函式 (核心重構)                 ---
# ==============================================================================

def calc_trade_extremes(prices, dates, buy_signals, sell_signals):
    """
    計算單次交易內部的最大跌幅和最大漲幅（以股價為基準）
    
    Args:
        prices (list or np.ndarray): 價格序列
        dates (list): 日期序列 (datetime objects)
        buy_signals (list of tuples/dicts): 買入信號 [(date, price), ...] 或 [{'date': dt, 'price': p}, ...]
        sell_signals (list of tuples/dicts): 賣出信號
        
    Returns:
        tuple: (最大跌幅百分比, 最大漲幅百分比)
    """
    if not buy_signals or not sell_signals or not isinstance(dates, (list, pd.core.indexes.datetimes.DatetimeIndex)) or len(dates) == 0:
        return 0.0, 0.0

    date_to_idx = {pd.to_datetime(d).date(): i for i, d in enumerate(dates)}
    worst_drop = 0.0
    best_gain = 0.0
    
    completed_trades = min(len(buy_signals), len(sell_signals))
    
    for i in range(completed_trades):
        try:
            buy_info = buy_signals[i]
            sell_info = sell_signals[i]

            buy_date = pd.to_datetime(buy_info[0] if isinstance(buy_info, tuple) else buy_info['date']).date()
            buy_price = buy_info[1] if isinstance(buy_info, tuple) else buy_info['price']
            sell_date = pd.to_datetime(sell_info[0] if isinstance(sell_info, tuple) else sell_info['date']).date()
            
            if buy_date not in date_to_idx or sell_date not in date_to_idx or buy_price <= 0:
                continue
                
            buy_idx = date_to_idx[buy_date]
            sell_idx = date_to_idx[sell_date]

            if buy_idx >= sell_idx: continue
            
            trade_period_prices = prices[buy_idx : sell_idx + 1]
            if len(trade_period_prices) == 0: continue
            
            min_price_in_trade = min(trade_period_prices)
            max_price_in_trade = max(trade_period_prices)
            
            drop_pct = (min_price_in_trade - buy_price) / buy_price
            gain_pct = (max_price_in_trade - buy_price) / buy_price
            
            worst_drop = min(worst_drop, drop_pct)
            best_gain = max(best_gain, gain_pct)
            
        except (IndexError, TypeError, KeyError, ZeroDivisionError) as e:
            logger.warning(f"計算交易極值時跳過一筆交易，原因: {e}")
            continue
    
    return worst_drop * 100, best_gain * 100

def calculate_performance_metrics(portfolio_values, dates, buy_signals, sell_signals, prices, risk_free_rate=0.0):
    """
    計算完整的策略績效指標。
    
    Args:
        portfolio_values (list or np.ndarray): 投資組合價值序列
        dates (list): 日期序列
        buy_signals (list): 買入信號
        sell_signals (list): 賣出信號
        prices (list): 股價序列
        risk_free_rate (float): 無風險利率
        
    Returns:
        dict: 包含所有績效指標的字典
    """
    metrics = {
        'total_return': 0.0, 'max_drawdown': 1.0, 'sharpe_ratio': 0.0,
        'profit_factor': 0.01, 'win_rate_pct': 0.0, 'trade_count': 0,
        'average_trade_return': 0.0, 'std_dev': 1.0,
        'max_trade_drop_pct': 0.0, 'max_trade_gain_pct': 0.0,
    }

    if portfolio_values is None or len(portfolio_values) < 2:
        return metrics

    # 1. 總報酬率 & 標準差
    final_value = portfolio_values[-1]
    metrics['total_return'] = final_value - 1.0
    metrics['std_dev'] = np.std(portfolio_values) if len(portfolio_values) > 1 else 0.001

    # 2. 最大回撤
    pv_np = np.array(portfolio_values)
    running_max = np.maximum.accumulate(pv_np)
    safe_running_max = np.where(running_max == 0, 1, running_max)
    drawdowns = (running_max - pv_np) / safe_running_max
    metrics['max_drawdown'] = np.max(drawdowns) if len(drawdowns) > 0 else 1.0

    # 3. 夏普比率
    if metrics['std_dev'] > 0:
        daily_returns = pd.Series(portfolio_values).pct_change().dropna()
        excess_returns = daily_returns - (risk_free_rate / 252)
        metrics['sharpe_ratio'] = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
    else:
        metrics['sharpe_ratio'] = 0.0

    # 4. 交易相關指標
    completed_trades = min(len(buy_signals), len(sell_signals))
    metrics['trade_count'] = completed_trades
    
    if completed_trades > 0:
        wins, total_profit, total_loss = 0, 0.0, 0.0
        total_trade_return_pct = 0.0
        
        for i in range(completed_trades):
            try:
                buy_info = buy_signals[i]
                sell_info = sell_signals[i]
                buy_price = buy_info[1] if isinstance(buy_info, tuple) else buy_info['price']
                sell_price = sell_info[1] if isinstance(sell_info, tuple) else sell_info['price']

                if buy_price > 0 and np.isfinite(buy_price) and np.isfinite(sell_price):
                    profit = sell_price - buy_price
                    if profit > 0:
                        wins += 1
                        total_profit += profit
                    else:
                        total_loss += abs(profit)
                    total_trade_return_pct += (profit / buy_price)
            except (IndexError, TypeError, KeyError, ZeroDivisionError):
                continue

        metrics['win_rate_pct'] = (wins / completed_trades) * 100
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 1.0)
        metrics['average_trade_return'] = total_trade_return_pct / completed_trades if completed_trades > 0 else 0.0
    
    # 5. 單次交易最大漲跌幅
    max_drop, max_gain = calc_trade_extremes(prices, dates, buy_signals, sell_signals)
    metrics['max_trade_drop_pct'] = max_drop
    metrics['max_trade_gain_pct'] = max_gain
    
    return metrics