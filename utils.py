
import os
import pymysql
import yfinance as yf
import datetime
import logging
from datetime import timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger("FinSimU.utils")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Utils] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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

STOCK_CACHE_DURATION = timedelta(minutes=5)

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

def calc_trade_extremes(buy_signals, sell_signals, commission_rate=0.0035):
    if not buy_signals or not sell_signals:
        return 0.0, 0.0

    max_profit_pct = 0.0
    max_loss_pct = 0.0
    num_trades = min(len(buy_signals), len(sell_signals))

    for i in range(num_trades):
        try:
            buy_price = buy_signals[i]['price']
            sell_price = sell_signals[i]['price']
            if buy_price > 1e-9 and np.isfinite(buy_price) and np.isfinite(sell_price):
                cost_basis = buy_price * (1 + commission_rate)
                net_proceeds = sell_price * (1 - commission_rate)
                profit_loss_pct = (net_proceeds - cost_basis) / cost_basis
                if profit_loss_pct > max_profit_pct:
                    max_profit_pct = profit_loss_pct
                elif profit_loss_pct < max_loss_pct:
                    max_loss_pct = profit_loss_pct
        except (IndexError, TypeError, KeyError):
            continue
    return max_profit_pct * 100, max_loss_pct * 100

def calculate_performance_metrics(portfolio_values, dates, buy_signals, sell_signals, prices, risk_free_rate=0.025, commission_rate=0.0035):
    metrics = {
        'total_return': 0.0, 'annualized_return': 0.0, 'max_drawdown': 1.0, 
        'sharpe_ratio': 0.0, 'sortino_ratio': 0.0,
        'profit_factor': 0.0, 'win_rate_pct': 0.0, 'pl_ratio': 0.0,
        'trade_count': 0, 'average_trade_return': 0.0,
        'volatility': 0.0, 'max_trade_drop_pct': 0.0, 'max_trade_gain_pct': 0.0
    }

    if portfolio_values is None or len(portfolio_values) < 2:
        return metrics


    pv_series = pd.Series(portfolio_values, index=pd.to_datetime(dates))
    daily_returns = pv_series.pct_change().dropna()
    
    # --- 1. 報酬率指標 ---
    metrics['total_return'] = pv_series.iloc[-1] - 1.0
    
    time_delta_days = (pv_series.index[-1] - pv_series.index[0]).days
    years = time_delta_days / 365.25
    if years > 0 and pv_series.iloc[-1] > 0:
        metrics['annualized_return'] = (pv_series.iloc[-1] ** (1 / years)) - 1
    
    # --- 2. 風險指標 ---
    metrics['volatility'] = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else 0.0

    running_max = pv_series.cummax()
    drawdowns = (running_max - pv_series) / running_max
    metrics['max_drawdown'] = drawdowns.max() if not drawdowns.empty else 1.0

    # --- 3. 交易分析指標 ---
    completed_trades = min(len(buy_signals), len(sell_signals))
    metrics['trade_count'] = completed_trades
    
    if completed_trades > 0:
        wins, losses = 0, 0
        total_profit, total_loss = 0.0, 0.0
        total_net_trade_return_pct = 0.0
        
        for i in range(completed_trades):
            try:
                buy_price = buy_signals[i]['price']
                sell_price = sell_signals[i]['price']
                if buy_price > 0 and np.isfinite(buy_price) and np.isfinite(sell_price):
                    cost_basis = buy_price * (1 + commission_rate)
                    net_proceeds = sell_price * (1 - commission_rate)
                    net_profit = net_proceeds - cost_basis
                    
                    if net_profit > 0:
                        wins += 1
                        total_profit += net_profit
                    else:
                        losses += 1
                        total_loss += abs(net_profit)
                    total_net_trade_return_pct += (net_profit / cost_basis)
            except (IndexError, TypeError, KeyError, ZeroDivisionError):
                continue

        metrics['win_rate_pct'] = (wins / completed_trades) * 100 if completed_trades > 0 else 0.0
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else total_profit
        
        avg_win = total_profit / wins if wins > 0 else 0
        avg_loss = total_loss / losses if losses > 0 else 0
        metrics['pl_ratio'] = avg_win / avg_loss if avg_loss > 0 else avg_win

        metrics['average_trade_return'] = total_net_trade_return_pct / completed_trades if completed_trades > 0 else 0.0
    
    # --- 4. 風險調整後報酬指標 ---
    if metrics['volatility'] > 0 and not daily_returns.empty:
        excess_returns = daily_returns - (risk_free_rate / 252)
        metrics['sharpe_ratio'] = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252) if not negative_returns.empty else 0.0
    if downside_std > 0:
        metrics['sortino_ratio'] = (metrics['annualized_return'] - risk_free_rate) / downside_std

    # --- 5. 單筆交易極值 ---
    max_gain, max_drop = calc_trade_extremes(buy_signals, sell_signals, commission_rate)
    metrics['max_trade_drop_pct'] = max_drop
    metrics['max_trade_gain_pct'] = max_gain
    
    return metrics
