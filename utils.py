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
# ---                  3. 統一績效與交易分析函式 (核心重構 v2.2)               ---
# ==============================================================================

# === MODIFICATION START: The `calc_trade_extremes` function is now commission-aware ===
def calc_trade_extremes(buy_signals, sell_signals, commission_rate=0.005):
    """
    (v2.2) 計算所有已完成交易中的最大單筆「已實現」獲利和虧損百分比（已納入手續費）。
    """
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
                # 模擬買入成本（包含手續費）
                cost_basis = buy_price * (1 + commission_rate)
                # 模擬賣出淨收入（扣除手續費）
                net_proceeds = sell_price * (1 - commission_rate)
                
                # 計算淨盈虧百分比
                profit_loss_pct = (net_proceeds - cost_basis) / cost_basis
                
                if profit_loss_pct > max_profit_pct:
                    max_profit_pct = profit_loss_pct
                elif profit_loss_pct < max_loss_pct:
                    max_loss_pct = profit_loss_pct
        except (IndexError, TypeError, KeyError):
            continue
    
    return max_profit_pct * 100, max_loss_pct * 100
# === MODIFICATION END ===

# === MODIFICATION START: The main metrics function is now fully commission-aware ===
def calculate_performance_metrics(portfolio_values, dates, buy_signals, sell_signals, prices, risk_free_rate=0.04, commission_rate=0.005):
    """
    (v2.2) 計算完整的策略績效指標（所有指標均納入手續費）。
    """
    metrics = {
        'total_return': 0.0, 'max_drawdown': 1.0, 'sharpe_ratio': 0.0,
        'profit_factor': 0.01, 'win_rate_pct': 0.0, 'trade_count': 0,
        'average_trade_return': 0.0, 'std_dev': 1.0,
        'max_trade_drop_pct': 0.0, 'max_trade_gain_pct': 0.0,
    }

    if portfolio_values is None or len(portfolio_values) < 2:
        return metrics

    # 1. 總報酬率 & 標準差 (這些本來就是準確的，因為基於已扣手續費的 portfolio_values)
    final_value = portfolio_values[-1]
    metrics['total_return'] = final_value - 1.0
    metrics['std_dev'] = np.std(portfolio_values) if len(portfolio_values) > 1 else 0.001

    # 2. 最大回撤 (準確)
    pv_np = np.array(portfolio_values)
    running_max = np.maximum.accumulate(pv_np)
    safe_running_max = np.where(running_max == 0, 1, running_max)
    drawdowns = (running_max - pv_np) / safe_running_max
    metrics['max_drawdown'] = np.max(drawdowns) if len(drawdowns) > 0 else 1.0

    # 3. 夏普比率 (準確)
    if metrics['std_dev'] > 0:
        daily_returns = pd.Series(portfolio_values).pct_change().dropna()
        if not daily_returns.empty:
            excess_returns = daily_returns - (risk_free_rate / 252)
            if np.std(excess_returns) > 0:
                metrics['sharpe_ratio'] = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
    
    # 4. 交易相關指標 (需要在這裡納入手續費計算)
    completed_trades = min(len(buy_signals), len(sell_signals))
    metrics['trade_count'] = completed_trades
    
    if completed_trades > 0:
        wins, total_profit, total_loss = 0, 0.0, 0.0
        total_net_trade_return_pct = 0.0
        
        for i in range(completed_trades):
            try:
                buy_price = buy_signals[i]['price']
                sell_price = sell_signals[i]['price']

                if buy_price > 0 and np.isfinite(buy_price) and np.isfinite(sell_price):
                    # 模擬手續費影響
                    cost_basis = buy_price * (1 + commission_rate)
                    net_proceeds = sell_price * (1 - commission_rate)
                    
                    net_profit = net_proceeds - cost_basis
                    
                    if net_profit > 0:
                        wins += 1
                        total_profit += net_profit
                    else:
                        total_loss += abs(net_profit)
                    
                    total_net_trade_return_pct += (net_profit / cost_basis)

            except (IndexError, TypeError, KeyError, ZeroDivisionError):
                continue

        metrics['win_rate_pct'] = (wins / completed_trades) * 100 if completed_trades > 0 else 0.0
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0.01)
        metrics['average_trade_return'] = total_net_trade_return_pct / completed_trades if completed_trades > 0 else 0.0
    
    # 5. 單次交易最大已實現漲跌幅 (調用新的、已納入手續費的函式)
    max_gain, max_drop = calc_trade_extremes(buy_signals, sell_signals, commission_rate)
    metrics['max_trade_drop_pct'] = max_drop
    metrics['max_trade_gain_pct'] = max_gain
    
    return metrics
# === MODIFICATION END ===
