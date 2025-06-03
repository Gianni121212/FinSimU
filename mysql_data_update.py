import mysql.connector 
import pandas as pd
import numpy as np
import yfinance as yf

# 連接 MySQL 資料庫
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0912559910",
    database="testdb"
)
cursor = conn.cursor()

# 欄位檢查列表，包含技術指標以及新增的財務與價格資料欄位
columns_to_check = [
    "ma5", "ma20", "ma50", "ma120", "ma200",
    "bb_upper", "bb_middle", "bb_lower",
    "rsi", "wmsr", "psy", "bias6",
    "macd", "macd_signal", "macd_hist",
    "k", "d", "j",
    "pe_ratio", "market_cap",
    "open_price", "close_price", "high_price", "low_price", "volume"
]

# 檢查並新增欄位
for column in columns_to_check:
    cursor.execute(f"SHOW COLUMNS FROM stocks LIKE '{column}'")
    result = cursor.fetchone()
    if not result:
        cursor.execute(f"ALTER TABLE stocks ADD COLUMN {column} FLOAT DEFAULT NULL;")
        print(f"✅ {column.upper()} 欄位已新增")
    else:
        print(f"⚠️ {column.upper()} 欄位已存在，跳過新增")

# 取得所有股票代號 (Symbol) 從 stocks 表
cursor.execute("SELECT symbol FROM stocks;")
symbols = [row[0] for row in cursor.fetchall()]

# ======================
#  技術指標計算函數區
# ======================

def compute_moving_averages(close_prices, periods=[5, 20, 50, 120, 200]):
    ma_values = {}
    for period in periods:
        if len(close_prices) >= period:
            rolling_mean = close_prices.rolling(window=period).mean()
            ma_values[f"ma{period}"] = float(rolling_mean.iloc[-1]) if not rolling_mean.empty else None
        else:
            ma_values[f"ma{period}"] = None
    return ma_values

def compute_bollinger_bands(close_prices, window=20, num_std=2):
    middle_band = close_prices.rolling(window=window).mean()
    std_dev = close_prices.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band

def compute_rsi(prices, window=14):
    if len(prices) < window:
        return None
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    # 當平均損失為 0 時，RSI 預設為 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.dropna().empty else None


def compute_williams_r(high, low, close, period=14):
    if len(close) < period:
        return None
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    if williams_r.dropna().empty:
        return None
    return float(williams_r.iloc[-1])

def compute_psy(close_prices, period=12):
    if len(close_prices) < period:
        return None
    up_days = (close_prices.diff() > 0).astype(int)
    psy = up_days.rolling(period).sum() / period * 100
    if psy.dropna().empty:
        return None
    return float(psy.iloc[-1])

def compute_bias(close_prices, period=6):
    if len(close_prices) < period:
        return None
    ma = close_prices.rolling(period).mean()
    bias = (close_prices - ma) / ma * 100
    if bias.dropna().empty:
        return None
    return float(bias.iloc[-1])

def compute_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow:
        return None, None, None
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return (
        float(macd_line.iloc[-1]),
        float(signal_line.iloc[-1]),
        float(macd_hist.iloc[-1]),
    )

def compute_kdj(high, low, close, n=9, k_smooth=3, d_smooth=3):
    if len(close) < n:
        return None, None, None
    lowest_low = low.rolling(n).min()
    highest_high = high.rolling(n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    K = rsv.ewm(alpha=1/k_smooth).mean()
    D = K.ewm(alpha=1/d_smooth).mean()
    J = 3 * K - 2 * D
    return (
        float(K.iloc[-1]) if not K.dropna().empty else None,
        float(D.iloc[-1]) if not D.dropna().empty else None,
        float(J.iloc[-1]) if not J.dropna().empty else None
    )

# ======================
#    主程式執行區
# ======================
for symbol in symbols:
    try:
        yf_symbol = symbol.replace('.', '-')
        ticker = yf.Ticker(yf_symbol)
        stock_data = ticker.history(period="200d", interval="1d")
        
        if stock_data.empty:
            print(f"⚠️ {symbol} 沒有數據，跳過更新")
            continue

        # 取得最新一天的價格資訊
        latest_data = stock_data.iloc[-1]
        open_price = float(latest_data["Open"])
        close_price = float(latest_data["Close"])
        high_price = float(latest_data["High"])
        low_price = float(latest_data["Low"])
        volume = float(latest_data["Volume"])

        # 取得基本的財務數據
        info = ticker.info
        pe_ratio = info.get('trailingPE', None)
        market_cap = info.get('marketCap', None)

        # 計算技術指標
        ma_values = compute_moving_averages(stock_data["Close"])
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(stock_data["Close"])
        rsi_value = compute_rsi(stock_data["Close"])
        wmsr_value = compute_williams_r(stock_data["High"], stock_data["Low"], stock_data["Close"])
        psy_value = compute_psy(stock_data["Close"])
        bias6_value = compute_bias(stock_data["Close"])
        macd_val, macd_signal_val, macd_hist_val = compute_macd(stock_data["Close"])
        k_val, d_val, j_val = compute_kdj(stock_data["High"], stock_data["Low"], stock_data["Close"])

        # 更新 MySQL 資料庫 (包含所有欄位)
        update_query = """
            UPDATE stocks 
            SET ma5=%s, ma20=%s, ma50=%s, ma120=%s, ma200=%s,
                bb_upper=%s, bb_middle=%s, bb_lower=%s,
                rsi=%s, wmsr=%s, psy=%s, bias6=%s,
                macd=%s, macd_signal=%s, macd_hist=%s,
                k=%s, d=%s, j=%s,
                pe_ratio=%s, market_cap=%s,
                open_price=%s, close_price=%s, high_price=%s, low_price=%s, volume=%s
            WHERE symbol=%s
        """
        cursor.execute(update_query, (
            ma_values.get("ma5"), 
            ma_values.get("ma20"), 
            ma_values.get("ma50"),
            ma_values.get("ma120"), 
            ma_values.get("ma200"),
            float(bb_upper.dropna().iloc[-1]) if not bb_upper.dropna().empty else None,
            float(bb_middle.dropna().iloc[-1]) if not bb_middle.dropna().empty else None,
            float(bb_lower.dropna().iloc[-1]) if not bb_lower.dropna().empty else None,
            rsi_value,
            wmsr_value, 
            psy_value, 
            bias6_value,
            macd_val, 
            macd_signal_val, 
            macd_hist_val,
            k_val, 
            d_val, 
            j_val,
            pe_ratio,
            market_cap,
            open_price,
            close_price,
            high_price,
            low_price,
            volume,
            symbol
        ))
        conn.commit()
        print(f"✅ {symbol} 技術指標與財務、價格資料更新成功")
    except Exception as e:
        print(f"❌ {symbol} 更新失敗: {e}")

# 關閉資料庫連線
cursor.close()
conn.close()
print("所有股票更新完成！")
