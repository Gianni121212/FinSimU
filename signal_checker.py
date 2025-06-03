# signal_checker.py
import json
import datetime
import pymysql
import numpy as np
import traceback

# 從 model_train 導入需要的函數
try:
    from model_train import load_stock_data, precompute_indicators, run_strategy
except ImportError:
    print("錯誤：無法導入 model_train.py 中的函數。請確保該文件存在且位於正確的路徑。")
    exit()
except SyntaxError as e:
     print(f"錯誤：導入 model_train.py 時發生語法錯誤：{e}。請先修正 model_train.py 中的語法錯誤。")
     exit()

# --- 資料庫設定 ---
try:
    from strategy_optimizer import DB_CONFIG
except ImportError:
    print("警告：無法從 strategy_optimizer.py 導入 DB_CONFIG。將使用本地定義。")
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'root',
        'password': 'your_password', # <--- 請務必修改這裡的密碼
        'database': 'testdb',
        'charset': 'utf8mb4',
        'cursorclass': pymysql.cursors.DictCursor
    }

# --- 重要：定義優化的開始日期 ---
# 結束日期將自動設為最新
OPTIMIZATION_START_DATE = '2021-01-01' # <--- 必須與 strategy_optimizer.py 中的 START_DATE 相同
VIX_TICKER = '^VIX'

# --- 資料庫相關函數 (保持不變) ---
def get_db_connection():
    try: conn = pymysql.connect(**DB_CONFIG); return conn
    except pymysql.Error as e: print(f"資料庫連線錯誤: {e}"); return None

def fetch_strategies():
    conn = get_db_connection();
    if not conn: return []
    strategies = [];
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ticker, gene FROM stock_strategy WHERE gene IS NOT NULL AND gene != '' AND JSON_VALID(gene)")
            rows = cur.fetchall(); strategies = [(row['ticker'], row['gene']) for row in rows]
    except pymysql.Error as e: print(f"讀取 stock_strategy 時出錯: {e}")
    finally:
        if conn: conn.close()
    return strategies

def upsert_signal(ticker, status, price=None, signal_date=None):
    """把狀態寫入 stock_signal"""
    conn = get_db_connection();
    if not conn: print(f"[{ticker}] 無法連線資料庫，跳過更新信號。"); return False
    success = False;
    if signal_date is None: signal_date = datetime.date.today()
    elif isinstance(signal_date, datetime.datetime): signal_date = signal_date.date()

    try:
        with conn.cursor() as cur:
            sql = """
            INSERT INTO stock_signal (ticker, signal_date, status, current_price)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                signal_date = VALUES(signal_date),
                status      = VALUES(status),
                current_price = VALUES(current_price)
            """
            db_price = float(price) if price is not None and np.isfinite(price) else None
            cur.execute(sql, (ticker, signal_date, status, db_price))
        conn.commit(); success = True
    except pymysql.Error as e: print(f"[{ticker}] 更新 stock_signal 時出錯: {e}"); conn.rollback()
    except Exception as e: print(f"[{ticker}] 更新 stock_signal 時發生未知錯誤: {e}"); traceback.print_exc(); conn.rollback()
    finally:
        if conn: conn.close()
    return success

# --- 策略設定 (!!! 關鍵：必須與優化時使用的 strategy_config 完全一致 !!!) ---
# 請再次仔細核對
strategy_config = {
    'rsi_period_options': [6, 12, 21],
    'vix_ma_period_options': [5, 10, 15, 20], # 確保與 strategy_optimizer 一致
    'bb_length_options': [10, 20],
    'bb_std_options': [1.5, 2.0],
    'adx_period': 14,
    'bbi_periods': (3, 6, 12, 24),
    'ma_short_period': 5,
    'ma_long_period': 10,
    'commission_pct': 0.003,
}
print("--- Signal Checker 使用的策略配置 ---")
print(f"RSI Periods: {strategy_config['rsi_period_options']}")
print(f"VIX MA Periods: {strategy_config['vix_ma_period_options']}")
print(f"BB Lengths: {strategy_config['bb_length_options']}")
print(f"BB Stds: {strategy_config['bb_std_options']}")
print(f"回測數據開始日期: {OPTIMIZATION_START_DATE}, 結束日期: 最新") # 打印日期說明
print("-" * 38)


# --- 主檢查邏輯 ---
def run_signal_check():
    """主函數：獲取策略，檢查信號，更新資料庫"""
    print("開始檢查策略信號...")
    strategies_to_check = fetch_strategies()
    print(f"從資料庫獲取了 {len(strategies_to_check)} 個有效策略進行檢查。")

    if not strategies_to_check: print("未能獲取任何策略，程序終止。"); return

    checked_count = 0; error_count = 0

    for ticker, gene_json in strategies_to_check:
        print("-" * 30); print(f"正在檢查: {ticker}")
        try:
            # 解析 JSON 格式的基因列表
            try:
                gene = json.loads(gene_json)
                if not isinstance(gene, list) or len(gene) != 11: raise ValueError("格式不正確或長度不足 11")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[{ticker}] 錯誤：解析基因時出錯 ({e})。Gene: {gene_json}。跳過。"); error_count += 1; continue

            # 1. 載入數據 (***核心修改：結束日期設為 None 以獲取最新數據***)
            print(f"[{ticker}] 載入歷史數據 (從 {OPTIMIZATION_START_DATE} 到 最新)...")
            prices, dates, stock_df, vix_series = load_stock_data(
                ticker,
                vix_ticker=VIX_TICKER,
                start_date=OPTIMIZATION_START_DATE, # 使用定義的開始日期
                end_date=None                      # <--- 結束日期設為 None
            )
            if prices is None or not prices:
                print(f"[{ticker}] 載入最新數據失敗或數據為空，跳過。"); error_count += 1; continue
            # 獲取實際的結束日期用於記錄
            actual_end_date_str = dates[-1].strftime('%Y-%m-%d') if dates else "未知"
            print(f"[{ticker}] 數據載入完成，共 {len(prices)} 筆 (到 {actual_end_date_str})。")

            # 2. 預計算指標
            print(f"[{ticker}] 預計算指標...")
            precalc, ok = precompute_indicators(stock_df, vix_series, strategy_config)
            if not ok: print(f"[{ticker}] 預計算指標失敗，跳過。"); error_count += 1; continue
            print(f"[{ticker}] 指標預計算完成。")

            # 3. 取出對應的指標序列 (增加索引檢查)
            try:
                rsi_options_len = len(strategy_config['rsi_period_options']); vix_options_len = len(strategy_config['vix_ma_period_options']); bblen_options_len = len(strategy_config['bb_length_options']); bbstd_options_len = len(strategy_config['bb_std_options'])
                if not (0 <= gene[4] < rsi_options_len): raise IndexError(f"RSI Period index {gene[4]} 無效")
                if not (0 <= gene[5] < vix_options_len): raise IndexError(f"VIX MA Period index {gene[5]} 無效")
                if not (0 <= gene[6] < bblen_options_len): raise IndexError(f"BB Length index {gene[6]} 無效")
                if not (0 <= gene[7] < bbstd_options_len): raise IndexError(f"BB Std index {gene[7]} 無效")

                rsi_period = strategy_config['rsi_period_options'][gene[4]]; vix_ma_period = strategy_config['vix_ma_period_options'][gene[5]]; bb_len = strategy_config['bb_length_options'][gene[6]]; bb_std = strategy_config['bb_std_options'][gene[7]]
                rsi_list    = precalc['rsi'][rsi_period]; vix_ma_list = precalc['vix_ma'][vix_ma_period]; bbl_list    = precalc['bbl'][(bb_len, bb_std)]; bbm_list    = precalc['bbm'][(bb_len, bb_std)]
                fixed = precalc['fixed']; bbi_list = fixed['bbi_list']; adx_list = fixed['adx_list']; ma_short_list = fixed['ma_short_list']; ma_long_list  = fixed['ma_long_list']
            except IndexError as e: print(f"[{ticker}] 錯誤: 基因索引無效 ({e})。存儲的基因與當前 strategy_config 不匹配。跳過。"); error_count += 1; continue
            except KeyError as e: print(f"[{ticker}] 錯誤: 無法在預計算結果中找到指標鍵 ({e})。跳過。"); error_count += 1; continue

            # 4. 執行策略回測 (現在基於到最新的數據)
            print(f"[{ticker}] 執行策略回測 (從 {OPTIMIZATION_START_DATE} 到 {actual_end_date_str})...")
            portfolio_values, buy_signals, sell_signals = run_strategy(
                gene[0], gene[1], gene[8], gene[2], gene[3], gene[9], gene[10],
                strategy_config['commission_pct'], prices, dates,
                rsi_list, bbl_list, bbm_list, bbi_list, adx_list, vix_ma_list, ma_short_list, ma_long_list
            )
            print(f"[{ticker}] 回測完成。找到 {len(buy_signals)} 個買入信號，{len(sell_signals)} 個賣出信號 (在回測區間內)。")

            # 5. 判斷最新的狀態 (基於到最近一日的回測)
            last_buy_date = None; last_sell_date = None; status = 'hold'
            if buy_signals: last_buy_date = buy_signals[-1][0]
            if sell_signals: last_sell_date = sell_signals[-1][0]

            if last_buy_date is not None and (last_sell_date is None or last_buy_date > last_sell_date):
                status = 'buy'
            elif last_sell_date is not None and (last_buy_date is None or last_sell_date > last_buy_date):
                status = 'sell'

            print(f"[{ticker}] 回測區間內最後買入日期: {last_buy_date}, 最後賣出日期: {last_sell_date}, 判斷狀態: {status}")

            # 6. 更新資料庫
            last_price = prices[-1] if prices else None # 獲取最新的收盤價
            signal_date_to_store = datetime.date.today() # 將信號標記為今天的

            price_str = f"{last_price:.2f}" if last_price is not None and np.isfinite(last_price) else 'N/A'
            print(f"[{ticker}] 最新價格 ({actual_end_date_str}): {price_str}，記錄狀態為 {status} (日期: {signal_date_to_store})...")

            update_ok = upsert_signal(ticker, status, last_price, signal_date=signal_date_to_store)
            if update_ok: print(f"[{ticker}] 資料庫更新成功。")
            else: print(f"[{ticker}] 資料庫更新失敗。")
            checked_count += 1

        except KeyboardInterrupt: print("\n收到用戶中斷請求，停止檢查..."); break
        except Exception as e: print(f"處理 [{ticker}] 時發生未預期錯誤: {type(e).__name__}: {e}"); traceback.print_exc(); error_count += 1

    print("-" * 30); print("信號檢查完成。")
    print(f"成功檢查股票數量: {checked_count}"); print(f"發生錯誤或跳過的股票數量: {error_count}"); print("-" * 30)

# --- 執行入口 ---
if __name__ == "__main__":
    print("檢查並嘗試創建 stock_signal 資料表...")
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS stock_signal (
                    ticker VARCHAR(20) PRIMARY KEY,
                    signal_date DATE,
                    status ENUM('buy','sell','hold') NOT NULL,
                    current_price FLOAT,
                    INDEX (signal_date)
                );"""
                cur.execute(create_table_sql)
            conn.commit(); print("stock_signal 資料表已存在或創建成功。")
        except pymysql.Error as e: print(f"創建 stock_signal 資料表時出錯: {e}"); conn.rollback()
        finally: conn.close()
    else: print("無法連線資料庫，未能創建 stock_signal 資料表。"); exit()

    run_signal_check()