# -*- coding: utf-8 -*-
import pandas as pd
import pymysql
import datetime
import json
import time
import traceback
import numpy as np # 引入 numpy 用於檢查 NaN

# 從 model_train 導入需要的函數
# 確保 model_train.py 在 Python 的搜索路徑中 (通常放在同一個目錄下即可)
try:
    # 注意：確保 model_train.py 中的 genetic_algorithm_with_elitism 函數語法已修正
    from model_train import load_stock_data, precompute_indicators, genetic_algorithm_with_elitism
except ImportError:
    print("錯誤：無法導入 model_train.py 中的函數。請確保該文件存在且位於正確的路徑，並且沒有語法錯誤。")
    exit()
except SyntaxError as e:
     print(f"錯誤：導入 model_train.py 時發生語法錯誤：{e}。請先修正 model_train.py 中的語法錯誤。")
     exit()


# --- 資料庫設定 ---
# !!! 重要：請根據你的 MySQL 環境修改以下設定 !!!
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',          # 你的 MySQL 用戶名
    'password': '0912559910', # <--- 請務必修改這裡的密碼 !!!
    'database': 'testdb',      # 你建立的資料庫名稱
    'charset': 'utf8mb4',      # 建議使用 utf8mb4 以支持更廣泛字符
    'cursorclass': pymysql.cursors.DictCursor # 使用字典游標方便操作
}

# --- 資料庫操作函數 ---
def get_db_connection():
    """建立資料庫連線"""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except pymysql.Error as e:
        print(f"資料庫連線錯誤: {e}")
        return None

def save_to_mysql(ticker, gene, fitness):
    """將優化結果儲存到 MySQL 資料庫"""
    # 檢查 fitness 是否有效
    if not np.isfinite(fitness):
        print(f"  ⚠️ 警告: 無效的 fitness 值 ({fitness})，跳過儲存 {ticker}。")
        return False
    # 檢查 gene 是否有效
    if gene is None or not isinstance(gene, list) or len(gene) != 11:
         print(f"  ⚠️ 警告: 無效的 gene ({gene})，跳過儲存 {ticker}。")
         return False

    conn = get_db_connection()
    if not conn:
        print(f"[{ticker}] 無法連線資料庫，跳過儲存。")
        return False

    success = False
    try:
        with conn.cursor() as cursor:
            # 使用 REPLACE INTO，如果 ticker 已存在則更新，否則插入新紀錄
            sql = """
            REPLACE INTO stock_strategy (ticker, gene, fitness, last_backtest_date)
            VALUES (%s, %s, %s, %s)
            """
            # 將 Python list (gene) 轉換為 JSON 字符串儲存
            gene_json = json.dumps(gene) # 確保 gene 是可序列化的列表
            today_date = datetime.date.today()
            cursor.execute(sql, (ticker, gene_json, float(fitness), today_date)) # 確保 fitness 是 float
        conn.commit()
        print(f"  💾 資料庫儲存成功: {ticker}")
        success = True
    except pymysql.Error as db_err:
        print(f"  ❌ 資料庫錯誤 ({ticker}): {db_err}")
        if conn:
            conn.rollback() # 發生錯誤時回滾事務
    except Exception as e:
        print(f"  ❌ 儲存到資料庫時發生未知錯誤 ({ticker}): {e}")
        traceback.print_exc()
        if conn:
            conn.rollback()
    finally:
        if conn and conn.open: # 檢查連線是否存在且開啟
            conn.close()
    return success

# --- 批次優化主函數 ---
def run_batch_optimization():
    """執行批次優化，每個股票運行多次 GA"""
    # --- 全局設定 ---
    TICKER_CSV_FILE = "sp-100-index-03-14-2025.csv" # 包含股票代碼的 CSV 文件
    START_DATE = '2021-01-01' # 回測開始日期
    END_DATE = '2025-05-05'   # 回測結束日期 (可以設為 None 使用最新數據，但建議固定以保證可重複性)
    VIX_TICKER = '^VIX'        # VIX 指數代碼
    NUM_GA_RUNS_PER_STOCK = 3  # 設定每個股票運行 GA 的次數 (例如 3, 5, 10)

    # --- 策略和 GA 參數設定 ---
    # !!! 關鍵：這裡的 _options 列表必須與 signal_checker.py 中的完全一致 !!!
    strategy_config = {
        'rsi_period_options': [6, 12, 21],
        'vix_ma_period_options': [5, 10, 15, 20],      # 確保與 signal_checker 一致
        'bb_length_options': [10, 20],
        'bb_std_options': [1.5, 2.0],
        'adx_period': 14,
        'bbi_periods': (3, 6, 12, 24),
        'ma_short_period': 5,
        'ma_long_period': 10,
        'commission_pct': 0.003, # 佣金率
    }
    ga_config = {
        'generations': 50,         # 根據需要調整代數
        'population_size': 80,     # 根據需要調整種群大小
        'crossover_rate': 0.7,
        'mutation_rate': 0.25,
        'elitism_size': 5,         # 精英數量
        'tournament_size': 7,
        'mutation_amount_range': (-3, 3),
        'vix_mutation_amount_range': (-2, 2),
        'adx_mutation_amount_range': (-2, 2),
        'show_process': False,     # 批次處理時通常關閉 GA 的詳細過程顯示
        'rsi_threshold_range': (10, 40, 45, 75),
        'vix_threshold_range': (15, 35),
        'adx_threshold_range': (20, 40),
        # --- 自動從 strategy_config 複製參數 ---
        'rsi_period_options': strategy_config['rsi_period_options'],
        'vix_ma_period_options': strategy_config['vix_ma_period_options'],
        'bb_length_options': strategy_config['bb_length_options'],
        'bb_std_options': strategy_config['bb_std_options'],
        'adx_period': strategy_config['adx_period'],
        'bbi_periods': strategy_config['bbi_periods'],
        'ma_short_period': strategy_config['ma_short_period'],
        'ma_long_period': strategy_config['ma_long_period'],
        'commission_rate': strategy_config['commission_pct'],
    }
    # --- 結束設定 ---

    print("-" * 60)
    print("開始批次執行股票策略優化...")
    print(f"讀取股票列表: {TICKER_CSV_FILE}")
    print(f"回測期間: {START_DATE} 到 {END_DATE}")
    print(f"VIX 代碼: {VIX_TICKER}")
    print(f"*** 每個股票將運行 GA {NUM_GA_RUNS_PER_STOCK} 次，並選取最佳結果 ***")
    print("使用的策略配置選項:")
    print(f"  RSI Periods: {strategy_config['rsi_period_options']}")
    print(f"  VIX MA Periods: {strategy_config['vix_ma_period_options']}")
    print(f"  BB Lengths: {strategy_config['bb_length_options']}")
    print(f"  BB Stds: {strategy_config['bb_std_options']}")
    print(f"GA 參數 (單次運行): Gen={ga_config['generations']}, Pop={ga_config['population_size']}, Elitism={ga_config['elitism_size']}, Runs/Stock={NUM_GA_RUNS_PER_STOCK}")
    print(f"資料庫主機: {DB_CONFIG['host']}, 資料庫: {DB_CONFIG['database']}")
    print("-" * 60)

    # --- 讀取股票代碼 ---
    try:
        df_tickers = pd.read_csv(TICKER_CSV_FILE)
        if 'Symbol' not in df_tickers.columns:
             print(f"錯誤: CSV 文件 '{TICKER_CSV_FILE}' 中缺少 'Symbol' 列。")
             return
        tickers = df_tickers['Symbol'].dropna().unique().tolist()
        # 過濾掉可能無效的代碼 (例如包含 '^', '/', 或其他非標準字符)
        tickers = [t for t in tickers if isinstance(t, str) and t.isalnum() and len(t) > 0]
        # 也可以用更寬鬆的規則，例如允許 '.'
        # tickers = [t for t in tickers if isinstance(t, str) and t.replace('.', '').isalnum() and len(t) > 0 and '^' not in t]

        # 排除最後一行可能的說明文字（更健壯的方式）
        if tickers and "Downloaded from" in tickers[-1]:
            tickers = tickers[:-1]

        print(f"找到 {len(tickers)} 個有效的股票代碼進行處理。")
        if not tickers:
            print("錯誤: 未在 CSV 中找到有效的股票代碼。")
            return
    except FileNotFoundError:
        print(f"錯誤: 找不到股票代碼文件 '{TICKER_CSV_FILE}'。")
        return
    except Exception as e:
        print(f"讀取 CSV 文件時發生錯誤: {e}")
        return

    total_start_time = time.time()
    optimization_success_count = 0 # 成功完成優化的股票數
    db_save_success_count = 0    # 成功儲存到DB的股票數
    error_skip_count = 0         # 因錯誤跳過的股票數

    # --- 遍歷每個股票代碼 ---
    for i, ticker in enumerate(tickers):
        ticker_start_time = time.time()
        print(f"\n[{i+1}/{len(tickers)}] 🚀 處理中: {ticker}...")
        optimization_completed = False # 標記該股票是否完成優化流程

        try:
            # 1. 載入數據
            print(f"  - 載入數據 ({START_DATE} to {END_DATE})...")
            prices, dates, stock_df, vix_series = load_stock_data(
                ticker,
                vix_ticker=VIX_TICKER,
                start_date=START_DATE,
                end_date=END_DATE
            )
            if prices is None or dates is None or stock_df is None or vix_series is None or not prices:
                print(f"  ❌ 無法載入或處理 {ticker} 的數據，跳過此股票。")
                error_skip_count += 1
                continue

            # 2. 預計算指標
            print(f"  - 預計算指標...")
            indicator_calc_start = time.time()
            precalculated_indicators, indicators_ready = precompute_indicators(stock_df, vix_series, strategy_config)
            indicator_calc_time = time.time() - indicator_calc_start
            print(f"    指標計算耗時: {indicator_calc_time:.2f} 秒")
            if not indicators_ready:
                print(f"  ❌ {ticker} 的指標計算失敗，跳過此股票。")
                error_skip_count += 1
                continue

            # 3. 執行多次遺傳演算法優化
            print(f"  - 執行 {NUM_GA_RUNS_PER_STOCK} 次遺傳演算法...")
            stock_run_results = [] # 儲存該股票每次 GA 運行的結果
            total_ga_time_stock = 0

            for run_num in range(NUM_GA_RUNS_PER_STOCK):
                run_start_time = time.time()
                print(f"    > GA 運行 {run_num + 1}/{NUM_GA_RUNS_PER_STOCK}...")
                best_gene_run, best_fitness_run = genetic_algorithm_with_elitism(
                    prices, dates,
                    precalculated_indicators,
                    ga_params=ga_config
                )
                run_time = time.time() - run_start_time
                total_ga_time_stock += run_time

                if best_gene_run is not None and np.isfinite(best_fitness_run):
                    stock_run_results.append({'gene': best_gene_run, 'fitness': best_fitness_run})
                    print(f"      完成 ({run_time:.2f} 秒) - Fitness: {best_fitness_run:.4f}")
                else:
                    print(f"      失敗 ({run_time:.2f} 秒) - 未找到有效解。")

            print(f"    所有 GA 運行總耗時: {total_ga_time_stock:.2f} 秒")

            # 4. 從多次運行中選取最佳結果
            if not stock_run_results: # 如果所有運行都失敗
                print(f"  ❌ {ticker} 的所有 {NUM_GA_RUNS_PER_STOCK} 次 GA 運行均未找到有效解，跳過此股票。")
                error_skip_count += 1
                continue

            overall_best_run_for_stock = max(stock_run_results, key=lambda item: item['fitness'])
            final_best_gene = overall_best_run_for_stock['gene']
            final_best_fitness = overall_best_run_for_stock['fitness']

            print(f"  ✅ {ticker} 多次優化完成 - 選取最佳 Fitness: {final_best_fitness:.4f} (來自 {len(stock_run_results)} 次成功運行)")
            optimization_completed = True # 標記優化成功

            # 5. 儲存最終最佳結果到資料庫
            print(f"  - 儲存最佳結果到資料庫...")
            save_success = save_to_mysql(ticker, final_best_gene, final_best_fitness)
            if save_success:
                db_save_success_count += 1 # 記錄成功存儲

        except KeyboardInterrupt:
            print("\n🛑 收到用戶中斷請求，停止批次處理...")
            break # 跳出外層循環
        except MemoryError:
             print(f"  ❌ 處理 {ticker} 時發生內存錯誤，跳過此股票。")
             error_skip_count += 1
             # 給系統一點時間恢復
             time.sleep(5)
        except Exception as e:
            print(f"  ❌ 處理 {ticker} 時發生未預期錯誤: {type(e).__name__}: {e}")
            traceback.print_exc()
            error_skip_count += 1
            # 短暫延遲避免連續錯誤
            time.sleep(2)

        finally:
            if optimization_completed:
                optimization_success_count += 1 # 只有真正完成優化流程的才計入
            ticker_time = time.time() - ticker_start_time
            print(f"  ⏱️ {ticker} 處理總耗時: {ticker_time:.2f} 秒")

    # --- 批次處理結束 ---
    total_time = time.time() - total_start_time
    print("-" * 60)
    print("批次優化處理完成！")
    print(f"總耗時: {total_time:.2f} 秒 ({total_time/60:.2f} 分鐘)")
    print(f"成功完成優化的股票數量: {optimization_success_count}")
    print(f"成功儲存到資料庫的股票數量: {db_save_success_count}")
    print(f"因錯誤或數據問題跳過的股票數量: {error_skip_count}")
    print(f"總處理股票數量 (嘗試): {optimization_success_count + error_skip_count} / {len(tickers)}")
    print("-" * 60)


if __name__ == '__main__':
    # --- 執行主函數 ---
    run_batch_optimization()