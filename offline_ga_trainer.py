# -*- coding: utf-8 -*-
import csv
import json
import datetime
import logging
import os
import sys
import time
import random
import traceback
import re

import pymysql
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import secrets # 用於生成唯一文件名

# --- 從 ga_engine 導入 GA 相關功能 ---
try:
    from ga_engine import (
        ga_load_stock_data,
        ga_precompute_indicators,
        genetic_algorithm_with_elitism,
        run_strategy,
        format_ga_gene_parameters_to_text, # 確保導入此函數
        STRATEGY_CONFIG_SHARED_GA,
        GA_PARAMS_CONFIG
    )
    GA_ENGINE_IMPORTED = True
    logger_ga_engine = logging.getLogger("GAEngine") 
    if not logger_ga_engine.hasHandlers():
        handler_ge = logging.StreamHandler()
        formatter_ge = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
        handler_ge.setFormatter(formatter_ge)
        logger_ga_engine.addHandler(handler_ge)
        logger_ga_engine.setLevel(logging.INFO)

except ImportError as e:
    logging.critical(f"致命錯誤: 無法從 ga_engine.py 導入: {e}。離線訓練器無法運行。")
    # 提供 dummy 函數以避免直接崩潰，儘管 trainer 的意義不大
    def ga_load_stock_data(*args, **kwargs): logging.error("ga_load_stock_data (dummy) called!"); return None, None, None, None
    def ga_precompute_indicators(*args, **kwargs): logging.error("ga_precompute_indicators (dummy) called!"); return {}, False
    def genetic_algorithm_with_elitism(*args, **kwargs): logging.error("genetic_algorithm_with_elitism (dummy) called!"); return None, -float('inf')
    def run_strategy(*args, **kwargs): logging.error("run_strategy (dummy) called!"); return [], [], []
    def format_ga_gene_parameters_to_text(*args, **kwargs): logging.error("format_ga_gene_parameters_to_text (dummy) called!"); return "參數描述不可用 (導入錯誤)"
    STRATEGY_CONFIG_SHARED_GA = {}
    GA_PARAMS_CONFIG = {}
    GA_ENGINE_IMPORTED = False
    sys.exit("離線訓練器因缺少 ga_engine 而無法運行。請檢查導入。")


# --- Settings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("OfflineGATrainer")

# --- Database Config & Helper ---
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD"), 
    'database': os.getenv("DB_NAME", "finsimu_db"),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'connect_timeout': 10
}

if DB_CONFIG['password'] is None:
    logger.error("致命錯誤: 未在環境變數中找到資料庫密碼 (DB_PASSWORD)。")
    sys.exit("請設置 DB_PASSWORD 環境變數。")


def get_db_connection():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except pymysql.Error as e:
        logger.error(f"資料庫連接錯誤: {e}")
        return None

def execute_db_query(query, args=None, fetch_one=False, fetch_all=False, commit=False, conn_param=None):
    conn_to_use = conn_param if conn_param else get_db_connection()
    if not conn_to_use:
        logger.error(f"資料庫查詢失敗: 無連接. 查詢: {query}")
        return None
    result = None
    try:
        with conn_to_use.cursor() as cursor:
            cursor.execute(query, args)
            if commit:
                if not conn_param: conn_to_use.commit()
                result = cursor.lastrowid if cursor.lastrowid else cursor.rowcount
            elif fetch_one: result = cursor.fetchone()
            elif fetch_all: result = cursor.fetchall()
    except pymysql.Error as e:
        logger.error(f"資料庫查詢錯誤: {e}\n查詢: {query}\n參數: {args}", exc_info=True)
        if conn_to_use and commit and not conn_param: conn_to_use.rollback()
        return None
    finally:
        if conn_to_use and not conn_param: conn_to_use.close()
    return result

# --- 延遲設定 ---
PROCESS_BATCH_SIZE = GA_PARAMS_CONFIG.get('trainer_process_batch_size', 50)  # 每處理多少支股票後暫停
PAUSE_DURATION_SECONDS = GA_PARAMS_CONFIG.get('trainer_pause_duration_seconds', 10 * 60)  # 暫停時長（秒）

# --- 繪圖函數 ---
def plot_ga_strategy_results(dates, prices, portfolio_values, buy_signals, sell_signals, 
                             ticker, market_type, strategy_params_desc, 
                             final_fitness, charts_static_dir="static/charts"):
    if not dates or not prices or not portfolio_values:
        logger.warning(f"繪圖失敗({ticker}): 缺少日期、價格或投資組合數據。")
        return None

    if not os.path.exists(charts_static_dir):
        try:
            os.makedirs(charts_static_dir)
            logger.info(f"已創建繪圖目錄: {charts_static_dir}")
        except OSError as e:
            logger.error(f"創建繪圖目錄 {charts_static_dir} 失敗: {e}")
            return None
            
    fig = make_subplots(rows=1, cols=1)

    initial_price = prices[0] if prices and np.isfinite(prices[0]) and prices[0] > 1e-9 else 1.0
    norm_prices = np.array(prices, dtype=float) / initial_price

    fig.add_trace(go.Scatter(x=dates, y=norm_prices, mode='lines', name=f'{ticker} 標準化價格', line=dict(color='grey', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=portfolio_values, mode='lines', name=f'GA策略 (適應度: {final_fitness:.4f})', line=dict(color='blue', width=1.5)), row=1, col=1)

    if buy_signals:
        buy_dates = [s[0] for s in buy_signals]
        buy_prices_norm = np.array([s[1] for s in buy_signals], dtype=float) / initial_price
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices_norm, mode='markers', name='買入點', 
                                 marker=dict(color='green', size=8, symbol='triangle-up', line=dict(width=1, color='DarkSlateGrey'))), row=1, col=1)
    if sell_signals:
        sell_dates = [s[0] for s in sell_signals]
        sell_prices_norm = np.array([s[1] for s in sell_signals], dtype=float) / initial_price
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices_norm, mode='markers', name='賣出點',
                                 marker=dict(color='red', size=8, symbol='triangle-down', line=dict(width=1, color='DarkSlateGrey'))), row=1, col=1)

    # 簡化標題中的策略參數描述長度
    strategy_params_short = (strategy_params_desc[:150] + '...') if len(strategy_params_desc) > 150 else strategy_params_desc
    title_text = f"{ticker} ({market_type}) - GA策略回測<br><sub>適應度: {final_fitness:.4f} | 策略(簡): {strategy_params_short.replace('<br>', ' ').replace('<b>', '').replace('</b>','')}</sub>"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=12)), # 調整標題字體大小
        xaxis_title="日期",
        yaxis_title="標準化價值 / 策略價值",
        legend_title_text='圖例',
        template='plotly_white', 
        height=500,
        margin=dict(l=50, r=50, t=100, b=50) # 增加頂部邊距給標題
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(0,0,0,0.1)')
    
    chart_filename_base = f"ga_train_{ticker.replace('.', '_')}_{market_type}_{secrets.token_hex(3)}.html" # 縮短隨機碼
    chart_path_disk = os.path.join(charts_static_dir, chart_filename_base)
    
    try:
        fig.write_html(chart_path_disk, full_html=False, include_plotlyjs='cdn')
        logger.info(f"已為 {ticker} ({market_type}) 保存GA策略回測圖表至: {chart_path_disk}")
        return chart_path_disk 
    except Exception as e_plot:
        logger.error(f"為 {ticker} ({market_type}) 保存GA策略回測圖表失敗: {e_plot}", exc_info=True)
        return None

# --- 離線訓練主函數 ---
def run_offline_training(stock_list_csv_path, market_type_for_csv,
                         train_start_date, train_end_date,
                         system_ai_user_id=0, 
                         num_ga_runs=GA_PARAMS_CONFIG.get('offline_trainer_runs_per_stock', 20)):
    if not GA_ENGINE_IMPORTED:
        logger.critical("GA 引擎未導入。離線訓練無法繼續。")
        return

    logger.info(f"--- 開始為市場 {market_type_for_csv} 從 {stock_list_csv_path} 進行離線 GA 訓練 ---")
    logger.info(f"訓練期間: {train_start_date} 至 {train_end_date}")
    logger.info(f"每支股票的 GA 運行次數: {num_ga_runs}")
    logger.info(f"每處理 {PROCESS_BATCH_SIZE} 支股票後將暫停 {PAUSE_DURATION_SECONDS // 60} 分鐘。") 
    logger.info(f"使用 GA_PARAMS_CONFIG: Generations={GA_PARAMS_CONFIG.get('generations')}, Population={GA_PARAMS_CONFIG.get('population_size')}")

    tickers_to_train = []
    try:
        with open(stock_list_csv_path, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            symbol_col_name = None
            possible_symbol_cols = ['Symbol', 'symbol', 'Ticker', 'ticker', '股票代號']
            for col_name_option in possible_symbol_cols:
                if col_name_option in reader.fieldnames:
                    symbol_col_name = col_name_option
                    break
            if not symbol_col_name:
                logger.error(f"CSV 檔案 {stock_list_csv_path} 必須包含以下任一股票代號列名: {possible_symbol_cols}。")
                return
            
            for row in reader:
                symbol_value = row.get(symbol_col_name)
                if symbol_value and symbol_value.strip():
                    tickers_to_train.append(symbol_value.strip().upper())
    except FileNotFoundError:
        logger.error(f"股票列表 CSV 檔案未找到: {stock_list_csv_path}")
        return
    except Exception as e:
        logger.error(f"讀取 CSV {stock_list_csv_path} 時發生錯誤: {e}")
        return

    if not tickers_to_train:
        logger.info(f"在 {stock_list_csv_path} 中未找到可訓練的股票代號。")
        return

    logger.info(f"找到 {len(tickers_to_train)} 個股票代號為市場 {market_type_for_csv} 進行訓練。")

    ga_params_to_use = GA_PARAMS_CONFIG.copy()
    strategy_config_to_use = STRATEGY_CONFIG_SHARED_GA.copy()

    # 確定 static/charts 目錄的路徑
    # 假設 offline_ga_trainer.py 與 app.py 在同一父目錄下 (例如 StockAnalyzer/)
    # 且 static 文件夾位於 StockAnalyzer/static
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.dirname(current_script_dir) # 假設此腳本在專案的子目錄中
                                                        # 如果此腳本就在專案根目錄，則 project_root_dir = current_script_dir
    # 如果你的結構是 StockAnalyzer/offline_ga_trainer.py 和 StockAnalyzer/app.py, StockAnalyzer/static/charts
    # 則 project_root_dir 應該是 current_script_dir (如果 offline_ga_trainer.py 在根目錄)
    # 或者 os.path.dirname(current_script_dir) (如果 offline_ga_trainer.py 在類似 'scripts' 的子目錄)
    # 最安全的方式是確保你知道 static 文件夾相對於此腳本的確切路徑
    # 這裡假設 offline_ga_trainer.py 和 app.py 在同一個目錄，而 static 是其子目錄
    # 如果 offline_ga_trainer.py 在根目錄，而 app.py 也在根目錄，則：
    charts_dir_for_saving = os.path.join(current_script_dir, "static", "charts")
    if not os.path.isdir(os.path.join(current_script_dir, "static")): # 檢查 static 目錄是否存在
        logger.warning(f"警告：在 {current_script_dir} 下未找到 'static' 目錄。圖表可能無法正確保存或提供服務。")
        # 你可能需要手動創建 static/charts 或調整路徑


    for i, ticker_raw in enumerate(tickers_to_train):
        logger.info(f"--- ({i+1}/{len(tickers_to_train)}) 處理股票代號: {ticker_raw} (市場: {market_type_for_csv}) ---")

        if i > 0 and i % PROCESS_BATCH_SIZE == 0: 
            logger.info(f"已處理 {i} 支股票，現在暫停 {PAUSE_DURATION_SECONDS // 60} 分鐘...")
            time.sleep(PAUSE_DURATION_SECONDS)
            logger.info("暫停結束，繼續訓練...")
        
        ticker = ticker_raw
        if market_type_for_csv == "TW" and not ticker.endswith(".TW"):
            if re.fullmatch(r'\d{4,6}', ticker): 
                ticker = f"{ticker}.TW"
            else:
                logger.warning(f"股票代號 {ticker_raw} (來自TW列表) 格式看起來不正確。跳過。")
                continue
        elif market_type_for_csv == "US" and ticker.endswith(".TW"):
            logger.warning(f"股票代號 {ticker_raw} (來自US列表) 看起來像台股代號。跳過。")
            continue

        try:
            force_retrain = True
            if not force_retrain:
                existing_training = execute_db_query(
                    """SELECT game_end_date, game_completed_at
                       FROM ai_vs_user_games
                       WHERE stock_ticker = %s AND market_type = %s AND user_id = %s
                             AND ai_strategy_gene IS NOT NULL
                       ORDER BY game_end_date DESC, game_completed_at DESC LIMIT 1""",
                    (ticker, market_type_for_csv, system_ai_user_id), fetch_one=True
                )
                if existing_training:
                    train_end_date_obj = datetime.datetime.strptime(train_end_date, "%Y-%m-%d").date()
                    if existing_training['game_end_date'] >= train_end_date_obj:
                        logger.info(f"跳過 {ticker}: 資料庫中已存在於 {train_end_date} 或之後結束的有效訓練數據。")
                        continue
                    else:
                        logger.info(f"{ticker}: 資料庫中存在舊的訓練數據 (結束於 {existing_training['game_end_date']})，將重新訓練至 {train_end_date}。")
            else:
                 logger.info(f"為 {ticker} 強制執行重新訓練。")

            prices, dates, stock_df, vix_series = ga_load_stock_data(
                ticker, vix_ticker="^VIX", 
                start_date=train_start_date, end_date=train_end_date,
                verbose=False 
            )

            if not prices or not dates or stock_df is None or stock_df.empty or vix_series is None or vix_series.empty:
                logger.error(f"股票 {ticker} 數據加載失敗。跳過 GA 訓練。")
                continue

            precalculated_indicators, indicators_ready = ga_precompute_indicators(
                stock_df, vix_series, strategy_config_to_use, verbose=False
            )

            if not indicators_ready:
                logger.error(f"股票 {ticker} 指標預計算失敗。跳過 GA 訓練。")
                continue

            rsi_lists_for_ga = precalculated_indicators.get('rsi', {})
            vix_ma_lists_for_ga = precalculated_indicators.get('vix_ma', {})
            bbl_lists_for_ga = precalculated_indicators.get('bbl', {})
            bbm_lists_for_ga = precalculated_indicators.get('bbm', {})
            adx_list_for_ga = precalculated_indicators.get('fixed', {}).get('adx_list', [])
            ma_short_list_for_ga = precalculated_indicators.get('fixed', {}).get('ma_short_list', [])
            ma_long_list_for_ga = precalculated_indicators.get('fixed', {}).get('ma_long_list', [])
            
            if not all([rsi_lists_for_ga, vix_ma_lists_for_ga, bbl_lists_for_ga, bbm_lists_for_ga, 
                        adx_list_for_ga, ma_short_list_for_ga, ma_long_list_for_ga]):
                logger.error(f"股票 {ticker} 的一個或多個必要預計算指標列表缺失或為空。跳過 GA。")
                continue

            overall_best_gene_for_ticker = None
            overall_best_fitness_for_ticker = -float('inf')

            logger.info(f"開始為 {ticker} 進行 {num_ga_runs} 次 GA 運行...")
            start_ticker_ga_time = time.time()

            for run_num in range(num_ga_runs):
                if not ga_params_to_use.get('show_process', False): 
                    logger.info(f"  GA 運行 {run_num + 1}/{num_ga_runs} (股票: {ticker}) 開始...")

                current_best_gene_run, current_fitness_run = genetic_algorithm_with_elitism(
                    prices, dates,
                    rsi_lists_for_ga, vix_ma_lists_for_ga,
                    bbl_lists_for_ga, bbm_lists_for_ga,
                    adx_list_for_ga,
                    ma_short_list_for_ga, ma_long_list_for_ga,
                    ga_params=ga_params_to_use
                )
                if current_best_gene_run and current_fitness_run > overall_best_fitness_for_ticker:
                    overall_best_fitness_for_ticker = current_fitness_run
                    overall_best_gene_for_ticker = current_best_gene_run

                if not ga_params_to_use.get('show_process', False):
                     fitness_display_str = f"{current_fitness_run:.4f}" if current_best_gene_run and np.isfinite(current_fitness_run) else 'N/A (無效或無基因)'
                     logger.info(f"  GA 運行 {run_num + 1}/{num_ga_runs} (股票: {ticker}) | 適應度: {fitness_display_str}")

            ticker_ga_duration = time.time() - start_ticker_ga_time
            logger.info(f"{ticker} 的所有 {num_ga_runs} 次 GA 運行完成，耗時 {ticker_ga_duration:.2f} 秒。總體最佳適應度: {overall_best_fitness_for_ticker:.4f}")

            if overall_best_gene_for_ticker and overall_best_fitness_for_ticker > -float('inf'):
                best_gene = overall_best_gene_for_ticker
                chosen_rsi_p = strategy_config_to_use['rsi_period_options'][best_gene[4]]
                chosen_vix_ma_p = strategy_config_to_use['vix_ma_period_options'][best_gene[5]]
                chosen_bb_l = strategy_config_to_use['bb_length_options'][best_gene[6]]
                chosen_bb_s = strategy_config_to_use['bb_std_options'][best_gene[7]]

                final_portfolio_values, final_buy_signals, final_sell_signals = run_strategy(
                    best_gene[0], best_gene[1], best_gene[8], best_gene[2], 
                    best_gene[3], best_gene[9],
                    strategy_config_to_use['commission_pct'],
                    prices, dates,
                    rsi_lists_for_ga[chosen_rsi_p],
                    bbl_lists_for_ga[(chosen_bb_l, chosen_bb_s)],
                    bbm_lists_for_ga[(chosen_bb_l, chosen_bb_s)],
                    adx_list_for_ga,
                    vix_ma_lists_for_ga[chosen_vix_ma_p],
                    ma_short_list_for_ga,
                    ma_long_list_for_ga
                )
                
                strategy_desc_for_plot = format_ga_gene_parameters_to_text(best_gene, strategy_config_to_use)

                plot_ga_strategy_results(
                    dates, prices, final_portfolio_values, 
                    final_buy_signals, final_sell_signals,
                    ticker, market_type_for_csv, strategy_desc_for_plot,
                    overall_best_fitness_for_ticker,
                    charts_static_dir=charts_dir_for_saving 
                )

                game_data = {
                    "user_id": system_ai_user_id,
                    "market_type": market_type_for_csv,
                    "stock_ticker": ticker,
                    "game_start_date": datetime.datetime.strptime(train_start_date, "%Y-%m-%d").date(),
                    "game_end_date": datetime.datetime.strptime(train_end_date, "%Y-%m-%d").date(),
                    "ai_strategy_gene": json.dumps(overall_best_gene_for_ticker), 
                    "ai_initial_cash": 1.0, 
                    "user_initial_cash": 1.0, 
                    "ai_final_portfolio_value": float(overall_best_fitness_for_ticker),
                    "user_final_portfolio_value": None, 
                    "game_completed_at": datetime.datetime.now()
                }
                insert_query = """
                    INSERT INTO ai_vs_user_games
                    (user_id, market_type, stock_ticker, game_start_date, game_end_date,
                     ai_strategy_gene, ai_initial_cash, user_initial_cash, ai_final_portfolio_value,
                     user_final_portfolio_value, game_completed_at)
                    VALUES (%(user_id)s, %(market_type)s, %(stock_ticker)s, %(game_start_date)s, %(game_end_date)s,
                            %(ai_strategy_gene)s, %(ai_initial_cash)s, %(user_initial_cash)s, %(ai_final_portfolio_value)s,
                            %(user_final_portfolio_value)s, %(game_completed_at)s)
                    ON DUPLICATE KEY UPDATE
                        game_start_date = VALUES(game_start_date),
                        game_end_date = VALUES(game_end_date),
                        ai_strategy_gene = VALUES(ai_strategy_gene),
                        ai_final_portfolio_value = VALUES(ai_final_portfolio_value),
                        game_completed_at = VALUES(game_completed_at);
                """
                result_id = execute_db_query(insert_query, game_data, commit=True)
                if result_id:
                    logger.info(f"已成功將 {ticker} 的最佳 GA 結果保存/更新到資料庫 (ID/受影響行數: {result_id})。")
                else:
                    logger.error(f"保存 {ticker} 的最佳 GA 結果到資料庫失敗。")
            else:
                logger.warning(f"{num_ga_runs} 次運行後，未找到 {ticker} 的有效最佳基因。跳過資料庫保存和繪圖。")

        except Exception as e:
            logger.error(f"為 {ticker} 進行 GA 訓練時發生錯誤: {e}", exc_info=True)
        
        time.sleep(random.uniform(0.3, 0.8)) 

    logger.info(f"--- 市場 {market_type_for_csv} 的離線 GA 訓練完成。 ---")


if __name__ == "__main__":
    nasdaq_csv_file = "nasdaq-100-index-03-14-2025.csv"
    sp100_csv_file = "sp-100-index-03-14-2025.csv"
    taiex_csv_file = "TAIEX_constituents.csv" # 假設這是你的台股列表文件名

    TRAIN_START_DATE = "2022-01-01" 
    TRAIN_END_DATE = "2025-05-25"   # 或者可以設置為動態的日期，例如 datetime.date.today().strftime("%Y-%m-%d")

    SYSTEM_AI_TRAINER_USER_ID = 2 
    num_ga_runs_config = GA_PARAMS_CONFIG.get('offline_trainer_runs_per_stock', 50) 

    # 決定是否運行某個市場的訓練
    run_nasdaq = True if os.path.exists(nasdaq_csv_file) else False
    run_sp100 = True if os.path.exists(sp100_csv_file) else False
    run_taiex = True if os.path.exists(taiex_csv_file) else False

    if not (run_nasdaq or run_sp100 or run_taiex):
        logger.error("所有指定的股票列表 CSV 文件均未找到。訓練無法開始。")
        sys.exit("請提供至少一個有效的股票列表 CSV 文件。")

    if run_nasdaq:
        logger.info("\n=== 訓練美國股票 (NASDAQ 100) ===")
        run_offline_training(nasdaq_csv_file, "US", TRAIN_START_DATE, TRAIN_END_DATE, SYSTEM_AI_TRAINER_USER_ID, num_ga_runs=num_ga_runs_config)
    else:
        logger.warning(f"{nasdaq_csv_file} 未找到。跳過 NASDAQ 訓練。")


    if run_sp100:
        logger.info("\n=== 訓練美國股票 (S&P 100) ===")
        run_offline_training(sp100_csv_file, "US", TRAIN_START_DATE, TRAIN_END_DATE, SYSTEM_AI_TRAINER_USER_ID, num_ga_runs=num_ga_runs_config)
    else:
        logger.warning(f"{sp100_csv_file} 未找到。跳過 S&P 100 訓練。")

    if run_taiex:
        logger.info("\n=== 訓練台灣股票 (TAIEX 成分股) ===")
        run_offline_training(taiex_csv_file, "TW", TRAIN_START_DATE, TRAIN_END_DATE, SYSTEM_AI_TRAINER_USER_ID, num_ga_runs=num_ga_runs_config)
    else:
        logger.warning(f"{taiex_csv_file} 未找到。跳過 TAIEX 訓練。")
        
    logger.info("所有離線 GA 訓練任務已完成。")