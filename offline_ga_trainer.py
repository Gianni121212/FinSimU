
import csv
import json
import datetime
import os
import sys
import time
import random
import traceback
import pymysql
import pandas as pd
import numpy as np
import re
# --- 導入核心模組 ---
try:
    from ga_engine import (
        ga_load_data,
        ga_precompute_indicators,
        genetic_algorithm_unified,
        run_strategy_numba_core,
        format_ga_gene_parameters_to_text,
        STRATEGY_CONFIG_SHARED_GA,
        GA_PARAMS_CONFIG,
        GENE_MAP,
        STRAT_NAMES,
        NSGA2_AVAILABLE
    )
    from utils import calculate_performance_metrics
    GA_ENGINE_IMPORTED = True
    print("[OfflineTrainer] ✅ 核心模組載入成功")
except ImportError as e:
    print(f"❌ 致命錯誤: 無法從 ga_engine.py 或 utils.py 導入: {e}")
    print("請確保 ga_engine.py 和 utils.py 檔案存在且完整。")
    sys.exit(1)

print(f"[OfflineTrainer] 🚀 AI遺傳演算法離線訓練器 v5.2 啟動")
print(f"[OfflineTrainer] NSGA-II 支援狀態: {'✅ 已啟用' if NSGA2_AVAILABLE else '❌ 未安裝 pymoo'}")

class TrainingConfig:
    ENABLE_NSGA2 = True  # True=多目標優化, False=傳統GA
    NUM_GA_RUNS_PER_STOCK = 50  # 每支股票運行幾次GA (建議30-100)
    TOP_N_STRATEGIES_TO_SAVE = 3  # 保存最佳N個策略
    TRAIN_START_DATE = "2022-09-06"  # 訓練開始日期
    TRAIN_END_DATE = "2025-09-06"    # 訓練結束日期
    SYSTEM_AI_USER_ID = 2  # 系統AI用戶ID
    

    STOCKS_PER_BATCH = 70      # 每處理N支股票暫停一次
    PAUSE_DURATION_MINUTES = 5  # 暫停N分鐘
    INDIVIDUAL_STOCK_DELAY = (0.8, 2.0)  # 每支股票間隨機延遲秒數範圍
    
    
    NSGA2_CONFIG = {
        'nsga2_selection_method': 'custom_balance',  # 🔧 可選方法：
        'min_required_trades': 5,      # 最少交易次數要求
        'generations': 5,             # NSGA-II 迭代次數
        'population_size': 70,         # NSGA-II 種群大小
        'show_process': False,         # 是否顯示詳細過程
        
        'custom_weights': {
            'total_return_weight': 0.45,      # 總報酬率權重
            'avg_trade_return_weight': 0.35,  # 平均交易報酬率權重 
            'win_rate_weight': 0.10,          # 勝率權重
            'trade_count_weight': 0,       # 交易次數權重
            'drawdown_weight': 0.10           # 回撤懲罰權重
        },
        
        'aggressive_settings': {
            'return_threshold': 0.30,   
            'total_weight': 0.65,      
            'avg_trade_weight': 0.35,  
        }
    }
    
    TRADITIONAL_GA_CONFIG = {
        'generations': 45,               # 傳統GA迭代次數  
        'population_size': 80,           # 傳統GA種群大小
        'no_trade_penalty_factor': 0.05, # 無交易懲罰因子
        'low_trade_penalty_factor': 0.75, # 低交易懲罰因子
        'show_process': False,           # 是否顯示詳細過程
    }
    RISK_FREE_RATE = 0.025
    STOCK_LIST_FILES = {
        'TAIEX': "tw_stock.csv",      # 台股清單
        'NASDAQ': "usa_stock.csv",  # NASDAQ 100
        'SP100': "sp-100-index-03-14-2025.csv",       # S&P 100
    }
    
    MARKETS_TO_TRAIN = {
        'TAIEX': True,  
        'NASDAQ': True,  
        'SP100': False,   
    }

DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME", "finsimu_db"),
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor,
    'connect_timeout': 15
}

if not DB_CONFIG['password']:
    print("❌ 致命錯誤: 未設定 DB_PASSWORD 環境變數")
    print("請在環境變數中設定資料庫密碼")
    sys.exit(1)

def get_db_connection():
    """建立資料庫連接"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except pymysql.Error as e:
        print(f"❌ 資料庫連接錯誤: {e}")
        return None

def calculate_detailed_metrics_for_traditional_ga(gene_result, prices, dates, precalculated, ga_params):
        try:
            def get_indicator_list(name, gene_indices, opt_keys, precalc_data):
                params = [ga_params[k][gene_result[g_idx]] for g_idx, k in zip(gene_indices, opt_keys)]
                key = tuple(params) if len(params) > 1 else params[0]
                return np.array(precalc_data.get(name, {}).get(key, [np.nan] * len(prices)))

            vix_ma_arr = get_indicator_list('vix_ma', [GENE_MAP['vix_ma_p']], ['vix_ma_period_options'], precalculated)
            sent_ma_arr = get_indicator_list('sentiment_ma', [GENE_MAP['sentiment_ma_p']], ['sentiment_ma_period_options'], precalculated)
            rsi_arr = get_indicator_list('rsi', [GENE_MAP['rsi_p']], ['rsi_period_options'], precalculated)
            adx_arr = get_indicator_list('adx', [GENE_MAP['adx_p']], ['adx_period_options'], precalculated)
            bb_key_indices = [GENE_MAP['bb_l_p'], GENE_MAP['bb_s_p']]
            bb_key_opts = ['bb_length_options', 'bb_std_options']
            bbl_arr = get_indicator_list('bbl', bb_key_indices, bb_key_opts, precalculated)
            bbm_arr = get_indicator_list('bbm', bb_key_indices, bb_key_opts, precalculated)
            bbu_arr = get_indicator_list('bbu', bb_key_indices, bb_key_opts, precalculated)
            ma_s_arr = get_indicator_list('ma', [GENE_MAP['ma_s_p']], ['ma_period_options'], precalculated)
            ma_l_arr = get_indicator_list('ma', [GENE_MAP['ma_l_p']], ['ma_period_options'], precalculated)
            ema_s_arr = get_indicator_list('ema_s', [GENE_MAP['ema_s_p']], ['ema_s_period_options'], precalculated)
            ema_m_arr = get_indicator_list('ema_m', [GENE_MAP['ema_m_p']], ['ema_m_period_options'], precalculated)
            ema_l_arr = get_indicator_list('ema_l', [GENE_MAP['ema_l_p']], ['ema_l_period_options'], precalculated)
            atr_arr = get_indicator_list('atr', [GENE_MAP['atr_p']], ['atr_period_options'], precalculated)
            atr_ma_arr = get_indicator_list('atr_ma', [GENE_MAP['atr_p']], ['atr_period_options'], precalculated)
            kd_key_indices = [GENE_MAP['kd_k_p'], GENE_MAP['kd_d_p'], GENE_MAP['kd_s_p']]
            kd_key_opts = ['kd_k_period_options', 'kd_d_period_options', 'kd_smooth_period_options']
            k_arr = get_indicator_list('kd_k', kd_key_indices, kd_key_opts, precalculated)
            d_arr = get_indicator_list('kd_d', kd_key_indices, kd_key_opts, precalculated)
            macd_key_indices = [GENE_MAP['macd_f_p'], GENE_MAP['macd_s_p'], GENE_MAP['macd_sig_p']]
            macd_key_opts = ['macd_fast_period_options', 'macd_slow_period_options', 'macd_signal_period_options']
            macd_line_arr = get_indicator_list('macd_line', macd_key_indices, macd_key_opts, precalculated)
            macd_signal_arr = get_indicator_list('macd_signal', macd_key_indices, macd_key_opts, precalculated)

            (portfolio_values, buy_indices, buy_prices, sell_indices, sell_prices, num_trades_from_numba) = run_strategy_numba_core(
                np.array(gene_result, dtype=np.float64), np.array(prices),
                vix_ma_arr, sent_ma_arr, rsi_arr, adx_arr,
                bbl_arr, bbm_arr, bbu_arr, ma_s_arr, ma_l_arr,
                ema_s_arr, ema_m_arr, ema_l_arr, atr_arr, atr_ma_arr,
                k_arr, d_arr, macd_line_arr, macd_signal_arr,
                ga_params['commission_rate'], 61
            )
            buy_signals_formatted = [{'date': dates[i], 'price': buy_prices[idx]} for idx, i in enumerate(buy_indices)]
            sell_signals_formatted = [{'date': dates[i], 'price': sell_prices[idx]} for idx, i in enumerate(sell_indices)]
            
            detailed_metrics = calculate_performance_metrics(
                portfolio_values.tolist(),
                dates,
                buy_signals_formatted,
                sell_signals_formatted,
                prices,
                risk_free_rate=ga_params.get('risk_free_rate', 0.04),
                commission_rate=ga_params.get('commission_rate', 0.005)
            )
            return detailed_metrics

        except Exception as e:
            print(f"❌ 計算詳細指標時發生錯誤: {e}")
            traceback.print_exc()
            # 返回一個包含所有鍵的失敗物件，以避免後續錯誤
            return {
                'total_return': 0, 'max_drawdown': 1, 'profit_factor': 0.01,
                'trade_count': 0, 'std_dev': 1, 'win_rate_pct': 0, 'sharpe_ratio': 0,
                'average_trade_return': 0, 'max_trade_drop_pct': 0.0, 'max_trade_gain_pct': 0.0
            }
def run_offline_training(stock_list_csv_path, market_type, config):
    print(f"\n{'='*80}")
    print(f" 開始為市場 {market_type} 進行離線 GA 訓練")
    print(f" 股票清單: {stock_list_csv_path}")
    print(f" 訓練期間: {config.TRAIN_START_DATE} ~ {config.TRAIN_END_DATE}")
    print(f" 每股運行次數: {config.NUM_GA_RUNS_PER_STOCK}")
    print(f" 保存最佳策略數: {config.TOP_N_STRATEGIES_TO_SAVE}")
    print(f" 優化方法: {'NSGA-II 多目標優化' if config.ENABLE_NSGA2 else '傳統單目標GA'}")
    if config.ENABLE_NSGA2:
        print(f" 選擇方法: {config.NSGA2_CONFIG['nsga2_selection_method']}")
    print(f" 暫停設定: 每{config.STOCKS_PER_BATCH}支暫停{config.PAUSE_DURATION_MINUTES}分鐘")
    print(f" 新功能: 單次交易最大跌幅/漲幅分析")
    print(f"{'='*80}")

    if config.ENABLE_NSGA2 and not NSGA2_AVAILABLE:
        print("  NSGA-II 已啟用但 pymoo 未安裝，自動切換為傳統 GA")
        config.ENABLE_NSGA2 = False

    try:
        tickers_df = pd.read_csv(stock_list_csv_path)
        symbol_col = next((col for col in ['Symbol', 'symbol', 'Ticker', 'ticker', '股票代號', 'Code', 'code']
                          if col in tickers_df.columns), None)
        if not symbol_col:
            print(f"❌ CSV檔案 {stock_list_csv_path} 必須包含股票代號欄位")
            return
        tickers_to_train = tickers_df[symbol_col].dropna().astype(str).str.strip().str.upper().tolist()
    except Exception as e:
        print(f"❌ 讀取股票清單檔案錯誤: {e}")
        return

    print(f"📊 找到 {len(tickers_to_train)} 支股票待訓練")

    ga_params = GA_PARAMS_CONFIG.copy()
    ga_params['nsga2_enabled'] = config.ENABLE_NSGA2
    ga_params['risk_free_rate'] = config.RISK_FREE_RATE
    if config.ENABLE_NSGA2:
        ga_params.update(config.NSGA2_CONFIG)
        print(f"🔧 使用 NSGA-II 多目標優化配置 (選擇方法: {config.NSGA2_CONFIG['nsga2_selection_method']})")
    else:
        ga_params.update(config.TRADITIONAL_GA_CONFIG)
        print("🔧 使用傳統單目標 GA 配置")

    sentiment_csv_file = '2021-2025每週新聞及情緒分析.csv'
    if not os.path.exists(sentiment_csv_file):
        print(f"⚠️  情緒分析檔案未找到: {sentiment_csv_file}")
        sentiment_csv_file = None
    else:
        print(f"✅ 已載入市場情緒數據: {sentiment_csv_file}")

    successful_trainings = 0
    failed_trainings = 0
    
    for i, ticker_raw in enumerate(tickers_to_train):
        current_stock_num = i + 1
        
        if i > 0 and i % config.STOCKS_PER_BATCH == 0:
            pause_seconds = config.PAUSE_DURATION_MINUTES * 60
            print(f"\n⏸️  已處理 {i} 支股票，暫停 {config.PAUSE_DURATION_MINUTES} 分鐘...")
            time.sleep(pause_seconds)
            print("▶️  繼續訓練中...")

        print(f"\n{'─'*60}")
        print(f"📈 ({current_stock_num}/{len(tickers_to_train)}) 正在處理: {ticker_raw} ({market_type})")
        print(f"{'─'*60}")

        try:
            ticker = None
            prices, dates, stock_df, vix_series, sentiment_series = None, None, None, None, None

            is_tw_numerical = market_type == "TW" and re.fullmatch(r'\d{4,6}', ticker_raw)

            if is_tw_numerical:
                for suffix in ['.TW', '.TWO']:
                    potential_ticker = f"{ticker_raw}{suffix}"
                    print(f"🔍 正在嘗試載入 {potential_ticker} 的歷史數據...")
                    (prices, dates, stock_df, vix_series, sentiment_series) = ga_load_data(
                        potential_ticker,
                        start_date=config.TRAIN_START_DATE,
                        end_date=config.TRAIN_END_DATE,
                        sentiment_csv_path=sentiment_csv_file,
                        verbose=False
                    )
                    if prices and len(prices) > 0:
                        ticker = potential_ticker # 成功找到，確認 ticker
                        break # 跳出迴圈
            else:
      
                ticker = ticker_raw
                print(f"🔍 正在載入 {ticker} 的歷史數據...")
                (prices, dates, stock_df, vix_series, sentiment_series) = ga_load_data(
                    ticker,
                    start_date=config.TRAIN_START_DATE,
                    end_date=config.TRAIN_END_DATE,
                    sentiment_csv_path=sentiment_csv_file,
                    verbose=False
                )

            if not prices or len(prices) < 100:
                print(f"⚠️  {ticker_raw} 數據不足或載入失敗 (已嘗試 .TW/.TWO)，跳過處理")
                failed_trainings += 1
                continue
            
            print(f"✅ 成功載入 {ticker} 的 {len(prices)} 個交易日數據")
            
            print(f"⚙️  正在預計算技術指標...")
            precalculated, indicator_ready = ga_precompute_indicators(
                stock_df, vix_series, STRATEGY_CONFIG_SHARED_GA,
                sentiment_series=sentiment_series, verbose=False
            )

            if not indicator_ready:
                print(f"⚠️  {ticker} 技術指標預計算失敗，跳過處理")
                failed_trainings += 1
                continue
            
            print(f"✅ 技術指標預計算完成")

            strategy_pool = []
            print(f"🚀 開始 {config.NUM_GA_RUNS_PER_STOCK} 輪 GA 優化...")

            for run_num in range(config.NUM_GA_RUNS_PER_STOCK):
                if run_num % 10 == 0 and run_num > 0:
                    print(f"   進度: {run_num}/{config.NUM_GA_RUNS_PER_STOCK} ({run_num/config.NUM_GA_RUNS_PER_STOCK*100:.1f}%)")
                result = genetic_algorithm_unified(prices, dates, precalculated, ga_params)
                if result is None or result[0] is None:
                    continue
                
                gene_result, _ = result # 我們不再信任 performance_result

                final_accurate_metrics = calculate_detailed_metrics_for_traditional_ga(
                    gene_result, prices, dates, precalculated, ga_params
                )
                main_fitness = final_accurate_metrics.get('total_return', -np.inf)

        
                if run_num == 0:
                    print(f"   🎯 策略結果預覽 (使用精確計算):")
                    print(f"      總報酬率: {final_accurate_metrics.get('total_return', 0)*100:.2f}%")
                    print(f"      平均交易報酬率: {final_accurate_metrics.get('average_trade_return', 0)*100:.3f}%")
                    print(f"      交易次數: {final_accurate_metrics.get('trade_count', 0)}")
                    print(f"      🆕 最大跌幅: {final_accurate_metrics.get('max_trade_drop_pct', 0):.2f}%")
                    print(f"      🆕 最大漲幅: {final_accurate_metrics.get('max_trade_gain_pct', 0):.2f}%")
                
       
                strategy_pool.append({
                    'fitness': main_fitness, 
                    'gene': tuple(gene_result), 
                    'metrics': final_accurate_metrics 
                })
            
            if not strategy_pool:
                print(f"⚠️  未找到有效策略，跳過 {ticker}")
                failed_trainings += 1
                continue

            print(f"📊 分析 {len(strategy_pool)} 個候選策略...")
            unique_genes = {}
            for s in strategy_pool:
                gene_tuple = s['gene']
                if gene_tuple not in unique_genes or s['fitness'] > unique_genes[gene_tuple]['fitness']:
                    unique_genes[gene_tuple] = {'fitness': s['fitness'], 'metrics': s['metrics']}
            top_champions = sorted(
                [{'gene': list(gene), **data} for gene, data in unique_genes.items()],
                key=lambda x: (
                    x['metrics'].get('total_return', -np.inf),
                    x['metrics'].get('average_trade_return', -np.inf),
                    x['metrics'].get('win_rate_pct', -np.inf),
                    x['metrics'].get('trade_count', -np.inf)
                ),
                reverse=True
            )[:config.TOP_N_STRATEGIES_TO_SAVE]

            if not top_champions:
                print(f"⚠️  無法確定冠軍策略，跳過 {ticker}")
                failed_trainings += 1
                continue

            print(f"🎉 為 {ticker} 找到 {len(top_champions)} 個優秀策略")
            best_strategy = top_champions[0]['metrics']
            print(f"🥇 最佳策略表現:")
            print(f"   📈 總報酬率: {best_strategy.get('total_return', 0)*100:.2f}%")
            print(f"   💰 平均交易報酬率: {best_strategy.get('average_trade_return', 0)*100:.3f}%")
            print(f"   🎯 勝率: {best_strategy.get('win_rate_pct', 0):.1f}%")
            print(f"   🔢 交易次數: {best_strategy.get('trade_count', 0)}")
            print(f"   📉 最大回撤: {best_strategy.get('max_drawdown', 0)*100:.2f}%")
            print(f"   🆕 單次交易最大跌幅: {best_strategy.get('max_trade_drop_pct', 0):.2f}%")
            print(f"   🆕 單次交易最大漲幅: {best_strategy.get('max_trade_gain_pct', 0):.2f}%")

            success = save_strategies_to_database(top_champions, ticker, market_type, config)
            if success:
                print(f"💾 成功保存 {ticker} 的策略到資料庫")
                successful_trainings += 1
            else:
                print(f"❌ 保存 {ticker} 策略到資料庫失敗")
                failed_trainings += 1
        except Exception as e_ticker:
            print(f"❌ 處理 {ticker_raw} 時發生錯誤: {e_ticker}")
            print(f"錯誤詳情: {traceback.format_exc()}")
            failed_trainings += 1
        delay_seconds = random.uniform(*config.INDIVIDUAL_STOCK_DELAY)
        time.sleep(delay_seconds)

    total_stocks = len(tickers_to_train)
    print(f"\n{'='*80}")
    print(f"🎊 市場 {market_type} 訓練完成!")
    print(f"📊 總結統計:")
    print(f"   📈 成功訓練: {successful_trainings}/{total_stocks} ({successful_trainings/total_stocks*100:.1f}%)")
    print(f"   ❌ 失败/跳過: {failed_trainings}/{total_stocks} ({failed_trainings/total_stocks*100:.1f}%)")
    print(f"{'='*80}")

def save_strategies_to_database(top_champions, ticker, market_type, config):
    conn = get_db_connection()
    if not conn:
        print("❌ 無法連接資料庫")
        return False

    try:
        with conn.cursor() as cursor:

            cursor.execute(
                """DELETE FROM ai_vs_user_games 
                   WHERE user_id = %s AND market_type = %s AND stock_ticker = %s 
                   AND strategy_rank > 0 
                   AND (strategy_details LIKE %s OR ai_strategy_gene LIKE %s)""",
                (config.SYSTEM_AI_USER_ID, market_type, ticker,
                 '%System A%', '%"length": 28%')  
            )

            for rank, champion in enumerate(top_champions):
                best_gene = champion['gene']
                metrics = champion['metrics']


                game_data = {
                    "user_id": config.SYSTEM_AI_USER_ID,
                    "market_type": market_type,
                    "stock_ticker": ticker,
                    "game_start_date": config.TRAIN_START_DATE,
                    "game_end_date": config.TRAIN_END_DATE,
                    "ai_strategy_gene": json.dumps(best_gene),
                    "ai_final_portfolio_value": metrics.get('total_return', 0) + 1.0,
                    "strategy_rank": rank + 1,
                    "strategy_details": format_ga_gene_parameters_to_text(best_gene),
                    "period_return_pct": metrics.get('total_return', 0) * 100,
                    "max_drawdown_pct": metrics.get('max_drawdown', 0) * 100,
                    "win_rate_pct": metrics.get('win_rate_pct', 0.0),
                    "total_trades": metrics.get('trade_count', 0),
                    "profit_factor": metrics.get('profit_factor', 0.0),
                    "sharpe_ratio": metrics.get('sharpe_ratio', 0.0),
                    "average_trade_return_pct": metrics.get('average_trade_return', 0.0) * 100,  
                    "max_trade_drop_pct": metrics.get('max_trade_drop_pct', 0.0),
                    "max_trade_gain_pct": metrics.get('max_trade_gain_pct', 0.0) 
                }

                # 📝 執行資料庫插入/更新
                insert_query = """
                INSERT INTO ai_vs_user_games (
                    user_id, market_type, stock_ticker, game_start_date, game_end_date,
                    ai_strategy_gene, ai_final_portfolio_value, strategy_rank, strategy_details,
                    period_return_pct, max_drawdown_pct, win_rate_pct, total_trades,
                    profit_factor, sharpe_ratio, average_trade_return_pct, 
                    max_trade_drop_pct, max_trade_gain_pct, created_at
                ) VALUES (
                    %(user_id)s, %(market_type)s, %(stock_ticker)s, %(game_start_date)s, %(game_end_date)s,
                    %(ai_strategy_gene)s, %(ai_final_portfolio_value)s, %(strategy_rank)s, %(strategy_details)s,
                    %(period_return_pct)s, %(max_drawdown_pct)s, %(win_rate_pct)s, %(total_trades)s,
                    %(profit_factor)s, %(sharpe_ratio)s, %(average_trade_return_pct)s,
                    %(max_trade_drop_pct)s, %(max_trade_gain_pct)s, NOW()
                ) ON DUPLICATE KEY UPDATE
                    game_start_date = VALUES(game_start_date),
                    game_end_date = VALUES(game_end_date),
                    ai_strategy_gene = VALUES(ai_strategy_gene),
                    ai_final_portfolio_value = VALUES(ai_final_portfolio_value),
                    strategy_details = VALUES(strategy_details),
                    period_return_pct = VALUES(period_return_pct),
                    max_drawdown_pct = VALUES(max_drawdown_pct),
                    win_rate_pct = VALUES(win_rate_pct),
                    total_trades = VALUES(total_trades),
                    profit_factor = VALUES(profit_factor),
                    sharpe_ratio = VALUES(sharpe_ratio),
                    average_trade_return_pct = VALUES(average_trade_return_pct),
                    max_trade_drop_pct = VALUES(max_trade_drop_pct),
                    max_trade_gain_pct = VALUES(max_trade_gain_pct),
                    updated_at = NOW()
                """

                cursor.execute(insert_query, game_data)

            conn.commit()
            return True

    except Exception as e_db:
        print(f"❌ 資料庫保存錯誤: {e_db}")
        traceback.print_exc()
        if conn:
            conn.rollback()
        return False

    finally:
        if conn:
            conn.close()
def analyze_training_scope(config):
  
    print(f"\n{'='*80}")
    print(f"📊 系統A訓練範圍分析")
    print(f"{'='*80}")
    
    total_stocks = 0
    available_files = []
    
    for market_name, file_path in config.STOCK_LIST_FILES.items():
        if not config.MARKETS_TO_TRAIN.get(market_name, False):
            print(f"⏭️  跳過 {market_name}: 已停用")
            continue
            
        if not os.path.exists(file_path):
            print(f"❌ {market_name}: 檔案不存在 - {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            symbol_col = next((col for col in ['Symbol', 'symbol', 'Ticker', 'ticker', '股票代號', 'Code', 'code']
                              if col in df.columns), None)
            
            if symbol_col:
                count = len(df[symbol_col].dropna())
                total_stocks += count
                available_files.append((market_name, file_path, count))
                print(f"✅ {market_name}: {count} 支股票 - {file_path}")
            else:
                print(f"⚠️  {market_name}: 找不到股票代號欄位 - {file_path}")
                
        except Exception as e:
            print(f"❌ {market_name}: 讀取錯誤 - {e}")
    
    if total_stocks == 0:
        print(f"❌ 沒有可訓練的股票!")
        return False
    

    avg_time_per_ga_run = 8  # 秒
    total_ga_runs = total_stocks * config.NUM_GA_RUNS_PER_STOCK
    estimated_time_hours = (total_ga_runs * avg_time_per_ga_run) / 3600
    
    # 暫停時間計算
    num_pauses = total_stocks // config.STOCKS_PER_BATCH
    pause_time_hours = (num_pauses * config.PAUSE_DURATION_MINUTES) / 60
    
    total_estimated_hours = estimated_time_hours + pause_time_hours
    print(f" 總股票數: {total_stocks}") 
    print(f"{'='*80}")
    return True

if __name__ == "__main__":
    # 📋 載入配置
    config = TrainingConfig()
    
    print(f"\n🚀 AI遺傳演算法離線訓練器 - 系統A v5.2 啟動")
    print(f"⚙️  優化引擎: {'NSGA-II 多目標優化' if config.ENABLE_NSGA2 else '傳統遺傳算法'}")
    print(f"🧬 基因系統: 28基因多策略系統A")
    print(f"🆕 新功能: 單次交易最大跌幅/漲幅分析")
    
   
    if config.ENABLE_NSGA2 and not NSGA2_AVAILABLE:
        print("❌ NSGA-II 已啟用但未安裝 'pymoo' 套件")
        print("💡 解決方案: pip install pymoo")
        print("🔄 自動切換為傳統 GA 模式...")
        config.ENABLE_NSGA2 = False

  
    if not analyze_training_scope(config):
        print("❌ 訓練範圍分析失敗，程式結束")
        sys.exit(1)


    print(f"\n⚠️  即將開始大規模系統A GA訓練，請確認配置無誤")
    print(f"💡 如需修改參數，請編輯 TrainingConfig 類別")
    
 
    markets_to_process = [
        ('TAIEX', 'TW'),   # 台股
        ('NASDAQ', 'US'),  # NASDAQ 100  
        ('SP100', 'US'),   # S&P 100
    ]
    
    start_time = time.time()
    
    for market_name, market_code in markets_to_process:
        if not config.MARKETS_TO_TRAIN.get(market_name, False):
            print(f"\n⏭️  跳過 {market_name} 市場 (已停用)")
            continue
            
        file_path = config.STOCK_LIST_FILES.get(market_name)
        if not file_path or not os.path.exists(file_path):
            print(f"\n❌ 跳過 {market_name} 市場 (檔案不存在)")
            continue
        
        print(f"\n開始訓練 {market_name} 市場 (系統A)")
        run_offline_training(file_path, market_code, config)
    
    print(f" 所有系統A訓練任務完成!")
 
