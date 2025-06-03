# ga_engine.py
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import numba
import random
import logging
import time
import traceback 
import re 

# --- Settings ---
logger = logging.getLogger("GAEngine") 
if not logger.hasHandlers(): 
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- GA Configuration ---
STRATEGY_CONFIG_SHARED_GA = {
    'rsi_period_options': [6, 12, 21], 
    'vix_ma_period_options': [1, 2, 5, 10, 20],
    'bb_length_options': [10, 15, 20], 
    'bb_std_options': [1.5, 2.0], 
    'adx_period': 14,
    'ma_short_period': 5, 
    'ma_long_period': 10, 
    'commission_pct': 0.003, 
}

GA_PARAMS_CONFIG = {
    'generations': 20,          
    'population_size': 100,     
    'crossover_rate': 0.7, 
    'mutation_rate': 0.25,
    'elitism_size': 2,          
    'tournament_size': 3,       
    'mutation_amount_range': (-3, 3),   
    'vix_mutation_amount_range': (-2, 2), 
    'adx_mutation_amount_range': (-2, 2), 
    'show_process': False, 
    'rsi_threshold_range': (10, 45, 46, 75), 
    'vix_threshold_range': (15, 30),       
    'adx_threshold_range': (20, 40),       
    'rsi_period_options': STRATEGY_CONFIG_SHARED_GA['rsi_period_options'],
    'vix_ma_period_options': STRATEGY_CONFIG_SHARED_GA['vix_ma_period_options'],
    'bb_length_options': STRATEGY_CONFIG_SHARED_GA['bb_length_options'],
    'bb_std_options': STRATEGY_CONFIG_SHARED_GA['bb_std_options'],
    'adx_period': STRATEGY_CONFIG_SHARED_GA['adx_period'],
    'ma_short_period': STRATEGY_CONFIG_SHARED_GA['ma_short_period'],
    'ma_long_period': STRATEGY_CONFIG_SHARED_GA['ma_long_period'],
    'commission_rate': STRATEGY_CONFIG_SHARED_GA['commission_pct'],
    'offline_trainer_runs_per_stock': 60, 
    'ai_watchlist_size_config': 150, # Watchlist 大小配置
    'ondemand_train_generations': 20, # 即時訓練代數
    'ondemand_train_population': 60, # 即時訓練種群大小
    'ondemand_train_runs': 50,        # 即時訓練運行次數
}


# --- GA 核心函數定義 ---

def ga_load_stock_data(ticker, vix_ticker="^VIX", start_date=None, end_date=None, verbose=False):
    if verbose: logger.info(f"Attempting to load data for {ticker} and {vix_ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download([ticker, vix_ticker], start=start_date, end=end_date, progress=False, auto_adjust=False, timeout=30) 
        if data is None or data.empty:
            if verbose: logger.warning(f"Error: yfinance.download returned None or empty DataFrame for {ticker}.")
            return None, None, None, None
        if not isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns: 
                logger.warning(f"Warning: VIX data for {vix_ticker} might be missing for {ticker}. Proceeding with stock data only.")
                stock_data = data.copy()
                vix_data = pd.Series(np.nan, index=stock_data.index)
                vix_data.name = 'VIX_Close'
            else:
                if verbose: logger.warning(f"Error: Expected MultiIndex columns for {ticker}, got {data.columns}. Check tickers.")
                return None, None, None, None
        else: 
            stock_columns_present = ticker in data.columns.get_level_values(1)
            vix_columns_present = vix_ticker in data.columns.get_level_values(1)

            if not stock_columns_present:
                if verbose: logger.error(f"Error: Missing data for stock {ticker}.")
                return None, None, None, None

            stock_data = data.loc[:, pd.IndexSlice[:, ticker]]
            stock_data.columns = stock_data.columns.droplevel(1) # 已修正
            
            if not vix_columns_present:
                if verbose: logger.warning(f"Warning: Missing VIX data for {vix_ticker} (for {ticker} analysis), will use NaNs.")
                vix_data = pd.Series(np.nan, index=stock_data.index) 
                vix_data.name = 'VIX_Close'
            else:
                vix_data_slice = data.loc[:, pd.IndexSlice['Close', vix_ticker]]
                if isinstance(vix_data_slice, pd.DataFrame):
                    if vix_data_slice.shape[1] == 1: vix_data = vix_data_slice.iloc[:, 0]
                    else: 
                        if verbose: logger.error(f"Error: Could not extract VIX Close as a Series (shape: {vix_data_slice.shape}) for {ticker}.")
                        return None, None, None, None
                elif isinstance(vix_data_slice, pd.Series): vix_data = vix_data_slice
                else:
                    if verbose: logger.error(f"Error: Unrecognized VIX data type: {type(vix_data_slice)} for {ticker}.")
                    return None, None, None, None
                vix_data.name = 'VIX_Close'
            
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        missing_essential_cols = [col for col in ['Close', 'High', 'Low'] if col not in stock_data.columns] 
        if missing_essential_cols:
            if verbose: logger.error(f"Error: Missing essential columns {missing_essential_cols} for {ticker}.")
            return None, None, None, None
        for col in required_cols: 
             if col not in stock_data.columns:
                 stock_data[col] = np.nan 
                 if verbose: logger.warning(f"Warning: Stock data for {ticker} missing '{col}', filled with NaN.")

        simplified_df = stock_data[required_cols].copy()
        
        if 'VIX_Close' not in simplified_df.columns: 
             vix_data = vix_data.reindex(simplified_df.index)
        elif 'VIX_Close' in simplified_df.columns: 
            vix_data = simplified_df['VIX_Close']
            simplified_df = simplified_df.drop(columns=['VIX_Close'], errors='ignore')

        aligned_data = pd.concat([simplified_df, vix_data], axis=1, join='inner')

        if aligned_data.empty:
            if verbose: logger.error(f"Error: No common dates found between {ticker} and {vix_ticker} after join.")
            return None, None, None, None

        if aligned_data.isnull().values.any():
            if verbose: logger.warning(f"Warning: NaNs found after alignment for {ticker}. Filling...")
            if 'VIX_Close' in aligned_data.columns and aligned_data['VIX_Close'].isnull().any():
                aligned_data['VIX_Close'] = aligned_data['VIX_Close'].ffill().bfill() 
            
            numeric_stock_cols = [col for col in required_cols if col in aligned_data.columns and pd.api.types.is_numeric_dtype(aligned_data[col])]
            for col in numeric_stock_cols:
                if aligned_data[col].isnull().any():
                     aligned_data[col] = aligned_data[col].ffill().bfill()

            if aligned_data.isnull().values.any():
                nan_cols_still = aligned_data.columns[aligned_data.isnull().any()].tolist()
                if any(col in nan_cols_still for col in ['Close', 'High', 'Low']):
                    if verbose: logger.error(f"Error: Could not fill NaNs for essential HLC columns in {ticker}: {nan_cols_still}.")
                    return None, None, None, None
                logger.warning(f"Proceeding with some NaNs in non-essential columns for {ticker}: {nan_cols_still}")

        final_stock_df = aligned_data[required_cols].copy()
        final_vix_series = aligned_data['VIX_Close'].copy() if 'VIX_Close' in aligned_data else pd.Series(np.nan, index=final_stock_df.index)

        prices = final_stock_df['Close'].tolist()
        dates = final_stock_df.index.tolist() 

        logger.info(f"Successfully loaded and aligned {len(prices)} data points for {ticker} from {start_date} to {end_date}.")
        return prices, dates, final_stock_df, final_vix_series
    except Exception as e:
        logger.error(f"Error during data loading for {ticker}: {type(e).__name__}: {e}", exc_info=True)
        return None, None, None, None

def ga_precompute_indicators(stock_df, vix_series, strategy_config, verbose=False):
    precalculated_rsi = {}
    precalculated_vix_ma = {}
    precalculated_bbl = {}
    precalculated_bbm = {}
    precalculated_fixed = {} 
    if verbose: logger.info(f"  Starting indicator pre-calculation (Stock DF shape: {stock_df.shape}, VIX Series len: {len(vix_series if vix_series is not None else [])})...")

    try:
        if stock_df.empty or not all(col in stock_df.columns for col in ['Close', 'High', 'Low']):
            logger.error("    Error: Stock DataFrame is empty or missing essential HLC columns for precomputation.")
            return {}, False
        
        close_prices = stock_df['Close']
        if close_prices.isnull().all(): 
            logger.error("    Error: All 'Close' prices are NaN. Cannot compute indicators.")
            return {}, False

        for rsi_p in strategy_config['rsi_period_options']:
            if len(close_prices.dropna()) < rsi_p : 
                logger.warning(f"    Warning: Not enough data for RSI period {rsi_p}. Filling with NaNs."); 
                precalculated_rsi[rsi_p] = [np.nan] * len(close_prices); continue
            rsi_values = ta.rsi(close_prices, length=rsi_p)
            precalculated_rsi[rsi_p] = rsi_values.tolist() if rsi_values is not None else [np.nan] * len(close_prices)
        if verbose: logger.info(f"    RSI ({len(precalculated_rsi)} variants) processed.")

        for vix_ma_p in strategy_config['vix_ma_period_options']:
            if vix_series is None or vix_series.empty or len(vix_series.dropna()) < vix_ma_p: 
                logger.warning(f"    Warning: Not enough data for VIX MA period {vix_ma_p} or VIX series is invalid. Filling with NaNs.")
                precalculated_vix_ma[vix_ma_p] = [np.nan] * len(vix_series if vix_series is not None and not vix_series.empty else close_prices); continue 
            vix_ma_values = vix_series.rolling(window=vix_ma_p).mean()
            precalculated_vix_ma[vix_ma_p] = vix_ma_values.tolist() if vix_ma_values is not None else [np.nan] * len(vix_series)
        if verbose: logger.info(f"    VIX MA ({len(precalculated_vix_ma)} variants) processed.")
        
        for bb_l in strategy_config['bb_length_options']:
            for bb_s in strategy_config['bb_std_options']:
                key = (bb_l, bb_s)
                if len(close_prices.dropna()) < bb_l:
                    logger.warning(f"    Warning: Not enough data for BB L={bb_l}. Filling with NaNs."); 
                    precalculated_bbl[key] = [np.nan] * len(close_prices)
                    precalculated_bbm[key] = [np.nan] * len(close_prices)
                    continue
                bbands = ta.bbands(close_prices, length=bb_l, std=bb_s)
                bbl_col_name = next((col for col in bbands.columns if 'BBL' in col), None) 
                bbm_col_name = next((col for col in bbands.columns if 'BBM' in col), None)
                if bbands is None or not bbl_col_name or not bbm_col_name: 
                    precalculated_bbl[key] = [np.nan] * len(close_prices)
                    precalculated_bbm[key] = [np.nan] * len(close_prices)
                else:
                    precalculated_bbl[key] = bbands[bbl_col_name].tolist()
                    precalculated_bbm[key] = bbands[bbm_col_name].tolist()
        if verbose: logger.info(f"    Bollinger Bands (BBL/BBM, {len(precalculated_bbl)} variants) processed.")

        if len(stock_df['High'].dropna()) < strategy_config['adx_period'] or \
           len(stock_df['Low'].dropna()) < strategy_config['adx_period'] or \
           len(close_prices.dropna()) < strategy_config['adx_period']:
            logger.warning(f"    Warning: Not enough HLC data for ADX period {strategy_config['adx_period']}. ADX will be NaN.")
            precalculated_fixed['adx_list'] = [np.nan] * len(close_prices)
        else:
            adx_df = ta.adx(stock_df['High'], stock_df['Low'], close_prices, length=strategy_config['adx_period'])
            adx_col_name = next((col for col in adx_df.columns if 'ADX' in col), None) 
            precalculated_fixed['adx_list'] = adx_df[adx_col_name].tolist() if adx_df is not None and adx_col_name else [np.nan] * len(close_prices)
        if verbose: logger.info("    ADX processed.")
        
        if len(close_prices.dropna()) < strategy_config['ma_short_period']:
            precalculated_fixed['ma_short_list'] = [np.nan] * len(close_prices)
        else:
            ma_short = close_prices.rolling(window=strategy_config['ma_short_period']).mean()
            precalculated_fixed['ma_short_list'] = ma_short.tolist() if ma_short is not None else [np.nan] * len(close_prices)

        if len(close_prices.dropna()) < strategy_config['ma_long_period']:
            precalculated_fixed['ma_long_list'] = [np.nan] * len(close_prices)
        else:
            ma_long = close_prices.rolling(window=strategy_config['ma_long_period']).mean()
            precalculated_fixed['ma_long_list'] = ma_long.tolist() if ma_long is not None else [np.nan] * len(close_prices)
        
        if verbose : logger.info("    MAs processed.")
        if verbose: logger.info("  All required indicator pre-calculations finished.")
        return {'rsi': precalculated_rsi, 'vix_ma': precalculated_vix_ma,
                'bbl': precalculated_bbl, 'bbm': precalculated_bbm, 
                'fixed': precalculated_fixed}, True
    except Exception as e:
        if verbose: logger.error(f"  Unexpected error during indicator pre-calculation: {type(e).__name__}: {e}", exc_info=True)
        return {}, False

@numba.jit(nopython=True)
def run_strategy_numba_core(rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold, vix_threshold,
                            low_vol_exit_strategy, high_vol_entry_choice,
                            commission_rate,
                            prices_arr,
                            rsi_arr, bbl_arr, bbm_arr, adx_arr, vix_ma_arr, ma_short_arr, ma_long_arr,
                            start_trading_iloc):
    T = len(prices_arr); portfolio_values_arr = np.full(T, 1.0, dtype=np.float64)
    max_signals = T // 2 + 1 
    buy_signal_indices = np.full(max_signals, -1, dtype=np.int64); buy_signal_prices = np.full(max_signals, np.nan, dtype=np.float64); buy_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    sell_signal_indices = np.full(max_signals, -1, dtype=np.int64); sell_signal_prices = np.full(max_signals, np.nan, dtype=np.float64); sell_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    buy_count = 0; sell_count = 0; cash = 1.0; stock = 0.0; position = 0; last_valid_portfolio_value = 1.0
    rsi_crossed_exit_level_after_buy = False 
    high_vol_entry_type = -1 

    start_trading_iloc = max(1, start_trading_iloc) 
    if start_trading_iloc >= T: 
        portfolio_values_arr[:] = 1.0 
        return portfolio_values_arr, buy_signal_indices[:0], buy_signal_prices[:0], buy_signal_rsis[:0], \
               sell_signal_indices[:0], sell_signal_prices[:0], sell_signal_rsis[:0]
    portfolio_values_arr[:start_trading_iloc] = 1.0 

    for i in range(start_trading_iloc, T):
        current_price = prices_arr[i]; rsi_i, rsi_prev = rsi_arr[i], rsi_arr[i-1]; bbl_i = bbl_arr[i]; bbm_i = bbm_arr[i]; adx_i = adx_arr[i]; vix_ma_i = vix_ma_arr[i]
        ma_short_i, ma_long_i = ma_short_arr[i], ma_long_arr[i]; ma_short_prev, ma_long_prev = ma_short_arr[i-1], ma_long_arr[i-1]

        required_values = (rsi_i, rsi_prev, current_price, bbl_i, bbm_i, adx_i, vix_ma_i, ma_short_i, ma_long_i, ma_short_prev, ma_long_prev)
        is_valid_point = True
        for val_idx_loop in range(len(required_values)): 
            if not np.isfinite(required_values[val_idx_loop]): is_valid_point = False; break
        
        if not is_valid_point: 
            current_portfolio_value_check_loop = cash if position == 0 else (stock * current_price if np.isfinite(current_price) else np.nan)
            portfolio_values_arr[i] = last_valid_portfolio_value if np.isnan(current_portfolio_value_check_loop) else current_portfolio_value_check_loop
            if not np.isnan(current_portfolio_value_check_loop): last_valid_portfolio_value = current_portfolio_value_check_loop
            continue

        if position == 0: 
            is_high_vol_market = vix_ma_i >= vix_threshold
            buy_triggered = False; entry_type_for_this_buy = -1 
            if is_high_vol_market: 
                if high_vol_entry_choice == 0: 
                    if (current_price <= bbl_i) and (rsi_i < rsi_buy_entry_threshold):
                        buy_triggered = True; entry_type_for_this_buy = 0
                else: 
                    if (current_price <= bbl_i) and (adx_i > adx_threshold):
                        buy_triggered = True; entry_type_for_this_buy = 1
            else: 
                if (ma_short_prev < ma_long_prev) and (ma_short_i >= ma_long_i): 
                    buy_triggered = True; entry_type_for_this_buy = -1 
            
            if buy_triggered and current_price > 1e-9: 
                cost_of_commission = cash * commission_rate; investment_amount = cash - cost_of_commission
                if investment_amount > 0: 
                    stock = investment_amount / current_price; cash = 0.0; position = 1
                    rsi_crossed_exit_level_after_buy = False 
                    high_vol_entry_type = entry_type_for_this_buy 
                    if buy_count < max_signals: buy_signal_indices[buy_count] = i; buy_signal_prices[buy_count] = current_price; buy_signal_rsis[buy_count] = rsi_i; buy_count += 1
        
        elif position == 1: 
            sell_triggered = False
            if high_vol_entry_type == 0: 
                if rsi_i >= rsi_exit_threshold: rsi_crossed_exit_level_after_buy = True 
                if rsi_crossed_exit_level_after_buy and rsi_i < rsi_exit_threshold: sell_triggered = True 
            elif high_vol_entry_type == 1: 
                if current_price >= bbm_i: sell_triggered = True 
            elif high_vol_entry_type == -1: 
                if low_vol_exit_strategy == 0: 
                    if current_price < ma_short_i: sell_triggered = True
                else: 
                    if (ma_short_prev > ma_long_prev) and (ma_short_i <= ma_long_i): sell_triggered = True
            
            if sell_triggered:
                proceeds_from_sale = stock * current_price; cost_of_commission_sell = proceeds_from_sale * commission_rate; cash = proceeds_from_sale - cost_of_commission_sell
                stock = 0.0; position = 0
                rsi_crossed_exit_level_after_buy = False; high_vol_entry_type = -1 
                if sell_count < max_signals: sell_signal_indices[sell_count] = i; sell_signal_prices[sell_count] = current_price; sell_signal_rsis[sell_count] = rsi_i; sell_count += 1
        
        current_stock_value_calc = stock * current_price if position == 1 else 0.0
        current_portfolio_value_update_loop = cash + current_stock_value_calc
        portfolio_values_arr[i] = last_valid_portfolio_value if np.isnan(current_portfolio_value_update_loop) else current_portfolio_value_update_loop
        if not np.isnan(current_portfolio_value_update_loop): last_valid_portfolio_value = current_portfolio_value_update_loop
    
    if T > 0 and np.isnan(portfolio_values_arr[-1]): portfolio_values_arr[-1] = last_valid_portfolio_value 
    return portfolio_values_arr, buy_signal_indices[:buy_count], buy_signal_prices[:buy_count], buy_signal_rsis[:buy_count], sell_signal_indices[:sell_count], sell_signal_prices[:sell_count], sell_signal_rsis[:sell_count]

def run_strategy(rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold, vix_threshold, low_vol_exit_strategy, high_vol_entry_choice,
                 commission_rate,
                 prices, dates, 
                 rsi_list, bbl_list, bbm_list, adx_list, vix_ma_list, ma_short_list, ma_long_list):
    T = len(prices)
    if T == 0: return [1.0], [], [] 

    prices_arr = np.array(prices, dtype=np.float64); rsi_arr = np.array(rsi_list, dtype=np.float64)
    bbl_arr = np.array(bbl_list, dtype=np.float64); bbm_arr = np.array(bbm_list, dtype=np.float64)
    adx_arr = np.array(adx_list, dtype=np.float64); vix_ma_arr = np.array(vix_ma_list, dtype=np.float64)
    ma_short_arr = np.array(ma_short_list, dtype=np.float64); ma_long_arr = np.array(ma_long_list, dtype=np.float64)

    def get_first_valid_iloc_local(indicator_arr_func_scope): 
        valid_indices = np.where(np.isfinite(indicator_arr_func_scope))[0]
        return valid_indices[0] if len(valid_indices) > 0 else T 
    
    start_iloc_rsi = get_first_valid_iloc_local(rsi_arr) if len(rsi_arr) > 0 else T
    start_iloc_bbl = get_first_valid_iloc_local(bbl_arr) if len(bbl_arr) > 0 else T
    start_iloc_bbm = get_first_valid_iloc_local(bbm_arr) if len(bbm_arr) > 0 else T
    start_iloc_adx = get_first_valid_iloc_local(adx_arr) if len(adx_arr) > 0 else T
    start_iloc_vix_ma = get_first_valid_iloc_local(vix_ma_arr) if len(vix_ma_arr) > 0 else T
    start_iloc_ma_short = get_first_valid_iloc_local(ma_short_arr) if len(ma_short_arr) > 0 else T
    start_iloc_ma_long = get_first_valid_iloc_local(ma_long_arr) if len(ma_long_arr) > 0 else T
    
    start_trading_iloc_calc = max(start_iloc_rsi, start_iloc_bbl, start_iloc_bbm, start_iloc_adx, start_iloc_vix_ma, start_iloc_ma_short, start_iloc_ma_long)
    start_trading_iloc_calc += 1 

    if start_trading_iloc_calc >= T: 
        return [1.0] * T, [], [] 

    start_trading_iloc_final = max(start_trading_iloc_calc, 1) 

    portfolio_values_arr, buy_indices, buy_prices, buy_rsis, sell_indices, sell_prices, sell_rsis = \
        run_strategy_numba_core(
            float(rsi_buy_entry_threshold), float(rsi_exit_threshold), float(adx_threshold), float(vix_threshold),
            int(low_vol_exit_strategy), int(high_vol_entry_choice), 
            float(commission_rate),
            prices_arr, rsi_arr, bbl_arr, bbm_arr, adx_arr, vix_ma_arr, ma_short_arr, ma_long_arr, 
            start_trading_iloc_final
        )

    buy_signals = []; sell_signals = []
    for idx, price_val, rsi_val_s in zip(buy_indices, buy_prices, buy_rsis): 
        if idx != -1 and idx < len(dates): buy_signals.append((dates[idx], price_val, rsi_val_s))
    for idx, price_val, rsi_val_s in zip(sell_indices, sell_prices, sell_rsis):
         if idx != -1 and idx < len(dates): sell_signals.append((dates[idx], price_val, rsi_val_s))
    return portfolio_values_arr.tolist(), buy_signals, sell_signals

def genetic_algorithm_with_elitism(prices, dates,
                                   precalculated_rsi_lists,   
                                   precalculated_vix_ma_lists, 
                                   precalculated_bbl_lists,   
                                   precalculated_bbm_lists,   
                                   adx_list,                  
                                   ma_short_list, ma_long_list, 
                                   ga_params):                
    generations = ga_params['generations']; population_size = ga_params['population_size']; crossover_rate = ga_params['crossover_rate']
    mutation_rate = ga_params['mutation_rate']; elitism_size = ga_params['elitism_size']; tournament_size = ga_params['tournament_size']
    mutation_amount_range = ga_params['mutation_amount_range']
    show_ga_generations = ga_params.get('show_process', False) 

    rsi_threshold_range_cfg = ga_params['rsi_threshold_range']; vix_threshold_range_cfg = ga_params['vix_threshold_range']
    adx_threshold_range_cfg = ga_params['adx_threshold_range']
    rsi_period_options_cfg = ga_params['rsi_period_options']; num_rsi_options = len(rsi_period_options_cfg)
    vix_ma_period_options_cfg = ga_params['vix_ma_period_options']; num_vix_ma_options = len(vix_ma_period_options_cfg)
    bb_length_options_cfg = ga_params['bb_length_options']; num_bb_len_options = len(bb_length_options_cfg)
    bb_std_options_cfg = ga_params['bb_std_options']; num_bb_std_options = len(bb_std_options_cfg)
    commission_rate_cfg = ga_params['commission_rate']
    vix_mutation_amount_range_cfg = ga_params.get('vix_mutation_amount_range', mutation_amount_range) 
    adx_mutation_amount_range_cfg = ga_params.get('adx_mutation_amount_range', mutation_amount_range) 

    T = len(prices)
    if T < 2: 
        if show_ga_generations: logger.error(f"Error: Data length {T} is too short for GA.")
        return None, 0

    population = []; attempts, max_attempts = 0, population_size * 200 
    min_buy_cfg, max_buy_cfg, min_exit_cfg, max_exit_cfg = rsi_threshold_range_cfg
    min_vix_cfg, max_vix_cfg = vix_threshold_range_cfg
    min_adx_cfg, max_adx_cfg = adx_threshold_range_cfg
    
    while len(population) < population_size and attempts < max_attempts:
        buy_entry_thr_g = random.randint(min_buy_cfg, max_buy_cfg)
        exit_thr_g = random.randint(max(buy_entry_thr_g + 1, min_exit_cfg), max_exit_cfg) 
        vix_thr_g = random.randint(min_vix_cfg, max_vix_cfg)
        low_vol_exit_g = random.choice([0, 1]) 
        rsi_p_choice_g = random.randint(0, num_rsi_options - 1) 
        vix_ma_p_choice_g = random.randint(0, num_vix_ma_options - 1) 
        bb_len_choice_g = random.randint(0, num_bb_len_options - 1) 
        bb_std_choice_g = random.randint(0, num_bb_std_options - 1) 
        adx_thr_g = random.randint(min_adx_cfg, max_adx_cfg) 
        hv_entry_choice_g = random.choice([0, 1]) 
        
        gene_g = [buy_entry_thr_g, exit_thr_g, vix_thr_g, low_vol_exit_g, rsi_p_choice_g, vix_ma_p_choice_g, bb_len_choice_g, bb_std_choice_g, adx_thr_g, hv_entry_choice_g]
        if (0<gene_g[0]<gene_g[1]<100 and min_buy_cfg<=gene_g[0]<=max_buy_cfg and min_exit_cfg<=gene_g[1]<=max_exit_cfg and 
            min_vix_cfg<=gene_g[2]<=max_vix_cfg and gene_g[3] in [0,1] and 
            0<=gene_g[4]<num_rsi_options and 0<=gene_g[5]<num_vix_ma_options and 
            0<=gene_g[6]<num_bb_len_options and 0<=gene_g[7]<num_bb_std_options and 
            min_adx_cfg<=gene_g[8]<=max_adx_cfg and gene_g[9] in [0,1]):
             population.append(gene_g)
        attempts += 1
    if not population or len(population) < population_size : 
        if show_ga_generations: logger.error(f"Error: Could not generate sufficient initial population ({len(population)}/{population_size}).")
        return None, 0
    
    best_gene_overall_val = population[0][:]; best_fitness_overall_val = -float('inf') 

    for generation in range(generations):
        fitness_scores_list = [] 
        for gene_idx_loop, current_gene_loop in enumerate(population):
            try:
                chosen_rsi_period_loop = rsi_period_options_cfg[current_gene_loop[4]]
                chosen_vix_ma_period_loop = vix_ma_period_options_cfg[current_gene_loop[5]]
                chosen_bb_length_loop = bb_length_options_cfg[current_gene_loop[6]]
                chosen_bb_std_loop = bb_std_options_cfg[current_gene_loop[7]]
                
                current_rsi_list_loop = precalculated_rsi_lists[chosen_rsi_period_loop]
                current_vix_ma_list_loop = precalculated_vix_ma_lists[chosen_vix_ma_period_loop]
                current_bbl_list_loop = precalculated_bbl_lists[(chosen_bb_length_loop, chosen_bb_std_loop)]
                current_bbm_list_loop = precalculated_bbm_lists[(chosen_bb_length_loop, chosen_bb_std_loop)]
                
                portfolio_values_loop, _, _ = run_strategy( 
                    current_gene_loop[0], current_gene_loop[1], current_gene_loop[8], current_gene_loop[2], current_gene_loop[3], current_gene_loop[9],
                    commission_rate_cfg, prices, dates,
                    current_rsi_list_loop, current_bbl_list_loop, current_bbm_list_loop, adx_list, current_vix_ma_list_loop, ma_short_list, ma_long_list
                )
                final_value_loop = next((p_val_loop for p_val_loop in reversed(portfolio_values_loop) if np.isfinite(p_val_loop)), -np.inf) 
                fitness_scores_list.append(final_value_loop)
            except (IndexError, KeyError) as e_eval_loop: 
                if show_ga_generations: logger.warning(f"  Eval Error (Gene: {current_gene_loop}): {e_eval_loop}. Fitness -inf.")
                fitness_scores_list.append(-np.inf)
            except Exception as e_unexp_loop: 
                if show_ga_generations: logger.error(f"  Unexpected Eval Error (Gene: {current_gene_loop}): {e_unexp_loop}.", exc_info=True)
                fitness_scores_list.append(-np.inf)

        fitness_array_gen = np.array(fitness_scores_list)
        valid_fitness_mask_gen = np.isfinite(fitness_array_gen) & (fitness_array_gen > -np.inf) 
        valid_indices_gen = np.where(valid_fitness_mask_gen)[0]
        valid_fitness_count_gen = len(valid_indices_gen)
        
        if valid_fitness_count_gen == 0: 
             if show_ga_generations: logger.warning(f"Gen {generation+1} - Warning: All individuals invalid.")
             continue 
        
        sorted_valid_indices_gen = valid_indices_gen[np.argsort(fitness_array_gen[valid_indices_gen])[::-1]] 
        num_elites_gen = min(elitism_size, valid_fitness_count_gen) 
        elite_indices_gen = sorted_valid_indices_gen[:num_elites_gen]
        elites_gen = [population[i_gen][:] for i_gen in elite_indices_gen] 
        
        current_best_fitness_in_gen_val = fitness_array_gen[elite_indices_gen[0]] if num_elites_gen > 0 else -np.inf
        if current_best_fitness_in_gen_val > best_fitness_overall_val: 
            best_fitness_overall_val = current_best_fitness_in_gen_val
            best_gene_overall_val = population[elite_indices_gen[0]][:]

        if show_ga_generations and (generation + 1) % 1 == 0: 
            gen_best_str_log = f"{current_best_fitness_in_gen_val:.4f}" if num_elites_gen > 0 else "N/A"
            overall_best_str_log = "N/A"
            if best_fitness_overall_val > -np.inf and best_gene_overall_val:
                bo_lv_exit_str_log = "Price<MA" if best_gene_overall_val[3] == 0 else "MACross"
                bo_rsi_p_str_log = rsi_period_options_cfg[best_gene_overall_val[4]]
                bo_vix_ma_p_str_log = vix_ma_period_options_cfg[best_gene_overall_val[5]]
                bo_bb_l_str_log = bb_length_options_cfg[best_gene_overall_val[6]]
                bo_bb_s_str_log = bb_std_options_cfg[best_gene_overall_val[7]]
                bo_hv_entry_str_log = "BB+RSI" if best_gene_overall_val[9] == 0 else "BB+ADX"
                overall_best_params_readable_log = (
                    f"BuyE={best_gene_overall_val[0]},ExitR={best_gene_overall_val[1]},VIX_T={best_gene_overall_val[2]},LVExit={bo_lv_exit_str_log},"
                    f"RSI_P={bo_rsi_p_str_log},VIX_MA_P={bo_vix_ma_p_str_log},BB_L={bo_bb_l_str_log},BB_S={bo_bb_s_str_log},"
                    f"ADX_T={best_gene_overall_val[8]},HVEntry={bo_hv_entry_str_log}"
                )
                overall_best_str_log = f"{best_fitness_overall_val:.4f} (Params: {overall_best_params_readable_log})"
            logger.info(f"Gen {generation+1}/{generations} | Best(G): {gen_best_str_log} | Best(O): {overall_best_str_log} | Valid: {valid_fitness_count_gen}/{population_size}")

        selected_parents_list = []; num_parents_to_select_val = population_size - num_elites_gen
        if num_parents_to_select_val <= 0: population = elites_gen[:population_size]; continue 
        effective_tournament_size_val = min(tournament_size, valid_fitness_count_gen) 
        if effective_tournament_size_val <= 0: population = elites_gen[:population_size]; continue 

        for _ in range(num_parents_to_select_val):
             aspirant_indices_local_val = np.random.choice(len(valid_indices_gen), size=effective_tournament_size_val, replace=False)
             aspirant_indices_global_val = valid_indices_gen[aspirant_indices_local_val]
             winner_global_idx_val = aspirant_indices_global_val[np.argmax(fitness_array_gen[aspirant_indices_global_val])]
             selected_parents_list.append(population[winner_global_idx_val][:])
        
        offspring_list = []; parent_indices_list = list(range(len(selected_parents_list))); random.shuffle(parent_indices_list)
        num_pairs_val = len(parent_indices_list) // 2
        for i_pair_val in range(num_pairs_val):
            p1_val, p2_val = selected_parents_list[parent_indices_list[2*i_pair_val]], selected_parents_list[parent_indices_list[2*i_pair_val + 1]]
            child1_val, child2_val = p1_val[:], p2_val[:]
            if random.random() < crossover_rate: 
                 crossover_point_val = random.randint(1, 9); 
                 child1_new_val = p1_val[:crossover_point_val] + p2_val[crossover_point_val:]
                 child2_new_val = p2_val[:crossover_point_val] + p1_val[crossover_point_val:]
                 valid_c1_val = (0<child1_new_val[0]<child1_new_val[1]<100 and min_buy_cfg<=child1_new_val[0]<=max_buy_cfg and min_exit_cfg<=child1_new_val[1]<=max_exit_cfg and min_vix_cfg<=child1_new_val[2]<=max_vix_cfg and child1_new_val[3] in [0,1] and 0<=child1_new_val[4]<num_rsi_options and 0<=child1_new_val[5]<num_vix_ma_options and 0<=child1_new_val[6]<num_bb_len_options and 0<=child1_new_val[7]<num_bb_std_options and min_adx_cfg<=child1_new_val[8]<=max_adx_cfg and child1_new_val[9] in [0,1])
                 valid_c2_val = (0<child2_new_val[0]<child2_new_val[1]<100 and min_buy_cfg<=child2_new_val[0]<=max_buy_cfg and min_exit_cfg<=child2_new_val[1]<=max_exit_cfg and min_vix_cfg<=child2_new_val[2]<=max_vix_cfg and child2_new_val[3] in [0,1] and 0<=child2_new_val[4]<num_rsi_options and 0<=child2_new_val[5]<num_vix_ma_options and 0<=child2_new_val[6]<num_bb_len_options and 0<=child2_new_val[7]<num_bb_std_options and min_adx_cfg<=child2_new_val[8]<=max_adx_cfg and child2_new_val[9] in [0,1])
                 child1_val = child1_new_val if valid_c1_val else p1_val[:] 
                 child2_val = child2_new_val if valid_c2_val else p2_val[:]
            offspring_list.append(child1_val); offspring_list.append(child2_val)
        if len(parent_indices_list) % 2 != 0: offspring_list.append(selected_parents_list[parent_indices_list[-1]][:]) 

        mut_min_val, mut_max_val = mutation_amount_range
        vix_mut_min_actual_val, vix_mut_max_actual_val = vix_mutation_amount_range_cfg
        adx_mut_min_actual_val, adx_mut_max_actual_val = adx_mutation_amount_range_cfg

        for i_offspring_val in range(len(offspring_list)):
            if random.random() < mutation_rate: 
                gene_to_mutate_val = offspring_list[i_offspring_val]; original_gene_val = gene_to_mutate_val[:]; mutate_idx_val = random.randint(0, 9) 
                if mutate_idx_val == 3: gene_to_mutate_val[3] = 1 - gene_to_mutate_val[3] 
                elif mutate_idx_val == 4: 
                    if num_rsi_options > 1:
                        new_choice_val = random.randint(0, num_rsi_options - 1)
                        while new_choice_val == original_gene_val[4]: new_choice_val = random.randint(0, num_rsi_options - 1) 
                        gene_to_mutate_val[4] = new_choice_val
                elif mutate_idx_val == 5: 
                    if num_vix_ma_options > 1:
                        new_choice_val = random.randint(0, num_vix_ma_options - 1)
                        while new_choice_val == original_gene_val[5]: new_choice_val = random.randint(0, num_vix_ma_options - 1)
                        gene_to_mutate_val[5] = new_choice_val
                elif mutate_idx_val == 6: 
                    if num_bb_len_options > 1:
                        new_choice_val = random.randint(0, num_bb_len_options - 1)
                        while new_choice_val == original_gene_val[6]: new_choice_val = random.randint(0, num_bb_len_options - 1)
                        gene_to_mutate_val[6] = new_choice_val
                elif mutate_idx_val == 7: 
                     if num_bb_std_options > 1:
                        new_choice_val = random.randint(0, num_bb_std_options - 1)
                        while new_choice_val == original_gene_val[7]: new_choice_val = random.randint(0, num_bb_std_options - 1)
                        gene_to_mutate_val[7] = new_choice_val
                elif mutate_idx_val == 9: gene_to_mutate_val[9] = 1 - gene_to_mutate_val[9] 
                else: 
                    if mutate_idx_val == 2: mutation_amount_val = random.randint(vix_mut_min_actual_val, vix_mut_max_actual_val); is_zero_range_val = (vix_mut_min_actual_val == 0 and vix_mut_max_actual_val == 0)
                    elif mutate_idx_val == 8: mutation_amount_val = random.randint(adx_mut_min_actual_val, adx_mut_max_actual_val); is_zero_range_val = (adx_mut_min_actual_val == 0 and adx_mut_max_actual_val == 0)
                    else: mutation_amount_val = random.randint(mut_min_val, mut_max_val); is_zero_range_val = (mut_min_val == 0 and mut_max_val == 0)
                    
                    if mutation_amount_val == 0 and not is_zero_range_val: 
                        while mutation_amount_val == 0:
                            if mutate_idx_val == 2: mutation_amount_val = random.randint(vix_mut_min_actual_val, vix_mut_max_actual_val)
                            elif mutate_idx_val == 8: mutation_amount_val = random.randint(adx_mut_min_actual_val, adx_mut_max_actual_val)
                            else: mutation_amount_val = random.randint(mut_min_val, mut_max_val)
                    
                    gene_to_mutate_val[mutate_idx_val] += mutation_amount_val
                    gene_to_mutate_val[0] = max(min_buy_cfg, min(gene_to_mutate_val[0], max_buy_cfg))
                    gene_to_mutate_val[1] = max(gene_to_mutate_val[0] + 1, min_exit_cfg, min(gene_to_mutate_val[1], max_exit_cfg)) 
                    gene_to_mutate_val[0] = max(min_buy_cfg, min(gene_to_mutate_val[0], gene_to_mutate_val[1] - 1, max_buy_cfg)) 
                    gene_to_mutate_val[2] = max(min_vix_cfg, min(gene_to_mutate_val[2], max_vix_cfg))
                    gene_to_mutate_val[8] = max(min_adx_cfg, min(gene_to_mutate_val[8], max_adx_cfg))
                
                final_valid_val = (0<gene_to_mutate_val[0]<gene_to_mutate_val[1]<100 and min_buy_cfg<=gene_to_mutate_val[0]<=max_buy_cfg and min_exit_cfg<=gene_to_mutate_val[1]<=max_exit_cfg and min_vix_cfg<=gene_to_mutate_val[2]<=max_vix_cfg and gene_to_mutate_val[3] in [0,1] and 0<=gene_to_mutate_val[4]<num_rsi_options and 0<=gene_to_mutate_val[5]<num_vix_ma_options and 0<=gene_to_mutate_val[6]<num_bb_len_options and 0<=gene_to_mutate_val[7]<num_bb_std_options and min_adx_cfg<=gene_to_mutate_val[8]<=max_adx_cfg and gene_to_mutate_val[9] in [0,1])
                if not final_valid_val: offspring_list[i_offspring_val] = original_gene_val 
        
        population = elites_gen + offspring_list; population = population[:population_size] 

    if best_fitness_overall_val == -float('inf'): 
        if show_ga_generations: logger.error("Error: GA finished without finding any valid solution.")
        return None, 0
    return best_gene_overall_val, best_fitness_overall_val


def check_ga_buy_signal_at_latest_point(
    rsi_buy_entry_threshold, 
    adx_threshold_gene, 
    vix_threshold_gene,
    high_vol_entry_choice_gene,
    current_price_latest,
    rsi_latest, rsi_prev,
    bbl_latest, 
    adx_latest,
    vix_ma_latest,
    ma_short_latest, ma_long_latest,
    ma_short_prev, ma_long_prev
):
    all_values = [
        rsi_buy_entry_threshold, adx_threshold_gene, vix_threshold_gene, high_vol_entry_choice_gene,
        current_price_latest, rsi_latest, rsi_prev, bbl_latest, adx_latest, vix_ma_latest,
        ma_short_latest, ma_long_latest, ma_short_prev, ma_long_prev
    ]
    
    if any(val is None or (isinstance(val, float) and not np.isfinite(val)) for val in all_values):
        logger.debug(f"GA Buy Signal Check: 發現無效的輸入數據。數值: {all_values}")
        return False 

    buy_condition = False
    is_high_vol = vix_ma_latest >= vix_threshold_gene

    if is_high_vol: 
        if high_vol_entry_choice_gene == 0: 
            if (current_price_latest <= bbl_latest) and (rsi_latest < rsi_buy_entry_threshold):
                buy_condition = True
        else: 
            if (current_price_latest <= bbl_latest) and (adx_latest > adx_threshold_gene):
                buy_condition = True
    else: 
        if (ma_short_prev < ma_long_prev) and (ma_short_latest >= ma_long_latest): 
            buy_condition = True
    
    return buy_condition and current_price_latest > 1e-9


def generate_buy_reason(gene, current_price_latest,
                        rsi_latest, rsi_prev, bbl_latest, adx_latest, vix_ma_latest,
                        ma_short_latest, ma_long_latest, ma_short_prev, ma_long_prev,
                        strategy_config=STRATEGY_CONFIG_SHARED_GA):
    try:
        rsi_buy_entry_threshold = int(gene[0])
        vix_threshold_gene = int(gene[2])
        rsi_period_idx = int(gene[4])
        vix_ma_period_idx = int(gene[5])
        bb_len_idx = int(gene[6])
        bb_std_idx = int(gene[7])
        adx_threshold_gene = int(gene[8])
        high_vol_entry_choice_gene = int(gene[9])

        is_high_vol = vix_ma_latest >= vix_threshold_gene
        reason_parts = []

        chosen_rsi_p_str = strategy_config['rsi_period_options'][rsi_period_idx]
        chosen_vix_ma_p_str = strategy_config['vix_ma_period_options'][vix_ma_period_idx]
        chosen_bb_l_str = strategy_config['bb_length_options'][bb_len_idx]
        chosen_bb_s_str = strategy_config['bb_std_options'][bb_std_idx]
        adx_period_str = strategy_config['adx_period']
        ma_short_str = strategy_config['ma_short_period']
        ma_long_str = strategy_config['ma_long_period']

        if is_high_vol: 
            reason_parts.append(f"偵測為高波動市場 (VIX {chosen_vix_ma_p_str}日均線值 {vix_ma_latest:.2f} ≥ 風險門檻 {vix_threshold_gene})")
            if high_vol_entry_choice_gene == 0: 
                reason_parts.append(f"當前價格 {current_price_latest:.2f} ≤ 布林帶下軌 {bbl_latest:.2f} (週期:{chosen_bb_l_str},標準差:{chosen_bb_s_str})")
                reason_parts.append(f"且 RSI({chosen_rsi_p_str}) 指標值 {rsi_latest:.2f} < 設定的買入區間 {rsi_buy_entry_threshold}")
            else: 
                reason_parts.append(f"當前價格 {current_price_latest:.2f} ≤ 布林帶下軌 {bbl_latest:.2f} (週期:{chosen_bb_l_str},標準差:{chosen_bb_s_str})")
                reason_parts.append(f"且 ADX({adx_period_str}) 指標值 {adx_latest:.2f} > 設定的趨勢強度門檻 {adx_threshold_gene}")
        else: 
            reason_parts.append(f"偵測為低波動市場 (VIX {chosen_vix_ma_p_str}日均線值 {vix_ma_latest:.2f} < 風險門檻 {vix_threshold_gene})")
            reason_parts.append(f"發生均線黃金交叉 (MA{ma_short_str} ({ma_short_latest:.2f}) 上穿 MA{ma_long_str} ({ma_long_latest:.2f}))")
        
        return "； ".join(reason_parts) 
    except IndexError: 
        logger.error(f"生成買入原因時因基因 {gene} 發生索引錯誤。請確保策略配置與基因結構一致。")
        return "GA買入信號，但詳細原因生成失敗(策略配置索引錯誤)。"
    except Exception as e: 
        logger.error(f"為基因 {gene} 生成買入原因時發生錯誤: {e}", exc_info=True)
        return "GA買入信號，但詳細原因生成時發生未知錯誤。"

def format_ga_gene_parameters_to_text(gene, strategy_config=STRATEGY_CONFIG_SHARED_GA):
    if not gene or len(gene) != 10:
        logger.warning(f"嘗試格式化無效的GA基因: {gene}")
        return "無效的GA策略基因。"
    try:
        rsi_buy_entry = gene[0]
        rsi_exit_ref = gene[1] 
        vix_thresh = gene[2]
        low_vol_exit_choice_code = gene[3]
        rsi_p_idx = gene[4]
        vix_ma_p_idx = gene[5]
        bb_len_idx = gene[6]
        bb_std_idx = gene[7]
        adx_thresh = gene[8]
        hv_entry_choice_code = gene[9]

        def get_option_safe(options_list, index, default_desc="未知選項"):
            try:
                return options_list[index]
            except IndexError:
                logger.warning(f"基因索引 {index} 超出選項列表 {options_list} 的範圍。")
                return default_desc

        rsi_p_str = get_option_safe(strategy_config.get('rsi_period_options', []), rsi_p_idx, f"RSI週期索引{rsi_p_idx}")
        vix_ma_p_str = get_option_safe(strategy_config.get('vix_ma_period_options', []), vix_ma_p_idx, f"VIX MA週期索引{vix_ma_p_idx}")
        bb_l_str = get_option_safe(strategy_config.get('bb_length_options', []), bb_len_idx, f"BB長度索引{bb_len_idx}")
        bb_s_str = get_option_safe(strategy_config.get('bb_std_options', []), bb_std_idx, f"BB標準差索引{bb_std_idx}")
        
        adx_p_str = strategy_config.get('adx_period', 'N/A')
        ma_s_str = strategy_config.get('ma_short_period', 'N/A')
        ma_l_str = strategy_config.get('ma_long_period', 'N/A')

        hv_entry_readable_desc = f"價格觸及布林下軌(週期:{bb_l_str},標準差:{bb_s_str})且RSI({rsi_p_str})低於買入門檻({rsi_buy_entry})" \
            if hv_entry_choice_code == 0 else \
            f"價格觸及布林下軌(週期:{bb_l_str},標準差:{bb_s_str})且ADX({adx_p_str})高於趨勢門檻({adx_thresh})"
        
        lv_exit_readable_desc = f"價格跌破MA{ma_s_str}" if low_vol_exit_choice_code == 0 else f"MA{ma_s_str}下穿MA{ma_l_str}（死亡交叉）"
        
        sell_trigger_readable_desc = (
            f"RSI({rsi_p_str})反彈至{rsi_exit_ref}後回落 (若為BB+RSI買入)； "
            f"或價格觸及布林中軌(週期:{bb_l_str},標準差:{bb_s_str}) (若為BB+ADX買入)； "
            f"或符合低波動市場賣出條件 ({lv_exit_readable_desc})"
        )

        params_text = f"""
        <b>買入條件:</b>
        - 低波動時: MA{ma_s_str} 黃金交叉 MA{ma_l_str} (條件: VIX {vix_ma_p_str}日均線 < {vix_thresh})
        - 高波動時 ({'BB+RSI策略' if hv_entry_choice_code == 0 else 'BB+ADX策略'}): {hv_entry_readable_desc}
        <b>主要賣出參考條件:</b>
        - {sell_trigger_readable_desc}
        """
        # 清理每行開頭的空白，以獲得更整潔的HTML輸出
        return "\n".join([line.strip() for line in params_text.strip().split('\n') if line.strip()])

    except IndexError as ie:
        logger.error(f"格式化GA基因參數時發生索引錯誤: {ie}，基因: {gene}，策略配置選項可能不匹配。")
        return "解析GA策略參數時發生索引錯誤，請檢查策略配置。"
    except Exception as e:
        logger.error(f"格式化GA基因參數時發生未知錯誤: {e}，基因: {gene}", exc_info=True)
        return f"解析GA策略參數時發生未知錯誤。"