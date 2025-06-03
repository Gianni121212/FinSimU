# -*- coding: utf-8 -*-
import random
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback
import time
import logging
import pandas_ta as ta
import numba
import os

# --- Settings ---
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['axes.unicode_minus'] = False

# --- Load Stock and VIX Data (保持您提供的版本，增加 verbose 控制) ---
def load_stock_data(ticker, vix_ticker="^VIX", start_date=None, end_date=None, verbose=False):
    if verbose: print(f"Attempting to load data for {ticker} and {vix_ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download([ticker, vix_ticker], start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data is None or data.empty:
            if verbose: print(f"Error: yfinance.download returned None or empty DataFrame for {ticker}.")
            return None, None, None, None
        if not isinstance(data.columns, pd.MultiIndex):
            if verbose: print(f"Error: Expected MultiIndex columns for {ticker}, got {data.columns}. Check tickers.")
            return None, None, None, None

        stock_columns_present = ticker in data.columns.get_level_values(1)
        vix_columns_present = vix_ticker in data.columns.get_level_values(1)

        if not stock_columns_present:
            if verbose: print(f"Error: Missing data for stock {ticker}.")
            return None, None, None, None

        stock_data = data.loc[:, pd.IndexSlice[:, ticker]]
        stock_data.columns = stock_data.columns.droplevel(1)

        if not vix_columns_present:
            if verbose: print(f"Warning: Missing VIX data for {vix_ticker} (for {ticker} analysis), will use NaNs. VIX-related strategies might be affected.")
            vix_data = pd.Series(np.nan, index=stock_data.index)
            vix_data.name = 'VIX_Close'
        else:
            vix_data_slice = data.loc[:, pd.IndexSlice['Close', vix_ticker]]
            if isinstance(vix_data_slice, pd.DataFrame):
                if vix_data_slice.shape[1] == 1:
                    vix_data = vix_data_slice.iloc[:, 0]
                else:
                    if verbose: print(f"Error: Could not extract VIX Close as a Series (shape: {vix_data_slice.shape}) for {ticker}.")
                    return None, None, None, None
            elif isinstance(vix_data_slice, pd.Series):
                vix_data = vix_data_slice
            else:
                if verbose: print(f"Error: Unrecognized VIX data type: {type(vix_data_slice)} for {ticker}.")
                return None, None, None, None
            vix_data.name = 'VIX_Close'

        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        missing_essential_cols = [col for col in ['Close', 'High', 'Low'] if col not in stock_data.columns]
        if missing_essential_cols:
            if verbose: print(f"Error: Missing essential columns {missing_essential_cols} for {ticker} needed for ADX calculation.")
            return None, None, None, None
        for col in required_cols:
             if col not in stock_data.columns:
                 stock_data[col] = np.nan
                 if verbose: print(f"Warning: Stock data for {ticker} missing '{col}', filled with NaN.")

        simplified_df = stock_data[required_cols].copy()
        if not vix_columns_present:
            vix_data = vix_data.reindex(simplified_df.index)

        aligned_data = pd.concat([simplified_df, vix_data], axis=1, join='inner')

        if aligned_data.empty:
            if verbose: print(f"Error: No common dates found between {ticker} and {vix_ticker} after join.")
            return None, None, None, None

        if aligned_data.isnull().values.any():
            if verbose: print(f"Warning: NaNs found after alignment for {ticker}. Filling...")
            if 'VIX_Close' in aligned_data.columns and aligned_data['VIX_Close'].isnull().any():
                aligned_data['VIX_Close'] = aligned_data['VIX_Close'].ffill().bfill()
            numeric_stock_cols = simplified_df.select_dtypes(include=np.number).columns
            for col in numeric_stock_cols:
                if col in aligned_data.columns and aligned_data[col].isnull().any():
                     aligned_data[col] = aligned_data[col].ffill().bfill()

            if aligned_data.isnull().values.any():
                nan_cols_still = aligned_data.columns[aligned_data.isnull().any()].tolist()
                if verbose: print(f"Error: Could not fill all NaNs for {ticker}. Columns with remaining NaNs: {nan_cols_still}.")
                return None, None, None, None

        final_stock_df = aligned_data[required_cols].copy()
        final_vix_series = aligned_data['VIX_Close'].copy()
        prices = final_stock_df['Close'].tolist()
        dates = final_stock_df.index.tolist()
        if verbose or not verbose : print(f"Successfully loaded and aligned {len(prices)} data points for {ticker}.")
        return prices, dates, final_stock_df, final_vix_series
    except Exception as e:
        print(f"\nError during data loading for {ticker}: {type(e).__name__}: {e}"); traceback.print_exc()
        return None, None, None, None

# --- Indicator Pre-calculation (不計算BBU) ---
def precompute_indicators(stock_df, vix_series, strategy_config, verbose=False):
    precalculated_rsi = {}
    precalculated_vix_ma = {}
    precalculated_bbl = {}
    precalculated_bbm = {}
    precalculated_fixed = {}
    indicators_ready = True
    if verbose: print(f"Starting indicator pre-calculation (Stock DF shape: {stock_df.shape}, VIX Series len: {len(vix_series)})...")

    try:
        for rsi_p in strategy_config['rsi_period_options']:
            rsi_values = ta.rsi(stock_df['Close'], length=rsi_p)
            if rsi_values is None or rsi_values.isnull().all():
                if verbose: print(f"  Error: RSI calculation failed for period {rsi_p}")
                indicators_ready = False; break
            precalculated_rsi[rsi_p] = rsi_values.tolist()
        if not indicators_ready: return {}, False
        if verbose: print(f"  RSI ({len(precalculated_rsi)} variants) OK.")

        for vix_ma_p in strategy_config['vix_ma_period_options']:
            vix_ma_values = vix_series.rolling(window=vix_ma_p).mean()
            if vix_ma_values is None or vix_ma_values.isnull().all():
                precalculated_vix_ma[vix_ma_p] = [np.nan] * len(vix_series)
                if verbose: print(f"  Warning: VIX MA (period {vix_ma_p}) calculation failed, filled with NaNs.")
            else:
                precalculated_vix_ma[vix_ma_p] = vix_ma_values.tolist()
        if verbose: print(f"  VIX MA ({len(precalculated_vix_ma)} variants) OK (or filled with NaN).")

        for bb_l in strategy_config['bb_length_options']:
            for bb_s in strategy_config['bb_std_options']:
                bbands = ta.bbands(stock_df['Close'], length=bb_l, std=bb_s)
                bbl_col_name = next((col for col in bbands.columns if 'BBL' in col), None)
                bbm_col_name = next((col for col in bbands.columns if 'BBM' in col), None)
                if bbands is None or not bbl_col_name or not bbm_col_name or \
                   bbands[bbl_col_name].isnull().all() or bbands[bbm_col_name].isnull().all():
                    if verbose: print(f"  Error: Bollinger Bands (BBL/BBM) calculation failed for L={bb_l}, S={bb_s}")
                    indicators_ready = False; break
                precalculated_bbl[(bb_l, bb_s)] = bbands[bbl_col_name].tolist()
                precalculated_bbm[(bb_l, bb_s)] = bbands[bbm_col_name].tolist()
            if not indicators_ready: break
        if not indicators_ready: return {}, False
        if verbose: print(f"  Bollinger Bands (BBL/BBM, {len(precalculated_bbl)} variants) OK.")

        bbi_periods = strategy_config['bbi_periods']
        sma1 = stock_df['Close'].rolling(window=bbi_periods[0]).mean(); sma2 = stock_df['Close'].rolling(window=bbi_periods[1]).mean()
        sma3 = stock_df['Close'].rolling(window=bbi_periods[2]).mean(); sma4 = stock_df['Close'].rolling(window=bbi_periods[3]).mean()
        bbi_values = (sma1 + sma2 + sma3 + sma4) / 4
        if bbi_values is None or bbi_values.isnull().all():
            if verbose: print("  Error: BBI calculation failed"); indicators_ready = False
        else:
            precalculated_fixed['bbi_list'] = bbi_values.tolist()
            if verbose: print("  BBI OK.")

        if indicators_ready:
            adx_df = ta.adx(stock_df['High'], stock_df['Low'], stock_df['Close'], length=strategy_config['adx_period'])
            adx_col_name = next((col for col in adx_df.columns if 'ADX' in col), None)
            if adx_df is None or not adx_col_name or adx_df[adx_col_name].isnull().all():
                if verbose: print(f"  Error: ADX calculation failed"); indicators_ready = False
            else:
                precalculated_fixed['adx_list'] = adx_df[adx_col_name].tolist()
                if verbose: print("  ADX OK.")

        if indicators_ready:
            ma_short = stock_df['Close'].rolling(window=strategy_config['ma_short_period']).mean()
            ma_long = stock_df['Close'].rolling(window=strategy_config['ma_long_period']).mean()
            if ma_short is None or ma_long is None or ma_short.isnull().all() or ma_long.isnull().all():
                if verbose: print(f"  Error: Moving Averages calculation failed"); indicators_ready = False
            else:
                precalculated_fixed['ma_short_list'] = ma_short.tolist()
                precalculated_fixed['ma_long_list'] = ma_long.tolist()
                if verbose: print("  MAs OK.")

        if indicators_ready:
            if verbose: print("All indicator pre-calculations finished.")
            return {'rsi': precalculated_rsi, 'vix_ma': precalculated_vix_ma,
                    'bbl': precalculated_bbl, 'bbm': precalculated_bbm,
                    'fixed': precalculated_fixed}, True
        else:
            if verbose: print("Error during indicator pre-calculation.")
            return {}, False
    except Exception as e:
        if verbose: print(f"Unexpected error during indicator pre-calculation: {type(e).__name__}: {e}"); traceback.print_exc()
        return {}, False

# --- Numba Core Strategy (保持您提供的 model_train2.py 的邏輯) ---
@numba.jit(nopython=True)
def run_strategy_numba_core(rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold, vix_threshold,
                            low_vol_exit_strategy_MA, high_vol_entry_choice, low_vol_strategy_choice,
                            commission_rate,
                            prices_arr,
                            rsi_arr, bbl_arr, bbm_arr, bbi_arr, adx_arr, vix_ma_arr, ma_short_arr, ma_long_arr,
                            start_trading_iloc):
    T = len(prices_arr); portfolio_values_arr = np.full(T, 1.0, dtype=np.float64)
    max_signals = T // 2 + 1
    buy_signal_indices = np.full(max_signals, -1, dtype=np.int64); buy_signal_prices = np.full(max_signals, np.nan, dtype=np.float64); buy_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    sell_signal_indices = np.full(max_signals, -1, dtype=np.int64); sell_signal_prices = np.full(max_signals, np.nan, dtype=np.float64); sell_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    buy_count = 0; sell_count = 0; cash = 1.0; stock = 0.0; position = 0; last_valid_portfolio_value = 1.0
    rsi_crossed_exit_level_after_buy = False
    high_vol_entry_type = -1
    low_vol_entry_type = -1

    start_trading_iloc = max(1, start_trading_iloc)
    if start_trading_iloc >= T:
        portfolio_values_arr[:] = 1.0
        return portfolio_values_arr, buy_signal_indices[:0], buy_signal_prices[:0], buy_signal_rsis[:0], \
               sell_signal_indices[:0], sell_signal_prices[:0], sell_signal_rsis[:0]
    portfolio_values_arr[:start_trading_iloc] = 1.0

    for i in range(start_trading_iloc, T):
        current_price = prices_arr[i]; rsi_i, rsi_prev = rsi_arr[i], rsi_arr[i-1]; bbl_i = bbl_arr[i]; bbm_i = bbm_arr[i]
        bbi_i, bbi_prev = bbi_arr[i], bbi_arr[i-1]
        adx_i = adx_arr[i]; vix_ma_i = vix_ma_arr[i]
        ma_short_i, ma_long_i = ma_short_arr[i], ma_long_arr[i]; ma_short_prev, ma_long_prev = ma_short_arr[i-1], ma_long_arr[i-1]

        required_values = (rsi_i, rsi_prev, current_price, bbl_i, bbm_i, bbi_i, bbi_prev, adx_i, ma_short_i, ma_long_i, ma_short_prev, ma_long_prev)
        is_valid = True
        for val in required_values:
            if not np.isfinite(val): is_valid = False; break
        vix_is_valid = np.isfinite(vix_ma_i)

        if not is_valid:
            current_portfolio_value = cash if position == 0 else (stock * current_price if np.isfinite(current_price) else np.nan)
            portfolio_values_arr[i] = last_valid_portfolio_value if np.isnan(current_portfolio_value) else current_portfolio_value
            if not np.isnan(current_portfolio_value): last_valid_portfolio_value = current_portfolio_value
            continue

        is_high_vol = vix_is_valid and (vix_ma_i >= vix_threshold)
        if not is_high_vol or high_vol_entry_type != 0: rsi_crossed_exit_level_after_buy = False
        if not is_high_vol: high_vol_entry_type = -1
        if is_high_vol or position == 0: low_vol_entry_type = -1

        if position == 0:
            buy_condition = False; entry_type_if_bought = -1; lv_entry_type_if_bought = -1
            if is_high_vol:
                if high_vol_entry_choice == 0:
                    if (current_price <= bbl_i) and (rsi_i < rsi_buy_entry_threshold):
                        buy_condition = True; entry_type_if_bought = 0
                else:
                    if (current_price <= bbl_i) and (adx_i > adx_threshold):
                        buy_condition = True; entry_type_if_bought = 1
            else:
                if low_vol_strategy_choice == 0:
                    if (ma_short_prev < ma_long_prev) and (ma_short_i >= ma_long_i):
                        buy_condition = True; lv_entry_type_if_bought = 0
                else:
                    if current_price > bbi_i:
                        buy_condition = True; lv_entry_type_if_bought = 1
            if buy_condition and current_price > 1e-9:
                cost = cash * commission_rate; amount_to_invest = cash - cost
                if amount_to_invest > 0:
                    stock = amount_to_invest / current_price; cash = 0.0; position = 1
                    rsi_crossed_exit_level_after_buy = False
                    high_vol_entry_type = entry_type_if_bought; low_vol_entry_type = lv_entry_type_if_bought
                    if buy_count < max_signals: buy_signal_indices[buy_count] = i; buy_signal_prices[buy_count] = current_price; buy_signal_rsis[buy_count] = rsi_i; buy_count += 1
        elif position == 1:
            sell_condition = False
            if high_vol_entry_type == 0:
                if rsi_i >= rsi_exit_threshold: rsi_crossed_exit_level_after_buy = True
                if rsi_crossed_exit_level_after_buy and rsi_i < rsi_exit_threshold: sell_condition = True
            elif high_vol_entry_type == 1: sell_condition = (current_price >= bbm_i)
            elif low_vol_entry_type == 0:
                if low_vol_exit_strategy_MA == 0: sell_condition = (current_price < ma_short_i)
                else: sell_condition = (ma_short_prev > ma_long_prev) and (ma_short_i <= ma_long_i)
            elif low_vol_entry_type == 1:
                sell_condition = (current_price < bbi_i)
            if sell_condition:
                proceeds = stock * current_price; cost = proceeds * commission_rate; cash = proceeds - cost
                stock = 0.0; position = 0
                rsi_crossed_exit_level_after_buy = False; high_vol_entry_type = -1; low_vol_entry_type = -1
                if sell_count < max_signals: sell_signal_indices[sell_count] = i; sell_signal_prices[sell_count] = current_price; sell_signal_rsis[sell_count] = rsi_i; sell_count += 1
        current_stock_value = stock * current_price if position == 1 else 0.0
        current_portfolio_value = cash + current_stock_value
        portfolio_values_arr[i] = last_valid_portfolio_value if np.isnan(current_portfolio_value) else current_portfolio_value
        if not np.isnan(current_portfolio_value): last_valid_portfolio_value = current_portfolio_value
    if T > 0 and np.isnan(portfolio_values_arr[-1]): portfolio_values_arr[-1] = last_valid_portfolio_value
    return portfolio_values_arr, buy_signal_indices[:buy_count], buy_signal_prices[:buy_count], buy_signal_rsis[:buy_count], sell_signal_indices[:sell_count], sell_signal_prices[:sell_count], sell_signal_rsis[:sell_count]

# --- Wrapper function (保持您提供的版本) ---
def run_strategy(rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold, vix_threshold,
                 low_vol_exit_strategy_MA, high_vol_entry_choice, low_vol_strategy_choice,
                 commission_rate,
                 prices, dates,
                 rsi_list, bbl_list, bbm_list, bbi_list, adx_list, vix_ma_list, ma_short_list, ma_long_list):
    T = len(prices)
    if T == 0: return [1.0], [], []
    prices_arr = np.array(prices, dtype=np.float64); rsi_arr = np.array(rsi_list, dtype=np.float64)
    bbl_arr = np.array(bbl_list, dtype=np.float64); bbm_arr = np.array(bbm_list, dtype=np.float64)
    bbi_arr = np.array(bbi_list, dtype=np.float64); adx_arr = np.array(adx_list, dtype=np.float64)
    vix_ma_arr = np.array(vix_ma_list, dtype=np.float64)
    ma_short_arr = np.array(ma_short_list, dtype=np.float64); ma_long_arr = np.array(ma_long_list, dtype=np.float64)
    def get_first_valid_iloc(indicator_arr):
        valid_indices = np.where(np.isfinite(indicator_arr))[0]
        return valid_indices[0] if len(valid_indices) > 0 else T
    start_iloc_rsi = get_first_valid_iloc(rsi_arr) if len(rsi_arr) > 0 else T
    start_iloc_bbl = get_first_valid_iloc(bbl_arr) if len(bbl_arr) > 0 else T
    start_iloc_bbm = get_first_valid_iloc(bbm_arr) if len(bbm_arr) > 0 else T
    start_iloc_bbi = get_first_valid_iloc(bbi_arr) if len(bbi_arr) > 0 else T
    start_iloc_adx = get_first_valid_iloc(adx_arr) if len(adx_arr) > 0 else T
    start_iloc_vix_ma = get_first_valid_iloc(vix_ma_arr) if len(vix_ma_arr) > 0 else T
    start_iloc_ma_short = get_first_valid_iloc(ma_short_arr) if len(ma_short_arr) > 0 else T
    start_iloc_ma_long = get_first_valid_iloc(ma_long_arr) if len(ma_long_arr) > 0 else T
    start_trading_iloc = max(start_iloc_rsi, start_iloc_bbl, start_iloc_bbm, start_iloc_bbi, start_iloc_adx, start_iloc_vix_ma, start_iloc_ma_short, start_iloc_ma_long) + 1
    if start_trading_iloc >= T: return [1.0] * T, [], []
    start_trading_iloc = max(start_trading_iloc, 1)
    portfolio_values_arr, buy_indices, buy_prices, buy_rsis, sell_indices, sell_prices, sell_rsis = \
        run_strategy_numba_core(
            float(rsi_buy_entry_threshold), float(rsi_exit_threshold), float(adx_threshold), float(vix_threshold),
            int(low_vol_exit_strategy_MA), int(high_vol_entry_choice), int(low_vol_strategy_choice),
            float(commission_rate),
            prices_arr, rsi_arr, bbl_arr, bbm_arr, bbi_arr, adx_arr, vix_ma_arr, ma_short_arr, ma_long_arr,
            start_trading_iloc
        )
    buy_signals = []; sell_signals = []
    for idx, price, rsi_val in zip(buy_indices, buy_prices, buy_rsis):
        if idx != -1 and idx < len(dates): buy_signals.append((dates[idx], price, rsi_val))
    for idx, price, rsi_val in zip(sell_indices, sell_prices, sell_rsis):
         if idx != -1 and idx < len(dates): sell_signals.append((dates[idx], price, rsi_val))
    return portfolio_values_arr.tolist(), buy_signals, sell_signals

# --- Genetic Algorithm (保持您提供的版本，並調整show_process的影響) ---
def genetic_algorithm_with_elitism(prices, dates,
                                   precalculated_indicators,
                                   ga_params):
    generations = ga_params['generations']; population_size = ga_params['population_size']; crossover_rate = ga_params['crossover_rate']
    mutation_rate = ga_params['mutation_rate']; elitism_size = ga_params['elitism_size']; tournament_size = ga_params['tournament_size']
    mutation_amount_range = ga_params['mutation_amount_range']
    show_ga_generations = ga_params.get('show_process', False)

    rsi_threshold_range = ga_params['rsi_threshold_range']; vix_threshold_range = ga_params['vix_threshold_range']
    adx_threshold_range = ga_params['adx_threshold_range']
    rsi_period_options = ga_params['rsi_period_options']; num_rsi_options = len(rsi_period_options)
    vix_ma_period_options = ga_params['vix_ma_period_options']; num_vix_ma_options = len(vix_ma_period_options)
    bb_length_options = ga_params['bb_length_options']; num_bb_len_options = len(bb_length_options)
    bb_std_options = ga_params['bb_std_options']; num_bb_std_options = len(bb_std_options)
    commission_rate = ga_params['commission_rate']
    vix_mutation_amount_range = ga_params.get('vix_mutation_amount_range', mutation_amount_range)
    adx_mutation_amount_range = ga_params.get('adx_mutation_amount_range', mutation_amount_range)

    fixed_inds = precalculated_indicators.get('fixed', {})
    bbi_list = fixed_inds.get('bbi_list'); adx_list = fixed_inds.get('adx_list')
    ma_short_list = fixed_inds.get('ma_short_list'); ma_long_list = fixed_inds.get('ma_long_list')
    if any(lst is None for lst in [bbi_list, adx_list, ma_short_list, ma_long_list]):
        if show_ga_generations: print("Error: Missing one or more fixed indicator lists in precalculated_indicators['fixed'].")
        return None, 0

    T = len(prices)
    if T < 2:
        if show_ga_generations: print(f"Error: Data length {T} too short.")
        return None, 0

    population = []; attempts, max_attempts = 0, population_size * 200
    min_buy, max_buy, min_exit, max_exit = rsi_threshold_range; min_vix, max_vix = vix_threshold_range; min_adx, max_adx = adx_threshold_range
    while len(population) < population_size and attempts < max_attempts:
        buy_entry_thr = random.randint(min_buy, max_buy); exit_thr = random.randint(max(buy_entry_thr + 1, min_exit), max_exit); vix_thr = random.randint(min_vix, max_vix)
        low_vol_exit_MA = random.choice([0, 1]); rsi_p_choice = random.randint(0, num_rsi_options - 1); vix_ma_p_choice = random.randint(0, num_vix_ma_options - 1)
        bb_len_choice = random.randint(0, num_bb_len_options - 1); bb_std_choice = random.randint(0, num_bb_std_options - 1)
        adx_thr = random.randint(min_adx, max_adx); hv_entry_choice = random.choice([0, 1])
        low_vol_strat_choice = random.choice([0, 1])
        gene = [buy_entry_thr, exit_thr, vix_thr, low_vol_exit_MA, rsi_p_choice, vix_ma_p_choice, bb_len_choice, bb_std_choice, adx_thr, hv_entry_choice, low_vol_strat_choice]
        if (0<gene[0]<gene[1]<100 and min_buy<=gene[0]<=max_buy and min_exit<=gene[1]<=max_exit and min_vix<=gene[2]<=max_vix and gene[3] in [0,1] and 0<=gene[4]<num_rsi_options and 0<=gene[5]<num_vix_ma_options and 0<=gene[6]<num_bb_len_options and 0<=gene[7]<num_bb_std_options and min_adx<=gene[8]<=max_adx and gene[9] in [0,1] and gene[10] in [0,1]):
             population.append(gene)
        attempts += 1
    if not population or len(population) < population_size :
        if show_ga_generations: print(f"Error: Could not generate sufficient initial population ({len(population)}/{population_size}).")
        return None, 0
    best_gene_overall = population[0][:]; best_fitness_overall = -float('inf')

    for generation in range(generations):
        fitness = []
        for gene_idx, gene in enumerate(population):
            try:
                chosen_rsi_period = rsi_period_options[gene[4]]; chosen_vix_ma_period = vix_ma_period_options[gene[5]]
                chosen_bb_length = bb_length_options[gene[6]]; chosen_bb_std = bb_std_options[gene[7]]
                rsi_list = precalculated_indicators['rsi'][chosen_rsi_period]
                vix_ma_list = precalculated_indicators['vix_ma'][chosen_vix_ma_period]
                bbl_list = precalculated_indicators['bbl'][(chosen_bb_length, chosen_bb_std)]
                bbm_list = precalculated_indicators['bbm'][(chosen_bb_length, chosen_bb_std)]
                portfolio_values, _, _ = run_strategy(
                    gene[0], gene[1], gene[8], gene[2], gene[3], gene[9], gene[10],
                    commission_rate, prices, dates,
                    rsi_list, bbl_list, bbm_list, bbi_list, adx_list, vix_ma_list, ma_short_list, ma_long_list
                )
                final_value = next((p for p in reversed(portfolio_values) if np.isfinite(p)), -np.inf); fitness.append(final_value)
            except (IndexError, KeyError) as e_eval:
                if show_ga_generations: print(f"  Eval Error (Gene: {gene}): {e_eval}. Fitness -inf.")
                fitness.append(-np.inf)
            except Exception as e_unexp:
                if show_ga_generations: print(f"  Unexpected Eval Error (Gene: {gene}): {e_unexp}. Fitness -inf."); traceback.print_exc()
                fitness.append(-np.inf)

        fitness_array = np.array(fitness); valid_fitness_mask = np.isfinite(fitness_array) & (fitness_array > -np.inf); valid_indices = np.where(valid_fitness_mask)[0]; valid_fitness_count = len(valid_indices)
        if valid_fitness_count == 0:
             if show_ga_generations: print(f"Gen {generation+1} - Warning: All individuals invalid.")
             continue
        sorted_valid_indices = valid_indices[np.argsort(fitness_array[valid_indices])[::-1]]; num_elites = min(elitism_size, valid_fitness_count); elite_indices = sorted_valid_indices[:num_elites]; elites = [population[i][:] for i in elite_indices]
        current_best_fitness_in_gen = fitness_array[elite_indices[0]] if num_elites > 0 else -np.inf
        if current_best_fitness_in_gen > best_fitness_overall: best_fitness_overall = current_best_fitness_in_gen; best_gene_overall = population[elite_indices[0]][:]

        if show_ga_generations and (generation + 1) % 10 == 0:
            gen_best_str = f"{current_best_fitness_in_gen:.4f}" if num_elites > 0 else "N/A"
            overall_best_str = "N/A"
            if best_fitness_overall > -np.inf and best_gene_overall:
                bo_lv_exit_ma_str = "Price<MA" if best_gene_overall[3] == 0 else "MACross"
                bo_rsi_p_str = rsi_period_options[best_gene_overall[4]]
                bo_vix_ma_p_str = vix_ma_period_options[best_gene_overall[5]]
                bo_bb_l_str = bb_length_options[best_gene_overall[6]]
                bo_bb_s_str = bb_std_options[best_gene_overall[7]]
                bo_hv_entry_str = "BB+RSI" if best_gene_overall[9] == 0 else "BB+ADX"
                bo_lv_strat_str = "MA" if best_gene_overall[10] == 0 else "BBI"
                overall_best_params_readable = (
                    f"BuyE={best_gene_overall[0]},ExitR={best_gene_overall[1]},VIX_T={best_gene_overall[2]},LVExitMA={bo_lv_exit_ma_str},"
                    f"RSI_P={bo_rsi_p_str},VIX_MA_P={bo_vix_ma_p_str},BB_L={bo_bb_l_str},BB_S={bo_bb_s_str},"
                    f"ADX_T={best_gene_overall[8]},HVEntry={bo_hv_entry_str},LVStrat={bo_lv_strat_str}"
                )
                overall_best_str = f"{best_fitness_overall:.4f} (Params: {overall_best_params_readable})"
            print(f"Gen {generation+1}/{generations} | Best(G): {gen_best_str} | Best(O): {overall_best_str} | Valid: {valid_fitness_count}/{population_size}")

        selected_parents = []; num_parents_to_select = population_size - num_elites
        if num_parents_to_select <= 0: population = elites[:population_size]; continue
        effective_tournament_size = min(tournament_size, valid_fitness_count)
        if effective_tournament_size <= 0: population = elites[:population_size]; continue

        for _ in range(num_parents_to_select):
             aspirant_indices_local = np.random.choice(len(valid_indices), size=effective_tournament_size, replace=False); aspirant_indices_global = valid_indices[aspirant_indices_local]; winner_global_idx = aspirant_indices_global[np.argmax(fitness_array[aspirant_indices_global])]; selected_parents.append(population[winner_global_idx][:])
        offspring = []; parent_indices = list(range(len(selected_parents))); random.shuffle(parent_indices); num_pairs = len(parent_indices) // 2
        for i_pair in range(num_pairs):
            p1, p2 = selected_parents[parent_indices[2*i_pair]], selected_parents[parent_indices[2*i_pair + 1]]; child1, child2 = p1[:], p2[:]
            if random.random() < crossover_rate:
                 crossover_point = random.randint(1, 10); child1_new = p1[:crossover_point] + p2[crossover_point:]; child2_new = p2[:crossover_point] + p1[crossover_point:]
                 valid_c1 = (0<child1_new[0]<child1_new[1]<100 and min_buy<=child1_new[0]<=max_buy and min_exit<=child1_new[1]<=max_exit and min_vix<=child1_new[2]<=max_vix and child1_new[3] in [0,1] and 0<=child1_new[4]<num_rsi_options and 0<=child1_new[5]<num_vix_ma_options and 0<=child1_new[6]<num_bb_len_options and 0<=child1_new[7]<num_bb_std_options and min_adx<=child1_new[8]<=max_adx and child1_new[9] in [0,1] and child1_new[10] in [0,1])
                 valid_c2 = (0<child2_new[0]<child2_new[1]<100 and min_buy<=child2_new[0]<=max_buy and min_exit<=child2_new[1]<=max_exit and min_vix<=child2_new[2]<=max_vix and child2_new[3] in [0,1] and 0<=child2_new[4]<num_rsi_options and 0<=child2_new[5]<num_vix_ma_options and 0<=child2_new[6]<num_bb_len_options and 0<=child2_new[7]<num_bb_std_options and min_adx<=child2_new[8]<=max_adx and child2_new[9] in [0,1] and child2_new[10] in [0,1])
                 child1 = child1_new if valid_c1 else p1[:]; child2 = child2_new if valid_c2 else p2[:]
            offspring.append(child1); offspring.append(child2)
        if len(parent_indices) % 2 != 0: offspring.append(selected_parents[parent_indices[-1]][:])

        mut_min, mut_max = mutation_amount_range
        vix_mut_min_actual, vix_mut_max_actual = vix_mutation_amount_range
        adx_mut_min_actual, adx_mut_max_actual = adx_mutation_amount_range

        for i_offspring in range(len(offspring)):
            if random.random() < mutation_rate:
                gene_to_mutate = offspring[i_offspring]; original_gene = gene_to_mutate[:]; mutate_idx = random.randint(0, 10)
                if mutate_idx == 3: gene_to_mutate[3] = 1 - gene_to_mutate[3]
                elif mutate_idx == 4:
                    if num_rsi_options > 1:
                        new_choice = random.randint(0, num_rsi_options - 1)
                        while new_choice == original_gene[4]: new_choice = random.randint(0, num_rsi_options - 1)
                        gene_to_mutate[4] = new_choice
                elif mutate_idx == 5:
                    if num_vix_ma_options > 1:
                        new_choice = random.randint(0, num_vix_ma_options - 1)
                        while new_choice == original_gene[5]: new_choice = random.randint(0, num_vix_ma_options - 1)
                        gene_to_mutate[5] = new_choice
                elif mutate_idx == 6:
                    if num_bb_len_options > 1:
                        new_choice = random.randint(0, num_bb_len_options - 1)
                        while new_choice == original_gene[6]: new_choice = random.randint(0, num_bb_len_options - 1)
                        gene_to_mutate[6] = new_choice
                elif mutate_idx == 7:
                    if num_bb_std_options > 1:
                        new_choice = random.randint(0, num_bb_std_options - 1)
                        while new_choice == original_gene[7]: new_choice = random.randint(0, num_bb_std_options - 1)
                        gene_to_mutate[7] = new_choice
                elif mutate_idx == 9: gene_to_mutate[9] = 1 - gene_to_mutate[9]
                elif mutate_idx == 10: gene_to_mutate[10] = 1 - gene_to_mutate[10]
                else:
                    if mutate_idx == 2: mutation_amount = random.randint(vix_mut_min_actual, vix_mut_max_actual); is_zero_range = (vix_mut_min_actual == 0 and vix_mut_max_actual == 0)
                    elif mutate_idx == 8: mutation_amount = random.randint(adx_mut_min_actual, adx_mut_max_actual); is_zero_range = (adx_mut_min_actual == 0 and adx_mut_max_actual == 0)
                    else: mutation_amount = random.randint(mut_min, mut_max); is_zero_range = (mut_min == 0 and mut_max == 0)
                    if mutation_amount == 0 and not is_zero_range:
                        while mutation_amount == 0:
                            if mutate_idx == 2: mutation_amount = random.randint(vix_mut_min_actual, vix_mut_max_actual)
                            elif mutate_idx == 8: mutation_amount = random.randint(adx_mut_min_actual, adx_mut_max_actual)
                            else: mutation_amount = random.randint(mut_min, mut_max)
                    gene_to_mutate[mutate_idx] += mutation_amount
                    gene_to_mutate[0] = max(min_buy, min(gene_to_mutate[0], max_buy))
                    gene_to_mutate[1] = max(gene_to_mutate[0] + 1, min_exit, min(gene_to_mutate[1], max_exit))
                    gene_to_mutate[0] = max(min_buy, min(gene_to_mutate[0], gene_to_mutate[1] - 1, max_buy))
                    gene_to_mutate[2] = max(min_vix, min(gene_to_mutate[2], max_vix))
                    gene_to_mutate[8] = max(min_adx, min(gene_to_mutate[8], max_adx))
                final_valid = (0<gene_to_mutate[0]<gene_to_mutate[1]<100 and min_buy<=gene_to_mutate[0]<=max_buy and min_exit<=gene_to_mutate[1]<=max_exit and min_vix<=gene_to_mutate[2]<=max_vix and gene_to_mutate[3] in [0,1] and 0<=gene_to_mutate[4]<num_rsi_options and 0<=gene_to_mutate[5]<num_vix_ma_options and 0<=gene_to_mutate[6]<num_bb_len_options and 0<=gene_to_mutate[7]<num_bb_std_options and min_adx<=gene_to_mutate[8]<=max_adx and gene_to_mutate[9] in [0,1] and gene_to_mutate[10] in [0,1])
                if not final_valid: offspring[i_offspring] = original_gene
        population = elites + offspring; population = population[:population_size]
    if best_fitness_overall == -float('inf'):
        if show_ga_generations: print("Error: GA finished without finding any valid solution.")
        return None, 0
    return best_gene_overall, best_fitness_overall

# --- Main Execution Block (維持您提供的版本結構，但優化最終結果輸出) ---
if __name__ == "__main__":
    print("Executing model_train2_final_v3.py as a standalone script (for testing)...")
    config = { 'ticker': 'NVDA', 'vix_ticker': '^VIX', 'start_date': '2021-01-01', 'end_date': '2025-05-03' }
    strategy_config_main = { # Renamed to avoid conflict with outer 'strategy_config' from your provided code
        'rsi_period_options': [6, 12, 21], 'vix_ma_period_options': [2, 5, 10],
        'bb_length_options': [10, 20], 'bb_std_options': [1.5, 2.0], 'adx_period': 14,
        'bbi_periods': (3, 6, 12, 24), 'ma_short_period': 5, 'ma_long_period': 10,
        'commission_pct': 0.003,
    }
    ga_params_config = {
        'generations': 20, 'population_size': 100, 'crossover_rate': 0.7, 'mutation_rate': 0.25,
        'elitism_size': 1, 'tournament_size': 2, 'mutation_amount_range': (-3, 3),
        'vix_mutation_amount_range': (-2, 2), 'adx_mutation_amount_range': (-2, 2),
        'show_process': False, # GA代數詳細輸出開關 (来自您提供的版本)
        'rsi_threshold_range': (10, 40, 45, 75), 'vix_threshold_range': (15, 35),
        'adx_threshold_range': (20, 40),
        'rsi_period_options': strategy_config_main['rsi_period_options'],
        'vix_ma_period_options': strategy_config_main['vix_ma_period_options'],
        'bb_length_options': strategy_config_main['bb_length_options'],
        'bb_std_options': strategy_config_main['bb_std_options'],
        'adx_period': strategy_config_main['adx_period'], 'bbi_periods': strategy_config_main['bbi_periods'],
        'ma_short_period': strategy_config_main['ma_short_period'], 'ma_long_period': strategy_config_main['ma_long_period'],
        'commission_rate': strategy_config_main['commission_pct'],
    }
    run_config_main = { 'num_runs': 100} # Single GA run for faster testing of this feature
    plot_config_main = {
        'show_plot': True, # Default to True for the custom period backtest
        'figure_size': (15, 10), 'buy_marker_color': 'lime', 'sell_marker_color': 'red',
        'strategy_line_color': 'blue', 'price_line_color': 'darkgrey', 'plot_bbands': True,
        'bb_line_color': 'grey', 'bb_fill_color': 'grey', 'plot_mas': True, # Enable MAs for better viz
        'ma_short_color': 'lightblue', 'ma_long_color': 'lightcoral',
    }

    # --- GA Optimization Phase ---
    print(f"--- GA Optimization Phase for {config['ticker']} ({config['start_date']} to {config['end_date']}) ---")
    ga_prices, ga_dates, ga_stock_df, ga_vix_series = load_stock_data(
        config['ticker'], vix_ticker=config['vix_ticker'],
        start_date=config['start_date'], end_date=config['end_date'],
        verbose=False # Keep GA phase data loading quiet
    )

    if ga_prices and ga_dates and ga_stock_df is not None and ga_vix_series is not None:
        print("Pre-calculating indicators for GA phase...")
        ga_precalculated_indicators, ga_indicators_ready = precompute_indicators(
            ga_stock_df, ga_vix_series, strategy_config_main, verbose=False
        )

        if ga_indicators_ready:
            print(f"Starting {run_config_main['num_runs']} GA runs...")
            overall_best_params_for_ga = None
            overall_best_fitness_for_ga = -float('inf')

            for run_idx in range(run_config_main['num_runs']):
                current_best_params, current_fitness = genetic_algorithm_with_elitism(
                    ga_prices, ga_dates,
                    ga_precalculated_indicators,
                    ga_params=ga_params_config
                )
                if current_best_params and current_fitness > overall_best_fitness_for_ga:
                    overall_best_fitness_for_ga = current_fitness
                    overall_best_params_for_ga = current_best_params
                # Per-run summary from your original code
                if current_best_params and np.isfinite(current_fitness):
                    lv_exit_ma_str_run = "Price<MA" if current_best_params[3] == 0 else "MACross"
                    chosen_rsi_p_run = strategy_config_main['rsi_period_options'][current_best_params[4]]
                    # ... (rest of the per-run parameter string construction) ...
                    print(f"  GA Run {run_idx + 1} Best Value: {current_fitness:.4f}. (Params summary can be added here if needed)")


            if overall_best_params_for_ga:
                print("\n--- GA Optimization Phase Complete ---")
                print(f"Best Fitness from GA: {overall_best_fitness_for_ga:.4f}")
                # Detailed printout of GA best params
                bp_ga = overall_best_params_for_ga
                chosen_rsi_p_ga = strategy_config_main['rsi_period_options'][bp_ga[4]]
                chosen_vix_ma_p_ga = strategy_config_main['vix_ma_period_options'][bp_ga[5]]
                chosen_bb_l_ga = strategy_config_main['bb_length_options'][bp_ga[6]]
                chosen_bb_s_ga = strategy_config_main['bb_std_options'][bp_ga[7]]
                hv_entry_ga = "BB+RSI" if bp_ga[9] == 0 else "BB+ADX"
                lv_strat_ga = "MA Crossover" if bp_ga[10] == 0 else "Pure BBI"
                print(f"Best GA Params: RSI_P={chosen_rsi_p_ga}, VIX_MA_P={chosen_vix_ma_p_ga}, BB=({chosen_bb_l_ga},{chosen_bb_s_ga}), "
                      f"VIX_T={bp_ga[2]}, ADX_T={bp_ga[8]}, BuyE={bp_ga[0]}, ExitR={bp_ga[1]}, "
                      f"HVEntry={hv_entry_ga}, LVStrat={lv_strat_ga}, LVExitMA_Choice={bp_ga[3]}")


                # --- Custom Period Backtest Phase ---
                print("\n--- Custom Period Backtest with GA Best Parameters ---")
                # Define your custom backtest period here
                custom_start_date = "2021-01-01" # EXAMPLE: Different start date
                custom_end_date = "2025-05-03"   # EXAMPLE: Different end date
                print(f"Custom Backtest Period: {custom_start_date} to {custom_end_date}")

                backtest_prices, backtest_dates, backtest_stock_df, backtest_vix_series = load_stock_data(
                    config['ticker'], vix_ticker=config['vix_ticker'],
                    start_date=custom_start_date, end_date=custom_end_date, verbose=True
                )

                if backtest_prices and backtest_dates and backtest_stock_df is not None and backtest_vix_series is not None:
                    print("Pre-calculating indicators for custom backtest period...")
                    backtest_precalculated_indicators, backtest_indicators_ready = precompute_indicators(
                        backtest_stock_df, backtest_vix_series, strategy_config_main, verbose=True
                    )

                    if backtest_indicators_ready:
                        print("Running strategy on custom period with GA best parameters...")
                        # Extract chosen params for clarity (already done above as chosen_..._ga)
                        chosen_rsi_list_bt = backtest_precalculated_indicators['rsi'][chosen_rsi_p_ga]
                        chosen_vix_ma_list_bt = backtest_precalculated_indicators['vix_ma'][chosen_vix_ma_p_ga]
                        chosen_bbl_list_bt = backtest_precalculated_indicators['bbl'][(int(chosen_bb_l_ga), float(chosen_bb_s_ga))]
                        chosen_bbm_list_bt = backtest_precalculated_indicators['bbm'][(int(chosen_bb_l_ga), float(chosen_bb_s_ga))]
                        fixed_bbi_bt = backtest_precalculated_indicators['fixed']['bbi_list']
                        fixed_adx_bt = backtest_precalculated_indicators['fixed']['adx_list']
                        fixed_ma_short_bt = backtest_precalculated_indicators['fixed']['ma_short_list']
                        fixed_ma_long_bt = backtest_precalculated_indicators['fixed']['ma_long_list']

                        backtest_portfolio_values, backtest_buy_signals, backtest_sell_signals = run_strategy(
                            overall_best_params_for_ga[0], overall_best_params_for_ga[1], overall_best_params_for_ga[8],
                            overall_best_params_for_ga[2], overall_best_params_for_ga[3], overall_best_params_for_ga[9],
                            overall_best_params_for_ga[10],
                            strategy_config_main['commission_pct'],
                            backtest_prices, backtest_dates,
                            chosen_rsi_list_bt, chosen_bbl_list_bt, chosen_bbm_list_bt,
                            fixed_bbi_bt, fixed_adx_bt, chosen_vix_ma_list_bt,
                            fixed_ma_short_bt, fixed_ma_long_bt
                        )

                        if backtest_portfolio_values:
                            final_backtest_value = next((p for p in reversed(backtest_portfolio_values) if np.isfinite(p)), 1.0)
                            print(f"Custom Period Final Portfolio Value: {final_backtest_value:.4f}")

                            num_bt_trades = min(len(backtest_buy_signals), len(backtest_sell_signals))
                            winning_bt_trades = 0
                            if num_bt_trades > 0:
                                for i_bt_trade in range(num_bt_trades):
                                    if backtest_sell_signals[i_bt_trade][1] > backtest_buy_signals[i_bt_trade][1]:
                                        winning_bt_trades +=1
                                bt_win_rate = (winning_bt_trades / num_bt_trades) * 100 if num_bt_trades > 0 else 0.0
                                print(f"  Trades in custom period: {num_bt_trades}, Wins: {winning_bt_trades}, Win Rate: {bt_win_rate:.2f}%")


                            if plot_config_main['show_plot']:
                                print("Plotting custom period backtest...")
                                fig_bt, ax_bt = plt.subplots(figsize=plot_config_main['figure_size'])
                                initial_price_bt = backtest_prices[0] if backtest_prices and np.isfinite(backtest_prices[0]) and backtest_prices[0] > 1e-9 else 1.0
                                normalized_prices_bt = np.array(backtest_prices, dtype=float) / initial_price_bt
                                ax_bt.plot(backtest_dates, normalized_prices_bt, label=f"{config['ticker']} Norm Price (Custom Period)", color=plot_config_main['price_line_color'], linewidth=1.0, alpha=0.7, zorder=1)
                                ax_bt.plot(backtest_dates, backtest_portfolio_values, label=f'GA Strategy (Custom Period)', color=plot_config_main['strategy_line_color'], linewidth=2.0, zorder=3)

                                if plot_config_main['plot_bbands']:
                                    try:
                                        bbands_bt_plot = ta.bbands(backtest_stock_df['Close'], length=int(chosen_bb_l_ga), std=float(chosen_bb_s_ga))
                                        bbl_col_bt = next((col for col in bbands_bt_plot.columns if 'BBL' in col), None)
                                        bbm_col_bt = next((col for col in bbands_bt_plot.columns if 'BBM' in col), None)
                                        bbu_col_bt = next((col for col in bbands_bt_plot.columns if 'BBU' in col), None)
                                        if bbl_col_bt and bbm_col_bt and bbu_col_bt:
                                            norm_bbl_bt = bbands_bt_plot[bbl_col_bt]/initial_price_bt; norm_bbm_bt = bbands_bt_plot[bbm_col_bt]/initial_price_bt; norm_bbu_bt = bbands_bt_plot[bbu_col_bt]/initial_price_bt
                                            bb_label_bt = f'BB({chosen_bb_l_ga},{chosen_bb_s_ga})'
                                            ax_bt.plot(backtest_dates, norm_bbm_bt, label=f'{bb_label_bt} Mid', color=plot_config_main['bb_line_color'], linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
                                            ax_bt.plot(backtest_dates, norm_bbu_bt, label=f'{bb_label_bt} Upper/Lower', color=plot_config_main['bb_line_color'], linestyle=':', linewidth=0.8, alpha=0.7, zorder=2)
                                            ax_bt.plot(backtest_dates, norm_bbl_bt, color=plot_config_main['bb_line_color'], linestyle=':', linewidth=0.8, alpha=0.7, zorder=2)
                                            ax_bt.fill_between(backtest_dates, norm_bbl_bt, norm_bbu_bt, color=plot_config_main['bb_fill_color'], alpha=0.1, zorder=1)
                                    except Exception as plot_bb_e_bt: print(f"Could not plot Bollinger Bands for custom period (ignored): {plot_bb_e_bt}")

                                if plot_config_main['plot_mas']:
                                    try:
                                        ma_s_bt_plot = backtest_stock_df['Close'].rolling(window=strategy_config_main['ma_short_period']).mean() / initial_price_bt
                                        ma_l_bt_plot = backtest_stock_df['Close'].rolling(window=strategy_config_main['ma_long_period']).mean() / initial_price_bt
                                        ax_bt.plot(backtest_dates, ma_s_bt_plot, label=f'MA({strategy_config_main["ma_short_period"]})', color=plot_config_main['ma_short_color'], linestyle='-.', linewidth=0.7, alpha=0.8, zorder=2)
                                        ax_bt.plot(backtest_dates, ma_l_bt_plot, label=f'MA({strategy_config_main["ma_long_period"]})', color=plot_config_main['ma_long_color'], linestyle='-.', linewidth=0.7, alpha=0.8, zorder=2)
                                    except Exception as plot_ma_e_bt: print(f"Could not plot MAs for custom period (ignored): {plot_ma_e_bt}")


                                if backtest_buy_signals:
                                    buy_dates_bt, buy_prices_bt, _ = zip(*backtest_buy_signals)
                                    ax_bt.scatter(buy_dates_bt, np.array(buy_prices_bt,dtype=float)/initial_price_bt, label='Buy (Custom)', marker='^', color='cyan', s=120, edgecolors='black', zorder=5)
                                if backtest_sell_signals:
                                    sell_dates_bt, sell_prices_bt, _ = zip(*backtest_sell_signals)
                                    ax_bt.scatter(sell_dates_bt, np.array(sell_prices_bt,dtype=float)/initial_price_bt, label='Sell (Custom)', marker='v', color='magenta', s=120, edgecolors='black', zorder=5)

                                title_bt_plot = (f"{config['ticker']} - GA Strategy Backtest on Custom Period: {custom_start_date} to {custom_end_date}\n"
                                                 f"Using GA Params: RSI_P={chosen_rsi_p_ga}, VIX_MA_P={chosen_vix_ma_p_ga}, BB=({chosen_bb_l_ga},{chosen_bb_s_ga}), VIX_T={bp_ga[2]}, ADX_T={bp_ga[8]}\n"
                                                 f"BuyE={bp_ga[0]}, ExitR={bp_ga[1]} | HVEntry:{hv_entry_ga}, LVStrat:{lv_strat_ga}, LVExitMA_Choice:{bp_ga[3]}")
                                ax_bt.set_title(title_bt_plot, fontsize=9)
                                ax_bt.legend(loc='upper left', fontsize=7)
                                fig_bt.tight_layout()
                                plt.show()
                        else: # show_plot is False for custom period
                             print("Plotting for custom period is disabled via plot_config_main['show_plot'].")
                    else: # backtest_indicators_ready is False
                        print("Error: Indicator pre-calculation failed for custom backtest period.")
                else: # backtest data loading failed
                    print("Error: Could not load data for custom backtest period.")
            else: # overall_best_params_for_ga is None
                print("\nError: GA optimization did not yield any best parameters to use for custom backtest.")
        else: # ga_indicators_ready is False
            print("\nError: Indicator pre-calculation failed for GA phase. Cannot proceed.")
    else: # GA phase data loading failed
        print("\nError: Data loading failed for GA phase. Program terminated.")

    print("\nmodel_train2_final_v3.py standalone test execution finished.")