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
from datetime import datetime as dt_datetime, timedelta

# --- Settings ---
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['axes.unicode_minus'] = False

# --- Data Loading Functions ---
def parse_week_string(week_str):
    try:
        if '-' in week_str and len(week_str.split('-')[0].split('/')) == 3 :
            year_part = week_str.split('/')[0]
            start_date_str = week_str.split('-')[0]
            end_date_day_month_str = week_str.split('-')[1]
            if len(end_date_day_month_str.split('/')) == 2:
                end_date_str = f"{year_part}/{end_date_day_month_str}"
            else:
                end_date_str = end_date_day_month_str
            start_date = pd.to_datetime(start_date_str, format='%Y/%m/%d')
            end_date = pd.to_datetime(end_date_str, format='%Y/%m/%d')
            return start_date, end_date
        elif '/' in week_str and len(week_str.split('/')) == 2 and week_str.split('/')[1].isdigit():
            year, week_num_str = week_str.split('/')
            week_num = int(week_num_str)
            start_date = pd.to_datetime(f'{year}-W{week_num-1}-1', format='%Y-W%W-%w')
            end_date = start_date + timedelta(days=6)
            return start_date, end_date
        else:
            parts = week_str.split('/')
            if len(parts) == 3:
                single_date = pd.to_datetime(week_str, format='%Y/%m/%d')
                return single_date, single_date
            else:
                # print(f"Warning: Unrecognized week string format: {week_str}") # Keep this quiet for now
                return None, None
    except Exception:
        # print(f"Error parsing week string '{week_str}': {e}") # Keep this quiet for now
        return None, None

def load_sentiment_data(csv_filepath, verbose=False):
    try:
        sentiment_df = pd.read_csv(csv_filepath, encoding='utf-8-sig')
        if verbose: print(f"Loaded sentiment data with columns: {sentiment_df.columns.tolist()}")
        sentiment_df.rename(columns={'年/週': 'WeekString', '情緒分數': 'SentimentScore'}, inplace=True, errors='ignore')
        if 'WeekString' not in sentiment_df.columns or 'SentimentScore' not in sentiment_df.columns:
            print("Error: Sentiment CSV must contain '年/週' (or 'WeekString') and '情緒分數' (or 'SentimentScore') columns.")
            return None

        daily_sentiments = []
        for _, row in sentiment_df.iterrows():
            week_str = str(row['WeekString']).strip()
            score = row['SentimentScore']
            if pd.isna(score):
                if verbose: print(f"Skipping row with NaN score for week: {week_str}")
                continue
            start_date, end_date = parse_week_string(week_str)
            if start_date and end_date:
                current_date = start_date
                while current_date <= end_date:
                    daily_sentiments.append({'Date': current_date, 'SentimentScore': float(score)})
                    current_date += timedelta(days=1)
            # elif verbose: # Reduce noise
            #     print(f"Could not parse week string: {week_str}")
        if not daily_sentiments:
            print("Error: No daily sentiment data could be generated from CSV.")
            return None
        daily_sentiment_df = pd.DataFrame(daily_sentiments)
        daily_sentiment_df['Date'] = pd.to_datetime(daily_sentiment_df['Date'])
        daily_sentiment_df = daily_sentiment_df.set_index('Date')
        daily_sentiment_df = daily_sentiment_df[~daily_sentiment_df.index.duplicated(keep='first')]
        if verbose: print(f"Processed daily sentiment data: {daily_sentiment_df.shape[0]} entries.")
        return daily_sentiment_df['SentimentScore']
    except Exception as e:
        print(f"Error loading or processing sentiment data from {csv_filepath}: {e}"); traceback.print_exc()
        return None

def load_stock_and_sentiment_data(ticker, sentiment_csv_path, start_date=None, end_date=None, verbose=False):
    if verbose: print(f"Attempting to load data for {ticker} and sentiment from {start_date} to {end_date}...")
    try:
        stock_data_raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if stock_data_raw is None or stock_data_raw.empty:
            if verbose: print(f"Error: yfinance.download returned None or empty DataFrame for {ticker}.")
            return None, None, None, None

        # Ensure stock_data_raw.columns is simplified if it's MultiIndex from yf.download for single ticker
        if isinstance(stock_data_raw.columns, pd.MultiIndex):
             # This case should ideally not happen for single ticker if yf.download behaves as expected
             # but if it does, we might need to select the ticker level if present, or assume it's already flat
             if ticker in stock_data_raw.columns.get_level_values(1): # Check if ticker name is in the second level
                stock_data_raw = stock_data_raw.xs(ticker, level=1, axis=1)
             elif len(stock_data_raw.columns.levels) > 1 : # If still multi-level but not by ticker name (unlikely for single ticker)
                stock_data_raw.columns = stock_data_raw.columns.droplevel(0) # Try dropping the top level

        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        missing_essential_cols = [col for col in ['Close', 'High', 'Low'] if col not in stock_data_raw.columns]
        if missing_essential_cols:
            if verbose: print(f"Error: Missing essential columns {missing_essential_cols} for {ticker}.")
            return None, None, None, None
        for col in required_cols:
             if col not in stock_data_raw.columns:
                 stock_data_raw[col] = np.nan
                 if verbose: print(f"Warning: Stock data for {ticker} missing '{col}', filled with NaN.")

        stock_df_simplified = stock_data_raw[required_cols].copy()
        if not isinstance(stock_df_simplified.index, pd.DatetimeIndex): # Ensure index is DatetimeIndex
            stock_df_simplified.index = pd.to_datetime(stock_df_simplified.index)

    except Exception as e:
        print(f"\nError during stock data loading for {ticker}: {type(e).__name__}: {e}"); traceback.print_exc()
        return None, None, None, None

    daily_sentiment_series = load_sentiment_data(sentiment_csv_path, verbose=verbose)
    if daily_sentiment_series is None:
        print(f"Error: Failed to load or process sentiment data for {ticker}.")
        return None, None, None, None

    if not isinstance(daily_sentiment_series.index, pd.DatetimeIndex):
        daily_sentiment_series.index = pd.to_datetime(daily_sentiment_series.index)
    
    # --- CRITICAL FIX for JOIN ---
    # Ensure both DataFrames to be joined have a simple DatetimeIndex (not MultiIndex)
    # stock_df_simplified should already have a simple DatetimeIndex
    # daily_sentiment_series is a Series with a DatetimeIndex

    # Pandas join on Series with DataFrame should work if indices are compatible
    # Let's try merging on index explicitly to be sure
    try:
        aligned_data = stock_df_simplified.merge(daily_sentiment_series.rename('SentimentScore'), # Rename series for merge
                                                 left_index=True,
                                                 right_index=True,
                                                 how='left') # Keep all stock dates, fill sentiment
    except Exception as e_merge:
        print(f"Error during explicit merge of stock and sentiment data: {e_merge}")
        traceback.print_exc()
        # Fallback or debug: print index details
        if verbose:
            print("Stock Index Info:", stock_df_simplified.index)
            print("Sentiment Index Info:", daily_sentiment_series.index)
        return None, None, None, None


    aligned_data['SentimentScore'] = aligned_data['SentimentScore'].ffill().bfill()
    numeric_stock_cols = stock_df_simplified.select_dtypes(include=np.number).columns
    for col in numeric_stock_cols:
        if col in aligned_data.columns and aligned_data[col].isnull().any():
             aligned_data[col] = aligned_data[col].ffill().bfill()
    aligned_data.dropna(subset=required_cols + ['SentimentScore'], inplace=True)

    if aligned_data.empty:
        if verbose: print(f"Error: No common dates or data left after aligning stock and sentiment for {ticker}.")
        return None, None, None, None

    final_stock_df = aligned_data[required_cols].copy()
    final_sentiment_series = aligned_data['SentimentScore'].copy()
    prices = final_stock_df['Close'].tolist()
    dates = final_stock_df.index.tolist()

    if len(final_sentiment_series) != len(prices):
        if verbose: print(f"Warning: Length mismatch after alignment for {ticker}. Sentiment: {len(final_sentiment_series)}, Prices: {len(prices)}. Reindexing sentiment.")
        final_sentiment_series = final_sentiment_series.reindex(final_stock_df.index).ffill().bfill()
        if final_sentiment_series.isnull().any():
             if verbose: print(f"Error: Sentiment series still has NaNs after reindexing for {ticker}.")
             return None, None, None, None

    print(f"Successfully loaded and aligned {len(prices)} data points for {ticker} with sentiment from {start_date} to {end_date}.")
    return prices, dates, final_stock_df, final_sentiment_series


# --- Indicator Pre-calculation (Modified for Sentiment MA) ---
def precompute_indicators_with_sentiment(stock_df, sentiment_series, strategy_config, verbose=False):
    precalculated_rsi = {}
    precalculated_sentiment_ma = {}
    precalculated_bbl = {}
    precalculated_bbm = {}
    precalculated_fixed = {}
    indicators_ready = True
    if verbose: print(f"  Starting indicator pre-calculation with sentiment (Stock DF shape: {stock_df.shape}, Sentiment Series len: {len(sentiment_series)})...")

    try:
        for rsi_p in strategy_config['rsi_period_options']:
            rsi_values = ta.rsi(stock_df['Close'], length=rsi_p)
            if rsi_values is None or rsi_values.isnull().all():
                if verbose: print(f"    Error: RSI calculation failed for period {rsi_p}")
                indicators_ready = False; break
            precalculated_rsi[rsi_p] = rsi_values.tolist()
        if not indicators_ready: return {}, False
        if verbose: print(f"    RSI ({len(precalculated_rsi)} variants) OK.")

        for sent_ma_p in strategy_config['sentiment_ma_period_options']:
            sent_ma_values = sentiment_series.rolling(window=sent_ma_p).mean()
            if sent_ma_values is None or sent_ma_values.isnull().all():
                precalculated_sentiment_ma[sent_ma_p] = [np.nan] * len(sentiment_series)
                if verbose: print(f"    Warning: Sentiment MA (period {sent_ma_p}) calculation failed, filled with NaNs.")
            else:
                precalculated_sentiment_ma[sent_ma_p] = sent_ma_values.tolist()
        if verbose: print(f"    Sentiment MA ({len(precalculated_sentiment_ma)} variants) OK.")

        for bb_l in strategy_config['bb_length_options']:
            for bb_s in strategy_config['bb_std_options']:
                bbands = ta.bbands(stock_df['Close'], length=bb_l, std=bb_s)
                bbl_col_name = next((col for col in bbands.columns if 'BBL' in col), None)
                bbm_col_name = next((col for col in bbands.columns if 'BBM' in col), None)
                if bbands is None or not bbl_col_name or not bbm_col_name or \
                   bbands[bbl_col_name].isnull().all() or bbands[bbm_col_name].isnull().all():
                    if verbose: print(f"    Error: Bollinger Bands (BBL/BBM) calculation failed for L={bb_l}, S={bb_s}")
                    indicators_ready = False; break
                precalculated_bbl[(bb_l, bb_s)] = bbands[bbl_col_name].tolist()
                precalculated_bbm[(bb_l, bb_s)] = bbands[bbm_col_name].tolist()
            if not indicators_ready: break
        if not indicators_ready: return {}, False
        if verbose: print(f"    Bollinger Bands (BBL/BBM, {len(precalculated_bbl)} variants) OK.")

        adx_df = ta.adx(stock_df['High'], stock_df['Low'], stock_df['Close'], length=strategy_config['adx_period'])
        adx_col_name = next((col for col in adx_df.columns if 'ADX' in col), None)
        if adx_df is None or not adx_col_name or adx_df[adx_col_name].isnull().all():
            if verbose: print(f"    Error: ADX calculation failed"); indicators_ready = False
        else:
            precalculated_fixed['adx_list'] = adx_df[adx_col_name].tolist()
            if verbose: print("    ADX OK.")

        if indicators_ready:
            ma_short = stock_df['Close'].rolling(window=strategy_config['ma_short_period']).mean()
            ma_long = stock_df['Close'].rolling(window=strategy_config['ma_long_period']).mean()
            if ma_short is None or ma_long is None or ma_short.isnull().all() or ma_long.isnull().all():
                if verbose: print(f"    Error: Moving Averages calculation failed"); indicators_ready = False
            else:
                precalculated_fixed['ma_short_list'] = ma_short.tolist()
                precalculated_fixed['ma_long_list'] = ma_long.tolist()
                if verbose: print("    MAs OK.")

        if indicators_ready:
            if verbose: print("  All required indicator pre-calculations finished for sentiment-based strategy.")
            return {'rsi': precalculated_rsi, 'sentiment_ma': precalculated_sentiment_ma,
                    'bbl': precalculated_bbl, 'bbm': precalculated_bbm,
                    'fixed': precalculated_fixed}, True
        else:
            if verbose: print("  Error during required indicator pre-calculation for sentiment-based strategy.")
            return {}, False
    except Exception as e:
        if verbose: print(f"  Unexpected error during indicator pre-calculation: {type(e).__name__}: {e}"); traceback.print_exc()
        return {}, False

# --- Numba Core Strategy (Modified for Sentiment) ---
@numba.jit(nopython=True)
def run_strategy_numba_core_sentiment(rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold,
                                      sentiment_threshold_risk_on,
                                      low_vol_exit_strategy, high_vol_entry_choice,
                                      commission_rate,
                                      prices_arr,
                                      rsi_arr, bbl_arr, bbm_arr, adx_arr,
                                      sentiment_ma_arr,
                                      ma_short_arr, ma_long_arr,
                                      start_trading_iloc):
    T = len(prices_arr); portfolio_values_arr = np.full(T, 1.0, dtype=np.float64)
    max_signals = T // 2 + 1
    buy_signal_indices = np.full(max_signals, -1, dtype=np.int64); buy_signal_prices = np.full(max_signals, np.nan, dtype=np.float64); buy_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    sell_signal_indices = np.full(max_signals, -1, dtype=np.int64); sell_signal_prices = np.full(max_signals, np.nan, dtype=np.float64); sell_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    buy_count = 0; sell_count = 0; cash = 1.0; stock = 0.0; position = 0; last_valid_portfolio_value = 1.0
    rsi_crossed_exit_level_after_buy = False
    risk_off_entry_type = -1

    start_trading_iloc = max(1, start_trading_iloc)
    if start_trading_iloc >= T:
        portfolio_values_arr[:] = 1.0
        return portfolio_values_arr, buy_signal_indices[:0], buy_signal_prices[:0], buy_signal_rsis[:0], \
               sell_signal_indices[:0], sell_signal_prices[:0], sell_signal_rsis[:0]
    portfolio_values_arr[:start_trading_iloc] = 1.0

    for i in range(start_trading_iloc, T):
        current_price = prices_arr[i]; rsi_i, rsi_prev = rsi_arr[i], rsi_arr[i-1]; bbl_i = bbl_arr[i]; bbm_i = bbm_arr[i]; adx_i = adx_arr[i]
        sentiment_ma_i = sentiment_ma_arr[i]
        ma_short_i, ma_long_i = ma_short_arr[i], ma_long_arr[i]; ma_short_prev, ma_long_prev = ma_short_arr[i-1], ma_long_arr[i-1]

        required_values = (rsi_i, rsi_prev, current_price, bbl_i, bbm_i, adx_i, sentiment_ma_i, ma_short_i, ma_long_i, ma_short_prev, ma_long_prev)
        is_valid = True
        for val in required_values:
            if not np.isfinite(val): is_valid = False; break
        if not is_valid:
            current_portfolio_value = cash if position == 0 else (stock * current_price if np.isfinite(current_price) else np.nan)
            portfolio_values_arr[i] = last_valid_portfolio_value if np.isnan(current_portfolio_value) else current_portfolio_value
            if not np.isnan(current_portfolio_value): last_valid_portfolio_value = current_portfolio_value
            continue

        if position == 0:
            is_risk_on_sentiment = sentiment_ma_i >= sentiment_threshold_risk_on
            buy_condition = False; entry_type_if_bought_risk_off = -1
            if not is_risk_on_sentiment:
                if high_vol_entry_choice == 0:
                    if (current_price <= bbl_i) and (rsi_i < rsi_buy_entry_threshold):
                        buy_condition = True; entry_type_if_bought_risk_off = 0
                else:
                    if (current_price <= bbl_i) and (adx_i > adx_threshold):
                        buy_condition = True; entry_type_if_bought_risk_off = 1
            else:
                if (ma_short_prev < ma_long_prev) and (ma_short_i >= ma_long_i):
                    buy_condition = True; entry_type_if_bought_risk_off = -1
            if buy_condition and current_price > 1e-9:
                cost = cash * commission_rate; amount_to_invest = cash - cost
                if amount_to_invest > 0:
                    stock = amount_to_invest / current_price; cash = 0.0; position = 1
                    rsi_crossed_exit_level_after_buy = False
                    risk_off_entry_type = entry_type_if_bought_risk_off
                    if buy_count < max_signals: buy_signal_indices[buy_count] = i; buy_signal_prices[buy_count] = current_price; buy_signal_rsis[buy_count] = rsi_i; buy_count += 1
        elif position == 1:
            sell_condition = False
            if risk_off_entry_type == 0:
                if rsi_i >= rsi_exit_threshold: rsi_crossed_exit_level_after_buy = True
                if rsi_crossed_exit_level_after_buy and rsi_i < rsi_exit_threshold: sell_condition = True
            elif risk_off_entry_type == 1: sell_condition = (current_price >= bbm_i)
            elif risk_off_entry_type == -1:
                if low_vol_exit_strategy == 0: sell_condition = (current_price < ma_short_i)
                else: sell_condition = (ma_short_prev > ma_long_prev) and (ma_short_i <= ma_long_i)
            if sell_condition:
                proceeds = stock * current_price; cost = proceeds * commission_rate; cash = proceeds - cost
                stock = 0.0; position = 0
                rsi_crossed_exit_level_after_buy = False; risk_off_entry_type = -1
                if sell_count < max_signals: sell_signal_indices[sell_count] = i; sell_signal_prices[sell_count] = current_price; sell_signal_rsis[sell_count] = rsi_i; sell_count += 1
        current_stock_value = stock * current_price if position == 1 else 0.0
        current_portfolio_value = cash + current_stock_value
        portfolio_values_arr[i] = last_valid_portfolio_value if np.isnan(current_portfolio_value) else current_portfolio_value
        if not np.isnan(current_portfolio_value): last_valid_portfolio_value = current_portfolio_value
    if T > 0 and np.isnan(portfolio_values_arr[-1]): portfolio_values_arr[-1] = last_valid_portfolio_value
    return portfolio_values_arr, buy_signal_indices[:buy_count], buy_signal_prices[:buy_count], buy_signal_rsis[:buy_count], sell_signal_indices[:sell_count], sell_signal_prices[:sell_count], sell_signal_rsis[:sell_count]

# --- Wrapper function (Modified for Sentiment) ---
def run_strategy_sentiment(rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold,
                           sentiment_threshold_risk_on,
                           low_vol_exit_strategy, high_vol_entry_choice,
                           commission_rate,
                           prices, dates,
                           rsi_list, bbl_list, bbm_list, adx_list,
                           sentiment_ma_list,
                           ma_short_list, ma_long_list):
    T = len(prices)
    if T == 0: return [1.0], [], []
    prices_arr = np.array(prices, dtype=np.float64); rsi_arr = np.array(rsi_list, dtype=np.float64)
    bbl_arr = np.array(bbl_list, dtype=np.float64); bbm_arr = np.array(bbm_list, dtype=np.float64)
    adx_arr = np.array(adx_list, dtype=np.float64); sentiment_ma_arr = np.array(sentiment_ma_list, dtype=np.float64)
    ma_short_arr = np.array(ma_short_list, dtype=np.float64); ma_long_arr = np.array(ma_long_list, dtype=np.float64)

    def get_first_valid_iloc(indicator_arr):
        valid_indices = np.where(np.isfinite(indicator_arr))[0]
        return valid_indices[0] if len(valid_indices) > 0 else T
    start_iloc_rsi = get_first_valid_iloc(rsi_arr) if len(rsi_arr) > 0 else T
    start_iloc_bbl = get_first_valid_iloc(bbl_arr) if len(bbl_arr) > 0 else T
    start_iloc_bbm = get_first_valid_iloc(bbm_arr) if len(bbm_arr) > 0 else T
    start_iloc_adx = get_first_valid_iloc(adx_arr) if len(adx_arr) > 0 else T
    start_iloc_sentiment_ma = get_first_valid_iloc(sentiment_ma_arr) if len(sentiment_ma_arr) > 0 else T
    start_iloc_ma_short = get_first_valid_iloc(ma_short_arr) if len(ma_short_arr) > 0 else T
    start_iloc_ma_long = get_first_valid_iloc(ma_long_arr) if len(ma_long_arr) > 0 else T
    start_trading_iloc = max(start_iloc_rsi, start_iloc_bbl, start_iloc_bbm, start_iloc_adx, start_iloc_sentiment_ma, start_iloc_ma_short, start_iloc_ma_long) + 1
    if start_trading_iloc >= T: return [1.0] * T, [], []
    start_trading_iloc = max(start_trading_iloc, 1)

    portfolio_values_arr, buy_indices, buy_prices, buy_rsis, sell_indices, sell_prices, sell_rsis = \
        run_strategy_numba_core_sentiment(
            float(rsi_buy_entry_threshold), float(rsi_exit_threshold), float(adx_threshold),
            float(sentiment_threshold_risk_on),
            int(low_vol_exit_strategy), int(high_vol_entry_choice),
            float(commission_rate),
            prices_arr, rsi_arr, bbl_arr, bbm_arr, adx_arr,
            sentiment_ma_arr,
            ma_short_arr, ma_long_arr,
            start_trading_iloc
        )
    buy_signals = []; sell_signals = []
    for idx, price, rsi_val in zip(buy_indices, buy_prices, buy_rsis):
        if idx != -1 and idx < len(dates): buy_signals.append((dates[idx], price, rsi_val))
    for idx, price, rsi_val in zip(sell_indices, sell_prices, sell_rsis):
         if idx != -1 and idx < len(dates): sell_signals.append((dates[idx], price, rsi_val))
    return portfolio_values_arr.tolist(), buy_signals, sell_signals

# --- Genetic Algorithm (Modified for Sentiment) ---
def genetic_algorithm_with_elitism_sentiment(prices, dates,
                                             precalculated_indicators,
                                             ga_params):
    generations = ga_params['generations']; population_size = ga_params['population_size']; crossover_rate = ga_params['crossover_rate']
    mutation_rate = ga_params['mutation_rate']; elitism_size = ga_params['elitism_size']; tournament_size = ga_params['tournament_size']
    mutation_amount_range = ga_params['mutation_amount_range']
    show_ga_generations = ga_params.get('show_process', False)

    rsi_threshold_range = ga_params['rsi_threshold_range']
    sentiment_threshold_range = ga_params['sentiment_threshold_range']
    adx_threshold_range = ga_params['adx_threshold_range']
    rsi_period_options = ga_params['rsi_period_options']; num_rsi_options = len(rsi_period_options)
    sentiment_ma_period_options = ga_params['sentiment_ma_period_options']; num_sentiment_ma_options = len(sentiment_ma_period_options)
    bb_length_options = ga_params['bb_length_options']; num_bb_len_options = len(bb_length_options)
    bb_std_options = ga_params['bb_std_options']; num_bb_std_options = len(bb_std_options)
    commission_rate = ga_params['commission_rate']
    sentiment_mutation_amount_range = ga_params.get('sentiment_mutation_amount_range', mutation_amount_range)
    adx_mutation_amount_range = ga_params.get('adx_mutation_amount_range', mutation_amount_range)

    fixed_inds = precalculated_indicators.get('fixed', {})
    adx_list = fixed_inds.get('adx_list')
    ma_short_list = fixed_inds.get('ma_short_list'); ma_long_list = fixed_inds.get('ma_long_list')
    if any(lst is None for lst in [adx_list, ma_short_list, ma_long_list]):
        if show_ga_generations: print("Error: Missing ADX or MA lists in precalculated_indicators['fixed'].")
        return None, 0

    T = len(prices)
    if T < 2:
        if show_ga_generations: print(f"Error: Data length {T} too short.")
        return None, 0

    population = []; attempts, max_attempts = 0, population_size * 200
    min_buy, max_buy, min_exit, max_exit = rsi_threshold_range
    min_sent_thr, max_sent_thr = sentiment_threshold_range
    min_adx, max_adx = adx_threshold_range

    while len(population) < population_size and attempts < max_attempts:
        buy_entry_thr = random.randint(min_buy, max_buy); exit_thr = random.randint(max(buy_entry_thr + 1, min_exit), max_exit)
        sentiment_thr_val = random.randint(min_sent_thr, max_sent_thr)
        low_vol_exit = random.choice([0, 1]); rsi_p_choice = random.randint(0, num_rsi_options - 1)
        sentiment_ma_p_choice = random.randint(0, num_sentiment_ma_options - 1)
        bb_len_choice = random.randint(0, num_bb_len_options - 1); bb_std_choice = random.randint(0, num_bb_std_options - 1)
        adx_thr_val = random.randint(min_adx, max_adx); hv_entry_choice = random.choice([0, 1])
        gene = [buy_entry_thr, exit_thr, sentiment_thr_val, low_vol_exit, rsi_p_choice, sentiment_ma_p_choice, bb_len_choice, bb_std_choice, adx_thr_val, hv_entry_choice]
        if (0<gene[0]<gene[1]<100 and min_buy<=gene[0]<=max_buy and min_exit<=gene[1]<=max_exit and
            min_sent_thr<=gene[2]<=max_sent_thr and gene[3] in [0,1] and
            0<=gene[4]<num_rsi_options and 0<=gene[5]<num_sentiment_ma_options and
            0<=gene[6]<num_bb_len_options and 0<=gene[7]<num_bb_std_options and
            min_adx<=gene[8]<=max_adx and gene[9] in [0,1]):
             population.append(gene)
        attempts += 1
    if not population or len(population) < population_size :
        if show_ga_generations: print(f"Error: Could not generate sufficient initial population ({len(population)}/{population_size}).")
        return None, 0
    best_gene_overall = population[0][:]; best_fitness_overall = -float('inf')

    for generation in range(generations):
        fitness = []
        for gene in population:
            try:
                chosen_rsi_period = rsi_period_options[gene[4]]
                chosen_sentiment_ma_period = sentiment_ma_period_options[gene[5]]
                chosen_bb_length = bb_length_options[gene[6]]
                chosen_bb_std = bb_std_options[gene[7]]
                rsi_list = precalculated_indicators['rsi'][chosen_rsi_period]
                sentiment_ma_list = precalculated_indicators['sentiment_ma'][chosen_sentiment_ma_period]
                bbl_list = precalculated_indicators['bbl'][(chosen_bb_length, chosen_bb_std)]
                bbm_list = precalculated_indicators['bbm'][(chosen_bb_length, chosen_bb_std)]
                portfolio_values, _, _ = run_strategy_sentiment(
                    gene[0], gene[1], gene[8], gene[2], gene[3], gene[9],
                    commission_rate, prices, dates,
                    rsi_list, bbl_list, bbm_list, adx_list, sentiment_ma_list, ma_short_list, ma_long_list
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
                bo_lv_exit_str = "Price<MA" if best_gene_overall[3] == 0 else "MACross"
                bo_rsi_p_str = rsi_period_options[best_gene_overall[4]]
                bo_sent_ma_p_str = sentiment_ma_period_options[best_gene_overall[5]]
                bo_bb_l_str = bb_length_options[best_gene_overall[6]]
                bo_bb_s_str = bb_std_options[best_gene_overall[7]]
                bo_hv_entry_str = "BB+RSI" if best_gene_overall[9] == 0 else "BB+ADX"
                overall_best_params_readable = (
                    f"BuyE={best_gene_overall[0]},ExitR={best_gene_overall[1]},SentT={best_gene_overall[2]},LVExit={bo_lv_exit_str},"
                    f"RSI_P={bo_rsi_p_str},SentMA_P={bo_sent_ma_p_str},BB_L={bo_bb_l_str},BB_S={bo_bb_s_str},"
                    f"ADX_T={best_gene_overall[8]},HVEntry={bo_hv_entry_str}"
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
                 crossover_point = random.randint(1, 9);
                 child1_new = p1[:crossover_point] + p2[crossover_point:]; child2_new = p2[:crossover_point] + p1[crossover_point:]
                 valid_c1 = (0<child1_new[0]<child1_new[1]<100 and min_buy<=child1_new[0]<=max_buy and min_exit<=child1_new[1]<=max_exit and min_sent_thr<=child1_new[2]<=max_sent_thr and child1_new[3] in [0,1] and 0<=child1_new[4]<num_rsi_options and 0<=child1_new[5]<num_sentiment_ma_options and 0<=child1_new[6]<num_bb_len_options and 0<=child1_new[7]<num_bb_std_options and min_adx<=child1_new[8]<=max_adx and child1_new[9] in [0,1])
                 valid_c2 = (0<child2_new[0]<child2_new[1]<100 and min_buy<=child2_new[0]<=max_buy and min_exit<=child2_new[1]<=max_exit and min_sent_thr<=child2_new[2]<=max_sent_thr and child2_new[3] in [0,1] and 0<=child2_new[4]<num_rsi_options and 0<=child2_new[5]<num_sentiment_ma_options and 0<=child2_new[6]<num_bb_len_options and 0<=child2_new[7]<num_bb_std_options and min_adx<=child2_new[8]<=max_adx and child2_new[9] in [0,1])
                 child1 = child1_new if valid_c1 else p1[:]; child2 = child2_new if valid_c2 else p2[:]
            offspring.append(child1); offspring.append(child2)
        if len(parent_indices) % 2 != 0: offspring.append(selected_parents[parent_indices[-1]][:])

        mut_min, mut_max = mutation_amount_range
        sent_mut_min_actual, sent_mut_max_actual = sentiment_mutation_amount_range
        adx_mut_min_actual, adx_mut_max_actual = adx_mutation_amount_range

        for i_offspring in range(len(offspring)):
            if random.random() < mutation_rate:
                gene_to_mutate = offspring[i_offspring]; original_gene = gene_to_mutate[:]; mutate_idx = random.randint(0, 9)
                if mutate_idx == 3: gene_to_mutate[3] = 1 - gene_to_mutate[3]
                elif mutate_idx == 4:
                    if num_rsi_options > 1:
                        new_choice = random.randint(0, num_rsi_options - 1)
                        while new_choice == original_gene[4]: new_choice = random.randint(0, num_rsi_options - 1)
                        gene_to_mutate[4] = new_choice
                elif mutate_idx == 5:
                    if num_sentiment_ma_options > 1:
                        new_choice = random.randint(0, num_sentiment_ma_options - 1)
                        while new_choice == original_gene[5]: new_choice = random.randint(0, num_sentiment_ma_options - 1)
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
                else:
                    if mutate_idx == 2: mutation_amount = random.randint(sent_mut_min_actual, sent_mut_max_actual); is_zero_range = (sent_mut_min_actual == 0 and sent_mut_max_actual == 0)
                    elif mutate_idx == 8: mutation_amount = random.randint(adx_mut_min_actual, adx_mut_max_actual); is_zero_range = (adx_mut_min_actual == 0 and adx_mut_max_actual == 0)
                    else: mutation_amount = random.randint(mut_min, mut_max); is_zero_range = (mut_min == 0 and mut_max == 0)
                    if mutation_amount == 0 and not is_zero_range:
                        while mutation_amount == 0:
                            if mutate_idx == 2: mutation_amount = random.randint(sent_mut_min_actual, sent_mut_max_actual)
                            elif mutate_idx == 8: mutation_amount = random.randint(adx_mut_min_actual, adx_mut_max_actual)
                            else: mutation_amount = random.randint(mut_min, mut_max)
                    gene_to_mutate[mutate_idx] += mutation_amount
                    gene_to_mutate[0] = max(min_buy, min(gene_to_mutate[0], max_buy))
                    gene_to_mutate[1] = max(gene_to_mutate[0] + 1, min_exit, min(gene_to_mutate[1], max_exit))
                    gene_to_mutate[0] = max(min_buy, min(gene_to_mutate[0], gene_to_mutate[1] - 1, max_buy))
                    gene_to_mutate[2] = max(min_sent_thr, min(gene_to_mutate[2], max_sent_thr))
                    gene_to_mutate[8] = max(min_adx, min(gene_to_mutate[8], max_adx))
                final_valid = (0<gene_to_mutate[0]<gene_to_mutate[1]<100 and min_buy<=gene_to_mutate[0]<=max_buy and min_exit<=gene_to_mutate[1]<=max_exit and min_sent_thr<=gene_to_mutate[2]<=max_sent_thr and gene_to_mutate[3] in [0,1] and 0<=gene_to_mutate[4]<num_rsi_options and 0<=gene_to_mutate[5]<num_sentiment_ma_options and 0<=gene_to_mutate[6]<num_bb_len_options and 0<=gene_to_mutate[7]<num_bb_std_options and min_adx<=gene_to_mutate[8]<=max_adx and gene_to_mutate[9] in [0,1])
                if not final_valid: offspring[i_offspring] = original_gene
        population = elites + offspring; population = population[:population_size]
    if best_fitness_overall == -float('inf'):
        if show_ga_generations: print("Error: GA finished without finding any valid solution.")
        return None, 0
    return best_gene_overall, best_fitness_overall

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Executing model_train2_sentiment_replace_vix.py as a standalone script...")

    sentiment_csv_file = '2021-2025每週新聞及情緒分析.csv'

    ga_train_config = {
        'ticker': 'TSLA',
        'start_date': '2022-01-01', 'end_date': '2023-06-30',
        'description': "GA Optimization with Sentiment (In-Sample)"
    }
    strategy_config_shared = {
        'rsi_period_options': [7, 14, 21],
        'sentiment_ma_period_options': [1,2, 4],
        'bb_length_options': [15, 20],
        'bb_std_options': [1.5, 2.0, 2.5],
        'adx_period': 14,
        'ma_short_period': 5, 'ma_long_period': 10,
        'commission_pct': 0.002,
    }
    ga_params_config = {
        'generations': 15, 'population_size': 40, 'crossover_rate': 0.7, 'mutation_rate': 0.25,
        'elitism_size': 3, 'tournament_size': 5, 'mutation_amount_range': (-5, 5),
        'sentiment_mutation_amount_range': (-3, 3),
        'adx_mutation_amount_range': (-2, 2),
        'show_process': False, # GA generation verbosity
        'rsi_threshold_range': (15, 40, 50, 80),
        'sentiment_threshold_range': (30, 70), # Example range for sentiment_threshold_risk_on
        'adx_threshold_range': (18, 35),
        'rsi_period_options': strategy_config_shared['rsi_period_options'],
        'sentiment_ma_period_options': strategy_config_shared['sentiment_ma_period_options'],
        'bb_length_options': strategy_config_shared['bb_length_options'],
        'bb_std_options': strategy_config_shared['bb_std_options'],
        'adx_period': strategy_config_shared['adx_period'],
        'ma_short_period': strategy_config_shared['ma_short_period'],
        'ma_long_period': strategy_config_shared['ma_long_period'],
        'commission_rate': strategy_config_shared['commission_pct'],
    }
    run_config_main = { 'num_runs': 100 }
    plot_config_shared = {
        'show_plot': True, 'figure_size': (16, 10), 'buy_marker_color': 'lime',
        'sell_marker_color': 'red', 'strategy_line_color': 'cyan', 'price_line_color': 'grey',
        'plot_bbands': True, 'bb_line_color': 'orange', 'bb_fill_color': 'moccasin',
        'plot_mas': True, 'ma_short_color': 'blue', 'ma_long_color': 'purple',
        'plot_sentiment_ma': True, 'sentiment_ma_color': 'magenta'
    }

    print(f"\n--- Phase 1: GA Optimization ({ga_train_config['description']}) ---")
    print(f"Period: {ga_train_config['start_date']} to {ga_train_config['end_date']} for Ticker: {ga_train_config['ticker']}")

    train_prices, train_dates, train_stock_df, train_sentiment_series = load_stock_and_sentiment_data(
        ga_train_config['ticker'], sentiment_csv_file,
        start_date=ga_train_config['start_date'], end_date=ga_train_config['end_date'],
        verbose=ga_params_config['show_process']
    )

    best_params_from_ga = None
    if train_prices and train_dates and train_stock_df is not None and train_sentiment_series is not None:
        if not ga_params_config['show_process']: print("  Pre-calculating indicators for GA training period...")
        train_precalculated_indicators, train_indicators_ready = precompute_indicators_with_sentiment(
            train_stock_df, train_sentiment_series, strategy_config_shared, verbose=ga_params_config['show_process']
        )

        if train_indicators_ready:
            if not ga_params_config['show_process']:
                print(f"  Starting GA optimization ({ga_params_config['generations']} generations, {ga_params_config['population_size']} population)...")
            ga_start_time = time.time()
            
            # For genetic_algorithm_with_elitism_sentiment, we pass the whole precalculated_indicators dict
            overall_best_fitness_for_ga_phase = -float('inf')
            best_params_for_ga_phase = None

            for run_idx in range(run_config_main['num_runs']):
                if ga_params_config['show_process']: print(f"--- GA Run {run_idx + 1}/{run_config_main['num_runs']} ---")
                
                current_best_params_run, current_fitness_run = genetic_algorithm_with_elitism_sentiment( # Call the new GA
                    train_prices, train_dates,
                    train_precalculated_indicators, # Pass the whole dict
                    ga_params=ga_params_config
                )
                if current_best_params_run and current_fitness_run > overall_best_fitness_for_ga_phase:
                    overall_best_fitness_for_ga_phase = current_fitness_run
                    best_params_for_ga_phase = current_best_params_run
                
                if not ga_params_config['show_process'] and current_best_params_run:
                     print(f"    GA Run {run_idx + 1} Best Value: {current_fitness_run:.4f}")


            ga_time_taken = time.time() - ga_start_time
            if not ga_params_config['show_process']: print(f"  GA optimization (all runs) finished in {ga_time_taken:.2f} seconds.")

            if best_params_for_ga_phase:
                best_params_from_ga = best_params_for_ga_phase
                print(f"  Best Fitness from GA (In-Sample, {run_config_main['num_runs']} runs): {overall_best_fitness_for_ga_phase:.4f}")
                bp_ga = best_params_from_ga
                chosen_rsi_p_str = strategy_config_shared['rsi_period_options'][bp_ga[4]]
                chosen_sent_ma_p_str = strategy_config_shared['sentiment_ma_period_options'][bp_ga[5]]
                chosen_bb_l_str = strategy_config_shared['bb_length_options'][bp_ga[6]]
                chosen_bb_s_str = strategy_config_shared['bb_std_options'][bp_ga[7]]
                hv_entry_desc = "BB+RSI" if bp_ga[9] == 0 else "BB+ADX"
                lv_exit_desc = "Price < MA_Short" if bp_ga[3] == 0 else "MA CrossDown"

                print(f"  Best GA Parameters (Sentiment-based):")
                print(f"    RSI Buy Entry: {bp_ga[0]}, RSI Exit Ref: {bp_ga[1]}")
                print(f"    Sentiment Threshold (Risk-On if MA > thr): {bp_ga[2]}")
                print(f"    Low Vol MA Exit Choice [{bp_ga[3]}]: {lv_exit_desc}")
                print(f"    RSI Period: {chosen_rsi_p_str} (Choice Code: {bp_ga[4]})")
                print(f"    Sentiment MA Period: {chosen_sent_ma_p_str} (Choice Code: {bp_ga[5]})")
                print(f"    BB Length: {chosen_bb_l_str} (Choice Code: {bp_ga[6]})")
                print(f"    BB StdDev: {chosen_bb_s_str} (Choice Code: {bp_ga[7]})")
                print(f"    ADX Threshold: {bp_ga[8]}")
                print(f"    Risk-Off (like High VIX) Entry Choice [{bp_ga[9]}]: {hv_entry_desc}")
                if not ga_params_config['show_process']: print("-" * 40)

                if plot_config_shared['show_plot']:
                    print(f"  Plotting GA Best Strategy on Training Data ({ga_train_config['start_date']} to {ga_train_config['end_date']})...")
                    
                    train_rsi_list_plot = train_precalculated_indicators['rsi'][chosen_rsi_p_str]
                    train_sent_ma_list_plot = train_precalculated_indicators['sentiment_ma'][chosen_sent_ma_p_str]
                    train_bbl_list_plot = train_precalculated_indicators['bbl'][(int(chosen_bb_l_str), float(chosen_bb_s_str))]
                    train_bbm_list_plot = train_precalculated_indicators['bbm'][(int(chosen_bb_l_str), float(chosen_bb_s_str))]
                    train_adx_list_plot = train_precalculated_indicators['fixed']['adx_list']
                    train_ma_short_list_plot = train_precalculated_indicators['fixed']['ma_short_list']
                    train_ma_long_list_plot = train_precalculated_indicators['fixed']['ma_long_list']

                    train_portfolio_values, train_buy_signals, train_sell_signals = run_strategy_sentiment(
                        bp_ga[0], bp_ga[1], bp_ga[8], bp_ga[2], bp_ga[3], bp_ga[9],
                        strategy_config_shared['commission_pct'],
                        train_prices, train_dates,
                        train_rsi_list_plot, train_bbl_list_plot, train_bbm_list_plot,
                        train_adx_list_plot, train_sent_ma_list_plot,
                        train_ma_short_list_plot, train_ma_long_list_plot
                    )
                    fig_train, ax_train1 = plt.subplots(figsize=plot_config_shared['figure_size'])
                    initial_price_train = train_prices[0] if train_prices and np.isfinite(train_prices[0]) and train_prices[0] > 1e-9 else 1.0
                    
                    color_price = 'tab:grey'
                    ax_train1.set_xlabel('Date', fontsize=12)
                    ax_train1.set_ylabel('Normalized Value (Start = 1)', color=color_price, fontsize=12)
                    ax_train1.plot(train_dates, np.array(train_prices, dtype=float)/initial_price_train, label=f"{ga_train_config['ticker']} Norm Price (Train)", color=color_price, alpha=0.7)
                    ax_train1.plot(train_dates, train_portfolio_values, label=f'GA Strategy (Train) - Fitness: {overall_best_fitness_for_ga_phase:.4f}', color=plot_config_shared['strategy_line_color'])
                    ax_train1.tick_params(axis='y', labelcolor=color_price)
                    if train_buy_signals: ax_train1.scatter([s[0] for s in train_buy_signals], np.array([s[1] for s in train_buy_signals],dtype=float)/initial_price_train, label='Buy', marker='^', color=plot_config_shared['buy_marker_color'], s=80, edgecolors='k', zorder=5)
                    if train_sell_signals: ax_train1.scatter([s[0] for s in train_sell_signals], np.array([s[1] for s in train_sell_signals],dtype=float)/initial_price_train, label='Sell', marker='v', color=plot_config_shared['sell_marker_color'], s=80, edgecolors='k', zorder=5)
                    
                    if plot_config_shared['plot_sentiment_ma']:
                        ax_train2 = ax_train1.twinx()
                        color_sentiment = plot_config_shared['sentiment_ma_color']
                        ax_train2.set_ylabel('Sentiment MA', color=color_sentiment, fontsize=12)
                        ax_train2.plot(train_dates, train_sent_ma_list_plot, color=color_sentiment, linestyle='--', label=f'Sentiment MA({chosen_sent_ma_p_str})', alpha=0.7)
                        ax_train2.tick_params(axis='y', labelcolor=color_sentiment)
                        ax_train2.axhline(y=bp_ga[2], color='gray', linestyle=':', linewidth=1, label=f'Sentiment Threshold ({bp_ga[2]})')

                    fig_train.suptitle(f"{ga_train_config['ticker']} - GA Optimized Strategy with Sentiment (Training: {ga_train_config['start_date']} to {ga_train_config['end_date']})", fontsize=10)
                    lines, labels = ax_train1.get_legend_handles_labels()
                    if plot_config_shared['plot_sentiment_ma']:
                        lines2, labels2 = ax_train2.get_legend_handles_labels()
                        ax_train2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=7)
                    else:
                         ax_train1.legend(lines, labels, loc='upper left', fontsize=7)
                    ax_train1.grid(True, alpha=0.3)
                    fig_train.tight_layout(rect=[0, 0, 1, 0.96])
                    plt.show()
            else:
                print("  GA optimization did not yield valid parameters for the training period.")
        else:
            print("  Indicator pre-calculation failed for GA training period.")
    else:
        print(f"  Data loading failed for GA training period ({ga_train_config['ticker']}).")


    if best_params_from_ga:
        print(f"\n--- Phase 2: Out-of-Sample Test with GA Best Parameters (Sentiment Strategy) ---")
        test_config_params = {
            'ticker': ga_train_config['ticker'],
            'start_date': '2023-07-01',
            'end_date': '2025-05-17',
            'description': "Out-of-Sample Test with Sentiment"
        }
        print(f"Period: {test_config_params['start_date']} to {test_config_params['end_date']} for Ticker: {test_config_params['ticker']}")

        test_prices, test_dates, test_stock_df, test_sentiment_series = load_stock_and_sentiment_data(
            test_config_params['ticker'], sentiment_csv_file,
            start_date=test_config_params['start_date'], end_date=test_config_params['end_date'],
            verbose=ga_params_config['show_process']
        )

        if test_prices and test_dates and test_stock_df is not None and test_sentiment_series is not None:
            if not ga_params_config['show_process']: print("  Pre-calculating indicators for OOS test period...")
            test_precalculated_indicators, test_indicators_ready = precompute_indicators_with_sentiment(
                test_stock_df, test_sentiment_series, strategy_config_shared, verbose=ga_params_config['show_process']
            )

            if test_indicators_ready:
                if not ga_params_config['show_process']: print("  Running strategy on OOS test period with GA best parameters...")
                bp_ga_final_test = best_params_from_ga
                
                chosen_rsi_p_test_str = strategy_config_shared['rsi_period_options'][bp_ga_final_test[4]]
                chosen_sent_ma_p_test_str = strategy_config_shared['sentiment_ma_period_options'][bp_ga_final_test[5]]
                chosen_bb_l_test_str = strategy_config_shared['bb_length_options'][bp_ga_final_test[6]]
                chosen_bb_s_test_str = strategy_config_shared['bb_std_options'][bp_ga_final_test[7]]

                test_rsi_list = test_precalculated_indicators['rsi'][chosen_rsi_p_test_str]
                test_sent_ma_list = test_precalculated_indicators['sentiment_ma'][chosen_sent_ma_p_test_str]
                test_bbl_list = test_precalculated_indicators['bbl'][(int(chosen_bb_l_test_str), float(chosen_bb_s_test_str))]
                test_bbm_list = test_precalculated_indicators['bbm'][(int(chosen_bb_l_test_str), float(chosen_bb_s_test_str))]
                test_adx_list = test_precalculated_indicators['fixed']['adx_list']
                test_ma_short_list = test_precalculated_indicators['fixed']['ma_short_list']
                test_ma_long_list = test_precalculated_indicators['fixed']['ma_long_list']

                oos_portfolio_values, oos_buy_signals, oos_sell_signals = run_strategy_sentiment(
                    bp_ga_final_test[0], bp_ga_final_test[1], bp_ga_final_test[8], bp_ga_final_test[2], 
                    bp_ga_final_test[3], bp_ga_final_test[9],
                    strategy_config_shared['commission_pct'],
                    test_prices, test_dates,
                    test_rsi_list, test_bbl_list, test_bbm_list,
                    test_adx_list, test_sent_ma_list,
                    test_ma_short_list, test_ma_long_list
                )

                if oos_portfolio_values:
                    final_oos_value = next((p for p in reversed(oos_portfolio_values) if np.isfinite(p)), 1.0)
                    print(f"  Out-of-Sample Final Portfolio Value: {final_oos_value:.4f}")
                    num_oos_trades = min(len(oos_buy_signals), len(oos_sell_signals))
                    winning_oos_trades = 0
                    if num_oos_trades > 0:
                        for i_oos_trade in range(num_oos_trades):
                            if oos_sell_signals[i_oos_trade][1] > oos_buy_signals[i_oos_trade][1]:
                                winning_oos_trades += 1
                        oos_win_rate = (winning_oos_trades / num_oos_trades) * 100 if num_oos_trades > 0 else 0.0
                        print(f"    Trades in OOS period: {num_oos_trades}, Wins: {winning_oos_trades}, Win Rate: {oos_win_rate:.2f}%")
                    else:
                        print("    No completed trades in OOS period.")

                    if plot_config_shared['show_plot']:
                        if not ga_params_config['show_process']: print(f"  Plotting GA Best Strategy on Test Data ({test_config_params['start_date']} to {test_config_params['end_date']})...")
                        fig_test, ax_test1 = plt.subplots(figsize=plot_config_shared['figure_size'])
                        initial_price_test = test_prices[0] if test_prices and np.isfinite(test_prices[0]) and test_prices[0] > 1e-9 else 1.0
                        
                        color_price_test = 'tab:grey'
                        ax_test1.set_xlabel('Date', fontsize=12)
                        ax_test1.set_ylabel('Normalized Value (Start = 1)', color=color_price_test, fontsize=12)
                        ax_test1.plot(test_dates, np.array(test_prices, dtype=float)/initial_price_test, label=f"{test_config_params['ticker']} Norm Price (Test)", color=color_price_test, alpha=0.7)
                        ax_test1.plot(test_dates, oos_portfolio_values, label=f'GA Strategy (Test) - Value: {final_oos_value:.4f}', color='purple')
                        ax_test1.tick_params(axis='y', labelcolor=color_price_test)
                        if oos_buy_signals: ax_test1.scatter([s[0] for s in oos_buy_signals], np.array([s[1] for s in oos_buy_signals],dtype=float)/initial_price_test, label='Buy (OOS)', marker='^', color='cyan', s=80, edgecolors='k', zorder=5)
                        if oos_sell_signals: ax_test1.scatter([s[0] for s in oos_sell_signals], np.array([s[1] for s in oos_sell_signals],dtype=float)/initial_price_test, label='Sell (OOS)', marker='v', color='magenta', s=80, edgecolors='k', zorder=5)
                        
                        if plot_config_shared['plot_sentiment_ma']:
                            ax_test2 = ax_test1.twinx()
                            color_sent_test = plot_config_shared['sentiment_ma_color']
                            ax_test2.set_ylabel('Sentiment MA (Test)', color=color_sent_test, fontsize=12)
                            ax_test2.plot(test_dates, test_sent_ma_list, color=color_sent_test, linestyle='--', label=f'Sentiment MA({chosen_sent_ma_p_test_str}) (Test)', alpha=0.7)
                            ax_test2.tick_params(axis='y', labelcolor=color_sent_test)
                            ax_test2.axhline(y=bp_ga_final_test[2], color='gray', linestyle=':', linewidth=1, label=f'Sentiment Threshold ({bp_ga_final_test[2]})')

                        hv_entry_desc_test_plot = "BB+RSI" if bp_ga_final_test[9] == 0 else "BB+ADX"
                        lv_exit_desc_test_plot = "Price < MA_Short" if bp_ga_final_test[3] == 0 else "MA CrossDown"
                        commission_info_plot = f"{strategy_config_shared['commission_pct']*100:.3f}%"

                        title_bt_plot = (
                            f"{test_config_params['ticker']} - GA Strategy (Sentiment) OOS Test: {test_config_params['start_date']} to {test_config_params['end_date']}\n"
                            f"Using GA Params: RSI_P={chosen_rsi_p_test_str}, SentMA_P={chosen_sent_ma_p_test_str}, BB=({chosen_bb_l_test_str},{chosen_bb_s_test_str}), SentT={bp_ga_final_test[2]}, ADX_T={bp_ga_final_test[8]}\n"
                            f"BuyE={bp_ga_final_test[0]}, ExitR={bp_ga_final_test[1]} | RiskOff Entry Choice[{bp_ga_final_test[9]}]: {hv_entry_desc_test_plot} | RiskOn MA Exit Choice[{bp_ga_final_test[3]}]: {lv_exit_desc_test_plot} | Comm: {commission_info_plot}"
                        )
                        fig_test.suptitle(title_bt_plot, fontsize=9)
                        lines1, labels1 = ax_test1.get_legend_handles_labels()
                        if plot_config_shared['plot_sentiment_ma']:
                            lines2, labels2 = ax_test2.get_legend_handles_labels()
                            ax_test2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)
                        else:
                            ax_test1.legend(lines1, labels1, loc='upper left', fontsize=7)
                        ax_test1.grid(True, alpha=0.3)
                        fig_test.tight_layout(rect=[0, 0, 1, 0.95])
                        plt.show()
                else:
                    print("  Could not run strategy on OOS test period (no portfolio values).")
            else:
                print("  Indicator pre-calculation failed for OOS test period.")
        else:
            print(f"  Data loading failed for OOS test period ({test_config_params['ticker']}).")
    else:
        print("\n--- Skipping Out-of-Sample Test: GA Optimization did not yield best parameters. ---")

    print("\nmodel_train2_sentiment_replace_vix.py standalone test execution finished.")