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
from datetime import datetime as dt_datetime, timedelta, timezone
import urllib.parse

# --- Settings ---
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['axes.unicode_minus'] = False

# --- Data Loading Functions ---
def parse_week_string_for_sentiment(week_str):
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
            else: return None, None
    except Exception: return None, None

def load_sentiment_data_for_unified(csv_filepath, verbose=False):
    try:
        sentiment_df = pd.read_csv(csv_filepath, encoding='utf-8-sig')
        sentiment_df.rename(columns={'年/週': 'WeekString', '情緒分數': 'SentimentScore'}, inplace=True, errors='ignore')
        if 'WeekString' not in sentiment_df.columns or 'SentimentScore' not in sentiment_df.columns:
            if verbose: print("Error: Sentiment CSV must contain '年/週' and '情緒分數' columns.")
            return None
        daily_sentiments = []
        for _, row in sentiment_df.iterrows():
            week_str = str(row['WeekString']).strip(); score = row['SentimentScore']
            if pd.isna(score): continue
            start_date, end_date = parse_week_string_for_sentiment(week_str)
            if start_date and end_date:
                current_date = start_date
                while current_date <= end_date:
                    daily_sentiments.append({'Date': current_date, 'SentimentScore': float(score)})
                    current_date += timedelta(days=1)
        if not daily_sentiments:
            if verbose: print("Warning: No daily sentiment data could be generated from CSV.")
            return None
        daily_sentiment_df = pd.DataFrame(daily_sentiments)
        daily_sentiment_df['Date'] = pd.to_datetime(daily_sentiment_df['Date'])
        daily_sentiment_df = daily_sentiment_df.set_index('Date')
        daily_sentiment_df = daily_sentiment_df[~daily_sentiment_df.index.duplicated(keep='first')]
        return daily_sentiment_df['SentimentScore']
    except Exception as e:
        if verbose: print(f"Error loading sentiment data: {e}");
        return None

def load_data_unified(ticker, vix_ticker, sentiment_csv_path, start_date=None, end_date=None, verbose=False):
    if verbose: print(f"Loading unified data for {ticker}, VIX:{vix_ticker}, Sentiment:{sentiment_csv_path} from {start_date} to {end_date}...")

    tickers_to_load = [ticker]
    if vix_ticker:
        tickers_to_load.append(vix_ticker)

    try:
        data_yf = yf.download(tickers_to_load, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if data_yf is None or data_yf.empty:
            if verbose: print(f"Error: yfinance.download returned empty for {tickers_to_load}.")
            return None, None, None, None, None

        if isinstance(data_yf.columns, pd.MultiIndex) and ticker in data_yf.columns.get_level_values(1):
            stock_data = data_yf.loc[:, pd.IndexSlice[:, ticker]]
            stock_data.columns = stock_data.columns.droplevel(1)
        elif not isinstance(data_yf.columns, pd.MultiIndex) and ticker in tickers_to_load and len(tickers_to_load) == 1 :
             stock_data = data_yf
        else:
            if verbose: print(f"Error: Could not extract stock data for {ticker} from yfinance download.")
            return None, None, None, None, None

        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        if not all(col in stock_data.columns for col in ['Close','High','Low']):
            if verbose: print(f"Error: Stock data for {ticker} missing essential price columns."); return None,None,None,None,None
        for col in required_cols:
            if col not in stock_data.columns: stock_data[col] = np.nan # Add missing optional columns as NaN
        stock_df_simplified = stock_data[required_cols].copy()
        if not isinstance(stock_df_simplified.index, pd.DatetimeIndex):
            stock_df_simplified.index = pd.to_datetime(stock_df_simplified.index)

        final_vix_series = None
        if vix_ticker:
            if isinstance(data_yf.columns, pd.MultiIndex) and vix_ticker in data_yf.columns.get_level_values(1):
                vix_data_slice = data_yf.loc[:, pd.IndexSlice['Close', vix_ticker]]
                if isinstance(vix_data_slice, pd.DataFrame) and vix_data_slice.shape[1] == 1:
                    final_vix_series = vix_data_slice.iloc[:, 0].rename('VIX_Close')
                elif isinstance(vix_data_slice, pd.Series):
                    final_vix_series = vix_data_slice.rename('VIX_Close')
                if final_vix_series is not None and not isinstance(final_vix_series.index, pd.DatetimeIndex):
                     final_vix_series.index = pd.to_datetime(final_vix_series.index)
            else: # VIX ticker provided but not found in multi-index, or data_yf is single-ticker VIX
                if not isinstance(data_yf.columns, pd.MultiIndex) and vix_ticker in tickers_to_load and len(tickers_to_load) == 1:
                    # This case means only VIX was downloaded, which is unusual but handle it
                    if 'Close' in data_yf.columns:
                         final_vix_series = data_yf['Close'].rename('VIX_Close')
                else:
                    if verbose: print(f"Warning: VIX ticker {vix_ticker} provided but not found as expected. VIX data will be NaNs.")
        if final_vix_series is None: # Ensure it's a Series even if no VIX data
            final_vix_series = pd.Series(np.nan, index=stock_df_simplified.index).rename('VIX_Close')


    except Exception as e_stock_vix:
        if verbose: print(f"Error loading stock/VIX data for {ticker}: {e_stock_vix}");
        return None,None,None,None,None

    daily_sentiment_series = load_sentiment_data_for_unified(sentiment_csv_path, verbose)
    if daily_sentiment_series is None:
        if verbose: print(f"Warning: Failed to load sentiment data. Sentiment features will be NaNs.")
        daily_sentiment_series = pd.Series(np.nan, index=stock_df_simplified.index).rename('SentimentScore')

    aligned_df = stock_df_simplified.copy()
    if final_vix_series is not None:
        aligned_df = aligned_df.join(final_vix_series, how='left')
    if daily_sentiment_series is not None:
        aligned_df = aligned_df.join(daily_sentiment_series, how='left')

    # Fill NaNs for VIX and Sentiment *before* dropping rows based on stock data
    if 'VIX_Close' in aligned_df.columns and aligned_df['VIX_Close'].isnull().any():
        aligned_df['VIX_Close'] = aligned_df['VIX_Close'].ffill().bfill()
    if 'SentimentScore' in aligned_df.columns and aligned_df['SentimentScore'].isnull().any():
        aligned_df['SentimentScore'] = aligned_df['SentimentScore'].ffill().bfill()

    # Fill NaNs for stock data required columns *after* VIX/Sentiment join, then drop rows if essential stock data still NaN
    for col in required_cols:
        if col in aligned_df.columns and aligned_df[col].isnull().any():
            aligned_df[col] = aligned_df[col].ffill().bfill() # ffill then bfill for stock data

    aligned_df.dropna(subset=['Close', 'High', 'Low'], inplace=True) # Only drop if essential price data is missing

    if aligned_df.empty:
        if verbose: print(f"Error: Data for {ticker} became empty after alignment and NaN handling.")
        return None,None,None,None,None

    final_stock_df = aligned_df[required_cols].copy()
    final_vix_series_aligned = aligned_df['VIX_Close'].copy() if 'VIX_Close' in aligned_df else pd.Series(np.nan, index=final_stock_df.index).rename('VIX_Close')
    final_sentiment_series_aligned = aligned_df['SentimentScore'].copy() if 'SentimentScore' in aligned_df else pd.Series(np.nan, index=final_stock_df.index).rename('SentimentScore')

    prices = final_stock_df['Close'].tolist()
    dates = final_stock_df.index.tolist()

    if verbose: print(f"Successfully loaded and unified data for {ticker}: {len(prices)} points from {start_date} to {end_date}.")
    return prices, dates, final_stock_df, final_vix_series_aligned, final_sentiment_series_aligned


# --- Unified Indicator Pre-calculation (MODIFIED) ---
def precompute_indicators_unified(stock_df, vix_series, sentiment_series, strategy_config, verbose=False):
    precalc = {
        'rsi': {}, 'vix_ma': {}, 'sentiment_ma': {},
        'bbl': {}, 'bbm': {},
        'adx': {}, 'ma_short': {}, 'ma_long': {} # MODIFIED: ADX, MA_Short, MA_Long will store variants
    }
    indicators_ready = True
    if verbose: print("  Starting unified indicator pre-calculation...")

    try:
        # RSI
        for rsi_p in strategy_config['rsi_period_options']:
            rsi_values = ta.rsi(stock_df['Close'], length=rsi_p)
            precalc['rsi'][rsi_p] = rsi_values.tolist() if rsi_values is not None else [np.nan] * len(stock_df)
        if verbose: print(f"    RSI ({len(precalc['rsi'])} variants) OK.")

        # VIX MA
        if vix_series is not None and not vix_series.isnull().all():
            for vix_ma_p in strategy_config.get('vix_ma_period_options', []):
                vix_ma_values = vix_series.rolling(window=vix_ma_p, min_periods=1).mean() # Added min_periods=1
                precalc['vix_ma'][vix_ma_p] = vix_ma_values.tolist() if not vix_ma_values.isnull().all() else [np.nan] * len(vix_series)
            if verbose: print(f"    VIX MA ({len(precalc['vix_ma'])} variants) OK or NaN.")
        else:
            if verbose: print("    VIX data not available or all NaNs, VIX MA set to NaNs.")
            for vix_ma_p in strategy_config.get('vix_ma_period_options', []):
                 precalc['vix_ma'][vix_ma_p] = [np.nan] * len(stock_df)

        # Sentiment MA
        if sentiment_series is not None and not sentiment_series.isnull().all():
            for sent_ma_p in strategy_config.get('sentiment_ma_period_options', []):
                sent_ma_values = sentiment_series.rolling(window=sent_ma_p, min_periods=1).mean() # Added min_periods=1
                precalc['sentiment_ma'][sent_ma_p] = sent_ma_values.tolist() if not sent_ma_values.isnull().all() else [np.nan] * len(sentiment_series)
            if verbose: print(f"    Sentiment MA ({len(precalc['sentiment_ma'])} variants) OK or NaN.")
        else:
            if verbose: print("    Sentiment data not available or all NaNs, Sentiment MA set to NaNs.")
            for sent_ma_p in strategy_config.get('sentiment_ma_period_options', []):
                 precalc['sentiment_ma'][sent_ma_p] = [np.nan] * len(stock_df)

        # BBands
        for bb_l in strategy_config['bb_length_options']:
            for bb_s in strategy_config['bb_std_options']:
                bbands = ta.bbands(stock_df['Close'], length=bb_l, std=bb_s)
                bbl_col_name_actual = next((col for col in bbands.columns if 'BBL' in col), None) if bbands is not None else None
                bbm_col_name_actual = next((col for col in bbands.columns if 'BBM' in col), None) if bbands is not None else None

                if bbands is None or not bbl_col_name_actual or not bbm_col_name_actual or \
                   bbands[bbl_col_name_actual].isnull().all() or bbands[bbm_col_name_actual].isnull().all():
                     if verbose: print(f"    Warning: Bollinger Bands (BBL/BBM) calculation failed or all NaNs for L={bb_l}, S={bb_s}")
                     precalc['bbl'][(bb_l, bb_s)] = [np.nan] * len(stock_df)
                     precalc['bbm'][(bb_l, bb_s)] = [np.nan] * len(stock_df)
                else:
                    precalc['bbl'][(bb_l, bb_s)] = bbands[bbl_col_name_actual].tolist()
                    precalc['bbm'][(bb_l, bb_s)] = bbands[bbm_col_name_actual].tolist()
        if verbose: print(f"    BBands ({len(precalc['bbl'])} variants) OK or NaN.")

        # ADX (MODIFIED for multiple periods)
        adx_period_options = strategy_config.get('adx_period_options', [14])
        for adx_p in adx_period_options:
            adx_data = ta.adx(stock_df['High'], stock_df['Low'], stock_df['Close'], length=adx_p)
            adx_col_name = f'ADX_{adx_p}'
            if adx_data is None or adx_col_name not in adx_data.columns or adx_data[adx_col_name].isnull().all():
                if verbose: print(f"    Warning: ADX calculation failed or all NaNs for period {adx_p}")
                precalc['adx'][adx_p] = [np.nan] * len(stock_df)
            else:
                precalc['adx'][adx_p] = adx_data[adx_col_name].tolist()
        if verbose: print(f"    ADX ({len(precalc['adx'])} variants) OK or NaN.")

        # MA Short (MODIFIED for multiple periods)
        ma_short_period_options = strategy_config.get('ma_short_period_options', [5])
        for ma_s_p in ma_short_period_options:
            ma_s_values = ta.sma(stock_df['Close'], length=ma_s_p)
            if ma_s_values is None or ma_s_values.isnull().all():
                if verbose: print(f"    Warning: MA Short calculation failed or all NaNs for period {ma_s_p}")
                precalc['ma_short'][ma_s_p] = [np.nan] * len(stock_df)
            else:
                precalc['ma_short'][ma_s_p] = ma_s_values.tolist()
        if verbose: print(f"    MA Short ({len(precalc['ma_short'])} variants) OK or NaN.")

        # MA Long (MODIFIED for multiple periods)
        ma_long_period_options = strategy_config.get('ma_long_period_options', [10])
        for ma_l_p in ma_long_period_options:
            ma_l_values = ta.sma(stock_df['Close'], length=ma_l_p)
            if ma_l_values is None or ma_l_values.isnull().all():
                if verbose: print(f"    Warning: MA Long calculation failed or all NaNs for period {ma_l_p}")
                precalc['ma_long'][ma_l_p] = [np.nan] * len(stock_df)
            else:
                precalc['ma_long'][ma_l_p] = ma_l_values.tolist()
        if verbose: print(f"    MA Long ({len(precalc['ma_long'])} variants) OK or NaN.")

    except Exception as e:
        if verbose: print(f"  Error in precompute_indicators_unified: {e}"); traceback.print_exc()
        indicators_ready = False

    # Check if essential indicator types have at least one valid precalculated series
    essential_indicator_types_check = ['rsi', 'adx', 'ma_short', 'ma_long', 'bbl'] # bbm is tied to bbl
    for ind_type in essential_indicator_types_check:
        if not precalc.get(ind_type) or not any(
            (isinstance(lst, list) and not pd.Series(lst).isnull().all())
            for lst in precalc[ind_type].values()
        ):
            if verbose: print(f"    Error: All variants for essential indicator type '{ind_type}' failed pre-calculation or are all NaNs.")
            indicators_ready = False
            break
            
    return precalc, indicators_ready

# --- Unified Numba Core Strategy ---
@numba.jit(nopython=True)
def run_strategy_numba_core_unified(
    rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold,
    external_indicator_threshold,
    low_vol_exit_strategy, high_vol_entry_choice,
    regime_indicator_choice,
    commission_rate,
    prices_arr, rsi_arr, bbl_arr, bbm_arr, adx_arr, # adx_arr is now the chosen ADX series
    external_indicator_ma_arr,
    ma_short_arr, ma_long_arr, # These are now chosen MA series
    start_trading_iloc):

    T = len(prices_arr); portfolio_values_arr = np.full(T, 1.0, dtype=np.float64)
    max_signals = T // 2 + 1
    buy_signal_indices = np.full(max_signals, -1, dtype=np.int64); buy_signal_prices = np.full(max_signals, np.nan, dtype=np.float64); buy_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    sell_signal_indices = np.full(max_signals, -1, dtype=np.int64); sell_signal_prices = np.full(max_signals, np.nan, dtype=np.float64); sell_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    buy_count = 0; sell_count = 0; cash = 1.0; stock = 0.0; position = 0; last_valid_portfolio_value = 1.0
    rsi_crossed_exit_level_after_buy = False
    risk_off_entry_type = -1 # -1: normal MA, 0: risk-off BB+RSI, 1: risk-off BB+ADX

    start_trading_iloc = max(1, start_trading_iloc)
    if start_trading_iloc >= T:
        portfolio_values_arr[:] = 1.0
        return portfolio_values_arr, buy_signal_indices[:0], buy_signal_prices[:0], buy_signal_rsis[:0], \
               sell_signal_indices[:0], sell_signal_prices[:0], sell_signal_rsis[:0]

    portfolio_values_arr[:start_trading_iloc] = 1.0

    for i in range(start_trading_iloc, T):
        current_price = prices_arr[i];
        rsi_i, rsi_prev = rsi_arr[i], rsi_arr[i-1];
        bbl_i = bbl_arr[i]; bbm_i = bbm_arr[i];
        adx_i = adx_arr[i] # Chosen ADX value for this step
        ext_indicator_ma_i = external_indicator_ma_arr[i]
        ma_short_i, ma_long_i = ma_short_arr[i], ma_long_arr[i]; # Chosen MA values
        ma_short_prev, ma_long_prev = ma_short_arr[i-1], ma_long_arr[i-1]

        current_values_for_decision = (current_price, rsi_i, bbl_i, bbm_i, adx_i, ext_indicator_ma_i, ma_short_i, ma_long_i)
        prev_values_for_decision = (rsi_prev, ma_short_prev, ma_long_prev)

        is_data_point_valid = True
        for val_idx in range(len(current_values_for_decision)): # Numba compatible loop
            if not np.isfinite(current_values_for_decision[val_idx]): is_data_point_valid = False; break
        if is_data_point_valid:
            for val_idx in range(len(prev_values_for_decision)):
                if not np.isfinite(prev_values_for_decision[val_idx]): is_data_point_valid = False; break

        if not is_data_point_valid:
            current_portfolio_value = cash
            if position == 1:
                if np.isfinite(current_price):
                    current_portfolio_value = stock * current_price
                else:
                    current_portfolio_value = np.nan

            if np.isfinite(current_portfolio_value):
                portfolio_values_arr[i] = current_portfolio_value
                last_valid_portfolio_value = current_portfolio_value
            else:
                portfolio_values_arr[i] = last_valid_portfolio_value
            continue

        is_risk_off_regime = False
        if regime_indicator_choice == 0:
            is_risk_off_regime = ext_indicator_ma_i >= external_indicator_threshold
        elif regime_indicator_choice == 1:
            is_risk_off_regime = ext_indicator_ma_i <= external_indicator_threshold

        if not (is_risk_off_regime and risk_off_entry_type == 0) :
            rsi_crossed_exit_level_after_buy = False

        if not is_risk_off_regime:
            risk_off_entry_type = -1

        if position == 0:
            buy_condition = False; entry_type_if_bought_this_step = -1
            if is_risk_off_regime:
                if high_vol_entry_choice == 0:
                    if (current_price <= bbl_i) and (rsi_i < rsi_buy_entry_threshold):
                        buy_condition = True; entry_type_if_bought_this_step = 0
                else:
                    if (current_price <= bbl_i) and (adx_i > adx_threshold): # adx_threshold is gene[9]
                        buy_condition = True; entry_type_if_bought_this_step = 1
            else:
                if (ma_short_prev < ma_long_prev) and (ma_short_i >= ma_long_i):
                    buy_condition = True; entry_type_if_bought_this_step = -1

            if buy_condition and current_price > 1e-9:
                cost = cash * commission_rate; amount_to_invest = cash - cost
                if amount_to_invest > 0:
                    stock = amount_to_invest / current_price; cash = 0.0; position = 1
                    risk_off_entry_type = entry_type_if_bought_this_step
                    rsi_crossed_exit_level_after_buy = False
                    if buy_count < max_signals:
                        buy_signal_indices[buy_count] = i; buy_signal_prices[buy_count] = current_price;
                        buy_signal_rsis[buy_count] = rsi_i; buy_count += 1

        elif position == 1:
            sell_condition = False
            if risk_off_entry_type == 0:
                if rsi_i >= rsi_exit_threshold:
                    rsi_crossed_exit_level_after_buy = True
                if rsi_crossed_exit_level_after_buy and rsi_i < rsi_exit_threshold:
                    sell_condition = True
            elif risk_off_entry_type == 1:
                if current_price >= bbm_i:
                    sell_condition = True
            elif risk_off_entry_type == -1:
                if low_vol_exit_strategy == 0:
                    sell_condition = (current_price < ma_short_i)
                else:
                    sell_condition = (ma_short_prev > ma_long_prev) and (ma_short_i <= ma_long_i)

            if sell_condition:
                proceeds = stock * current_price; cost = proceeds * commission_rate; cash = proceeds - cost
                stock = 0.0; position = 0
                if sell_count < max_signals:
                    sell_signal_indices[sell_count] = i; sell_signal_prices[sell_count] = current_price;
                    sell_signal_rsis[sell_count] = rsi_i; sell_count += 1

        current_stock_value = stock * current_price if position == 1 and np.isfinite(current_price) else 0.0
        if position == 1 and not np.isfinite(current_price):
            current_portfolio_value = np.nan
        else:
            current_portfolio_value = cash + current_stock_value

        if np.isfinite(current_portfolio_value):
            portfolio_values_arr[i] = current_portfolio_value
            last_valid_portfolio_value = current_portfolio_value
        else:
            portfolio_values_arr[i] = last_valid_portfolio_value

    if T > 0 and np.isnan(portfolio_values_arr[-1]):
        portfolio_values_arr[-1] = last_valid_portfolio_value

    return portfolio_values_arr, buy_signal_indices[:buy_count], buy_signal_prices[:buy_count], buy_signal_rsis[:buy_count], \
           sell_signal_indices[:sell_count], sell_signal_prices[:sell_count], sell_signal_rsis[:sell_count]

# --- Unified Wrapper function ---
# run_strategy_unified (NO CHANGE NEEDED HERE, it already accepts the lists)
def run_strategy_unified(
    rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold,
    vix_or_sentiment_threshold,
    low_vol_exit_strategy, high_vol_entry_choice,
    regime_indicator_choice,
    external_indicator_ma_list,
    commission_rate,
    prices, dates,
    rsi_list, bbl_list, bbm_list, adx_list,
    ma_short_list, ma_long_list):

    T = len(prices)
    if T == 0: return [1.0] * (1 if T == 0 else T) , [], [] # Return default for empty prices, ensure list for T=0

    # Ensure all indicator lists have the same length as prices, filling with NaNs if necessary
    def sanitize_list(lst, length, list_name="Indicator"):
        if not isinstance(lst, list):
            # print(f"Warning: {list_name} is not a list. Converting. Length: {length}")
            lst = [] # Ensure it's a list
        if len(lst) != length:
            # print(f"Warning: {list_name} length mismatch. Expected {length}, got {len(lst)}. Filling with NaNs.")
            return [np.nan] * length
        return lst

    rsi_list = sanitize_list(rsi_list, T, "RSI List")
    bbl_list = sanitize_list(bbl_list, T, "BBL List")
    bbm_list = sanitize_list(bbm_list, T, "BBM List")
    adx_list = sanitize_list(adx_list, T, "ADX List")
    external_indicator_ma_list = sanitize_list(external_indicator_ma_list, T, "External MA List")
    ma_short_list = sanitize_list(ma_short_list, T, "MA Short List")
    ma_long_list = sanitize_list(ma_long_list, T, "MA Long List")

    prices_arr = np.array(prices, dtype=np.float64)
    rsi_arr = np.array(rsi_list, dtype=np.float64)
    bbl_arr = np.array(bbl_list, dtype=np.float64)
    bbm_arr = np.array(bbm_list, dtype=np.float64)
    adx_arr = np.array(adx_list, dtype=np.float64)
    external_indicator_ma_arr_np = np.array(external_indicator_ma_list, dtype=np.float64)
    ma_short_arr = np.array(ma_short_list, dtype=np.float64)
    ma_long_arr = np.array(ma_long_list, dtype=np.float64)

    def get_first_valid_iloc(indicator_arr):
        if indicator_arr is None or len(indicator_arr) == 0: # Handle empty or None arrays
            return T
        valid_indices = np.where(np.isfinite(indicator_arr))[0]
        return valid_indices[0] if len(valid_indices) > 0 else T

    start_trading_iloc = 0
    # Only consider essential indicators for determining the start trading iloc
    # External indicator MA might be all NaNs if VIX/Sentiment is not used or data is missing
    # BBL/BBM might also be all NaNs if BBands calculation failed for a specific gene.
    # RSI, ADX, MA_Short, MA_Long are more consistently expected.
    essential_indicator_arrays_for_start = [rsi_arr, adx_arr, ma_short_arr, ma_long_arr]
    # Also consider bbl_arr and bbm_arr if they are not all NaNs
    if bbl_arr is not None and not np.all(np.isnan(bbl_arr)):
        essential_indicator_arrays_for_start.append(bbl_arr)
    if bbm_arr is not None and not np.all(np.isnan(bbm_arr)):
        essential_indicator_arrays_for_start.append(bbm_arr)


    for arr in essential_indicator_arrays_for_start:
        start_trading_iloc = max(start_trading_iloc, get_first_valid_iloc(arr))

    start_trading_iloc += 1 # Need i and i-1 to be valid for some prev calculations

    if start_trading_iloc >= T :
        # print(f"Warning: Not enough valid data points to start trading. Start iloc {start_trading_iloc} >= T {T}")
        return [1.0] * T, [], []

    start_trading_iloc = max(start_trading_iloc, 1) # Ensure it's at least 1 for i-1 access in Numba

    portfolio_values_arr, buy_indices, buy_prices, buy_rsis, sell_indices, sell_prices, sell_rsis = \
        run_strategy_numba_core_unified(
            float(rsi_buy_entry_threshold), float(rsi_exit_threshold), float(adx_threshold),
            float(vix_or_sentiment_threshold),
            int(low_vol_exit_strategy), int(high_vol_entry_choice),
            int(regime_indicator_choice),
            float(commission_rate),
            prices_arr, rsi_arr, bbl_arr, bbm_arr, adx_arr,
            external_indicator_ma_arr_np,
            ma_short_arr, ma_long_arr,
            start_trading_iloc
        )
    buy_signals = []; sell_signals = []
    if dates: # Ensure dates list is not empty
        for idx, price, rsi_val in zip(buy_indices, buy_prices, buy_rsis):
            if idx != -1 and idx < len(dates): buy_signals.append((dates[idx], price, rsi_val))
        for idx, price, rsi_val in zip(sell_indices, sell_prices, sell_rsis):
             if idx != -1 and idx < len(dates): sell_signals.append((dates[idx], price, rsi_val))
    return portfolio_values_arr.tolist(), buy_signals, sell_signals
# --- Unified Genetic Algorithm (MODIFIED for 15-element gene) ---
def genetic_algorithm_with_elitism_unified(prices, dates,
                                           precalculated_indicators,
                                           ga_params):
    generations = ga_params['generations']; population_size = ga_params['population_size']; crossover_rate = ga_params['crossover_rate']
    mutation_rate = ga_params['mutation_rate']; elitism_size = ga_params['elitism_size']; tournament_size = ga_params['tournament_size']
    mutation_amount_range = ga_params['mutation_amount_range']
    show_ga_generations = ga_params.get('show_process', False)

    rsi_threshold_range = ga_params['rsi_threshold_range']
    vix_threshold_range = ga_params['vix_threshold_range']
    sentiment_threshold_range = ga_params['sentiment_threshold_range']
    adx_threshold_range_param = ga_params['adx_threshold_range'] # This is for gene[9] (ADX strategy threshold)

    rsi_period_options = ga_params['rsi_period_options']; num_rsi_options = len(rsi_period_options)
    vix_ma_period_options = ga_params.get('vix_ma_period_options', []); num_vix_ma_options = len(vix_ma_period_options)
    sentiment_ma_period_options = ga_params.get('sentiment_ma_period_options', []); num_sentiment_ma_options = len(sentiment_ma_period_options)
    bb_length_options = ga_params['bb_length_options']; num_bb_len_options = len(bb_length_options)
    bb_std_options = ga_params['bb_std_options']; num_bb_std_options = len(bb_std_options)
    commission_rate = ga_params['commission_rate']

    # MODIFIED: Get new period options for ADX, MA_Short, MA_Long
    adx_period_options_ga = ga_params['adx_period_options']; num_adx_options = len(adx_period_options_ga)
    ma_short_period_options_ga = ga_params['ma_short_period_options']; num_ma_short_options = len(ma_short_period_options_ga)
    ma_long_period_options_ga = ga_params['ma_long_period_options']; num_ma_long_options = len(ma_long_period_options_ga)

    vix_mutation_amount_range_actual = ga_params.get('vix_mutation_amount_range', mutation_amount_range)
    sentiment_mutation_amount_range_actual = ga_params.get('sentiment_mutation_amount_range', mutation_amount_range)
    adx_mutation_amount_range_actual = ga_params.get('adx_mutation_amount_range', mutation_amount_range) # For gene[9]

    T = len(prices)
    if T < 2:
        if show_ga_generations: print(f"Error: Data length {T} too short for GA.")
        return None, 0

    population = []; attempts, max_attempts = 0, population_size * 300 # Increased attempts
    min_buy, max_buy, min_exit, max_exit = rsi_threshold_range
    min_vix_thr, max_vix_thr = vix_threshold_range
    min_sent_thr, max_sent_thr = sentiment_threshold_range
    min_adx_strat_thr, max_adx_strat_thr = adx_threshold_range_param # For gene[9]

    while len(population) < population_size and attempts < max_attempts:
        gene = [0]*15 # MODIFIED: Gene length is 15
        gene[0] = random.randint(min_buy, max_buy)
        gene[1] = random.randint(max(gene[0] + 1, min_exit), max_exit)
        gene[2] = random.randint(min_vix_thr, max_vix_thr)
        gene[3] = random.randint(min_sent_thr, max_sent_thr)
        gene[4] = random.choice([0, 1])
        gene[5] = random.randint(0, num_rsi_options - 1) if num_rsi_options > 0 else 0
        gene[11] = random.choice([0, 1])
        if gene[11] == 0:
            gene[6] = random.randint(0, num_vix_ma_options - 1) if num_vix_ma_options > 0 else 0
        else:
            gene[6] = random.randint(0, num_sentiment_ma_options - 1) if num_sentiment_ma_options > 0 else 0
        gene[7] = random.randint(0, num_bb_len_options - 1) if num_bb_len_options > 0 else 0
        gene[8] = random.randint(0, num_bb_std_options - 1) if num_bb_std_options > 0 else 0
        gene[9] = random.randint(min_adx_strat_thr, max_adx_strat_thr) # ADX strategy threshold
        gene[10] = random.choice([0, 1])

        # MODIFIED: New gene elements for ADX/MA periods
        gene[12] = random.randint(0, num_adx_options - 1) if num_adx_options > 0 else 0
        gene[13] = random.randint(0, num_ma_short_options - 1) if num_ma_short_options > 0 else 0
        gene[14] = random.randint(0, num_ma_long_options - 1) if num_ma_long_options > 0 else 0
        
        # Initial check for MA Long > MA Short (can be made more robust or rely on fitness penalty)
        if num_ma_short_options > 0 and num_ma_long_options > 0:
            short_p_val = ma_short_period_options_ga[gene[13]]
            long_p_val = ma_long_period_options_ga[gene[14]]
            if long_p_val <= short_p_val: # If invalid, try to pick a valid long period
                valid_long_indices = [idx for idx, lp in enumerate(ma_long_period_options_ga) if lp > short_p_val]
                if valid_long_indices:
                    gene[14] = random.choice(valid_long_indices)
                # else: this gene might be penalized by fitness if no valid long period exists for the short one

        valid = (0 < gene[0] < gene[1] < 100)
        if valid: population.append(gene)
        attempts += 1

    if not population or len(population) < population_size :
        if show_ga_generations: print(f"Error: Could not generate sufficient initial population ({len(population)}/{population_size}).")
        return None, 0
    best_gene_overall = population[0][:]; best_fitness_overall = -float('inf')

    for generation in range(generations):
        fitness = []
        for gene_idx, gene in enumerate(population):
            try:
                chosen_rsi_period = rsi_period_options[gene[5]] if num_rsi_options > 0 else None
                chosen_bb_length = bb_length_options[gene[7]] if num_bb_len_options > 0 else None
                chosen_bb_std = bb_std_options[gene[8]] if num_bb_std_options > 0 else None

                # MODIFIED: Get chosen ADX, MA_Short, MA_Long periods
                chosen_adx_period = adx_period_options_ga[gene[12]] if num_adx_options > 0 else None
                chosen_ma_short_period = ma_short_period_options_ga[gene[13]] if num_ma_short_options > 0 else None
                chosen_ma_long_period = ma_long_period_options_ga[gene[14]] if num_ma_long_options > 0 else None

                # Constraint: MA Long period > MA Short period
                if chosen_ma_short_period is not None and \
                   chosen_ma_long_period is not None and \
                   chosen_ma_long_period <= chosen_ma_short_period:
                    fitness.append(-np.inf) # Penalize invalid MA combination
                    if show_ga_generations and (generation + 1) % 20 == 0 and gene_idx < 5 : # Reduce verbosity
                         print(f"  Penalized Gene {gene_idx}: Invalid MA periods (L:{chosen_ma_long_period} <= S:{chosen_ma_short_period}).")
                    continue

                rsi_list_current = precalculated_indicators['rsi'].get(chosen_rsi_period, [np.nan]*T) if chosen_rsi_period else [np.nan]*T
                bbl_list_current, bbm_list_current = [np.nan]*T, [np.nan]*T
                if chosen_bb_length is not None and chosen_bb_std is not None:
                    bb_key = (chosen_bb_length, chosen_bb_std)
                    bbl_list_current = precalculated_indicators['bbl'].get(bb_key, [np.nan]*T)
                    bbm_list_current = precalculated_indicators['bbm'].get(bb_key, [np.nan]*T)

                # MODIFIED: Get precalculated lists for chosen ADX, MA_Short, MA_Long
                adx_list_current = precalculated_indicators['adx'].get(chosen_adx_period, [np.nan]*T) if chosen_adx_period else [np.nan]*T
                ma_short_list_current = precalculated_indicators['ma_short'].get(chosen_ma_short_period, [np.nan]*T) if chosen_ma_short_period else [np.nan]*T
                ma_long_list_current = precalculated_indicators['ma_long'].get(chosen_ma_long_period, [np.nan]*T) if chosen_ma_long_period else [np.nan]*T

                external_indicator_ma_list_current = [np.nan]*T
                current_external_threshold = 0
                regime_choice = gene[11]
                if regime_choice == 0:
                    chosen_ext_period = vix_ma_period_options[gene[6]] if num_vix_ma_options > 0 else None
                    if chosen_ext_period:
                        external_indicator_ma_list_current = precalculated_indicators['vix_ma'].get(chosen_ext_period, [np.nan]*T)
                    current_external_threshold = gene[2]
                else:
                    chosen_ext_period = sentiment_ma_period_options[gene[6]] if num_sentiment_ma_options > 0 else None
                    if chosen_ext_period:
                        external_indicator_ma_list_current = precalculated_indicators['sentiment_ma'].get(chosen_ext_period, [np.nan]*T)
                    current_external_threshold = gene[3]

                portfolio_values, _, _ = run_strategy_unified(
                    gene[0], gene[1], gene[9], # rsi_buy, rsi_exit, adx_strategy_threshold
                    current_external_threshold,
                    gene[4], gene[10],
                    regime_choice,
                    external_indicator_ma_list_current,
                    commission_rate,
                    prices, dates,
                    rsi_list_current, bbl_list_current, bbm_list_current,
                    adx_list_current, # Pass chosen ADX list
                    ma_short_list_current, ma_long_list_current # Pass chosen MA lists
                )
                final_value = next((p for p in reversed(portfolio_values) if np.isfinite(p) and p is not None), -np.inf);
                fitness.append(final_value)

            except (IndexError, KeyError) as e_eval:
                if show_ga_generations: print(f"  Eval Error (Gene {gene_idx}: {gene}): {e_eval}. Fitness -inf.")
                fitness.append(-np.inf)
            except Exception as e_unexp:
                if show_ga_generations: print(f"  Unexpected Eval Error (Gene {gene_idx}: {gene}): {e_unexp}. Fitness -inf."); traceback.print_exc()
                fitness.append(-np.inf)

        fitness_array = np.array(fitness, dtype=float);
        valid_fitness_mask = np.isfinite(fitness_array) & (fitness_array > -np.inf) & (~np.isnan(fitness_array))
        valid_indices = np.where(valid_fitness_mask)[0];
        valid_fitness_count = len(valid_indices)

        if valid_fitness_count == 0:
             if show_ga_generations: print(f"Gen {generation+1} - Warning: All individuals invalid or fitness NaN.")
             if generation == generations -1 and best_fitness_overall == -float('inf'):
                 print("GA failed to find any valid solution across all generations.")
                 return None, 0
             continue

        sorted_valid_indices = valid_indices[np.argsort(fitness_array[valid_indices])[::-1]];
        num_elites = min(elitism_size, valid_fitness_count);
        elite_indices = sorted_valid_indices[:num_elites];
        elites = [population[i][:] for i in elite_indices]

        current_best_fitness_in_gen = fitness_array[elite_indices[0]] if num_elites > 0 else -np.inf
        if current_best_fitness_in_gen > best_fitness_overall:
            best_fitness_overall = current_best_fitness_in_gen;
            best_gene_overall = population[elite_indices[0]][:]

        if show_ga_generations and (generation + 1) % 10 == 0:
            gen_best_str = f"{current_best_fitness_in_gen:.4f}" if num_elites > 0 else "N/A"
            overall_best_str = "N/A"
            if best_fitness_overall > -np.inf and best_gene_overall:
                bo_lv_exit_str = "Price<MA" if best_gene_overall[4] == 0 else "MACross"
                bo_rsi_p_str = rsi_period_options[best_gene_overall[5]] if num_rsi_options > 0 and best_gene_overall[5] < num_rsi_options else "N/A"
                bo_regime_choice_str = "VIX" if best_gene_overall[11] == 0 else "Sentiment"
                bo_ext_p_str = ""
                if best_gene_overall[11] == 0:
                    bo_ext_p_str = vix_ma_period_options[best_gene_overall[6]] if num_vix_ma_options > 0 and best_gene_overall[6] < num_vix_ma_options else "N/A"
                else:
                    bo_ext_p_str = sentiment_ma_period_options[best_gene_overall[6]] if num_sentiment_ma_options > 0 and best_gene_overall[6] < num_sentiment_ma_options else "N/A"
                bo_bb_l_str = bb_length_options[best_gene_overall[7]] if num_bb_len_options > 0 and best_gene_overall[7] < num_bb_len_options else "N/A"
                bo_bb_s_str = bb_std_options[best_gene_overall[8]] if num_bb_std_options > 0 and best_gene_overall[8] < num_bb_std_options else "N/A"
                bo_hv_entry_str = "BB+RSI" if best_gene_overall[10] == 0 else "BB+ADX"
                # MODIFIED: Add new params to readable string
                bo_adx_p_val_str = adx_period_options_ga[best_gene_overall[12]] if num_adx_options > 0 and best_gene_overall[12] < num_adx_options else "N/A"
                bo_ma_s_p_val_str = ma_short_period_options_ga[best_gene_overall[13]] if num_ma_short_options > 0 and best_gene_overall[13] < num_ma_short_options else "N/A"
                bo_ma_l_p_val_str = ma_long_period_options_ga[best_gene_overall[14]] if num_ma_long_options > 0 and best_gene_overall[14] < num_ma_long_options else "N/A"

                overall_best_params_readable = (
                    f"RSI(P:{bo_rsi_p_str}, Buy:{best_gene_overall[0]}/Exit:{best_gene_overall[1]}), "
                    f"Regime:{bo_regime_choice_str}(P:{bo_ext_p_str}, ThrVIX:{best_gene_overall[2]}/Sent:{best_gene_overall[3]}), "
                    f"LVExit:{bo_lv_exit_str}, BB(L:{bo_bb_l_str},S:{bo_bb_s_str}), ADX_Strat_T:{best_gene_overall[9]}, "
                    f"RiskOffEntry:{bo_hv_entry_str}, ADX_P:{bo_adx_p_val_str}, MA_S_P:{bo_ma_s_p_val_str}, MA_L_P:{bo_ma_l_p_val_str}" # MODIFIED
                )
                overall_best_str = f"{best_fitness_overall:.4f} ({overall_best_params_readable})"
            print(f"Gen {generation+1}/{generations} | Best(G): {gen_best_str} | Best(O): {overall_best_str} | Valid: {valid_fitness_count}/{population_size}")

        selected_parents = []; num_parents_to_select = population_size - num_elites
        if num_parents_to_select <= 0: population = elites[:population_size]; continue

        effective_tournament_size = min(tournament_size, valid_fitness_count)
        if effective_tournament_size <= 0:
            if valid_fitness_count > 0 :
                 selected_parents = [population[valid_indices[0]][:] for _ in range(num_parents_to_select)]
            else:
                 population = elites[:population_size]; continue

        if effective_tournament_size > 0:
            for _ in range(num_parents_to_select):
                aspirant_indices_local = np.random.choice(len(valid_indices), size=effective_tournament_size, replace=True);
                aspirant_indices_global = valid_indices[aspirant_indices_local];
                winner_global_idx = aspirant_indices_global[np.argmax(fitness_array[aspirant_indices_global])];
                selected_parents.append(population[winner_global_idx][:])

        offspring = []; parent_indices = list(range(len(selected_parents))); random.shuffle(parent_indices); num_pairs = len(parent_indices) // 2
        for i_pair in range(num_pairs):
            p1, p2 = selected_parents[parent_indices[2*i_pair]], selected_parents[parent_indices[2*i_pair + 1]]; child1, child2 = p1[:], p2[:]
            if random.random() < crossover_rate:
                 crossover_point = random.randint(1, 14); # MODIFIED: Gene length 15 (0-14), so point up to 14
                 child1_new = p1[:crossover_point] + p2[crossover_point:]; child2_new = p2[:crossover_point] + p1[crossover_point:]
                 if 0 < child1_new[0] < child1_new[1] < 100: child1 = child1_new # Basic RSI check
                 if 0 < child2_new[0] < child2_new[1] < 100: child2 = child2_new
            offspring.append(child1); offspring.append(child2)
        if len(parent_indices) % 2 != 0: offspring.append(selected_parents[parent_indices[-1]][:])

        mut_min, mut_max = mutation_amount_range
        vix_mut_min_actual, vix_mut_max_actual = vix_mutation_amount_range_actual
        sentiment_mut_min_actual, sentiment_mut_max_actual = sentiment_mutation_amount_range_actual
        adx_mut_min_actual, adx_mut_max_actual = adx_mutation_amount_range_actual

        for i_offspring in range(len(offspring)):
            if random.random() < mutation_rate:
                gene_to_mutate = offspring[i_offspring]; mutate_idx = random.randint(0, 14) # MODIFIED: Mutate up to index 14
                if mutate_idx == 4: gene_to_mutate[4] = 1 - gene_to_mutate[4]
                elif mutate_idx == 10: gene_to_mutate[10] = 1 - gene_to_mutate[10]
                elif mutate_idx == 11:
                    gene_to_mutate[11] = 1 - gene_to_mutate[11]
                    if gene_to_mutate[11] == 0: gene_to_mutate[6] = random.randint(0, num_vix_ma_options - 1) if num_vix_ma_options > 0 else 0
                    else: gene_to_mutate[6] = random.randint(0, num_sentiment_ma_options - 1) if num_sentiment_ma_options > 0 else 0
                elif mutate_idx == 5: # rsi_period_choice
                    if num_rsi_options > 0: gene_to_mutate[5] = random.randint(0, num_rsi_options - 1)
                elif mutate_idx == 6: # ext_ma_period_choice
                    num_options = num_vix_ma_options if gene_to_mutate[11] == 0 else num_sentiment_ma_options
                    if num_options > 0: gene_to_mutate[6] = random.randint(0, num_options - 1)
                elif mutate_idx == 7: # bb_length_choice
                    if num_bb_len_options > 0: gene_to_mutate[7] = random.randint(0, num_bb_len_options - 1)
                elif mutate_idx == 8: # bb_std_choice
                    if num_bb_std_options > 0: gene_to_mutate[8] = random.randint(0, num_bb_std_options - 1)
                # MODIFIED: Mutation for new period choices
                elif mutate_idx == 12: # adx_period_choice
                    if num_adx_options > 0: gene_to_mutate[12] = random.randint(0, num_adx_options - 1)
                elif mutate_idx == 13: # ma_short_period_choice
                    if num_ma_short_options > 0: gene_to_mutate[13] = random.randint(0, num_ma_short_options - 1)
                elif mutate_idx == 14: # ma_long_period_choice
                    if num_ma_long_options > 0: gene_to_mutate[14] = random.randint(0, num_ma_long_options - 1)
                else: # Numeric parameter mutation (gene[0,1,2,3,9])
                    mut_amount = 0
                    if mutate_idx == 2: mut_amount = random.randint(vix_mut_min_actual, vix_mut_max_actual)
                    elif mutate_idx == 3: mut_amount = random.randint(sentiment_mut_min_actual, sentiment_mut_max_actual)
                    elif mutate_idx == 9: mut_amount = random.randint(adx_mut_min_actual, adx_mut_max_actual) # ADX Strategy Threshold
                    else: mut_amount = random.randint(mut_min, mut_max) # For RSI thresholds

                    gene_to_mutate[mutate_idx] += mut_amount
                    gene_to_mutate[0] = max(min_buy, min(gene_to_mutate[0], max_buy))
                    gene_to_mutate[1] = max(min_exit, min(gene_to_mutate[1], max_exit))
                    if gene_to_mutate[0] >= gene_to_mutate[1]:
                        gene_to_mutate[0] = max(min_buy, gene_to_mutate[1] -1)
                        gene_to_mutate[0] = max(1, gene_to_mutate[0])
                        if gene_to_mutate[0] >= gene_to_mutate[1]:
                           gene_to_mutate[0] = min_buy
                           gene_to_mutate[1] = max(min_buy + 1, min_exit)
                    gene_to_mutate[2] = max(min_vix_thr, min(gene_to_mutate[2], max_vix_thr))
                    gene_to_mutate[3] = max(min_sent_thr, min(gene_to_mutate[3], max_sent_thr))
                    gene_to_mutate[9] = max(min_adx_strat_thr, min(gene_to_mutate[9], max_adx_strat_thr)) # ADX Strategy Threshold

        population = elites + offspring; population = population[:population_size]

    if best_fitness_overall == -float('inf') or best_gene_overall is None:
        if show_ga_generations: print("Error: GA finished without finding any valid solution or best gene.")
        return None, 0
    return best_gene_overall, best_fitness_overall

# --- Main Execution Block (MODIFIED) ---
if __name__ == "__main__":
    print("Executing newest_model.py as a standalone script...")
    sentiment_csv_file_main = '2021-2025每週新聞及情緒分析.csv'
    if not os.path.exists(sentiment_csv_file_main):
        print(f"ERROR: Sentiment CSV file not found at {sentiment_csv_file_main}")
        exit()


    ga_train_config = {
        'ticker': 'UNH', 'vix_ticker': '^VIX', #'2330.TW',
        'start_date': '2024-01-01', 'end_date': '2025-05-23', # MODIFIED: Shorter training period
        'description': "GA Optimization (Unified - ADX/MA Periods Optimized)"
    }
    strategy_config_shared = { # MODIFIED: Added ADX/MA period options
        'rsi_period_options': [7, 14, 21],
        'vix_ma_period_options': [2, 5, 10, 20],
        'sentiment_ma_period_options': [1, 2, 4], # Weekly sentiment, so MA periods are in weeks
        'bb_length_options': [10, 20],
        'bb_std_options': [1.5, 2.0, 2.5],
        'adx_period_options': [7, 14, 21],
        'ma_short_period_options': [5, 10],
        'ma_long_period_options': [10,20,50], # Ensure options allow Long > Short
        'commission_pct': 0.0025,
    }
    ga_params_config = { # MODIFIED: Added ADX/MA period options here too
        'generations': 10, 'population_size': 50, # Slightly adjusted
        'crossover_rate': 0.8, 'mutation_rate': 0.35, # Slightly adjusted
        'elitism_size': 7, 'tournament_size': 7, 'mutation_amount_range': (-7, 7), # Wider for RSI
        'vix_mutation_amount_range': (-4, 4),
        'sentiment_mutation_amount_range': (-7, 7),
        'adx_mutation_amount_range': (-4, 4), # For ADX strategy threshold (gene[9])
        'show_process': False,
        'rsi_threshold_range': (5, 45, 46, 80), # min_buy, max_buy, min_exit, max_exit
        'vix_threshold_range': (10, 40),
        'sentiment_threshold_range': (10, 90),
        'adx_threshold_range': (10, 40), # For ADX strategy threshold (gene[9])
        'rsi_period_options': strategy_config_shared['rsi_period_options'],
        'vix_ma_period_options': strategy_config_shared['vix_ma_period_options'],
        'sentiment_ma_period_options': strategy_config_shared['sentiment_ma_period_options'],
        'bb_length_options': strategy_config_shared['bb_length_options'],
        'bb_std_options': strategy_config_shared['bb_std_options'],
        'adx_period_options': strategy_config_shared['adx_period_options'], # Pass to GA
        'ma_short_period_options': strategy_config_shared['ma_short_period_options'], # Pass to GA
        'ma_long_period_options': strategy_config_shared['ma_long_period_options'], # Pass to GA
        'commission_rate': strategy_config_shared['commission_pct'],
    }
    run_config_main = { 'num_runs': 200 } # Reduced for quicker testing, increase for real runs
    plot_config_shared = {
        'show_plot': True, 'figure_size': (18, 12), 'buy_marker_color': 'lime',
        'sell_marker_color': 'deeppink', 'strategy_line_color': '#007bff', 'price_line_color': 'darkgrey',
        'plot_bbands': True, 'bb_line_color': 'gold', 'bb_fill_color': 'lightyellow',
        'plot_mas': True, 'ma_short_color': 'deepskyblue', 'ma_long_color': 'tomato',
        'plot_ext_indicator_ma': True, 'ext_indicator_ma_color': 'darkviolet'
    }

    print(f"\n--- Phase 1: GA Optimization ({ga_train_config['description']}) ---")
    print(f"Period: {ga_train_config['start_date']} to {ga_train_config['end_date']} for Ticker: {ga_train_config['ticker']}")

    train_prices, train_dates, train_stock_df, train_vix_series_unified, train_sentiment_series_unified = load_data_unified(
        ga_train_config['ticker'], ga_train_config['vix_ticker'], sentiment_csv_file_main,
        start_date=ga_train_config['start_date'], end_date=ga_train_config['end_date'],
        verbose=ga_params_config['show_process']
    )

    best_params_from_ga = None
    if train_prices and train_dates and train_stock_df is not None and not train_stock_df.empty:
        if not ga_params_config['show_process']: print("  Pre-calculating indicators for GA training period...")
        train_precalculated_indicators, train_indicators_ready = precompute_indicators_unified(
            train_stock_df, train_vix_series_unified, train_sentiment_series_unified,
            strategy_config_shared, verbose=ga_params_config['show_process'] # Pass full strategy_config
        )

        if train_indicators_ready:
            if not ga_params_config['show_process']:
                print(f"  Starting GA optimization ({ga_params_config['generations']} generations, {ga_params_config['population_size']} population)...")
            ga_start_time = time.time()

            overall_best_fitness_for_ga_phase = -float('inf')
            best_params_for_ga_phase = None

            for run_idx in range(run_config_main['num_runs']):
                if ga_params_config['show_process']: print(f"--- GA Run {run_idx + 1}/{run_config_main['num_runs']} ---")

                current_best_params_run, current_fitness_run = genetic_algorithm_with_elitism_unified(
                    train_prices, train_dates,
                    train_precalculated_indicators,
                    ga_params=ga_params_config # Pass full ga_params
                )
                if current_best_params_run and current_fitness_run is not None and np.isfinite(current_fitness_run) and current_fitness_run > overall_best_fitness_for_ga_phase:
                    overall_best_fitness_for_ga_phase = current_fitness_run
                    best_params_for_ga_phase = current_best_params_run

                if not ga_params_config['show_process'] and current_best_params_run and current_fitness_run is not None and np.isfinite(current_fitness_run):
                     print(f"    GA Run {run_idx + 1} Best Value: {current_fitness_run:.4f}")


            ga_time_taken = time.time() - ga_start_time
            if not ga_params_config['show_process']: print(f"  GA optimization (all runs) finished in {ga_time_taken:.2f} seconds.")

            if best_params_for_ga_phase:
                best_params_from_ga = best_params_for_ga_phase
                print(f"  Best Fitness from GA (In-Sample, {run_config_main['num_runs']} runs): {overall_best_fitness_for_ga_phase:.4f}")
                bp_ga = best_params_from_ga

                # MODIFIED: Update parameter descriptions
                val_rsi_buy = bp_ga[0]; val_rsi_exit = bp_ga[1]
                val_vix_thr = bp_ga[2]; val_sent_thr = bp_ga[3]
                val_lv_exit_choice_code = bp_ga[4]; desc_lv_exit_val = 'Price < MA_Short' if val_lv_exit_choice_code == 0 else 'MA CrossDown'
                val_rsi_p_choice_code = bp_ga[5]; desc_rsi_p_val = ga_params_config['rsi_period_options'][val_rsi_p_choice_code] if val_rsi_p_choice_code < len(ga_params_config['rsi_period_options']) else "N/A"
                val_regime_choice_code = bp_ga[11]; desc_regime_ind_name_val = "VIX MA" if val_regime_choice_code == 0 else "Sentiment MA"
                val_ext_p_choice_code = bp_ga[6]
                desc_chosen_ext_period_val_str = "N/A"
                if val_regime_choice_code == 0:
                    if val_ext_p_choice_code < len(ga_params_config['vix_ma_period_options']): desc_chosen_ext_period_val_str = str(ga_params_config['vix_ma_period_options'][val_ext_p_choice_code])
                else:
                    if val_ext_p_choice_code < len(ga_params_config['sentiment_ma_period_options']): desc_chosen_ext_period_val_str = str(ga_params_config['sentiment_ma_period_options'][val_ext_p_choice_code])
                val_bb_l_choice_code = bp_ga[7]; desc_bb_l_val = str(ga_params_config['bb_length_options'][val_bb_l_choice_code]) if val_bb_l_choice_code < len(ga_params_config['bb_length_options']) else "N/A"
                val_bb_s_choice_code = bp_ga[8]; desc_bb_s_val = str(ga_params_config['bb_std_options'][val_bb_s_choice_code]) if val_bb_s_choice_code < len(ga_params_config['bb_std_options']) else "N/A"
                val_adx_strat_thr = bp_ga[9] # ADX Strategy Threshold
                val_hv_entry_choice_code = bp_ga[10]; desc_hv_entry_val = 'BB+RSI' if val_hv_entry_choice_code == 0 else 'BB+ADX'
                # New parameters
                val_adx_p_choice_code = bp_ga[12]; desc_adx_p_val = ga_params_config['adx_period_options'][val_adx_p_choice_code] if val_adx_p_choice_code < len(ga_params_config['adx_period_options']) else "N/A"
                val_ma_s_p_choice_code = bp_ga[13]; desc_ma_s_p_val = ga_params_config['ma_short_period_options'][val_ma_s_p_choice_code] if val_ma_s_p_choice_code < len(ga_params_config['ma_short_period_options']) else "N/A"
                val_ma_l_p_choice_code = bp_ga[14]; desc_ma_l_p_val = ga_params_config['ma_long_period_options'][val_ma_l_p_choice_code] if val_ma_l_p_choice_code < len(ga_params_config['ma_long_period_options']) else "N/A"


                print(f"  Best GA Parameters (Unified Strategy - 15 elements):")
                print(f"    RSI Buy Entry: {val_rsi_buy}, RSI Exit Ref: {val_rsi_exit}, RSI Period: {desc_rsi_p_val}")
                print(f"    Regime: {desc_regime_ind_name_val}(P:{desc_chosen_ext_period_val_str}), VIX Thr: {val_vix_thr}, Sent Thr: {val_sent_thr}")
                print(f"    Low Vol MA Exit: {desc_lv_exit_val}, MA_Short_P: {desc_ma_s_p_val}, MA_Long_P: {desc_ma_l_p_val}")
                print(f"    BB Length: {desc_bb_l_val}, BB Std: {desc_bb_s_val}")
                print(f"    ADX Strat Thr: {val_adx_strat_thr}, ADX Period: {desc_adx_p_val}")
                print(f"    Risk-Off Entry: {desc_hv_entry_val}")
                if not ga_params_config['show_process']: print("-" * 40)

                if plot_config_shared['show_plot']:
                    print(f"  Plotting GA Best Strategy on Training Data ({ga_train_config['start_date']} to {ga_train_config['end_date']})...")
                    required_len_train = len(train_prices)
                    # MODIFIED: Get correct indicator lists for plotting based on best_params_from_ga
                    plot_rsi_list = train_precalculated_indicators['rsi'].get(int(desc_rsi_p_val), [np.nan]*required_len_train) if desc_rsi_p_val != "N/A" else [np.nan]*required_len_train
                    plot_bbl_list, plot_bbm_list = [np.nan]*required_len_train, [np.nan]*required_len_train
                    if desc_bb_l_val != "N/A" and desc_bb_s_val != "N/A":
                        try:
                            bb_key = (int(desc_bb_l_val), float(desc_bb_s_val))
                            plot_bbl_list = train_precalculated_indicators['bbl'].get(bb_key, [np.nan]*required_len_train)
                            plot_bbm_list = train_precalculated_indicators['bbm'].get(bb_key, [np.nan]*required_len_train)
                        except (ValueError, TypeError): pass
                    
                    plot_adx_list = train_precalculated_indicators['adx'].get(int(desc_adx_p_val), [np.nan]*required_len_train) if desc_adx_p_val != "N/A" else [np.nan]*required_len_train
                    plot_ma_short_list = train_precalculated_indicators['ma_short'].get(int(desc_ma_s_p_val), [np.nan]*required_len_train) if desc_ma_s_p_val != "N/A" else [np.nan]*required_len_train
                    plot_ma_long_list = train_precalculated_indicators['ma_long'].get(int(desc_ma_l_p_val), [np.nan]*required_len_train) if desc_ma_l_p_val != "N/A" else [np.nan]*required_len_train

                    plot_ext_ind_ma_list_train = [np.nan]*required_len_train
                    plot_ext_ind_threshold_train = val_vix_thr if val_regime_choice_code == 0 else val_sent_thr
                    plot_ext_ind_name_train_plot = f"{desc_regime_ind_name_val}({desc_chosen_ext_period_val_str})"
                    if desc_chosen_ext_period_val_str != "N/A":
                        try:
                            period_key = int(desc_chosen_ext_period_val_str)
                            if val_regime_choice_code == 0:
                                plot_ext_ind_ma_list_train = train_precalculated_indicators.get('vix_ma', {}).get(period_key, [np.nan]*required_len_train)
                            else:
                                plot_ext_ind_ma_list_train = train_precalculated_indicators.get('sentiment_ma', {}).get(period_key, [np.nan]*required_len_train)
                        except (ValueError, TypeError): pass
                    
                    # Sanitize list lengths before passing to strategy
                    all_plot_lists_train_check = [plot_rsi_list, plot_bbl_list, plot_bbm_list, plot_adx_list, plot_ma_short_list, plot_ma_long_list, plot_ext_ind_ma_list_train]
                    for k_idx_check, lst_check in enumerate(all_plot_lists_train_check):
                        if not isinstance(lst_check, list) or len(lst_check) != required_len_train:
                            all_plot_lists_train_check[k_idx_check] = [np.nan] * required_len_train
                    [plot_rsi_list, plot_bbl_list, plot_bbm_list, plot_adx_list, plot_ma_short_list, plot_ma_long_list, plot_ext_ind_ma_list_train] = all_plot_lists_train_check


                    train_portfolio_values, train_buy_signals, train_sell_signals = run_strategy_unified(
                        val_rsi_buy, val_rsi_exit, val_adx_strat_thr, # Use ADX strategy threshold
                        plot_ext_ind_threshold_train,
                        val_lv_exit_choice_code, val_hv_entry_choice_code,
                        val_regime_choice_code,
                        plot_ext_ind_ma_list_train,
                        strategy_config_shared['commission_pct'],
                        train_prices, train_dates,
                        plot_rsi_list, plot_bbl_list, plot_bbm_list, plot_adx_list,
                        plot_ma_short_list, plot_ma_long_list
                    )
                    # ... (Plotting logic mostly the same, ensure titles and legends are updated if needed) ...
                    fig_train, ax_train1 = plt.subplots(figsize=plot_config_shared['figure_size'])
                    initial_price_train_val = train_prices[0] if train_prices and np.isfinite(train_prices[0]) and train_prices[0] > 1e-9 else 1.0
                    
                    color_price = plot_config_shared['price_line_color']
                    ax_train1.set_xlabel('Date', fontsize=12); ax_train1.set_ylabel('Normalized Value', color=color_price, fontsize=12)
                    ax_train1.plot(train_dates, np.array(train_prices, dtype=float)/initial_price_train_val, label=f"{ga_train_config['ticker']} Price (Train, Norm)", color=color_price, alpha=0.7, linewidth=1)
                    ax_train1.plot(train_dates, train_portfolio_values, label=f'GA Strat (Train) Val: {overall_best_fitness_for_ga_phase:.4f}', color=plot_config_shared['strategy_line_color'], linewidth=1.5)
                    ax_train1.tick_params(axis='y', labelcolor=color_price)
                    if train_buy_signals: ax_train1.scatter([s[0] for s in train_buy_signals], np.array([s[1] for s in train_buy_signals],dtype=float)/initial_price_train_val, label='Buy', marker='^', color=plot_config_shared['buy_marker_color'], s=100, edgecolors='k', zorder=5, alpha=0.9)
                    if train_sell_signals: ax_train1.scatter([s[0] for s in train_sell_signals], np.array([s[1] for s in train_sell_signals],dtype=float)/initial_price_train_val, label='Sell', marker='v', color=plot_config_shared['sell_marker_color'], s=100, edgecolors='k', zorder=5, alpha=0.9)
                    
                    if plot_config_shared['plot_ext_indicator_ma'] and any(np.isfinite(plot_ext_ind_ma_list_train)) :
                        ax_train2 = ax_train1.twinx()
                        color_ext_ind = plot_config_shared['ext_indicator_ma_color']
                        ax_train2.set_ylabel(plot_ext_ind_name_train_plot, color=color_ext_ind, fontsize=12)
                        ax_train2.plot(train_dates, plot_ext_ind_ma_list_train, color=color_ext_ind, linestyle='--', label=plot_ext_ind_name_train_plot, alpha=0.6, linewidth=1.5)
                        ax_train2.tick_params(axis='y', labelcolor=color_ext_ind)
                        ax_train2.axhline(y=plot_ext_ind_threshold_train, color='gray', linestyle=':', linewidth=1.5, label=f'Threshold ({plot_ext_ind_threshold_train})')
                    
                    fig_train.suptitle(f"{ga_train_config['ticker']} - GA Optimized (Training: {ga_train_config['start_date']} to {ga_train_config['end_date']})", fontsize=14, weight='bold')
                    lines, labels = ax_train1.get_legend_handles_labels()
                    if plot_config_shared['plot_ext_indicator_ma'] and any(np.isfinite(plot_ext_ind_ma_list_train)) and 'ax_train2' in locals():
                        lines2, labels2 = ax_train2.get_legend_handles_labels()
                        ax_train1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=8)
                    else:
                        ax_train1.legend(lines, labels, loc='upper left', fontsize=8)
                    ax_train1.grid(True, alpha=0.3); fig_train.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

            else:
                print("  GA optimization did not yield valid parameters for the training period.")
        else:
            print("  Indicator pre-calculation failed or returned insufficient data for GA training period.")
    else:
        print(f"  Data loading failed for GA training period ({ga_train_config['ticker']}).")

    if best_params_from_ga:
        print(f"\n--- Phase 2: Out-of-Sample Test with GA Best Parameters (Unified Strategy) ---")
        test_config_params = { # MODIFIED: OOS Test Period
            'ticker': ga_train_config['ticker'], 'vix_ticker': ga_train_config['vix_ticker'],
            'start_date': '2024-01-01', 'end_date': '2025-05-23', # Example OOS dates
            'description': "Out-of-Sample Test with Unified Strategy"
        }
        print(f"Period: {test_config_params['start_date']} to {test_config_params['end_date']} for Ticker: {test_config_params['ticker']}")

        test_prices, test_dates, test_stock_df, test_vix_series_unified, test_sentiment_series_unified = load_data_unified(
            test_config_params['ticker'], test_config_params['vix_ticker'], sentiment_csv_file_main,
            start_date=test_config_params['start_date'], end_date=test_config_params['end_date'],
            verbose=ga_params_config['show_process']
        )

        if test_prices and test_dates and test_stock_df is not None and not test_stock_df.empty:
            if not ga_params_config['show_process']: print("  Pre-calculating indicators for OOS test period...")
            test_precalculated_indicators, test_indicators_ready = precompute_indicators_unified(
                test_stock_df, test_vix_series_unified, test_sentiment_series_unified,
                strategy_config_shared, verbose=ga_params_config['show_process'] # Use same shared config
            )

            if test_indicators_ready:
                if not ga_params_config['show_process']: print("  Running strategy on OOS test period with GA best parameters...")
                bp_ga_final_test = best_params_from_ga # Use the best gene from training

                # MODIFIED: Extract OOS parameters based on the 15-element gene
                oos_rsi_buy = bp_ga_final_test[0]; oos_rsi_exit = bp_ga_final_test[1]
                oos_vix_thr = bp_ga_final_test[2]; oos_sent_thr = bp_ga_final_test[3]
                oos_lv_exit_choice_code = bp_ga_final_test[4]
                oos_rsi_p_choice_code = bp_ga_final_test[5]; desc_rsi_p_oos = ga_params_config['rsi_period_options'][oos_rsi_p_choice_code] if oos_rsi_p_choice_code < len(ga_params_config['rsi_period_options']) else "N/A"
                oos_regime_choice_code = bp_ga_final_test[11]
                oos_ext_p_choice_code = bp_ga_final_test[6]
                desc_chosen_ext_period_oos = "N/A"
                current_oos_external_threshold = 0
                if oos_regime_choice_code == 0:
                    if oos_ext_p_choice_code < len(ga_params_config['vix_ma_period_options']): desc_chosen_ext_period_oos = str(ga_params_config['vix_ma_period_options'][oos_ext_p_choice_code])
                    current_oos_external_threshold = oos_vix_thr
                else:
                    if oos_ext_p_choice_code < len(ga_params_config['sentiment_ma_period_options']): desc_chosen_ext_period_oos = str(ga_params_config['sentiment_ma_period_options'][oos_ext_p_choice_code])
                    current_oos_external_threshold = oos_sent_thr
                oos_bb_l_choice_code = bp_ga_final_test[7]; desc_bb_l_oos = str(ga_params_config['bb_length_options'][oos_bb_l_choice_code]) if oos_bb_l_choice_code < len(ga_params_config['bb_length_options']) else "N/A"
                oos_bb_s_choice_code = bp_ga_final_test[8]; desc_bb_s_oos = str(ga_params_config['bb_std_options'][oos_bb_s_choice_code]) if oos_bb_s_choice_code < len(ga_params_config['bb_std_options']) else "N/A"
                oos_adx_strat_thr_val = bp_ga_final_test[9]
                oos_hv_entry_choice_code = bp_ga_final_test[10]
                # New params for OOS
                oos_adx_p_choice_code = bp_ga_final_test[12]; desc_adx_p_oos = ga_params_config['adx_period_options'][oos_adx_p_choice_code] if oos_adx_p_choice_code < len(ga_params_config['adx_period_options']) else "N/A"
                oos_ma_s_p_choice_code = bp_ga_final_test[13]; desc_ma_s_p_oos = ga_params_config['ma_short_period_options'][oos_ma_s_p_choice_code] if oos_ma_s_p_choice_code < len(ga_params_config['ma_short_period_options']) else "N/A"
                oos_ma_l_p_choice_code = bp_ga_final_test[14]; desc_ma_l_p_oos = ga_params_config['ma_long_period_options'][oos_ma_l_p_choice_code] if oos_ma_l_p_choice_code < len(ga_params_config['ma_long_period_options']) else "N/A"


                required_len_oos = len(test_prices)
                # MODIFIED: Get correct indicator lists for OOS plotting
                test_rsi_list = test_precalculated_indicators['rsi'].get(int(desc_rsi_p_oos), [np.nan]*required_len_oos) if desc_rsi_p_oos != "N/A" else [np.nan]*required_len_oos
                test_bbl_list, test_bbm_list = [np.nan]*required_len_oos, [np.nan]*required_len_oos
                if desc_bb_l_oos != "N/A" and desc_bb_s_oos != "N/A":
                    try:
                        bb_key_oos = (int(desc_bb_l_oos), float(desc_bb_s_oos))
                        test_bbl_list = test_precalculated_indicators['bbl'].get(bb_key_oos, [np.nan]*required_len_oos)
                        test_bbm_list = test_precalculated_indicators['bbm'].get(bb_key_oos, [np.nan]*required_len_oos)
                    except (ValueError, TypeError): pass
                
                test_adx_list = test_precalculated_indicators['adx'].get(int(desc_adx_p_oos), [np.nan]*required_len_oos) if desc_adx_p_oos != "N/A" else [np.nan]*required_len_oos
                test_ma_short_list = test_precalculated_indicators['ma_short'].get(int(desc_ma_s_p_oos), [np.nan]*required_len_oos) if desc_ma_s_p_oos != "N/A" else [np.nan]*required_len_oos
                test_ma_long_list = test_precalculated_indicators['ma_long'].get(int(desc_ma_l_p_oos), [np.nan]*required_len_oos) if desc_ma_l_p_oos != "N/A" else [np.nan]*required_len_oos

                oos_external_indicator_ma_list_for_run = [np.nan]*required_len_oos
                if desc_chosen_ext_period_oos != "N/A":
                    try:
                        period_key = int(desc_chosen_ext_period_oos)
                        if oos_regime_choice_code == 0:
                            oos_external_indicator_ma_list_for_run = test_precalculated_indicators.get('vix_ma',{}).get(period_key, [np.nan]*required_len_oos)
                        else:
                            oos_external_indicator_ma_list_for_run = test_precalculated_indicators.get('sentiment_ma',{}).get(period_key, [np.nan]*required_len_oos)
                    except (ValueError, TypeError): pass

                all_plot_lists_oos_check = [test_rsi_list, test_bbl_list, test_bbm_list, test_adx_list, test_ma_short_list, test_ma_long_list, oos_external_indicator_ma_list_for_run]
                for k_idx_oos_check, lst_oos_check in enumerate(all_plot_lists_oos_check):
                    if not isinstance(lst_oos_check, list) or len(lst_oos_check) != required_len_oos:
                         all_plot_lists_oos_check[k_idx_oos_check] = [np.nan] * required_len_oos
                [test_rsi_list, test_bbl_list, test_bbm_list, test_adx_list, test_ma_short_list, test_ma_long_list, oos_external_indicator_ma_list_for_run] = all_plot_lists_oos_check


                oos_portfolio_values, oos_buy_signals, oos_sell_signals = run_strategy_unified(
                    oos_rsi_buy, oos_rsi_exit, oos_adx_strat_thr_val,
                    current_oos_external_threshold,
                    oos_lv_exit_choice_code, oos_hv_entry_choice_code,
                    oos_regime_choice_code,
                    oos_external_indicator_ma_list_for_run,
                    strategy_config_shared['commission_pct'],
                    test_prices, test_dates,
                    test_rsi_list, test_bbl_list, test_bbm_list, test_adx_list,
                    test_ma_short_list, test_ma_long_list
                )

                if oos_portfolio_values:
                    # ... (OOS statistics and plotting logic mostly the same, ensure titles and legends are updated) ...
                    final_oos_value = next((p for p in reversed(oos_portfolio_values) if np.isfinite(p) and p is not None), 1.0)
                    print(f"  Out-of-Sample Final Portfolio Value: {final_oos_value:.4f}")
                    num_oos_trades = 0
                    if oos_buy_signals and oos_sell_signals:
                        num_oos_trades = min(len(oos_buy_signals), len(oos_sell_signals))
                    
                    winning_oos_trades = 0
                    if num_oos_trades > 0:
                        for i_oos_trade in range(num_oos_trades):
                            if len(oos_sell_signals) > i_oos_trade and len(oos_buy_signals) > i_oos_trade:
                                if oos_sell_signals[i_oos_trade][1] > oos_buy_signals[i_oos_trade][1]:
                                    winning_oos_trades += 1
                        oos_win_rate = (winning_oos_trades / num_oos_trades) * 100 if num_oos_trades > 0 else 0.0
                        print(f"    Trades in OOS period: {num_oos_trades}, Wins: {winning_oos_trades}, Win Rate: {oos_win_rate:.2f}%")
                    else:
                        print("    No completed trades in OOS period.")

                    if plot_config_shared['show_plot']:
                        if not ga_params_config['show_process']: print(f"  Plotting GA Best Strategy on Test Data ({test_config_params['start_date']} to {test_config_params['end_date']})...")
                        fig_test, ax_test1 = plt.subplots(figsize=plot_config_shared['figure_size'])
                        initial_price_test_val = test_prices[0] if test_prices and np.isfinite(test_prices[0]) and test_prices[0] > 1e-9 else 1.0
                        
                        color_price_test = plot_config_shared['price_line_color']
                        ax_test1.set_xlabel('Date', fontsize=12); ax_test1.set_ylabel('Normalized Value', color=color_price_test, fontsize=12)
                        ax_test1.plot(test_dates, np.array(test_prices, dtype=float)/initial_price_test_val, label=f"{test_config_params['ticker']} Price (Test, Norm)", color=color_price_test, alpha=0.7, linewidth=1)
                        ax_test1.plot(test_dates, oos_portfolio_values, label=f'GA Strat (Test) Val: {final_oos_value:.4f}', color='purple', linewidth=1.5)
                        ax_test1.tick_params(axis='y', labelcolor=color_price_test)
                        if oos_buy_signals: ax_test1.scatter([s[0] for s in oos_buy_signals], np.array([s[1] for s in oos_buy_signals],dtype=float)/initial_price_test_val, label='Buy (OOS)', marker='^', color='cyan', s=100, edgecolors='k', zorder=5, alpha=0.9)
                        if oos_sell_signals: ax_test1.scatter([s[0] for s in oos_sell_signals], np.array([s[1] for s in oos_sell_signals],dtype=float)/initial_price_test_val, label='Sell (OOS)', marker='v', color='magenta', s=100, edgecolors='k', zorder=5, alpha=0.9)
                        
                        plot_ext_ind_name_test_plot = f"{('VIX MA' if oos_regime_choice_code == 0 else 'Sentiment MA')}({desc_chosen_ext_period_oos})"
                        if plot_config_shared['plot_ext_indicator_ma'] and any(np.isfinite(oos_external_indicator_ma_list_for_run)):
                            ax_test2 = ax_test1.twinx()
                            color_ext_ind_test = plot_config_shared['ext_indicator_ma_color']
                            ax_test2.set_ylabel(plot_ext_ind_name_test_plot, color=color_ext_ind_test, fontsize=12)
                            ax_test2.plot(test_dates, oos_external_indicator_ma_list_for_run, color=color_ext_ind_test, linestyle='--', label=plot_ext_ind_name_test_plot, alpha=0.6, linewidth=1.5)
                            ax_test2.tick_params(axis='y', labelcolor=color_ext_ind_test)
                            ax_test2.axhline(y=current_oos_external_threshold, color='gray', linestyle=':', linewidth=1.5, label=f'Threshold ({current_oos_external_threshold})')

                        commission_info_plot = f"{strategy_config_shared['commission_pct']*100:.3f}%"
                        desc_lv_exit_oos_plot = 'Price < MA_Short' if oos_lv_exit_choice_code == 0 else 'MA CrossDown'
                        desc_hv_entry_oos_plot = 'BB+RSI' if oos_hv_entry_choice_code == 0 else 'BB+ADX'

                        title_bt_plot = ( # MODIFIED: Update plot title for new params
                            f"{test_config_params['ticker']} - GA Unified OOS Test: {test_config_params['start_date']} to {test_config_params['end_date']}\n"
                            f"Regime: {plot_ext_ind_name_test_plot}, Thr: {current_oos_external_threshold} | RSI(P:{desc_rsi_p_oos}, B:{oos_rsi_buy}/E:{oos_rsi_exit})\n"
                            f"BB(L:{desc_bb_l_oos},S:{desc_bb_s_oos}) | ADX(P:{desc_adx_p_oos}, Strat_T:{oos_adx_strat_thr_val})\n"
                            f"RiskOff Entry: {desc_hv_entry_oos_plot} | RiskOn MA(S:{desc_ma_s_p_oos},L:{desc_ma_l_p_oos}) Exit: {desc_lv_exit_oos_plot} | Comm: {commission_info_plot}"
                        )
                        fig_test.suptitle(title_bt_plot, fontsize=9, weight='bold') # Adjusted font size
                        lines1, labels1 = ax_test1.get_legend_handles_labels()
                        if plot_config_shared['plot_ext_indicator_ma'] and any(np.isfinite(oos_external_indicator_ma_list_for_run)) and 'ax_test2' in locals():
                            lines2_oos, labels2_oos = ax_test2.get_legend_handles_labels()
                            ax_test1.legend(lines1 + lines2_oos, labels1 + labels2_oos, loc='best', fontsize=7) # Adjusted font size
                        else:
                            ax_test1.legend(lines1, labels1, loc='best', fontsize=7)
                        ax_test1.grid(True, alpha=0.3); fig_test.tight_layout(rect=[0, 0.03, 1, 0.91]); plt.show() # Adjusted rect
                else:
                    print("  Could not run strategy on OOS test period (no portfolio values).")
            else:
                print("  Indicator pre-calculation failed or returned insufficient data for OOS test period.")
        else:
            print(f"  Data loading failed for OOS test period ({test_config_params['ticker']}).")
    else:
        print("\n--- Skipping Out-of-Sample Test: GA Optimization did not yield best parameters. ---")

    print("\nnewest_model.py standalone test execution finished.")