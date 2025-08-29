# ga_engine.py - 完整版支援 NSGA-II 多目標優化與平均交易報酬率 (修復版)

# 版本: 2.1 - 修復NSGA-II基因生成和交易獎勵機制

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import numba
import random
import time
import traceback
import re
import os
import json
from datetime import datetime as dt_datetime, timedelta

# NSGA-II 支援
try:
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    NSGA2_AVAILABLE = True
    print("[GAEngine] NSGA-II 支援已載入。")
except ImportError:
    NSGA2_AVAILABLE = False
    print("[GAEngine] WARN: NSGA-II 套件 (pymoo) 未安裝。將使用傳統單目標 GA。請執行 'pip install pymoo' 安裝以啟用多目標優化。")

# --- GA Configuration ---
STRATEGY_CONFIG_SHARED_GA = {
    'vix_ma_period_options': [1, 2, 5, 10, 20],
    'sentiment_ma_period_options': [1, 2, 4],
    'rsi_period_options': [7, 14, 21],
    'adx_period_options': [7, 14, 21],
    'bb_length_options': [10, 20],
    'bb_std_options': [1.5, 2.0],
    'ma_period_options': [5, 10, 20, 50, 60],
    'ema_s_period_options': [5, 8],
    'ema_m_period_options': [8, 10, 13],
    'ema_l_period_options': [13, 21, 34],
    'atr_period_options': [10, 14, 20],
    'kd_k_period_options': [9, 14],
    'kd_d_period_options': [3],
    'kd_smooth_period_options': [3],
    'macd_fast_period_options': [8, 12],
    'macd_slow_period_options': [21, 26],
    'macd_signal_period_options': [9],
    'commission_rate': 0.005,
}

GA_PARAMS_CONFIG = {
    'generations': 10, 'population_size': 70, 'crossover_rate': 0.8,
    'mutation_rate': 0.3, 'elitism_size': 5, 'tournament_size': 7,
    'show_process': False, 'offline_trainer_runs_per_stock': 60,
    'vix_threshold_range': (15, 30),
    'sentiment_threshold_range': (25, 75),
    'rsi_threshold_range': (15, 45, 46, 85),
    'kd_threshold_range': (10, 30, 70, 90),
    'adx_threshold_range': (15, 40),
    'min_trades_for_full_score': 4,
    'no_trade_penalty_factor': 0.1,
    'low_trade_penalty_factor': 0.75,
    # NSGA-II 特定參數
    'nsga2_enabled': True,
    'nsga2_objectives_num': 4,
    'nsga2_selection_method': 'custom_balance',
    'min_required_trades': 5,
    'nsga2_no_trade_penalty_return': -0.5,
    'nsga2_no_trade_penalty_max_drawdown': 1.0,
    'nsga2_no_trade_penalty_std_dev': 1.0,
    'nsga2_no_trade_penalty_profit_factor': 0.01,
    **STRATEGY_CONFIG_SHARED_GA
}

# --- Gene Map and Names ---
GENE_MAP = {
    'regime_choice': 0, 'normal_strat': 1, 'risk_off_strat': 2, 'vix_thr': 3,
    'sentiment_thr': 4, 'rsi_buy_thr': 5, 'rsi_sell_thr': 6, 'kd_buy_thr': 7,
    'kd_sell_thr': 8, 'adx_thr': 9, 'vix_ma_p': 10, 'sentiment_ma_p': 11,
    'rsi_p': 12, 'adx_p': 13, 'ma_s_p': 14, 'ma_l_p': 15, 'ema_s_p': 16,
    'ema_m_p': 17, 'ema_l_p': 18, 'atr_p': 19, 'kd_k_p': 20, 'kd_d_p': 21,
    'kd_s_p': 22, 'macd_f_p': 23, 'macd_s_p': 24, 'macd_sig_p': 25,
    'bb_l_p': 26, 'bb_s_p': 27
}

STRAT_NAMES = ["MA Cross", "Triple EMA", "MA+MACD+RSI", "EMA+RSI", "BB+RSI", "BB+ADX", "ATR+KD", "BB+MACD"]

# --- Data Loading Functions --- (保持不變)
def parse_week_string_for_sentiment(week_str):
    """解析週期字串用於情緒數據"""
    try:
        if '-' in week_str and len(week_str.split('-')[0].split('/')) == 3:
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
                return None, None
    except Exception:
        return None, None

def load_sentiment_data_for_unified(csv_filepath, verbose=False):
    """載入並轉換情緒數據為日期序列"""
    try:
        sentiment_df = pd.read_csv(csv_filepath, encoding='utf-8-sig')
        sentiment_df.rename(columns={'年/週': 'WeekString', '情緒分數': 'SentimentScore'}, inplace=True, errors='ignore')
        if 'WeekString' not in sentiment_df.columns or 'SentimentScore' not in sentiment_df.columns:
            if verbose:
                print("Sentiment CSV must contain '年/週' and '情緒分數' columns.")
            return None

        daily_sentiments = []
        for _, row in sentiment_df.iterrows():
            week_str = str(row['WeekString']).strip()
            score = row['SentimentScore']
            if pd.isna(score):
                continue
            start_date, end_date = parse_week_string_for_sentiment(week_str)
            if start_date and end_date:
                current_date = start_date
                while current_date <= end_date:
                    daily_sentiments.append({'Date': current_date, 'SentimentScore': float(score)})
                    current_date += timedelta(days=1)

        if not daily_sentiments:
            if verbose:
                print("No daily sentiment data could be generated from CSV.")
            return None

        daily_sentiment_df = pd.DataFrame(daily_sentiments)
        daily_sentiment_df['Date'] = pd.to_datetime(daily_sentiment_df['Date'])
        daily_sentiment_df = daily_sentiment_df.set_index('Date')
        daily_sentiment_df = daily_sentiment_df[~daily_sentiment_df.index.duplicated(keep='first')]
        return daily_sentiment_df['SentimentScore']
    except Exception as e:
        if verbose:
            print(f"Error loading sentiment data: {e}")
        return None

def ga_load_data(ticker, vix_ticker="^VIX", start_date=None, end_date=None, verbose=False, sentiment_csv_path=None, retries=3, delay=5):
    """統一的數據載入函數，支援重試機制"""
    if verbose:
        print(f"[GAEngine] Loading unified data for {ticker}, VIX:{vix_ticker}, Sentiment:{sentiment_csv_path}...")

    for attempt in range(retries):
        try:
            tickers_to_load = list(set([t for t in [ticker, vix_ticker] if t]))
            data_yf = yf.download(tickers_to_load, start=start_date, end=end_date, progress=False, auto_adjust=False, timeout=20)

            if data_yf is None or data_yf.empty:
                raise ValueError(f"yfinance download returned empty for {tickers_to_load}")

            if isinstance(data_yf.columns, pd.MultiIndex):
                if ticker not in data_yf.columns.get_level_values(1):
                    raise ValueError(f"Ticker '{ticker}' not found in downloaded MultiIndex columns.")
                stock_data = data_yf.loc[:, pd.IndexSlice[:, ticker]]
                stock_data.columns = stock_data.columns.droplevel(1)
            elif not isinstance(data_yf.columns, pd.MultiIndex) and ticker in tickers_to_load and len(tickers_to_load) == 1:
                stock_data = data_yf
            else:
                raise ValueError(f"Could not extract stock data for '{ticker}'. Available columns: {data_yf.columns}")

            required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
            if not all(col in stock_data.columns for col in ['Close', 'High', 'Low']):
                raise ValueError(f"Stock data for {ticker} missing essential price columns.")

            for col in required_cols:
                if col not in stock_data.columns:
                    stock_data[col] = np.nan

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
                elif not isinstance(data_yf.columns, pd.MultiIndex) and vix_ticker in tickers_to_load and len(tickers_to_load) == 1:
                    if 'Close' in data_yf.columns:
                        final_vix_series = data_yf['Close'].rename('VIX_Close')

            if final_vix_series is not None and not isinstance(final_vix_series.index, pd.DatetimeIndex):
                final_vix_series.index = pd.to_datetime(final_vix_series.index)

            if final_vix_series is None:
                if verbose and vix_ticker:
                    print(f"[GAEngine] WARN: VIX data for {vix_ticker} not found. Will be NaNs.")
                final_vix_series = pd.Series(np.nan, index=stock_df_simplified.index, name='VIX_Close')

            daily_sentiment_series = None
            if sentiment_csv_path and os.path.exists(sentiment_csv_path):
                daily_sentiment_series = load_sentiment_data_for_unified(sentiment_csv_path, verbose)

            if daily_sentiment_series is None:
                daily_sentiment_series = pd.Series(np.nan, index=stock_df_simplified.index, name='SentimentScore')

            aligned_df = stock_df_simplified.copy()
            aligned_df = aligned_df.join(final_vix_series, how='left')
            aligned_df = aligned_df.join(daily_sentiment_series, how='left')

            for col in ['VIX_Close', 'SentimentScore']:
                if col in aligned_df.columns and aligned_df[col].isnull().any():
                    aligned_df[col] = aligned_df[col].ffill().bfill()

            for col in required_cols:
                if col in aligned_df.columns and aligned_df[col].isnull().any():
                    aligned_df[col] = aligned_df[col].ffill().bfill()

            aligned_df.dropna(subset=['Close', 'High', 'Low'], inplace=True)

            if aligned_df.empty:
                raise ValueError(f"Data for {ticker} became empty after alignment and cleaning.")

            final_stock_df = aligned_df[required_cols].copy()
            final_vix_series_aligned = aligned_df.get('VIX_Close')
            final_sentiment_series_aligned = aligned_df.get('SentimentScore')

            prices = final_stock_df['Close'].tolist()
            dates = final_stock_df.index.tolist()

            if verbose:
                print(f"[GAEngine] Successfully loaded and unified data for {ticker}: {len(prices)} points.")

            return prices, dates, final_stock_df, final_vix_series_aligned, final_sentiment_series_aligned

        except Exception as e:
            print(f"[GAEngine] WARN: Attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            if attempt + 1 < retries:
                print(f"[GAEngine] Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"[GAEngine] ERROR: All {retries} attempts failed for {ticker}. Could not extract stock data.")
                return None, None, None, None, None

    return None, None, None, None, None

def ga_precompute_indicators(stock_df, vix_series, strategy_config, sentiment_series=None, verbose=False):
    """預計算所有技術指標"""
    precalc = {
        'rsi': {}, 'vix_ma': {}, 'sentiment_ma': {}, 'bbl': {}, 'bbm': {}, 'bbu': {}, 'adx': {},
        'ema_s': {}, 'ema_m': {}, 'ema_l': {}, 'atr': {}, 'atr_ma': {},
        'kd_k': {}, 'kd_d': {}, 'macd_line': {}, 'macd_signal': {}, 'ma': {}
    }

    if verbose:
        print("[GAEngine] Starting indicator pre-calculation...")

    try:
        df_len = len(stock_df)

        def calc_or_nan(func, **kwargs):
            try:
                res = func(**kwargs)
                return res.tolist() if isinstance(res, pd.Series) else [np.nan] * df_len
            except Exception:
                return [np.nan] * df_len

        for p in strategy_config['rsi_period_options']:
            precalc['rsi'][p] = calc_or_nan(ta.rsi, source=stock_df['Close'], length=p)

        if vix_series is not None and not vix_series.empty:
            for p in strategy_config['vix_ma_period_options']:
                precalc['vix_ma'][p] = vix_series.rolling(p).mean().tolist()

        if sentiment_series is not None and not sentiment_series.empty:
            for p in strategy_config['sentiment_ma_period_options']:
                precalc['sentiment_ma'][p] = sentiment_series.rolling(p).mean().tolist()

        for l in strategy_config['bb_length_options']:
            for s in strategy_config['bb_std_options']:
                bbands = ta.bbands(stock_df['Close'], length=l, std=s)
                key = (l, s)
                if bbands is not None and not bbands.empty:
                    precalc['bbl'][key] = bbands[f'BBL_{l}_{float(s)}'].tolist()
                    precalc['bbm'][key] = bbands[f'BBM_{l}_{float(s)}'].tolist()
                    precalc['bbu'][key] = bbands[f'BBU_{l}_{float(s)}'].tolist()

        for p in strategy_config['ema_s_period_options']:
            precalc['ema_s'][p] = calc_or_nan(ta.ema, source=stock_df['Close'], length=p)

        for p in strategy_config['ema_m_period_options']:
            precalc['ema_m'][p] = calc_or_nan(ta.ema, source=stock_df['Close'], length=p)

        for p in strategy_config['ema_l_period_options']:
            precalc['ema_l'][p] = calc_or_nan(ta.ema, source=stock_df['Close'], length=p)

        for p in strategy_config['atr_period_options']:
            atr = ta.atr(stock_df['High'], stock_df['Low'], stock_df['Close'], length=p)
            if atr is not None:
                precalc['atr'][p] = atr.tolist()
                precalc['atr_ma'][p] = atr.rolling(p).mean().tolist()

        for k in strategy_config['kd_k_period_options']:
            for d in strategy_config['kd_d_period_options']:
                for s in strategy_config['kd_smooth_period_options']:
                    stoch = ta.stoch(stock_df['High'], stock_df['Low'], stock_df['Close'], k=k, d=d, smooth_k=s)
                    key = (k, d, s)
                    if stoch is not None and not stoch.empty:
                        precalc['kd_k'][key] = stoch[f'STOCHk_{k}_{d}_{s}'].tolist()
                        precalc['kd_d'][key] = stoch[f'STOCHd_{k}_{d}_{s}'].tolist()

        for f in strategy_config['macd_fast_period_options']:
            for s in strategy_config['macd_slow_period_options']:
                if f >= s:
                    continue
                for sig in strategy_config['macd_signal_period_options']:
                    macd = ta.macd(stock_df['Close'], fast=f, slow=s, signal=sig)
                    key = (f, s, sig)
                    if macd is not None and not macd.empty:
                        precalc['macd_line'][key] = macd[f'MACD_{f}_{s}_{sig}'].tolist()
                        precalc['macd_signal'][key] = macd[f'MACDs_{f}_{s}_{sig}'].tolist()

        for p in strategy_config['ma_period_options']:
            precalc['ma'][p] = calc_or_nan(ta.sma, source=stock_df['Close'], length=p)

        for p in strategy_config['adx_period_options']:
            adx = ta.adx(stock_df['High'], stock_df['Low'], stock_df['Close'], length=p)
            if adx is not None and not adx.empty:
                precalc['adx'][p] = adx[f'ADX_{p}'].tolist()

        if verbose:
            print("[GAEngine] Indicator pre-calculation finished.")

        return precalc, True

    except Exception as e:
        print(f"[GAEngine] ERROR: Error in precompute_indicators: {e}")
        traceback.print_exc()
        return precalc, False

# --- Core Numba Strategy --- (保持不變，已經正確使用numba)
@numba.jit(nopython=True)
def run_strategy_numba_core(
    gene_arr, prices_arr, vix_ma_arr, sentiment_ma_arr,
    rsi_arr, adx_arr, bbl_arr, bbm_arr, bbu_arr, ma_s_arr, ma_l_arr,
    ema_s_arr, ema_m_arr, ema_l_arr, atr_arr, atr_ma_arr, k_arr, d_arr,
    macd_line_arr, macd_signal_arr, commission_rate, start_trading_iloc):
    """Numba加速的策略核心 - 基於28基因系統A策略執行買賣決策"""

    regime_indicator_choice = int(gene_arr[0])
    normal_regime_strat_choice = int(gene_arr[1])
    risk_off_regime_strat_choice = int(gene_arr[2])
    vix_threshold = gene_arr[3]
    sentiment_threshold = gene_arr[4]
    rsi_buy_thr = gene_arr[5]
    rsi_sell_thr = gene_arr[6]
    kd_buy_thr = gene_arr[7]
    kd_sell_thr = gene_arr[8]
    adx_thr = gene_arr[9]

    T = len(prices_arr)
    portfolio_values_arr = np.full(T, 1.0)
    max_trade_signals = T // 2 + 1
    buy_indices_temp = np.full(max_trade_signals, -1, dtype=np.int64)
    buy_prices_temp = np.full(max_trade_signals, np.nan)
    sell_indices_temp = np.full(max_trade_signals, -1, dtype=np.int64)
    sell_prices_temp = np.full(max_trade_signals, np.nan)
    buy_count, sell_count, cash, stock, position = 0, 0, 1.0, 0.0, 0
    last_valid_portfolio_value = 1.0
    entry_strategy_type = -1
    atr_stop_loss_price = 0.0
    rsi_sell_armed = False

    portfolio_values_arr[:start_trading_iloc] = 1.0

    for i in range(start_trading_iloc, T):
        is_risk_off_regime = False
        if regime_indicator_choice == 0:
            if np.isfinite(vix_ma_arr[i]) and vix_ma_arr[i] >= vix_threshold:
                is_risk_off_regime = True
        else:
            if np.isfinite(sentiment_ma_arr[i]) and sentiment_ma_arr[i] <= sentiment_threshold:
                is_risk_off_regime = True

        price = prices_arr[i]
        if not np.isfinite(price):
            portfolio_values_arr[i] = last_valid_portfolio_value
            continue

        buy_condition, sell_condition = False, False

        if position == 0:
            strategy_to_use = risk_off_regime_strat_choice if is_risk_off_regime else normal_regime_strat_choice

            if strategy_to_use == 0:  # MA Cross
                if (np.isfinite(ma_s_arr[i-1]) and np.isfinite(ma_l_arr[i-1]) and
                    ma_s_arr[i-1] < ma_l_arr[i-1] and ma_s_arr[i] >= ma_l_arr[i]):
                    buy_condition = True
            elif strategy_to_use == 1:  # Triple EMA
                if (np.isfinite(ema_s_arr[i]) and np.isfinite(ema_m_arr[i]) and np.isfinite(ema_l_arr[i]) and
                    ema_s_arr[i] > ema_m_arr[i] and ema_m_arr[i] > ema_l_arr[i]):
                    buy_condition = True
            elif strategy_to_use == 2:  # MA+MACD+RSI
                if (np.isfinite(ma_l_arr[i]) and np.isfinite(macd_line_arr[i]) and np.isfinite(rsi_arr[i]) and
                    price > ma_l_arr[i] and macd_line_arr[i] > 0 and
                    macd_line_arr[i] > macd_signal_arr[i] and rsi_arr[i] > 50):
                    buy_condition = True
            elif strategy_to_use == 3:  # EMA+RSI
                if (np.isfinite(ema_l_arr[i]) and np.isfinite(rsi_arr[i]) and
                    price > ema_l_arr[i] and rsi_arr[i] > 50):
                    buy_condition = True
            elif strategy_to_use == 4:  # BB+RSI
                if (np.isfinite(bbl_arr[i]) and np.isfinite(rsi_arr[i]) and
                    price <= bbl_arr[i] and rsi_arr[i] < rsi_buy_thr):
                    buy_condition = True
            elif strategy_to_use == 5:  # BB+ADX
                if (np.isfinite(bbl_arr[i]) and np.isfinite(adx_arr[i]) and
                    price <= bbl_arr[i] and adx_arr[i] > adx_thr):
                    buy_condition = True
            elif strategy_to_use == 6:  # ATR+KD
                if (np.isfinite(k_arr[i-1]) and np.isfinite(d_arr[i-1]) and
                    np.isfinite(atr_arr[i]) and np.isfinite(atr_ma_arr[i]) and
                    k_arr[i] < kd_buy_thr and k_arr[i-1] < d_arr[i-1] and
                    k_arr[i] >= d_arr[i] and atr_arr[i] > atr_ma_arr[i]):
                    buy_condition = True
            elif strategy_to_use == 7:  # BB+MACD
                if (np.isfinite(bbu_arr[i]) and np.isfinite(macd_line_arr[i]) and np.isfinite(macd_signal_arr[i]) and
                    price > bbu_arr[i] and macd_line_arr[i] > macd_signal_arr[i] and macd_line_arr[i] > 0):
                    buy_condition = True

            if buy_condition and price > 1e-9:
                stock = (cash * (1 - commission_rate)) / price
                cash = 0.0
                position = 1
                rsi_sell_armed = False
                entry_strategy_type = strategy_to_use
                if entry_strategy_type == 6 and np.isfinite(atr_arr[i]):
                    atr_stop_loss_price = price - 1.5 * atr_arr[i]
                if buy_count < max_trade_signals:
                    buy_indices_temp[buy_count] = i
                    buy_prices_temp[buy_count] = price
                    buy_count += 1

        elif position == 1:
            sell_strategy_to_use = entry_strategy_type

            if sell_strategy_to_use == 0:  # MA Cross
                if (np.isfinite(ma_s_arr[i-1]) and np.isfinite(ma_l_arr[i-1]) and
                    ma_s_arr[i-1] > ma_l_arr[i-1] and ma_s_arr[i] <= ma_l_arr[i]):
                    sell_condition = True
            elif sell_strategy_to_use == 1:  # Triple EMA
                if (np.isfinite(ema_s_arr[i]) and np.isfinite(ema_m_arr[i]) and
                    ema_s_arr[i] < ema_m_arr[i]):
                    sell_condition = True
            elif sell_strategy_to_use == 2:  # MA+MACD+RSI
                if (np.isfinite(ma_l_arr[i]) and np.isfinite(macd_line_arr[i]) and
                    (price < ma_l_arr[i] or macd_line_arr[i] < macd_signal_arr[i])):
                    sell_condition = True
            elif sell_strategy_to_use == 3:  # EMA+RSI
                if np.isfinite(ema_l_arr[i]) and price < ema_l_arr[i]:
                    sell_condition = True
            elif sell_strategy_to_use == 4:  # BB+RSI
                rsi = rsi_arr[i]
                if np.isfinite(rsi):
                    if rsi >= rsi_sell_thr:
                        rsi_sell_armed = True
                    if rsi_sell_armed and rsi < rsi_sell_thr:
                        sell_condition = True
            elif sell_strategy_to_use == 5:  # BB+ADX
                if np.isfinite(bbm_arr[i]) and price >= bbm_arr[i]:
                    sell_condition = True
            elif sell_strategy_to_use == 6:  # ATR+KD
                if ((np.isfinite(k_arr[i]) and np.isfinite(d_arr[i]) and
                     k_arr[i] > kd_sell_thr and k_arr[i] < d_arr[i]) or
                    (atr_stop_loss_price > 0 and price <= atr_stop_loss_price)):
                    sell_condition = True
            elif sell_strategy_to_use == 7:  # BB+MACD
                if np.isfinite(bbm_arr[i]) and price < bbm_arr[i]:
                    sell_condition = True

            if sell_condition:
                cash = (stock * price) * (1 - commission_rate)
                stock = 0.0
                position = 0
                entry_strategy_type = -1
                rsi_sell_armed = False
                atr_stop_loss_price = 0.0
                if sell_count < max_trade_signals:
                    sell_indices_temp[sell_count] = i
                    sell_prices_temp[sell_count] = price
                    sell_count += 1

        current_portfolio_value = cash + (stock * price)
        if np.isfinite(current_portfolio_value):
            portfolio_values_arr[i] = current_portfolio_value
            last_valid_portfolio_value = current_portfolio_value
        else:
            portfolio_values_arr[i] = last_valid_portfolio_value

    return (portfolio_values_arr,
            buy_indices_temp[:buy_count], buy_prices_temp[:buy_count],
            sell_indices_temp[:sell_count], sell_prices_temp[:sell_count],
            sell_count)

# === 🔥 修復：新增有效基因採樣器 ===
class ValidGASampling(Sampling):
    """自定義的有效基因採樣器，確保與傳統GA一致"""
    
    def __init__(self, ga_params):
        super().__init__()
        self.ga_params = ga_params
        
    def _do(self, problem, n_samples, **kwargs):
        GENE_LENGTH = len(GENE_MAP)
        population = []
        
        # 複製傳統GA的基因生成邏輯
        vix_thr_min, vix_thr_max = self.ga_params['vix_threshold_range']
        sent_thr_min, sent_thr_max = self.ga_params['sentiment_threshold_range']
        rsi_buy_min, rsi_buy_max, rsi_sell_min, rsi_sell_max = self.ga_params['rsi_threshold_range']
        kd_buy_min, kd_buy_max, kd_sell_min, kd_sell_max = self.ga_params['kd_threshold_range']
        adx_thr_min, adx_thr_max = self.ga_params['adx_threshold_range']
        
        p_opts = {key: self.ga_params.get(key, []) for key in STRATEGY_CONFIG_SHARED_GA.keys() if 'options' in key}
        key_mapper = {
            'vix_ma_period_options': 'vix_ma_p', 'sentiment_ma_period_options': 'sentiment_ma_p', 
            'rsi_period_options': 'rsi_p', 'adx_period_options': 'adx_p', 
            'ma_period_options': ['ma_s_p', 'ma_l_p'], 'ema_s_period_options': 'ema_s_p',
            'ema_m_period_options': 'ema_m_p', 'ema_l_period_options': 'ema_l_p', 
            'atr_period_options': 'atr_p', 'kd_k_period_options': 'kd_k_p', 
            'kd_d_period_options': 'kd_d_p', 'kd_smooth_period_options': 'kd_s_p',
            'macd_fast_period_options': 'macd_f_p', 'macd_slow_period_options': 'macd_s_p',
            'macd_signal_period_options': 'macd_sig_p', 'bb_length_options': 'bb_l_p', 
            'bb_std_options': 'bb_s_p',
        }
        
        num_p_opts = {}
        for config_key, gene_key_or_keys in key_mapper.items():
            if config_key in p_opts:
                num_options = len(p_opts[config_key])
                if isinstance(gene_key_or_keys, list):
                    for gene_key in gene_key_or_keys:
                        num_p_opts[gene_key] = num_options
                else:
                    num_p_opts[gene_key_or_keys] = num_options
        
        def is_gene_valid(gene):
            """🔥 完全複製傳統GA的驗證邏輯"""
            # MA期間約束
            ma_s_p_idx = gene[GENE_MAP['ma_s_p']]
            ma_l_p_idx = gene[GENE_MAP['ma_l_p']]
            if p_opts['ma_period_options'][ma_s_p_idx] >= p_opts['ma_period_options'][ma_l_p_idx]:
                return False
                
            # EMA期間約束
            ema_s_p_idx = gene[GENE_MAP['ema_s_p']]
            ema_m_p_idx = gene[GENE_MAP['ema_m_p']]
            ema_l_p_idx = gene[GENE_MAP['ema_l_p']]
            if not (p_opts['ema_s_period_options'][ema_s_p_idx] < 
                   p_opts['ema_m_period_options'][ema_m_p_idx] < 
                   p_opts['ema_l_period_options'][ema_l_p_idx]):
                return False
                
            # MACD期間約束
            macd_f_p_idx = gene[GENE_MAP['macd_f_p']]
            macd_s_p_idx = gene[GENE_MAP['macd_s_p']]
            if p_opts['macd_fast_period_options'][macd_f_p_idx] >= p_opts['macd_slow_period_options'][macd_s_p_idx]:
                return False
                
            return True
        
        # 生成有效基因
        attempts = 0
        max_attempts = n_samples * 1000
        
        print(f"[GAEngine] NSGA-II 正在生成 {n_samples} 個有效基因...")
        
        while len(population) < n_samples and attempts < max_attempts:
            gene = np.zeros(GENE_LENGTH, dtype=int)
            
            # 🔥 完全複製傳統GA的基因生成邏輯
            gene[GENE_MAP['regime_choice']] = random.randint(0, 1)
            gene[GENE_MAP['normal_strat']] = random.randint(0, 7)
            gene[GENE_MAP['risk_off_strat']] = random.randint(0, 7)
            gene[GENE_MAP['vix_thr']] = random.randint(vix_thr_min, vix_thr_max)
            gene[GENE_MAP['sentiment_thr']] = random.randint(sent_thr_min, sent_thr_max)
            gene[GENE_MAP['rsi_buy_thr']] = random.randint(rsi_buy_min, rsi_buy_max)
            gene[GENE_MAP['rsi_sell_thr']] = random.randint(rsi_sell_min, rsi_sell_max)
            gene[GENE_MAP['kd_buy_thr']] = random.randint(kd_buy_min, kd_buy_max)
            gene[GENE_MAP['kd_sell_thr']] = random.randint(kd_sell_min, kd_sell_max)
            gene[GENE_MAP['adx_thr']] = random.randint(adx_thr_min, adx_thr_max)
            
            for key, num_opts in num_p_opts.items():
                if num_opts > 0:
                    gene[GENE_MAP[key]] = random.randint(0, num_opts - 1)
            
            if is_gene_valid(gene):
                population.append(gene)
            
            attempts += 1
        
        if len(population) < n_samples:
            print(f"[GAEngine] WARNING: 只生成了 {len(population)}/{n_samples} 個有效基因")
            # 填充不足的部分
            while len(population) < n_samples:
                if population:
                    population.append(population[0])  # 複製第一個有效基因
                else:
                    # 緊急情況：生成一個基本有效的基因
                    emergency_gene = np.zeros(GENE_LENGTH, dtype=int)
                    emergency_gene[GENE_MAP['ma_s_p']] = 0  # 最短MA
                    emergency_gene[GENE_MAP['ma_l_p']] = len(p_opts.get('ma_period_options', [5, 10])) - 1  # 最長MA
                    emergency_gene[GENE_MAP['ema_s_p']] = 0
                    emergency_gene[GENE_MAP['ema_m_p']] = 0
                    emergency_gene[GENE_MAP['ema_l_p']] = 0
                    emergency_gene[GENE_MAP['macd_f_p']] = 0
                    emergency_gene[GENE_MAP['macd_s_p']] = len(p_opts.get('macd_slow_period_options', [21])) - 1
                    population.append(emergency_gene)
        
        print(f"[GAEngine] NSGA-II 成功生成 {len(population)} 個有效基因（嘗試 {attempts} 次）")
        return np.array(population, dtype=float)  # NSGA-II需要float類型

# === NSGA-II 多目標優化類別 ===
class MultiObjectiveStrategyProblem(Problem):
    """多目標策略優化問題定義 - 包含平均交易報酬率和交易次數約束"""

    def __init__(self, prices, dates, precalculated_indicators, ga_params):
        self.prices = prices
        self.dates = dates
        self.precalculated = precalculated_indicators
        self.ga_params = ga_params

        GENE_LENGTH = len(GENE_MAP)
        xl = np.zeros(GENE_LENGTH)
        xu = np.zeros(GENE_LENGTH)

        # 策略選擇變數
        xl[GENE_MAP['regime_choice']] = 0; xu[GENE_MAP['regime_choice']] = 1
        xl[GENE_MAP['normal_strat']] = 0; xu[GENE_MAP['normal_strat']] = 7
        xl[GENE_MAP['risk_off_strat']] = 0; xu[GENE_MAP['risk_off_strat']] = 7

        # 閾值變數
        vix_min, vix_max = ga_params['vix_threshold_range']
        xl[GENE_MAP['vix_thr']] = vix_min; xu[GENE_MAP['vix_thr']] = vix_max

        sent_min, sent_max = ga_params['sentiment_threshold_range']
        xl[GENE_MAP['sentiment_thr']] = sent_min; xu[GENE_MAP['sentiment_thr']] = sent_max

        rsi_buy_min, rsi_buy_max, rsi_sell_min, rsi_sell_max = ga_params['rsi_threshold_range']
        xl[GENE_MAP['rsi_buy_thr']] = rsi_buy_min; xu[GENE_MAP['rsi_buy_thr']] = rsi_buy_max
        xl[GENE_MAP['rsi_sell_thr']] = rsi_sell_min; xu[GENE_MAP['rsi_sell_thr']] = rsi_sell_max

        kd_buy_min, kd_buy_max, kd_sell_min, kd_sell_max = ga_params['kd_threshold_range']
        xl[GENE_MAP['kd_buy_thr']] = kd_buy_min; xu[GENE_MAP['kd_buy_thr']] = kd_buy_max
        xl[GENE_MAP['kd_sell_thr']] = kd_sell_min; xu[GENE_MAP['kd_sell_thr']] = kd_sell_max

        adx_min, adx_max = ga_params['adx_threshold_range']
        xl[GENE_MAP['adx_thr']] = adx_min; xu[GENE_MAP['adx_thr']] = adx_max

        # 參數選擇變數範圍
        p_opts = {key: ga_params.get(key, []) for key in STRATEGY_CONFIG_SHARED_GA.keys() if 'options' in key}

        key_mapper = {
            'vix_ma_period_options': 'vix_ma_p', 'sentiment_ma_period_options': 'sentiment_ma_p', 'rsi_period_options': 'rsi_p',
            'adx_period_options': 'adx_p', 'ma_period_options': ['ma_s_p', 'ma_l_p'], 'ema_s_period_options': 'ema_s_p',
            'ema_m_period_options': 'ema_m_p', 'ema_l_period_options': 'ema_l_p', 'atr_period_options': 'atr_p',
            'kd_k_period_options': 'kd_k_p', 'kd_d_period_options': 'kd_d_p', 'kd_smooth_period_options': 'kd_s_p',
            'macd_fast_period_options': 'macd_f_p', 'macd_slow_period_options': 'macd_s_p', 'macd_signal_period_options': 'macd_sig_p',
            'bb_length_options': 'bb_l_p', 'bb_std_options': 'bb_s_p',
        }

        for config_key, gene_key_or_keys in key_mapper.items():
            if config_key in p_opts and len(p_opts[config_key]) > 0:
                if isinstance(gene_key_or_keys, list):
                    for gene_key in gene_key_or_keys:
                        if gene_key in GENE_MAP:
                            xl[GENE_MAP[gene_key]] = 0; xu[GENE_MAP[gene_key]] = len(p_opts[config_key]) - 1
                else:
                    if gene_key_or_keys in GENE_MAP:
                        xl[GENE_MAP[gene_key_or_keys]] = 0; xu[GENE_MAP[gene_key_or_keys]] = len(p_opts[config_key]) - 1

        # 4個目標 + 1個約束條件
        n_obj = 4  # 目標：報酬率、回撤、獲利因子、平均交易報酬率
        n_constr = 1  # 約束：交易次數 >= 5

        super().__init__(n_var=GENE_LENGTH, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, type_var=int)

    def _evaluate(self, X, out, *args, **kwargs):
        """評估目標函數和約束條件"""
        objectives = np.zeros((X.shape[0], 4))
        constraints = np.zeros((X.shape[0], 1))

        for i, gene in enumerate(X):
            try:
                portfolio_values, buy_indices, buy_prices, sell_indices, sell_prices, num_trades_from_numba = self._run_backtest_raw(gene)

                metrics = self._calculate_metrics(
                    portfolio_values, buy_indices, buy_prices,
                    sell_indices, sell_prices, num_trades_from_numba,
                    self.ga_params
                )

                # 目標1：最大化總報酬率（轉為最小化負報酬率）
                objectives[i, 0] = -metrics['total_return'] * 1.5  # 增加權重以強調報酬率

                # 目標2：最小化最大回撤
                objectives[i, 1] = metrics['max_drawdown']

                # 目標3：最大化獲利因子（轉為最小化負獲利因子）
                objectives[i, 2] = -metrics['profit_factor']

                # 目標4：最大化平均交易報酬率（轉為最小化負平均交易報酬率）
                objectives[i, 3] = -metrics['average_trade_return']

                # 約束條件：交易次數 >= min_required_trades (約束值 <= 0 表示滿足約束)
                min_trades = self.ga_params.get('min_required_trades', 5)
                constraints[i, 0] = min_trades - metrics['trade_count']

            except Exception as e:
                print(f"[GAEngine] ERROR: 評估基因 {i} 時發生錯誤: {e}")
                objectives[i, :] = [
                    -self.ga_params.get('nsga2_no_trade_penalty_return', 0.5),
                    self.ga_params.get('nsga2_no_trade_penalty_max_drawdown', 1.0),
                    -self.ga_params.get('nsga2_no_trade_penalty_profit_factor', 0.01),
                    -0.001
                ]
                constraints[i, 0] = 10  # 違反約束

        out["F"] = objectives
        out["G"] = constraints

    def _run_backtest_raw(self, gene):
        """執行回測，返回原始數據和計數"""
        def get_indicator_list(name, gene_indices, opt_keys):
            params = [self.ga_params[k][int(gene[g_idx])] for g_idx, k in zip(gene_indices, opt_keys)]
            key = tuple(params) if len(params) > 1 else params[0]
            indicator_data = self.precalculated.get(name, {}).get(key, [np.nan] * len(self.prices))
            return np.array(indicator_data, dtype=np.float64)

        vix_ma_arr = get_indicator_list('vix_ma', [GENE_MAP['vix_ma_p']], ['vix_ma_period_options'])
        sent_ma_arr = get_indicator_list('sentiment_ma', [GENE_MAP['sentiment_ma_p']], ['sentiment_ma_period_options'])
        rsi_arr = get_indicator_list('rsi', [GENE_MAP['rsi_p']], ['rsi_period_options'])
        adx_arr = get_indicator_list('adx', [GENE_MAP['adx_p']], ['adx_period_options'])

        bb_key = (self.ga_params['bb_length_options'][int(gene[GENE_MAP['bb_l_p']])],
                 self.ga_params['bb_std_options'][int(gene[GENE_MAP['bb_s_p']])])
        bbl_arr = np.array(self.precalculated.get('bbl', {}).get(bb_key, [np.nan] * len(self.prices)))
        bbm_arr = np.array(self.precalculated.get('bbm', {}).get(bb_key, [np.nan] * len(self.prices)))
        bbu_arr = np.array(self.precalculated.get('bbu', {}).get(bb_key, [np.nan] * len(self.prices)))

        ma_s_arr = get_indicator_list('ma', [GENE_MAP['ma_s_p']], ['ma_period_options'])
        ma_l_arr = get_indicator_list('ma', [GENE_MAP['ma_l_p']], ['ma_period_options'])

        ema_s_arr = get_indicator_list('ema_s', [GENE_MAP['ema_s_p']], ['ema_s_period_options'])
        ema_m_arr = get_indicator_list('ema_m', [GENE_MAP['ema_m_p']], ['ema_m_period_options'])
        ema_l_arr = get_indicator_list('ema_l', [GENE_MAP['ema_l_p']], ['ema_l_period_options'])

        atr_p = self.ga_params['atr_period_options'][int(gene[GENE_MAP['atr_p']])]
        atr_arr = np.array(self.precalculated.get('atr', {}).get(atr_p, [np.nan] * len(self.prices)))
        atr_ma_arr = np.array(self.precalculated.get('atr_ma', {}).get(atr_p, [np.nan] * len(self.prices)))

        kd_key = (self.ga_params['kd_k_period_options'][int(gene[GENE_MAP['kd_k_p']])],
                 self.ga_params['kd_d_period_options'][int(gene[GENE_MAP['kd_d_p']])],
                 self.ga_params['kd_smooth_period_options'][int(gene[GENE_MAP['kd_s_p']])])
        k_arr = np.array(self.precalculated.get('kd_k', {}).get(kd_key, [np.nan] * len(self.prices)))
        d_arr = np.array(self.precalculated.get('kd_d', {}).get(kd_key, [np.nan] * len(self.prices)))

        macd_key = (self.ga_params['macd_fast_period_options'][int(gene[GENE_MAP['macd_f_p']])],
                   self.ga_params['macd_slow_period_options'][int(gene[GENE_MAP['macd_s_p']])],
                   self.ga_params['macd_signal_period_options'][int(gene[GENE_MAP['macd_sig_p']])])
        macd_line_arr = np.array(self.precalculated.get('macd_line', {}).get(macd_key, [np.nan] * len(self.prices)))
        macd_signal_arr = np.array(self.precalculated.get('macd_signal', {}).get(macd_key, [np.nan] * len(self.prices)))

        portfolio_values, buy_indices_numba, buy_prices_numba, sell_indices_numba, sell_prices_numba, sell_count_numba = run_strategy_numba_core(
            np.array(gene, dtype=np.float64),
            np.array(self.prices, dtype=np.float64),
            vix_ma_arr, sent_ma_arr, rsi_arr, adx_arr,
            bbl_arr, bbm_arr, bbu_arr, ma_s_arr, ma_l_arr,
            ema_s_arr, ema_m_arr, ema_l_arr, atr_arr, atr_ma_arr,
            k_arr, d_arr, macd_line_arr, macd_signal_arr,
            self.ga_params['commission_rate'], 61
        )

        return portfolio_values, buy_indices_numba, buy_prices_numba, sell_indices_numba, sell_prices_numba, sell_count_numba

    def _calculate_metrics(self, portfolio_values, buy_indices, buy_prices, sell_indices, sell_prices, completed_trades_count, ga_params_config_for_penalties):
        """🔥 修復：計算策略績效指標 - 包含交易獎勵機制與平均交易報酬率"""
        
        no_trade_return_penalty = ga_params_config_for_penalties.get('nsga2_no_trade_penalty_return', -0.5)
        no_trade_max_drawdown_penalty = ga_params_config_for_penalties.get('nsga2_no_trade_penalty_max_drawdown', 1.0)
        no_trade_std_dev_penalty = ga_params_config_for_penalties.get('nsga2_no_trade_penalty_std_dev', 1.0)
        no_trade_profit_factor_penalty = ga_params_config_for_penalties.get('nsga2_no_trade_penalty_profit_factor', 0.01)

        if len(portfolio_values) == 0 or not np.isfinite(portfolio_values).any():
            return {
                'total_return': no_trade_return_penalty,
                'max_drawdown': no_trade_max_drawdown_penalty,
                'profit_factor': no_trade_profit_factor_penalty,
                'trade_count': 0,
                'std_dev': no_trade_std_dev_penalty,
                'win_rate_pct': 0.0,
                'average_trade_return': 0.0
            }

        final_value = portfolio_values[-1] if np.isfinite(portfolio_values[-1]) else (portfolio_values[np.isfinite(portfolio_values)][-1] if np.isfinite(portfolio_values).any() else 1.0)
        total_return_actual = final_value - 1.0

        # 🔥 新增：加入交易獎勵機制（與傳統GA一致）
        trade_bonus = 1.0
        min_trades = ga_params_config_for_penalties.get('min_trades_for_full_score', 4)
        no_trade_penalty = ga_params_config_for_penalties.get('no_trade_penalty_factor', 0.1)
        low_trade_penalty = ga_params_config_for_penalties.get('low_trade_penalty_factor', 0.75)
        
        if completed_trades_count == 0:
            trade_bonus = no_trade_penalty
        elif completed_trades_count < min_trades:
            trade_bonus = low_trade_penalty
        
        # 調整總報酬率（模擬傳統GA的適應度計算）
        adjusted_final_value = final_value * trade_bonus
        adjusted_total_return = adjusted_final_value - 1.0

        if completed_trades_count == 0:
            if adjusted_total_return > 0.01:
                portfolio_values_clean = portfolio_values[np.isfinite(portfolio_values)]
                running_max = np.maximum.accumulate(portfolio_values_clean)
                safe_running_max = np.where(running_max == 0, 1, running_max)
                drawdowns = (running_max - portfolio_values_clean) / safe_running_max
                max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
                std_dev = np.std(portfolio_values_clean) if len(portfolio_values_clean) > 1 else 0.001

                return {
                    'total_return': adjusted_total_return,
                    'max_drawdown': max_drawdown,
                    'profit_factor': no_trade_profit_factor_penalty,
                    'trade_count': 0,
                    'std_dev': std_dev,
                    'win_rate_pct': 0.0,
                    'average_trade_return': 0.0
                }
            else:
                return {
                    'total_return': no_trade_return_penalty,
                    'max_drawdown': no_trade_max_drawdown_penalty,
                    'profit_factor': no_trade_profit_factor_penalty,
                    'trade_count': 0,
                    'std_dev': no_trade_std_dev_penalty,
                    'win_rate_pct': 0.0,
                    'average_trade_return': 0.0
                }

        # --- 正常有交易的策略計算 ---
        total_return = adjusted_total_return  # 🔥 使用調整後的報酬率

        portfolio_values_clean = portfolio_values[np.isfinite(portfolio_values)]
        running_max = np.maximum.accumulate(portfolio_values_clean)
        safe_running_max = np.where(running_max == 0, 1, running_max)
        drawdowns = (running_max - portfolio_values_clean) / safe_running_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        std_dev = np.std(portfolio_values_clean) if len(portfolio_values_clean) > 1 else 0.001
         
        # === 【新增】從配置中獲取無風險利率 ===
        risk_free_rate = ga_params_config_for_penalties.get('risk_free_rate', 0.04)
        
        # === 【新增】計算夏普比率 ===
        sharpe_ratio = 0.0
        if std_dev > 1e-9:
            daily_returns = pd.Series(portfolio_values_clean).pct_change().dropna()
            if not daily_returns.empty:
                excess_returns = daily_returns - (risk_free_rate / 252)
                if np.std(excess_returns) > 0:
                    sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)
                    
        total_profit = 0.0
        total_loss = 0.0
        wins = 0

        # 🔥 新增：計算平均交易報酬率
        total_trade_returns = 0.0
        valid_trades = 0

        for i in range(completed_trades_count):
            buy_p = buy_prices[i]
            sell_p = sell_prices[i]
            if np.isfinite(buy_p) and np.isfinite(sell_p) and buy_p > 0:
                trade_return = sell_p - buy_p
                trade_return_pct = trade_return / buy_p

                if trade_return > 0:
                    total_profit += trade_return
                    wins += 1
                else:
                    total_loss += abs(trade_return)

                total_trade_returns += trade_return_pct
                valid_trades += 1

        profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else no_trade_profit_factor_penalty)
        win_rate_pct = (wins / completed_trades_count) * 100 if completed_trades_count > 0 else 0.0

        # 平均交易報酬率
        average_trade_return = total_trade_returns / valid_trades if valid_trades > 0 else 0.0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'profit_factor': max(profit_factor, no_trade_profit_factor_penalty),
            'trade_count': completed_trades_count,
            'std_dev': max(std_dev, 0.001),
            'win_rate_pct': win_rate_pct,
            'average_trade_return': average_trade_return,
            'sharpe_ratio': sharpe_ratio  
        }

# =======================================================================================
# 檔案: ga_engine.py
# 請用以下完整函數替換您檔案中現有的 select_best_from_pareto 函數
# =======================================================================================

def select_best_from_pareto(pareto_genes, pareto_objectives, prices, precalculated_indicators, selection_method='custom_balance', ga_params=GA_PARAMS_CONFIG):
    """從帕累托前沿選擇最佳策略 - 支援平均交易報酬率"""
    if len(pareto_genes) == 0:
        return None, {}

    all_metrics_on_pareto_front = []
    temp_problem_instance = MultiObjectiveStrategyProblem(
        prices=prices, dates=[], precalculated_indicators=precalculated_indicators, ga_params=ga_params
    )

    for gene_idx, gene in enumerate(pareto_genes):
        try:
            portfolio_values, buy_indices, buy_prices, sell_indices, sell_prices, num_trades_actual = temp_problem_instance._run_backtest_raw(np.array(gene))
            metrics = temp_problem_instance._calculate_metrics(
                portfolio_values, buy_indices, buy_prices,
                sell_indices, sell_prices, num_trades_actual, ga_params
            )
            all_metrics_on_pareto_front.append(metrics)
        except Exception as e:
            print(f"[GAEngine] WARN: 計算帕累托解指標時出錯: {gene} -> {e}")
            all_metrics_on_pareto_front.append({
                'total_return': ga_params.get('nsga2_no_trade_penalty_return', -0.5),
                'max_drawdown': ga_params.get('nsga2_no_trade_penalty_max_drawdown', 1.0),
                'profit_factor': ga_params.get('nsga2_no_trade_penalty_profit_factor', 0.01),
                'trade_count': 0,
                'std_dev': ga_params.get('nsga2_no_trade_penalty_std_dev', 1.0),
                'win_rate_pct': 0.0,
                'average_trade_return': 0.0
            })

    best_idx = 0
    if selection_method == 'return':
        best_idx = np.argmax([m['total_return'] for m in all_metrics_on_pareto_front])
        
    elif selection_method == 'average_trade_return':
        best_idx = np.argmax([m['average_trade_return'] for m in all_metrics_on_pareto_front])
        
    elif selection_method == 'expectancy_balanced':
        expectancy_scores = []
        for metrics in all_metrics_on_pareto_front:
            avg_trade_return = metrics['average_trade_return']
            win_rate = metrics['win_rate_pct'] / 100
            total_return = metrics['total_return']
            expectancy = (avg_trade_return * win_rate * 0.6) + (total_return * 0.4)
            expectancy_scores.append(expectancy)
        best_idx = np.argmax(expectancy_scores)
        
    elif selection_method == 'aggressive':
        # 🔥 新增：激進高報酬率選擇方法
        high_return_threshold = 0.30  # 30% 報酬率門檻
        high_return_indices = [
            i for i, m in enumerate(all_metrics_on_pareto_front)
            if m['total_return'] > high_return_threshold and m['trade_count'] >= ga_params.get('min_required_trades', 1)
        ]

        if high_return_indices:
            combined_scores = []
            for i in high_return_indices:
                metrics = all_metrics_on_pareto_front[i]
                score = (metrics['total_return'] * 0.7) + (metrics['average_trade_return'] * 0.3)
                combined_scores.append(score)
            best_relative_idx = np.argmax(combined_scores)
            best_idx = high_return_indices[best_relative_idx]
            print(f"[GAEngine] return_aggressive: 在 {len(high_return_indices)} 個高收益解中選擇最佳 (報酬率 {all_metrics_on_pareto_front[best_idx]['total_return']*100:.2f}%)")
        else:
            valid_solutions = [
                i for i, m in enumerate(all_metrics_on_pareto_front)
                if m['trade_count'] >= ga_params.get('min_required_trades', 1)
            ]
            if valid_solutions:
                returns_in_valid = [all_metrics_on_pareto_front[i]['total_return'] for i in valid_solutions]
                best_relative_idx = np.argmax(returns_in_valid)
                best_idx = valid_solutions[best_relative_idx]
            else:
                best_idx = np.argmax([m['total_return'] for m in all_metrics_on_pareto_front])
                print(f"[GAEngine] return_aggressive: 警告 - 使用不滿足交易次數要求的解")
    
    elif selection_method == 'custom_balance':
        # 🔥🔥🔥 --- 修正後的自定義權重邏輯 --- 🔥🔥🔥
        
        # 從 ga_params 獲取用戶定義的權重，若無則使用預設值
        custom_weights = ga_params.get('custom_weights', {
            'total_return_weight': 0.35, 'avg_trade_return_weight': 0.30,
            'win_rate_weight': 0.25, 'trade_count_weight': 0.05, 'drawdown_weight': 0.05
        })
        print(f"[GAEngine] 使用自定義權重進行選擇: {custom_weights}")

        # 從所有策略中提取各項指標
        all_returns = np.array([m['total_return'] for m in all_metrics_on_pareto_front])
        all_avg_trade_returns = np.array([m['average_trade_return'] for m in all_metrics_on_pareto_front])
        all_win_rates = np.array([m.get('win_rate_pct', 0) for m in all_metrics_on_pareto_front])
        all_trade_counts = np.array([m['trade_count'] for m in all_metrics_on_pareto_front])
        all_max_drawdowns = np.array([m['max_drawdown'] for m in all_metrics_on_pareto_front])

        # 定義正規化函數，將所有指標縮放到 0-1 之間，以便公平比較
        def normalize(arr):
            min_val, max_val = np.min(arr), np.max(arr)
            # 避免除以零
            if (max_val - min_val) > 1e-9:
                return (arr - min_val) / (max_val - min_val)
            else:
                return np.full_like(arr, 0.5) # 如果所有值都相同，則返回中間值

        # 正規化各項指標
        norm_returns = normalize(all_returns)
        norm_avg_trade_returns = normalize(all_avg_trade_returns)
        norm_win_rates = normalize(all_win_rates)
        norm_trade_counts = normalize(all_trade_counts)
        # 對於最大回撤，值越小越好，所以正規化後用1減去，使其變為越大越好
        norm_drawdowns_inv = 1 - normalize(all_max_drawdowns)

        # 根據用戶定義的權重計算每個策略的最終平衡分數
        balanced_scores = (
            norm_returns * custom_weights.get('total_return_weight', 0.35) +
            norm_avg_trade_returns * custom_weights.get('avg_trade_return_weight', 0.30) +
            norm_win_rates * custom_weights.get('win_rate_weight', 0.25) +
            norm_trade_counts * custom_weights.get('trade_count_weight', 0.05) +
            norm_drawdowns_inv * custom_weights.get('drawdown_weight', 0.05)
        )
        
        # 選擇分數最高的策略
        best_idx = np.argmax(balanced_scores)
        
    else:  # 'sharpe' 或其他未定義的方法作為預設
        best_idx = 0
        best_sharpe = -np.inf
        for i, metrics in enumerate(all_metrics_on_pareto_front):
            returns = metrics['total_return']
            std_dev = metrics.get('std_dev', 1) # 使用 .get 避免 KeyError
            sharpe = returns / std_dev if std_dev > 1e-9 and returns > 0 else (0 if returns <= 0 else np.inf)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_idx = i

    best_gene = pareto_genes[best_idx]
    best_metrics = all_metrics_on_pareto_front[best_idx]

    # 計算夏普比率並加入最終結果
    if 'std_dev' in best_metrics and best_metrics['std_dev'] > 1e-9:
        best_metrics['sharpe_ratio'] = best_metrics['total_return'] / best_metrics['std_dev']
    else:
        best_metrics['sharpe_ratio'] = 0

    return best_gene, best_metrics

# --- 統一的遺傳算法函數入口 ---
def genetic_algorithm_unified(prices, dates, precalculated_indicators, ga_params, seed_genes=None):
    """統一的遺傳算法函數，支援 NSGA-II 多目標優化和傳統單目標 GA"""
    use_nsga2 = ga_params.get('nsga2_enabled', False) and NSGA2_AVAILABLE

    if use_nsga2:
        print("[GAEngine] 使用 NSGA-II 多目標優化。")
        return nsga2_optimize(prices, dates, precalculated_indicators, ga_params)
    else:
        print("[GAEngine] 使用傳統單目標遺傳算法。")
        return genetic_algorithm_unified_original(prices, dates, precalculated_indicators, ga_params, seed_genes)

def genetic_algorithm_unified_original(prices, dates, precalculated_indicators, ga_params, seed_genes=None):
    """原有的單目標遺傳算法（保持完全兼容）"""
    GENE_LENGTH = len(GENE_MAP)

    generations, pop_size, crossover_rate, mutation_rate, elitism_size, tournament_size = \
        ga_params['generations'], ga_params['population_size'], ga_params['crossover_rate'], \
        ga_params['mutation_rate'], ga_params['elitism_size'], ga_params['tournament_size']

    show_ga = ga_params.get('show_process', False)
    min_trades = ga_params.get('min_trades_for_full_score', 4)
    no_trade_penalty = ga_params.get('no_trade_penalty_factor', 0.1)
    low_trade_penalty = ga_params.get('low_trade_penalty_factor', 0.75)

    vix_thr_min, vix_thr_max = ga_params['vix_threshold_range']
    sent_thr_min, sent_thr_max = ga_params['sentiment_threshold_range']
    rsi_buy_min, rsi_buy_max, rsi_sell_min, rsi_sell_max = ga_params['rsi_threshold_range']
    kd_buy_min, kd_buy_max, kd_sell_min, kd_sell_max = ga_params['kd_threshold_range']
    adx_thr_min, adx_thr_max = ga_params['adx_threshold_range']

    p_opts = {key: ga_params.get(key, []) for key in STRATEGY_CONFIG_SHARED_GA.keys() if 'options' in key}

    key_mapper = {
        'vix_ma_period_options': 'vix_ma_p', 'sentiment_ma_period_options': 'sentiment_ma_p', 'rsi_period_options': 'rsi_p',
        'adx_period_options': 'adx_p', 'ma_period_options': ['ma_s_p', 'ma_l_p'], 'ema_s_period_options': 'ema_s_p',
        'ema_m_period_options': 'ema_m_p', 'ema_l_period_options': 'ema_l_p', 'atr_period_options': 'atr_p',
        'kd_k_period_options': 'kd_k_p', 'kd_d_period_options': 'kd_d_p', 'kd_smooth_period_options': 'kd_s_p',
        'macd_fast_period_options': 'macd_f_p', 'macd_slow_period_options': 'macd_s_p', 'macd_signal_period_options': 'macd_sig_p',
        'bb_length_options': 'bb_l_p', 'bb_std_options': 'bb_s_p',
    }

    num_p_opts = {}
    for config_key, gene_key_or_keys in key_mapper.items():
        if config_key in p_opts:
            num_options = len(p_opts[config_key])
            if isinstance(gene_key_or_keys, list):
                for gene_key in gene_key_or_keys:
                    num_p_opts[gene_key] = num_options
            else:
                num_p_opts[gene_key_or_keys] = num_options

    def is_gene_valid(gene):
        ma_s_p_idx = gene[GENE_MAP['ma_s_p']]
        ma_l_p_idx = gene[GENE_MAP['ma_l_p']]
        if p_opts['ma_period_options'][ma_s_p_idx] >= p_opts['ma_period_options'][ma_l_p_idx]:
            return False

        ema_s_p_idx = gene[GENE_MAP['ema_s_p']]
        ema_m_p_idx = gene[GENE_MAP['ema_m_p']]
        ema_l_p_idx = gene[GENE_MAP['ema_l_p']]
        if not (p_opts['ema_s_period_options'][ema_s_p_idx] < p_opts['ema_m_period_options'][ema_m_p_idx] < p_opts['ema_l_period_options'][ema_l_p_idx]):
            return False

        macd_f_p_idx = gene[GENE_MAP['macd_f_p']]
        macd_s_p_idx = gene[GENE_MAP['macd_s_p']]
        if p_opts['macd_fast_period_options'][macd_f_p_idx] >= p_opts['macd_slow_period_options'][macd_s_p_idx]:
            return False

        return True

    population = []
    if seed_genes and isinstance(seed_genes, list):
        for seed in seed_genes:
            if isinstance(seed, list) and len(seed) == GENE_LENGTH and is_gene_valid(seed):
                population.append(seed)

    if population and show_ga:
        print(f"[GAEngine] 成功注入 {len(population)} 個種子基因到初始種群中。")

    for _ in range(pop_size * 500):
        if len(population) >= pop_size:
            break

        gene = np.zeros(GENE_LENGTH, dtype=int)
        gene[GENE_MAP['regime_choice']] = random.randint(0, 1)
        gene[GENE_MAP['normal_strat']] = random.randint(0, 7)
        gene[GENE_MAP['risk_off_strat']] = random.randint(0, 7)
        gene[GENE_MAP['vix_thr']] = random.randint(vix_thr_min, vix_thr_max)
        gene[GENE_MAP['sentiment_thr']] = random.randint(sent_thr_min, sent_thr_max)
        gene[GENE_MAP['rsi_buy_thr']] = random.randint(rsi_buy_min, rsi_buy_max)
        gene[GENE_MAP['rsi_sell_thr']] = random.randint(rsi_sell_min, rsi_sell_max)
        gene[GENE_MAP['kd_buy_thr']] = random.randint(kd_buy_min, kd_buy_max)
        gene[GENE_MAP['kd_sell_thr']] = random.randint(kd_sell_min, kd_sell_max)
        gene[GENE_MAP['adx_thr']] = random.randint(adx_thr_min, adx_thr_max)

        for key, num_opts in num_p_opts.items():
            if num_opts > 0:
                gene[GENE_MAP[key]] = random.randint(0, num_opts - 1)

        if is_gene_valid(gene):
            population.append(gene.tolist())

    if len(population) < pop_size:
        print(f"[GAEngine] ERROR: 無法生成足夠的初始種群 ({len(population)}/{pop_size}).")
        return None, 0

    best_gene_overall, best_fitness_overall = population[0], -np.inf

    for generation in range(generations):
        fitness_scores = []
        for gene in population:
            def get_precalc_list(indicator_name, key):
                return precalculated_indicators.get(indicator_name, {}).get(key, [np.nan] * len(prices))

            vix_ma_arr = np.array(get_precalc_list('vix_ma', p_opts['vix_ma_period_options'][gene[GENE_MAP['vix_ma_p']]]))
            sent_ma_arr = np.array(get_precalc_list('sentiment_ma', p_opts['sentiment_ma_period_options'][gene[GENE_MAP['sentiment_ma_p']]]))
            rsi_arr = np.array(get_precalc_list('rsi', p_opts['rsi_period_options'][gene[GENE_MAP['rsi_p']]]))
            adx_arr = np.array(get_precalc_list('adx', p_opts['adx_period_options'][gene[GENE_MAP['adx_p']]]))

            bb_key = (p_opts['bb_length_options'][gene[GENE_MAP['bb_l_p']]], p_opts['bb_std_options'][gene[GENE_MAP['bb_s_p']]])
            bbl_arr, bbm_arr, bbu_arr = np.array(get_precalc_list('bbl', bb_key)), np.array(get_precalc_list('bbm', bb_key)), np.array(get_precalc_list('bbu', bb_key))

            ma_s_arr, ma_l_arr = np.array(get_precalc_list('ma', p_opts['ma_period_options'][gene[GENE_MAP['ma_s_p']]])), np.array(get_precalc_list('ma', p_opts['ma_period_options'][gene[GENE_MAP['ma_l_p']]]))

            ema_s_arr, ema_m_arr, ema_l_arr = np.array(get_precalc_list('ema_s', p_opts['ema_s_period_options'][gene[GENE_MAP['ema_s_p']]])), np.array(get_precalc_list('ema_m', p_opts['ema_m_period_options'][gene[GENE_MAP['ema_m_p']]])), np.array(get_precalc_list('ema_l', p_opts['ema_l_period_options'][gene[GENE_MAP['ema_l_p']]]))

            atr_p = p_opts['atr_period_options'][gene[GENE_MAP['atr_p']]]
            atr_arr, atr_ma_arr = np.array(get_precalc_list('atr', atr_p)), np.array(get_precalc_list('atr_ma', atr_p))

            kd_key = (p_opts['kd_k_period_options'][gene[GENE_MAP['kd_k_p']]], p_opts['kd_d_period_options'][gene[GENE_MAP['kd_d_p']]], p_opts['kd_smooth_period_options'][gene[GENE_MAP['kd_s_p']]])
            k_arr, d_arr = np.array(get_precalc_list('kd_k', kd_key)), np.array(get_precalc_list('kd_d', kd_key))

            macd_key = (p_opts['macd_fast_period_options'][gene[GENE_MAP['macd_f_p']]], p_opts['macd_slow_period_options'][gene[GENE_MAP['macd_s_p']]], p_opts['macd_signal_period_options'][gene[GENE_MAP['macd_sig_p']]])
            macd_line_arr, macd_signal_arr = np.array(get_precalc_list('macd_line', macd_key)), np.array(get_precalc_list('macd_signal', macd_key))

            portfolio_values, buy_indices_numba, buy_prices_numba, sell_indices_numba, sell_prices_numba, num_trades_from_numba = run_strategy_numba_core(
                np.array(gene, dtype=np.float64), np.array(prices, dtype=np.float64), vix_ma_arr, sent_ma_arr, rsi_arr, adx_arr,
                bbl_arr, bbm_arr, bbu_arr, ma_s_arr, ma_l_arr, ema_s_arr, ema_m_arr, ema_l_arr, atr_arr, atr_ma_arr,
                k_arr, d_arr, macd_line_arr, macd_signal_arr, ga_params['commission_rate'], 61
            )

            final_value = portfolio_values[-1]
            trade_bonus = 1.0
            if num_trades_from_numba == 0:
                trade_bonus = no_trade_penalty
            elif num_trades_from_numba < min_trades:
                trade_bonus = low_trade_penalty

            fitness = final_value * trade_bonus
            fitness_scores.append(fitness if np.isfinite(fitness) else -np.inf)

        valid_indices = np.where(np.isfinite(fitness_scores))[0]
        if len(valid_indices) == 0:
            continue

        fitness_array = np.array(fitness_scores)
        sorted_indices = valid_indices[np.argsort(fitness_array[valid_indices])[::-1]]

        elites = [population[i] for i in sorted_indices[:elitism_size]]
        current_best_fitness = fitness_array[sorted_indices[0]]
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_gene_overall = elites[0]

        if show_ga and (generation + 1) % 5 == 0:
            print(f"[GAEngine] Gen {generation+1}/{generations} | Best Fitness: {best_fitness_overall:.4f}")

        new_population = elites[:]

        while len(new_population) < pop_size:
            p1_idx = np.random.choice(sorted_indices, size=tournament_size)
            p2_idx = np.random.choice(sorted_indices, size=tournament_size)
            parent1 = population[p1_idx[np.argmax(fitness_array[p1_idx])]]
            parent2 = population[p2_idx[np.argmax(fitness_array[p2_idx])]]

            child1, child2 = parent1[:], parent2[:]

            if random.random() < crossover_rate:
                crossover_point = random.randint(1, GENE_LENGTH - 2)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]

            for child in [child1, child2]:
                if random.random() < mutation_rate:
                    idx_to_mutate = random.randint(0, GENE_LENGTH - 1)
                    if GENE_MAP['vix_thr'] <= idx_to_mutate <= GENE_MAP['adx_thr']:
                        child[idx_to_mutate] += random.randint(-5, 5)
                        child[GENE_MAP['vix_thr']] = max(vix_thr_min, min(child[GENE_MAP['vix_thr']], vix_thr_max))
                        child[GENE_MAP['sentiment_thr']] = max(sent_thr_min, min(child[GENE_MAP['sentiment_thr']], sent_thr_max))
                        child[GENE_MAP['rsi_buy_thr']] = max(rsi_buy_min, min(child[GENE_MAP['rsi_buy_thr']], rsi_buy_max))
                        child[GENE_MAP['rsi_sell_thr']] = max(rsi_sell_min, min(child[GENE_MAP['rsi_sell_thr']], rsi_sell_max))
                        child[GENE_MAP['kd_buy_thr']] = max(kd_buy_min, min(child[GENE_MAP['kd_buy_thr']], kd_buy_max))
                        child[GENE_MAP['kd_sell_thr']] = max(kd_sell_min, min(child[GENE_MAP['kd_sell_thr']], kd_sell_max))
                        child[GENE_MAP['adx_thr']] = max(adx_thr_min, min(child[GENE_MAP['adx_thr']], adx_thr_max))
                    else:
                        key = next((k for k, v in GENE_MAP.items() if v == idx_to_mutate), None)
                        if key in num_p_opts and num_p_opts[key] > 0:
                            child[idx_to_mutate] = random.randint(0, num_p_opts[key] - 1)

                if is_gene_valid(child1):
                    new_population.append(child1)
                if len(new_population) < pop_size and is_gene_valid(child2):
                    new_population.append(child2)

        population = new_population[:pop_size]

    return best_gene_overall, best_fitness_overall

def nsga2_optimize(prices, dates, precalculated_indicators, ga_params):
    """NSGA-II 多目標優化主函數 - 完整修復版"""
    if not NSGA2_AVAILABLE:
        print("[GAEngine] ERROR: NSGA-II 不可用，無法執行多目標優化。")
        return None, None

    print("[GAEngine] 開始 NSGA-II 多目標優化...")

    try:
        problem = MultiObjectiveStrategyProblem(prices, dates, precalculated_indicators, ga_params)

        algorithm = NSGA2(
            pop_size=ga_params.get('population_size', 50),
            sampling=ValidGASampling(ga_params),
            crossover=SBX(prob=ga_params.get('crossover_rate', 0.8), eta=15),
            mutation=PM(prob=ga_params.get('mutation_rate', 0.1), eta=20),
            eliminate_duplicates=True
        )

        res = minimize(
            problem,
            algorithm,
            ('n_gen', ga_params.get('generations', 20)),
            verbose=ga_params.get('show_process', False)
        )

        if res.X is None or len(res.X) == 0:
            print("[GAEngine] WARN: NSGA-II 未找到有效解。")
            return None, None

        pareto_genes = [gene.astype(int).tolist() for gene in res.X]
        pareto_objectives = res.F

        best_gene, best_performance_metrics = select_best_from_pareto(
            pareto_genes, pareto_objectives, prices, precalculated_indicators,
            ga_params.get('nsga2_selection_method', 'custom_balance'), ga_params
        )

        if best_gene is None:
            print("[GAEngine] WARN: 從帕累托前沿選擇最佳解失敗。")
            return None, None

        print(f"[GAEngine] NSGA-II 完成，找到 {len(pareto_genes)} 個帕累托最佳解。")
        print(f"[GAEngine] 選中最佳解 (方法: {ga_params.get('nsga2_selection_method', 'custom_balance')}): "
              f"報酬率={best_performance_metrics.get('total_return', 0)*100:.2f}%, "
              f"回撤={best_performance_metrics.get('max_drawdown', 0)*100:.2f}%, "
              f"平均交易報酬={best_performance_metrics.get('average_trade_return', 0)*100:.3f}%, "
              f"交易次數={best_performance_metrics.get('trade_count', 0)}")

        return best_gene, best_performance_metrics

    except Exception as e:
        print(f"[GAEngine] ERROR: NSGA-II 優化過程中發生錯誤: {e}")
        traceback.print_exc()
        return None, None

# --- 其餘輔助函數保持不變 ---
def check_ga_buy_signal_at_latest_point(
    gene, current_price_latest, vix_ma_latest, rsi_latest, bbl_latest, adx_latest,
    ma_short_latest, ma_long_latest, ma_short_prev, ma_long_prev):
    """在最新數據點檢查買入信號"""
    try:
        regime_choice, normal_strat, risk_off_strat = int(gene[0]), int(gene[1]), int(gene[2])
        vix_threshold, sentiment_threshold = gene[3], gene[4]
        rsi_buy_thr, _ = gene[5], gene[6]
        kd_buy_thr, _ = gene[7], gene[8]
        adx_thr = gene[9]

        all_values = [
            current_price_latest, vix_ma_latest, rsi_latest, bbl_latest, adx_latest,
            ma_short_latest, ma_long_latest, ma_short_prev, ma_long_prev
        ]

        if any(val is None or (isinstance(val, float) and not np.isfinite(val)) for val in all_values):
            return False

        is_risk_off_regime = (regime_choice == 0 and vix_ma_latest >= vix_threshold)
        strategy_to_use = risk_off_strat if is_risk_off_regime else normal_strat

        buy_condition = False

        if strategy_to_use == 0:
            if ma_short_prev < ma_long_prev and ma_short_latest >= ma_long_latest:
                buy_condition = True
        elif strategy_to_use == 4:
            if current_price_latest <= bbl_latest and rsi_latest < rsi_buy_thr:
                buy_condition = True
        elif strategy_to_use == 5:
            if current_price_latest <= bbl_latest and adx_latest > adx_thr:
                buy_condition = True

        return buy_condition and current_price_latest > 1e-9

    except (IndexError, TypeError):
        return False

def format_ga_gene_parameters_to_text(gene):
    """
    
    將系統A基因參數轉換為詳細、統一且易於理解的中文策略描述。
    """
    try:
        if not gene or len(gene) != len(GENE_MAP):
            return "基因格式錯誤 (長度不符)"

        # --------------------------------------------------
        # 1. 解析基因，獲取所有需要的參數
        # --------------------------------------------------
        config = GA_PARAMS_CONFIG
        
        # 市場狀態判斷
        regime_choice = gene[GENE_MAP['regime_choice']]
        regime_indicator_name = "VIX波動率指標" if regime_choice == 0 else "市場情緒指標"
        
        if regime_choice == 0:
            regime_threshold = gene[GENE_MAP['vix_thr']]
            vix_ma_period = config['vix_ma_period_options'][gene[GENE_MAP['vix_ma_p']]]
            regime_indicator_details = f"VIX {vix_ma_period}日均線"
            regime_condition_desc = f"≥ {regime_threshold}"
        else:
            regime_threshold = gene[GENE_MAP['sentiment_thr']]
            sentiment_ma_period = config['sentiment_ma_period_options'][gene[GENE_MAP['sentiment_ma_p']]]
            regime_indicator_details = f"市場情緒 {sentiment_ma_period}週均線"
            regime_condition_desc = f"≤ {regime_threshold}"

        # 策略選擇
        low_vol_strat_idx = gene[GENE_MAP['normal_strat']]
        high_vol_strat_idx = gene[GENE_MAP['risk_off_strat']]
        
        # --------------------------------------------------
        # 2. 為每種策略生成簡潔的描述和參數
        # --------------------------------------------------
        
        def get_strategy_description(strat_idx):
            """輔助函式：根據策略索引生成描述和所需參數"""
            
            # 買入條件描述
            buy_desc = ""
            # 賣出條件描述
            sell_desc = ""
            # 關鍵參數描述
            params_desc = ""
            
            gene_map = GENE_MAP
            
            if strat_idx == 0: # MA Cross
                ma_s = config['ma_period_options'][gene[gene_map['ma_s_p']]]
                ma_l = config['ma_period_options'][gene[gene_map['ma_l_p']]]
                buy_desc = f"短期均線({ma_s}日)上穿長期均線({ma_l}日)。"
                sell_desc = "均線死叉時賣出。"
                params_desc = f"均線交叉 ({ma_s}日 vs {ma_l}日)"
            elif strat_idx == 1: # Triple EMA
                ema_s = config['ema_s_period_options'][gene[gene_map['ema_s_p']]]
                ema_m = config['ema_m_period_options'][gene[gene_map['ema_m_p']]]
                ema_l = config['ema_l_period_options'][gene[gene_map['ema_l_p']]]
                buy_desc = f"短期EMA({ema_s}日) > 中期EMA({ema_m}日) > 長期EMA({ema_l}日)形成多頭排列。"
                sell_desc = "短期EMA下穿中期EMA時賣出。"
                params_desc = f"三重EMA ({ema_s}/{ema_m}/{ema_l}日)"
            elif strat_idx == 2: # MA+MACD+RSI
                ma_l = config['ma_period_options'][gene[gene_map['ma_l_p']]]
                rsi_p = config['rsi_period_options'][gene[gene_map['rsi_p']]]
                buy_desc = f"價格高於{ma_l}日均線，且MACD金叉、RSI({rsi_p}日)強勢。"
                sell_desc = "價格跌破長期均線或MACD死叉時賣出。"
                params_desc = f"均線({ma_l}日), MACD, RSI({rsi_p}日)"
            elif strat_idx == 3: # EMA+RSI
                ema_l = config['ema_l_period_options'][gene[gene_map['ema_l_p']]]
                rsi_p = config['rsi_period_options'][gene[gene_map['rsi_p']]]
                buy_desc = f"價格確認站穩於長期EMA({ema_l}日)之上，且RSI({rsi_p}日)顯示上漲動能。"
                sell_desc = "價格跌破長期EMA時賣出。"
                params_desc = f"長期EMA({ema_l}日), RSI({rsi_p}日)"
            elif strat_idx == 4: # BB+RSI
                bb_l = config['bb_length_options'][gene[gene_map['bb_l_p']]]
                bb_s = config['bb_std_options'][gene[gene_map['bb_s_p']]]
                rsi_p = config['rsi_period_options'][gene[gene_map['rsi_p']]]
                rsi_buy = gene[gene_map['rsi_buy_thr']]
                rsi_sell = gene[gene_map['rsi_sell_thr']]
                buy_desc = f"價格觸及布林帶下軌，且RSI({rsi_p}日)進入超賣區(<{rsi_buy})。"
                sell_desc = f"RSI進入超買區(>{rsi_sell})後回落時賣出。"
                params_desc = f"布林帶({bb_l}日, {bb_s}x), RSI({rsi_p}日, 買<{rsi_buy})"
            elif strat_idx == 5: # BB+ADX
                bb_l = config['bb_length_options'][gene[gene_map['bb_l_p']]]
                bb_s = config['bb_std_options'][gene[gene_map['bb_s_p']]]
                adx_p = config['adx_period_options'][gene[gene_map['adx_p']]]
                adx_thr = gene[gene_map['adx_thr']]
                buy_desc = f"價格觸及布林帶下軌，且ADX({adx_p}日)高於{adx_thr}確認趨勢強度。"
                sell_desc = "價格回歸至布林帶中軌時賣出。"
                params_desc = f"布林帶({bb_l}日, {bb_s}x), ADX({adx_p}日, >{adx_thr})"
            elif strat_idx == 6: # ATR+KD
                kd_buy = gene[gene_map['kd_buy_thr']]
                kd_sell = gene[gene_map['kd_sell_thr']]
                buy_desc = f"KD指標在低檔(K<{kd_buy})發生黃金交叉，且市場波動率放大。"
                sell_desc = f"KD指標進入高檔(K>{kd_sell})或觸發ATR移動停損時賣出。"
                params_desc = f"KD指標 (買<{kd_buy}), ATR波動率"
            elif strat_idx == 7: # BB+MACD
                bb_l = config['bb_length_options'][gene[gene_map['bb_l_p']]]
                bb_s = config['bb_std_options'][gene[gene_map['bb_s_p']]]
                buy_desc = "價格強勢突破布林帶上軌，且MACD指標確認上漲動能。"
                sell_desc = "價格回落至布林帶中軌以下時賣出。"
                params_desc = f"布林帶({bb_l}日, {bb_s}x), MACD"
            else:
                return "未知策略", "未知", "未知"

            return buy_desc, sell_desc, params_desc

        low_vol_buy, low_vol_sell, low_vol_params = get_strategy_description(low_vol_strat_idx)
        high_vol_buy, high_vol_sell, high_vol_params = get_strategy_description(high_vol_strat_idx)

        # --------------------------------------------------
        # 3. 組合最終的描述字串
        # --------------------------------------------------

        # 策略標籤
        strategy_styles = {
            "趨勢追蹤型": [0, 1, 2, 3, 7],
            "反轉交易型": [4, 5, 6]
        }
        
        style = "混合型"
        if low_vol_strat_idx in strategy_styles["趨勢追蹤型"] and high_vol_strat_idx in strategy_styles["趨勢追蹤型"]:
            style = "趨勢追蹤型"
        elif low_vol_strat_idx in strategy_styles["反轉交易型"] and high_vol_strat_idx in strategy_styles["反轉交易型"]:
            style = "反轉交易型"

        strategy_tag = f"波動率切換型 {style} 策略"

        # 組合輸出
        description = f"""

核心邏輯:
• 根據市場風險變化，在不同交易邏輯間自動切換。
• 使用 {regime_indicator_name} 判斷市場為「高波動」或「低波動」狀態。

進場條件:
• [低波動市場]: {low_vol_buy}
• [高波動市場]: {high_vol_buy}

出場條件:
• [低波動市場]: {low_vol_sell}
• [高波動市場]: {high_vol_sell}

關鍵參數:
• 市場狀態指標: {regime_indicator_details} (閾值: {regime_condition_desc})
• 低波動指標: {low_vol_params}
• 高波動指標: {high_vol_params}"""

        return description

    except Exception as e:
        return f"策略參數解析錯誤：{str(e)}"

def check_module_integrity():
    """檢查模組完整性和依賴項"""
    print("[GAEngine] === 模組完整性檢查 ===")
    
    # 檢查核心函數
    required_functions = [
        'ga_load_data', 'ga_precompute_indicators', 'genetic_algorithm_unified',
        'run_strategy_numba_core', 'format_ga_gene_parameters_to_text'
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if func_name not in globals():
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"[GAEngine] ❌ 缺少函數: {missing_functions}")
        return False
    else:
        print("[GAEngine] ✅ 所有核心函數完整")
    
    # 檢查配置
    required_configs = ['GENE_MAP', 'STRATEGY_CONFIG_SHARED_GA', 'GA_PARAMS_CONFIG', 'STRAT_NAMES']
    missing_configs = []
    for config_name in required_configs:
        if config_name not in globals():
            missing_configs.append(config_name)
    
    if missing_configs:
        print(f"[GAEngine] ❌ 缺少配置: {missing_configs}")
        return False
    else:
        print("[GAEngine] ✅ 所有配置完整")
    
    # 檢查 NSGA-II 支援
    print(f"[GAEngine] NSGA-II 支援: {'✅ 可用' if NSGA2_AVAILABLE else '❌ 不可用 (請安裝 pymoo)'}")
    
    # 檢查 Numba 支援
    try:
        import numba
        print(f"[GAEngine] ✅ Numba 已安裝: v{numba.__version__}")
    except ImportError:
        print("[GAEngine] ❌ Numba 未安裝 (性能將大幅下降)")
        return False
    
    # 檢查其他依賴
    dependencies = {
        'pandas': 'pd', 'numpy': 'np', 'yfinance': 'yf', 
        'pandas_ta': 'ta', 'datetime': 'dt_datetime'
    }
    
    for dep_name, alias in dependencies.items():
        try:
            if alias in globals():
                print(f"[GAEngine] ✅ {dep_name} 已載入")
            else:
                print(f"[GAEngine] ⚠️  {dep_name} 可能未正確載入")
        except:
            print(f"[GAEngine] ❌ {dep_name} 載入失敗")
    
    print("[GAEngine] === 檢查完成 ===")
    print(f"[GAEngine] 基因長度: {len(GENE_MAP)} 位")
    print(f"[GAEngine] 策略數量: {len(STRAT_NAMES)} 種")
    print(f"[GAEngine] 模組版本: v2.1 (修復版)")
    
    return True

# 檢查模組完整性
if __name__ == "__main__":
    check_module_integrity()
else:
    print("[GAEngine] ga_engine.py v2.1 模組已載入完成（修復版）")
    print("[GAEngine] 修復功能：NSGA-II 基因採樣器 + 交易獎勵機制 + 平均交易報酬率")
