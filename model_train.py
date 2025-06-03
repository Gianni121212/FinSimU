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
# 設定 matplotlib 支持中文顯示（如果需要且已安裝字體）
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 或其他中文字體如 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# --- Load Stock and VIX Data ---
def load_stock_data(ticker, vix_ticker="^VIX", start_date=None, end_date=None):
    """
    從 yfinance 載入股票和 VIX 數據，進行對齊和清理。
    (已更新 fillna 用法)
    返回:
        prices (list): 收盤價列表。
        dates (list): 日期列表。
        final_stock_df (pd.DataFrame): 對齊後的股票數據 DataFrame。
        final_vix_series (pd.Series): 對齊後的 VIX 收盤價 Series。
    """
    print(f"嘗試載入 {ticker} 和 {vix_ticker} 從 {start_date} 到 {end_date} 的數據...")
    try:
        # 下載數據
        data = yf.download([ticker, vix_ticker], start=start_date, end=end_date, progress=False, auto_adjust=False)

        # --- 健壯性檢查 ---
        if data is None or data.empty:
            print(f"錯誤: yfinance.download 返回了 None 或空的 DataFrame ({ticker})。")
            return None, None, None, None
        if not isinstance(data.columns, pd.MultiIndex):
            print(f"錯誤: 預期為 MultiIndex 列，但得到 {data.columns} ({ticker})。請檢查代碼。")
            return None, None, None, None
        # 檢查股票和 VIX 數據是否存在
        stock_columns_present = ticker in data.columns.get_level_values(1)
        vix_columns_present = vix_ticker in data.columns.get_level_values(1)
        if not stock_columns_present:
            print(f"錯誤: 缺少股票數據 {ticker}。")
            return None, None, None, None
        if not vix_columns_present:
            print(f"錯誤: 缺少 VIX 數據 {vix_ticker} (用於 {ticker} 分析)。")
            vix_data = pd.Series(np.nan, index=data.index) # 創建一個 NaN Series
            vix_data.name = 'VIX_Close'
            print(f"警告: 缺少 VIX 數據 {vix_ticker}，將使用 NaN 值。VIX相關策略將失效。")
        else:
             # 提取 VIX 收盤價
            vix_data_slice = data.loc[:, pd.IndexSlice['Close', vix_ticker]]
            if isinstance(vix_data_slice, pd.DataFrame):
                if vix_data_slice.shape[1] == 1:
                    vix_data = vix_data_slice.iloc[:, 0]
                else:
                     print(f"錯誤: 無法將 VIX 收盤價提取為 Series (形狀: {vix_data_slice.shape}) ({ticker})。")
                     return None, None, None, None
            elif isinstance(vix_data_slice, pd.Series):
                 vix_data = vix_data_slice
            else:
                 print(f"錯誤: 無法識別的 VIX 數據類型: {type(vix_data_slice)} ({ticker})。")
                 return None, None, None, None
            vix_data.name = 'VIX_Close'

        # 提取股票數據
        stock_data = data.loc[:, pd.IndexSlice[:, ticker]]
        stock_data.columns = stock_data.columns.droplevel(1) # 移除 MultiIndex 的頂層

        # 檢查必需的股票列
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        missing_cols = [col for col in required_cols if col not in stock_data.columns]
        if not all(col in stock_data.columns for col in ['Close', 'High', 'Low']):
            print(f"錯誤: 缺少計算 ADX 所需的基礎列 (Close, High, Low) ({ticker})。")
            return None, None, None, None
        for col in missing_cols:
            if col not in ['Close', 'High', 'Low']:
                stock_data[col] = np.nan
                print(f"警告: 股票數據缺少 '{col}' 列，已用 NaN 填充 ({ticker})。")

        # 選擇需要的列並複製
        simplified_df = stock_data[required_cols].copy()

        # 合併與對齊 (Inner Join)
        aligned_data = pd.concat([simplified_df, vix_data], axis=1, join='inner')

        if aligned_data.empty:
            print(f"錯誤: 股票和 VIX 數據之間沒有共同的日期 ({ticker})。")
            return None, None, None, None

        # 處理對齊後的 NaN 值 (非常重要)
        if aligned_data.isnull().values.any():
            print(f"警告: 對齊後發現 NaN 值，正在嘗試填充 ({ticker})...")
            # *** 修正 fillna 用法 ***
            if aligned_data['VIX_Close'].isnull().any():
                aligned_data['VIX_Close'] = aligned_data['VIX_Close'].ffill().bfill()
            numeric_stock_cols = simplified_df.select_dtypes(include=np.number).columns
            aligned_data[numeric_stock_cols] = aligned_data[numeric_stock_cols].ffill().bfill()
            # *** 修正結束 ***

            # 再次檢查是否還有 NaN
            if aligned_data.isnull().values.any():
                rows_with_nan = aligned_data[aligned_data.isnull().any(axis=1)]
                print(f"錯誤: 無法填充所有 NaN 值。剩餘 NaN 的行示例:\n{rows_with_nan.head()} ({ticker})。")
                return None, None, None, None # 更安全的做法是報錯

        # 提取最終數據
        final_stock_df = aligned_data[required_cols]
        final_vix_series = aligned_data['VIX_Close']
        prices = final_stock_df['Close'].tolist()
        dates = final_stock_df.index.tolist() # 使用對齊後的索引

        print(f"成功載入並對齊 {len(prices)} 個數據點 ({ticker})。")
        return prices, dates, final_stock_df, final_vix_series

    except Exception as e:
        print(f"\n載入數據時發生錯誤 ({ticker}): {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None, None, None


# --- Indicator Pre-calculation (保持不變) ---
def precompute_indicators(stock_df, vix_series, strategy_config):
    """
    預先計算所有策略需要的指標變體。
    """
    precalculated_rsi = {}
    precalculated_vix_ma = {}
    precalculated_bbl = {}
    precalculated_bbm = {}
    precalculated_fixed = {}
    indicators_ready = True
    print(f"開始預計算指標 (Stock DF shape: {stock_df.shape}, VIX Series len: {len(vix_series)})...")
    try:
        # RSI
        for rsi_p in strategy_config['rsi_period_options']:
            rsi_values = ta.rsi(stock_df['Close'], length=rsi_p)
            if rsi_values is None or rsi_values.isnull().all():
                print(f"錯誤: RSI 計算失敗 (週期 {rsi_p})")
                indicators_ready = False; break
            precalculated_rsi[rsi_p] = rsi_values.tolist()
        if not indicators_ready: return {}, False
        print(f"  RSI ({len(precalculated_rsi)} variants) OK.")
        # VIX MA
        for vix_ma_p in strategy_config['vix_ma_period_options']:
            vix_ma_values = vix_series.rolling(window=vix_ma_p).mean()
            if vix_ma_values is None or vix_ma_values.isnull().all():
                # print(f"錯誤: VIX MA 計算失敗 (週期 {vix_ma_p})") # 允許失敗
                precalculated_vix_ma[vix_ma_p] = [np.nan] * len(vix_series) # 用 NaN 填充
                print(f"  警告: VIX MA (週期 {vix_ma_p}) 計算失敗，將使用 NaN。")
            else:
                precalculated_vix_ma[vix_ma_p] = vix_ma_values.tolist()
        print(f"  VIX MA ({len(precalculated_vix_ma)} variants) OK (or filled with NaN).")
        # Bollinger Bands
        for bb_l in strategy_config['bb_length_options']:
            for bb_s in strategy_config['bb_std_options']:
                bbands = ta.bbands(stock_df['Close'], length=bb_l, std=bb_s)
                bbl_col_name = next((col for col in bbands.columns if 'BBL' in col), None)
                bbm_col_name = next((col for col in bbands.columns if 'BBM' in col), None)
                if bbands is None or not bbl_col_name or not bbm_col_name or \
                   bbands[bbl_col_name].isnull().all() or bbands[bbm_col_name].isnull().all():
                    print(f"錯誤: 布林帶計算失敗 (L={bb_l}, S={bb_s})")
                    indicators_ready = False; break
                precalculated_bbl[(bb_l, bb_s)] = bbands[bbl_col_name].tolist()
                precalculated_bbm[(bb_l, bb_s)] = bbands[bbm_col_name].tolist()
            if not indicators_ready: break
        if not indicators_ready: return {}, False
        print(f"  Bollinger Bands ({len(precalculated_bbl)} variants) OK.")
        # Fixed Indicators: BBI
        bbi_periods = strategy_config['bbi_periods']
        sma1 = stock_df['Close'].rolling(window=bbi_periods[0]).mean()
        sma2 = stock_df['Close'].rolling(window=bbi_periods[1]).mean()
        sma3 = stock_df['Close'].rolling(window=bbi_periods[2]).mean()
        sma4 = stock_df['Close'].rolling(window=bbi_periods[3]).mean()
        bbi_values = (sma1 + sma2 + sma3 + sma4) / 4
        if bbi_values is None or bbi_values.isnull().all(): print("錯誤: BBI 計算失敗"); indicators_ready = False
        else: precalculated_fixed['bbi_list'] = bbi_values.tolist(); print("  BBI OK.")
        # Fixed Indicators: ADX
        if indicators_ready:
            adx_df = ta.adx(stock_df['High'], stock_df['Low'], stock_df['Close'], length=strategy_config['adx_period'])
            adx_col_name = next((col for col in adx_df.columns if 'ADX' in col), None)
            if adx_df is None or not adx_col_name or adx_df[adx_col_name].isnull().all(): print(f"錯誤: ADX 計算失敗"); indicators_ready = False
            else: precalculated_fixed['adx_list'] = adx_df[adx_col_name].tolist(); print("  ADX OK.")
        # Fixed Indicators: MAs
        if indicators_ready:
            ma_short = stock_df['Close'].rolling(window=strategy_config['ma_short_period']).mean()
            ma_long = stock_df['Close'].rolling(window=strategy_config['ma_long_period']).mean()
            if ma_short is None or ma_long is None or ma_short.isnull().all() or ma_long.isnull().all(): print(f"錯誤: 移動平均線計算失敗"); indicators_ready = False
            else: precalculated_fixed['ma_short_list'] = ma_short.tolist(); precalculated_fixed['ma_long_list'] = ma_long.tolist(); print("  MAs OK.")

        if indicators_ready: print("所有指標預計算完成。"); return {'rsi': precalculated_rsi, 'vix_ma': precalculated_vix_ma, 'bbl': precalculated_bbl, 'bbm': precalculated_bbm, 'fixed': precalculated_fixed}, True
        else: print("指標預計算中發生錯誤。"); return {}, False
    except Exception as e: print(f"預計算指標時發生未預期錯誤: {type(e).__name__}: {e}"); traceback.print_exc(); return {}, False

# --- run_strategy_numba_core (保持不變) ---
@numba.jit(nopython=True)
def run_strategy_numba_core(rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold, vix_threshold,
                            low_vol_exit_strategy_MA, high_vol_entry_choice, low_vol_strategy_choice, # Optimized params
                            commission_rate, # Fixed commission
                            prices_arr, # NumPy arrays
                            rsi_arr, bbl_arr, bbm_arr, bbi_arr, adx_arr, vix_ma_arr, ma_short_arr, ma_long_arr, # Indicator arrays
                            start_trading_iloc):
    # ... (Numba 核心代碼保持不變) ...
    T = len(prices_arr)
    if T == 0: return np.array([1.0]), np.array([-1]), np.array([np.nan]), np.array([np.nan]), np.array([-1]), np.array([np.nan]), np.array([np.nan])
    portfolio_values_arr = np.full(T, 1.0, dtype=np.float64)
    max_signals = T // 2 + 1
    buy_signal_indices = np.full(max_signals, -1, dtype=np.int64); buy_signal_prices = np.full(max_signals, np.nan, dtype=np.float64); buy_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    sell_signal_indices = np.full(max_signals, -1, dtype=np.int64); sell_signal_prices = np.full(max_signals, np.nan, dtype=np.float64); sell_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    buy_count = 0; sell_count = 0; cash = 1.0; stock = 0.0; position = 0; last_valid_portfolio_value = 1.0
    rsi_crossed_exit_level_after_buy = False; high_vol_entry_type = -1; low_vol_entry_type = -1
    start_trading_iloc = max(1, start_trading_iloc)
    if start_trading_iloc >= T: portfolio_values_arr[:] = 1.0; return portfolio_values_arr, buy_signal_indices[:0], buy_signal_prices[:0], buy_signal_rsis[:0], sell_signal_indices[:0], sell_signal_prices[:0], sell_signal_rsis[:0]
    portfolio_values_arr[:start_trading_iloc] = 1.0
    for i in range(start_trading_iloc, T):
        current_price = prices_arr[i]; rsi_i, rsi_prev = rsi_arr[i], rsi_arr[i-1]; bbl_i = bbl_arr[i]; bbm_i = bbm_arr[i]; bbi_i = bbi_arr[i]; adx_i = adx_arr[i]; vix_ma_i = vix_ma_arr[i]
        ma_short_i, ma_long_i = ma_short_arr[i], ma_long_arr[i]; ma_short_prev, ma_long_prev = ma_short_arr[i-1], ma_long_arr[i-1]
        required_values = (rsi_i, rsi_prev, current_price, bbl_i, bbm_i, bbi_i, adx_i, vix_ma_i, ma_short_i, ma_long_i, ma_short_prev, ma_long_prev)
        is_valid = True;
        for val in required_values:
            if not np.isfinite(val): is_valid = False; break
        vix_valid = np.isfinite(vix_ma_i)
        if not is_valid:
            current_eval_value = cash if position == 0 else (stock * current_price if np.isfinite(current_price) else np.nan)
            portfolio_values_arr[i] = last_valid_portfolio_value if np.isnan(current_eval_value) else current_eval_value
            if not np.isnan(current_eval_value): last_valid_portfolio_value = current_eval_value
            continue
        is_high_vol = vix_valid and (vix_ma_i >= vix_threshold)
        if not is_high_vol or high_vol_entry_type != 0: rsi_crossed_exit_level_after_buy = False
        if not is_high_vol: high_vol_entry_type = -1
        if is_high_vol or position == 0: low_vol_entry_type = -1
        if position == 0:
            buy_condition = False; entry_type_if_bought = -1; lv_entry_type_if_bought = -1
            if is_high_vol:
                if high_vol_entry_choice == 0: # BB+RSI
                    if (current_price <= bbl_i) and (rsi_i < rsi_buy_entry_threshold): buy_condition = True; entry_type_if_bought = 0
                else: # BB+ADX
                    if (current_price <= bbl_i) and (adx_i > adx_threshold): buy_condition = True; entry_type_if_bought = 1
            else: # low_vol
                if low_vol_strategy_choice == 0: # MA Crossover
                    if (ma_short_prev < ma_long_prev) and (ma_short_i >= ma_long_i): buy_condition = True; lv_entry_type_if_bought = 0
                else: # Pure BBI Signal
                    if current_price > bbi_i: buy_condition = True; lv_entry_type_if_bought = 1
            if buy_condition and current_price > 1e-9:
                cost = cash * commission_rate; amount_to_invest = cash - cost
                if amount_to_invest > 0:
                    stock = amount_to_invest / current_price; cash = 0.0; position = 1; rsi_crossed_exit_level_after_buy = False
                    high_vol_entry_type = entry_type_if_bought; low_vol_entry_type = lv_entry_type_if_bought
                    if buy_count < max_signals: buy_signal_indices[buy_count] = i; buy_signal_prices[buy_count] = current_price; buy_signal_rsis[buy_count] = rsi_i; buy_count += 1
        elif position == 1:
            sell_condition = False
            if high_vol_entry_type == 0: # BB+RSI Exit
                if rsi_i >= rsi_exit_threshold: rsi_crossed_exit_level_after_buy = True
                if rsi_crossed_exit_level_after_buy and rsi_i < rsi_exit_threshold: sell_condition = True
            elif high_vol_entry_type == 1: # BB+ADX Exit
                sell_condition = (current_price >= bbm_i)
            elif low_vol_entry_type == 0: # MA Cross Exit
                if low_vol_exit_strategy_MA == 0: sell_condition = (current_price < ma_short_i)
                else: sell_condition = (ma_short_prev > ma_long_prev) and (ma_short_i <= ma_long_i)
            elif low_vol_entry_type == 1: # Pure BBI Exit
                sell_condition = (current_price < bbi_i)
            if sell_condition:
                proceeds = stock * current_price; cost = proceeds * commission_rate; cash = proceeds - cost; stock = 0.0; position = 0
                rsi_crossed_exit_level_after_buy = False; high_vol_entry_type = -1; low_vol_entry_type = -1
                if sell_count < max_signals: sell_signal_indices[sell_count] = i; sell_signal_prices[sell_count] = current_price; sell_signal_rsis[sell_count] = rsi_i; sell_count += 1
        current_stock_value = stock * current_price if position == 1 else 0.0; current_portfolio_value = cash + current_stock_value
        portfolio_values_arr[i] = last_valid_portfolio_value if np.isnan(current_portfolio_value) else current_portfolio_value
        if not np.isnan(current_portfolio_value): last_valid_portfolio_value = current_portfolio_value
    if T > 0 and np.isnan(portfolio_values_arr[-1]): portfolio_values_arr[-1] = last_valid_portfolio_value
    return (portfolio_values_arr, buy_signal_indices[:buy_count], buy_signal_prices[:buy_count], buy_signal_rsis[:buy_count], sell_signal_indices[:sell_count], sell_signal_prices[:sell_count], sell_signal_rsis[:sell_count])

# --- run_strategy Wrapper (保持不變) ---
def run_strategy(rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold, vix_threshold,
                 low_vol_exit_strategy_MA, high_vol_entry_choice, low_vol_strategy_choice, # Optimized params
                 commission_rate, # Fixed commission
                 prices, dates, # Base data
                 rsi_list, bbl_list, bbm_list, bbi_list, adx_list, vix_ma_list, ma_short_list, ma_long_list): # Pre-calculated lists
    # ... (Wrapper 函數保持不變) ...
    T = len(prices);
    if T == 0: return [1.0], [], []
    prices_arr = np.array(prices, dtype=np.float64); rsi_arr = np.array(rsi_list, dtype=np.float64); bbl_arr = np.array(bbl_list, dtype=np.float64); bbm_arr = np.array(bbm_list, dtype=np.float64); bbi_arr = np.array(bbi_list, dtype=np.float64); adx_arr = np.array(adx_list, dtype=np.float64); vix_ma_arr = np.array(vix_ma_list, dtype=np.float64); ma_short_arr = np.array(ma_short_list, dtype=np.float64); ma_long_arr = np.array(ma_long_list, dtype=np.float64)
    def get_first_valid_iloc(indicator_arr): valid_indices = np.where(np.isfinite(indicator_arr))[0]; return valid_indices[0] if len(valid_indices) > 0 else T
    start_iloc_rsi = get_first_valid_iloc(rsi_arr) if len(rsi_arr) > 0 else T; start_iloc_bbl = get_first_valid_iloc(bbl_arr) if len(bbl_arr) > 0 else T; start_iloc_bbm = get_first_valid_iloc(bbm_arr) if len(bbm_arr) > 0 else T; start_iloc_bbi = get_first_valid_iloc(bbi_arr) if len(bbi_arr) > 0 else T; start_iloc_adx = get_first_valid_iloc(adx_arr) if len(adx_arr) > 0 else T; start_iloc_vix_ma = get_first_valid_iloc(vix_ma_arr) if len(vix_ma_arr) > 0 else T; start_iloc_ma_short = get_first_valid_iloc(ma_short_arr) if len(ma_short_arr) > 0 else T; start_iloc_ma_long = get_first_valid_iloc(ma_long_arr) if len(ma_long_arr) > 0 else T
    start_trading_iloc = max(start_iloc_rsi, start_iloc_bbl, start_iloc_bbm, start_iloc_bbi, start_iloc_adx, start_iloc_vix_ma, start_iloc_ma_short, start_iloc_ma_long) + 1
    if start_trading_iloc >= T: print(f"警告: 有效數據不足，無法開始交易 (需要 {start_trading_iloc} 數據點, 只有 {T})"); return [1.0] * T, [], []
    start_trading_iloc = max(start_trading_iloc, 1)
    portfolio_values_arr, buy_indices, buy_prices, buy_rsis, sell_indices, sell_prices, sell_rsis = \
        run_strategy_numba_core(float(rsi_buy_entry_threshold), float(rsi_exit_threshold), float(adx_threshold), float(vix_threshold), int(low_vol_exit_strategy_MA), int(high_vol_entry_choice), int(low_vol_strategy_choice), float(commission_rate), prices_arr, rsi_arr, bbl_arr, bbm_arr, bbi_arr, adx_arr, vix_ma_arr, ma_short_arr, ma_long_arr, start_trading_iloc)
    buy_signals = []; sell_signals = []
    for idx, price, rsi_val in zip(buy_indices, buy_prices, buy_rsis):
        if idx != -1 and idx < len(dates): buy_signals.append((dates[idx], price, rsi_val))
    for idx, price, rsi_val in zip(sell_indices, sell_prices, sell_rsis):
         if idx != -1 and idx < len(dates): sell_signals.append((dates[idx], price, rsi_val))
    return portfolio_values_arr.tolist(), buy_signals, sell_signals


# --- Genetic Algorithm (保持不變) ---
# --- Genetic Algorithm (修正變異部分語法) ---
def genetic_algorithm_with_elitism(prices, dates, # Base data
                                   # --- 預計算的指標列表/字典 ---
                                   precalculated_indicators, # 包含 'rsi', 'vix_ma', 'bbl', 'bbm', 'fixed' 的字典
                                   ga_params):
    """
    優化策略參數，包括低波動策略選擇（MA vs 純 BBI）。
    基因: [rsi_buy_entry, rsi_exit_ref, vix_thr, low_vol_exit_MA, rsi_p_choice, vix_ma_p_choice, bb_len_choice, bb_std_choice, adx_threshold, high_vol_entry_choice, low_vol_strategy_choice]
    """
    # --- 解包 GA 參數 ---
    generations = ga_params['generations']
    population_size = ga_params['population_size']
    crossover_rate = ga_params['crossover_rate']
    mutation_rate = ga_params['mutation_rate']
    elitism_size = ga_params['elitism_size']
    tournament_size = ga_params['tournament_size']
    mutation_amount_range = ga_params['mutation_amount_range']
    vix_mutation_amount_range = ga_params.get('vix_mutation_amount_range', mutation_amount_range) # VIX 變異範圍
    adx_mutation_amount_range = ga_params.get('adx_mutation_amount_range', mutation_amount_range) # ADX 變異範圍
    show_process = ga_params.get('show_process', False) # 預設不顯示過程

    # --- 參數範圍和選項 ---
    rsi_threshold_range = ga_params['rsi_threshold_range']
    vix_threshold_range = ga_params['vix_threshold_range']
    adx_threshold_range = ga_params['adx_threshold_range']
    rsi_period_options = ga_params['rsi_period_options']
    vix_ma_period_options = ga_params['vix_ma_period_options']
    bb_length_options = ga_params['bb_length_options']
    bb_std_options = ga_params['bb_std_options']
    commission_rate = ga_params['commission_rate']

    # 固定的指標參數 (用於提取對應的列表)
    adx_list = precalculated_indicators['fixed']['adx_list']
    bbi_list = precalculated_indicators['fixed']['bbi_list']
    ma_short_list = precalculated_indicators['fixed']['ma_short_list']
    ma_long_list = precalculated_indicators['fixed']['ma_long_list']

    # --- 其他變數 ---
    T = len(prices)
    num_rsi_options = len(rsi_period_options)
    num_vix_ma_options = len(vix_ma_period_options)
    num_bb_len_options = len(bb_length_options)
    num_bb_std_options = len(bb_std_options)

    # --- 檢查數據長度 ---
    if T < 2: print("錯誤: 數據長度過短，無法運行 GA。"); return None, 0

    # --- 初始化種群 (Gene: 11 個元素) ---
    population = []; attempts, max_attempts = 0, population_size * 100
    min_buy, max_buy, min_exit, max_exit = rsi_threshold_range; min_vix, max_vix = vix_threshold_range; min_adx, max_adx = adx_threshold_range
    while len(population) < population_size and attempts < max_attempts:
        buy_entry_thr = random.randint(min_buy, max_buy); exit_thr = random.randint(max(buy_entry_thr + 1, min_exit), max_exit); vix_thr = random.randint(min_vix, max_vix); low_vol_exit_MA = random.choice([0, 1]); rsi_p_choice = random.randint(0, num_rsi_options - 1); vix_ma_p_choice = random.randint(0, num_vix_ma_options - 1); bb_len_choice = random.randint(0, num_bb_len_options - 1); bb_std_choice = random.randint(0, num_bb_std_options - 1); adx_thr = random.randint(min_adx, max_adx); hv_entry_choice = random.choice([0, 1]); low_vol_strat_choice = random.choice([0, 1])
        gene = [buy_entry_thr, exit_thr, vix_thr, low_vol_exit_MA, rsi_p_choice, vix_ma_p_choice, bb_len_choice, bb_std_choice, adx_thr, hv_entry_choice, low_vol_strat_choice]
        valid_gene = (0<gene[0]<gene[1]<100 and min_buy<=gene[0]<=max_buy and min_exit<=gene[1]<=max_exit and min_vix<=gene[2]<=max_vix and gene[3] in [0,1] and 0<=gene[4]<num_rsi_options and 0<=gene[5]<num_vix_ma_options and 0<=gene[6]<num_bb_len_options and 0<=gene[7]<num_bb_std_options and min_adx<=gene[8]<=max_adx and gene[9] in [0,1] and gene[10] in [0,1])
        if valid_gene: population.append(gene)
        attempts += 1
    if not population: print(f"錯誤: 無法生成有效的初始種群 (嘗試 {attempts} 次)。"); return None, 0
    population = population[:population_size]; best_gene_overall = population[0][:]; best_fitness_overall = -float('inf')

    # --- GA 迭代循環 ---
    for generation in range(generations):
        fitness = [] # 儲存當前代所有個體的適應度
        # --- 評估每個個體的適應度 ---
        for gene in population:
            try: # 添加 try-except 捕捉可能的索引錯誤
                chosen_rsi_period = rsi_period_options[gene[4]]
                chosen_vix_ma_period = vix_ma_period_options[gene[5]]
                chosen_bb_length = bb_length_options[gene[6]]
                chosen_bb_std = bb_std_options[gene[7]]

                rsi_list = precalculated_indicators['rsi'][chosen_rsi_period]
                vix_ma_list = precalculated_indicators['vix_ma'][chosen_vix_ma_period]
                bbl_list = precalculated_indicators['bbl'][(chosen_bb_length, chosen_bb_std)]
                bbm_list = precalculated_indicators['bbm'][(chosen_bb_length, chosen_bb_std)]

                portfolio_values, _, _ = run_strategy(
                    gene[0], gene[1], gene[8], gene[2], gene[3], gene[9], gene[10], # 傳遞基因中的 7 個策略參數
                    commission_rate,                                                # 佣金率
                    prices, dates,                                                  # 基礎數據
                    rsi_list, bbl_list, bbm_list, bbi_list, adx_list, vix_ma_list,  # 指標列表
                    ma_short_list, ma_long_list
                )
                final_value = next((p for p in reversed(portfolio_values) if np.isfinite(p)), -np.inf)
                fitness.append(final_value)
            except IndexError as e:
                 print(f"錯誤: 評估基因 {gene} 時發生索引錯誤 ({e})。可能是基因無效或配置不匹配。適應度設為 -inf。")
                 fitness.append(-np.inf)
            except KeyError as e:
                 print(f"錯誤: 評估基因 {gene} 時找不到指標鍵 ({e})。適應度設為 -inf。")
                 fitness.append(-np.inf)
            except Exception as e:
                 print(f"錯誤: 評估基因 {gene} 時發生未知錯誤: {e}")
                 traceback.print_exc()
                 fitness.append(-np.inf)


        # --- 選擇、交叉、變異 ---
        fitness_array = np.array(fitness)
        valid_fitness_mask = np.isfinite(fitness_array) & (fitness_array > -np.inf)
        valid_indices = np.where(valid_fitness_mask)[0]
        valid_fitness_count = len(valid_indices)

        if valid_fitness_count == 0:
             if show_process: print(f"第 {generation+1} 代 - 警告: 所有個體均無效。")
             num_elites = min(elitism_size, len(population)); elites = [population[i][:] for i in range(num_elites)] # 保留上一代可能的精英
             new_random_pop = []; attempts = 0
             while len(new_random_pop) < population_size - num_elites and attempts < max_attempts:
                 buy_entry_thr = random.randint(min_buy, max_buy); exit_thr = random.randint(max(buy_entry_thr + 1, min_exit), max_exit); vix_thr = random.randint(min_vix, max_vix); low_vol_exit_MA = random.choice([0, 1]); rsi_p_choice = random.randint(0, num_rsi_options - 1); vix_ma_p_choice = random.randint(0, num_vix_ma_options - 1); bb_len_choice = random.randint(0, num_bb_len_options - 1); bb_std_choice = random.randint(0, num_bb_std_options - 1); adx_thr = random.randint(min_adx, max_adx); hv_entry_choice = random.choice([0, 1]); low_vol_strat_choice = random.choice([0, 1])
                 gene = [buy_entry_thr, exit_thr, vix_thr, low_vol_exit_MA, rsi_p_choice, vix_ma_p_choice, bb_len_choice, bb_std_choice, adx_thr, hv_entry_choice, low_vol_strat_choice]
                 valid_gene = (0<gene[0]<gene[1]<100 and min_buy<=gene[0]<=max_buy and min_exit<=gene[1]<=max_exit and min_vix<=gene[2]<=max_vix and gene[3] in [0,1] and 0<=gene[4]<num_rsi_options and 0<=gene[5]<num_vix_ma_options and 0<=gene[6]<num_bb_len_options and 0<=gene[7]<num_bb_std_options and min_adx<=gene[8]<=max_adx and gene[9] in [0,1] and gene[10] in [0,1])
                 if valid_gene: new_random_pop.append(gene)
                 attempts += 1
             population = elites + new_random_pop; population = population[:population_size]
             continue

        # --- 精英選擇 ---
        sorted_valid_indices = valid_indices[np.argsort(fitness_array[valid_indices])[::-1]]
        num_elites = min(elitism_size, valid_fitness_count)
        elite_indices = sorted_valid_indices[:num_elites]
        elites = [population[i][:] for i in elite_indices]

        current_best_fitness_in_gen = fitness_array[elite_indices[0]] if num_elites > 0 else -np.inf
        if current_best_fitness_in_gen > best_fitness_overall:
            best_fitness_overall = current_best_fitness_in_gen
            best_gene_overall = population[elite_indices[0]][:]

        if show_process and (generation + 1) % 10 == 0:
             gen_best_str = f"{current_best_fitness_in_gen:.4f}" if num_elites > 0 else "N/A"
             if best_fitness_overall > -np.inf:
                 # ... (顯示參數部分不變) ...
                 lv_exit_ma_str = "Price<MA" if best_gene_overall[3] == 0 else "MACross"; chosen_rsi_p_str = rsi_period_options[best_gene_overall[4]]; chosen_vix_ma_p_str = vix_ma_period_options[best_gene_overall[5]]; chosen_bb_l_str = bb_length_options[best_gene_overall[6]]; chosen_bb_s_str = bb_std_options[best_gene_overall[7]]; hv_entry_str = "BB+RSI" if best_gene_overall[9] == 0 else "BB+ADX"; lv_strat_str = "MA" if best_gene_overall[10] == 0 else "BBI"
                 overall_best_params_str = (f"BuyE={best_gene_overall[0]},ExitR={best_gene_overall[1]},VIX_T={best_gene_overall[2]},LVExitMA={lv_exit_ma_str},RSI_P={chosen_rsi_p_str},VIX_MA_P={chosen_vix_ma_p_str},BB_L={chosen_bb_l_str},BB_S={chosen_bb_s_str},ADX_T={best_gene_overall[8]},HVEntry={hv_entry_str},LVStrat={lv_strat_str}")
                 overall_best_str = f"{best_fitness_overall:.4f} (Params: {overall_best_params_str})"
             else: overall_best_str = "N/A"
             print(f"第 {generation+1}/{generations} 代 | 當代最佳: {gen_best_str} | 全局最佳: {overall_best_str} | 有效個體: {valid_fitness_count}/{population_size}")

        # --- 錦標賽選擇 ---
        selected_parents = []; num_parents_to_select = population_size - num_elites
        if num_parents_to_select <= 0: population = elites[:population_size]; continue
        effective_tournament_size = min(tournament_size, valid_fitness_count)
        if effective_tournament_size <= 0: population = elites[:population_size]; continue
        for _ in range(num_parents_to_select):
             aspirant_indices_local = np.random.choice(len(valid_indices), size=effective_tournament_size, replace=False)
             aspirant_indices_global = valid_indices[aspirant_indices_local]
             winner_global_idx = aspirant_indices_global[np.argmax(fitness_array[aspirant_indices_global])]
             selected_parents.append(population[winner_global_idx][:])

        # --- 交叉 ---
        offspring = []; parent_indices = list(range(len(selected_parents))); random.shuffle(parent_indices); num_pairs = len(parent_indices) // 2
        for i in range(num_pairs):
            p1_idx = parent_indices[2*i]; p2_idx = parent_indices[2*i + 1]; p1, p2 = selected_parents[p1_idx], selected_parents[p2_idx]; child1, child2 = p1[:], p2[:]
            if random.random() < crossover_rate:
                 crossover_point = random.randint(1, 10); child1_new = p1[:crossover_point] + p2[crossover_point:]; child2_new = p2[:crossover_point] + p1[crossover_point:]
                 valid_c1 = (0<child1_new[0]<child1_new[1]<100 and min_buy<=child1_new[0]<=max_buy and min_exit<=child1_new[1]<=max_exit and min_vix<=child1_new[2]<=max_vix and child1_new[3] in [0,1] and 0<=child1_new[4]<num_rsi_options and 0<=child1_new[5]<num_vix_ma_options and 0<=child1_new[6]<num_bb_len_options and 0<=child1_new[7]<num_bb_std_options and min_adx<=child1_new[8]<=max_adx and child1_new[9] in [0,1] and child1_new[10] in [0,1])
                 valid_c2 = (0<child2_new[0]<child2_new[1]<100 and min_buy<=child2_new[0]<=max_buy and min_exit<=child2_new[1]<=max_exit and min_vix<=child2_new[2]<=max_vix and child2_new[3] in [0,1] and 0<=child2_new[4]<num_rsi_options and 0<=child2_new[5]<num_vix_ma_options and 0<=child2_new[6]<num_bb_len_options and 0<=child2_new[7]<num_bb_std_options and min_adx<=child2_new[8]<=max_adx and child2_new[9] in [0,1] and child2_new[10] in [0,1])
                 child1 = child1_new if valid_c1 else p1[:]; child2 = child2_new if valid_c2 else p2[:]
            offspring.append(child1); offspring.append(child2)
        if len(parent_indices) % 2 != 0: offspring.append(selected_parents[parent_indices[-1]][:])

        # --- 變異 ---
        mut_min, mut_max = mutation_amount_range; vix_mut_min, vix_mut_max = vix_mutation_amount_range; adx_mut_min, adx_mut_max = adx_mutation_amount_range
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                gene_to_mutate = offspring[i]
                original_gene = gene_to_mutate[:] # 保存原始基因
                mutate_idx = random.randint(0, 10) # 隨機選擇變異點 (0 到 10)

                if mutate_idx == 3: # 變異 低波動MA退出策略 (Index 3)
                    gene_to_mutate[3] = 1 - gene_to_mutate[3]
                # ============ 修正語法開始 ============
                elif mutate_idx == 4: # 變異 RSI 週期選擇 (Index 4)
                    if num_rsi_options > 1:
                        original_choice = gene_to_mutate[4]
                        new_choice = random.randint(0, num_rsi_options - 1)
                        # 確保變異到不同的選項
                        while new_choice == original_choice:
                            new_choice = random.randint(0, num_rsi_options - 1)
                        gene_to_mutate[4] = new_choice
                elif mutate_idx == 5: # 變異 VIX MA 週期選擇 (Index 5)
                    if num_vix_ma_options > 1:
                        original_choice = gene_to_mutate[5]
                        new_choice = random.randint(0, num_vix_ma_options - 1)
                        while new_choice == original_choice:
                            new_choice = random.randint(0, num_vix_ma_options - 1)
                        gene_to_mutate[5] = new_choice
                elif mutate_idx == 6: # 變異 BB 長度選擇 (Index 6)
                     if num_bb_len_options > 1:
                        original_choice = gene_to_mutate[6]
                        new_choice = random.randint(0, num_bb_len_options - 1)
                        while new_choice == original_choice:
                            new_choice = random.randint(0, num_bb_len_options - 1)
                        gene_to_mutate[6] = new_choice
                elif mutate_idx == 7: # 變異 BB 標準差選擇 (Index 7)
                     if num_bb_std_options > 1:
                        original_choice = gene_to_mutate[7]
                        new_choice = random.randint(0, num_bb_std_options - 1)
                        while new_choice == original_choice:
                            new_choice = random.randint(0, num_bb_std_options - 1)
                        gene_to_mutate[7] = new_choice
                # ============ 修正語法結束 ============
                elif mutate_idx == 9: # 變異 高波動入場策略 (Index 9)
                    gene_to_mutate[9] = 1 - gene_to_mutate[9]
                elif mutate_idx == 10: # 變異 低波動策略 (Index 10)
                     gene_to_mutate[10] = 1 - gene_to_mutate[10]
                else: # 變異數值型閾值
                    # ... (閾值變異部分保持不變) ...
                    if mutate_idx == 2: mutation_amount = random.randint(vix_mut_min, vix_mut_max); is_zero_range = (vix_mut_min == 0 and vix_mut_max == 0)
                    elif mutate_idx == 8: mutation_amount = random.randint(adx_mut_min, adx_mut_max); is_zero_range = (adx_mut_min == 0 and adx_mut_max == 0)
                    else: mutation_amount = random.randint(mut_min, mut_max); is_zero_range = (mut_min == 0 and mut_max == 0)
                    if mutation_amount == 0 and not is_zero_range:
                         while mutation_amount == 0:
                             if mutate_idx == 2: mutation_amount = random.randint(vix_mut_min, vix_mut_max)
                             elif mutate_idx == 8: mutation_amount = random.randint(adx_mut_min, adx_mut_max)
                             else: mutation_amount = random.randint(mut_min, mut_max)
                    gene_to_mutate[mutate_idx] += mutation_amount
                    gene_to_mutate[0] = max(min_buy, min(gene_to_mutate[0], max_buy)); gene_to_mutate[1] = max(gene_to_mutate[0] + 1, min_exit, min(gene_to_mutate[1], max_exit)); gene_to_mutate[0] = max(min_buy, min(gene_to_mutate[0], gene_to_mutate[1] - 1, max_buy)); gene_to_mutate[2] = max(min_vix, min(gene_to_mutate[2], max_vix)); gene_to_mutate[8] = max(min_adx, min(gene_to_mutate[8], max_adx))

                # --- 變異後最終驗證 ---
                final_valid = (
                    0 < gene_to_mutate[0] < gene_to_mutate[1] < 100 and min_buy <= gene_to_mutate[0] <= max_buy and
                    min_exit <= gene_to_mutate[1] <= max_exit and min_vix <= gene_to_mutate[2] <= max_vix and
                    gene_to_mutate[3] in [0, 1] and 0 <= gene_to_mutate[4] < num_rsi_options and
                    0 <= gene_to_mutate[5] < num_vix_ma_options and 0 <= gene_to_mutate[6] < num_bb_len_options and
                    0 <= gene_to_mutate[7] < num_bb_std_options and min_adx <= gene_to_mutate[8] <= max_adx and
                    gene_to_mutate[9] in [0, 1] and gene_to_mutate[10] in [0, 1]
                )
                if not final_valid:
                    offspring[i] = original_gene[:] # 恢復原始基因

        # --- 組成新一代種群 ---
        population = elites + offspring
        population = population[:population_size] # 確保種群大小正確

    # --- GA 結束 ---
    if best_fitness_overall == -float('inf'):
        print("錯誤: GA 運行結束，未能找到任何有效的解決方案。")
        return None, 0

    # 返回找到的最佳基因和對應的適應度
    return best_gene_overall, best_fitness_overall


# ... (文件的其餘部分，包括 __main__ 測試塊，保持不變) ...

# --- Main Block for Testing (保持不變) ---
if __name__ == "__main__":
    print("正在執行 model_train.py 作為獨立腳本 (用於測試)...")
    # ... (測試部分的代碼與之前相同) ...
    # --- Load Data ---
    start_load_time = time.time()
    test_config = { 'ticker': 'SMCI', 'vix_ticker': '^VIX', 'start_date': '2022-01-01', 'end_date': '2025-05-17' }
    test_strategy_config = { 'rsi_period_options': [6, 12], 'vix_ma_period_options': [5, 10, 20], 'bb_length_options': [20], 'bb_std_options': [2.0], 'adx_period': 14, 'bbi_periods': (3, 6, 12, 24), 'ma_short_period': 5, 'ma_long_period': 10, 'commission_pct': 0.003, }
    test_ga_config = { 'generations': 10, 'population_size': 20, 'crossover_rate': 0.7, 'mutation_rate': 0.25, 'elitism_size': 2, 'tournament_size': 5, 'mutation_amount_range': (-2, 2), 'vix_mutation_amount_range': (-1, 1), 'adx_mutation_amount_range': (-1, 1), 'show_process': True, 'rsi_threshold_range': (15, 35, 50, 70), 'vix_threshold_range': (18, 32), 'adx_threshold_range': (22, 38), 'rsi_period_options': test_strategy_config['rsi_period_options'], 'vix_ma_period_options': test_strategy_config['vix_ma_period_options'], 'bb_length_options': test_strategy_config['bb_length_options'], 'bb_std_options': test_strategy_config['bb_std_options'], 'adx_period': test_strategy_config['adx_period'], 'bbi_periods': test_strategy_config['bbi_periods'], 'ma_short_period': test_strategy_config['ma_short_period'], 'ma_long_period': test_strategy_config['ma_long_period'], 'commission_rate': test_strategy_config['commission_pct'], }
    test_plot_config = { 'show_plot': True, 'figure_size': (14, 8), 'buy_marker_color': 'lime', 'sell_marker_color': 'red', 'strategy_line_color': 'blue', 'price_line_color': 'darkgrey', 'plot_bbands': True, 'bb_line_color': 'grey', 'bb_fill_color': 'grey', 'plot_mas': True, 'ma_short_color': 'lightblue', 'ma_long_color': 'lightcoral', }
    prices, dates, stock_df, vix_series = load_stock_data(test_config['ticker'], vix_ticker=test_config['vix_ticker'], start_date=test_config['start_date'], end_date=test_config['end_date'])
    load_time = time.time() - start_load_time; print(f"測試數據載入耗時 {load_time:.2f} 秒。")
    if prices and dates and stock_df is not None and vix_series is not None:
        indicator_calc_start = time.time(); precalculated_indicators, indicators_ready = precompute_indicators(stock_df, vix_series, test_strategy_config); indicator_calc_time = time.time() - indicator_calc_start; print(f"測試指標預計算耗時 {indicator_calc_time:.2f} 秒。")
        if indicators_ready:
            print(f"\n對齊後的數據點數: {len(prices)}."); print(f"開始運行遺傳演算法 (測試模式)..."); ga_start_time = time.time()
            best_params_test, final_value_test = genetic_algorithm_with_elitism(prices, dates, precalculated_indicators, ga_params=test_ga_config)
            ga_time = time.time() - ga_start_time; print(f"\n遺傳演算法 (測試模式) 耗時 {ga_time:.2f} 秒。")
            if best_params_test:
                print(f"\n--- 測試運行最佳結果 ---"); print(f"股票: {test_config['ticker']}"); print(f"最佳最終組合價值 (從 1 開始): {final_value_test:.4f}")
                best_buy_entry, best_exit_ref, best_vix_thr, best_low_vol_exit_MA, best_rsi_p_choice, best_vix_ma_p_choice, best_bb_l_choice, best_bb_s_choice, best_adx_thr, best_hv_entry_choice, best_lv_strat_choice = best_params_test
                best_rsi_p = test_strategy_config['rsi_period_options'][best_rsi_p_choice]; best_vix_ma_p = test_strategy_config['vix_ma_period_options'][best_vix_ma_p_choice]; best_bb_l = test_strategy_config['bb_length_options'][best_bb_l_choice]; best_bb_s = test_strategy_config['bb_std_options'][best_bb_s_choice]; low_vol_exit_desc = "Price < MA_Short" if best_low_vol_exit_MA == 0 else "MA CrossDown"; hv_entry_desc = "BB+RSI" if best_hv_entry_choice == 0 else "BB+ADX"; lv_strat_desc = "MA Crossover" if best_lv_strat_choice == 0 else "Pure BBI Signal"; commission_info = f"{test_strategy_config['commission_pct']*100:.3f}%"
                print(f"優化後的參數:"); print(f"  RSI Period: {best_rsi_p}, VIX MA Period: {best_vix_ma_p}, BB: ({best_bb_l},{best_bb_s})"); print(f"  VIX Threshold: {best_vix_thr}, ADX Threshold: {best_adx_thr}"); print(f"  HV Entry: {hv_entry_desc} (RSI Buy<{best_buy_entry}, RSI Exit Ref>{best_exit_ref})"); print(f"  LV Strategy: {lv_strat_desc} (MA Exit: {low_vol_exit_desc})"); print(f"  Commission: {commission_info}")
                if test_plot_config['show_plot']:
                    print("\n繪製測試結果圖表..."); plot_start_time = time.time()
                    final_rsi_list = precalculated_indicators['rsi'][best_rsi_p]; final_vix_ma_list = precalculated_indicators['vix_ma'][best_vix_ma_p]; final_bbl_list = precalculated_indicators['bbl'][(best_bb_l, best_bb_s)]; final_bbm_list = precalculated_indicators['bbm'][(best_bb_l, best_bb_s)]; final_bbi_list = precalculated_indicators['fixed']['bbi_list']; final_adx_list = precalculated_indicators['fixed']['adx_list']; final_ma_short_list = precalculated_indicators['fixed']['ma_short_list']; final_ma_long_list = precalculated_indicators['fixed']['ma_long_list']
                    best_strategy_values, buy_signals, sell_signals = run_strategy(best_buy_entry, best_exit_ref, best_adx_thr, best_vix_thr, best_low_vol_exit_MA, best_hv_entry_choice, best_lv_strat_choice, test_strategy_config['commission_pct'], prices, dates, final_rsi_list, final_bbl_list, final_bbm_list, final_bbi_list, final_adx_list, final_vix_ma_list, final_ma_short_list, final_ma_long_list)
                    if best_strategy_values and len(best_strategy_values) == len(prices):
                         initial_price = prices[0] if prices and np.isfinite(prices[0]) and prices[0] > 1e-9 else 1.0; prices_float = np.array(prices, dtype=np.float64); valid_price_mask = np.isfinite(prices_float)
                         if not valid_price_mask.all(): print("警告: 價格數據包含 NaN 或 Inf，可能影響標準化。"); prices_float[~valid_price_mask] = 1.0
                         normalized_prices = prices_float / initial_price; fig, ax1 = plt.subplots(figsize=test_plot_config['figure_size'])
                         ax1.plot(dates, normalized_prices, label=f'{test_config["ticker"]} Norm Price', color=test_plot_config['price_line_color'], linewidth=1.0, alpha=0.7, zorder=1); ax1.plot(dates, best_strategy_values, label=f'優化策略 (測試)', color=test_plot_config['strategy_line_color'], linewidth=2.0, zorder=3)
                         ax1.set_xlabel('日期', fontsize=12); ax1.set_ylabel('標準化價值 (起始=1)', fontsize=12); ax1.tick_params(axis='y'); ax1.grid(True, linestyle='--', alpha=0.6)
                         print("\n--- 買入信號 (測試) ---");
                         if buy_signals: buy_dates_raw, buy_prices_raw, _ = zip(*buy_signals); buy_prices_normalized = np.array(buy_prices_raw, dtype=float) / initial_price; ax1.scatter(buy_dates_raw, buy_prices_normalized, label='買入', marker='^', color=test_plot_config['buy_marker_color'], s=80, edgecolors='black', zorder=5); print(f"找到 {len(buy_signals)} 個買入信號")
                         else: print("未生成買入信號。")
                         print("\n--- 賣出信號 (測試) ---");
                         if sell_signals: sell_dates_raw, sell_prices_raw, _ = zip(*sell_signals); sell_prices_normalized = np.array(sell_prices_raw, dtype=float) / initial_price; ax1.scatter(sell_dates_raw, sell_prices_normalized, label='賣出', marker='v', color=test_plot_config['sell_marker_color'], s=80, edgecolors='black', zorder=5); print(f"找到 {len(sell_signals)} 個賣出信號")
                         else: print("未生成賣出信號。")
                         if test_plot_config['plot_bbands']:
                             try:
                                 bbands_plot = ta.bbands(stock_df['Close'], length=best_bb_l, std=best_bb_s); bbl_col = next((col for col in bbands_plot.columns if 'BBL' in col), None); bbm_col = next((col for col in bbands_plot.columns if 'BBM' in col), None); bbu_col = next((col for col in bbands_plot.columns if 'BBU' in col), None)
                                 if bbl_col and bbm_col and bbu_col: norm_bbl = bbands_plot[bbl_col]/initial_price; norm_bbm = bbands_plot[bbm_col]/initial_price; norm_bbu = bbands_plot[bbu_col]/initial_price; bb_label = f'BB({best_bb_l},{best_bb_s})'; ax1.plot(dates, norm_bbm, label=f'{bb_label} 中軌', color=test_plot_config['bb_line_color'], linestyle='--', linewidth=0.8, alpha=0.7, zorder=2); ax1.plot(dates, norm_bbu, label=f'{bb_label} 上/下軌', color=test_plot_config['bb_line_color'], linestyle=':', linewidth=0.8, alpha=0.7, zorder=2); ax1.plot(dates, norm_bbl, color=test_plot_config['bb_line_color'], linestyle=':', linewidth=0.8, alpha=0.7, zorder=2); ax1.fill_between(dates, norm_bbl, norm_bbu, color=test_plot_config['bb_fill_color'], alpha=0.1, zorder=1)
                             except Exception as plot_bb_e: print(f"無法繪製布林帶 (已忽略): {plot_bb_e}")
                         if test_plot_config['plot_mas']:
                              try: ma_s_plot = stock_df['Close'].rolling(window=test_strategy_config['ma_short_period']).mean()/initial_price; ma_l_plot = stock_df['Close'].rolling(window=test_strategy_config['ma_long_period']).mean()/initial_price; ax1.plot(dates, ma_s_plot, label=f'MA({test_strategy_config["ma_short_period"]})', color=test_plot_config['ma_short_color'], linestyle='-.', linewidth=0.7, alpha=0.8, zorder=2); ax1.plot(dates, ma_l_plot, label=f'MA({test_strategy_config["ma_long_period"]})', color=test_plot_config['ma_long_color'], linestyle='-.', linewidth=0.7, alpha=0.8, zorder=2)
                              except Exception as plot_ma_e: print(f"無法繪製移動平均線 (已忽略): {plot_ma_e}")
                         title = (f'{test_config["ticker"]} 策略回測 (測試模式) | 佣金: {commission_info}\n' f'最佳參數: RSI_P={best_rsi_p}, VIX_MA_P={best_vix_ma_p}, BB=({best_bb_l},{best_bb_s}), VIX_T={best_vix_thr}, ADX_T={best_adx_thr}\n' f'HV Entry={hv_entry_desc}(Buy<{best_buy_entry}, ExitRef>{best_exit_ref}) | LV Strat={lv_strat_desc}(Exit={low_vol_exit_desc})'); plt.title(title, fontsize=9)
                         lines, labels = ax1.get_legend_handles_labels(); ax1.legend(lines, labels, loc='upper left', fontsize=7); fig.tight_layout(); plot_time = time.time() - plot_start_time; print(f"\n繪圖耗時 {plot_time:.2f} 秒。"); plt.show()
                    else: print("錯誤: 無法獲取有效的策略價值曲線進行繪圖。")
                else: print("\n測試模式下未啟用繪圖。")
            else: print("\n錯誤: 測試運行的遺傳演算法未能找到解決方案。")
        else: print("\n錯誤: 測試指標預計算失敗，無法運行遺傳演算法。")
    else: print("\n錯誤: 測試數據載入或處理失敗。")
    print("\nmodel_train.py 獨立測試執行完畢。")