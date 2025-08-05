# ga_engine_b.py (v2.1 - 系統B多目標優化修復版)
"""
AI遺傳演算法引擎 - 系統B (10基因交易策略)
支援 NSGA-II 多目標優化 + 平均交易報酬率
===============================================
版本: v2.1
基因結構: 10個參數 (RSI策略為主)
更新: 2025/07/05
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import numba
import random
import time
import traceback
import logging
import json
from datetime import datetime as dt_datetime, timedelta

# --- NSGA-II 多目標優化支援 ---
try:
    from pymoo.core.problem import Problem
    from pymoo.core.sampling import Sampling
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    NSGA2_AVAILABLE = True
    print("[GAEngine_B] ✅ NSGA-II 多目標優化支援已載入")
except ImportError:
    NSGA2_AVAILABLE = False
    print("[GAEngine_B] ⚠️  NSGA-II 未安裝，將使用傳統GA (pip install pymoo)")

# --- 日誌設定 ---
logger = logging.getLogger("GAEngine_B")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════════
# 📊 系統B配置 (10基因策略)
# ═══════════════════════════════════════════════════════════════

STRATEGY_CONFIG_B = {
    'rsi_period_options': [6, 12, 21],
    'vix_ma_period_options': [1, 2, 5, 10, 20],
    'bb_length_options': [10, 15, 20],
    'bb_std_options': [1.5, 2.0],
    'adx_period': 14,
    'ma_short_period': 5,
    'ma_long_period': 10,
    'commission_pct': 0.003,
}

GA_PARAMS_CONFIG_B = {
    'generations': 15,
    'population_size': 60,
    'crossover_rate': 0.7,
    'mutation_rate': 0.25,
    'elitism_size': 3,
    'tournament_size': 5,
    'show_process': False,
    'mutation_amount_range': (-3, 3),
    'vix_mutation_amount_range': (-2, 2),
    'adx_mutation_amount_range': (-2, 2),
    'rsi_threshold_range': (10, 45, 46, 85),  # (min_buy, max_buy, min_exit, max_exit)
    'vix_threshold_range': (15, 30),
    'adx_threshold_range': (20, 40),
    'commission_rate': 0.003,
    'min_trades_for_full_score': 4,
    'no_trade_penalty_factor': 0.1,
    'low_trade_penalty_factor': 0.75,
    # NSGA-II 特定配置
    'nsga2_enabled': False,
    'nsga2_selection_method': 'custom_balance',
    'min_required_trades': 5,
    'nsga2_no_trade_penalty_return': -0.5,
    'nsga2_no_trade_penalty_max_drawdown': 1.0,
    'nsga2_no_trade_penalty_profit_factor': 0.01,
    **STRATEGY_CONFIG_B
}

# 基因對應表 (10基因系統)
GENE_MAP_B = {
    'rsi_buy_entry': 0,      # RSI買入閾值
    'rsi_exit': 1,           # RSI賣出閾值  
    'vix_threshold': 2,      # VIX波動率閾值
    'low_vol_exit': 3,       # 低波動賣出策略選擇
    'rsi_period_choice': 4,  # RSI週期選擇
    'vix_ma_choice': 5,      # VIX MA週期選擇
    'bb_length_choice': 6,   # 布林帶長度選擇
    'bb_std_choice': 7,      # 布林帶標準差選擇
    'adx_threshold': 8,      # ADX閾值
    'high_vol_entry': 9      # 高波動進場策略選擇
}

# ═══════════════════════════════════════════════════════════════
# 📈 數據載入函數 
# ═══════════════════════════════════════════════════════════════

def load_stock_data_b(ticker, vix_ticker="^VIX", start_date=None, end_date=None, verbose=False):
    """系統B專用數據載入函數"""
    if verbose: 
        logger.info(f"正在載入 {ticker} 和 {vix_ticker} 的數據 ({start_date} ~ {end_date})")
    
    try:
        data = yf.download([ticker, vix_ticker], start=start_date, end=end_date, 
                          progress=False, auto_adjust=False, timeout=30)
        
        if data is None or data.empty:
            if verbose: 
                logger.warning(f"yfinance 未能載入 {ticker} 的數據")
            return None, None, None, None

        # 處理單一股票情況
        if not isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns:
                logger.warning(f"VIX數據可能缺失，僅使用股票數據")
                stock_data = data.copy()
                vix_data = pd.Series(np.nan, index=stock_data.index, name='VIX_Close')
            else:
                if verbose: 
                    logger.error(f"預期多重索引列，但得到 {data.columns}")
                return None, None, None, None
        else:
            # 處理多重索引情況
            stock_present = ticker in data.columns.get_level_values(1)
            vix_present = vix_ticker in data.columns.get_level_values(1)
            
            if not stock_present:
                if verbose: 
                    logger.error(f"缺少股票 {ticker} 的數據")
                return None, None, None, None
            
            # 提取股票數據
            stock_data = data.loc[:, pd.IndexSlice[:, ticker]]
            stock_data.columns = stock_data.columns.droplevel(1)
            
            # 提取VIX數據
            if not vix_present:
                if verbose: 
                    logger.warning(f"缺少VIX數據，將使用NaN填充")
                vix_data = pd.Series(np.nan, index=stock_data.index, name='VIX_Close')
            else:
                vix_data_slice = data.loc[:, pd.IndexSlice['Close', vix_ticker]]
                if isinstance(vix_data_slice, pd.DataFrame):
                    if vix_data_slice.shape[1] == 1:
                        vix_data = vix_data_slice.iloc[:, 0]
                    else:
                        if verbose: 
                            logger.error(f"無法提取VIX收盤價 (形狀: {vix_data_slice.shape})")
                        return None, None, None, None
                elif isinstance(vix_data_slice, pd.Series):
                    vix_data = vix_data_slice
                else:
                    if verbose: 
                        logger.error(f"無法識別的VIX數據類型: {type(vix_data_slice)}")
                    return None, None, None, None
                vix_data.name = 'VIX_Close'

        # 檢查必要列
        required_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        missing_essential = [col for col in ['Close', 'High', 'Low'] if col not in stock_data.columns]
        
        if missing_essential:
            if verbose: 
                logger.error(f"缺少必要列 {missing_essential}")
            return None, None, None, None

        # 補充缺失列
        for col in required_cols:
            if col not in stock_data.columns:
                stock_data[col] = np.nan
                if verbose: 
                    logger.warning(f"列 '{col}' 缺失，已用NaN填充")

        # 數據對齊和清理
        simplified_df = stock_data[required_cols].copy()
        vix_data = vix_data.reindex(simplified_df.index)
        aligned_data = pd.concat([simplified_df, vix_data], axis=1, join='inner')

        if aligned_data.empty:
            if verbose: 
                logger.error(f"股票和VIX數據沒有共同日期")
            return None, None, None, None

        # 處理NaN值
        if aligned_data.isnull().values.any():
            if verbose: 
                logger.warning(f"發現NaN值，正在填充...")
            
            # 填充VIX數據
            if 'VIX_Close' in aligned_data.columns and aligned_data['VIX_Close'].isnull().any():
                aligned_data['VIX_Close'] = aligned_data['VIX_Close'].ffill().bfill()
            
            # 填充股票數據
            numeric_cols = [col for col in required_cols if col in aligned_data.columns 
                           and pd.api.types.is_numeric_dtype(aligned_data[col])]
            for col in numeric_cols:
                if aligned_data[col].isnull().any():
                    aligned_data[col] = aligned_data[col].ffill().bfill()

            # 檢查是否還有關鍵列的NaN
            if aligned_data.isnull().values.any():
                nan_cols = aligned_data.columns[aligned_data.isnull().any()].tolist()
                if any(col in nan_cols for col in ['Close', 'High', 'Low']):
                    if verbose: 
                        logger.error(f"無法填充關鍵列的NaN: {nan_cols}")
                    return None, None, None, None
                logger.warning(f"部分非關鍵列仍有NaN: {nan_cols}")

        # 最終數據準備
        final_stock_df = aligned_data[required_cols].copy()
        final_vix_series = aligned_data['VIX_Close'].copy() if 'VIX_Close' in aligned_data else pd.Series(np.nan, index=final_stock_df.index)
        
        prices = final_stock_df['Close'].tolist()
        dates = final_stock_df.index.tolist()

        logger.info(f"成功載入 {len(prices)} 個數據點 for {ticker}")
        return prices, dates, final_stock_df, final_vix_series

    except Exception as e:
        logger.error(f"載入 {ticker} 數據時發生錯誤: {type(e).__name__}: {e}", exc_info=True)
        return None, None, None, None

# ═══════════════════════════════════════════════════════════════
# 📊 技術指標預計算
# ═══════════════════════════════════════════════════════════════

def precompute_indicators_b(stock_df, vix_series, strategy_config, verbose=False):
    """系統B專用指標預計算"""
    precalculated = {
        'rsi': {},
        'vix_ma': {},
        'bbl': {},
        'bbm': {},
        'fixed': {}
    }
    
    if verbose:
        print(f"[GAEngine_B] 開始指標預計算 (股票數據: {stock_df.shape}, VIX序列: {len(vix_series)})")

    try:
        # 計算RSI
        for rsi_p in strategy_config['rsi_period_options']:
            rsi_values = ta.rsi(stock_df['Close'], length=rsi_p)
            if rsi_values is None or rsi_values.isnull().all():
                if verbose:
                    print(f"[GAEngine_B] ❌ RSI({rsi_p}) 計算失敗")
                return {}, False
            precalculated['rsi'][rsi_p] = rsi_values.tolist()
        
        if verbose:
            print(f"[GAEngine_B] ✅ RSI計算完成 ({len(precalculated['rsi'])} 種週期)")

        # 計算VIX移動平均
        for vix_ma_p in strategy_config['vix_ma_period_options']:
            vix_ma_values = vix_series.rolling(window=vix_ma_p).mean()
            if vix_ma_values is None or vix_ma_values.isnull().all():
                precalculated['vix_ma'][vix_ma_p] = [np.nan] * len(vix_series)
                if verbose:
                    print(f"[GAEngine_B] ⚠️ VIX_MA({vix_ma_p}) 計算失敗，用NaN填充")
            else:
                precalculated['vix_ma'][vix_ma_p] = vix_ma_values.tolist()
        
        if verbose:
            print(f"[GAEngine_B] ✅ VIX移動平均計算完成 ({len(precalculated['vix_ma'])} 種週期)")

        # 計算布林帶
        for bb_l in strategy_config['bb_length_options']:
            for bb_s in strategy_config['bb_std_options']:
                bbands = ta.bbands(stock_df['Close'], length=bb_l, std=bb_s)
                if bbands is None or bbands.empty:
                    if verbose:
                        print(f"[GAEngine_B] ❌ 布林帶({bb_l}, {bb_s}) 計算失敗")
                    return {}, False
                
                # 尋找布林帶列名
                bbl_col = next((col for col in bbands.columns if 'BBL' in col), None)
                bbm_col = next((col for col in bbands.columns if 'BBM' in col), None)
                
                if not bbl_col or not bbm_col or bbands[bbl_col].isnull().all() or bbands[bbm_col].isnull().all():
                    if verbose:
                        print(f"[GAEngine_B] ❌ 布林帶({bb_l}, {bb_s}) 列不完整")
                    return {}, False
                
                precalculated['bbl'][(bb_l, bb_s)] = bbands[bbl_col].tolist()
                precalculated['bbm'][(bb_l, bb_s)] = bbands[bbm_col].tolist()

        if verbose:
            print(f"[GAEngine_B] ✅ 布林帶計算完成 ({len(precalculated['bbl'])} 種組合)")

        # 計算固定週期指標
        # ADX
        adx_df = ta.adx(stock_df['High'], stock_df['Low'], stock_df['Close'], 
                       length=strategy_config['adx_period'])
        if adx_df is None or adx_df.empty:
            if verbose:
                print(f"[GAEngine_B] ❌ ADX計算失敗")
            return {}, False
        
        adx_col = next((col for col in adx_df.columns if 'ADX' in col), None)
        if not adx_col or adx_df[adx_col].isnull().all():
            if verbose:
                print(f"[GAEngine_B] ❌ ADX列不完整")
            return {}, False
        
        precalculated['fixed']['adx_list'] = adx_df[adx_col].tolist()

        # 移動平均
        ma_short = stock_df['Close'].rolling(window=strategy_config['ma_short_period']).mean()
        ma_long = stock_df['Close'].rolling(window=strategy_config['ma_long_period']).mean()
        
        if ma_short is None or ma_long is None or ma_short.isnull().all() or ma_long.isnull().all():
            if verbose:
                print(f"[GAEngine_B] ❌ 移動平均計算失敗")
            return {}, False
        
        precalculated['fixed']['ma_short_list'] = ma_short.tolist()
        precalculated['fixed']['ma_long_list'] = ma_long.tolist()

        if verbose:
            print(f"[GAEngine_B] ✅ 固定週期指標計算完成")
            print(f"[GAEngine_B] 🎉 所有指標預計算完成")

        return precalculated, True

    except Exception as e:
        if verbose:
            print(f"[GAEngine_B] ❌ 指標預計算過程中發生錯誤: {type(e).__name__}: {e}")
            traceback.print_exc()
        return {}, False

# ═══════════════════════════════════════════════════════════════
# 🔥 Numba加速策略核心 (系統B)
# ═══════════════════════════════════════════════════════════════

@numba.jit(nopython=True)
def run_strategy_numba_core_b(
    rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold, vix_threshold,
    low_vol_exit_strategy, high_vol_entry_choice, commission_rate,
    prices_arr, rsi_arr, bbl_arr, bbm_arr, adx_arr, vix_ma_arr, 
    ma_short_arr, ma_long_arr, start_trading_iloc):
    """
    系統B Numba加速策略核心 (10基因策略)
    
    基因結構:
    0: rsi_buy_entry_threshold - RSI買入閾值
    1: rsi_exit_threshold - RSI賣出閾值  
    2: vix_threshold - VIX波動率閾值
    3: low_vol_exit_strategy - 低波動退出策略 (0/1)
    4: rsi_period_choice - RSI週期選擇
    5: vix_ma_choice - VIX MA週期選擇
    6: bb_length_choice - 布林帶長度選擇
    7: bb_std_choice - 布林帶標準差選擇
    8: adx_threshold - ADX閾值
    9: high_vol_entry_choice - 高波動進場策略 (0=BB+RSI, 1=BB+ADX)
    """
    
    T = len(prices_arr)
    portfolio_values_arr = np.full(T, 1.0, dtype=np.float64)
    
    # 交易信號記錄
    max_signals = T // 2 + 1
    buy_signal_indices = np.full(max_signals, -1, dtype=np.int64)
    buy_signal_prices = np.full(max_signals, np.nan, dtype=np.float64)
    buy_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    sell_signal_indices = np.full(max_signals, -1, dtype=np.int64)
    sell_signal_prices = np.full(max_signals, np.nan, dtype=np.float64)
    sell_signal_rsis = np.full(max_signals, np.nan, dtype=np.float64)
    
    # 交易狀態變數
    buy_count = 0
    sell_count = 0
    cash = 1.0
    stock = 0.0
    position = 0  # 0=空倉, 1=持倉
    last_valid_portfolio_value = 1.0
    
    # RSI策略狀態追蹤
    rsi_crossed_exit_level_after_buy = False
    high_vol_entry_type = -1  # 記錄高波動進場類型
    
    # 確保開始交易位置有效
    start_trading_iloc = max(1, start_trading_iloc)
    if start_trading_iloc >= T:
        portfolio_values_arr[:] = 1.0
        return (portfolio_values_arr, 
                buy_signal_indices[:0], buy_signal_prices[:0], buy_signal_rsis[:0],
                sell_signal_indices[:0], sell_signal_prices[:0], sell_signal_rsis[:0])
    
    # 初始化投資組合價值
    portfolio_values_arr[:start_trading_iloc] = 1.0
    
    # 主要交易迴圈
    for i in range(start_trading_iloc, T):
        current_price = prices_arr[i]
        rsi_i = rsi_arr[i]
        rsi_prev = rsi_arr[i-1] if i > 0 else rsi_arr[i]
        bbl_i = bbl_arr[i]
        bbm_i = bbm_arr[i]
        adx_i = adx_arr[i]
        vix_ma_i = vix_ma_arr[i]
        ma_short_i = ma_short_arr[i]
        ma_long_i = ma_long_arr[i]
        ma_short_prev = ma_short_arr[i-1] if i > 0 else ma_short_arr[i]
        ma_long_prev = ma_long_arr[i-1] if i > 0 else ma_long_arr[i]
        
        # 檢查所有必要數值的有效性
        required_values = (rsi_i, rsi_prev, current_price, bbl_i, bbm_i, adx_i, 
                          vix_ma_i, ma_short_i, ma_long_i, ma_short_prev, ma_long_prev)
        
        is_valid = True
        for val in required_values:
            if not np.isfinite(val):
                is_valid = False
                break
        
        if not is_valid:
            # 如果數據無效，維持上一期投資組合價值
            current_portfolio_value = cash if position == 0 else (stock * current_price if np.isfinite(current_price) else np.nan)
            portfolio_values_arr[i] = last_valid_portfolio_value if np.isnan(current_portfolio_value) else current_portfolio_value
            if not np.isnan(current_portfolio_value):
                last_valid_portfolio_value = current_portfolio_value
            continue
        
        # 📈 買入邏輯 (空倉時)
        if position == 0:
            # 判斷市場狀態
            is_high_vol = vix_ma_i >= vix_threshold
            
            buy_condition = False
            entry_type_if_bought = -1
            
            if is_high_vol:
                # 高波動市場策略
                if high_vol_entry_choice == 0:
                    # BB+RSI策略: 價格觸及下軌 且 RSI低於買入閾值
                    if (current_price <= bbl_i) and (rsi_i < rsi_buy_entry_threshold):
                        buy_condition = True
                        entry_type_if_bought = 0
                else:
                    # BB+ADX策略: 價格觸及下軌 且 ADX確認趨勢強度
                    if (current_price <= bbl_i) and (adx_i > adx_threshold):
                        buy_condition = True
                        entry_type_if_bought = 1
            else:
                # 低波動市場策略: MA交叉
                if (ma_short_prev < ma_long_prev) and (ma_short_i >= ma_long_i):
                    buy_condition = True
                    entry_type_if_bought = -1  # 標記為低波動策略
            
            # 執行買入
            if buy_condition and current_price > 1e-9:
                cost = cash * commission_rate
                amount_to_invest = cash - cost
                if amount_to_invest > 0:
                    stock = amount_to_invest / current_price
                    cash = 0.0
                    position = 1
                    rsi_crossed_exit_level_after_buy = False
                    high_vol_entry_type = entry_type_if_bought
                    
                    # 記錄買入信號
                    if buy_count < max_signals:
                        buy_signal_indices[buy_count] = i
                        buy_signal_prices[buy_count] = current_price
                        buy_signal_rsis[buy_count] = rsi_i
                        buy_count += 1
        
        # 📉 賣出邏輯 (持倉時)
        elif position == 1:
            sell_condition = False
            
            if high_vol_entry_type == 0:
                # BB+RSI策略的賣出邏輯
                if rsi_i >= rsi_exit_threshold:
                    rsi_crossed_exit_level_after_buy = True
                if rsi_crossed_exit_level_after_buy and rsi_i < rsi_exit_threshold:
                    sell_condition = True
            
            elif high_vol_entry_type == 1:
                # BB+ADX策略的賣出邏輯: 價格回到布林帶中軌之上
                sell_condition = (current_price >= bbm_i)
            
            elif high_vol_entry_type == -1:
                # 低波動策略的賣出邏輯
                if low_vol_exit_strategy == 0:
                    # 策略0: 價格跌破短期MA
                    sell_condition = (current_price < ma_short_i)
                else:
                    # 策略1: MA死叉
                    sell_condition = (ma_short_prev > ma_long_prev) and (ma_short_i <= ma_long_i)
            
            # 執行賣出
            if sell_condition:
                proceeds = stock * current_price
                cost = proceeds * commission_rate
                cash = proceeds - cost
                stock = 0.0
                position = 0
                
                # 重置狀態
                rsi_crossed_exit_level_after_buy = False
                high_vol_entry_type = -1
                
                # 記錄賣出信號
                if sell_count < max_signals:
                    sell_signal_indices[sell_count] = i
                    sell_signal_prices[sell_count] = current_price
                    sell_signal_rsis[sell_count] = rsi_i
                    sell_count += 1
        
        # 📊 更新投資組合價值
        current_stock_value = stock * current_price if position == 1 else 0.0
        current_portfolio_value = cash + current_stock_value
        
        portfolio_values_arr[i] = last_valid_portfolio_value if np.isnan(current_portfolio_value) else current_portfolio_value
        if not np.isnan(current_portfolio_value):
            last_valid_portfolio_value = current_portfolio_value
    
    # 確保最後一期有有效值
    if T > 0 and np.isnan(portfolio_values_arr[-1]):
        portfolio_values_arr[-1] = last_valid_portfolio_value
    
    return (portfolio_values_arr,
            buy_signal_indices[:buy_count], buy_signal_prices[:buy_count], buy_signal_rsis[:buy_count],
            sell_signal_indices[:sell_count], sell_signal_prices[:sell_count], sell_signal_rsis[:sell_count])

# ═══════════════════════════════════════════════════════════════
# 🎯 策略包裝函數
# ═══════════════════════════════════════════════════════════════

def run_strategy_b(rsi_buy_entry_threshold, rsi_exit_threshold, adx_threshold, vix_threshold,
                   low_vol_exit_strategy, high_vol_entry_choice, commission_rate,
                   prices, dates, rsi_list, bbl_list, bbm_list, adx_list, 
                   vix_ma_list, ma_short_list, ma_long_list):
    """系統B策略包裝函數"""
    
    T = len(prices)
    if T == 0:
        return [1.0], [], []
    
    # 轉換為numpy數組
    prices_arr = np.array(prices, dtype=np.float64)
    rsi_arr = np.array(rsi_list, dtype=np.float64)
    bbl_arr = np.array(bbl_list, dtype=np.float64)
    bbm_arr = np.array(bbm_list, dtype=np.float64)
    adx_arr = np.array(adx_list, dtype=np.float64)
    vix_ma_arr = np.array(vix_ma_list, dtype=np.float64)
    ma_short_arr = np.array(ma_short_list, dtype=np.float64)
    ma_long_arr = np.array(ma_long_list, dtype=np.float64)
    
    # 找到所有指標都有效的開始位置
    def get_first_valid_iloc(indicator_arr):
        valid_indices = np.where(np.isfinite(indicator_arr))[0]
        return valid_indices[0] if len(valid_indices) > 0 else T
    
    start_locs = [
        get_first_valid_iloc(rsi_arr),
        get_first_valid_iloc(bbl_arr),
        get_first_valid_iloc(bbm_arr),
        get_first_valid_iloc(adx_arr),
        get_first_valid_iloc(vix_ma_arr),
        get_first_valid_iloc(ma_short_arr),
        get_first_valid_iloc(ma_long_arr)
    ]
    
    start_trading_iloc = max(start_locs) + 1
    if start_trading_iloc >= T:
        return [1.0] * T, [], []
    
    start_trading_iloc = max(start_trading_iloc, 1)
    
    # 執行策略
    (portfolio_values_arr, buy_indices, buy_prices, buy_rsis, 
     sell_indices, sell_prices, sell_rsis) = run_strategy_numba_core_b(
        float(rsi_buy_entry_threshold), float(rsi_exit_threshold), 
        float(adx_threshold), float(vix_threshold),
        int(low_vol_exit_strategy), int(high_vol_entry_choice),
        float(commission_rate),
        prices_arr, rsi_arr, bbl_arr, bbm_arr, adx_arr, vix_ma_arr,
        ma_short_arr, ma_long_arr, start_trading_iloc
    )
    
    # 轉換信號格式
    buy_signals = []
    sell_signals = []
    
    for idx, price, rsi_val in zip(buy_indices, buy_prices, buy_rsis):
        if idx != -1 and idx < len(dates):
            buy_signals.append((dates[idx], price, rsi_val))
    
    for idx, price, rsi_val in zip(sell_indices, sell_prices, sell_rsis):
        if idx != -1 and idx < len(dates):
            sell_signals.append((dates[idx], price, rsi_val))
    
    return portfolio_values_arr.tolist(), buy_signals, sell_signals

# ═══════════════════════════════════════════════════════════════
# 🧬 有效基因採樣器 (系統B)
# ═══════════════════════════════════════════════════════════════

class ValidGASampling_B(Sampling):
    """系統B的有效基因採樣器"""
    
    def __init__(self, ga_params):
        super().__init__()
        self.ga_params = ga_params
        
    def _do(self, problem, n_samples, **kwargs):
        population = []
        
        # 提取基因範圍
        min_buy, max_buy, min_exit, max_exit = self.ga_params['rsi_threshold_range']
        min_vix, max_vix = self.ga_params['vix_threshold_range']
        min_adx, max_adx = self.ga_params['adx_threshold_range']
        
        num_rsi_options = len(self.ga_params['rsi_period_options'])
        num_vix_ma_options = len(self.ga_params['vix_ma_period_options'])
        num_bb_len_options = len(self.ga_params['bb_length_options'])
        num_bb_std_options = len(self.ga_params['bb_std_options'])
        
        def is_gene_valid(gene):
            """檢查系統B基因有效性"""
            # RSI閾值約束: 買入閾值 < 賣出閾值
            if gene[0] >= gene[1]:
                return False
            return True
        
        print(f"[GAEngine_B] NSGA-II 正在生成 {n_samples} 個有效基因...")
        
        attempts = 0
        max_attempts = n_samples * 500
        
        while len(population) < n_samples and attempts < max_attempts:
            # 生成10基因
            gene = np.zeros(10, dtype=int)
            
            # 基因0: RSI買入閾值
            gene[0] = random.randint(min_buy, max_buy)
            
            # 基因1: RSI賣出閾值 (必須大於買入閾值)
            gene[1] = random.randint(max(gene[0] + 1, min_exit), max_exit)
            
            # 基因2: VIX閾值
            gene[2] = random.randint(min_vix, max_vix)
            
            # 基因3: 低波動退出策略 (0或1)
            gene[3] = random.choice([0, 1])
            
            # 基因4: RSI週期選擇
            gene[4] = random.randint(0, num_rsi_options - 1)
            
            # 基因5: VIX MA週期選擇
            gene[5] = random.randint(0, num_vix_ma_options - 1)
            
            # 基因6: 布林帶長度選擇
            gene[6] = random.randint(0, num_bb_len_options - 1)
            
            # 基因7: 布林帶標準差選擇
            gene[7] = random.randint(0, num_bb_std_options - 1)
            
            # 基因8: ADX閾值
            gene[8] = random.randint(min_adx, max_adx)
            
            # 基因9: 高波動進場策略 (0或1)
            gene[9] = random.choice([0, 1])
            
            if is_gene_valid(gene):
                population.append(gene)
            
            attempts += 1
        
        if len(population) < n_samples:
            print(f"[GAEngine_B] WARNING: 只生成了 {len(population)}/{n_samples} 個有效基因")
            # 填充不足
            while len(population) < n_samples:
                if population:
                    population.append(population[0])
                else:
                    # 緊急基因
                    emergency_gene = np.array([25, 65, 20, 0, 0, 0, 0, 0, 25, 0], dtype=int)
                    population.append(emergency_gene)
        
        print(f"[GAEngine_B] NSGA-II 成功生成 {len(population)} 個有效基因（嘗試 {attempts} 次）")
        return np.array(population, dtype=float)

# ═══════════════════════════════════════════════════════════════
# 🎯 NSGA-II 多目標優化問題定義 (系統B)
# ═══════════════════════════════════════════════════════════════

class MultiObjectiveStrategyProblem_B(Problem):
    """系統B多目標策略優化問題"""
    
    def __init__(self, prices, dates, precalculated_indicators, ga_params):
        self.prices = prices
        self.dates = dates
        self.precalculated = precalculated_indicators
        self.ga_params = ga_params
        
        # 設定變數範圍
        xl = np.zeros(10)
        xu = np.zeros(10)
        
        min_buy, max_buy, min_exit, max_exit = ga_params['rsi_threshold_range']
        min_vix, max_vix = ga_params['vix_threshold_range']
        min_adx, max_adx = ga_params['adx_threshold_range']
        
        # 基因範圍
        xl[0] = min_buy; xu[0] = max_buy  # RSI買入閾值
        xl[1] = min_exit; xu[1] = max_exit  # RSI賣出閾值
        xl[2] = min_vix; xu[2] = max_vix  # VIX閾值
        xl[3] = 0; xu[3] = 1  # 低波動退出策略
        xl[4] = 0; xu[4] = len(ga_params['rsi_period_options']) - 1  # RSI週期
        xl[5] = 0; xu[5] = len(ga_params['vix_ma_period_options']) - 1  # VIX MA週期
        xl[6] = 0; xu[6] = len(ga_params['bb_length_options']) - 1  # BB長度
        xl[7] = 0; xu[7] = len(ga_params['bb_std_options']) - 1  # BB標準差
        xl[8] = min_adx; xu[8] = max_adx  # ADX閾值
        xl[9] = 0; xu[9] = 1  # 高波動進場策略
        
        # 4個目標 + 1個約束
        super().__init__(n_var=10, n_obj=4, n_constr=1, xl=xl, xu=xu, type_var=int)
    
    def _evaluate(self, X, out, *args, **kwargs):
        """評估目標函數和約束"""
        objectives = np.zeros((X.shape[0], 4))
        constraints = np.zeros((X.shape[0], 1))
        
        for i, gene in enumerate(X):
            try:
                portfolio_values, buy_signals, sell_signals = self._run_backtest_raw(gene)
                metrics = self._calculate_metrics(portfolio_values, buy_signals, sell_signals)
                
                # 目標函數 (轉為最小化)
                objectives[i, 0] = -metrics['total_return'] * 1.5  # 最大化報酬率
                objectives[i, 1] = metrics['max_drawdown']  # 最小化回撤
                objectives[i, 2] = -metrics['profit_factor']  # 最大化獲利因子
                objectives[i, 3] = -metrics['average_trade_return']  # 最大化平均交易報酬
                
                # 約束條件: 交易次數 >= 最少要求
                min_trades = self.ga_params.get('min_required_trades', 5)
                constraints[i, 0] = min_trades - metrics['trade_count']
                
            except Exception as e:
                print(f"[GAEngine_B] 評估基因 {i} 時錯誤: {e}")
                objectives[i, :] = [-0.5, 1.0, -0.01, -0.001]
                constraints[i, 0] = 10
        
        out["F"] = objectives
        out["G"] = constraints
    
    def _run_backtest_raw(self, gene):
        """執行回測"""
        # 提取基因參數
        rsi_buy_entry = gene[0]
        rsi_exit = gene[1]
        vix_threshold = gene[2]
        low_vol_exit_strategy = int(gene[3])
        rsi_period_choice = int(gene[4])
        vix_ma_choice = int(gene[5])
        bb_length_choice = int(gene[6])
        bb_std_choice = int(gene[7])
        adx_threshold = gene[8]
        high_vol_entry_choice = int(gene[9])
        
        # 獲取對應的指標數據
        rsi_period = self.ga_params['rsi_period_options'][rsi_period_choice]
        vix_ma_period = self.ga_params['vix_ma_period_options'][vix_ma_choice]
        bb_length = self.ga_params['bb_length_options'][bb_length_choice]
        bb_std = self.ga_params['bb_std_options'][bb_std_choice]
        
        rsi_list = self.precalculated['rsi'][rsi_period]
        vix_ma_list = self.precalculated['vix_ma'][vix_ma_period]
        bbl_list = self.precalculated['bbl'][(bb_length, bb_std)]
        bbm_list = self.precalculated['bbm'][(bb_length, bb_std)]
        adx_list = self.precalculated['fixed']['adx_list']
        ma_short_list = self.precalculated['fixed']['ma_short_list']
        ma_long_list = self.precalculated['fixed']['ma_long_list']
        
        # 執行策略
        portfolio_values, buy_signals, sell_signals = run_strategy_b(
            rsi_buy_entry, rsi_exit, adx_threshold, vix_threshold,
            low_vol_exit_strategy, high_vol_entry_choice,
            self.ga_params['commission_rate'],
            self.prices, self.dates,
            rsi_list, bbl_list, bbm_list, adx_list, 
            vix_ma_list, ma_short_list, ma_long_list
        )
        
        return portfolio_values, buy_signals, sell_signals
    
    def _calculate_metrics(self, portfolio_values, buy_signals, sell_signals):
        """計算績效指標 (包含平均交易報酬率和勝率) - 修復版"""
        if not portfolio_values or len(portfolio_values) < 2:
            return {
                'total_return': -0.5, 'max_drawdown': 1.0, 'profit_factor': 0.01,
                'trade_count': 0, 'average_trade_return': 0.0, 'win_rate_pct': 0.0
            }

        final_value = portfolio_values[-1]
        total_return = final_value - 1.0

        # 🔥 加入交易獎勵機制
        completed_trades = min(len(buy_signals), len(sell_signals))
        trade_bonus = 1.0

        if completed_trades == 0:
            trade_bonus = self.ga_params.get('no_trade_penalty_factor', 0.1)
        elif completed_trades < self.ga_params.get('min_trades_for_full_score', 4):
            trade_bonus = self.ga_params.get('low_trade_penalty_factor', 0.75)

        # 調整報酬率
        adjusted_total_return = (final_value * trade_bonus) - 1.0

        # 計算最大回撤
        portfolio_arr = np.array(portfolio_values)
        running_max = np.maximum.accumulate(portfolio_arr)
        drawdowns = (running_max - portfolio_arr) / running_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # 🔥 修復：計算交易指標 + 勝率
        total_profit = 0.0
        total_loss = 0.0
        total_trade_returns = 0.0
        valid_trades = 0
        winning_trades = 0  # 🌟 新增：獲利交易次數統計

        for i in range(completed_trades):
            try:
                buy_price = buy_signals[i][1]
                sell_price = sell_signals[i][1]
                
                if buy_price > 0 and np.isfinite(buy_price) and np.isfinite(sell_price):
                    trade_return = sell_price - buy_price
                    trade_return_pct = trade_return / buy_price

                    if trade_return > 0:
                        total_profit += trade_return
                        winning_trades += 1  # 🌟 統計獲利交易
                    else:
                        total_loss += abs(trade_return)

                    total_trade_returns += trade_return_pct
                    valid_trades += 1
            except (IndexError, TypeError):
                continue

        profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 1.0)
        average_trade_return = total_trade_returns / valid_trades if valid_trades > 0 else 0.0
        
        # 🌟 修復：計算勝率
        win_rate_pct = (winning_trades / valid_trades) * 100 if valid_trades > 0 else 0.0

        return {
            'total_return': adjusted_total_return,
            'max_drawdown': max_drawdown,
            'profit_factor': max(profit_factor, 0.01),
            'trade_count': completed_trades,
            'average_trade_return': average_trade_return,
            'win_rate_pct': win_rate_pct  # 🌟 新增：返回勝率
        }


# ═══════════════════════════════════════════════════════════════
# 🚀 統一遺傳算法函數 (系統B)
# ═══════════════════════════════════════════════════════════════

def genetic_algorithm_unified_b(prices, dates, precalculated_indicators, ga_params):
    """系統B統一遺傳算法函數"""
    use_nsga2 = ga_params.get('nsga2_enabled', False) and NSGA2_AVAILABLE
    
    if use_nsga2:
        print("[GAEngine_B] 使用 NSGA-II 多目標優化")
        return nsga2_optimize_b(prices, dates, precalculated_indicators, ga_params)
    else:
        print("[GAEngine_B] 使用傳統單目標GA")
        # 導入並調用原有的遺傳算法函數
        from model_train2 import genetic_algorithm_with_elitism
        
        # 準備數據格式
        rsi_lists = precalculated_indicators['rsi']
        vix_ma_lists = precalculated_indicators['vix_ma']
        bbl_lists = precalculated_indicators['bbl']
        bbm_lists = precalculated_indicators['bbm']
        adx_list = precalculated_indicators['fixed']['adx_list']
        ma_short_list = precalculated_indicators['fixed']['ma_short_list']
        ma_long_list = precalculated_indicators['fixed']['ma_long_list']
        
        result = genetic_algorithm_with_elitism(
            prices, dates, rsi_lists, vix_ma_lists, bbl_lists, bbm_lists,
            adx_list, ma_short_list, ma_long_list, ga_params
        )
        
        return result

def nsga2_optimize_b(prices, dates, precalculated_indicators, ga_params):
    """系統B NSGA-II 多目標優化"""
    if not NSGA2_AVAILABLE:
        print("[GAEngine_B] ERROR: NSGA-II 不可用")
        return None, None
    
    print("[GAEngine_B] 開始 NSGA-II 多目標優化...")
    
    try:
        problem = MultiObjectiveStrategyProblem_B(prices, dates, precalculated_indicators, ga_params)
        
        algorithm = NSGA2(
            pop_size=ga_params.get('population_size', 60),
            sampling=ValidGASampling_B(ga_params),
            crossover=SBX(prob=ga_params.get('crossover_rate', 0.7), eta=15),
            mutation=PM(prob=ga_params.get('mutation_rate', 0.25), eta=20),
            eliminate_duplicates=True
        )
        
        res = minimize(
            problem,
            algorithm,
            ('n_gen', ga_params.get('generations', 15)),
            verbose=ga_params.get('show_process', False)
        )
        
        if res.X is None or len(res.X) == 0:
            print("[GAEngine_B] WARN: NSGA-II 未找到有效解")
            return None, None
        
        pareto_genes = [gene.astype(int).tolist() for gene in res.X]
        pareto_objectives = res.F
        
        best_gene, best_metrics = select_best_from_pareto_b(
            pareto_genes, pareto_objectives, prices, dates, precalculated_indicators,
            ga_params.get('nsga2_selection_method', 'custom_balance'), ga_params
        )
        
        if best_gene is None:
            print("[GAEngine_B] WARN: 帕累托前沿選擇失敗")
            return None, None
        
        print(f"[GAEngine_B] NSGA-II 完成，找到 {len(pareto_genes)} 個帕累托解")
        print(f"[GAEngine_B] 最佳解: 報酬率={best_metrics.get('total_return', 0)*100:.2f}%, "
              f"平均交易報酬={best_metrics.get('average_trade_return', 0)*100:.3f}%, "
              f"交易次數={best_metrics.get('trade_count', 0)}")
        
        return best_gene, best_metrics
        
    except Exception as e:
        print(f"[GAEngine_B] ERROR: NSGA-II 優化錯誤: {e}")
        traceback.print_exc()
        return None, None

def select_best_from_pareto_b(pareto_genes, pareto_objectives, prices, dates, precalculated, selection_method, ga_params):
    """從帕累托前沿選擇最佳策略 (系統B) - 修復權重讀取"""
    if len(pareto_genes) == 0:
        return None, {}
    
    all_metrics = []
    temp_problem = MultiObjectiveStrategyProblem_B(prices, dates, precalculated, ga_params)
    
    for gene in pareto_genes:
        try:
            portfolio_values, buy_signals, sell_signals = temp_problem._run_backtest_raw(np.array(gene))
            metrics = temp_problem._calculate_metrics(portfolio_values, buy_signals, sell_signals)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"[GAEngine_B] WARN: 計算帕累托解指標錯誤: {e}")
            all_metrics.append({
                'total_return': -0.5, 'max_drawdown': 1.0, 'profit_factor': 0.01,
                'trade_count': 0, 'average_trade_return': 0.0, 'win_rate_pct': 0.0
            })
    
    best_idx = 0
    if selection_method == 'custom_balance':
        # 🔥🔥🔥 --- 系統B 修正 --- 🔥🔥🔥
        custom_weights = ga_params.get('custom_weights', {
            'total_return_weight': 0.35, 'avg_trade_return_weight': 0.30,
            'win_rate_weight': 0.25, 'trade_count_weight': 0.05, 'drawdown_weight': 0.05
        })
        print(f"[GAEngine_B] 使用自定義權重進行選擇: {custom_weights}")

        def normalize(arr):
            min_val, max_val = np.min(arr), np.max(arr)
            return (arr - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-9 else np.full_like(arr, 0.5)

        all_returns = np.array([m['total_return'] for m in all_metrics])
        all_avg_trade_returns = np.array([m['average_trade_return'] for m in all_metrics])
        all_win_rates = np.array([m.get('win_rate_pct', 0) for m in all_metrics])
        all_trade_counts = np.array([m['trade_count'] for m in all_metrics])
        all_max_drawdowns = np.array([m['max_drawdown'] for m in all_metrics])

        norm_returns = normalize(all_returns)
        norm_avg_trade_returns = normalize(all_avg_trade_returns)
        norm_win_rates = normalize(all_win_rates)
        norm_trade_counts = normalize(all_trade_counts)
        norm_drawdowns_inv = 1 - normalize(all_max_drawdowns)

        balanced_scores = (
            norm_returns * custom_weights['total_return_weight'] +
            norm_avg_trade_returns * custom_weights['avg_trade_return_weight'] +
            norm_win_rates * custom_weights['win_rate_weight'] +
            norm_trade_counts * custom_weights['trade_count_weight'] +
            norm_drawdowns_inv * custom_weights['drawdown_weight']
        )
        best_idx = np.argmax(balanced_scores)
        
    else:
        # 其他選擇方法 (例如: 'return', 'average_trade_return')
        if selection_method == 'return':
            best_idx = np.argmax([m['total_return'] for m in all_metrics])
        elif selection_method == 'average_trade_return':
            best_idx = np.argmax([m['average_trade_return'] for m in all_metrics])
        else: # 預設 fallback
            best_idx = np.argmax([m['total_return'] for m in all_metrics])

    return pareto_genes[best_idx], all_metrics[best_idx]

# ═══════════════════════════════════════════════════════════════
# 🎨 策略描述函數 (系統B)
# ═══════════════════════════════════════════════════════════════

def format_gene_parameters_to_text_b(gene):
    """將系統B基因轉換為詳細的中文策略描述 - 只顯示實際使用的參數"""
    try:
        if not gene or len(gene) != 10:
            return "系統B基因格式錯誤"
        
        # 🧬 解析10基因參數
        rsi_buy_entry = gene[0]
        rsi_exit = gene[1]
        vix_threshold = gene[2]
        low_vol_exit = gene[3]
        rsi_period_choice = gene[4]
        vix_ma_choice = gene[5]
        bb_length_choice = gene[6]
        bb_std_choice = gene[7]
        adx_threshold = gene[8]
        high_vol_entry = gene[9]
        
        # 📊 獲取對應的參數值
        rsi_period = STRATEGY_CONFIG_B['rsi_period_options'][rsi_period_choice]
        vix_ma_period = STRATEGY_CONFIG_B['vix_ma_period_options'][vix_ma_choice]
        bb_length = STRATEGY_CONFIG_B['bb_length_options'][bb_length_choice]
        bb_std = STRATEGY_CONFIG_B['bb_std_options'][bb_std_choice]
        
        # 🎯 分析實際使用的策略和指標
        used_indicators = set(['vix_ma', 'rsi', 'bb'])  # 基本必用指標
        strategy_details = []
        
        # 高波動市場策略分析
        if high_vol_entry == 0:
            high_vol_strategy_name = "布林帶+RSI反轉策略"
            high_vol_strategy_desc = f"價格觸及布林帶({bb_length}日, {bb_std}倍標準差)下軌 且 RSI({rsi_period}日)<{rsi_buy_entry} 時買入"
            high_vol_exit_desc = f"RSI升至{rsi_exit}以上後回落時賣出"
            used_indicators.add('rsi')
        else:
            high_vol_strategy_name = "布林帶+ADX趨勢策略"
            high_vol_strategy_desc = f"價格觸及布林帶({bb_length}日, {bb_std}倍標準差)下軌 且 ADX(14日)>{adx_threshold} 確認趨勢強度時買入"
            high_vol_exit_desc = f"價格回升至布林帶中軌以上時賣出"
            used_indicators.add('adx')
        
        # 低波動市場策略分析
        low_vol_strategy_name = "均線交叉策略"
        low_vol_strategy_desc = "短期5日均線上穿長期10日均線時買入"
        
        if low_vol_exit == 0:
            low_vol_exit_desc = "價格跌破短期5日均線時賣出"
        else:
            low_vol_exit_desc = "短期5日均線下穿長期10日均線時賣出"
        
        used_indicators.add('ma')
        
        # 📋 組裝策略執行邏輯
        strategy_details.append(f"【高波動市場策略】{high_vol_strategy_name}")
        strategy_details.append(f"  進場條件: {high_vol_strategy_desc}")
        strategy_details.append(f"  出場條件: {high_vol_exit_desc}")
        strategy_details.append(f"【低波動市場策略】{low_vol_strategy_name}")
        strategy_details.append(f"  進場條件: {low_vol_strategy_desc}")
        strategy_details.append(f"  出場條件: {low_vol_exit_desc}")
        
        # 🌟 組裝最終描述
        description = f"""

🎯 【核心策略】
   市場狀態判斷: VIX {vix_ma_period}日移動平均 ≥ {vix_threshold} 為高波動市場
   高波動策略: {high_vol_strategy_name}
   低波動策略: {low_vol_strategy_name}

🚦 【交易信號邏輯】"""

        for detail in strategy_details:
            if detail.startswith("【"):
                description += f"\n   ► {detail}"
            else:
                description += f"\n     {detail}"

        description += f"\n\n⚙️  【技術指標配置】（僅顯示策略實際使用項目）"
        
        # 只顯示實際使用的指標
        if 'vix_ma' in used_indicators:
            description += f"\n   • VIX波動率指標: {vix_ma_period}日移動平均（市場狀態判斷）"
            
        if 'rsi' in used_indicators:
            description += f"\n   • RSI相對強弱指標: {rsi_period}日週期（買入<{rsi_buy_entry}, 賣出>{rsi_exit}）"
            
        if 'bb' in used_indicators:
            description += f"\n   • 布林帶指標: {bb_length}日週期, {bb_std}倍標準差（支撐阻力判斷）"
            
        if 'adx' in used_indicators:
            description += f"\n   • ADX趨勢強度: 固定14日週期（趨勢確認>{adx_threshold}）"
            
        if 'ma' in used_indicators:
            description += f"\n   • 移動平均線: 短期5日, 長期10日（趨勢跟隨）"

        description += f"\n\n📈 【策略執行流程】"
        description += f"\n   1️⃣  每日開盤前計算VIX {vix_ma_period}日移動平均值"
        description += f"\n   2️⃣  判斷市場狀態：VIX≥{vix_threshold}為高波動，<{vix_threshold}為低波動"
        description += f"\n   3️⃣  根據市場狀態選擇對應策略執行交易信號"
        description += f"\n   4️⃣  嚴格按照進出場條件執行，避免情緒化交易"


        description += f"\n\n🔍 【適用市場環境】"
        if high_vol_entry == 0:
            description += f"\n   📊 高波動期間: 擅長捕捉超跌反彈機會"
        else:
            description += f"\n   📊 高波動期間: 擅長捕捉趨勢突破機會"
        description += f"\n   📊 低波動期間: 穩健跟隨中長期趨勢"
        description += f"\n   📊 震盪市場: 利用布林帶邊界進行區間交易"
        description += f"\n   📊 趨勢市場: 均線系統確保趨勢跟隨能力"

        return description
        
    except Exception as e:
        return f"系統B策略參數解析錯誤：{str(e)}"


# ═══════════════════════════════════════════════════════════════
# 🔍 模組完整性檢查
# ═══════════════════════════════════════════════════════════════

def check_module_integrity_b():
    """檢查系統B模組完整性"""
    print("[GAEngine_B] === 系統B模組完整性檢查 ===")
    
    # 檢查核心函數
    required_functions = [
        'load_stock_data_b', 'precompute_indicators_b', 'genetic_algorithm_unified_b',
        'run_strategy_numba_core_b', 'run_strategy_b', 'format_gene_parameters_to_text_b'
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if func_name not in globals():
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"[GAEngine_B] ❌ 缺少函數: {missing_functions}")
        return False
    else:
        print("[GAEngine_B] ✅ 所有核心函數完整")
    
    # 檢查配置
    required_configs = ['GENE_MAP_B', 'STRATEGY_CONFIG_B', 'GA_PARAMS_CONFIG_B']
    missing_configs = []
    for config_name in required_configs:
        if config_name not in globals():
            missing_configs.append(config_name)
    
    if missing_configs:
        print(f"[GAEngine_B] ❌ 缺少配置: {missing_configs}")
        return False
    else:
        print("[GAEngine_B] ✅ 所有配置完整")
    
    print(f"[GAEngine_B] NSGA-II 支援: {'✅ 可用' if NSGA2_AVAILABLE else '❌ 不可用'}")
    
    # 檢查 Numba
    try:
        import numba
        print(f"[GAEngine_B] ✅ Numba: v{numba.__version__}")
    except ImportError:
        print("[GAEngine_B] ❌ Numba 未安裝")
        return False
    
    print("[GAEngine_B] === 檢查完成 ===")
    print(f"[GAEngine_B] 基因長度: 10 位 (RSI主導型)")
    print(f"[GAEngine_B] 策略類型: 波動率切換型多策略")
    print(f"[GAEngine_B] 模組版本: v2.1 (修復版)")
    
    return True

# 模組載入時自動檢查
if __name__ == "__main__":
    check_module_integrity_b()
else:
    print("[GAEngine_B] 🚀 ga_engine_b.py v2.1 模組已載入完成（修復版）")
    print("[GAEngine_B] 🎯 系統B功能：10基因RSI策略 + NSGA-II多目標優化 + 平均交易報酬率")