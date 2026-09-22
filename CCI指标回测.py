"""
CCI指标回测
============
测试CCI指标在铜和锡期货日线数据上的表现
策略1: 超买超卖区间 (+100/-100)
策略2: CCI均线金叉死叉
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

print("="*80)
print("CCI指标回测 - 铜 vs 锡".center(80))
print("="*80)

# 加载数据
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

# CCI计算函数
def calculate_cci(df, length=20):
    """计算CCI指标"""
    data = df.copy()
    data['tp'] = (data['high'] + data['low'] + data['close']) / 3  # Typical Price
    data['tp_ma'] = data['tp'].rolling(window=length).mean()
    data['tp_dev'] = data['tp'].rolling(window=length).std()

    # CCI = (TP - MA) / (0.015 * Deviation)
    data['cci'] = (data['tp'] - data['tp_ma']) / (0.015 * data['tp_dev'])
    data['cci'] = data['cci'].fillna(0)

    return data

# CCI均线
def calculate_cci_ma(df, cci_length=14, ma_type='SMA'):
    """计算CCI的移动平均线"""
    data = df.copy()

    if ma_type == 'SMA':
        data['cci_ma'] = data['cci'].rolling(window=cci_length).mean()
    elif ma_type == 'EMA':
        data['cci_ma'] = data['cci'].ewm(span=cci_length, adjust=False).mean()

    data['cci_ma'] = data['cci_ma'].fillna(0)
    return data

# ============ 策略1: 超买超卖区间 ============
def backtest_cci_overbought(df, cci_length=20, buy_threshold=-100, sell_threshold=100):
    """
    策略1: CCI超买超卖区间
    - CCI < -100: 超卖，买入
    - CCI > +100: 超买，卖出
    """
    data = calculate_cci(df, length=cci_length)

    balance = 100000
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    closes = data['close'].values
    opens = data['open'].values
    cci = data['cci'].values

    for i in range(cci_length + 1, len(data)-1):
        price = closes[i]
        next_open = opens[i+1]
        cci_val = cci[i]

        # 平仓逻辑
        if position > 0:
            if cci_val > sell_threshold:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0

        # 开仓逻辑
        if position == 0:
            if cci_val < buy_threshold:
                # 固定仓位：2倍杠杆
                qty = int(balance * 2.0 / (next_open * 5))
                qty = max(1, qty)
                position = qty
                entry_price = next_open

        # 计算当前权益
        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * 5
        equity.append(val)

    # 计算指标
    if not trades or len(trades) < 5:
        return None

    ret = (equity[-1] - 100000) / 100000 * 100
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100

    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    max_dd = dd.min()

    winning = [t for t in trades if t > 0]
    losing = [t for t in trades if t <= 0]
    profit_factor = abs(sum(winning) / sum(losing)) if losing else 0

    return {
        'return_pct': ret,
        'max_dd': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'final_balance': equity[-1]
    }

# ============ 策略2: CCI均线金叉死叉 ============
def backtest_cci_cross(df, cci_length=20, ma_length=14, ma_type='SMA'):
    """
    策略2: CCI均线金叉死叉
    - CCI上穿CCI_MA: 金叉，买入
    - CCI下穿CCI_MA: 死叉，卖出
    """
    data = calculate_cci(df, length=cci_length)
    data = calculate_cci_ma(data, cci_length=ma_length, ma_type=ma_type)

    balance = 100000
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    closes = data['close'].values
    opens = data['open'].values
    cci = data['cci'].values
    cci_ma = data['cci_ma'].values

    start_idx = cci_length + ma_length + 1

    for i in range(start_idx, len(data)-1):
        price = closes[i]
        next_open = opens[i+1]
        cci_val = cci[i]
        cci_ma_val = cci_ma[i]
        cci_prev = cci[i-1]
        cci_ma_prev = cci_ma[i-1]

        # 平仓逻辑
        if position > 0:
            # 死叉：CCI下穿CCI_MA
            if cci_prev > cci_ma_prev and cci_val <= cci_ma_val:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0

        # 开仓逻辑
        if position == 0:
            # 金叉：CCI上穿CCI_MA
            if cci_prev <= cci_ma_prev and cci_val > cci_ma_val:
                qty = int(balance * 2.0 / (next_open * 5))
                qty = max(1, qty)
                position = qty
                entry_price = next_open

        # 计算当前权益
        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * 5
        equity.append(val)

    # 计算指标
    if not trades or len(trades) < 5:
        return None

    ret = (equity[-1] - 100000) / 100000 * 100
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100

    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    max_dd = dd.min()

    winning = [t for t in trades if t > 0]
    losing = [t for t in trades if t <= 0]
    profit_factor = abs(sum(winning) / sum(losing)) if losing else 0

    return {
        'return_pct': ret,
        'max_dd': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'final_balance': equity[-1]
    }

# ============ 主测试流程 ============

symbols = {
    'cu': '铜',
    'sn': '锡'
}

results = {}

for symbol_key, symbol_name in symbols.items():
    print(f"\n{'='*80}")
    print(f"{symbol_name}期货 ({symbol_key.upper()}) - CCI指标回测".center(80))
    print(f"{'='*80}")

    df_full = DATA_POOL[symbol_key].copy()
    df_full.sort_index(inplace=True)

    # 分割训练集和测试集
    train_df = df_full.loc[:'2023-12-31'].copy()
    test_df = df_full.loc['2024-01-01':].copy()

    print(f"\n数据分割:")
    print(f"  训练集: {len(train_df)}根K线 ({train_df.index[0].strftime('%Y-%m-%d')} 至 {train_df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  测试集: {len(test_df)}根K线 ({test_df.index[0].strftime('%Y-%m-%d')} 至 {test_df.index[-1].strftime('%Y-%m-%d')})")

    results[symbol_key] = {}

    # ============ 策略1: 超买超卖区间 ============
    print(f"\n{'-'*80}")
    print("策略1: CCI超买超卖区间 (+100/-100)".center(80))
    print(f"{'-'*80}")

    train_result1 = backtest_cci_overbought(train_df, cci_length=20, buy_threshold=-100, sell_threshold=100)
    test_result1 = backtest_cci_overbought(test_df, cci_length=20, buy_threshold=-100, sell_threshold=100)

    if train_result1 and test_result1:
        print(f"\n{'数据集':<10} {'收益率':<12} {'最大回撤':<12} {'交易次数':<10} {'胜率':<10} {'盈亏比':<10}")
        print("-"*80)
        print(f"{'训练集':<10} {train_result1['return_pct']:>+10.2f}% {train_result1['max_dd']:>10.2f}% "
              f"{train_result1['total_trades']:>8} {train_result1['win_rate']:>9.1f}% {train_result1['profit_factor']:>9.2f}")
        print(f"{'测试集':<10} {test_result1['return_pct']:>+10.2f}% {test_result1['max_dd']:>10.2f}% "
              f"{test_result1['total_trades']:>8} {test_result1['win_rate']:>9.1f}% {test_result1['profit_factor']:>9.2f}")

        results[symbol_key]['overbought'] = {
            'train': train_result1,
            'test': test_result1
        }

    # ============ 策略2: CCI均线金叉死叉 ============
    print(f"\n{'-'*80}")
    print("策略2: CCI均线金叉死叉 (SMA 14)".center(80))
    print(f"{'-'*80}")

    train_result2 = backtest_cci_cross(train_df, cci_length=20, ma_length=14, ma_type='SMA')
    test_result2 = backtest_cci_cross(test_df, cci_length=20, ma_length=14, ma_type='SMA')

    if train_result2 and test_result2:
        print(f"\n{'数据集':<10} {'收益率':<12} {'最大回撤':<12} {'交易次数':<10} {'胜率':<10} {'盈亏比':<10}")
        print("-"*80)
        print(f"{'训练集':<10} {train_result2['return_pct']:>+10.2f}% {train_result2['max_dd']:>10.2f}% "
              f"{train_result2['total_trades']:>8} {train_result2['win_rate']:>9.1f}% {train_result2['profit_factor']:>9.2f}")
        print(f"{'测试集':<10} {test_result2['return_pct']:>+10.2f}% {test_result2['max_dd']:>10.2f}% "
              f"{test_result2['total_trades']:>8} {test_result2['win_rate']:>9.1f}% {test_result2['profit_factor']:>9.2f}")

        results[symbol_key]['cross'] = {
            'train': train_result2,
            'test': test_result2
        }

    # ============ 策略2b: CCI EMA金叉死叉 ============
    print(f"\n{'-'*80}")
    print("策略2b: CCI均线金叉死叉 (EMA 14)".center(80))
    print(f"{'-'*80}")

    train_result2b = backtest_cci_cross(train_df, cci_length=20, ma_length=14, ma_type='EMA')
    test_result2b = backtest_cci_cross(test_df, cci_length=20, ma_length=14, ma_type='EMA')

    if train_result2b and test_result2b:
        print(f"\n{'数据集':<10} {'收益率':<12} {'最大回撤':<12} {'交易次数':<10} {'胜率':<10} {'盈亏比':<10}")
        print("-"*80)
        print(f"{'训练集':<10} {train_result2b['return_pct']:>+10.2f}% {train_result2b['max_dd']:>10.2f}% "
              f"{train_result2b['total_trades']:>8} {train_result2b['win_rate']:>9.1f}% {train_result2b['profit_factor']:>9.2f}")
        print(f"{'测试集':<10} {test_result2b['return_pct']:>+10.2f}% {test_result2b['max_dd']:>10.2f}% "
              f"{test_result2b['total_trades']:>8} {test_result2b['win_rate']:>9.1f}% {test_result2b['profit_factor']:>9.2f}")

        results[symbol_key]['cross_ema'] = {
            'train': train_result2b,
            'test': test_result2b
        }

# ============ 总结对比 ============
print(f"\n{'='*80}")
print("总结对比 - 测试集表现".center(80))
print(f"{'='*80}")

print(f"\n{'品种':<8} {'策略':<25} {'收益率':<12} {'最大回撤':<12} {'交易次数':<10} {'胜率':<10}")
print("-"*80)

for symbol_key, symbol_name in symbols.items():
    if 'overbought' in results[symbol_key]:
        r = results[symbol_key]['overbought']['test']
        print(f"{symbol_name:<8} {'CCI超买超卖':<25} {r['return_pct']:>+10.2f}% {r['max_dd']:>10.2f}% "
              f"{r['total_trades']:>8} {r['win_rate']:>9.1f}%")

    if 'cross' in results[symbol_key]:
        r = results[symbol_key]['cross']['test']
        print(f"{symbol_name:<8} {'CCI SMA金叉死叉':<25} {r['return_pct']:>+10.2f}% {r['max_dd']:>10.2f}% "
              f"{r['total_trades']:>8} {r['win_rate']:>9.1f}%")

    if 'cross_ema' in results[symbol_key]:
        r = results[symbol_key]['cross_ema']['test']
        print(f"{symbol_name:<8} {'CCI EMA金叉死叉':<25} {r['return_pct']:>+10.2f}% {r['max_dd']:>10.2f}% "
              f"{r['total_trades']:>8} {r['win_rate']:>9.1f}%")

print(f"\n{'='*80}")
print("回测完成！")
print(f"{'='*80}")
