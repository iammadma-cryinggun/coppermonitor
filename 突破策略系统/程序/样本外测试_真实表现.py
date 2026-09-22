"""
LME铜突破策略 - 样本外测试（Out-of-Sample Testing）
======================================================
这是量化交易的"金标准"——用从未见过的数据验证策略

测试方案：
- 训练集：2016-2021年（用来优化参数）
- 测试集：2024-2026年（完全没见过，验证真实表现）

只有测试集的结果才是真实的！
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# ============ 加载数据 ============
print("加载数据...")
with open('../原始数据/data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df = DATA_POOL['cu'].copy()
df.sort_index(inplace=True)

# ============ 加载优化参数（训练集得出） ============
with open('../优化结果/突破策略_进阶版_最优参数.json', 'r', encoding='utf-8') as f:
    opt_result = json.load(f)
    params = opt_result['params']

print(f"\n使用训练集优化出的参数:")
print(f"  唐奇安周期: {params['donchian_period']}日")
print(f"  止损周期: {params['stop_period']}日")
print(f"  EMA周期: {params['ema_period']}日")
print(f"  ADX阈值: {params['adx_threshold']}")
print(f"  ATR倍数: {params['atr_multiplier']}")

# ============ 数据分割 ============
# 训练集：2016-2021年
train_start = '2016-01-01'
train_end = '2021-12-31'

# 测试集：2024-2026年（样本外！）
test_start = '2024-01-01'
test_end = '2026-12-31'

df_train = df.loc[train_start:train_end].copy()
df_test = df.loc[test_start:test_end].copy()

print(f"\n数据分割:")
print(f"  训练集: {train_start} ~ {train_end} ({len(df_train)}根K线)")
print(f"  测试集: {test_start} ~ {test_end} ({len(df_test)}根K线) ← 样本外！")

# ============ 固定参数 ============
INITIAL_CAPITAL = 100000
CONTRACT_SIZE = 5
MAX_LEVERAGE = 2.0

# ============ 计算指标 ============
def calculate_indicators(df, params):
    """计算所有技术指标"""
    df = df.copy()

    # 唐奇安通道
    df['donchian_high'] = df['high'].rolling(window=params['donchian_period']).max()
    df['donchian_low'] = df['low'].rolling(window=params['donchian_period']).min()
    df['donchian_middle'] = (df['donchian_high'] + df['donchian_low']) / 2

    # EMA
    df['ema_filter'] = df['close'].ewm(span=params['ema_period'], adjust=False).mean()

    # ADX
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)

    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']

    df['pdm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['ndm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    adx_window = 14
    df['tr_s'] = df['tr'].rolling(adx_window).mean()
    df['pdm_s'] = df['pdm'].rolling(adx_window).mean()
    df['ndm_s'] = df['ndm'].rolling(adx_window).mean()

    df['pdi'] = 100 * (df['pdm_s'] / df['tr_s'])
    df['ndi'] = 100 * (df['ndm_s'] / df['tr_s'])
    df['dx'] = 100 * abs(df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi'])
    df['adx'] = df['dx'].rolling(adx_window).mean()

    # ATR
    df['atr'] = df['tr'].rolling(20).mean()

    # 移动止损线
    df['stop_low'] = df['low'].rolling(window=params['stop_period']).min()

    return df

# ============ 回测核心 ============
def backtest(df, params, dataset_name="数据集"):
    """回测函数"""
    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    peak_capital = capital
    position = None

    trades = []
    capital_history = []

    min_idx = max(params['donchian_period'], params['stop_period'], params['ema_period'], 50) + 1

    for i in range(min_idx, len(df) - 1):
        prev = df.iloc[i - 1]
        current = df.iloc[i]
        next_bar = df.iloc[i + 1]

        if pd.isna(current['adx']) or pd.isna(current['ema_filter']) or pd.isna(current['atr']):
            continue

        # 持仓管理
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if current['low'] <= current['stop_low']:
                exit_price = min(current['stop_low'], current['open'])
                exit_triggered = True
                exit_reason = 'stop_loss'

            elif current['close'] < current['donchian_middle']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'trend_reversal'

            elif current['close'] < current['ema_filter']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'ema_break'

            if exit_triggered:
                pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                capital += pnl
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': current.name,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': pnl / (capital - pnl) * 100 if capital > pnl else 0,
                    'exit_reason': exit_reason
                })
                position = None

        # 开仓逻辑
        if position is None:
            breakout = current['close'] > prev['donchian_high']
            adx_ok = current['adx'] > params['adx_threshold']
            ema_ok = current['close'] > current['ema_filter']

            if breakout and adx_ok and ema_ok:
                entry_price = next_bar['open']
                contract_value = entry_price * CONTRACT_SIZE

                contracts = int((capital * 1.5) / contract_value)
                if contracts < 1:
                    contracts = 1

                total_notional = contracts * contract_value
                actual_leverage = total_notional / capital
                if actual_leverage > MAX_LEVERAGE:
                    contracts = int((capital * MAX_LEVERAGE) / contract_value)
                    if contracts < 1:
                        continue
                    total_notional = contracts * contract_value
                    actual_leverage = total_notional / capital

                position = {
                    'entry_date': next_bar.name,
                    'entry_price': entry_price,
                    'contracts': contracts,
                    'leverage': actual_leverage
                }

        capital_history.append(capital)
        if capital > peak_capital:
            peak_capital = capital

    # 统计
    if not trades:
        return {
            'dataset': dataset_name,
            'total_trades': 0,
            'return_pct': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'trades': []
        }

    trades_df = pd.DataFrame(trades)

    capital_series = pd.Series(capital_history)
    peak = capital_series.cummax()
    drawdown = (capital_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    return_pct = (capital_history[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100

    return {
        'dataset': dataset_name,
        'total_trades': len(trades_df),
        'return_pct': return_pct,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trades': trades_df
    }

# ============ 运行测试 ============
print("\n" + "="*80)
print("样本外测试 - 真实表现验证".center(80))
print("="*80)

# 1. 训练集回测（应该是不错的，因为参数是基于这个优化的）
print("\n【第1步】训练集回测（2016-2021）")
print("-"*80)
result_train = backtest(df_train, params, "训练集")
print(f"总交易:    {result_train['total_trades']}笔")
print(f"总收益率:  {result_train['return_pct']:+.2f}%")
print(f"最大回撤:  {result_train['max_drawdown']:.2f}%")
print(f"胜率:      {result_train['win_rate']:.1f}%")

# 2. 测试集回测（这才是真实表现！）
print("\n【第2步】测试集回测（2024-2026）← 样本外！")
print("-"*80)
result_test = backtest(df_test, params, "测试集")
print(f"总交易:    {result_test['total_trades']}笔")
print(f"总收益率:  {result_test['return_pct']:+.2f}%")
print(f"最大回撤:  {result_test['max_drawdown']:.2f}%")
print(f"胜率:      {result_test['win_rate']:.1f}%")

# ============ 对比分析 ============
print("\n" + "="*80)
print("样本内 vs 样本外 对比".center(80))
print("="*80)

comparison = pd.DataFrame({
    '指标': ['总收益率', '最大回撤', '胜率', '交易次数'],
    '训练集（样本内）': [
        f"{result_train['return_pct']:+.2f}%",
        f"{result_train['max_drawdown']:.2f}%",
        f"{result_train['win_rate']:.1f}%",
        result_train['total_trades']
    ],
    '测试集（样本外）': [
        f"{result_test['return_pct']:+.2f}%",
        f"{result_test['max_drawdown']:.2f}%",
        f"{result_test['win_rate']:.1f}%",
        result_test['total_trades']
    ]
})

print(comparison.to_string(index=False))

# ============ 判断是否过拟合 ============
print("\n" + "="*80)
print("过拟合检测".center(80))
print("="*80)

performance_drop = result_train['return_pct'] - result_test['return_pct']
drawdown_increase = abs(result_test['max_drawdown']) - abs(result_train['max_drawdown'])

if performance_drop > 20:
    print(f"⚠️  警告：收益率大幅下降 {performance_drop:.1f}%")
    print(f"   可能存在过拟合！")
elif performance_drop > 10:
    print(f"⚡ 注意：收益率下降 {performance_drop:.1f}%")
    print(f"   轻微过拟合，可接受")
else:
    print(f"✅ 收益率表现稳定")

if drawdown_increase > 10:
    print(f"⚠️  警告：回撤增加 {drawdown_increase:.1f}%")
    print(f"   样本外风险更高！")
elif drawdown_increase > 5:
    print(f"⚡ 注意：回撤增加 {drawdown_increase:.1f}%")
else:
    print(f"✅ 回撤控制良好")

# ============ 测试集交易明细 ============
if result_test['total_trades'] > 0:
    print("\n" + "="*80)
    print("测试集交易明细（样本外真实交易）".center(80))
    print("="*80)

    trades_test = result_test['trades'][[
        'entry_date', 'exit_date', 'entry_price', 'exit_price',
        'contracts', 'pnl', 'pnl_pct', 'exit_reason'
    ]]

    trades_test.columns = [
        '入场日期', '出场日期', '入场价', '出场价',
        '手数', '盈亏', '盈亏%', '出场原因'
    ]

    print(trades_test.to_string(index=False))

print("\n" + "="*80)
print("结论：只有测试集的结果才是实盘的真实预期！".center(80))
print("="*80)
