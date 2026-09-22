"""
LME铜EMA+MACD策略 - 样本外测试
================================
验证50次迭代的最优参数在未见数据上的表现

测试方案：
- 训练集：2016-2021年（优化参数）
- 测试集：2024-2026年（样本外验证）
"""

import pandas as pd
import numpy as np
import pickle
import json

# ============ 加载数据 ============
print("正在加载铜数据...")
with open('../原始数据/data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df = DATA_POOL['cu'].copy()
df.sort_index(inplace=True)

# ============ 加载50次迭代的最优参数 ============
with open('../优化结果/EMA_MACD_用户版本_只做多.json', 'r', encoding='utf-8') as f:
    opt_result = json.load(f)
    params = [
        opt_result['params']['ema_fast'],
        opt_result['params']['ema_slow'],
        opt_result['params']['macd_fast'],
        opt_result['params']['macd_slow'],
        opt_result['params']['macd_signal'],
        opt_result['params']['atr_multiplier']
    ]

print(f"\n使用50次迭代的最优参数:")
print(f"  EMA: {params[0]}/{params[1]}日")
print(f"  MACD: ({params[2]}, {params[3]}, {params[4]})")
print(f"  ATR止损: {params[5]:.2f}倍")

# ============ 固定参数 ============
INITIAL_CAPITAL = 100000
CONTRACT_SIZE = 5
MAX_LEVERAGE = 2.0

# ============ 数据分割 ============
train_start = '2016-01-01'
train_end = '2021-12-31'
test_start = '2024-01-01'
test_end = '2026-12-31'

df_train = df.loc[train_start:train_end].copy()
df_test = df.loc[test_start:test_end].copy()

print(f"\n数据分割:")
print(f"  训练集: {train_start} ~ {train_end} ({len(df_train)}根K线)")
print(f"  测试集: {test_start} ~ {test_end} ({len(df_test)}根K线) ← 样本外！")

# ============ 计算指标 ============
def calculate_indicators(data, p):
    data = data.copy()
    data['ema_fast'] = data['close'].ewm(span=int(p[0]), adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=int(p[1]), adjust=False).mean()

    ema_m_fast = data['close'].ewm(span=int(p[2]), adjust=False).mean()
    ema_m_slow = data['close'].ewm(span=int(p[3]), adjust=False).mean()
    data['diff'] = ema_m_fast - ema_m_slow
    data['dea'] = data['diff'].ewm(span=int(p[4]), adjust=False).mean()

    data['h-l'] = data['high'] - data['low']
    data['h-pc'] = abs(data['high'] - data['close'].shift(1))
    data['l-pc'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    data['atr'] = data['tr'].rolling(20).mean()

    return data

# ============ 回测函数 ============
def backtest(data, params, dataset_name="数据集"):
    data = calculate_indicators(data, params)

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades_list = []
    equity = []

    emas_f = data['ema_fast'].values
    emas_s = data['ema_slow'].values
    diffs = data['diff'].values
    atrs = data['atr'].values
    closes = data['close'].values
    opens = data['open'].values

    min_idx = 50
    for i in range(min_idx, len(data)-1):
        cross_up = (emas_f[i-1] < emas_s[i-1]) and (emas_f[i] > emas_s[i])
        cross_down = (emas_f[i-1] > emas_s[i-1]) and (emas_f[i] < emas_s[i])
        macd_condition = diffs[i] > 0

        next_open = opens[i+1]

        if position > 0:
            stop_price = entry_price - (atrs[i] * params[5])
            if next_open < stop_price or cross_down:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * CONTRACT_SIZE
                balance += pnl
                trades_list.append({
                    'pnl': pnl,
                    'entry_date': data.index[i+1],
                    'exit_date': data.index[i+1]
                })
                position = 0

        if position == 0:
            if cross_up and macd_condition:
                stop_dist = atrs[i] * params[5]
                if stop_dist > 0:
                    qty = max(1, min(int(balance * 0.02 / (stop_dist * CONTRACT_SIZE)),
                                   int(balance * MAX_LEVERAGE / (next_open * CONTRACT_SIZE))))
                    position = qty
                    entry_price = next_open

        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * CONTRACT_SIZE
        equity.append(val)

    # 统计
    final_ret = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = len([t['pnl'] for t in trades_list if t['pnl'] > 0]) / len(trades_list) * 100 if trades_list else 0

    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_dd = drawdown.min()

    return {
        'dataset': dataset_name,
        'return_pct': final_ret,
        'max_drawdown': max_dd,
        'total_trades': len(trades_list),
        'win_rate': win_rate,
        'trades': trades_list
    }

# ============ 运行测试 ============
print("\n" + "="*80)
print("EMA+MACD策略 - 样本外测试".center(80))
print("="*80)

# 训练集回测
print("\n【训练集回测】2016-2021")
print("-"*80)
result_train = backtest(df_train, params, "训练集")
print(f"总交易:    {result_train['total_trades']}笔")
print(f"总收益率:  {result_train['return_pct']:+.2f}%")
print(f"最大回撤:  {result_train['max_drawdown']:.2f}%")
print(f"胜率:      {result_train['win_rate']:.1f}%")

# 测试集回测（样本外）
print("\n【测试集回测】2024-2026（样本外）")
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
drawdown_change = abs(result_test['max_drawdown']) - abs(result_train['max_drawdown'])

if performance_drop > 20:
    print(f"⚠️  警告：收益率大幅下降 {performance_drop:.1f}%")
    print(f"   策略可能存在过拟合！")
elif performance_drop > 10:
    print(f"⚡ 注意：收益率下降 {performance_drop:.1f}%")
    print(f"   轻微过拟合，需谨慎")
else:
    print(f"✅ 收益率表现稳定")

if result_test['return_pct'] < 0:
    print(f"❌ 严重警告：样本外亏损 {result_test['return_pct']:.1f}%")
    print(f"   策略在样本外完全失效！")

if drawdown_change > 10:
    print(f"⚠️  警告：回撤增加 {drawdown_change:.1f}%")

# ============ 测试集交易明细 ============
if result_test['total_trades'] > 0:
    print("\n" + "="*80)
    print("测试集交易明细（样本外真实交易）".center(80))
    print("="*80)

    for i, trade in enumerate(result_test['trades']):
        print(f"{i+1}. {trade['entry_date'].strftime('%Y-%m-%d')} 入场 | "
              f"盈亏: {trade['pnl']:+,.0f} ({trade['pnl']/INITIAL_CAPITAL*100:+.2f}%)")

print("\n" + "="*80)
print("结论：样本外表现是实盘的真实预期".center(80))
print("="*80)
