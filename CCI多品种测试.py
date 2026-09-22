"""
CCI(24,5) 多品种回测
==================
测试优化后的CCI参数在所有品种上的表现
"""

import pandas as pd
import numpy as np
import pickle

print("="*80)
print("CCI(24,5) 多品种回测".center(80))
print("="*80)

# 加载数据
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

# 计算CCI
def calculate_cci(df, length=24):
    data = df.copy()
    data['tp'] = (data['high'] + data['low'] + data['close']) / 3
    data['tp_ma'] = data['tp'].rolling(window=length).mean()
    data['tp_dev'] = data['tp'].rolling(window=length).std()
    data['cci'] = (data['tp'] - data['tp_ma']) / (0.015 * data['tp_dev'])
    data['cci'] = data['cci'].fillna(0)
    return data

# 回测函数
def backtest_cci(df, cci_length=24, ma_length=5):
    data = calculate_cci(df, length=cci_length)
    data['cci_ma'] = data['cci'].rolling(window=ma_length).mean()

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
        cci_val = cci[i]
        cci_ma_val = cci_ma[i]
        cci_prev = cci[i-1]
        cci_ma_prev = cci_ma[i-1]

        # 平仓逻辑
        if position > 0:
            if cci_prev >= cci_ma_prev and cci_val <= cci_ma_val:
                exit_p = opens[i+1]
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0

        # 开仓逻辑
        if position == 0:
            if cci_prev <= cci_ma_prev and cci_val > cci_ma_val:
                qty = int(balance * 2.0 / (opens[i+1] * 5))
                qty = max(1, qty)
                position = qty
                entry_price = opens[i+1]

        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * 5
        equity.append(val)

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

# ============ 测试所有品种 ============

symbols = {
    'cu': '铜',
    'al': '铝',
    'zn': '锌',
    'pb': '铅',
    'ni': '镍',
    'sn': '锡'
}

results = {}
results_train = {}

print(f"\n{'='*80}")
print(f"{'品种':<8} {'训练收益':<12} {'训练回撤':<12} {'训练交易':<10} {'训练胜率':<10} | "
      f"{'测试收益':<12} {'测试回撤':<12} {'测试交易':<10} {'测试胜率':<10}")
print("-"*80)

for symbol_key, symbol_name in symbols.items():
    if symbol_key not in DATA_POOL:
        continue

    df_full = DATA_POOL[symbol_key].copy()
    df_full.sort_index(inplace=True)

    # 分割训练集和测试集
    train_df = df_full.loc[:'2023-12-31'].copy()
    test_df = df_full.loc['2024-01-01':].copy()

    # 训练集回测
    train_result = backtest_cci(train_df, 24, 5)
    # 测试集回测
    test_result = backtest_cci(test_df, 24, 5)

    if train_result and test_result:
        results[symbol_key] = test_result
        results_train[symbol_key] = train_result

        print(f"{symbol_name:<8} "
              f"{train_result['return_pct']:>+10.2f}% {train_result['max_dd']:>10.2f}% "
              f"{train_result['total_trades']:>8} {train_result['win_rate']:>9.1f}% | "
              f"{test_result['return_pct']:>+10.2f}% {test_result['max_dd']:>10.2f}% "
              f"{test_result['total_trades']:>8} {test_result['win_rate']:>9.1f}%")

print("="*80)

# ============ 排序和分析 ============
print("\n" + "="*80)
print("测试集收益率排名".center(80))
print("="*80)

# 按测试集收益率排序
sorted_results = sorted(results.items(), key=lambda x: x[1]['return_pct'], reverse=True)

print(f"\n{'排名':<6} {'品种':<8} {'收益率':<12} {'最大回撤':<12} {'交易次数':<10} {'胜率':<10} {'盈亏比':<10}")
print("-"*80)

for rank, (symbol_key, result) in enumerate(sorted_results, 1):
    symbol_name = symbols[symbol_key]
    print(f"{rank:<6} {symbol_name:<8} {result['return_pct']:>+10.2f}% {result['max_dd']:>10.2f}% "
          f"{result['total_trades']:>8} {result['win_rate']:>9.1f}% {result['profit_factor']:>9.2f}")

# ============ 与默认参数对比 ============
print("\n" + "="*80)
print("优化参数 vs 默认参数对比 (测试集)".center(80))
print("="*80)

print(f"\n{'品种':<8} {'默认收益':<12} {'优化收益':<12} {'优化回撤':<12} {'差异':<10} {'评价':<10}")
print("-"*80)

for symbol_key, symbol_name in symbols.items():
    if symbol_key not in results:
        continue

    df_full = DATA_POOL[symbol_key].copy()
    df_full.sort_index(inplace=True)
    test_df = df_full.loc['2024-01-01':].copy()

    # 测试默认参数
    default_result = backtest_cci(test_df, 20, 14)
    optimized_result = results[symbol_key]

    if default_result:
        diff = optimized_result['return_pct'] - default_result['return_pct']

        # 评价
        if diff > 20:
            evaluation = "优秀"
        elif diff > 10:
            evaluation = "良好"
        elif diff > 0:
            evaluation = "提升"
        elif diff > -10:
            evaluation = "接近"
        else:
            evaluation = "较差"

        print(f"{symbol_name:<8} {default_result['return_pct']:>+10.2f}% "
              f"{optimized_result['return_pct']:>+10.2f}% {optimized_result['max_dd']:>10.2f}% "
              f"{diff:>+8.2f}% {evaluation:<10}")

# ============ 统计分析 ============
print("\n" + "="*80)
print("统计分析".center(80))
print("="*80)

test_returns = [r['return_pct'] for r in results.values()]
test_dds = [r['max_dd'] for r in results.values()]
test_win_rates = [r['win_rate'] for r in results.values()]

print(f"\n测试集表现 ({len(results)}个品种):")
print(f"  平均收益率: {np.mean(test_returns):+.2f}%")
print(f"  收益率标准差: {np.std(test_returns):.2f}%")
print(f"  最高收益率: {max(test_returns):+.2f}%")
print(f"  最低收益率: {min(test_returns):+.2f}%")
print(f"  胜率 > 50%的品种: {len([r for r in results.values() if r['win_rate'] > 50])}个")
print(f"  收益率 > 50%的品种: {len([r for r in results.values() if r['return_pct'] > 50])}个")
print(f"  收益率 > 0%的品种: {len([r for r in results.values() if r['return_pct'] > 0])}个")

print(f"\n平均回撤: {np.mean(test_dds):.2f}%")
print(f"  最小回撤: {min(test_dds):.2f}%")
print(f"  最大回撤: {max(test_dds):.2f}%")

print(f"\n平均胜率: {np.mean(test_win_rates):.1f}%")
print(f"  最高胜率: {max(test_win_rates):.1f}%")
print(f"  最低胜率: {min(test_win_rates):.1f}%")

# ============ 多品种组合 ============
print("\n" + "="*80)
print("等权重多品种组合收益".center(80))
print("="*80)

print(f"\n如果同时交易所有6个品种，等权重分配资金:")
print(f"  每个品种分配: 100000 / 6 = 16667元")
print(f"  总资金: 100000元")
print(f"  总收益: {np.mean(test_returns):+.2f}%")
print(f"  最差情况: {min(test_returns):+.2f}% (如果只选了最差的品种)")
print(f"  最好情况: {max(test_returns):+.2f}% (如果只选了最好的品种)")

print("\n" + "="*80)
print("测试完成！")
print("="*80)
