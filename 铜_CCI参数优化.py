"""
铜期货 - CCI参数优化
==================
专门针对铜进行CCI参数网格搜索
"""

import pandas as pd
import numpy as np
import pickle
import itertools

print("="*80)
print("铜期货 - CCI参数优化".center(80))
print("="*80)

# 加载数据
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

# 计算CCI
def calculate_cci(df, length=20):
    data = df.copy()
    data['tp'] = (data['high'] + data['low'] + data['close']) / 3
    data['tp_ma'] = data['tp'].rolling(window=length).mean()
    data['tp_dev'] = data['tp'].rolling(window=length).std()
    data['cci'] = (data['tp'] - data['tp_ma']) / (0.015 * data['tp_dev'])
    data['cci'] = data['cci'].fillna(0)
    return data

# 回测函数
def backtest_cci(df, cci_length, ma_length):
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
        'final_balance': equity[-1],
        'cci_length': cci_length,
        'ma_length': ma_length
    }

# ============ 主测试流程 ============

symbol_key = 'cu'
symbol_name = '铜'

df_full = DATA_POOL[symbol_key].copy()
df_full.sort_index(inplace=True)

train_df = df_full.loc[:'2023-12-31'].copy()
test_df = df_full.loc['2024-01-01':].copy()

print(f"\n{symbol_name}期货数据:")
print(f"  训练集: {len(train_df)}根K线 (2016-2023)")
print(f"  测试集: {len(test_df)}根K线 (2024-2026)")

# ============ 定义参数网格 ============
# 基于锡的最优参数是(24,5)，铜可能在附近
cci_length_grid = list(range(14, 36, 2))  # [14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
ma_length_grid = list(range(3, 16, 1))    # [3, 4, 5, 6, 7, ..., 15]

print(f"\n参数网格:")
print(f"  CCI长度: {cci_length_grid} (共{len(cci_length_grid)}个)")
print(f"  MA长度:  {ma_length_grid} (共{len(ma_length_grid)}个)")
print(f"  总组合数: {len(cci_length_grid) * len(ma_length_grid)}")

# 生成所有参数组合
param_combinations = list(itertools.product(cci_length_grid, ma_length_grid))
print(f"  有效组合数: {len(param_combinations)}")

# ============ 运行网格搜索 ============
print("\n" + "="*80)
print("开始网格搜索...".center(80))
print("="*80)

import time
start_time = time.time()

results = []
for i, (cci_len, ma_len) in enumerate(param_combinations):
    result = backtest_cci(train_df, cci_len, ma_len)
    if result:
        results.append(result)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"已测试: {i+1}/{len(param_combinations)} ({(i+1)/len(param_combinations)*100:.1f}%) | "
              f"耗时: {elapsed:.1f}秒 | 有效结果: {len(results)}", flush=True)

elapsed = time.time() - start_time
print(f"\n网格搜索完成! 总耗时: {elapsed:.1f}秒")
print(f"有效参数组合: {len(results)}")

# ============ 分析结果 ============
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('return_pct', ascending=False)

print("\n" + "="*80)
print("训练集Top 30参数组合".center(80))
print("="*80)

print(f"\n{'排名':<6} {'收益率':<10} {'胜率':<8} {'回撤':<10} {'交易':<8} {'盈亏比':<8} {'参数':<20}")
print("-"*80)

for idx, row in results_df.head(30).iterrows():
    rank = results_df.index.get_loc(idx) + 1
    param_str = f"CCI({row['cci_length']},MA{row['ma_length']})"
    print(f"{rank:<6} {row['return_pct']:>+8.2f}% {row['win_rate']:>7.1f}% {row['max_dd']:>8.2f}% "
          f"{row['total_trades']:>6} {row['profit_factor']:>7.2f} {param_str}")

# ============ 测试Top参数的样本外表现 ============
print("\n" + "="*80)
print("Top 15参数样本外验证".center(80))
print("="*80)

top_params = results_df.head(15)

out_of_sample_results = []
for idx, row in top_params.iterrows():
    cci_len = int(row['cci_length'])
    ma_len = int(row['ma_length'])

    try:
        test_result = backtest_cci(test_df, cci_len, ma_len)

        if test_result:
            test_result['train_return'] = row['return_pct']
            test_result['train_win_rate'] = row['win_rate']
            test_result['train_dd'] = row['max_dd']
            out_of_sample_results.append(test_result)

            print(f"\n排名 {results_df.index.get_loc(idx)+1}: CCI({cci_len},MA{ma_len})")
            print(f"  训练: {row['return_pct']:+.2f}% | 测试: {test_result['return_pct']:+.2f}% | "
                  f"差异: {test_result['return_pct']-row['return_pct']:+.2f}%")
    except Exception as e:
        print(f"\n排名 {results_df.index.get_loc(idx)+1}: CCI({cci_len},MA{ma_len}) - 测试失败: {e}")

# ============ 与默认参数对比 ============
print("\n" + "="*80)
print("与默认参数对比".center(80))
print("="*80)

default_params = (20, 14)  # 默认参数
default_train = backtest_cci(train_df, default_params[0], default_params[1])
default_test = backtest_cci(test_df, default_params[0], default_params[1])

print(f"\n{'参数版本':<25} | {'训练收益':<12} | {'测试收益':<12} | {'测试胜率':<10} | {'测试回撤':<10}")
print("-"*80)
print(f"{'默认参数 CCI(20,14)':<25} | {default_train['return_pct']:>+10.2f}% | {default_test['return_pct']:>+10.2f}% | "
      f"{default_test['win_rate']:>9.1f}% | {default_test['max_dd']:>9.2f}%")

if out_of_sample_results:
    # 找到测试收益最高的
    best_grid = max(out_of_sample_results, key=lambda x: x['return_pct'])
    best_params_str = f"CCI({best_grid['cci_length']},MA{best_grid['ma_length']})"
    print(f"{best_params_str:<25} | {best_grid['train_return']:>+10.2f}% | {best_grid['return_pct']:>+10.2f}% | "
          f"{best_grid['win_rate']:>9.1f}% | {best_grid['max_dd']:>9.2f}%")

    improvement = best_grid['return_pct'] - default_test['return_pct']
    dd_improvement = best_grid['max_dd'] - default_test['max_dd']
    print("-"*80)
    if improvement > 10:
        print(f"\n[SUCCESS] 网格优化显著优于默认参数 {improvement:.2f}%")
        if dd_improvement < 0:
            print(f"          回撤也更小 {dd_improvement:.2f}%")
        print(f"          强烈推荐使用 CCI({best_grid['cci_length']}, MA{best_grid['ma_length']})")
    elif improvement > 0:
        print(f"\n[OK] 网格优化略优于默认参数 {improvement:.2f}%")
        print(f"   可以使用，但默认参数也很稳健")
    elif improvement > -10:
        print(f"\n[INFO] 网格优化与默认参数接近 ({improvement:+.2f}%)")
        print(f"       建议使用默认参数（更稳健）")
    else:
        print(f"\n[INFO] 默认参数优于网格优化 {abs(improvement):.2f}%")

# ============ 参数敏感性分析 ============
print("\n" + "="*80)
print("参数敏感性分析（测试集）".center(80))
print("="*80)

# 按CCI长度分组
print("\n按CCI长度分组（测试集平均表现）:")
print(f"{'CCI长度':<10} {'平均收益':<12} {'平均回撤':<12} {'平均胜率':<10} {'平均交易':<10}")
print("-"*80)

for cci_len in cci_length_grid:
    subset = results_df[results_df['cci_length'] == cci_len]
    if len(subset) > 0:
        # 在测试集上重新评估这些参数
        test_returns = []
        test_dds = []
        test_win_rates = []
        test_trades = []

        for _, row in subset.iterrows():
            try:
                test_res = backtest_cci(test_df, int(row['cci_length']), int(row['ma_length']))
                if test_res:
                    test_returns.append(test_res['return_pct'])
                    test_dds.append(test_res['max_dd'])
                    test_win_rates.append(test_res['win_rate'])
                    test_trades.append(test_res['total_trades'])
            except:
                pass

        if test_returns:
            avg_ret = np.mean(test_returns)
            avg_dd = np.mean(test_dds)
            avg_wr = np.mean(test_win_rates)
            avg_trades = np.mean(test_trades)
            print(f"{cci_len:<10} {avg_ret:>+10.2f}% {avg_dd:>10.2f}% {avg_wr:>9.1f}% {avg_trades:>8.0f}")

# 按MA长度分组
print("\n按MA长度分组（测试集平均表现）:")
print(f"{'MA长度':<10} {'平均收益':<12} {'平均回撤':<12} {'平均胜率':<10} {'平均交易':<10}")
print("-"*80)

for ma_len in ma_length_grid:
    subset = results_df[results_df['ma_length'] == ma_len]
    if len(subset) > 0:
        test_returns = []
        test_dds = []
        test_win_rates = []
        test_trades = []

        for _, row in subset.iterrows():
            try:
                test_res = backtest_cci(test_df, int(row['cci_length']), int(row['ma_length']))
                if test_res:
                    test_returns.append(test_res['return_pct'])
                    test_dds.append(test_res['max_dd'])
                    test_win_rates.append(test_res['win_rate'])
                    test_trades.append(test_res['total_trades'])
            except:
                pass

        if test_returns:
            avg_ret = np.mean(test_returns)
            avg_dd = np.mean(test_dds)
            avg_wr = np.mean(test_win_rates)
            avg_trades = np.mean(test_trades)
            print(f"{ma_len:<10} {avg_ret:>+10.2f}% {avg_dd:>10.2f}% {avg_wr:>9.1f}% {avg_trades:>8.0f}")

print("\n" + "="*80)
print("优化完成！")
print("="*80)
