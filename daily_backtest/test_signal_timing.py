# -*- coding: utf-8 -*-
"""
测试信号生成和价格对应的逻辑
对比：错误逻辑 vs 正确逻辑
"""

import pandas as pd
import numpy as np
from trend_prediction_tool import calculate_ghe, calculate_lppl_d
import os

os.chdir(r"D:\期货数据\铜期货监控\daily_backtest")

# 读取数据
df = pd.read_csv('SN_沪锡_日线_10年.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.rename(columns={'收盘价': 'close', '开盘价': 'open'})

print("="*80)
print("信号生成和价格对应逻辑测试")
print("="*80)

# 计算GHE和LPPL-D
print("\n正在计算GHE和LPPL-D...")
window_size = 200

ghe_values = []
lppl_values = []

for i in range(window_size, len(df)):
    window_data = df.iloc[i-window_size:i]
    ghe = calculate_ghe(window_data['close'])
    lppl = calculate_lppl_d(window_data['close'])

    if ghe is not None and lppl is not None:
        ghe_values.append(ghe)
        lppl_values.append(lppl)
    else:
        ghe_values.append(None)
        lppl_values.append(None)

# 添加到数据框
df_analysis = df.iloc[window_size:].copy()
df_analysis['ghe'] = ghe_values
df_analysis['lppl'] = lppl_values
df_analysis = df_analysis.dropna().reset_index(drop=True)

print(f"有效样本数: {len(df_analysis)}")

print(f"\n{'='*80}")
print("逻辑对比测试")
print(f"{'='*80}")

# 测试：GHE 0.50-0.60 + LPPL-D 0.7-0.9 组合
target_ghe_min = 0.50
target_ghe_max = 0.60
target_lppl_min = 0.7
target_lppl_max = 0.9

signal_dates = df_analysis[
    (df_analysis['ghe'] >= target_ghe_min) &
    (df_analysis['ghe'] <= target_ghe_max) &
    (df_analysis['lppl'] >= target_lppl_min) &
    (df_analysis['lppl'] <= target_lppl_max)
].copy()

print(f"\n找到信号: {len(signal_dates)}次")
print(f"GHE区间: {target_ghe_min}-{target_ghe_max}")
print(f"LPPL-D区间: {target_lppl_min}-{target_lppl_max}")

if len(signal_dates) == 0:
    print("\n没有找到符合条件的信号，尝试放宽条件...")
    signal_dates = df_analysis[
        (df_analysis['ghe'] >= 0.50) &
        (df_analysis['ghe'] <= 0.55) &
        (df_analysis['lppl'] >= 0.8) &
        (df_analysis['lppl'] <= 0.9)
    ].copy()
    print(f"放宽条件后找到信号: {len(signal_dates)}次")

# 对比两种逻辑
print(f"\n{'='*80}")
print("逻辑1：错误逻辑（之前用的）")
print(f"{'='*80}")
print(f"第i天收盘后生成信号 → 计算第i天收盘到第i+20天收盘的收益")
print(f"问题：无法在第i天收盘时交易！")

print(f"\n{'='*80}")
print("逻辑2：正确逻辑（实际交易）")
print(f"{'='*80}")
print(f"第i天收盘后生成信号 → 第i+1天开盘入场 → 计算第i+1天开盘到第i+21天收盘的收益")

# 计算两种逻辑的收益
results = []

for idx, row in signal_dates.iterrows():
    signal_idx = df[df['datetime'] == row['datetime']].index[0]

    # 逻辑1：错误（之前的计算方式）
    if signal_idx + 20 < len(df):
        entry_price_wrong = df.loc[signal_idx, 'close']
        exit_price_wrong = df.loc[signal_idx + 20, 'close']
        return_wrong = (exit_price_wrong - entry_price_wrong) / entry_price_wrong * 100
    else:
        return_wrong = None

    # 逻辑2：正确（实际交易方式）
    if signal_idx + 21 < len(df):
        entry_price_correct = df.loc[signal_idx + 1, 'open']  # 第二天开盘入场
        exit_price_correct = df.loc[signal_idx + 21, 'close']
        return_correct = (exit_price_correct - entry_price_correct) / entry_price_correct * 100
    else:
        return_correct = None

    results.append({
        'signal_date': row['datetime'],
        'ghe': row['ghe'],
        'lppl': row['lppl'],
        'return_wrong': return_wrong,
        'return_correct': return_correct
    })

results_df = pd.DataFrame(results)

# 统计对比
print(f"\n{'='*80}")
print("结果对比（前10个信号）")
print(f"{'='*80}")

print(f"\n{'信号日期':<12} {'GHE':<8} {'LPPL-D':<8} {'错误逻辑收益':<15} {'正确逻辑收益':<15} {'差异':<10}")
print("-"*80)

for _, row in results_df.head(10).iterrows():
    if row['return_wrong'] is not None and row['return_correct'] is not None:
        diff = row['return_correct'] - row['return_wrong']
        print(f"{str(row['signal_date']):<12} {row['ghe']:<8.3f} {row['lppl']:<8.3f} "
              f"{row['return_wrong']:>8.2f}%       {row['return_correct']:>8.2f}%       {diff:>7.2f}%")

# 统计分析
valid_results = results_df[(results_df['return_wrong'].notna()) &
                           (results_df['return_correct'].notna())]

print(f"\n{'='*80}")
print("统计分析（共{0}个有效信号）".format(len(valid_results)))
print(f"{'='*80}")

print(f"\n【错误逻辑】")
print(f"  平均收益: {valid_results['return_wrong'].mean():.2f}%")
print(f"  收益标准差: {valid_results['return_wrong'].std():.2f}%")
print(f"  最大收益: {valid_results['return_wrong'].max():.2f}%")
print(f"  最小收益: {valid_results['return_wrong'].min():.2f}%")
print(f"  上涨次数: {(valid_results['return_wrong'] > 0).sum()}/{len(valid_results)} "
      f"({(valid_results['return_wrong'] > 0).mean()*100:.1f}%)")

print(f"\n【正确逻辑】")
print(f"  平均收益: {valid_results['return_correct'].mean():.2f}%")
print(f"  收益标准差: {valid_results['return_correct'].std():.2f}%")
print(f"  最大收益: {valid_results['return_correct'].max():.2f}%")
print(f"  最小收益: {valid_results['return_correct'].min():.2f}%")
print(f"  上涨次数: {(valid_results['return_correct'] > 0).sum()}/{len(valid_results)} "
      f"({(valid_results['return_correct'] > 0).mean()*100:.1f}%)")

print(f"\n【差异分析】")
avg_diff = (valid_results['return_correct'] - valid_results['return_wrong']).mean()
print(f"  平均差异: {avg_diff:.2f}%")
print(f"  → 正确逻辑比错误逻辑{'高' if avg_diff > 0 else '低'}{abs(avg_diff):.2f}个百分点")

print(f"\n{'='*80}")
print("结论")
print(f"{'='*80}")

print(f"""
之前的分析存在逻辑错误：

1. 信号生成时间：
   - 第i天收盘后，用前200天数据计算GHE和LPPL-D
   - 信号在第i天收盘后才生成

2. 实际交易时间：
   - 第i天收盘时已无法交易
   - 最早只能在第i+1天开盘时入场

3. 收益计算差异：
   - 错误逻辑：第i天收盘 → 第i+20天收盘
   - 正确逻辑：第i+1天开盘 → 第i+21天收盘
   - 开盘价和收盘价之间的差异（隔夜跳空）会影响收益

4. 影响：
   - 如果市场经常跳空高开/低开
   - 两种逻辑的收益会有明显差异
   - 之前的"100%准确"可能是基于错误逻辑
""")

print("\n" + "="*80)
print("测试完成")
print("="*80)
