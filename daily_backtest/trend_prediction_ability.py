# -*- coding: utf-8 -*-
"""
趋势预测能力测试：GHE+LPPL组合
直接测试：给定当前的GHE和LPPL-D值，能否预测未来N天的涨跌？
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trend_prediction_tool import calculate_ghe, calculate_lppl_d
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.chdir(r"D:\期货数据\铜期货监控\daily_backtest")

# 读取数据
df = pd.read_csv('SN_沪锡_日线_10年.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.rename(columns={'收盘价': 'close'})

print("="*80)
print("趋势预测能力测试：GHE+LPPL组合")
print("="*80)

# 计算GHE和LPPL-D
print("\n正在计算GHE和LPPL-D...")
ghe_values = []
lppl_values = []

ghe_valid = 0
lppl_valid = 0
both_valid = 0

window_size = 200  # LPPL-D需要至少200天数据

for i in range(window_size, len(df)):
    window_data = df.iloc[i-window_size:i]
    ghe = calculate_ghe(window_data['close'])
    lppl = calculate_lppl_d(window_data['close'])

    if ghe is not None:
        ghe_valid += 1
    if lppl is not None:
        lppl_valid += 1
    if ghe is not None and lppl is not None:
        both_valid += 1

    ghe_values.append(ghe)
    lppl_values.append(lppl)

total = len(df) - window_size
print(f"GHE有效: {ghe_valid}/{total} ({ghe_valid/total*100:.1f}%)")
print(f"LPPL-D有效: {lppl_valid}/{total} ({lppl_valid/total*100:.1f}%)")
print(f"两者都有效: {both_valid}/{total} ({both_valid/total*100:.1f}%)")

# 创建分析数据框
analysis_df = pd.DataFrame({
    'date': df.iloc[window_size:].reset_index(drop=True)['datetime'],
    'close': df.iloc[window_size:].reset_index(drop=True)['close'],
    'ghe': ghe_values,
    'lppl': lppl_values
}).dropna()

print(f"最终分析样本数: {len(analysis_df)}天")

# 测试不同预测周期的收益
prediction_periods = [5, 10, 20]  # 预测未来5天、10天、20天

for period in prediction_periods:
    print(f"\n{'='*80}")
    print(f"测试：预测未来{period}天的趋势")
    print(f"{'='*80}")

    # 计算未来收益率
    analysis_df[f'future_return_{period}d'] = analysis_df['close'].pct_change(period).shift(-period) * 100

    # 删除最后几行（没有未来数据）
    valid_data = analysis_df.dropna(subset=[f'future_return_{period}d']).copy()
    print(f"有效样本数: {len(valid_data)}天")

    # 将GHE和LPPL-D分组
    valid_data['ghe_group'] = pd.cut(valid_data['ghe'], bins=[0, 0.3, 0.4, 0.5, 1.0],
                                      labels=['<0.3', '0.3-0.4', '0.4-0.5', '>=0.5'])
    valid_data['lppl_group'] = pd.cut(valid_data['lppl'], bins=[0, 0.3, 0.5, 0.8, 2.0],
                                       labels=['<0.3', '0.3-0.5', '0.5-0.8', '>=0.8'])

    # 分析1: GHE单独预测
    print(f"\n【GHE单独预测未来{period}天】")
    print(f"{'GHE区间':<12} {'样本数':<10} {'平均收益':<15} {'上涨概率':<12}")
    print("-"*60)

    ghe_performance = []
    for label in ['<0.3', '0.3-0.4', '0.4-0.5', '>=0.5']:
        group = valid_data[valid_data['ghe_group'] == label]
        if len(group) > 0:
            avg_return = group[f'future_return_{period}d'].mean()
            up_prob = (group[f'future_return_{period}d'] > 0).mean() * 100
            print(f"{label:<12} {len(group):<10} {avg_return:>8.2f}%      {up_prob:>8.1f}%")
            ghe_performance.append({
                'group': label,
                'count': len(group),
                'avg_return': avg_return,
                'up_prob': up_prob
            })

    # 分析2: LPPL-D单独预测
    print(f"\n【LPPL-D单独预测未来{period}天】")
    print(f"{'LPPL-D区间':<12} {'样本数':<10} {'平均收益':<15} {'上涨概率':<12}")
    print("-"*60)

    lppl_performance = []
    for label in ['<0.3', '0.3-0.5', '0.5-0.8', '>=0.8']:
        group = valid_data[valid_data['lppl_group'] == label]
        if len(group) > 0:
            avg_return = group[f'future_return_{period}d'].mean()
            up_prob = (group[f'future_return_{period}d'] > 0).mean() * 100
            print(f"{label:<12} {len(group):<10} {avg_return:>8.2f}%      {up_prob:>8.1f}%")
            lppl_performance.append({
                'group': label,
                'count': len(group),
                'avg_return': avg_return,
                'up_prob': up_prob
            })

    # 分析3: GHE+LPPL组合预测
    print(f"\n【GHE+LPPL组合预测未来{period}天】")
    print(f"{'GHE':<10} {'LPPL-D':<10} {'样本':<8} {'平均收益':<12} {'上涨概率':<10}")
    print("-"*70)

    combo_performance = []
    for ghe_label in ['<0.3', '0.3-0.4', '0.4-0.5', '>=0.5']:
        for lppl_label in ['<0.3', '0.3-0.5', '0.5-0.8', '>=0.8']:
            group = valid_data[(valid_data['ghe_group'] == ghe_label) &
                             (valid_data['lppl_group'] == lppl_label)]
            if len(group) >= 10:  # 只显示样本>=10的组合
                avg_return = group[f'future_return_{period}d'].mean()
                up_prob = (group[f'future_return_{period}d'] > 0).mean() * 100
                print(f"{ghe_label:<10} {lppl_label:<10} {len(group):<8} "
                      f"{avg_return:>8.2f}%    {up_prob:>8.1f}%")
                combo_performance.append({
                    'ghe': ghe_label,
                    'lppl': lppl_label,
                    'count': len(group),
                    'avg_return': avg_return,
                    'up_prob': up_prob
                })

    # 分析4: 找出最佳和最差组合
    if combo_performance:
        print(f"\n【最佳组合】（平均收益最高）")
        best = max(combo_performance, key=lambda x: x['avg_return'])
        print(f"  GHE={best['ghe']}, LPPL-D={best['lppl']}")
        print(f"  样本数: {best['count']}, 平均收益: {best['avg_return']:.2f}%, 上涨概率: {best['up_prob']:.1f}%")

        print(f"\n【最差组合】（平均收益最低）")
        worst = min(combo_performance, key=lambda x: x['avg_return'])
        print(f"  GHE={worst['ghe']}, LPPL-D={worst['lppl']}")
        print(f"  样本数: {worst['count']}, 平均收益: {worst['avg_return']:.2f}%, 上涨概率: {worst['up_prob']:.1f}%")

        print(f"\n【最高上涨概率】")
        highest_prob = max(combo_performance, key=lambda x: x['up_prob'])
        print(f"  GHE={highest_prob['ghe']}, LPPL-D={highest_prob['lppl']}")
        print(f"  样本数: {highest_prob['count']}, 平均收益: {highest_prob['avg_return']:.2f}%, 上涨概率: {highest_prob['up_prob']:.1f}%")

    # 分析5: 预测能力评估
    print(f"\n【预测能力评估】")

    # 整体基准
    overall_avg = valid_data[f'future_return_{period}d'].mean()
    overall_up = (valid_data[f'future_return_{period}d'] > 0).mean() * 100
    print(f"整体基准: 平均收益 {overall_avg:.2f}%, 上涨概率 {overall_up:.1f}%")

    # 检查是否有明显的规律
    if ghe_performance:
        ghe_returns = [x['avg_return'] for x in ghe_performance]
        ghe_range = max(ghe_returns) - min(ghe_returns)
        print(f"\nGHE预测能力: 收益差距 {ghe_range:.2f}%")
        if ghe_range > 1.0:
            print("  -> GHE有一定的预测能力")
        else:
            print("  -> GHE预测能力较弱")

    if lppl_performance:
        lppl_returns = [x['avg_return'] for x in lppl_performance]
        lppl_range = max(lppl_returns) - min(lppl_returns)
        print(f"\nLPPL-D预测能力: 收益差距 {lppl_range:.2f}%")
        if lppl_range > 1.0:
            print("  -> LPPL-D有一定的预测能力")
        else:
            print("  -> LPPL-D预测能力较弱")

# 最终结论
print(f"\n{'='*80}")
print("【最终结论】GHE+LPPL的趋势预测能力")
print(f"{'='*80}")

print("""
基于10年数据的实际测试，我们要回答一个问题：
给定当前的GHE和LPPL-D值，能否可靠地预测未来趋势？

判断标准：
1. 不同GHE/LPPL区间是否有显著的收益差异（>1%）
2. 是否存在稳定的高概率上涨组合（>60%）
3. 规律是否在不同预测周期都成立

请根据以上测试结果，判断GHE+LPPL是否具有趋势预测能力。
""")

print("\n" + "="*80)
print("测试完成")
print("="*80)
