# -*- coding: utf-8 -*-
"""
反向分析：价格涨跌 vs GHE+LPPL特征
从结果反推原因，而不是从原因预测结果
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
print("反向分析：价格涨跌 vs GHE+LPPL特征")
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
    ghe_values.append(ghe)
    lppl_values.append(lppl)

# 创建分析数据框
df_analysis = df.iloc[window_size:].reset_index(drop=True).copy()
df_analysis['ghe'] = ghe_values
df_analysis['lppl'] = lppl_values
df_analysis = df_analysis.dropna().reset_index(drop=True)

print(f"有效样本数: {len(df_analysis)}天")

# 定义涨跌幅度
change_thresholds = [0.10, 0.20]  # 10%, 20%

for threshold in change_thresholds:
    print(f"\n{'='*80}")
    print(f"分析：价格变化幅度 {threshold*100:.0f}%")
    print(f"{'='*80}")

    # 计算未来N天的累计涨跌幅
    for days in [5, 10, 20, 40]:
        print(f"\n--- 未来{days}天涨跌{threshold*100:.0f}%的时期 ---")

        # 计算未来涨跌幅
        df_analysis[f'future_change_{days}d'] = df_analysis['close'].pct_change(days).shift(-days)

        # 分类：大涨、小涨、横盘、小跌、大跌
        conditions = [
            df_analysis[f'future_change_{days}d'] >= threshold,  # 大涨
            (df_analysis[f'future_change_{days}d'] > 0) & (df_analysis[f'future_change_{days}d'] < threshold),  # 小涨
            (df_analysis[f'future_change_{days}d'] >= -threshold/2) & (df_analysis[f'future_change_{days}d'] <= threshold/2),  # 横盘
            (df_analysis[f'future_change_{days}d'] < 0) & (df_analysis[f'future_change_{days}d'] > -threshold),  # 小跌
            df_analysis[f'future_change_{days}d'] <= -threshold,  # 大跌
        ]
        labels = ['大涨', '小涨', '横盘', '小跌', '大跌']

        df_analysis['category'] = np.select(conditions, labels, default='横盘')

        # 删除无效数据
        valid_data = df_analysis[df_analysis[f'future_change_{days}d'].notna()].copy()

        if len(valid_data) == 0:
            continue

        print(f"\n有效样本: {len(valid_data)}天")

        # 统计每个类别的GHE和LPPL-D特征
        print(f"\n{'类别':<8} {'样本数':<8} {'GHE均值':<12} {'LPPL-D均值':<12}")
        print("-"*60)

        category_stats = {}

        for label in labels:
            category_data = valid_data[valid_data['category'] == label]

            if len(category_data) > 0:
                ghe_mean = category_data['ghe'].mean()
                ghe_std = category_data['ghe'].std()
                lppl_mean = category_data['lppl'].mean()
                lppl_std = category_data['lppl'].std()

                category_stats[label] = {
                    'count': len(category_data),
                    'ghe_mean': ghe_mean,
                    'ghe_std': ghe_std,
                    'lppl_mean': lppl_mean,
                    'lppl_std': lppl_std
                }

                print(f"{label:<8} {len(category_data):<8} "
                      f"{ghe_mean:>6.3f}±{ghe_std:<5.3f} "
                      f"{lppl_mean:>6.3f}±{lppl_std:<5.3f}")

        # 分析：大涨 vs 大跌的差异
        if '大涨' in category_stats and '大跌' in category_stats:
            print(f"\n【关键对比：大涨 vs 大跌】")
            print(f"GHE差异: {category_stats['大涨']['ghe_mean'] - category_stats['大跌']['ghe_mean']:.3f}")
            print(f"LPPL-D差异: {category_stats['大涨']['lppl_mean'] - category_stats['大跌']['lppl_mean']:.3f}")

            # 判断是否有明显差异
            ghe_diff = abs(category_stats['大涨']['ghe_mean'] - category_stats['大跌']['ghe_mean'])
            lppl_diff = abs(category_stats['大涨']['lppl_mean'] - category_stats['大跌']['lppl_mean'])

            if ghe_diff > 0.05:
                print(f"→ GHE有明显差异 ({ghe_diff:.3f})")
            else:
                print(f"→ GHE差异不明显 ({ghe_diff:.3f})")

            if lppl_diff > 0.2:
                print(f"→ LPPL-D有明显差异 ({lppl_diff:.3f})")
            else:
                print(f"→ LPPL-D差异不明显 ({lppl_diff:.3f})")

        # 分析：大涨 vs 其他所有类别
        if '大涨' in category_stats:
            print(f"\n【大涨时期的特征】")
            print(f"GHE: {category_stats['大涨']['ghe_mean']:.3f}±{category_stats['大涨']['ghe_std']:.3f}")
            print(f"LPPL-D: {category_stats['大涨']['lppl_mean']:.3f}±{category_stats['大涨']['lppl_std']:.3f}")

            # 检查大涨时期的GHE和LPPL-D主要分布区间
            big_rise_data = valid_data[valid_data['category'] == '大涨']
            print(f"\n大涨时期的GHE分布:")
            print(f"  <0.3: {(big_rise_data['ghe'] < 0.3).sum()} ({(big_rise_data['ghe'] < 0.3).mean()*100:.1f}%)")
            print(f"  0.3-0.4: {((big_rise_data['ghe'] >= 0.3) & (big_rise_data['ghe'] < 0.4)).sum()} "
                  f"({((big_rise_data['ghe'] >= 0.3) & (big_rise_data['ghe'] < 0.4)).mean()*100:.1f}%)")
            print(f"  0.4-0.5: {((big_rise_data['ghe'] >= 0.4) & (big_rise_data['ghe'] < 0.5)).sum()} "
                  f"({((big_rise_data['ghe'] >= 0.4) & (big_rise_data['ghe'] < 0.5)).mean()*100:.1f}%)")
            print(f"  ≥0.5: {(big_rise_data['ghe'] >= 0.5).sum()} ({(big_rise_data['ghe'] >= 0.5).mean()*100:.1f}%)")

            print(f"\n大涨时期的LPPL-D分布:")
            print(f"  <0.3: {(big_rise_data['lppl'] < 0.3).sum()} ({(big_rise_data['lppl'] < 0.3).mean()*100:.1f}%)")
            print(f"  0.3-0.5: {((big_rise_data['lppl'] >= 0.3) & (big_rise_data['lppl'] < 0.5)).sum()} "
                  f"({((big_rise_data['lppl'] >= 0.3) & (big_rise_data['lppl'] < 0.5)).mean()*100:.1f}%)")
            print(f"  0.5-0.8: {((big_rise_data['lppl'] >= 0.5) & (big_rise_data['lppl'] < 0.8)).sum()} "
                  f"({((big_rise_data['lppl'] >= 0.5) & (big_rise_data['lppl'] < 0.8)).mean()*100:.1f}%)")
            print(f"  ≥0.8: {(big_rise_data['lppl'] >= 0.8).sum()} ({(big_rise_data['lppl'] >= 0.8).mean()*100:.1f}%)")

# 最终总结
print(f"\n{'='*80}")
print("最终总结")
print(f"{'='*80}")

print("""
这种方法的优势：
1. 从结果反推原因，避免了"预测偏差"
2. 先找大涨/大跌的时期，再看这些时期的GHE+LPPL特征
3. 如果某个特征在大涨时期频繁出现，说明这个特征有预测价值

判断标准：
- 如果大涨时期GHE/LPPL有明显特征（比如>60%集中在某个区间）
- 说明这个特征确实有预测能力
- 如果大涨时期的GHE/LPPL分布很散乱
- 说明这个指标没有预测价值
""")

print("\n" + "="*80)
print("分析完成")
print("="*80)
