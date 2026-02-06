# -*- coding: utf-8 -*-
"""
快速分析：所有亏损交易的GHE+LPPL组合
"""

import pandas as pd
import numpy as np
import os

os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

# 读取数据
df = pd.read_csv('trades_with_ghe_lppl.csv')

# 分离亏损交易
losing_trades = df[df['pnl'] <= 0].copy()

print("="*80)
print(f"所有亏损交易深度分析（共{len(losing_trades)}笔）")
print("="*80)

# 1. 详细列出每笔亏损交易（前30笔）
print(f"\n{'序号':<6} {'入场日期':<12} {'GHE':<8} {'LPPL-D':<10} {'亏损率':<10} {'亏损额':<12} {'出场原因':<12}")
print("-"*110)

for i, (_, trade) in enumerate(losing_trades.head(30).iterrows(), 1):
    print(f"{i:<6} {trade['entry_datetime']:<12} "
          f"{trade['entry_ghe']:<8.3f} {trade['entry_lppl']:<10.3f} "
          f"{trade['pnl_pct']:<10.2f}% {trade['pnl']:<12,.0f}元 {trade['exit_reason']:<12}")

print(f"\n... 还有{len(losing_trades)-30}笔交易")

# 2. 统计分析
losing_ghe = losing_trades['entry_ghe']
losing_lppl = losing_trades['entry_lppl']

print(f"\n" + "="*80)
print("亏损交易的GHE+LPPL特征统计")
print("="*80)

print(f"\n【GHE特征】")
print(f"  均值: {losing_ghe.mean():.3f}")
print(f"  中位数: {losing_ghe.median():.3f}")
print(f"  标准差: {losing_ghe.std():.3f}")
print(f"  范围: {losing_ghe.min():.3f} ~ {losing_ghe.max():.3f}")

print(f"\n【LPPL-D特征】")
print(f"  均值: {losing_lppl.mean():.3f}")
print(f"  中位数: {losing_lppl.median():.3f}")
print(f"  标准差: {losing_lppl.std():.3f}")
print(f"  范围: {losing_lppl.min():.3f} ~ {losing_lppl.max():.3f}")

# 3. 区间分布
ghe_bins = [0, 0.3, 0.4, 0.5, 1.0]
ghe_labels = ['<0.3', '0.3-0.4', '0.4-0.5', '≥0.5']
losing_trades['ghe_group'] = pd.cut(losing_trades['entry_ghe'], bins=ghe_bins, labels=ghe_labels)

lppl_bins = [0, 0.3, 0.5, 0.8, 2.0]
lppl_labels = ['<0.3', '0.3-0.5', '0.5-0.8', '≥0.8']
losing_trades['lppl_group'] = pd.cut(losing_trades['entry_lppl'], bins=lppl_bins, labels=lppl_labels)

print(f"\n" + "="*80)
print("亏损交易的区间分布")
print("="*80)

print(f"\n【GHE区间分布】")
print(f"{'GHE区间':<12} {'亏损笔数':<12} {'占比':<10}")
print("-"*40)
for label in ghe_labels:
    count = len(losing_trades[losing_trades['ghe_group'] == label])
    print(f"{label:<12} {count:<12} {count/len(losing_trades)*100:<10.1f}%")

print(f"\n【LPPL-D区间分布】")
print(f"{'LPPL-D区间':<12} {'亏损笔数':<12} {'占比':<10}")
print("-"*40)
for label in lppl_labels:
    count = len(losing_trades[losing_trades['lppl_group'] == label])
    print(f"{label:<12} {count:<12} {count/len(losing_trades)*100:<10.1f}%")

# 4. 最危险的组合
print(f"\n" + "="*80)
print("最危险的GHE+LPPL组合（亏损最集中）")
print("="*80)

combo_stats = losing_trades.groupby(['ghe_group', 'lppl_group']).agg({
    'pnl': ['count', 'sum', 'mean']
}).round(2)

print(f"\n{'GHE':<10} {'LPPL-D':<10} {'亏损笔数':<10} {'总亏损':<15} {'平均亏损':<12}")
print("-"*80)

sorted_combos = []
for (ghe, lppl), group in combo_stats.groupby(level=[0, 1]):
    count = int(group[('pnl', 'count')].iloc[0])
    total_loss = group[('pnl', 'sum')].iloc[0]
    avg_loss = group[('pnl', 'mean')].iloc[0]

    sorted_combos.append({
        'ghe': ghe,
        'lppl': lppl,
        'count': count,
        'total_loss': total_loss,
        'avg_loss': avg_loss
    })

sorted_combos.sort(key=lambda x: x['count'], reverse=True)

for combo in sorted_combos:
    print(f"{combo['ghe']:<10} {combo['lppl']:<10} {combo['count']:<10} "
          f"{combo['total_loss']:>13,.0f}元 {combo['avg_loss']:>10.2f}%")

# 5. 最大亏损交易
print(f"\n" + "="*80)
print("亏损最严重的10笔交易")
print("="*80)

top_losses = losing_trades.nsmallest(10, 'pnl')
print(f"\n{'序号':<6} {'入场日期':<12} {'GHE':<8} {'LPPL-D':<10} {'亏损率':<10} {'亏损额':<12}")
print("-"*80)

for i, (_, trade) in enumerate(top_losses.iterrows(), 1):
    print(f"{i:<6} {trade['entry_datetime']:<12} "
          f"{trade['entry_ghe']:<8.3f} {trade['entry_lppl']:<10.3f} "
          f"{trade['pnl_pct']:<10.2f}% {trade['pnl']:<12,.0f}元")

print(f"\n" + "="*80)
print("分析完成")
print("="*80)
