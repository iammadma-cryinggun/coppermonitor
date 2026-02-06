# -*- coding: utf-8 -*-
"""
深度分析：所有亏损交易的GHE+LPPL特征
找出最容易亏钱的组合
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

# 读取数据
df = pd.read_csv('trades_with_ghe_lppl.csv')
df['entry_datetime'] = pd.to_datetime(df['entry_datetime'])
df['exit_datetime'] = pd.to_datetime(df['exit_datetime'])

# 分离亏损交易
losing_trades = df[df['pnl'] <= 0].copy()
winning_trades = df[df['pnl'] > 0].copy()

print("="*80)
print(f"所有亏损交易深度分析（共{len(losing_trades)}笔）")
print("="*80)

# 1. 详细列出每笔亏损交易
print(f"\n{'序号':<6} {'入场日期':<12} {'GHE':<8} {'LPPL-D':<10} {'亏损率':<10} {'亏损额':<12} {'出场原因':<12}")
print("-"*100)

for i, (_, trade) in enumerate(losing_trades.iterrows(), 1):
    print(f"{i:<6} {trade['entry_datetime'].date():<12} "
          f"{trade['entry_ghe']:<8.3f} {trade['entry_lppl']:<10.3f} "
          f"{trade['pnl_pct']:<10.2f}% {trade['pnl']:<12,.0f}元 {trade['exit_reason']:<12}")

# 2. 统计分析
print("\n" + "="*80)
print("亏损交易的GHE+LPPL特征统计")
print("="*80)

# GHE统计
losing_ghe = losing_trades['entry_ghe']
print(f"\n【GHE特征】")
print(f"  均值: {losing_ghe.mean():.3f}")
print(f"  中位数: {losing_ghe.median():.3f}")
print(f"  标准差: {losing_ghe.std():.3f}")
print(f"  最小值: {losing_ghe.min():.3f}")
print(f"  最大值: {losing_ghe.max():.3f}")

# LPPL-D统计
losing_lppl = losing_trades['entry_lppl']
print(f"\n【LPPL-D特征】")
print(f"  均值: {losing_lppl.mean():.3f}")
print(f"  中位数: {losing_lppl.median():.3f}")
print(f"  标准差: {losing_lppl.std():.3f}")
print(f"  最小值: {losing_lppl.min():.3f}")
print(f"  最大值: {losing_lppl.max():.3f}")

# 3. 亏损幅度分析
print("\n" + "="*80)
print("亏损幅度分析")
print("="*80)

print(f"\n最大10笔亏损:")
top_losses = losing_trades.nsmallest(10, 'pnl')
print(f"\n{'序号':<6} {'入场日期':<12} {'GHE':<8} {'LPPL-D':<10} {'亏损率':<10} {'亏损额':<12}")
print("-"*80)

for i, (_, trade) in enumerate(top_losses.iterrows(), 1):
    print(f"{i:<6} {trade['entry_datetime'].date():<12} "
          f"{trade['entry_ghe']:<8.3f} {trade['entry_lppl']:<10.3f} "
          f"{trade['pnl_pct']:<10.2f}% {trade['pnl']:<12,.0f}元")

# 统计亏损原因
print("\n" + "="*80)
print("亏损原因统计")
print("="*80)

exit_reason_counts = losing_trades['exit_reason'].value_counts()
print(f"\n{'出场原因':<15} {'笔数':<8} {'占比':<10} {'平均亏损率':<12}")
print("-"*60)

for reason, count in exit_reason_counts.items():
    avg_loss = losing_trades[losing_trades['exit_reason'] == reason]['pnl_pct'].mean()
    print(f"{reason:<15} {count:<8} {count/len(losing_trades)*100:<10.1f}% {avg_loss:<12.2f}%")

# 4. GHE和LPPL-D的区间分布
print("\n" + "="*80)
print("亏损交易的GHE和LPPL-D区间分布")
print("="*80)

# GHE区间
ghe_bins = [0, 0.3, 0.4, 0.5, 1.0]
ghe_labels = ['<0.3', '0.3-0.4', '0.4-0.5', '≥0.5']

losing_trades['ghe_group'] = pd.cut(losing_trades['entry_ghe'], bins=ghe_bins, labels=ghe_labels)

print(f"\n【GHE区间分布】")
print(f"\n{'GHE区间':<12} {'亏损交易数':<12} {'占比':<10}")
print("-"*40)

for label in ghe_labels:
    count = len(losing_trades[losing_trades['ghe_group'] == label])
    if count > 0:
        print(f"{label:<12} {count:<12} {count/len(losing_trades)*100:<10.1f}%")

# LPPL-D区间
lppl_bins = [0, 0.3, 0.5, 0.8, 2.0]
lppl_labels = ['<0.3', '0.3-0.5', '0.5-0.8', '≥0.8']

losing_trades['lppl_group'] = pd.cut(losing_trades['entry_lppl'], bins=lppl_bins, labels=lppl_labels)

print(f"\n【LPPL-D区间分布】")
print(f"\n{'LPPL-D区间':<12} {'亏损交易数':<12} {'占比':<10}")
print("-"*40)

for label in lppl_labels:
    count = len(losing_trades[losing_trades['lppl_group'] == label])
    if count > 0:
        print(f"{label:<12} {count:<12} {count/len(losing_trades)*100:<10.1f}%")

# 5. 最危险的组合（亏损最多的组合）
print("\n" + "="*80)
print("最危险的GHE+LPPL组合（亏损最集中）")
print("="*80)

losing_trades['ghe_group'] = pd.cut(losing_trades['entry_ghe'], bins=ghe_bins, labels=ghe_labels)
losing_trades['lppl_group'] = pd.cut(losing_trades['entry_lppl'], bins=lppl_bins, labels=lppl_labels)

combo_stats = losing_trades.groupby(['ghe_group', 'lppl_group']).agg({
    'pnl': ['count', 'sum', 'mean']
}).round(2)

print(f"\n{'GHE':<10} {'LPPL-D':<10} {'亏损笔数':<10} {'总亏损':<12} {'平均亏损率':<12}")
print("-"*70)

for (ghe, lppl), group in combo_stats.groupby(level=[0, 1]):
    count = int(group[('pnl', 'count')])
    total_loss = group[('pnl', 'sum')].iloc[0]
    avg_loss = group[('pnl', 'mean')].iloc[0]

    print(f"{ghe:<10} {lppl:<10} {count:<10} {total_loss:>11,.0f}元 {avg_loss:>10.2f}%")

# 6. 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1：亏损交易的GHE分布
ax1 = axes[0, 0]
ax1.hist(losing_trades['entry_ghe'], bins=20, color='red', alpha=0.7, edgecolor='black')
ax1.axvline(losing_trades['entry_ghe'].mean(), color='darkred', linestyle='--',
           linewidth=2, label=f'均值{losing_trades["entry_ghe"].mean():.3f}')
ax1.set_xlabel('GHE值', fontsize=12)
ax1.set_ylabel('亏损交易数量', fontsize=12)
ax1.set_title('亏损交易的GHE分布', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.3)

# 图2：亏损交易的LPPL-D分布
ax2 = axes[0, 1]
ax2.hist(losing_trades['entry_lppl'], bins=20, color='red', alpha=0.7, edgecolor='black')
ax2.axvline(losing_trades['entry_lppl'].mean(), color='darkred', linestyle='--',
           linewidth=2, label=f'均值{losing_trades["entry_lppl"].mean():.3f}')
ax2.set_xlabel('LPPL-D值', fontsize=12)
ax2.set_ylabel('亏损交易数量', fontsize=12)
ax2.set_title('亏损交易的LPPL-D分布', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.3)

# 图3：GHE vs LPPL-D散点图（按亏损幅度着色）
ax3 = axes[1, 0]
scatter = ax3.scatter(losing_trades['entry_ghe'], losing_trades['entry_lppl'],
                     c=losing_trades['pnl_pct'], cmap='Reds', s=80,
                     vmin=-2.5, vmax=0, edgecolors='black', linewidth=0.5)
ax3.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='GHE=0.5阈值')
ax3.axhline(y=0.8, color='blue', linestyle='--', linewidth=2, label='LPPL-D=0.8阈值')
ax3.set_xlabel('GHE', fontsize=12)
ax3.set_ylabel('LPPL-D', fontsize=12)
ax3.set_title('亏损交易的GHE-LPPL分布（颜色=亏损幅度）', fontsize=14, fontweight='bold')
ax3.legend()
plt.colorbar(scatter, ax=ax3, label='亏损率(%)')
ax3.grid(True, linestyle='--', alpha=0.3)

# 图4：最危险组合条形图
ax4 = axes[1, 1]

# 只显示交易数≥3的组合
combo_stats_filtered = combo_stats.copy()
dangerous_combos = []
for (ghe, lppl), group in combo_stats_filtered.groupby(level=[0, 1]):
    count = int(group[('pnl', 'count')])
    total_loss = group[('pnl', 'sum')].iloc[0]
    if count >= 3:
        dangerous_combos.append({
            'ghe': ghe,
            'lppl': lppl,
            'count': count,
            'total_loss': total_loss
        })

if dangerous_combos:
    dangerous_combos = sorted(dangerous_combos, key=lambda x: x['count'], reverse=True)
    combo_labels = [f"{d['ghe']}\n{d['lppl']}" for d in dangerous_combos]
    combo_counts = [d['count'] for d in dangerous_combos]
    combo_losses = [d['total_loss'] for d in dangerous_combos]

    bars = ax4.bar(combo_labels, combo_counts, color='red', alpha=0.7, edgecolor='black')
    ax4.set_ylabel('亏损交易数量', fontsize=12, color='red')
    ax4.set_xlabel('GHE / LPPL-D组合', fontsize=12)
    ax4.set_title('最危险的GHE+LPPL组合（≥3笔亏损）', fontsize=14, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 添加总亏损标签
    for i, (bar, loss) in enumerate(zip(bars, combo_losses)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{loss/1000:.1f}k', ha='center', fontsize=10, color='darkred')

plt.tight_layout()
output_file = 'D:/期货数据/铜期货监控/daily_backtest/losing_trades_deep_analysis.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {output_file}")
plt.show()

# 7. 最终建议
print("\n" + "="*80)
print("【最终建议】如何避免亏损")
print("="*80)

print(f"""
基于{len(losing_trades)}笔亏损交易的分析：

1. 最危险的GHE区间:
   - GHE ≥0.5: {len(losing_trades[losing_trades['entry_ghe'] >= 0.5])}笔亏损
   - 建议: 当GHE≥0.5时避免开仓

2. 最危险的LPPL-D区间:
   - LPPL-D ≥0.8: {len(losing_trades[losing_trades['entry_lppl'] >= 0.8])}笔亏损
   - 建议: 当LPPL-D≥0.8时避免开仓

3. 应该严格禁止的组合:
   - GHE ≥0.5 且 LPPL-D ≥0.8
   - 历史上这个组合亏损最严重

4. 相对安全的区间（从亏损交易中推断）:
   - GHE <0.3: {len(losing_trades[losing_trades['entry_ghe'] < 0.3])}笔亏损
   - LPPL-D <0.3: {len(losing_trades[losing_trades['entry_lppl'] < 0.3])}笔亏损
   - 但也要注意：安全不代表盈利，只是亏损较少
""")

print("\n" + "="*80)
print("分析完成")
print("="*80)
