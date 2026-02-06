# -*- coding: utf-8 -*-
"""
沪锡10年数据全景诊断
识别LPPL超指数增长区间
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_csv('SN_沪锡_日线_10年.csv')
df = df.rename(columns={
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low',
    '收盘价': 'close',
    '成交量': 'volume',
    '持仓量': 'hold'
})
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
print(f"数据量: {len(df)} 条")
print(f"价格区间: {df['close'].min():,.0f} ~ {df['close'].max():,.0f}")
print(f"对数价格区间: {np.log(df['close'].min()):.2f} ~ {np.log(df['close'].max()):.2f}")

# 创建图表
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# 1. 原始价格图
ax1 = axes[0]
ax1.plot(df['datetime'], df['close'], linewidth=1)
ax1.set_title('沪锡10年价格走势 (2016-2026)', fontsize=14, fontweight='bold')
ax1.set_ylabel('价格 (元/吨)')
ax1.grid(True, alpha=0.3)
ax1.legend(['收盘价'], loc='upper left')

# 标注最高点和最低点
max_idx = df['close'].idxmax()
min_idx = df['close'].idxmin()
ax1.scatter(df['datetime'].iloc[max_idx], df['close'].iloc[max_idx],
           color='red', s=100, zorder=5, label=f"最高 {df['close'].iloc[max_idx]:,.0f}")
ax1.scatter(df['datetime'].iloc[min_idx], df['close'].iloc[min_idx],
           color='green', s=100, zorder=5, label=f"最低 {df['close'].iloc[min_idx]:,.0f}")
ax1.legend()

# 2. 对数价格图 - LPPL的核心视图
ax2 = axes[1]
log_prices = np.log(df['close'].values)
ax2.plot(df['datetime'], log_prices, linewidth=1, color='purple')
ax2.set_title('对数价格走势 - LPPL泡沫识别', fontsize=14, fontweight='bold')
ax2.set_ylabel('ln(价格)')
ax2.grid(True, alpha=0.3)

# 标注可能的关键泡沫区间（大幅上涨后）
# 2021-2022年大涨
ax2.axvspan(pd.Timestamp('2021-01-01'), pd.Timestamp('2022-01-01'),
             alpha=0.2, color='red', label='2021-2022大涨期')
# 2024-2026年大涨
ax2.axvspan(pd.Timestamp('2024-01-01'), df['datetime'].iloc[-1],
             alpha=0.2, color='yellow', label='2024-2026当前涨期')
ax2.legend()

# 3. 计算收益率滚动窗口，识别超指数增长
ax3 = axes[2]
# 计算60日和120日滚动收益率
df['return_60'] = df['close'].pct_change(60)
df['return_120'] = df['close'].pct_change(120)

ax3.plot(df['datetime'], df['return_60'], label='60日收益率', linewidth=1)
ax3.plot(df['datetime'], df['return_120'], label='120日收益率', linewidth=1)
ax3.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='5%阈值')
ax3.axhline(y=0.10, color='r', linestyle='--', alpha=0.5, label='10%阈值')
ax3.set_title('滚动收益率 - 识别超指数增长', fontsize=14, fontweight='bold')
ax3.set_ylabel('收益率')
ax3.set_xlabel('日期')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig('SN_10year_diagnosis.png', dpi=150, bbox_inches='tight')
print("\n图表已保存: SN_10year_diagnosis.png")
plt.close()

# 识别最佳拟合区间
print("\n关键发现:")

# 找出最大的几次上涨
df['pct_change'] = df['close'].pct_change(20)
major_rises = df[df['pct_change'] > 0.2].sort_values('pct_change', ascending=False)

if len(major_rises) > 0:
    print("\n20日涨幅超过20%的日期:")
    for idx, row in major_rises.head(10).iterrows():
        print(f"  {row['datetime'].date()} : {row['pct_change']*100:.1f}%")

# 识别可能的泡沫区间（连续上涨且加速度增加）
print("\n可能的泡沫区间识别:")
bubble_candidates = []

# 方法1: 查找连续3个月涨幅超过50%的区间
for i in range(60, len(df)-60):
    window_start = df.iloc[i]['datetime']
    window_end = df.iloc[i+60]['datetime']
    start_price = df.iloc[i]['close']
    end_price = df.iloc[i+60]['close']
    pct_change = (end_price - start_price) / start_price

    if pct_change > 0.5:  # 60天涨幅超过50%
        bubble_candidates.append({
            'start': window_start,
            'end': window_end,
            'return': pct_change,
            'start_idx': i,
            'end_idx': i+60
        })

# 合并相邻的区间
merged_candidates = []
if bubble_candidates:
    current = bubble_candidates[0]
    for candidate in bubble_candidates[1:]:
        if candidate['start_idx'] <= current['end_idx'] + 30:  # 相邻或接近
            current['end'] = candidate['end']
            current['end_idx'] = candidate['end_idx']
            current['return'] = (df.loc[current['end_idx'], 'close'] - df.loc[current['start_idx'], 'close']) / df.loc[current['start_idx'], 'close']
        else:
            merged_candidates.append(current)
            current = candidate
    merged_candidates.append(current)

    print(f"\n找到 {len(merged_candidates)} 个可能的泡沫区间:")
    for i, bc in enumerate(merged_candidates, 1):
        duration = (bc['end'] - bc['start']).days
        print(f"  {i}. {bc['start'].date()} ~ {bc['end'].date()} ({duration}天)")
        print(f"     涨幅: {bc['return']*100:.1f}%")

# 推荐最佳拟合区间
print("\n推荐的LPPL拟合区间:")
if merged_candidates:
    best = merged_candidates[-1]  # 最近的泡沫区间
    print(f"  推荐: {best['start'].date()} ~ {best['end'].date()}")
    print(f"  理由: 最近期的泡沫，对当前交易最有参考意义")
    print(f"  涨幅: {best['return']*100:.1f}%")
    recommended_start = best['start_idx']
    recommended_size = best['end_idx'] - best['start_idx']
else:
    print("  推荐: 2024-01-01 ~ 当前")
    recommended_start = df[df['datetime'] >= pd.Timestamp('2024-01-01')].index[0]
    recommended_size = len(df) - recommended_start

print(f"  窗口大小: {recommended_size} 天")

print("\n" + "="*80)
print("下一步: 运行简化版LPPL拟合")
print("="*80)
