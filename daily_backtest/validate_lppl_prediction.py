# -*- coding: utf-8 -*-
"""
验证LPPL预测的准确性
检查预测临界点2026-01-06前后的实际价格走势
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# LPPL拟合参数
tc_date = pd.Timestamp('2026-01-06')
fit_start = pd.Timestamp('2025-10-21')
fit_end = pd.Timestamp('2026-01-29')

print("="*80)
print("LPPL预测验证")
print("="*80)

# 检查tc前后的价格变化
before_tc = df[df['datetime'] <= tc_date].iloc[-1]
after_tc = df[df['datetime'] > tc_date].iloc[1] if len(df[df['datetime'] > tc_date]) > 0 else None

print(f"\nLPPL预测临界点: {tc_date.date()}")
print(f"\n临界点前最后一个交易日 ({before_tc['datetime'].date()}):")
print(f"  收盘价: {before_tc['close']:,.0f}")

if after_tc is not None:
    print(f"\n临界点后第一个交易日 ({after_tc['datetime'].date()}):")
    print(f"  收盘价: {after_tc['close']:,.0f}")
    print(f"  变化: {(after_tc['close'] - before_tc['close']) / before_tc['close'] * 100:+.2f}%")

# 检查tc后10天、20天的表现
for days in [5, 10, 20]:
    future_date = tc_date + pd.Timedelta(days=days)
    future_data = df[df['datetime'] >= future_date]

    if len(future_data) > 0:
        future_price = future_data.iloc[0]['close']
        pct_change = (future_price - before_tc['close']) / before_tc['close'] * 100

        direction = "[下跌]" if pct_change < 0 else "[上涨]"
        print(f"\n{tc_date.date()}后{days}天 ({future_data.iloc[0]['datetime'].date()}):")
        print(f"  收盘价: {future_price:,.0f}")
        print(f"  累计变化: {pct_change:+.2f}% {direction}")

# 检查最高点和最低点
print(f"\n关键价格点分析:")
window_start = tc_date - pd.Timedelta(days=30)
window_end = tc_date + pd.Timedelta(days=30)
window_data = df[(df['datetime'] >= window_start) & (df['datetime'] <= window_end)]

if len(window_data) > 0:
    max_idx = window_data['close'].idxmax()
    min_idx = window_data['close'].idxmin()

    print(f"\n在 {window_start.date()} ~ {window_end.date()} 窗口内:")
    print(f"  最高点: {window_data.loc[max_idx, 'datetime'].date()} @ {window_data.loc[max_idx, 'close']:,.0f}")
    print(f"  最低点: {window_data.loc[min_idx, 'datetime'].date()} @ {window_data.loc[min_idx, 'close']:,.0f}")

    # 判断tc是否在峰值附近
    if window_data.loc[max_idx, 'datetime'] <= tc_date + pd.Timedelta(days=5):
        print(f"\n  [OK] 预测临界点 {tc_date.date()} 确实在峰值附近!")
    else:
        print(f"\n  [!] 预测临界点 {tc_date.date()} 不在峰值附近")

# 绘图验证
fig, ax = plt.subplots(figsize=(16, 8))

# 绘制价格
plot_data = df[(df['datetime'] >= fit_start) & (df['datetime'] <= df['datetime'].iloc[-1])]
ax.plot(plot_data['datetime'], plot_data['close'], 'o-', linewidth=2, markersize=4, label='收盘价')

# 标注拟合区间
ax.axvspan(fit_start, fit_end, alpha=0.2, color='blue', label='LPPL拟合区间')

# 标注临界点
ax.axvline(tc_date, color='red', linestyle='--', linewidth=2, label=f'LPPL预测临界点 {tc_date.date()}')

# 标注最高点
max_idx = plot_data['close'].idxmax()
ax.scatter(plot_data.loc[max_idx, 'datetime'], plot_data.loc[max_idx, 'close'],
           color='orange', s=200, zorder=5, marker='^', label=f"最高点 {plot_data.loc[max_idx, 'datetime'].date()}")

ax.set_title('LPPL预测验证 - 沪锡价格走势', fontsize=14, fontweight='bold')
ax.set_ylabel('价格 (元/吨)')
ax.set_xlabel('日期')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SN_lppl_validation.png', dpi=150, bbox_inches='tight')
print(f"\n验证图表已保存: SN_lppl_validation.png")

print("\n" + "="*80)
print("验证完成")
print("="*80)
