# -*- coding: utf-8 -*-
"""
检查4小时数据的时间戳分配
确认每根K线代表哪个时间段
"""

import pandas as pd
from datetime import datetime

# 读取数据
df = pd.read_csv('futures_data_4h/沪铜_4hour.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['date'] = df['datetime'].dt.date

print("="*80)
print("4小时数据时间戳分析")
print("="*80)

# 统计小时分布
print("\n小时分布:")
print(df['hour'].value_counts().sort_index())

# 查看连续的几根K线
print("\n\n连续10根K线示例:")
print("="*80)
for i in range(min(10, len(df))):
    row = df.iloc[i]
    print(f"{row['datetime']} | 小时: {row['hour']:2d} | 收盘价: {row['close']:.0f}")

# 查看某一天的所有K线
test_date = df['date'].iloc[5]
print(f"\n\n{test_date} 当天的所有K线:")
print("="*80)
day_klines = df[df['date'] == test_date]
for i, row in day_klines.iterrows():
    print(f"{row['datetime']} | 小时: {row['hour']:2d} | 收盘价: {row['close']:.0f} | 成交量: {row['volume']:.0f}")

print("\n\n推测:")
print("="*80)
print("根据时间戳，推测每根K线代表的时间段:")
print("  00:00 → 代表 20:00-00:00 (夜盘)")
print("  08:00 → 代表 04:00-08:00 (可能无交易)")
print("  12:00 → 代表 08:00-12:00 (日盘)")
print("  20:00 → 代表 16:00-20:00 (日盘收盘后)")
print("\n注意: 这些时间戳与实际期货交易时间不完全对应")
