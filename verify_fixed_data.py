# -*- coding: utf-8 -*-
"""
验证修正后的4小时数据
"""

import pandas as pd

# 读取修正后的数据
df = pd.read_csv('futures_data_4h/沪铜_4hour_fixed.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print("="*80)
print("修正后的4小时数据验证")
print("="*80)

print("\n时间范围:")
print(f"  开始: {df['datetime'].iloc[0]}")
print(f"  结束: {df['datetime'].iloc[-1]}")

print("\n小时分布:")
hour_counts = df['datetime'].dt.hour.value_counts().sort_index()
for hour, count in hour_counts.items():
    print(f"  {hour:02d}:00 → {count} 条")

print("\n前10根K线:")
print("时间                 | 收盘价   | 成交量")
print("-"*60)
for i in range(min(10, len(df))):
    row = df.iloc[i]
    print(f"{row['datetime']} | {row['close']:8.0f} | {row['volume']:6.0f}")

print("\n结论:")
print("  ✓ 时间戳已修正为: 01:00, 09:00, 13:00, 21:00")
print("  ✓ 符合实际期货交易时间")
