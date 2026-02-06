# -*- coding: utf-8 -*-
"""
测试线上数据获取的时间戳是否正确
"""

import sys
sys.path.insert(0, 'D:\\期货数据\\铜期货监控')

from china_futures_fetcher import ChinaFuturesFetcher

# 测试数据获取
fetcher = ChinaFuturesFetcher()

print("="*80)
print("测试线上数据获取 - 时间戳验证")
print("="*80)

# 获取沪铜数据
df = fetcher.get_historical_data('CU', days=10)

if df is not None:
    print("\n数据获取成功!")
    print(f"数据量: {len(df)} 条")
    print(f"时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    print("\n小时分布:")
    hour_counts = df['datetime'].dt.hour.value_counts().sort_index()
    for hour, count in hour_counts.items():
        print(f"  {hour:02d}:00 → {count} 条")

    print("\n最新5根K线:")
    print("时间                 | 收盘价   | 成交量")
    print("-"*60)
    for i in range(max(0, len(df)-5), len(df)):
        row = df.iloc[i]
        print(f"{row['datetime']} | {row['close']:8.0f} | {row['volume']:6.0f}")

    print("\n结论:")
    hours = sorted(hour_counts.keys())
    if hours == [1, 9, 13, 21]:
        print("  OK - 时间戳正确: 01:00, 09:00, 13:00, 21:00")
        print("  OK - 与回测数据格式一致")
    else:
        print(f"  WARNING - 时间戳不正确: {hours}")
else:
    print("数据获取失败")
