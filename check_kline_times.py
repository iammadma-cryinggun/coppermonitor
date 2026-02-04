# -*- coding: utf-8 -*-
"""
检查实际4小时K线时间点
"""
from china_futures_fetcher import ChinaFuturesFetcher
import pandas as pd

print("=" * 80)
print("检查4小时K线实际时间点")
print("=" * 80)

fetcher = ChinaFuturesFetcher()

# 检查几个品种
for code in ['SA', 'CU', 'NI']:
    print(f"\n{'='*60}")
    print(f"品种: {code}")
    print('='*60)

    df = fetcher.get_historical_data(code, days=3)
    if df is None or df.empty:
        print("数据为空")
        continue

    # 最新20条
    latest = df.tail(20)
    print("\n最新20条4小时K线:")
    for idx, row in latest.iterrows():
        time_str = str(row['datetime'])
        close_price = row['close']
        print(f"  {time_str}: {close_price:.2f}")

    # 检查时间点
    times = df['datetime'].dt.time.unique()
    print(f"\n所有时间点:")
    for t in sorted(times):
        count = len(df[df['datetime'].dt.time == t])
        print(f"  {t}: {count}条")
