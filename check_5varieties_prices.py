# -*- coding: utf-8 -*-
"""
检查5个品种的价格数据（用户截图中的品种）
检查12:00和16:00的价格是否真的相同
"""

from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime
import pandas as pd

print("=" * 80)
print("检查5个品种的价格数据（12:00 vs 16:00）")
print("=" * 80)

# 用户截图中的5个品种
varieties = {
    'NI': '沪镍',
    'V': 'PVC',
    'M': '豆粕',
    'SN': '沪锡',
    'FG': '玻璃'
}

fetcher = ChinaFuturesFetcher()

print(f"\n当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

results = []

for code, name in varieties.items():
    print(f"\n{'=' * 60}")
    print(f"{name} ({code})")
    print('=' * 60)

    df = fetcher.get_historical_data(code, days=3)

    if df is None or df.empty:
        print("[X] 数据为空")
        continue

    # 获取最新数据
    latest = df.tail(8)
    print(f"\n最新8条数据:")
    for idx, row in latest.iterrows():
        print(f"  {row['datetime']}: {row['close']:.2f}")

    # 提取12:00和16:00的数据
    from datetime import time as dt_time

    data_12 = df[df['datetime'].dt.time == dt_time(12, 0)]
    data_16 = df[df['datetime'].dt.time == dt_time(16, 0)]

    result = {'code': code, 'name': name}

    print(f"\n12:00数据:")
    if len(data_12) > 0:
        latest_12 = data_12.iloc[-1]
        print(f"  时间: {latest_12['datetime']}")
        print(f"  收盘价: {latest_12['close']:.2f}")
        result['price_12'] = latest_12['close']
    else:
        print("  [X] 未找到12:00数据")
        result['price_12'] = None

    print(f"\n16:00数据:")
    if len(data_16) > 0:
        latest_16 = data_16.iloc[-1]
        print(f"  时间: {latest_16['datetime']}")
        print(f"  收盘价: {latest_16['close']:.2f}")
        result['price_16'] = latest_16['close']
    else:
        print("  [X] 未找到16:00数据")
        result['price_16'] = None

    # 检查最新数据时间
    latest_time = df['datetime'].iloc[-1]
    latest_price = df['close'].iloc[-1]
    print(f"\n最新数据:")
    print(f"  时间: {latest_time}")
    print(f"  收盘价: {latest_price:.2f}")
    result['latest_time'] = latest_time
    result['latest_price'] = latest_price

    results.append(result)

print("\n" + "=" * 80)
print("汇总结果")
print("=" * 80)
print(f"{'品种':<10} {'12:00价格':<15} {'16:00价格':<15} {'最新价格':<15} {'最新时间'}")
print("-" * 80)
for r in results:
    price_12 = f"{r['price_12']:.2f}" if r['price_12'] else "N/A"
    price_16 = f"{r['price_16']:.2f}" if r['price_16'] else "N/A"
    latest_price = f"{r['latest_price']:.2f}"
    latest_time = str(r['latest_time'])
    print(f"{r['name']:<10} {price_12:<15} {price_16:<15} {latest_price:<15} {latest_time}")

print("\n" + "=" * 80)
print("分析结论:")
print("=" * 80)
print("""
重要发现：
1. 4小时K线时间点: 00:00, 04:00, 08:00, 12:00, 20:00
2. 没有16:00的K线! (16:00不是4小时K线的收盘时间)
3. 最新数据是20:00

实际情况：
- 16:00运行时: 最新可用数据是12:00收盘 (因为16:00 K线不存在)
- 20:00运行时: 最新可用数据是12:00收盘 (因为20:00 K线未收盘)
→ 两次都显示12:00的价格 → 价格相同!

解决方案：
方案1: 延迟到20:30运行，确保20:00 K线已收盘
方案2: 使用12:00和20:00的数据，明确标注数据时间
方案3: 添加延迟提示："数据延迟: X小时"
""")
