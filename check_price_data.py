# -*- coding: utf-8 -*-
"""
检查价格数据问题
"""

from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime
import pandas as pd

now = datetime.now()
print("=" * 80)
print("Telegram价格显示问题检查")
print("=" * 80)
print(f"\n当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"当前小时: {now.hour}")

# 应该显示的K线
current_hour = now.hour
if current_hour < 4:
    expected_bar = "00:00"
elif current_hour < 8:
    expected_bar = "04:00"
elif current_hour < 12:
    expected_bar = "08:00"
elif current_hour < 16:
    expected_bar = "12:00"
elif current_hour < 20:
    expected_bar = "16:00"
else:
    expected_bar = "20:00"

print(f"\n当前应该显示: {expected_bar} K线收盘价")

# 获取数据
fetcher = ChinaFuturesFetcher()
df = fetcher.get_historical_data('SA', days=5)

print(f"\n数据获取: {len(df)}条")
print(f"\n最新10条数据:")
print(df.tail(10)[['datetime', 'close']])

latest_time = df['datetime'].iloc[-1]
latest_price = df['close'].iloc[-1]
print(f"\n最新数据时间: {latest_time}")
print(f"最新收盘价: {latest_price}")

# 时间差
time_diff = now - latest_time
print(f"\n当前时间与数据时间差: {time_diff}")

# 分析
print("\n" + "=" * 80)
print("问题分析:")
print("=" * 80)
print(f"""
16:00和20:00价格相同的原因:

1. K线收盘时间问题:
   - 4小时K线时间点: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
   - 16:00运行时: 16:00 K线还没收盘，显示的是12:00收盘价
   - 20:00运行时: 20:00 K线还没收盘，显示的是16:00收盘价

2. 数据更新延迟:
   - akshare数据可能有延迟
   - 15分钟缓存可能导致数据未更新

3. 如果12:00和16:00的收盘价碰巧相同，就会看到相同的价格

解决方案:
   - 在Telegram消息中明确显示数据时间
   - 修改消息格式: "价格: 1217.0 (截至 20:00)"
""")
