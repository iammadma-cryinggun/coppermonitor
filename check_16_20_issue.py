# -*- coding: utf-8 -*-
"""
检查16:00和20:00的数据问题
"""

from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime, timedelta

print("=" * 80)
print("检查16:00和20:00数据问题")
print("=" * 80)

# 模拟16:00运行的情况
print("\n【模拟16:00运行】")
print("理论上应该显示: 12:00的收盘价 (因为16:00 K线还没收盘)")

fetcher = ChinaFuturesFetcher()
df = fetcher.get_historical_data('SA', days=3)

if df is not None and not df.empty:
    # 获取最新的几条数据
    latest = df.tail(5)
    print("\n最新5条数据:")
    print(latest[['datetime', 'open', 'high', 'low', 'close']])

    # 检查16:00的情况
    now_16 = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    print(f"\n如果16:00运行，当前时间是: {now_16.strftime('%Y-%m-%d %H:%M:%S')}")

    # 找到16:00之前的数据
    latest_time = df['datetime'].iloc[-1]
    latest_price = df['close'].iloc[-1]
    print(f"最新数据时间: {latest_time}")
    print(f"最新收盘价: {latest_price}")

    # 模拟20:00运行（4小时后）
    print("\n【模拟20:00运行】")
    print("理论上应该显示: 16:00的收盘价 (因为20:00 K线还没收盘)")

    # 由于数据实时更新，可能16:00的数据在20:00时才更新
    print(f"\n如果20:00运行，当前时间是: {(now_16 + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查是否有16:00的数据
    data_16 = df[df['datetime'].dt.time == datetime.strptime('16:00:00', '%H:%M:%S').time()]
    if len(data_16) > 0:
        print(f"找到16:00数据: {data_16.iloc[0]['datetime']}, 收盘价: {data_16.iloc[0]['close']}")
    else:
        print("未找到16:00的数据 (可能还没更新)")

    data_12 = df[df['datetime'].dt.time == datetime.strptime('12:00:00', '%H:%M:%S').time()]
    if len(data_12) > 0:
        print(f"找到12:00数据: {data_12.iloc[0]['datetime']}, 收盘价: {data_12.iloc[0]['close']}")

print("\n" + "=" * 80)
print("问题分析:")
print("=" * 80)
print("""
16:00和20:00消息相同的原因:

1. 4小时K线收盘时间:
   - 16:00 K线的收盘时间是16:00
   - 20:00 K线的收盘时间是20:00

2. 定时任务运行时间:
   - 16:00准时运行
   - 20:00准时运行

3. 数据获取情况:
   - 16:00运行时 (16:00:00): 16:00 K线正在交易，未收盘
     → 显示的是12:00的收盘价

   - 20:00运行时 (20:00:00): 20:00 K线正在交易，未收盘
     → 显示的是16:00的收盘价

4. 如果12:00和16:00的收盘价相同或接近:
   → 两次推送显示的价格就会一样

5. 数据更新延迟:
   - akshare (新浪财经) 可能有几分钟延迟
   - 16:00运行时，12:00的数据可能刚更新
   - 20:00运行时，16:00的数据可能刚更新或还没更新

6. 真正的问题:
   - 16:00运行 → 显示12:00收盘价 (可能是1217)
   - 20:00运行 → 显示16:00收盘价 (可能也是1217)
   - 两次显示相同价格 → 用户觉得消息重复

解决方案:
   方案1: 在消息中明确标注"数据时间"
         "价格: 1217 (数据时间: 12:00)"

   方案2: 延迟运行时间
         改为16:30和20:30运行，确保K线已收盘

   方案3: 实时价格
         使用实时tick数据而不是4小时K线
""")
