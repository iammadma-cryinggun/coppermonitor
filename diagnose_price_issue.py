# -*- coding: utf-8 -*-
"""
诊断Telegram价格显示问题
检查16:00和20:00运行时的数据情况
"""

from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime
import pandas as pd

def diagnose_data_issue():
    """诊断数据问题"""

    print("=" * 80)
    print("Telegram价格显示问题诊断")
    print("=" * 80)

    # 当前时间
    now = datetime.now()
    print(f"\n当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前小时: {now.hour}")

    # 应该显示的4小时K线时间点
    print(f"\n4小时K线时间点:")
    print("  00:00, 04:00, 08:00, 12:00, 16:00, 20:00")

    # 判断当前应该显示哪个时间点的数据
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

    print(f"\n当前时间应该显示: {expected_bar} K线的收盘价")

    # 测试数据获取
    print("\n" + "=" * 80)
    print("测试数据获取（纯碱 SA）")
    print("=" * 80)

    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('SA', days=5)

    if df is None or df.empty:
        print("❌ 数据获取失败")
        return

    print(f"\n✅ 数据获取成功: {len(df)}条")
    print(f"\n最新10条数据:")
    print(df.tail(10)[['datetime', 'open', 'high', 'low', 'close']])

    # 检查最新数据时间
    latest_time = df['datetime'].iloc[-1]
    latest_price = df['close'].iloc[-1]

    print(f"\n最新数据时间: {latest_time}")
    print(f"最新收盘价: {latest_price}")

    # 检查是否有重复的时间戳
    print(f"\n检查时间戳重复:")
    time_counts = df['datetime'].value_counts()
    duplicates = time_counts[time_counts > 1]
    if len(duplicates) > 0:
        print(f"❌ 发现重复时间戳:")
        print(duplicates)
    else:
        print(f"✅ 没有重复时间戳")

    # 检查是否有相同的价格
    print(f"\n检查收盘价重复:")
    price_counts = df['close'].value_counts()
    price_duplicates = price_counts[price_counts > 1]
    if len(price_duplicates) > 0:
        print(f"⚠️  发现相同收盘价 (这是正常的，可能在不同时间出现相同价格)")
        print(f"    例如: {dict(price_duplicates.head(3))}")
    else:
        print(f"✅ 所有收盘价都不同")

    # 问题分析
    print("\n" + "=" * 80)
    print("问题分析")
    print("=" * 80)

    # 计算时间差
    time_diff = now - latest_time
    print(f"\n当前时间与最新数据时间差: {time_diff}")

    # 判断是否是数据延迟
    if time_diff.total_seconds() > 4 * 3600:  # 超过4小时
        print(f"⚠️  数据延迟超过4小时，可能存在数据更新问题")
    elif time_diff.total_seconds() > 1 * 3600:  # 超过1小时
        print(f"⚠️  数据延迟超过1小时，当前K线可能还没收盘")
    else:
        print(f"✅ 数据延迟正常")

    # 判断16:00和20:00的情况
    print(f"\n16:00和20:00运行时的数据情况:")
    print(f"  - 16:00运行时: 最新数据应该是 12:00 K线 (收盘价)")
    print(f"  - 20:00运行时: 最新数据应该是 16:00 K线 (收盘价)")
    print(f"  - 如果两次运行价格相同，可能原因:")
    print(f"    1. 数据更新延迟（akshare数据还没更新）")
    print(f"    2. 12:00和16:00的收盘价碰巧相同")
    print(f"    3. 数据缓存问题（15分钟缓存）")

    # 检查缓存
    print(f"\n当前缓存状态:")
    print(f"  缓存时间: 15分钟")
    print(f"  如果16:00和20:00运行间隔小于15分钟，会使用缓存数据")

    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)
    print("""
1. 检查数据源：
   - akshare的数据可能更新延迟
   - 16:00和20:00运行时，最新的4小时K线可能还没收盘

2. 解决方案：
   - 方案A: 显示数据的时间戳，让用户知道是哪个时间点的数据
   - 方案B: 等待K线收盘后再运行（例如16:30和20:30运行）
   - 方案C: 添加数据延迟提示

3. 修改消息格式：
   在Telegram消息中明确显示数据时间：
   "价格: 1217.0 (截至 20:00)"
   而不是：
   "价格: 1217.0"
    """)

if __name__ == '__main__':
    diagnose_data_issue()
