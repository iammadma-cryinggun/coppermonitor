# -*- coding: utf-8 -*-
"""
测试修复后的定时运行逻辑
验证新的运行时间是否正确
"""

from datetime import datetime, timedelta

RUN_INTERVAL_HOURS = 4

def get_wait_seconds():
    """计算到下一个4小时整点+30分钟的等待时间

    4小时K线时间点: 00:00, 04:00, 08:00, 12:00, 20:00
    运行时间延迟30分钟: 00:30, 04:30, 08:30, 12:30, 20:30
    注意: 跳过16:00 (没有16:00 K线)
    """
    now = datetime.now()
    hour = now.hour

    # 4小时K线的实际时间点: 0, 4, 8, 12, 20 (跳过16)
    valid_hours = [0, 4, 8, 12, 20]

    # 找到下一个有效的运行时间
    next_hour = None
    for valid_hour in valid_hours:
        if valid_hour > hour:
            next_hour = valid_hour
            break

    # 如果没找到（已过20:00），下一个是0:00（次日）
    if next_hour is None:
        next_hour = 0

    # 添加30分钟延迟，确保K线已收盘
    next_time = now.replace(hour=next_hour, minute=30, second=0, microsecond=0)

    # 如果是0点，日期+1
    if next_hour == 0:
        next_time += timedelta(days=1)

    wait_seconds = (next_time - now).total_seconds()
    return wait_seconds, next_time


print("=" * 80)
print("测试修复后的定时运行逻辑")
print("=" * 80)

# 测试不同时间点
test_times = [
    "00:00", "01:00", "04:00", "08:00", "12:00",
    "15:00", "16:00", "19:00", "20:00", "23:00"
]

print("\n当前时间 -> 下次运行时间:")
print("-" * 80)

for test_time in test_times:
    hour, minute = map(int, test_time.split(':'))
    now = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)

    # 模拟get_wait_seconds逻辑（跳过16:00）
    valid_hours = [0, 4, 8, 12, 20]
    next_hour = None
    for valid_hour in valid_hours:
        if valid_hour > hour:
            next_hour = valid_hour
            break
    if next_hour is None:
        next_hour = 0

    next_time = now.replace(hour=next_hour, minute=30, second=0, microsecond=0)
    if next_hour == 0:
        next_time += timedelta(days=1)

    wait_seconds = (next_time - now).total_seconds()
    wait_minutes = wait_seconds / 60

    print(f"{test_time} -> {next_time.strftime('%Y-%m-%d %H:%M')} (等待 {wait_minutes:.0f} 分钟)")

print("\n" + "=" * 80)
print("新的运行时间表:")
print("=" * 80)
print("00:30 (数据时间: 00:00)")
print("04:30 (数据时间: 04:00)")
print("08:30 (数据时间: 08:00)")
print("12:30 (数据时间: 12:00)")
print("20:30 (数据时间: 20:00)")
print("\n注意: 没有16:30运行，因为没有16:00 K线")

print("\n" + "=" * 80)
print("当前状态:")
print("=" * 80)

wait_seconds, next_time = get_wait_seconds()
wait_minutes = wait_seconds / 60
wait_hours = wait_seconds / 3600

print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"下次运行: {next_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"等待时间: {wait_hours:.1f} 小时 ({wait_minutes:.0f} 分钟)")

print("\n" + "=" * 80)
print("修复说明:")
print("=" * 80)
print("""
核心修复：
1. 运行时间从整点改为整点后30分钟
   - 16:00 -> 12:30 (使用12:00数据)
   - 20:00 -> 20:30 (使用20:00数据)
   - 两次运行将显示不同时间的价格

2. 添加数据时间显示
   - 报告时间: 实际运行时间
   - 数据时间: K线收盘时间
   - 用户可以清楚知道数据是何时

3. 缩短缓存时间到5分钟
   - 确保获取最新数据
   - 避免使用过期数据

4. 显示完整时间戳（秒）
   - 用户可以看到精确运行时间
   - 便于追踪和调试
""")
