# -*- coding: utf-8 -*-
"""
测试最终修复后的定时运行逻辑
"""
from datetime import datetime, timedelta

# 实际4小时K线时间点: 00:00, 08:00, 12:00, 20:00
# 运行时间: 整点后30分钟 (0:30, 8:30, 12:30, 20:30)

def get_wait_seconds():
    """计算到下一个4小时K线整点的等待时间"""
    now = datetime.now()
    hour = now.hour

    valid_hours = [0, 8, 12, 20]
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
    return wait_seconds, next_time


print("=" * 80)
print("最终修复: 4小时K线实际时间点")
print("=" * 80)

print("\n实际4小时K线时间点 (从数据中确认):")
print("-" * 80)
print("00:00, 08:00, 12:00, 20:00")
print("\n注意: 没有04:00和16:00")
print("原因: akshare从60分钟数据重采样，跳过了某些时间点")

print("\n" + "=" * 80)
print("运行时间表 (K线收盘后30分钟):")
print("=" * 80)
print("00:30 -> 使用00:00 K线数据")
print("08:30 -> 使用08:00 K线数据")
print("12:30 -> 使用12:00 K线数据")
print("20:30 -> 使用20:00 K线数据")

print("\n" + "=" * 80)
print("问题回顾:")
print("=" * 80)
print("""
原问题: 16:00和20:00推送价格相同

原因分析:
1. 实际K线时间: 00:00, 08:00, 12:00, 20:00 (无16:00)
2. 原运行时间: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
3. 16:00运行时: 最新数据是12:00 (因为无16:00 K线)
4. 20:00运行时: K线未收盘，数据仍是12:00
5. 结果: 两次都显示12:00价格 → 消息相同

解决方案:
- 运行时间改为: 00:30, 08:30, 12:30, 20:30
- 确保K线已收盘后再运行
- 添加数据时间显示
""")

print("\n" + "=" * 80)
print("当前状态:")
print("=" * 80)

wait_seconds, next_time = get_wait_seconds()
wait_minutes = wait_seconds / 60
wait_hours = wait_seconds / 3600

print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"下次运行: {next_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"等待时间: {wait_hours:.1f} 小时 ({wait_minutes:.0f} 分钟)")

# 计算数据时间
if next_time.hour == 0:
    data_time = next_time.replace(minute=0)
elif next_time.hour == 8:
    data_time = next_time.replace(minute=0)
elif next_time.hour == 12:
    data_time = next_time.replace(minute=0)
elif next_time.hour == 20:
    data_time = next_time.replace(minute=0)

print(f"数据时间: {data_time.strftime('%Y-%m-%d %H:%M:%S')}")
