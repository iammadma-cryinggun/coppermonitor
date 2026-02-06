# -*- coding: utf-8 -*-
"""
检查并修复持仓记录中的时间戳问题
"""

import json
from pathlib import Path
from datetime import datetime

LOGS_DIR = Path('logs')
position_file = LOGS_DIR / 'position_status.json'

print("="*80)
print("持仓记录时间检查")
print("="*80)

if position_file.exists():
    with open(position_file, 'r', encoding='utf-8') as f:
        position = json.load(f)

    print(f"\n当前持仓状态:")
    print(f"  持仓中: {position['holding']}")

    if position['holding']:
        entry_dt = position['entry_datetime']
        print(f"  入场时间: {entry_dt}")
        print(f"  入场价格: {position['entry_price']}")
        print(f"  仓位: {position['position_size']}x")
        print(f"  止损价: {position['stop_loss']}")
        print(f"  信号ID: {position['signal_id']}")

        # 检查时间格式
        if entry_dt:
            try:
                dt = datetime.fromisoformat(entry_dt)
                hour = dt.hour
                print(f"\n  时间解析成功: {dt}")
                print(f"  小时: {hour:02d}")

                # 检查是否是正确的时间戳格式（01:00, 09:00, 13:00, 21:00）
                if hour in [1, 9, 13, 21]:
                    print(f"  OK - 时间戳格式正确")
                else:
                    print(f"  WARNING - 时间戳格式可能不正确（应该是 01:00, 09:00, 13:00, 21:00）")
                    print(f"  建议: 修正入场时间或重新记录")

                # 计算持仓天数
                days_held = (datetime.now() - dt).days
                print(f"  持仓天数: {days_held} 天")

            except Exception as e:
                print(f"  ERROR - 时间解析失败: {e}")
else:
    print("\n  没有持仓记录文件")

print("\n" + "="*80)
print("检查完成")
print("="*80)
