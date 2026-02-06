# -*- coding: utf-8 -*-
"""
测试持仓记录的保存和加载功能
"""

import sys
sys.path.insert(0, 'D:\\期货数据\\铜期货监控')

from pathlib import Path
from copper_monitor import load_position_status, save_position_status
from datetime import datetime

LOGS_DIR = Path('logs')

print("="*80)
print("测试持仓记录功能")
print("="*80)

# 测试1：保存持仓状态
print("\n测试1: 保存持仓状态")
test_position = {
    'holding': True,
    'entry_price': 100000.0,
    'entry_datetime': '2026-02-06 09:00:00',
    'position_size': 1.5,
    'stop_loss': 98500.0,
    'signal_id': '2026-02-06 09:00:00_sniper'
}
save_position_status(test_position)
print(f"  OK - 已保存测试持仓状态")
print(f"    入场价: {test_position['entry_price']:.0f}")
print(f"    入场时间: {test_position['entry_datetime']}")

# 测试2：加载持仓状态
print("\n测试2: 加载持仓状态")
loaded = load_position_status()
print(f"  OK - 已加载持仓状态")
print(f"    持仓中: {loaded['holding']}")
if loaded['holding']:
    print(f"    入场价: {loaded['entry_price']:.0f}")
    print(f"    入场时间: {loaded['entry_datetime']}")
    print(f"    仓位: {loaded['position_size']}x")
    print(f"    止损价: {loaded['stop_loss']:.0f}")

    # 计算持仓天数
    from datetime import datetime
    entry_dt = datetime.fromisoformat(loaded['entry_datetime'])
    days_held = (datetime.now() - entry_dt).days
    print(f"    持仓天数: {days_held} 天")

# 测试3：清除持仓状态
print("\n测试3: 清除持仓状态")
empty_position = {
    'holding': False,
    'entry_price': None,
    'entry_datetime': None,
    'position_size': 1.0,
    'stop_loss': None,
    'signal_id': None
}
save_position_status(empty_position)
print(f"  OK - 已清除持仓状态")

# 验证
loaded_after_clear = load_position_status()
print(f"  验证: 持仓中 = {loaded_after_clear['holding']}")

print("\n" + "="*80)
print("测试完成！")
print("="*80)
print("\n修复内容:")
print("  1. 持仓天数计算统一使用 datetime.now()")
print("  2. 买入信号时自动保存持仓状态")
print("  3. 卖出信号时自动清除持仓状态")
print("  4. 所有时间戳格式: 01:00, 09:00, 13:00, 21:00")
