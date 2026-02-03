# -*- coding: utf-8 -*-
"""
===================================
实盘跟踪分析
===================================
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# 文件路径
SIGNAL_LOG_PATH = 'D:/daily_stock_analysis/logs/signal_log.json'
TRACKING_PATH = 'D:/daily_stock_analysis/logs/performance_tracking.csv'
POSITION_PATH = 'D:/daily_stock_analysis/logs/position_status.json'

print("=" * 80)
print("沪铜策略 - 实盘跟踪分析")
print("=" * 80)

# 1. 信号统计
print("\n[1] 信号统计")
print("-" * 80)

signal_path = Path(SIGNAL_LOG_PATH)
if signal_path.exists():
    with open(signal_path, 'r', encoding='utf-8') as f:
        signals = json.load(f)

    total = len(signals)
    executed = len([s for s in signals if s['status'] == 'executed'])
    pending = len([s for s in signals if s['status'] == 'pending'])
    ignored = len([s for s in signals if s['status'] == 'ignored'])

    print(f"总信号数: {total}")
    print(f"已执行: {executed} ({executed/total*100:.1f}%)" if total > 0 else "已执行: 0")
    print(f"待处理: {pending}")
    print(f"已忽略: {ignored}")

    # 信号类型统计
    buy_signals = [s for s in signals if s['signal'].get('buy_signal', False)]
    sell_signals = [s for s in signals if s['signal'].get('sell_signal', False)]

    print(f"\n买入信号: {len(buy_signals)}")
    print(f"卖出信号: {len(sell_signals)}")

    # 最近5个信号
    print(f"\n最近5个信号:")
    for s in signals[-5:]:
        status_mark = {
            'pending': '[待]',
            'executed': '[OK]',
            'ignored': '[X]'
        }.get(s['status'], '?')

        signal_type = s['signal'].get('signal_type', 'none')
        position_size = s['signal'].get('position_size', 0)

        print(f"  {status_mark} {s['signal_datetime']} | {signal_type} | "
              f"仓位: {position_size:.1f}x | 状态: {s['status']}")
else:
    print("暂无信号记录")

# 2. 持仓状态
print("\n[2] 当前持仓")
print("-" * 80)

position_path = Path(POSITION_PATH)
if position_path.exists():
    with open(position_path, 'r', encoding='utf-8') as f:
        position = json.load(f)

    if position['holding']:
        print(f"持仓状态: 有持仓")
        print(f"  入场价: {position['entry_price']:.0f}")
        print(f"  入场时间: {position['entry_datetime']}")
        print(f"  仓位: {position['position_size']:.1f}x")
        print(f"  止损价: {position['stop_loss']:.0f}")

        # 计算持有天数
        entry_dt = datetime.fromisoformat(position['entry_datetime'])
        days_held = (datetime.now() - entry_dt).days
        print(f"  持有天数: {days_held} 天")
    else:
        print("持仓状态: 空仓")
else:
    print("持仓状态: 无记录")

# 3. 监控记录统计
print("\n[3] 监控记录")
print("-" * 80)

tracking_path = Path(TRACKING_PATH)
if tracking_path.exists():
    df = pd.read_csv(tracking_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"监控次数: {len(df)}")
    print(f"时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 最近记录
    print(f"\n最近5次监控:")
    for i, row in df.tail(5).iterrows():
        holding_mark = "[持仓]" if row['holding'] else "[空仓]"
        buy_mark = "[买]" if row['buy_signal'] else "[ ]"
        sell_mark = "[卖]" if row['sell_signal'] else "[ ]"

        print(f"  {holding_mark} {row['datetime']} | 价格: {row['price']:.0f} | "
              f"Ratio: {row['ratio']:.2f} | RSI: {row['rsi']:.1f} {buy_mark}{sell_mark}")
else:
    print("暂无监控记录")

# 4. 策略表现总结
print("\n[4] 策略表现总结")
print("-" * 80)

if signal_path.exists() and executed > 0:
    print("\n实盘执行统计:")
    print(f"  执行率: {executed}/{total} ({executed/total*100:.1f}%)")

    # 计算平均盈亏（如果有实际成交记录）
    executed_signals = [s for s in signals if s['status'] == 'executed' and s.get('actual_price')]

    if executed_signals:
        print(f"\n已执行信号详情:")
        for s in executed_signals:
            signal_type = s['signal'].get('signal_type', 'none')
            actual_price = s.get('actual_price', 0)
            action = s.get('actual_action', 'unknown')

            print(f"  {s['signal_datetime']} | {signal_type} | "
                  f"操作: {action} | 成交价: {actual_price:.0f} | "
                  f"备注: {s.get('notes', '')}")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
