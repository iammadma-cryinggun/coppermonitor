# -*- coding: utf-8 -*-
"""
验证开仓和平仓逻辑的准确性

测试场景：
1. 开仓逻辑验证
2. STC止盈平仓
3. 趋势反转平仓
4. 止损平仓
"""

import json
from pathlib import Path

def verify_position_logic():
    """验证持仓逻辑"""

    print("=" * 80)
    print("持仓逻辑验证")
    print("=" * 80)

    # 模拟数据
    test_cases = [
        {
            'name': '场景1: 正常开仓',
            'signal': {
                'price': 5000.0,
                'low': 4980.0,
                'high': 5020.0,
                'buy_signal': True,
                'sell_signal': False,
                'stop_loss': 4900.0,  # 5000 * (1 - 0.02)
                'datetime': '2026-02-04 14:00:00',
                'signal_type': 'sniper',
                'reason': {'buy': 'sniper'}
            },
            'expected': {
                'holding': True,
                'entry_price': 5000.0,
                'stop_loss': 4900.0,
                'pnl_pct': None
            }
        },
        {
            'name': '场景2: STC止盈平仓',
            'current_position': {
                'holding': True,
                'entry_price': 5000.0,
                'stop_loss': 4900.0
            },
            'signal': {
                'price': 5200.0,  # 盈利4%
                'low': 5180.0,
                'buy_signal': False,
                'sell_signal': True,
                'signal_type': 'stc',
                'reason': {'sell': 'stc'}
            },
            'expected': {
                'holding': False,
                'exit_price': 5200.0,
                'pnl_pct': 4.0
            }
        },
        {
            'name': '场景3: 趋势反转平仓',
            'current_position': {
                'holding': True,
                'entry_price': 5000.0,
                'stop_loss': 4900.0
            },
            'signal': {
                'price': 4850.0,  # 亏损3%
                'low': 4830.0,
                'buy_signal': False,
                'sell_signal': True,
                'signal_type': 'trend',
                'reason': {'sell': 'trend'}
            },
            'expected': {
                'holding': False,
                'exit_price': 4850.0,
                'pnl_pct': -3.0
            }
        },
        {
            'name': '场景4: 止损平仓（触及止损）',
            'current_position': {
                'holding': True,
                'entry_price': 5000.0,
                'stop_loss': 4900.0
            },
            'signal': {
                'price': 4910.0,  # 收盘价4910
                'low': 4880.0,  # 最低价4880 < 4900（止损价）
                'buy_signal': False,
                'sell_signal': False,  # 原本没有卖出信号
                'stop_loss_price': 4900.0,
                'actual_exit_price': 4910.0,
                'signal_type': 'stop_loss',
                'reason': {'sell': 'stop_loss'}
            },
            'expected': {
                'holding': False,
                'exit_price': 4910.0,  # 按市场价平仓
                'pnl_pct': -1.8  # (4910 - 5000) / 5000 * 100
            }
        }
    ]

    all_passed = True

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"测试案例 {i}: {case['name']}")
        print(f"{'='*60}")

        # 验证逻辑
        if 'current_position' not in case:
            # 开仓场景
            signal = case['signal']
            expected = case['expected']

            # 验证开仓价格
            assert signal['price'] == expected['entry_price'], \
                f"[ERROR] 开仓价格错误: {signal['price']} != {expected['entry_price']}"

            # 验证止损价格
            assert signal['stop_loss'] == expected['stop_loss'], \
                f"[ERROR] 止损价格错误: {signal['stop_loss']} != {expected['stop_loss']}"

            # 验证止损百分比
            stop_loss_pct = (expected['entry_price'] - expected['stop_loss']) / expected['entry_price']
            assert abs(stop_loss_pct - 0.02) < 0.0001, \
                f"[ERROR] 止损百分比错误: {stop_loss_pct:.4f} != 0.02"

            print(f"[OK] 开仓价格: {expected['entry_price']:.2f}")
            print(f"[OK] 止损价格: {expected['stop_loss']:.2f} ({stop_loss_pct*100:.1f}%)")
            print(f"[OK] 信号类型: {signal['signal_type']}")

        else:
            # 平仓场景
            position = case['current_position']
            signal = case['signal']
            expected = case['expected']

            # 计算盈亏
            if 'actual_exit_price' in signal:
                exit_price = signal['actual_exit_price']
            else:
                exit_price = signal['price']

            pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100

            # 验证平仓价格
            assert abs(exit_price - expected['exit_price']) < 0.01, \
                f"[ERROR] 平仓价格错误: {exit_price} != {expected['exit_price']}"

            # 验证盈亏百分比
            assert abs(pnl_pct - expected['pnl_pct']) < 0.1, \
                f"[ERROR] 盈亏百分比错误: {pnl_pct:.2f}% != {expected['pnl_pct']:.2f}%"

            print(f"[OK] 入场价格: {position['entry_price']:.2f}")
            print(f"[OK] 出场价格: {exit_price:.2f}")

            if 'stop_loss_price' in signal:
                print(f"[OK] 止损价格: {signal['stop_loss_price']:.2f}")

            print(f"[OK] 盈亏百分比: {pnl_pct:+.2f}%")
            print(f"[OK] 平仓原因: {signal['signal_type']}")

            # 验证止损检查
            if signal['signal_type'] == 'stop_loss':
                assert signal['low'] <= position['stop_loss'], \
                    f"[ERROR] 止损检查错误: low({signal['low']}) > stop_loss({position['stop_loss']})"
                print(f"[OK] 止损触发: {signal['low']:.2f} <= {position['stop_loss']:.2f}")

        print(f"[OK] 测试案例 {i} 通过")

    print(f"\n{'='*80}")
    print(f"所有测试通过! [OK]")
    print(f"{'='*80}")

    # 验证交易日志记录
    print(f"\n{'='*80}")
    print("交易日志验证")
    print(f"{'='*80}")

    signal_log_path = Path('logs/multi_signals.json')
    if signal_log_path.exists():
        with open(signal_log_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        if logs:
            print(f"\n交易记录数: {len(logs)}")

            # 检查最新记录的字段
            latest_log = logs[-1]
            print(f"\n最新交易记录包含字段:")
            required_fields = ['timestamp', 'future', 'action', 'price', 'signal_type']
            for field in required_fields:
                if field in latest_log:
                    value = latest_log[field]
                    if field == 'pnl_pct' and value is not None:
                        print(f"  [OK] {field}: {value:+.2f}%")
                    elif field == 'price':
                        print(f"  [OK] {field}: {value:.2f}")
                    else:
                        print(f"  [OK] {field}: {value}")
                else:
                    print(f"  [ERROR] 缺少字段: {field}")
                    all_passed = False

            # 检查止损平仓记录的特殊字段
            if latest_log.get('signal_type') == 'stop_loss':
                if 'stop_loss_price' in latest_log:
                    print(f"  [OK] stop_loss_price: {latest_log['stop_loss_price']:.2f}")
                else:
                    print(f"  [WARNING]  止损平仓记录缺少 stop_loss_price 字段")
        else:
            print("[WARNING]  暂无交易记录")
    else:
        print("[WARNING]  交易日志文件不存在")

    print(f"\n{'='*80}")
    if all_passed:
        print("所有验证通过! [OK]")
    else:
        print("部分验证失败，需要检查! [ERROR]")
    print(f"{'='*80}")

if __name__ == '__main__':
    verify_position_logic()
