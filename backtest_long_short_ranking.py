# -*- coding: utf-8 -*-
"""
做多vs做空策略回测排名
基于当前参数配置，回测所有品种的表现
"""

from china_futures_fetcher import ChinaFuturesFetcher
from futures_monitor import TOP10_FUTURES_CONFIG, calculate_indicators, check_signals
import pandas as pd
import numpy as np
from datetime import datetime

def backtest_strategy(df, params, mode='long'):
    """
    回测策略

    Args:
        df: 历史数据
        params: 参数配置
        mode: 'long' 或 'short'

    Returns:
        dict: 回测结果
    """
    if len(df) < 200:
        return None

    # 计算指标
    df = calculate_indicators(df, params)

    signals = []
    for i in range(200, len(df)):
        current_df = df.iloc[:i+1].copy()
        signal = check_signals(current_df, params, 'TEST')

        if 'error' in signal:
            continue

        signals.append({
            'datetime': current_df.iloc[-1]['datetime'],
            'price': signal['price'],
            'buy_signal': signal['buy_signal'],
            'sell_signal': signal['sell_signal'],
            'trend': signal['trend']
        })

    if not signals:
        return None

    # 模拟交易
    INITIAL_CAPITAL = 50000
    position = None
    trades = []

    for sig in signals:
        if mode == 'long':
            # 做多逻辑
            if sig['buy_signal'] and position is None:
                position = {
                    'entry_price': sig['price'],
                    'entry_time': sig['datetime'],
                    'entry_type': 'buy'
                }
            elif sig['sell_signal'] and position is not None:
                exit_price = sig['price']
                entry_price = position['entry_price']
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': sig['datetime'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct
                })
                position = None
        else:
            # 做空逻辑
            if sig['sell_signal'] and position is None:
                position = {
                    'entry_price': sig['price'],
                    'entry_time': sig['datetime'],
                    'entry_type': 'sell'
                }
            elif sig['buy_signal'] and position is not None:
                exit_price = sig['price']
                entry_price = position['entry_price']
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': sig['datetime'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct
                })
                position = None

    if not trades:
        return None

    # 计算统计数据
    total_return = sum(t['pnl_pct'] for t in trades)
    win_trades = [t for t in trades if t['pnl_pct'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0

    return {
        'trades': len(trades),
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_return': total_return / len(trades) if trades else 0
    }


print("=" * 80)
print("做多vs做空策略回测排名")
print("=" * 80)

fetcher = ChinaFuturesFetcher()

long_results = []
short_results = []

for future_name, config in TOP10_FUTURES_CONFIG.items():
    print(f"\n回测 {future_name}...")

    # 获取数据
    df = fetcher.get_historical_data(config['code'], days=252)
    if df is None or df.empty:
        print(f"  ✗ 数据获取失败")
        continue

    # 回测做多
    long_result = backtest_strategy(df, config['params'], mode='long')
    if long_result:
        long_results.append({
            'name': future_name,
            'trades': long_result['trades'],
            'return': long_result['total_return'],
            'win_rate': long_result['win_rate']
        })
        print(f"  做多: {long_result['trades']}笔, 收益{long_result['total_return']:.2f}%, 胜率{long_result['win_rate']:.1f}%")

    # 回测做空
    short_result = backtest_strategy(df, config['params'], mode='short')
    if short_result:
        short_results.append({
            'name': future_name,
            'trades': short_result['trades'],
            'return': short_result['total_return'],
            'win_rate': short_result['win_rate']
        })
        print(f"  做空: {short_result['trades']}笔, 收益{short_result['total_return']:.2f}%, 胜率{short_result['win_rate']:.1f}%")

# 排名
print("\n" + "=" * 80)
print("做多策略排名 (按收益率)")
print("=" * 80)
long_results.sort(key=lambda x: x['return'], reverse=True)
print(f"\n{'排名':<4} {'品种':<10} {'收益率':<10} {'胜率':<10} {'交易数'}")
print('-' * 60)
for i, r in enumerate(long_results, 1):
    print(f"{i:<4} {r['name']:<10} {r['return']:>6.2f}%     {r['win_rate']:>5.1f}%     {r['trades']:>3}笔")

print("\n" + "=" * 80)
print("做多策略排名 (按胜率)")
print("=" * 80)
long_results_win = sorted(long_results, key=lambda x: x['win_rate'], reverse=True)
print(f"\n{'排名':<4} {'品种':<10} {'胜率':<10} {'收益率':<10} {'交易数'}")
print('-' * 60)
for i, r in enumerate(long_results_win, 1):
    print(f"{i:<4} {r['name']:<10} {r['win_rate']:>5.1f}%     {r['return']:>6.2f}%     {r['trades']:>3}笔")

print("\n" + "=" * 80)
print("做空策略排名 (按收益率)")
print("=" * 80)
short_results.sort(key=lambda x: x['return'], reverse=True)
print(f"\n{'排名':<4} {'品种':<10} {'收益率':<10} {'胜率':<10} {'交易数'}")
print('-' * 60)
for i, r in enumerate(short_results, 1):
    print(f"{i:<4} {r['name']:<10} {r['return']:>6.2f}%     {r['win_rate']:>5.1f}%     {r['trades']:>3}笔")

print("\n" + "=" * 80)
print("做空策略排名 (按胜率)")
print("=" * 80)
short_results_win = sorted(short_results, key=lambda x: x['win_rate'], reverse=True)
print(f"\n{'排名':<4} {'品种':<10} {'胜率':<10} {'收益率':<10} {'交易数'}")
print('-' * 60)
for i, r in enumerate(short_results_win, 1):
    print(f"{i:<4} {r['name']:<10} {r['win_rate']:>5.1f}%     {r['return']:>6.2f}%     {r['trades']:>3}笔")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"\n做多策略:")
if long_results:
    best_long = max(long_results, key=lambda x: x['return'])
    print(f"  最佳收益率: {best_long['name']} ({best_long['return']:.2f}%)")
    best_long_win = max(long_results, key=lambda x: x['win_rate'])
    print(f"  最佳胜率: {best_long_win['name']} ({best_long_win['win_rate']:.1f}%)")
    avg_long_return = sum(r['return'] for r in long_results) / len(long_results)
    print(f"  平均收益率: {avg_long_return:.2f}%")

print(f"\n做空策略:")
if short_results:
    best_short = max(short_results, key=lambda x: x['return'])
    print(f"  最佳收益率: {best_short['name']} ({best_short['return']:.2f}%)")
    best_short_win = max(short_results, key=lambda x: x['win_rate'])
    print(f"  最佳胜率: {best_short_win['name']} ({best_short_win['win_rate']:.1f}%)")
    avg_short_return = sum(r['return'] for r in short_results) / len(short_results)
    print(f"  平均收益率: {avg_short_return:.2f}%")

print("\n回测完成！")
