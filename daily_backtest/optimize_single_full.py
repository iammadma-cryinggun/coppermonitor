# -*- coding: utf-8 -*-
"""
单品种日线数据参数优化（完整网格搜索）
测试品种：沪锡（SN）
完整参数范围
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_monitor import calculate_indicators, check_signals
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import json

def backtest_strategy(df, params):
    """回测策略"""
    # 计算指标
    df = calculate_indicators(df, params)

    # 生成信号
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
    position = None
    trades = []

    for sig in signals:
        if sig['buy_signal'] and position is None:
            position = {
                'entry_price': sig['price'],
                'entry_time': sig['datetime']
            }
        elif sig['sell_signal'] and position is not None:
            exit_price = sig['price']
            entry_price = position['entry_price']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': sig['datetime'],
                'pnl_pct': pnl_pct
            })
            position = None

    if not trades:
        return None

    total_return = sum(t['pnl_pct'] for t in trades)
    win_trades = [t for t in trades if t['pnl_pct'] > 0]
    win_rate = len(win_trades) / len(trades) * 100

    return {
        'trades': len(trades),
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_return': total_return / len(trades)
    }


def grid_optimization(csv_file):
    """完整网格优化"""
    print("="*80)
    print("日线数据完整参数网格优化 - 沪锡（SN）")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 读取数据
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    print(f"数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 条")
    print()

    # 完整参数网格
    param_grid = {
        'EMA_FAST': [3, 5, 7, 10, 12, 15],
        'EMA_SLOW': [10, 12, 15, 18, 20, 22, 25, 30],
        'RSI_FILTER': [30, 35, 40, 45, 50, 55],
        'RATIO_TRIGGER': [1.02, 1.05, 1.08, 1.10, 1.12, 1.15, 1.18, 1.20, 1.22, 1.25],
        'STC_SELL_ZONE': [65, 70, 75, 80, 85]
    }

    # 生成所有组合
    all_combinations = list(product(
        param_grid['EMA_FAST'],
        param_grid['EMA_SLOW'],
        param_grid['RSI_FILTER'],
        param_grid['RATIO_TRIGGER'],
        param_grid['STC_SELL_ZONE']
    ))

    print(f"完整参数网格:")
    print(f"  EMA_FAST: {param_grid['EMA_FAST']}")
    print(f"  EMA_SLOW: {param_grid['EMA_SLOW']}")
    print(f"  RSI_FILTER: {param_grid['RSI_FILTER']}")
    print(f"  RATIO_TRIGGER: {param_grid['RATIO_TRIGGER']}")
    print(f"  STC_SELL_ZONE: {param_grid['STC_SELL_ZONE']}")
    print()
    print(f"参数组合总数: {len(all_combinations)}")
    print()

    # 网格搜索
    results = []
    total = len(all_combinations)

    for i, (ema_fast, ema_slow, rsi_filter, ratio_trigger, stc_sell) in enumerate(all_combinations):
        if ema_fast >= ema_slow:
            continue  # 跳过无效组合

        params = {
            'EMA_FAST': ema_fast,
            'EMA_SLOW': ema_slow,
            'RSI_FILTER': rsi_filter,
            'RATIO_TRIGGER': ratio_trigger,
            'STC_SELL_ZONE': stc_sell,
            'STOP_LOSS_PCT': 0.02
        }

        result = backtest_strategy(df, params)

        if result and result['trades'] >= 5:  # 至少5笔交易
            results.append({
                'params': params,
                'return': result['total_return'],
                'trades': result['trades'],
                'win_rate': result['win_rate']
            })

        # 进度显示
        if (i + 1) % 500 == 0:
            print(f"进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    # 排序结果
    results_by_return = sorted(results, key=lambda x: x['return'], reverse=True)
    results_by_winrate = sorted(results, key=lambda x: x['win_rate'], reverse=True)

    # 显示结果
    print("\n" + "="*80)
    print("完整优化结果（按收益率排名 TOP 20）")
    print("="*80)
    print(f"\n{'排名':<4} {'收益率':<10} {'胜率':<8} {'交易数':<8} {'参数'}")
    print('-'*80)

    for i, r in enumerate(results_by_return[:20], 1):
        p = r['params']
        param_str = f"EMA({p['EMA_FAST']},{p['EMA_SLOW']}) RSI={p['RSI_FILTER']} RATIO={p['RATIO_TRIGGER']:.2f} STC={p['STC_SELL_ZONE']}"
        print(f"{i:<4} {r['return']:>6.2f}%     {r['win_rate']:>5.1f}%     {r['trades']:>4}笔   {param_str}")

    print("\n" + "="*80)
    print("完整优化结果（按胜率排名 TOP 20）")
    print("="*80)
    print(f"\n{'排名':<4} {'胜率':<8} {'收益率':<10} {'交易数':<8} {'参数'}")
    print('-'*80)

    for i, r in enumerate(results_by_winrate[:20], 1):
        p = r['params']
        param_str = f"EMA({p['EMA_FAST']},{p['EMA_SLOW']}) RSI={p['RSI_FILTER']} RATIO={p['RATIO_TRIGGER']:.2f} STC={p['STC_SELL_ZONE']}"
        print(f"{i:<4} {r['win_rate']:>5.1f}%     {r['return']:>6.2f}%     {r['trades']:>4}笔   {param_str}")

    # 最佳参数
    best = results_by_return[0]
    print("\n" + "="*80)
    print("最佳参数组合（最高收益）")
    print("="*80)
    print(f"\n收益率: {best['return']:.2f}%")
    print(f"胜率: {best['win_rate']:.1f}%")
    print(f"交易数: {best['trades']}笔")
    print(f"\n参数:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")

    # 保存结果
    output = {
        'symbol': 'SN',
        'name': '沪锡',
        'data_type': 'daily',
        'optimization_type': 'full_grid',
        'optimization_time': datetime.now().isoformat(),
        'total_combinations': total,
        'valid_results': len(results),
        'best_by_return': results_by_return[:50],
        'best_by_winrate': results_by_winrate[:50]
    }

    output_file = 'daily_backtest/SN_optimization_full_result.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_file}")

    print("\n" + "="*80)
    print(f"完整优化完成！耗时: {datetime.now()}")
    print("="*80)


if __name__ == "__main__":
    csv_file = "SN_沪锡_日线.csv"
    grid_optimization(csv_file)
