# -*- coding: utf-8 -*-
"""
日线数据完整网格优化 - 使用真实交易逻辑
考虑保证金、合约单位、止损止盈、复利计算
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_monitor import calculate_indicators
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

# 初始资金和风险控制
INITIAL_CAPITAL = 100000
MAX_POSITION_RATIO = 0.8
STOP_LOSS_PCT = 0.02

# 沪锡合约规格
CONTRACT_SIZE = 1
MARGIN_RATE = 0.13


def backtest_with_params(df, params):
    """
    使用指定参数进行真实交易回测

    返回: {
        'capital': 最终权益,
        'return': 收益率%,
        'trades': 交易次数,
        'win_rate': 胜率%,
        'win_trades': 盈利笔数,
        'avg_return': 平均收益率%
    }
    """
    # 计算指标
    df = calculate_indicators(df.copy(), params)

    # 回测
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 检查卖出/止损
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 止损
            if current['low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_triggered = True
                exit_reason = '止损'
            # STC止盈
            elif (prev['stc'] > params['STC_SELL_ZONE'] and
                  current['stc'] < prev['stc']):
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'STC止盈'
            # 趋势反转
            elif current['ema_fast'] < current['ema_slow']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = '趋势反转'

            if exit_triggered:
                pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': df.iloc[i]['datetime'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'reason': exit_reason
                })

                position = None
                continue

        # 开仓逻辑
        if position is None:
            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
            ratio_shrinking = current['ratio'] < current['ratio_prev']
            turning_up = current['macd_dif'] > prev['macd_dif']
            is_strong = current['rsi'] > params['RSI_FILTER']

            sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
            ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            buy_signal = sniper_signal or (ema_cross and is_strong)

            if buy_signal:
                # 计算可开仓手数（真实交易逻辑）
                entry_price = current['close']
                margin_per_contract = entry_price * CONTRACT_SIZE * MARGIN_RATE
                available_capital = capital * MAX_POSITION_RATIO
                max_contracts = int(available_capital / margin_per_contract)

                if max_contracts > 0:
                    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)

                    position = {
                        'entry_datetime': df.iloc[i]['datetime'],
                        'entry_price': entry_price,
                        'contracts': max_contracts,
                        'stop_loss': stop_loss_price
                    }

    # 计算统计数据
    if not trades:
        return None

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100

    return {
        'capital': capital,
        'return': total_return,
        'trades': len(trades),
        'win_rate': win_rate,
        'win_trades': len(win_trades),
        'avg_return': total_return / len(trades),
        'trade_list': trades
    }


def grid_optimization(csv_file):
    """完整网格优化 - 真实交易逻辑"""
    print("="*80)
    print("日线数据完整参数网格优化（真实交易逻辑） - 沪锡（SN）")
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
    tested = 0
    last_progress = 0

    for ema_fast, ema_slow, rsi_filter, ratio_trigger, stc_sell in all_combinations:
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

        result = backtest_with_params(df, params)

        if result:
            results.append({
                'params': params,
                'return': result['return'],
                'win_rate': result['win_rate'],
                'trades': result['trades'],
                'capital': result['capital']
            })

        tested += 1
        progress = int(tested / total * 100)

        # 每5%打印一次进度
        if progress >= last_progress + 5:
            print(f"进度: {tested}/{total} ({progress}%)")
            last_progress = progress

    print()

    # 排序结果
    results_by_return = sorted(results, key=lambda x: x['return'], reverse=True)
    results_by_win_rate = sorted(results, key=lambda x: x['win_rate'], reverse=True)

    # 打印TOP20收益率
    print("="*80)
    print(f"完整优化结果（按收益率排序） TOP 20组")
    print("="*80)
    print()
    print(f"{'排名':<6} {'收益率':<12} {'胜率':<10} {'交易数':<8} {'参数'}")
    print("-"*80)

    for i, r in enumerate(results_by_return[:20], 1):
        p = r['params']
        print(f"{i:<6} {r['return']:>8.2f}%     {r['win_rate']:>6.1f}%       "
              f"{r['trades']:<4}   EMA({p['EMA_FAST']},{p['EMA_SLOW']}) "
              f"RSI={p['RSI_FILTER']} RATIO={p['RATIO_TRIGGER']:.2f} STC={p['STC_SELL_ZONE']}")

    print()

    # 打印TOP20胜率
    print("="*80)
    print(f"完整优化结果（按胜率排序） TOP 20组")
    print("="*80)
    print()
    print(f"{'排名':<6} {'胜率':<10} {'收益率':<12} {'交易数':<8} {'参数'}")
    print("-"*80)

    for i, r in enumerate(results_by_win_rate[:20], 1):
        p = r['params']
        print(f"{i:<6} {r['win_rate']:>6.1f}%     {r['return']:>8.2f}%       "
              f"{r['trades']:<4}   EMA({p['EMA_FAST']},{p['EMA_SLOW']}) "
              f"RSI={p['RSI_FILTER']} RATIO={p['RATIO_TRIGGER']:.2f} STC={p['STC_SELL_ZONE']}")

    print()

    # 最优参数总结
    print("="*80)
    print("最优参数总结（真实交易逻辑）")
    print("="*80)
    print()

    best = results_by_return[0]
    print(f"收益率最优:")
    print(f"  收益率: {best['return']:.2f}%")
    print(f"  胜率: {best['win_rate']:.1f}%")
    print(f"  交易次数: {best['trades']}笔")
    print()
    print(f"参数:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")

    print()

    # 保存结果
    output_file = csv_file.replace('.csv', '_optimization_realistic_result.json')
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best_return': {
                'return': best['return'],
                'win_rate': best['win_rate'],
                'trades': best['trades'],
                'capital': best['capital'],
                'params': best['params']
            },
            'top20_by_return': results_by_return[:20],
            'top20_by_win_rate': results_by_win_rate[:20]
        }, f, indent=2, ensure_ascii=False)

    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    csv_file = "SN_沪锡_日线.csv"
    grid_optimization(csv_file)
