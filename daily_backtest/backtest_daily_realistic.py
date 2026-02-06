# -*- coding: utf-8 -*-
"""
日线数据真实交易回测
考虑保证金、合约单位、止损止盈等真实交易规则
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_monitor import TOP7_FUTURES_CONFIG, calculate_indicators
import pandas as pd
import numpy as np
from datetime import datetime

# 固定技术参数
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50

# 初始资金和风险控制
INITIAL_CAPITAL = 100000
MAX_POSITION_RATIO = 0.8  # 最大仓位80%
STOP_LOSS_PCT = 0.02  # 2%止损


def backtest_realistic(csv_file, config, future_name):
    """
    真实期货回测

    考虑：
    1. 真实保证金
    2. 合约单位
    3. 止损止盈
    4. 资金管理
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    if len(df) < 200:
        return None

    # 获取合约规格
    contract_size = config['contract_size']
    margin_rate = config['margin_rate']
    params = config['params']

    # 计算指标
    df = calculate_indicators(df, params)

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
                pnl = (exit_price - position['entry_price']) * position['contracts'] * contract_size
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

        # 开仓逻辑（只有在没有持仓时）
        if position is None:
            # 检查买入信号
            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
            ratio_shrinking = current['ratio'] < current['ratio_prev']
            turning_up = current['macd_dif'] > prev['macd_dif']
            is_strong = current['rsi'] > params['RSI_FILTER']

            sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong

            # EMA金叉
            ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

            buy_signal = sniper_signal or (ema_cross and is_strong)

            if buy_signal:
                # 计算可开仓数量
                entry_price = current['close']
                margin_per_contract = entry_price * contract_size * margin_rate

                # 最大可用资金
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
        'trades': len(trades),
        'final_capital': capital,
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_return': total_return / len(trades) if trades else 0,
        'trade_details': trades
    }


def main():
    """主函数"""
    print("=" * 80)
    print("日线数据真实交易回测（考虑保证金、合约单位、止损止盈）")
    print("=" * 80)
    print(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"初始资金: {INITIAL_CAPITAL:,} 元")
    print(f"止损幅度: {STOP_LOSS_PCT*100}%")
    print(f"最大仓位: {MAX_POSITION_RATIO*100}%")
    print()

    results = []

    # 回测每个品种
    for future_name, config in TOP7_FUTURES_CONFIG.items():
        code = config['code']
        csv_file = f"{code}_{future_name}_日线.csv"

        print(f"{'='*60}")
        print(f"回测 {future_name} ({code})...")
        print('='*60)

        print(f"  合约单位: {config['contract_size']}")
        print(f"  保证金率: {config['margin_rate']*100}%")
        print(f"  参数: EMA({config['params']['EMA_FAST']},{config['params']['EMA_SLOW']}) "
              f"RSI={config['params']['RSI_FILTER']} RATIO={config['params']['RATIO_TRIGGER']} "
              f"STC={config['params']['STC_SELL_ZONE']}")

        # 回测
        result = backtest_realistic(csv_file, config, future_name)

        if result:
            results.append({
                'name': future_name,
                'code': code,
                'trades': result['trades'],
                'final_capital': result['final_capital'],
                'return': result['total_return'],
                'win_rate': result['win_rate'],
                'avg_return': result['avg_return']
            })

            print(f"\n  [结果]")
            print(f"    交易次数: {result['trades']}笔")
            print(f"    最终权益: {result['final_capital']:,.2f} 元")
            print(f"    总收益率: {result['total_return']:.2f}%")
            print(f"    胜率: {result['win_rate']:.1f}%")
            print(f"    平均收益: {result['avg_return']:.2f}%")

            # 显示最近几笔交易
            print(f"\n  最近5笔交易:")
            for trade in result['trade_details'][-5:]:
                pnl_str = f"+{trade['pnl']:,.0f}" if trade['pnl'] > 0 else f"{trade['pnl']:,.0f}"
                print(f"    {trade['entry_datetime'].date()} -> {trade['exit_datetime'].date()} "
                      f"{trade['contracts']}手 {trade['entry_price']:.0f}->{trade['exit_price']:.0f} "
                      f"{pnl_str}元 ({trade['pnl_pct']:+.2f}%) [{trade['reason']}]")
        else:
            print(f"  [FAIL] 回测失败")

        print()

    # 排名
    if results:
        print("\n" + "=" * 80)
        print("真实交易回测排名（按收益率）")
        print("=" * 80)
        results_sorted = sorted(results, key=lambda x: x['return'], reverse=True)
        print(f"\n{'排名':<4} {'品种':<10} {'收益率':<12} {'胜率':<10} {'交易数':<8} {'最终权益'}")
        print('-'*80)
        for i, r in enumerate(results_sorted, 1):
            print(f"{i:<4} {r['name']:<10} {r['return']:>8.2f}%     "
                  f"{r['win_rate']:>6.1f}%     {r['trades']:>4}笔   {r['final_capital']:>10,.0f}")

        print("\n" + "=" * 80)
        print("真实交易回测排名（按胜率）")
        print("=" * 80)
        results_win = sorted(results, key=lambda x: x['win_rate'], reverse=True)
        print(f"\n{'排名':<4} {'品种':<10} {'胜率':<10} {'收益率':<12} {'交易数':<8} {'最终权益'}")
        print('-'*80)
        for i, r in enumerate(results_win, 1):
            print(f"{i:<4} {r['name']:<10} {r['win_rate']:>6.1f}%     "
                  f"{r['return']:>8.2f}%     {r['trades']:>4}笔   {r['final_capital']:>10,.0f}")

        print("\n" + "=" * 80)
        print("总结")
        print("=" * 80)
        print(f"\n总品种数: {len(results)}")
        print(f"平均收益率: {sum(r['return'] for r in results)/len(results):.2f}%")
        print(f"平均胜率: {sum(r['win_rate'] for r in results)/len(results):.1f}%")
        print(f"总交易数: {sum(r['trades'] for r in results)}笔")

        # 盈亏统计
        profit_variety = sum(1 for r in results if r['return'] > 0)
        print(f"\n盈利品种: {profit_variety}/{len(results)}")
        print(f"亏损品种: {len(results) - profit_variety}/{len(results)}")

    print("\n" + "=" * 80)
    print("回测完成")
    print("=" * 80)


if __name__ == "__main__":
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    main()
