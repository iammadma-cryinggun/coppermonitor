# -*- coding: utf-8 -*-
"""
沪锡日线回测 - 加入日线趋势过滤
只有日线EMA向上时才做多
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_monitor import calculate_indicators
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
MAX_POSITION_RATIO = 0.8
STOP_LOSS_PCT = 0.02

# 沪锡合约规格
CONTRACT_SIZE = 1
MARGIN_RATE = 0.13

# 日线优化最优参数
PARAMS_DAILY = {
    'EMA_FAST': 3,
    'EMA_SLOW': 15,
    'RSI_FILTER': 30,
    'RATIO_TRIGGER': 1.05,
    'STC_SELL_ZONE': 70,
    'STOP_LOSS_PCT': 0.02
}


def backtest_with_trend_filter(df, params):
    """
    回测 - 加入日线趋势过滤
    只有EMA快线>EMA慢线（趋势向上）时才允许做多
    """
    # 计算指标
    df = calculate_indicators(df, params)

    # 回测
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    filtered_signals = 0  # 被过滤的信号数量

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

        # 开仓逻辑 - 加入趋势过滤
        if position is None:
            # 【关键】检查日线趋势
            trend_up = current['ema_fast'] > current['ema_slow']

            # 如果趋势向下，直接跳过，不开仓
            if not trend_up:
                # 记录被过滤的信号（即使有买入信号也不开仓）
                ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
                ratio_shrinking = current['ratio'] < current['ratio_prev']
                turning_up = current['macd_dif'] > prev['macd_dif']
                is_strong = current['rsi'] > params['RSI_FILTER']

                sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
                ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
                buy_signal = sniper_signal or (ema_cross and is_strong)

                if buy_signal:
                    filtered_signals += 1

                continue

            # 趋势向上，检查买入信号
            ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
            ratio_shrinking = current['ratio'] < current['ratio_prev']
            turning_up = current['macd_dif'] > prev['macd_dif']
            is_strong = current['rsi'] > params['RSI_FILTER']

            sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
            ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            buy_signal = sniper_signal or (ema_cross and is_strong)

            if buy_signal:
                # 计算可开仓数量
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
        'filtered_signals': filtered_signals,
        'total_signals': len(trades) + filtered_signals,
        'win_rate': win_rate,
        'win_trades': len(win_trades),
        'loss_trades': len(trades) - len(win_trades),
        'avg_return': total_return / len(trades),
        'trade_list': trades
    }


def main():
    """主函数"""
    print("=" * 80)
    print("沪锡日线回测 - 加入日线趋势过滤")
    print("=" * 80)
    print(f"规则: 只有日线EMA向上时才做多")
    print(f"参数: EMA({PARAMS_DAILY['EMA_FAST']},{PARAMS_DAILY['EMA_SLOW']}) "
          f"RSI={PARAMS_DAILY['RSI_FILTER']} RATIO={PARAMS_DAILY['RATIO_TRIGGER']} "
          f"STC={PARAMS_DAILY['STC_SELL_ZONE']}")
    print()

    # 读取数据
    csv_file = "SN_沪锡_日线.csv"
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    print(f"数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 条")
    print()

    # 回测
    result = backtest_with_trend_filter(df, PARAMS_DAILY)

    if result:
        print("=" * 80)
        print("回测结果（加入趋势过滤）")
        print("=" * 80)
        print(f"\n初始资金: {INITIAL_CAPITAL:,.2f} 元")
        print(f"最终权益: {result['capital']:,.2f} 元")
        print(f"总收益: {result['capital'] - INITIAL_CAPITAL:,.2f} 元")
        print(f"收益率: {result['return']:.2f}%")
        print()
        print(f"交易次数: {result['trades']}笔")
        print(f"被过滤信号: {result['filtered_signals']}笔")
        print(f"总信号数: {result['total_signals']}笔")
        print(f"信号过滤率: {result['filtered_signals']/result['total_signals']*100:.1f}%")
        print()
        print(f"盈利交易: {result['win_trades']}笔")
        print(f"亏损交易: {result['loss_trades']}笔")
        print(f"胜率: {result['win_rate']:.1f}%")
        print(f"平均收益: {result['avg_return']:.2f}%")
        print()

        # 显示所有交易
        print("=" * 80)
        print("所有交易明细")
        print("=" * 80)
        print(f"\n{'序号':<4} {'入场日期':<12} {'出场日期':<12} {'手数':<4} {'入场价':<8} {'出场价':<8} "
              f"{'盈亏':<12} {'收益率':<8} {'原因'}")
        print('-'*80)

        for i, trade in enumerate(result['trade_list'], 1):
            pnl_str = f"+{trade['pnl']:,.0f}" if trade['pnl'] > 0 else f"{trade['pnl']:,.0f}"
            pct_str = f"+{trade['pnl_pct']:.2f}%" if trade['pnl_pct'] > 0 else f"{trade['pnl_pct']:.2f}%"
            print(f"{i:<4} {trade['entry_datetime'].date()} {trade['exit_datetime'].date()} "
                  f"{trade['contracts']:<4} {trade['entry_price']:<8.0f} {trade['exit_price']:<8.0f} "
                  f"{pnl_str:<12} {pct_str:<8} {trade['reason']}")

        # 统计分析
        print("\n" + "=" * 80)
        print("统计分析")
        print("=" * 80)

        wins = [t for t in result['trade_list'] if t['pnl'] > 0]
        losses = [t for t in result['trade_list'] if t['pnl'] <= 0]

        if wins:
            avg_win = sum(t['pnl'] for t in wins) / len(wins)
            max_win = max(t['pnl'] for t in wins)
            print(f"平均盈利: {avg_win:,.2f} 元")
            print(f"最大盈利: {max_win:,.2f} 元")

        if losses:
            avg_loss = sum(t['pnl'] for t in losses) / len(losses)
            max_loss = min(t['pnl'] for t in losses)
            print(f"平均亏损: {avg_loss:,.2f} 元")
            print(f"最大亏损: {max_loss:,.2f} 元")

        if wins and losses:
            profit_loss_ratio = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses))
            print(f"盈亏比: {profit_loss_ratio:.2f}")

        # 对比分析
        print("\n" + "=" * 80)
        print("效果对比")
        print("=" * 80)
        print(f"\n不加趋势过滤:")
        print(f"  交易次数: 29笔")
        print(f"  收益率: 500.04%")
        print(f"  胜率: 41.4%")
        print(f"  最终权益: 600,041元")
        print(f"\n加入趋势过滤:")
        print(f"  交易次数: {result['trades']}笔")
        print(f"  过滤信号: {result['filtered_signals']}笔")
        print(f"  收益率: {result['return']:.2f}%")
        print(f"  胜率: {result['win_rate']:.1f}%")
        print(f"  最终权益: {result['capital']:,.0f}元")

        if result['trades'] > 0:
            print(f"\n结论: 趋势过滤{'减少了' if result['trades'] < 29 else '增加了或维持了'}交易次数，"
                  f"{'提高了' if result['win_rate'] > 41.4 else '降低或维持了'}胜率")

    print("\n" + "=" * 80)
    print("回测完成")
    print("=" * 80)


if __name__ == "__main__":
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    main()
