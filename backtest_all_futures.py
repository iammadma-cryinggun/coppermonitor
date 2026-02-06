# -*- coding: utf-8 -*-
"""
批量回测22个期货品种
使用铜期货的纯策略参数
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 使用铜期货优化的参数
EMA_FAST = 5
EMA_SLOW = 15
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
STOP_LOSS_PCT = 0.02

def calculate_indicators(df):
    """计算技术指标"""
    # EMA
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    # MACD
    exp1 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['macd_hist'] = df['macd_dif'] - df['macd_dea']

    return df

def backtest(csv_file, future_name):
    """回测单个品种"""
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = calculate_indicators(df)

    INITIAL_CAPITAL = 10000
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(max(EMA_SLOW, MACD_SLOW) + 50, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # EMA交叉信号
        ema_cross_up = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
        ema_cross_down = (prev['ema_fast'] >= prev['ema_slow']) and (current['ema_fast'] < current['ema_slow'])

        # MACD柱状图
        macd_hist_positive = current['macd_hist'] > 0
        macd_hist_negative = current['macd_hist'] < 0

        # 买入：EMA上穿 AND MACD柱状图 > 0
        if ema_cross_up and macd_hist_positive and position is None:
            entry_price = current['close']
            position_value = capital
            amount = position_value / entry_price

            position = {
                'entry_datetime': current['datetime'],
                'entry_price': entry_price,
                'amount': amount,
                'stop_loss': entry_price * (1 - STOP_LOSS_PCT),
                'entry_index': i
            }

        # 卖出：EMA下穿 OR 触发止损
        elif position is not None:
            stop_loss_hit = current['low'] <= position['stop_loss']
            exit_signal = ema_cross_down or stop_loss_hit

            if exit_signal:
                exit_price = position['stop_loss'] if stop_loss_hit else current['close']
                exit_reason = 'stop_loss' if stop_loss_hit else 'ema_cross'

                pnl = (exit_price - position['entry_price']) * position['amount']
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': exit_reason,
                    'holding_bars': i - position['entry_index']
                })

                position = None

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])

    total_pnl = trades_df['pnl'].sum()
    return_pct = total_pnl / INITIAL_CAPITAL * 100
    win_rate = winning_trades / total_trades * 100

    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    return {
        'name': future_name,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'return_pct': return_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_capital': capital,
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
    }

def main():
    print("=" * 80)
    print("批量回测22个期货品种")
    print(f"策略: EMA({EMA_FAST},{EMA_SLOW}) + MACD + {STOP_LOSS_PCT*100:.0f}%止损")
    print("=" * 80)

    data_dir = Path('futures_data_4h')
    csv_files = sorted(data_dir.glob('*.csv'))

    results = []

    for csv_file in csv_files:
        future_name = csv_file.stem.replace('_4hour', '')
        print(f"\n正在回测: {future_name}...")

        try:
            result = backtest(csv_file, future_name)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  [FAIL] {e}")

    if not results:
        print("\n无有效结果")
        return

    # 排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)

    # 输出完整排名
    print("\n" + "=" * 80)
    print("回测结果排名（按收益率排序）")
    print("=" * 80)
    print(f"\n{'排名':<6} {'品种':<10} {'收益率':>10} {'胜率':>8} {'交易次数':>8} {'盈亏比':>8}")
    print("-" * 80)

    for i, row in results_df.iterrows():
        rank = results_df.index.get_loc(i) + 1
        print(f"{rank:<6} {row['name']:<10} {row['return_pct']:>+8.2f}% {row['win_rate']:>7.1f}% {row['total_trades']:>8} {row['profit_factor']:>8.2f}")

    # 统计
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)

    profitable = len(results_df[results_df['return_pct'] > 0])
    unprofitable = len(results_df[results_df['return_pct'] <= 0])

    print(f"\n总品种数:     {len(results_df)}")
    print(f"盈利品种:     {profitable}")
    print(f"亏损品种:     {unprofitable}")
    print(f"\n平均收益率:   {results_df['return_pct'].mean():+.2f}%")
    print(f"最高收益:     {results_df['return_pct'].max():+.2f}% ({results_df.iloc[0]['name']})")
    print(f"最低收益:     {results_df['return_pct'].min():+.2f}% ({results_df.iloc[-1]['name']})")
    print(f"收益率中位数: {results_df['return_pct'].median():+.2f}%")

    # 显示前5名详细信息
    print("\n" + "=" * 80)
    print("前5名详细信息")
    print("=" * 80)

    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n第{i+1}名: {row['name']}")
        print(f"  收益率:       {row['return_pct']:+.2f}%")
        print(f"  交易次数:     {row['total_trades']}")
        print(f"  胜率:         {row['win_rate']:.1f}%")
        print(f"  盈利次数:     {row['winning_trades']}")
        print(f"  亏损次数:     {row['losing_trades']}")
        print(f"  平均盈利:     {row['avg_win']:+,.2f}")
        print(f"  平均亏损:     {row['avg_loss']:+,.2f}")
        print(f"  盈亏比:       {row['profit_factor']:.2f}")
        print(f"  总盈亏:       {row['total_pnl']:+,.2f}")
        print(f"  最终资金:     {row['final_capital']:,.2f}")

if __name__ == '__main__':
    main()
