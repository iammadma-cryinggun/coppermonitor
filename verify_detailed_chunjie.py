# -*- coding: utf-8 -*-
"""
详细验证纯碱的优化结果
展示所有交易细节
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 固定参数
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50
STOP_LOSS_PCT = 0.02
INITIAL_CAPITAL = 100000
MARGIN_RATIO = 0.15
CONTRACT_SIZE = 5

def calculate_indicators(df, params):
    """计算技术指标"""
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

    exp1 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    stc_macd = df['close'].ewm(span=STC_FAST, adjust=False).mean() - \
               df['close'].ewm(span=STC_SLOW, adjust=False).mean()
    stoch_period = STC_LENGTH
    min_macd = stc_macd.rolling(window=stoch_period).min()
    max_macd = stc_macd.rolling(window=stoch_period).max()
    stoch_k = 100 * (stc_macd - min_macd) / (max_macd - min_macd).replace(0, np.nan)
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=3).mean()
    min_stoch_d = stoch_d.rolling(window=stoch_period).min()
    max_stoch_d = stoch_d.rolling(window=stoch_period).max()
    stc_raw = 100 * (stoch_d - min_stoch_d) / (max_stoch_d - min_stoch_d).replace(0, np.nan)
    stc_raw = stc_raw.fillna(50)
    df['stc'] = stc_raw.rolling(window=3).mean()

    df['ratio_prev'] = df['ratio'].shift(1)
    df['stc_prev'] = df['stc'].shift(1)

    return df

def detailed_backtest(df, params):
    """详细回测，记录所有交易"""
    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > params['RSI_FILTER']
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong
        buy_signal = sniper_signal or chase_signal

        # 计算仓位
        if df['ratio_prev'].iloc[i] > 0:
            if df['ratio_prev'].iloc[i] > 2.0:
                position_size = 2.0
            elif df['ratio_prev'].iloc[i] > 1.5:
                position_size = 1.5
            elif df['ratio_prev'].iloc[i] > 1.0:
                position_size = 1.2
            else:
                position_size = 1.0
        else:
            position_size = 1.0

        stop_loss = current['close'] * (1 - STOP_LOSS_PCT)

        # 买入
        if buy_signal and position is None:
            entry_price = current['close']
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATIO
            available_margin = capital * position_size
            contracts = int(available_margin / margin_per_contract)

            if contracts <= 0:
                continue

            position = {
                'entry_datetime': current['datetime'],
                'entry_price': entry_price,
                'contracts': contracts,
                'stop_loss': stop_loss,
                'entry_index': i,
                'position_size': position_size,
                'entry_capital': capital
            }

        # 卖出
        elif position is not None:
            for j in range(position['entry_index'] + 1, len(df)):
                bar = df.iloc[j]

                exit_triggered = False
                exit_price = None
                exit_reason = None

                if bar['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_triggered = True
                    exit_reason = '止损'
                elif (df['stc_prev'].iloc[j] > params['STC_SELL_ZONE'] and
                      bar['stc'] < df['stc_prev'].iloc[j]):
                    exit_price = bar['close']
                    exit_triggered = True
                    exit_reason = 'STC止盈'
                elif bar['ema_fast'] < bar['ema_slow']:
                    exit_price = bar['close']
                    exit_triggered = True
                    exit_reason = '趋势反转'

                if exit_triggered:
                    pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                    capital += pnl

                    trade_info = {
                        'entry_datetime': position['entry_datetime'],
                        'entry_price': position['entry_price'],
                        'exit_datetime': bar['datetime'],
                        'exit_price': exit_price,
                        'contracts': position['contracts'],
                        'position_size': position['position_size'],
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'return_pct': pnl / position['entry_capital'] * 100,
                        'capital_after': capital
                    }
                    trades.append(trade_info)

                    position = None
                    break

    return trades, capital

def main():
    print("=" * 100)
    print("纯碱详细回测验证")
    print("=" * 100)

    # 加载数据
    csv_file = Path('futures_data_4h/纯碱_4hour.csv')
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 最优参数
    params = {
        'EMA_FAST': 12,
        'EMA_SLOW': 10,
        'RSI_FILTER': 35,
        'RATIO_TRIGGER': 1.10,
        'STC_SELL_ZONE': 75
    }

    print(f"\n数据信息:")
    print(f"  品种: 纯碱")
    print(f"  数据量: {len(df)}条")
    print(f"  时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    print(f"\n优化参数:")
    print(f"  EMA({params['EMA_FAST']}, {params['EMA_SLOW']})")
    print(f"  RSI_FILTER: {params['RSI_FILTER']}")
    print(f"  RATIO_TRIGGER: {params['RATIO_TRIGGER']:.2f}")
    print(f"  STC_SELL_ZONE: {params['STC_SELL_ZONE']}")

    print(f"\n预期结果:")
    print(f"  收益率: +190.86%")
    print(f"  交易数: 39笔")
    print(f"  胜率: 66.7%")

    # 回测
    trades, final_capital = detailed_backtest(df, params)

    if not trades:
        print("\n[ERROR] 无交易记录")
        return

    # 统计
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    total_pnl = sum(t['pnl'] for t in trades)
    return_pct = total_pnl / INITIAL_CAPITAL * 100
    win_rate = winning_trades / total_trades * 100

    print(f"\n实际回测结果:")
    print(f"  收益率: {return_pct:+.2f}%")
    print(f"  交易数: {total_trades}笔")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  最终资金: {final_capital:,.2f}")

    # 对比
    if abs(return_pct - 190.86) < 0.1 and total_trades == 39:
        print(f"\n[验证通过] 与优化结果完全一致！")
    else:
        print(f"\n[验证失败] 与优化结果不一致！")

    # 详细交易记录
    print(f"\n{'='*100}")
    print("详细交易记录:")
    print(f"{'='*100}")

    print(f"\n{'序号':<6} {'入场时间':>20} {'入场价格':>10} {'出场时间':>20} {'出场价格':>10} "
          f"{'手数':>6} {'仓位':>6} {'出场原因':>10} {'盈亏':>12} {'收益率%':>10} {'资金后':>15}")

    print("-" * 160)

    for i, trade in enumerate(trades, 1):
        pnl_str = f"{trade['pnl']:+,.2f}" if trade['pnl'] >= 0 else f"{trade['pnl']:,.2f}"
        return_str = f"{trade['return_pct']:+.2f}%" if trade['pnl'] >= 0 else f"{trade['return_pct']:.2f}%"
        capital_str = f"{trade['capital_after']:,.2f}"

        print(f"{i:<6} {str(trade['entry_datetime']):>20} {trade['entry_price']:>10.2f} "
              f"{str(trade['exit_datetime']):>20} {trade['exit_price']:>10.2f} "
              f"{trade['contracts']:>6} {trade['position_size']:>6.1f} {trade['exit_reason']:>10} "
              f"{pnl_str:>12} {return_str:>10} {capital_str:>15}")

    # 盈亏分析
    print(f"\n{'='*100}")
    print("盈亏分析:")
    print(f"{'='*100}")

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    print(f"\n盈利交易 ({len(wins)}笔):")
    if wins:
        avg_win = np.mean([t['pnl'] for t in wins])
        max_win = max([t['pnl'] for t in wins])
        print(f"  平均盈利: {avg_win:,.2f}")
        print(f"  最大盈利: {max_win:,.2f}")
        print(f"  盈利总额: {sum(t['pnl'] for t in wins):,.2f}")

    print(f"\n亏损交易 ({len(losses)}笔):")
    if losses:
        avg_loss = np.mean([t['pnl'] for t in losses])
        max_loss = min([t['pnl'] for t in losses])
        print(f"  平均亏损: {avg_loss:,.2f}")
        print(f"  最大亏损: {max_loss:,.2f}")
        print(f"  亏损总额: {sum(t['pnl'] for t in losses):,.2f}")

    if wins and losses:
        profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses))
        print(f"\n盈亏比: {profit_factor:.2f}")

    # 出场原因统计
    print(f"\n出场原因统计:")
    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    for reason, count in exit_reasons.items():
        pct = count / total_trades * 100
        print(f"  {reason}: {count}笔 ({pct:.1f}%)")

if __name__ == '__main__':
    main()
