# -*- coding: utf-8 -*-
"""
PTA完整交易记录报告 - 验证数据准确性和时间连贯性
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime, timedelta

def generate_detailed_report():
    """生成详细的PTA交易记录报告"""

    print("=" * 150)
    print("PTA完整交易记录验证报告".center(150))
    print("=" * 150)

    # ========== 配置参数 ==========
    RSI_FILTER_LONG = 40
    RATIO_TRIGGER_LONG = 1.25
    STOP_LOSS_PCT_LONG = 0.02
    RSI_FILTER_SHORT = 60
    RATIO_TRIGGER_SHORT = 1.25
    STC_SELL_ZONE = 75
    STC_BUY_ZONE = 25

    # ========== 获取数据 ==========
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('TA', days=300)

    if df is None or df.empty:
        print("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # ========== 计算指标 ==========
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    def calculate_stc(series, fast_period=23, slow_period=50, period=10):
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        stoch_k = 100 * (macd - macd.rolling(window=period).min()) / \
                  (macd.rolling(window=period).max() - macd.rolling(window=period).min())
        stoch_d = stoch_k.ewm(span=period, adjust=False).mean()
        stoch_k_d = 100 * (stoch_k - stoch_d.rolling(window=period).min()) / \
                    (stoch_d.rolling(window=period).max() - stoch_d.rolling(window=period).min())
        stc = stoch_k_d.ewm(span=period, adjust=False).mean()
        return stc

    df['stc'] = calculate_stc(df['close'])
    df['stc_prev'] = df['stc'].shift(1)

    # ========== 回测生成交易记录 ==========
    trades = []
    position = None

    for i in range(50, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        if position is not None:
            exit_triggered = False
            exit_reason = None
            exit_price = None

            if position['direction'] == 'long':
                if current['low'] <= position['stop_loss']:
                    exit_triggered = True
                    exit_reason = 'stop_loss_long'
                    exit_price = current['close']
            elif position['direction'] == 'short':
                if current['high'] >= position['stop_loss']:
                    exit_triggered = True
                    exit_reason = 'stop_loss_short'
                    exit_price = current['close']

            if not exit_triggered:
                if position['direction'] == 'long':
                    stc_exit = (prev['stc'] > STC_SELL_ZONE) and (current['stc'] < prev['stc'])
                    trend_exit = current['ema_fast'] < current['ema_slow']
                    if stc_exit:
                        exit_triggered = True
                        exit_reason = 'stc_exit_long'
                        exit_price = current['close']
                    elif trend_exit:
                        exit_triggered = True
                        exit_reason = 'trend_exit_long'
                        exit_price = current['close']
                elif position['direction'] == 'short':
                    stc_exit = (prev['stc'] < STC_BUY_ZONE) and (current['stc'] > prev['stc'])
                    trend_exit = current['ema_fast'] > current['ema_slow']
                    if stc_exit:
                        exit_triggered = True
                        exit_reason = 'stc_exit_short'
                        exit_price = current['close']
                    elif trend_exit:
                        exit_triggered = True
                        exit_reason = 'trend_exit_short'
                        exit_price = current['close']

            if exit_triggered:
                if position['direction'] == 'long':
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                else:
                    pnl_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100

                trade = {
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'direction': position['direction'],
                    'signal_type': position['signal_type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'hold_bars': i - position['entry_index']
                }
                trades.append(trade)
                position = None
                continue

        if position is None:
            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe_long = (0 < current['ratio'] < RATIO_TRIGGER_LONG)
            ratio_shrinking_long = current['ratio'] < prev['ratio']
            turning_up_long = current['macd_dif'] > prev['macd_dif']
            is_strong_long = current['rsi'] > RSI_FILTER_LONG
            sniper_long = trend_up and ratio_safe_long and ratio_shrinking_long and turning_up_long and is_strong_long
            ema_golden_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            chase_long = ema_golden_cross and is_strong_long

            trend_down = current['ema_fast'] < current['ema_slow']
            ratio_safe_short = (-RATIO_TRIGGER_SHORT < current['ratio'] < 0)
            ratio_falling_short = current['ratio'] < prev['ratio']
            turning_down_short = current['macd_dif'] < prev['macd_dif']
            is_weak_short = current['rsi'] < RSI_FILTER_SHORT
            sniper_short = trend_down and ratio_safe_short and ratio_falling_short and turning_down_short and is_weak_short
            ema_death_cross = (prev['ema_fast'] >= prev['ema_slow']) and (current['ema_fast'] < current['ema_slow'])
            chase_short = ema_death_cross and is_weak_short

            if sniper_long or chase_long:
                signal_type = 'sniper_long' if sniper_long else 'chase_long'
                stop_loss = current['close'] * (1 - STOP_LOSS_PCT_LONG)
                position = {
                    'direction': 'long',
                    'signal_type': signal_type,
                    'entry_datetime': current['datetime'],
                    'entry_price': current['close'],
                    'stop_loss': stop_loss,
                    'entry_index': i
                }
            elif sniper_short or chase_short:
                signal_type = 'sniper_short' if sniper_short else 'chase_short'
                stop_loss = current['close'] * (1 + STOP_LOSS_PCT_LONG)
                position = {
                    'direction': 'short',
                    'signal_type': signal_type,
                    'entry_datetime': current['datetime'],
                    'entry_price': current['close'],
                    'stop_loss': stop_loss,
                    'entry_index': i
                }

    # ========== 合约规格 ==========
    CONTRACT_SIZE = 5  # PTA合约乘数：5吨/手
    MARGIN_RATE = 0.08  # 保证金率：8%

    print(f"\n【合约规格】")
    print(f"  品种：PTA (TA)")
    print(f"  合约乘数：{CONTRACT_SIZE} 吨/手")
    print(f"  保证金率：{MARGIN_RATE * 100}%")
    print(f"  交易单位：1手")
    print(f"  回测周期：{df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"  K线数量：{len(df)} 根")

    # ========== 时间连贯性验证 ==========
    print(f"\n{'=' * 150}")
    print("时间连贯性验证")
    print(f"{'=' * 150}")

    time_errors = []
    for i in range(len(trades) - 1):
        current_exit = trades[i]['exit_datetime']
        next_entry = trades[i + 1]['entry_datetime']
        if next_entry < current_exit:
            time_errors.append({
                'trade_1': i + 1,
                'trade_2': i + 2,
                'current_exit': current_exit,
                'next_entry': next_entry,
                'overlap': (next_entry - current_exit).total_seconds() / 3600
            })

    if time_errors:
        print(f"\n[X] 发现时间重叠错误：")
        for error in time_errors:
            print(f"  交易{error['trade_1']}出场时间：{error['current_exit']}")
            print(f"  交易{error['trade_2']}入场时间：{error['next_entry']}")
            print(f"  重叠时长：{abs(error['overlap']):.2f} 小时")
    else:
        print(f"\n[OK] 时间连贯性检查通过：所有交易无重叠")

    # ========== 详细交易记录 ==========
    print(f"\n{'=' * 150}")
    print("详细交易记录（共{}笔）".format(len(trades)))
    print(f"{'=' * 150}")

    print(f"\n{'-' * 150}")
    print(f"{'序号':<5} {'方向':<6} {'信号':<15} {'入场时间':<19} {'入场价':<8} {'出场时间':<19} {'出场价':<8} {'持仓':<8} {'保证金':<10} {'盈亏金额':<10} {'盈亏%':<8} {'退出原因':<15}")
    print(f"{'-' * 150}")

    total_pnl_amount = 0
    long_trades = []
    short_trades = []

    for i, trade in enumerate(trades, 1):
        direction_str = '做多' if trade['direction'] == 'long' else '做空'
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        margin = entry_price * CONTRACT_SIZE * MARGIN_RATE

        if trade['direction'] == 'long':
            pnl_amount = (exit_price - entry_price) * CONTRACT_SIZE
            long_trades.append(trade)
        else:
            pnl_amount = (entry_price - exit_price) * CONTRACT_SIZE
            short_trades.append(trade)

        total_pnl_amount += pnl_amount
        pnl_pct = trade['pnl_pct']
        pnl_str = f"{'+' if pnl_amount >= 0 else ''}{pnl_amount:.2f}"
        pnl_pct_str = f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"

        hold_hours = trade['hold_bars'] * 4  # 4小时K线
        hold_str = f"{hold_hours}h"

        print(f"{i:<5} {direction_str:<6} {trade['signal_type']:<15} "
              f"{str(trade['entry_datetime']):<19} {entry_price:<8.2f} "
              f"{str(trade['exit_datetime']):<19} {exit_price:<8.2f} "
              f"{hold_str:<8} {margin:<10.2f} {pnl_str:<10} {pnl_pct_str:<8} {trade['exit_reason']:<15}")

    # ========== 统计汇总 ==========
    print(f"\n{'=' * 150}")
    print("统计汇总")
    print(f"{'=' * 150}")

    long_win = [t for t in long_trades if t['pnl_pct'] > 0]
    long_loss = [t for t in long_trades if t['pnl_pct'] <= 0]
    short_win = [t for t in short_trades if t['pnl_pct'] > 0]
    short_loss = [t for t in short_trades if t['pnl_pct'] <= 0]

    print(f"\n【做多统计】")
    print(f"  交易次数：{len(long_trades)} 次")
    print(f"  获胜次数：{len(long_win)} 次")
    print(f"  失败次数：{len(long_loss)} 次")
    print(f"  胜率：{len(long_win) / len(long_trades) * 100:.1f}%")
    print(f"  总盈亏：{sum(t['pnl_pct'] for t in long_trades):.2f}%")
    print(f"  总盈亏金额：{sum((t['exit_price'] - t['entry_price']) * CONTRACT_SIZE for t in long_trades):.2f} 元")

    print(f"\n【做空统计】")
    print(f"  交易次数：{len(short_trades)} 次")
    print(f"  获胜次数：{len(short_win)} 次")
    print(f"  失败次数：{len(short_loss)} 次")
    print(f"  胜率：{len(short_win) / len(short_trades) * 100:.1f}%")
    print(f"  总盈亏：{sum(t['pnl_pct'] for t in short_trades):.2f}%")
    print(f"  总盈亏金额：{sum((t['entry_price'] - t['exit_price']) * CONTRACT_SIZE for t in short_trades):.2f} 元")

    print(f"\n【总体统计】")
    print(f"  总交易次数：{len(trades)} 次")
    print(f"  总盈亏：{sum(t['pnl_pct'] for t in trades):.2f}%")
    print(f"  总盈亏金额：{total_pnl_amount:.2f} 元")
    print(f"  平均每笔盈亏：{total_pnl_amount / len(trades):.2f} 元")

    # ========== 数据准确性验证 ==========
    print(f"\n{'=' * 150}")
    print("数据准确性验证（随机抽查5笔）")
    print(f"{'=' * 150}")

    import random
    sample_indices = random.sample(range(len(trades)), min(5, len(trades)))

    for idx in sample_indices:
        trade = trades[idx]
        print(f"\n  交易 #{idx + 1} ({trade['direction']}):")
        print(f"    入场：{trade['entry_price']} 元/吨")
        print(f"    出场：{trade['exit_price']} 元/吨")
        print(f"    合约乘数：{CONTRACT_SIZE} 吨/手")
        print(f"    保证金率：{MARGIN_RATE * 100}%")

        if trade['direction'] == 'long':
            expected_pnl = (trade['exit_price'] - trade['entry_price']) * CONTRACT_SIZE
            expected_pct = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100
        else:
            expected_pnl = (trade['entry_price'] - trade['exit_price']) * CONTRACT_SIZE
            expected_pct = (trade['entry_price'] - trade['exit_price']) / trade['entry_price'] * 100

        margin = trade['entry_price'] * CONTRACT_SIZE * MARGIN_RATE

        print(f"    保证金计算：{trade['entry_price']} × {CONTRACT_SIZE} × {MARGIN_RATE} = {margin:.2f} 元")
        print(f"    盈亏金额计算：{expected_pnl:.2f} 元")
        print(f"    盈亏百分比计算：{expected_pct:.2f}%")
        print(f"    系统记录：盈亏 {trade['pnl_pct']:.2f}%")

        if abs(expected_pct - trade['pnl_pct']) < 0.01:
            print(f"    [OK] 数据准确")
        else:
            print(f"    [ERROR] 数据不准确！差异：{abs(expected_pct - trade['pnl_pct']):.4f}%")

    # ========== 保存到文件 ==========
    output_file = Path('logs/pta_detailed_trades_report.txt')
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 150 + "\n")
        f.write("PTA完整交易记录报告\n")
        f.write("=" * 150 + "\n")
        f.write(f"\n生成时间：{datetime.now()}\n")
        f.write(f"合约规格：PTA (TA) {CONTRACT_SIZE}吨/手，保证金率{MARGIN_RATE*100}%\n")
        f.write(f"回测周期：{df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}\n")
        f.write(f"总交易：{len(trades)} 笔\n")

        f.write(f"\n{'-' * 150}\n")
        f.write(f"{'序号':<5} {'方向':<6} {'信号':<15} {'入场时间':<19} {'入场价':<8} {'出场时间':<19} {'出场价':<8} {'持仓':<8} {'保证金':<10} {'盈亏金额':<10} {'盈亏%':<8} {'退出原因':<15}\n")
        f.write(f"{'-' * 150}\n")

        for i, trade in enumerate(trades, 1):
            direction_str = '做多' if trade['direction'] == 'long' else '做空'
            pnl_amount = ((trade['exit_price'] - trade['entry_price']) * CONTRACT_SIZE if trade['direction'] == 'long'
                          else (trade['entry_price'] - trade['exit_price']) * CONTRACT_SIZE)
            pnl_pct = trade['pnl_pct']
            pnl_str = f"{'+' if pnl_amount >= 0 else ''}{pnl_amount:.2f}"
            pnl_pct_str = f"{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%"
            hold_hours = trade['hold_bars'] * 4
            hold_str = f"{hold_hours}h"
            margin = trade['entry_price'] * CONTRACT_SIZE * MARGIN_RATE

            f.write(f"{i:<5} {direction_str:<6} {trade['signal_type']:<15} "
                   f"{str(trade['entry_datetime']):<19} {trade['entry_price']:<8.2f} "
                   f"{str(trade['exit_datetime']):<19} {trade['exit_price']:<8.2f} "
                   f"{hold_str:<8} {margin:<10.2f} {pnl_str:<10} {pnl_pct_str:<8} {trade['exit_reason']:<15}\n")

        f.write(f"\n总盈亏金额：{total_pnl_amount:.2f} 元\n")
        f.write(f"平均每笔盈亏：{total_pnl_amount / len(trades):.2f} 元\n")

    print(f"\n详细报告已保存到：{output_file}")
    print(f"\n报告生成完成时间：{datetime.now()}")

if __name__ == '__main__':
    generate_detailed_report()
