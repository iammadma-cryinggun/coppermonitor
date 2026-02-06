# -*- coding: utf-8 -*-
"""
PTA只做多回测 - 使用之前优化后的最佳参数验证
与双向回测的做多部分对比
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime

def backtest_long_only_optimized_params():
    """使用之前优化后的最佳参数进行只做多回测"""

    print("=" * 100)
    print("PTA只做多回测 - 使用优化后的最佳参数")
    print("=" * 100)

    # ========== 使用之前优化后的最佳参数 ==========
    EMA_FAST = 12
    EMA_SLOW = 10  # 优化后的值是10，不是26！
    RSI_FILTER_LONG = 40
    RATIO_TRIGGER_LONG = 1.25
    STOP_LOSS_PCT_LONG = 0.02
    STC_SELL_ZONE = 75

    print(f"\n参数配置（与优化结果一致）:")
    print(f"  EMA_FAST={EMA_FAST}, EMA_SLOW={EMA_SLOW}")
    print(f"  RSI_FILTER={RSI_FILTER_LONG}, RATIO_TRIGGER={RATIO_TRIGGER_LONG}")
    print(f"  STOP_LOSS={STOP_LOSS_PCT_LONG*100}%, STC_SELL_ZONE={STC_SELL_ZONE}")

    # ========== 获取数据 ==========
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('TA', days=300)

    if df is None or df.empty:
        print("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"K线数量: {len(df)}")

    # ========== 计算指标 ==========
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()  # 关键：使用10而不是26

    exp1 = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    exp2 = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # STC指标
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

    # ========== 交易记录 ==========
    trades = []
    position = None

    # ========== 开始回测 ==========
    print(f"\n开始回测...")

    for i in range(50, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 如果有持仓，检查平仓或止损
        if position is not None:
            exit_triggered = False
            exit_reason = None
            exit_price = None

            # 止损检查
            if current['low'] <= position['stop_loss']:
                exit_triggered = True
                exit_reason = 'stop_loss'
                exit_price = current['close']

            # 平仓信号检查
            if not exit_triggered:
                stc_exit = (prev['stc'] > STC_SELL_ZONE) and (current['stc'] < prev['stc'])
                trend_exit = current['ema_fast'] < current['ema_slow']

                if stc_exit:
                    exit_triggered = True
                    exit_reason = 'stc_exit'
                    exit_price = current['close']
                elif trend_exit:
                    exit_triggered = True
                    exit_reason = 'trend_exit'
                    exit_price = current['close']

            # 如果触发平仓
            if exit_triggered:
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100

                trade = {
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'direction': 'long',
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

        # 如果没有持仓，检查开仓信号
        if position is None:
            # 做多信号
            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe_long = (0 < current['ratio'] < RATIO_TRIGGER_LONG)
            ratio_shrinking_long = current['ratio'] < prev['ratio']
            turning_up_long = current['macd_dif'] > prev['macd_dif']
            is_strong_long = current['rsi'] > RSI_FILTER_LONG

            sniper_long = trend_up and ratio_safe_long and ratio_shrinking_long and turning_up_long and is_strong_long

            ema_golden_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            chase_long = ema_golden_cross and is_strong_long

            # 开仓
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

    # ========== 合约规格 ==========
    CONTRACT_SIZE = 5
    MARGIN_RATE = 0.08

    # ========== 统计结果 ==========
    print(f"\n{'='*100}")
    print("回测结果统计")
    print(f"{'='*100}")

    if trades:
        win_trades = [t for t in trades if t['pnl_pct'] > 0]
        loss_trades = [t for t in trades if t['pnl_pct'] <= 0]

        win_rate = len(win_trades) / len(trades) * 100
        avg_win = np.mean([t['pnl_pct'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in loss_trades]) if loss_trades else 0
        total_pnl_pct = sum([t['pnl_pct'] for t in trades])
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        profit_factor = abs(sum([t['pnl_pct'] for t in win_trades]) / sum([t['pnl_pct'] for t in loss_trades])) if loss_trades else 0

        max_win = max([t['pnl_pct'] for t in trades])
        max_loss = min([t['pnl_pct'] for t in trades])

        # 计算盈亏金额（假设初始资金50000元，每次全仓）
        initial_capital = 50000
        total_pnl_amount = 0
        capital = initial_capital

        for t in trades:
            pnl_amount = (t['exit_price'] - t['entry_price']) * CONTRACT_SIZE * (capital / (t['entry_price'] * CONTRACT_SIZE * MARGIN_RATE))
            # 简化计算：按百分比
            pnl_amount = capital * t['pnl_pct'] / 100
            capital += pnl_amount
            total_pnl_amount += pnl_amount

        final_return = (capital - initial_capital) / initial_capital * 100

        print(f"\n【做多交易统计】")
        print(f"  交易次数: {len(trades)} 次")
        print(f"  获胜次数: {len(win_trades)} 次")
        print(f"  失败次数: {len(loss_trades)} 次")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  平均盈利: {avg_win:.2f}%")
        print(f"  平均亏损: {avg_loss:.2f}%")
        print(f"  平均盈亏: {avg_pnl:.2f}%")
        print(f"  总盈亏: {total_pnl_pct:.2f}%")
        print(f"  最终收益率: {final_return:.2f}%")
        print(f"  盈亏比: {profit_factor:.2f}")
        print(f"  最大盈利: {max_win:.2f}%")
        print(f"  最大亏损: {max_loss:.2f}%")

        # ========== 与之前优化结果对比 ==========
        print(f"\n{'='*100}")
        print("与之前优化结果对比")
        print(f"{'='*100}")

        print(f"\n{'项目':<20} {'之前优化结果':<20} {'本次回测':<20} {'差异':<20}")
        print(f"{'-'*80}")
        print(f"{'交易次数':<20} {'13笔':<20} {f'{len(trades)}笔':<20} {f'{len(trades)-13}笔':<20}")
        print(f"{'胜率':<20} {'84.6%':<20} {f'{win_rate:.1f}%':<20} {f'{win_rate-84.6:+.1f}%':<20}")
        print(f"{'收益率':<20} {'+167.47%':<20} {f'{final_return:+.2f}%':<20} {f'{final_return-167.47:+.2f}%':<20}")

        if len(trades) == 13 and abs(win_rate - 84.6) < 1:
            print(f"\n[OK] 结果完全一致！参数验证通过")
        else:
            print(f"\n[WARNING] 结果不一致，需要进一步检查")

        # ========== 详细交易记录 ==========
        print(f"\n{'='*100}")
        print("详细交易记录")
        print(f"{'='*100}")

        print(f"\n{'-'*100}")
        print(f"{'序号':<5} {'信号':<15} {'入场时间':<19} {'入场价':<8} {'出场时间':<19} {'出场价':<8} "
              f"{'持仓':<8} {'盈亏%':<8} {'退出原因':<15}")
        print(f"{'-'*100}")

        for i, trade in enumerate(trades, 1):
            pnl_pct_str = f"{'+' if trade['pnl_pct'] >= 0 else ''}{trade['pnl_pct']:.2f}%"
            hold_hours = trade['hold_bars'] * 4
            hold_str = f"{hold_hours}h"

            print(f"{i:<5} {trade['signal_type']:<15} "
                  f"{str(trade['entry_datetime']):<19} {trade['entry_price']:<8.2f} "
                  f"{str(trade['exit_datetime']):<19} {trade['exit_price']:<8.2f} "
                  f"{hold_str:<8} {pnl_pct_str:<8} {trade['exit_reason']:<15}")

    else:
        print("\n没有产生任何交易")

if __name__ == '__main__':
    backtest_long_only_optimized_params()
