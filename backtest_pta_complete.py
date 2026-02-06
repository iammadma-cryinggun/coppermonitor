# -*- coding: utf-8 -*-
"""
PTA完整回测 - 测试做多和做空的能力、胜率、盈亏比
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime

def backtest_pta_complete():
    """完整回测PTA做多和做空"""

    print("=" * 100)
    print("PTA做多做空完整回测分析")
    print("=" * 100)

    # ========== 配置参数 ==========
    # 做多参数
    RSI_FILTER_LONG = 40
    RATIO_TRIGGER_LONG = 1.25
    STOP_LOSS_PCT_LONG = 0.02  # 2%

    # 做空参数
    RSI_FILTER_SHORT = 60  # 100 - 40
    RATIO_TRIGGER_SHORT = 1.25

    # 平仓参数
    STC_SELL_ZONE = 75  # 平多仓
    STC_BUY_ZONE = 25   # 平空仓

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

    # STC指标
    def calculate_stc(series, fast_period=23, slow_period=50, period=10):
        """计算STC指标"""
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
    trades = []  # 所有交易
    position = None  # 当前持仓

    print(f"\n参数配置:")
    print(f"  做多: RSI_FILTER={RSI_FILTER_LONG}, RATIO_TRIGGER={RATIO_TRIGGER_LONG}, STOP_LOSS={STOP_LOSS_PCT_LONG*100}%")
    print(f"  做空: RSI_FILTER={RSI_FILTER_SHORT}, RATIO_TRIGGER={RATIO_TRIGGER_SHORT}")
    print(f"  平仓: STC_SELL_ZONE={STC_SELL_ZONE}, STC_BUY_ZONE={STC_BUY_ZONE}")

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
            if position['direction'] == 'long':
                # 做多止损：最低价触及
                if current['low'] <= position['stop_loss']:
                    exit_triggered = True
                    exit_reason = 'stop_loss_long'
                    exit_price = current['close']  # 按收盘价止损
            elif position['direction'] == 'short':
                # 做空止损：最高价触及
                if current['high'] >= position['stop_loss']:
                    exit_triggered = True
                    exit_reason = 'stop_loss_short'
                    exit_price = current['close']

            # 平仓信号检查
            if not exit_triggered:
                if position['direction'] == 'long':
                    # 平多仓信号
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
                    # 平空仓信号
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

            # 如果触发平仓
            if exit_triggered:
                # 计算盈亏
                if position['direction'] == 'long':
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                else:  # short
                    pnl_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100

                # 记录交易
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

        # 如果没有持仓，检查开仓信号
        if position is None:
            # ========== 做多信号 ==========
            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe_long = (0 < current['ratio'] < RATIO_TRIGGER_LONG)
            ratio_shrinking_long = current['ratio'] < prev['ratio']
            turning_up_long = current['macd_dif'] > prev['macd_dif']
            is_strong_long = current['rsi'] > RSI_FILTER_LONG

            sniper_long = trend_up and ratio_safe_long and ratio_shrinking_long and turning_up_long and is_strong_long

            ema_golden_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            chase_long = ema_golden_cross and is_strong_long

            # ========== 做空信号 ==========
            trend_down = current['ema_fast'] < current['ema_slow']
            ratio_safe_short = (-RATIO_TRIGGER_SHORT < current['ratio'] < 0)
            ratio_falling_short = current['ratio'] < prev['ratio']
            turning_down_short = current['macd_dif'] < prev['macd_dif']
            is_weak_short = current['rsi'] < RSI_FILTER_SHORT

            sniper_short = trend_down and ratio_safe_short and ratio_falling_short and turning_down_short and is_weak_short

            ema_death_cross = (prev['ema_fast'] >= prev['ema_slow']) and (current['ema_fast'] < current['ema_slow'])
            chase_short = ema_death_cross and is_weak_short

            # 开仓（做多优先）
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
                stop_loss = current['close'] * (1 + STOP_LOSS_PCT_LONG)  # 做空止损在上方

                position = {
                    'direction': 'short',
                    'signal_type': signal_type,
                    'entry_datetime': current['datetime'],
                    'entry_price': current['close'],
                    'stop_loss': stop_loss,
                    'entry_index': i
                }

    # ========== 统计结果 ==========
    print(f"\n{'='*100}")
    print("回测结果统计")
    print(f"{'='*100}")

    # 分离做多和做空交易
    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']

    # 做多统计
    if long_trades:
        long_win = [t for t in long_trades if t['pnl_pct'] > 0]
        long_loss = [t for t in long_trades if t['pnl_pct'] <= 0]

        long_win_rate = len(long_win) / len(long_trades) * 100
        long_avg_win = np.mean([t['pnl_pct'] for t in long_win]) if long_win else 0
        long_avg_loss = np.mean([t['pnl_pct'] for t in long_loss]) if long_loss else 0
        long_total_pnl = sum([t['pnl_pct'] for t in long_trades])
        long_avg_pnl = np.mean([t['pnl_pct'] for t in long_trades])
        long_profit_factor = abs(sum([t['pnl_pct'] for t in long_win]) / sum([t['pnl_pct'] for t in long_loss])) if long_loss else 0

        max_long_win = max([t['pnl_pct'] for t in long_trades])
        max_long_loss = min([t['pnl_pct'] for t in long_trades])
    else:
        long_win_rate = long_avg_win = long_avg_loss = long_total_pnl = long_avg_pnl = long_profit_factor = 0
        max_long_win = max_long_loss = 0

    # 做空统计
    if short_trades:
        short_win = [t for t in short_trades if t['pnl_pct'] > 0]
        short_loss = [t for t in short_trades if t['pnl_pct'] <= 0]

        short_win_rate = len(short_win) / len(short_trades) * 100
        short_avg_win = np.mean([t['pnl_pct'] for t in short_win]) if short_win else 0
        short_avg_loss = np.mean([t['pnl_pct'] for t in short_loss]) if short_loss else 0
        short_total_pnl = sum([t['pnl_pct'] for t in short_trades])
        short_avg_pnl = np.mean([t['pnl_pct'] for t in short_trades])
        short_profit_factor = abs(sum([t['pnl_pct'] for t in short_win]) / sum([t['pnl_pct'] for t in short_loss])) if short_loss else 0

        max_short_win = max([t['pnl_pct'] for t in short_trades])
        max_short_loss = min([t['pnl_pct'] for t in short_trades])
    else:
        short_win_rate = short_avg_win = short_avg_loss = short_total_pnl = short_avg_pnl = short_profit_factor = 0
        max_short_win = max_short_loss = 0

    # 总体统计
    all_trades = trades
    if all_trades:
        all_win = [t for t in all_trades if t['pnl_pct'] > 0]
        all_loss = [t for t in all_trades if t['pnl_pct'] <= 0]

        all_win_rate = len(all_win) / len(all_trades) * 100
        all_total_pnl = sum([t['pnl_pct'] for t in all_trades])
        all_avg_pnl = np.mean([t['pnl_pct'] for t in all_trades])
        all_profit_factor = abs(sum([t['pnl_pct'] for t in all_win]) / sum([t['pnl_pct'] for t in all_loss])) if all_loss else 0
    else:
        all_win_rate = all_total_pnl = all_avg_pnl = all_profit_factor = 0

    # ========== 打印结果 ==========
    print(f"\n【做多交易统计】")
    print(f"  交易次数: {len(long_trades)} 次")
    print(f"  胜率: {long_win_rate:.1f}% ({len(long_win)}胜 / {len(long_loss)}负)")
    print(f"  平均盈利: {long_avg_win:.2f}%")
    print(f"  平均亏损: {long_avg_loss:.2f}%")
    print(f"  平均盈亏: {long_avg_pnl:.2f}%")
    print(f"  总盈亏: {long_total_pnl:.2f}%")
    print(f"  盈亏比: {long_profit_factor:.2f}")
    print(f"  最大盈利: {max_long_win:.2f}%")
    print(f"  最大亏损: {max_long_loss:.2f}%")

    print(f"\n【做空交易统计】")
    print(f"  交易次数: {len(short_trades)} 次")
    print(f"  胜率: {short_win_rate:.1f}% ({len(short_win)}胜 / {len(short_loss)}负)")
    print(f"  平均盈利: {short_avg_win:.2f}%")
    print(f"  平均亏损: {short_avg_loss:.2f}%")
    print(f"  平均盈亏: {short_avg_pnl:.2f}%")
    print(f"  总盈亏: {short_total_pnl:.2f}%")
    print(f"  盈亏比: {short_profit_factor:.2f}")
    print(f"  最大盈利: {max_short_win:.2f}%")
    print(f"  最大亏损: {max_short_loss:.2f}%")

    print(f"\n【总体统计】")
    print(f"  总交易次数: {len(all_trades)} 次")
    print(f"  总胜率: {all_win_rate:.1f}%")
    print(f"  总盈亏: {all_total_pnl:.2f}%")
    print(f"  平均盈亏: {all_avg_pnl:.2f}%")
    print(f"  盈亏比: {all_profit_factor:.2f}")

    # ========== 对比分析 ==========
    print(f"\n{'='*100}")
    print("多空对比分析")
    print(f"{'='*100}")

    print(f"\n{'指标':<15} {'做多':<20} {'做空':<20} {'更优':<10}")
    print(f"{'-'*70}")

    print(f"{'交易次数':<15} {len(long_trades):<20} {len(short_trades):<20} {'做多' if len(long_trades) > len(short_trades) else '做空'}")
    print(f"{'胜率':<15} {long_win_rate:<20.2f} {short_win_rate:<20.2f} {'做多' if long_win_rate > short_win_rate else '做空'}")
    print(f"{'平均盈亏':<15} {long_avg_pnl:<20.2f} {short_avg_pnl:<20.2f} {'做多' if long_avg_pnl > short_avg_pnl else '做空'}")
    print(f"{'总盈亏':<15} {long_total_pnl:<20.2f} {short_total_pnl:<20.2f} {'做多' if long_total_pnl > short_total_pnl else '做空'}")
    print(f"{'盈亏比':<15} {long_profit_factor:<20.2f} {short_profit_factor:<20.2f} {'做多' if long_profit_factor > short_profit_factor else '做空'}")

    # ========== 退出原因统计 ==========
    print(f"\n{'='*100}")
    print("退出原因统计")
    print(f"{'='*100}")

    exit_reasons_long = {}
    exit_reasons_short = {}

    for t in trades:
        if t['direction'] == 'long':
            exit_reasons_long[t['exit_reason']] = exit_reasons_long.get(t['exit_reason'], 0) + 1
        else:
            exit_reasons_short[t['exit_reason']] = exit_reasons_short.get(t['exit_reason'], 0) + 1

    print(f"\n做多退出原因:")
    for reason, count in exit_reasons_long.items():
        print(f"  {reason}: {count} 次")

    print(f"\n做空退出原因:")
    for reason, count in exit_reasons_short.items():
        print(f"  {reason}: {count} 次")

    # ========== 合约规格 ==========
    CONTRACT_SIZE = 5  # PTA合约乘数：5吨/手
    MARGIN_RATE = 0.08  # 保证金率：8%

    # ========== 详细交易记录（全部） ==========
    print(f"\n{'='*140}")
    print("详细交易记录（全部）")
    print(f"{'='*140}")
    print(f"\n合约规格：PTA (TA)")
    print(f"  合约乘数：{CONTRACT_SIZE} 吨/手")
    print(f"  保证金率：{MARGIN_RATE*100}%")
    print(f"  假设：1手交易")

    print(f"\n{'-'*140}")
    print(f"{'序号':<5} {'方向':<6} {'信号':<15} {'入场时间':<19} {'入场价':<8} {'出场时间':<19} {'出场价':<8} {'保证金':<10} {'盈亏':<10} {'盈亏%':<8} {'原因':<20}")
    print(f"{'-'*140}")

    for i, trade in enumerate(trades, 1):
        direction_symbol = '做多' if trade['direction'] == 'long' else '做空'
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        margin = entry_price * CONTRACT_SIZE * MARGIN_RATE  # 保证金

        # 计算盈亏金额
        if trade['direction'] == 'long':
            pnl_amount = (exit_price - entry_price) * CONTRACT_SIZE
        else:  # short
            pnl_amount = (entry_price - exit_price) * CONTRACT_SIZE

        pnl_pct = trade['pnl_pct']
        pnl_str = f"{'+' if pnl_amount >= 0 else ''}{pnl_amount:.2f}"

        print(f"{i:<5} {direction_symbol:<6} {trade['signal_type']:<15} "
              f"{str(trade['entry_datetime']):<19} {entry_price:<8.2f} "
              f"{str(trade['exit_datetime']):<19} {exit_price:<8.2f} "
              f"{margin:<10.2f} {pnl_str:<10} {pnl_pct:<8.2f} {trade['exit_reason']:<20}")

    # ========== 保存交易记录 ==========
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        output_file = Path('logs/pta_backtest_trades.csv')
        output_file.parent.mkdir(exist_ok=True)
        trades_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n交易记录已保存到: {output_file}")

    # ========== 结论 ==========
    print(f"\n{'='*100}")
    print("结论")
    print(f"{'='*100}")

    if long_total_pnl > short_total_pnl:
        print(f"\n1. 做多表现更好（总盈亏: {long_total_pnl:.2f}% vs {short_total_pnl:.2f}%）")
    elif short_total_pnl > long_total_pnl:
        print(f"\n1. 做空表现更好（总盈亏: {short_total_pnl:.2f}% vs {long_total_pnl:.2f}%）")
    else:
        print(f"\n1. 多空表现持平")

    if long_win_rate > short_win_rate:
        print(f"\n2. 做多胜率更高（{long_win_rate:.1f}% vs {short_win_rate:.1f}%）")
    elif short_win_rate > long_win_rate:
        print(f"\n2. 做空胜率更高（{short_win_rate:.1f}% vs {long_win_rate:.1f}%）")

    print(f"\n3. 建议:")
    if all_total_pnl > 0:
        print(f"   [OK] 总体盈利，策略可行")
        print(f"   [OK] 双向交易可以启用")
    else:
        print(f"   [WARNING] 总体亏损，需要优化参数")

    print(f"\n回测完成时间: {datetime.now()}")

if __name__ == '__main__':
    backtest_pta_complete()
