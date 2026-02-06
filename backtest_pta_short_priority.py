# -*- coding: utf-8 -*-
"""
PTA双向回测 - 做空优先版本
验证：优先选择参数更严格、历史胜率更高的做空信号
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime

def backtest_short_priority():
    """做空优先的双向回测"""

    print("=" * 120)
    print("PTA双向回测 - 做空优先版本".center(120))
    print("=" * 120)

    CONTRACT_SIZE = 5
    MARGIN_RATE = 0.08
    STOP_LOSS_PCT = 0.02
    INITIAL_CAPITAL = 50000
    MAX_POSITION_RATIO = 0.80
    MAX_SINGLE_LOSS_PCT = 0.15

    # 做多最优参数
    LONG_PARAMS = {
        'EMA_FAST': 12,
        'EMA_SLOW': 10,
        'RSI_FILTER': 40,
        'RATIO_TRIGGER': 1.25,
        'STC_SELL_ZONE': 75
    }

    # 做空最优参数（更严格）
    SHORT_PARAMS = {
        'EMA_FAST': 7,
        'EMA_SLOW': 20,
        'RSI_FILTER': 45,
        'RATIO_TRIGGER': 1.05,
        'STC_BUY_ZONE': 15
    }

    print(f"\n【策略：做空优先】")
    print(f"  理由1：做空参数更严格（RSI<45 vs >40, RATIO<1.05 vs <1.25）")
    print(f"  理由2：做空历史胜率100% vs 做多77.8%")
    print(f"  理由3：可执行（不需要预知未来）")

    # 获取数据
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('TA', days=300)

    if df is None or df.empty:
        print("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 计算指标
    df['ema_fast_long'] = df['close'].ewm(span=LONG_PARAMS['EMA_FAST'], adjust=False).mean()
    df['ema_slow_long'] = df['close'].ewm(span=LONG_PARAMS['EMA_SLOW'], adjust=False).mean()
    exp1_long = df['close'].ewm(span=LONG_PARAMS['EMA_FAST'], adjust=False).mean()
    exp2_long = df['close'].ewm(span=LONG_PARAMS['EMA_SLOW'], adjust=False).mean()
    df['macd_dif_long'] = exp1_long - exp2_long
    df['macd_dea_long'] = df['macd_dif_long'].ewm(span=9, adjust=False).mean()
    df['ratio_long'] = np.where(df['macd_dea_long'] != 0, df['macd_dif_long'] / df['macd_dea_long'], 0)

    df['ema_fast_short'] = df['close'].ewm(span=SHORT_PARAMS['EMA_FAST'], adjust=False).mean()
    df['ema_slow_short'] = df['close'].ewm(span=SHORT_PARAMS['EMA_SLOW'], adjust=False).mean()
    exp1_short = df['close'].ewm(span=SHORT_PARAMS['EMA_FAST'], adjust=False).mean()
    exp2_short = df['close'].ewm(span=SHORT_PARAMS['EMA_SLOW'], adjust=False).mean()
    df['macd_dif_short'] = exp1_short - exp2_short
    df['macd_dea_short'] = df['macd_dif_short'].ewm(span=9, adjust=False).mean()
    df['ratio_short'] = np.where(df['macd_dea_short'] != 0, df['macd_dif_short'] / df['macd_dea_short'], 0)

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

    # 回测
    capital = INITIAL_CAPITAL
    position = None
    all_trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 持仓管理
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if position['direction'] == 'long':
                if current['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_triggered = True
                    exit_reason = '止损'
                else:
                    stc_exit = (prev['stc'] > LONG_PARAMS['STC_SELL_ZONE']) and (current['stc'] < prev['stc'])
                    trend_exit = current['ema_fast_long'] < current['ema_slow_long']

                    if stc_exit:
                        exit_price = current['close']
                        exit_triggered = True
                        exit_reason = 'STC止盈'
                    elif trend_exit:
                        exit_price = current['close']
                        exit_triggered = True
                        exit_reason = '趋势反转'

            elif position['direction'] == 'short':
                if current['high'] >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_triggered = True
                    exit_reason = '止损'
                else:
                    stc_exit = (prev['stc'] < SHORT_PARAMS['STC_BUY_ZONE']) and (current['stc'] > prev['stc'])
                    trend_exit = current['ema_fast_short'] > current['ema_slow_short']

                    if stc_exit:
                        exit_price = current['close']
                        exit_triggered = True
                        exit_reason = 'STC止盈'
                    elif trend_exit:
                        exit_price = current['close']
                        exit_triggered = True
                        exit_reason = '趋势反转'

            if exit_triggered:
                if position['direction'] == 'long':
                    pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                else:
                    pnl = (position['entry_price'] - exit_price) * position['contracts'] * CONTRACT_SIZE

                capital += pnl

                trade = {
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'direction': position['direction'],
                    'signal_type': position['signal_type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / position['capital_at_entry']) * 100,
                    'capital_after': capital,
                    'exit_reason': exit_reason
                }
                all_trades.append(trade)
                position = None
                continue

        # 开仓信号检查（做空优先！）
        if position is None:
            # 先评估做空
            trend_down_short = current['ema_fast_short'] < current['ema_slow_short']
            ratio_safe_short = (-SHORT_PARAMS['RATIO_TRIGGER'] < current['ratio_short'] < 0)
            ratio_falling_short = current['ratio_short'] < prev['ratio_short']
            turning_down_short = current['macd_dif_short'] < prev['macd_dif_short']
            is_weak_short = current['rsi'] < SHORT_PARAMS['RSI_FILTER']

            sniper_short = trend_down_short and ratio_safe_short and ratio_falling_short and turning_down_short and is_weak_short
            ema_death_cross = (prev['ema_fast_short'] >= prev['ema_slow_short']) and (current['ema_fast_short'] < current['ema_slow_short'])
            chase_short = ema_death_cross and is_weak_short

            # 做多信号
            trend_up_long = current['ema_fast_long'] > current['ema_slow_long']
            ratio_safe_long = (0 < current['ratio_long'] < LONG_PARAMS['RATIO_TRIGGER'])
            ratio_shrinking_long = current['ratio_long'] < prev['ratio_long']
            turning_up_long = current['macd_dif_long'] > prev['macd_dif_long']
            is_strong_long = current['rsi'] > LONG_PARAMS['RSI_FILTER']

            sniper_long = trend_up_long and ratio_safe_long and ratio_shrinking_long and turning_up_long and is_strong_long
            ema_golden_cross = (prev['ema_fast_long'] <= prev['ema_slow_long']) and (current['ema_fast_long'] > current['ema_slow_long'])
            chase_long = ema_golden_cross and is_strong_long

            # ========== 关键：做空优先 ==========
            if sniper_short or chase_short:
                signal_type = 'sniper_short' if sniper_short else 'chase_short'
                direction = 'short'
                entry_price = current['close']
                stop_loss = entry_price * (1 + STOP_LOSS_PCT)

            elif sniper_long or chase_long:
                signal_type = 'sniper_long' if sniper_long else 'chase_long'
                direction = 'long'
                entry_price = current['close']
                stop_loss = entry_price * (1 - STOP_LOSS_PCT)

            else:
                continue

            # 计算手数
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATE

            if direction == 'long':
                potential_loss_per_contract = (entry_price - stop_loss) * CONTRACT_SIZE
            else:
                potential_loss_per_contract = (stop_loss - entry_price) * CONTRACT_SIZE

            max_contracts_by_risk = int((capital * MAX_SINGLE_LOSS_PCT) / potential_loss_per_contract)
            max_contracts_by_margin = int((capital * MAX_POSITION_RATIO) / margin_per_contract)
            contracts = min(max_contracts_by_margin, max_contracts_by_risk)

            if contracts <= 0:
                continue

            position = {
                'entry_datetime': current['datetime'],
                'entry_price': entry_price,
                'contracts': contracts,
                'stop_loss': stop_loss,
                'direction': direction,
                'signal_type': signal_type,
                'capital_at_entry': capital,
                'entry_index': i
            }

    # 统计结果
    if not all_trades:
        print("无交易记录")
        return

    trades_df = pd.DataFrame(all_trades)

    long_trades = trades_df[trades_df['direction'] == 'long']
    short_trades = trades_df[trades_df['direction'] == 'short']

    long_total = len(long_trades)
    long_win = len(long_trades[long_trades['pnl'] > 0])
    long_win_rate = (long_win / long_total * 100) if long_total > 0 else 0
    long_total_pnl = long_trades['pnl'].sum()
    long_total_pnl_pct = (long_total_pnl / INITIAL_CAPITAL * 100)

    short_total = len(short_trades)
    short_win = len(short_trades[short_trades['pnl'] > 0])
    short_win_rate = (short_win / short_total * 100) if short_total > 0 else 0
    short_total_pnl = short_trades['pnl'].sum()
    short_total_pnl_pct = (short_total_pnl / INITIAL_CAPITAL * 100)

    total_trades = len(trades_df)
    total_win = len(trades_df[trades_df['pnl'] > 0])
    total_win_rate = (total_win / total_trades * 100) if total_trades > 0 else 0
    total_pnl = trades_df['pnl'].sum()
    total_pnl_pct = (total_pnl / INITIAL_CAPITAL * 100)
    final_capital = capital

    print(f"\n{'='*120}")
    print("回测结果")
    print(f"{'='*120}")

    print(f"\n【做多统计】")
    print(f"  交易: {long_total}笔 | 胜率: {long_win_rate:.1f}% | 盈亏: {long_total_pnl:+,.2f}元 ({long_total_pnl_pct:+.2f}%)")

    print(f"\n【做空统计】")
    print(f"  交易: {short_total}笔 | 胜率: {short_win_rate:.1f}% | 盈亏: {short_total_pnl:+,.2f}元 ({short_total_pnl_pct:+.2f}%)")

    print(f"\n【总体统计】")
    print(f"  总交易: {total_trades}笔 | 胜率: {total_win_rate:.1f}%")
    print(f"  初始资金: {INITIAL_CAPITAL:,.2f}元")
    print(f"  最终资金: {final_capital:,.2f}元")
    print(f"  总盈亏: {total_pnl:+,.2f}元 ({total_pnl_pct:+.2f}%)")

    print(f"\n{'='*120}")
    print("对比分析")
    print(f"{'='*120}")

    print(f"\n{'策略':<25} {'收益率':<15} {'交易次数':<15} {'胜率':<15}")
    print(f"{'-'*70}")
    print(f"{'只做多':<25} {'+167.47%':<15} {'13笔':<15} {'84.6%':<15}")
    print(f"{'做多优先双向':<25} {'+100.94%':<15} {'10笔':<15} {'80.0%':<15}")
    print(f"{'做空优先双向':<25} {total_pnl_pct:+.2f}%{' ':<11} {total_trades}笔{' ':<11} {total_win_rate:.1f}%{' ':<11}")

    # 详细交易
    print(f"\n{'='*120}")
    print("详细交易记录")
    print(f"{'='*120}")

    for i, trade in enumerate(all_trades, 1):
        direction_str = '做多' if trade['direction'] == 'long' else '做空'
        pnl_str = f"{'+' if trade['pnl'] >= 0 else ''}{trade['pnl']:.2f}"
        pnl_pct_str = f"{'+' if trade['pnl_pct'] >= 0 else ''}{trade['pnl_pct']:.2f}%"

        print(f"{i}. {direction_str} | {trade['entry_datetime']} ~ {trade['exit_datetime']} | "
              f"{trade['entry_price']:.2f} -> {trade['exit_price']:.2f} | {pnl_str} ({pnl_pct_str}) | {trade['exit_reason']}")

    return trades_df

if __name__ == '__main__':
    backtest_short_priority()
