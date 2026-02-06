# -*- coding: utf-8 -*-
"""
PTA对冲策略分析 - 修正版
正确处理保证金和资金分配
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime

def analyze_hedging_corrected():
    """正确的对冲策略分析"""

    print("=" * 120)
    print("PTA对冲策略分析 - 修正版".center(120))
    print("=" * 120)

    # 合约规格
    CONTRACT_SIZE = 5
    MARGIN_RATE = 0.08
    STOP_LOSS_PCT = 0.02
    INITIAL_CAPITAL = 50000

    # 做多最优参数
    LONG_PARAMS = {
        'EMA_FAST': 12,
        'EMA_SLOW': 10,
        'RSI_FILTER': 40,
        'RATIO_TRIGGER': 1.25,
        'STC_SELL_ZONE': 75
    }

    # 做空最优参数
    SHORT_PARAMS = {
        'EMA_FAST': 7,
        'EMA_SLOW': 20,
        'RSI_FILTER': 45,
        'RATIO_TRIGGER': 1.05,
        'STC_BUY_ZONE': 15
    }

    print(f"\n【参数配置】")
    print(f"  做多: EMA({LONG_PARAMS['EMA_FAST']}, {LONG_PARAMS['EMA_SLOW']}), RSI={LONG_PARAMS['RSI_FILTER']}")
    print(f"  做空: EMA({SHORT_PARAMS['EMA_FAST']}, {SHORT_PARAMS['EMA_SLOW']}), RSI={SHORT_PARAMS['RSI_FILTER']}")

    # 获取数据
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('TA', days=300)

    if df is None or df.empty:
        print("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 计算指标（做多和做空各一套）
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

    # RSI（共用）
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # STC（共用）
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

    # ========== 回测逻辑（修正版） ==========
    capital = INITIAL_CAPITAL
    position_long = None
    position_short = None
    all_trades = []

    print(f"\n开始回测...")
    print(f"  保证金率: {MARGIN_RATE * 100}%")
    print(f"  最大仓位: 80%（单边）")
    print(f"  最大仓位: 80%（对冲时多空各40%）")

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # ========== 持仓管理（多空分别管理） ==========

        # 管理多仓
        if position_long is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if current['low'] <= position_long['stop_loss']:
                exit_price = position_long['stop_loss']
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

            if exit_triggered:
                pnl = (exit_price - position_long['entry_price']) * position_long['contracts'] * CONTRACT_SIZE
                capital += pnl

                trade = {
                    'entry_datetime': position_long['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'direction': 'long',
                    'signal_type': position_long['signal_type'],
                    'entry_price': position_long['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position_long['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / position_long['capital_at_entry']) * 100,
                    'capital_after': capital,
                    'exit_reason': exit_reason,
                    'is_hedge': position_long.get('is_hedge', False),
                    'margin_used': position_long['margin_used']
                }
                all_trades.append(trade)
                position_long = None

        # 管理空仓
        if position_short is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if current['high'] >= position_short['stop_loss']:
                exit_price = position_short['stop_loss']
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
                pnl = (position_short['entry_price'] - exit_price) * position_short['contracts'] * CONTRACT_SIZE
                capital += pnl

                trade = {
                    'entry_datetime': position_short['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'direction': 'short',
                    'signal_type': position_short['signal_type'],
                    'entry_price': position_short['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position_short['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / position_short['capital_at_entry']) * 100,
                    'capital_after': capital,
                    'exit_reason': exit_reason,
                    'is_hedge': position_short.get('is_hedge', False),
                    'margin_used': position_short['margin_used']
                }
                all_trades.append(trade)
                position_short = None

        # ========== 开仓信号检查 ==========
        if position_long is not None or position_short is not None:
            continue  # 有持仓就不开新仓

        # 检查做多信号
        trend_up_long = current['ema_fast_long'] > current['ema_slow_long']
        ratio_safe_long = (0 < current['ratio_long'] < LONG_PARAMS['RATIO_TRIGGER'])
        ratio_shrinking_long = current['ratio_long'] < prev['ratio_long']
        turning_up_long = current['macd_dif_long'] > prev['macd_dif_long']
        is_strong_long = current['rsi'] > LONG_PARAMS['RSI_FILTER']

        sniper_long = trend_up_long and ratio_safe_long and ratio_shrinking_long and turning_up_long and is_strong_long
        ema_golden_cross = (prev['ema_fast_long'] <= prev['ema_slow_long']) and (current['ema_fast_long'] > current['ema_slow_long'])
        chase_long = ema_golden_cross and is_strong_long

        long_signal = sniper_long or chase_long

        # 检查做空信号
        trend_down_short = current['ema_fast_short'] < current['ema_slow_short']
        ratio_safe_short = (-SHORT_PARAMS['RATIO_TRIGGER'] < current['ratio_short'] < 0)
        ratio_falling_short = current['ratio_short'] < prev['ratio_short']
        turning_down_short = current['macd_dif_short'] < prev['macd_dif_short']
        is_weak_short = current['rsi'] < SHORT_PARAMS['RSI_FILTER']

        sniper_short = trend_down_short and ratio_safe_short and ratio_falling_short and turning_down_short and is_weak_short
        ema_death_cross = (prev['ema_fast_short'] >= prev['ema_slow_short']) and (current['ema_fast_short'] < current['ema_slow_short'])
        chase_short = ema_death_cross and is_weak_short

        short_signal = sniper_short or chase_short

        # ========== 对冲逻辑（修正版） ==========

        if long_signal and short_signal:
            # 对冲：同时开多空
            print(f"\n[{current['datetime']}] 检测到对冲信号！")

            entry_price = current['close']
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATE

            # ========== 修正：正确计算手数 ==========
            # 对冲时总共用80%资金，多空各40%
            total_margin_available = capital * 0.80
            margin_per_side = total_margin_available / 2  # 每边40%

            # 多仓手数
            potential_loss_long = entry_price * STOP_LOSS_PCT * CONTRACT_SIZE
            max_contracts_by_margin_long = int(margin_per_side / margin_per_contract)
            max_contracts_by_risk_long = int((capital * 0.15) / potential_loss_long)  # 每边风险15%
            contracts_long = min(max_contracts_by_margin_long, max_contracts_by_risk_long)

            # 空仓手数
            potential_loss_short = entry_price * STOP_LOSS_PCT * CONTRACT_SIZE
            max_contracts_by_margin_short = int(margin_per_side / margin_per_contract)
            max_contracts_by_risk_short = int((capital * 0.15) / potential_loss_short)
            contracts_short = min(max_contracts_by_margin_short, max_contracts_by_risk_short)

            if contracts_long > 0 and contracts_short > 0:
                # 计算实际保证金占用
                margin_used_long = contracts_long * margin_per_contract
                margin_used_short = contracts_short * margin_per_contract
                total_margin_used = margin_used_long + margin_used_short
                margin_ratio = total_margin_used / capital

                print(f"  开多仓: {contracts_long}手, 保证金{margin_used_long:,.0f}元 ({margin_used_long/capital*100:.1f}%)")
                print(f"  开空仓: {contracts_short}手, 保证金{margin_used_short:,.0f}元 ({margin_used_short/capital*100:.1f}%)")
                print(f"  总占用: {total_margin_used:,.0f}元 ({margin_ratio*100:.1f}%)")

                # 开多仓
                position_long = {
                    'entry_datetime': current['datetime'],
                    'entry_price': entry_price,
                    'contracts': contracts_long,
                    'stop_loss': entry_price * (1 - STOP_LOSS_PCT),
                    'signal_type': 'sniper_long' if sniper_long else 'chase_long',
                    'capital_at_entry': capital,
                    'is_hedge': True,
                    'margin_used': margin_used_long
                }

                # 开空仓
                position_short = {
                    'entry_datetime': current['datetime'],
                    'entry_price': entry_price,
                    'contracts': contracts_short,
                    'stop_loss': entry_price * (1 + STOP_LOSS_PCT),
                    'signal_type': 'sniper_short' if sniper_short else 'chase_short',
                    'capital_at_entry': capital,
                    'is_hedge': True,
                    'margin_used': margin_used_short
                }

        elif long_signal:
            # 只做多
            entry_price = current['close']
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATE
            potential_loss = entry_price * STOP_LOSS_PCT * CONTRACT_SIZE

            max_contracts_by_margin = int((capital * 0.80) / margin_per_contract)
            max_contracts_by_risk = int((capital * 0.15) / potential_loss)
            contracts = min(max_contracts_by_margin, max_contracts_by_risk)

            if contracts > 0:
                margin_used = contracts * margin_per_contract

                position_long = {
                    'entry_datetime': current['datetime'],
                    'entry_price': entry_price,
                    'contracts': contracts,
                    'stop_loss': entry_price * (1 - STOP_LOSS_PCT),
                    'signal_type': 'sniper_long' if sniper_long else 'chase_long',
                    'capital_at_entry': capital,
                    'is_hedge': False,
                    'margin_used': margin_used
                }

        elif short_signal:
            # 只做空
            entry_price = current['close']
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATE
            potential_loss = entry_price * STOP_LOSS_PCT * CONTRACT_SIZE

            max_contracts_by_margin = int((capital * 0.80) / margin_per_contract)
            max_contracts_by_risk = int((capital * 0.15) / potential_loss)
            contracts = min(max_contracts_by_margin, max_contracts_by_risk)

            if contracts > 0:
                margin_used = contracts * margin_per_contract

                position_short = {
                    'entry_datetime': current['datetime'],
                    'entry_price': entry_price,
                    'contracts': contracts,
                    'stop_loss': entry_price * (1 + STOP_LOSS_PCT),
                    'signal_type': 'sniper_short' if sniper_short else 'chase_short',
                    'capital_at_entry': capital,
                    'is_hedge': False,
                    'margin_used': margin_used
                }

    # 统计结果
    if not all_trades:
        print("\n无交易记录")
        return

    trades_df = pd.DataFrame(all_trades)

    # 分离对冲和非对冲交易
    hedge_trades = trades_df[trades_df['is_hedge'] == True]
    normal_trades = trades_df[trades_df['is_hedge'] == False]

    long_trades = trades_df[trades_df['direction'] == 'long']
    short_trades = trades_df[trades_df['direction'] == 'short']

    print(f"\n{'='*120}")
    print("回测结果（修正版）")
    print(f"{'='*120}")

    print(f"\n【总体统计】")
    print(f"  总交易: {len(trades_df)}笔")
    print(f"  对冲交易: {len(hedge_trades)//2}组 ({len(hedge_trades)}笔)")
    print(f"  普通交易: {len(normal_trades)}笔")
    print(f"  初始资金: {INITIAL_CAPITAL:,.2f}元")
    print(f"  最终资金: {capital:,.2f}元")
    print(f"  总盈亏: {capital - INITIAL_CAPITAL:+,.2f}元 ({(capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100:+.2f}%)")

    print(f"\n【做多统计】")
    print(f"  交易: {len(long_trades)}笔")
    if len(long_trades) > 0:
        long_win = len(long_trades[long_trades['pnl'] > 0])
        print(f"  胜率: {long_win / len(long_trades) * 100:.1f}%")
        print(f"  盈亏: {long_trades['pnl'].sum():+,.2f}元")

    print(f"\n【做空统计】")
    print(f"  交易: {len(short_trades)}笔")
    if len(short_trades) > 0:
        short_win = len(short_trades[short_trades['pnl'] > 0])
        print(f"  胜率: {short_win / len(short_trades) * 100:.1f}%")
        print(f"  盈亏: {short_trades['pnl'].sum():+,.2f}元")

    if len(hedge_trades) > 0:
        print(f"\n【对冲交易分析】")
        hedge_pnl = hedge_trades['pnl'].sum()
        hedge_win = len(hedge_trades[hedge_trades['pnl'] > 0])
        print(f"  对冲交易: {len(hedge_trades)}笔")
        print(f"  胜率: {hedge_win / len(hedge_trades) * 100:.1f}%")
        print(f"  盈亏: {hedge_pnl:+,.2f}元")

        # 保证金占用
        avg_margin = hedge_trades['margin_used'].mean()
        max_margin = hedge_trades['margin_used'].max()
        print(f"  平均保证金占用: {avg_margin:,.0f}元 ({avg_margin/INITIAL_CAPITAL*100:.1f}%)")
        print(f"  最大保证金占用: {max_margin:,.0f}元 ({max_margin/INITIAL_CAPITAL*100:.1f}%)")

    # 详细交易
    print(f"\n{'='*120}")
    print("详细交易记录")
    print(f"{'='*120}")

    for i, trade in enumerate(all_trades, 1):
        direction_str = '做多' if trade['direction'] == 'long' else '做空'
        hedge_str = '[对冲]' if trade['is_hedge'] else '[单边]'
        pnl_str = f"{'+' if trade['pnl'] >= 0 else ''}{trade['pnl']:.2f}"
        pnl_pct_str = f"{'+' if trade['pnl_pct'] >= 0 else ''}{trade['pnl_pct']:.2f}%"
        margin_str = f"保证金{trade['margin_used']:,.0f}元"

        print(f"{i}. {hedge_str} {direction_str} | {trade['entry_datetime']} ~ {trade['exit_datetime']} | "
              f"{trade['entry_price']:.2f} -> {trade['exit_price']:.2f} | {pnl_str} ({pnl_pct_str}) | {margin_str} | {trade['exit_reason']}")

    return trades_df

if __name__ == '__main__':
    analyze_hedging_corrected()
