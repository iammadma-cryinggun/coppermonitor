# -*- coding: utf-8 -*-
"""
PTA多空双向回测 - 做多和做空分别使用各自优化的最佳参数
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime

def backtest_bidirectional_optimized():
    """做多和做空分别使用各自优化的最佳参数"""

    print("=" * 120)
    print("PTA多空双向回测 - 分别优化参数版本".center(120))
    print("=" * 120)

    # ========== 合约规格 ==========
    CONTRACT_SIZE = 5
    MARGIN_RATE = 0.08

    # ========== 做多最优参数 ==========
    LONG_PARAMS = {
        'EMA_FAST': 12,
        'EMA_SLOW': 10,
        'RSI_FILTER_LONG': 40,
        'RATIO_TRIGGER_LONG': 1.25,
        'STC_SELL_ZONE': 75
    }

    # ========== 做空最优参数 ==========
    SHORT_PARAMS = {
        'EMA_FAST': 7,
        'EMA_SLOW': 20,
        'RSI_FILTER_SHORT': 45,
        'RATIO_TRIGGER_SHORT': 1.05,
        'STC_BUY_ZONE': 15
    }

    # ========== 通用参数 ==========
    STOP_LOSS_PCT = 0.02
    INITIAL_CAPITAL = 50000
    MAX_POSITION_RATIO = 0.80
    MAX_SINGLE_LOSS_PCT = 0.15

    print(f"\n【做多参数（最优）】")
    print(f"  EMA({LONG_PARAMS['EMA_FAST']}, {LONG_PARAMS['EMA_SLOW']})")
    print(f"  RSI_FILTER={LONG_PARAMS['RSI_FILTER_LONG']}")
    print(f"  RATIO_TRIGGER={LONG_PARAMS['RATIO_TRIGGER_LONG']}")
    print(f"  STC_SELL_ZONE={LONG_PARAMS['STC_SELL_ZONE']}")

    print(f"\n【做空参数（最优）】")
    print(f"  EMA({SHORT_PARAMS['EMA_FAST']}, {SHORT_PARAMS['EMA_SLOW']})")
    print(f"  RSI_FILTER_SHORT={SHORT_PARAMS['RSI_FILTER_SHORT']}")
    print(f"  RATIO_TRIGGER_SHORT={SHORT_PARAMS['RATIO_TRIGGER_SHORT']}")
    print(f"  STC_BUY_ZONE={SHORT_PARAMS['STC_BUY_ZONE']}")

    print(f"\n【资金管理】")
    print(f"  合约: {CONTRACT_SIZE}吨/手, 保证金{MARGIN_RATE*100}%")
    print(f"  最大仓位: {MAX_POSITION_RATIO*100}%, 单笔最大亏损: {MAX_SINGLE_LOSS_PCT*100}%")

    # ========== 获取数据 ==========
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('TA', days=300)

    if df is None or df.empty:
        print("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n【数据范围】")
    print(f"  时间: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"  K线数: {len(df)}")

    # ========== 计算指标（做多和做空各一套） ==========
    # 做多指标
    df['ema_fast_long'] = df['close'].ewm(span=LONG_PARAMS['EMA_FAST'], adjust=False).mean()
    df['ema_slow_long'] = df['close'].ewm(span=LONG_PARAMS['EMA_SLOW'], adjust=False).mean()

    exp1_long = df['close'].ewm(span=LONG_PARAMS['EMA_FAST'], adjust=False).mean()
    exp2_long = df['close'].ewm(span=LONG_PARAMS['EMA_SLOW'], adjust=False).mean()
    df['macd_dif_long'] = exp1_long - exp2_long
    df['macd_dea_long'] = df['macd_dif_long'].ewm(span=9, adjust=False).mean()
    df['ratio_long'] = np.where(df['macd_dea_long'] != 0, df['macd_dif_long'] / df['macd_dea_long'], 0)

    # 做空指标
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

    # STC指标（共用）
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

    # ========== 开始回测 ==========
    capital = INITIAL_CAPITAL
    position = None
    all_trades = []

    print(f"\n开始回测...")

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # ========== 持仓管理 ==========
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 止损检查
            if position['direction'] == 'long':
                if current['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_triggered = True
                    exit_reason = '止损'
            elif position['direction'] == 'short':
                if current['high'] >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_triggered = True
                    exit_reason = '止损'

            # 平仓信号检查
            if not exit_triggered:
                if position['direction'] == 'long':
                    # 使用做多参数平仓
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
                    # 使用做空参数平仓
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

            # 执行平仓
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
                    'stop_loss': position['stop_loss'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / position['capital_at_entry']) * 100,
                    'capital_after': capital,
                    'exit_reason': exit_reason
                }
                all_trades.append(trade)
                position = None
                continue

        # ========== 开仓信号检查 ==========
        if position is None:
            # 做多信号（使用做多参数）
            trend_up_long = current['ema_fast_long'] > current['ema_slow_long']
            ratio_safe_long = (0 < current['ratio_long'] < LONG_PARAMS['RATIO_TRIGGER_LONG'])
            ratio_shrinking_long = current['ratio_long'] < prev['ratio_long']
            turning_up_long = current['macd_dif_long'] > prev['macd_dif_long']
            is_strong_long = current['rsi'] > LONG_PARAMS['RSI_FILTER_LONG']

            sniper_long = trend_up_long and ratio_safe_long and ratio_shrinking_long and turning_up_long and is_strong_long
            ema_golden_cross = (prev['ema_fast_long'] <= prev['ema_slow_long']) and (current['ema_fast_long'] > current['ema_slow_long'])
            chase_long = ema_golden_cross and is_strong_long

            # 做空信号（使用做空参数）
            trend_down_short = current['ema_fast_short'] < current['ema_slow_short']
            ratio_safe_short = (-SHORT_PARAMS['RATIO_TRIGGER_SHORT'] < current['ratio_short'] < 0)
            ratio_falling_short = current['ratio_short'] < prev['ratio_short']
            turning_down_short = current['macd_dif_short'] < prev['macd_dif_short']
            is_weak_short = current['rsi'] < SHORT_PARAMS['RSI_FILTER_SHORT']

            sniper_short = trend_down_short and ratio_safe_short and ratio_falling_short and turning_down_short and is_weak_short
            ema_death_cross = (prev['ema_fast_short'] >= prev['ema_slow_short']) and (current['ema_fast_short'] < current['ema_slow_short'])
            chase_short = ema_death_cross and is_weak_short

            # 开仓（做多优先）
            if sniper_long or chase_long:
                signal_type = 'sniper_long' if sniper_long else 'chase_long'
                direction = 'long'
                entry_price = current['close']
                stop_loss = entry_price * (1 - STOP_LOSS_PCT)

            elif sniper_short or chase_short:
                signal_type = 'sniper_short' if sniper_short else 'chase_short'
                direction = 'short'
                entry_price = current['close']
                stop_loss = entry_price * (1 + STOP_LOSS_PCT)

            else:
                continue

            # 计算合约手数
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

    # ========== 统计结果 ==========
    print(f"\n{'='*120}")
    print("回测结果统计")
    print(f"{'='*120}")

    if not all_trades:
        print("无交易记录")
        return

    trades_df = pd.DataFrame(all_trades)

    # 分离做多和做空
    long_trades = trades_df[trades_df['direction'] == 'long']
    short_trades = trades_df[trades_df['direction'] == 'short']

    # 做多统计
    long_total = len(long_trades)
    long_win = len(long_trades[long_trades['pnl'] > 0])
    long_loss = len(long_trades[long_trades['pnl'] <= 0])
    long_win_rate = (long_win / long_total * 100) if long_total > 0 else 0
    long_total_pnl = long_trades['pnl'].sum()
    long_total_pnl_pct = (long_total_pnl / INITIAL_CAPITAL * 100)
    long_avg_pnl_pct = long_trades['pnl_pct'].mean()

    # 做空统计
    short_total = len(short_trades)
    short_win = len(short_trades[short_trades['pnl'] > 0])
    short_loss = len(short_trades[short_trades['pnl'] <= 0])
    short_win_rate = (short_win / short_total * 100) if short_total > 0 else 0
    short_total_pnl = short_trades['pnl'].sum()
    short_total_pnl_pct = (short_total_pnl / INITIAL_CAPITAL * 100)
    short_avg_pnl_pct = short_trades['pnl_pct'].mean()

    # 总体统计
    total_trades = len(trades_df)
    total_win = len(trades_df[trades_df['pnl'] > 0])
    total_win_rate = (total_win / total_trades * 100) if total_trades > 0 else 0
    total_pnl = trades_df['pnl'].sum()
    total_pnl_pct = (total_pnl / INITIAL_CAPITAL * 100)
    final_capital = capital

    print(f"\n【做多统计（使用做多最优参数）】")
    print(f"  交易次数: {long_total} 笔")
    print(f"  获胜: {long_win} 笔 | 失败: {long_loss} 笔")
    print(f"  胜率: {long_win_rate:.1f}%")
    print(f"  总盈亏: {long_total_pnl:+,.2f} 元 ({long_total_pnl_pct:+.2f}%)")
    print(f"  平均盈亏: {long_avg_pnl_pct:+.2f}%")

    print(f"\n【做空统计（使用做空最优参数）】")
    print(f"  交易次数: {short_total} 笔")
    print(f"  获胜: {short_win} 笔 | 失败: {short_loss} 笔")
    print(f"  胜率: {short_win_rate:.1f}%")
    print(f"  总盈亏: {short_total_pnl:+,.2f} 元 ({short_total_pnl_pct:+.2f}%)")
    print(f"  平均盈亏: {short_avg_pnl_pct:+.2f}%")

    print(f"\n【总体统计】")
    print(f"  总交易: {total_trades} 笔")
    print(f"  总获胜: {total_win} 笔 | 胜率: {total_win_rate:.1f}%")
    print(f"  初始资金: {INITIAL_CAPITAL:,.2f} 元")
    print(f"  最终资金: {final_capital:,.2f} 元")
    print(f"  总盈亏: {total_pnl:+,.2f} 元 ({total_pnl_pct:+.2f}%)")

    # ========== 详细交易记录 ==========
    print(f"\n{'='*120}")
    print("详细交易记录")
    print(f"{'='*120}")

    print(f"\n{'-'*120}")
    print(f"{'序号':<5} {'方向':<6} {'信号':<15} {'入场时间':<19} {'入场价':<8} {'手数':<4} "
          f"{'出场时间':<19} {'出场价':<8} {'止损价':<8} {'盈亏金额':<10} {'盈亏%':<8} {'原因':<10} {'资金后':<12}")
    print(f"{'-'*120}")

    for i, trade in enumerate(all_trades, 1):
        direction_str = '做多' if trade['direction'] == 'long' else '做空'
        pnl_str = f"{'+' if trade['pnl'] >= 0 else ''}{trade['pnl']:.2f}"
        pnl_pct_str = f"{'+' if trade['pnl_pct'] >= 0 else ''}{trade['pnl_pct']:.2f}%"
        capital_str = f"{trade['capital_after']:,.0f}"

        print(f"{i:<5} {direction_str:<6} {trade['signal_type']:<15} "
              f"{str(trade['entry_datetime']):<19} {trade['entry_price']:<8.2f} {trade['contracts']:<4} "
              f"{str(trade['exit_datetime']):<19} {trade['exit_price']:<8.2f} {trade['stop_loss']:<8.2f} "
              f"{pnl_str:<10} {pnl_pct_str:<8} {trade['exit_reason']:<10} {capital_str:<12}")

    # ========== 对比分析 ==========
    print(f"\n{'='*120}")
    print("对比分析")
    print(f"{'='*120}")

    print(f"\n{'项目':<25} {'只做多(优化)':<20} {'镜像参数双向':<20} {'分别优化双向':<20}")
    print(f"{'-'*90}")
    print(f"{'收益率':<25} {'+167.47%':<20} {'+51.43%':<20} {total_pnl_pct:+.2f}%")
    print(f"{'交易次数':<25} {'13笔':<20} {'13笔':<20} {total_trades}笔")
    print(f"{'胜率':<25} {'84.6%':<20} {'69.2%':<20} {total_win_rate:.1f}%")

    # 保存结果
    output_file = Path('logs/pta_bidirectional_optimized_trades.csv')
    output_file.parent.mkdir(exist_ok=True)
    trades_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n交易记录已保存到: {output_file}")

    return trades_df

if __name__ == '__main__':
    backtest_bidirectional_optimized()
