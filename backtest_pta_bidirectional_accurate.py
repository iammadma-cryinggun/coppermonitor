# -*- coding: utf-8 -*-
"""
PTA多空双向回测 - 使用准确逻辑（与优化脚本一致）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime

def backtest_bidirectional_accurate():
    """使用准确逻辑进行多空双向回测"""

    print("=" * 120)
    print("PTA多空双向回测 - 准确逻辑版本".center(120))
    print("=" * 120)

    # ========== 合约规格 ==========
    CONTRACT_SIZE = 5  # PTA: 5吨/手
    MARGIN_RATE = 0.08  # 8%保证金

    # ========== 参数配置 ==========
    # 做多参数
    EMA_FAST = 12
    EMA_SLOW = 10
    RSI_FILTER_LONG = 40
    RATIO_TRIGGER_LONG = 1.25
    STOP_LOSS_PCT = 0.02  # 2%止损
    STC_SELL_ZONE = 75  # 平多仓

    # 做空参数
    RSI_FILTER_SHORT = 60  # 100 - 40
    RATIO_TRIGGER_SHORT = 1.25
    STC_BUY_ZONE = 25  # 平空仓

    # 资金管理
    INITIAL_CAPITAL = 50000
    MAX_POSITION_RATIO = 0.80  # 最大仓位80%
    MAX_SINGLE_LOSS_PCT = 0.15  # 单笔最大亏损15%

    print(f"\n【参数配置】")
    print(f"  做多: EMA({EMA_FAST},{EMA_SLOW}), RSI={RSI_FILTER_LONG}, RATIO={RATIO_TRIGGER_LONG}, STC_SELL={STC_SELL_ZONE}")
    print(f"  做空: RSI={RSI_FILTER_SHORT}, RATIO={RATIO_TRIGGER_SHORT}, STC_BUY={STC_BUY_ZONE}")
    print(f"  止损: {STOP_LOSS_PCT*100}%")
    print(f"  合约: {CONTRACT_SIZE}吨/手, 保证金{MARGIN_RATE*100}%")
    print(f"  资金管理: 最大仓位{MAX_POSITION_RATIO*100}%, 单笔最大亏损{MAX_SINGLE_LOSS_PCT*100}%")

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

    # ========== 计算指标 ==========
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

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
                # 做多止损：低点触及止损价，按止损价平仓
                if current['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']  # 关键：按止损价平仓
                    exit_triggered = True
                    exit_reason = '止损'
            elif position['direction'] == 'short':
                # 做空止损：高点触及止损价，按止损价平仓
                if current['high'] >= position['stop_loss']:
                    exit_price = position['stop_loss']  # 关键：按止损价平仓
                    exit_triggered = True
                    exit_reason = '止损'

            # 平仓信号检查
            if not exit_triggered:
                if position['direction'] == 'long':
                    # 平多仓信号
                    stc_exit = (prev['stc'] > STC_SELL_ZONE) and (current['stc'] < prev['stc'])
                    trend_exit = current['ema_fast'] < current['ema_slow']

                    if stc_exit:
                        exit_price = current['close']
                        exit_triggered = True
                        exit_reason = 'STC止盈'
                    elif trend_exit:
                        exit_price = current['close']
                        exit_triggered = True
                        exit_reason = '趋势反转'

                elif position['direction'] == 'short':
                    # 平空仓信号
                    stc_exit = (prev['stc'] < STC_BUY_ZONE) and (current['stc'] > prev['stc'])
                    trend_exit = current['ema_fast'] > current['ema_slow']

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
                else:  # short
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
            # 做多信号
            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe_long = (0 < current['ratio'] < RATIO_TRIGGER_LONG)
            ratio_shrinking_long = current['ratio'] < prev['ratio']
            turning_up_long = current['macd_dif'] > prev['macd_dif']
            is_strong_long = current['rsi'] > RSI_FILTER_LONG

            sniper_long = trend_up and ratio_safe_long and ratio_shrinking_long and turning_up_long and is_strong_long

            ema_golden_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            chase_long = ema_golden_cross and is_strong_long

            # 做空信号
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

            # 计算合约手数（动态仓位管理）
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATE

            # 基于风险的最大手数
            if direction == 'long':
                potential_loss_per_contract = (entry_price - stop_loss) * CONTRACT_SIZE
            else:  # short
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

    print(f"\n【做多统计】")
    print(f"  交易次数: {long_total} 笔")
    print(f"  获胜: {long_win} 笔 | 失败: {long_loss} 笔")
    print(f"  胜率: {long_win_rate:.1f}%")
    print(f"  总盈亏: {long_total_pnl:+.2f} 元 ({long_total_pnl_pct:+.2f}%)")
    print(f"  平均盈亏: {long_avg_pnl_pct:+.2f}%")

    print(f"\n【做空统计】")
    print(f"  交易次数: {short_total} 笔")
    print(f"  获胜: {short_win} 笔 | 失败: {short_loss} 笔")
    print(f"  胜率: {short_win_rate:.1f}%")
    print(f"  总盈亏: {short_total_pnl:+.2f} 元 ({short_total_pnl_pct:+.2f}%)")
    print(f"  平均盈亏: {short_avg_pnl_pct:+.2f}%")

    print(f"\n【总体统计】")
    print(f"  总交易: {total_trades} 笔")
    print(f"  总获胜: {total_win} 笔 | 胜率: {total_win_rate:.1f}%")
    print(f"  初始资金: {INITIAL_CAPITAL:,.2f} 元")
    print(f"  最终资金: {final_capital:,.2f} 元")
    print(f"  总盈亏: {total_pnl:+.2f} 元 ({total_pnl_pct:+.2f}%)")

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

    # ========== 保存交易记录 ==========
    output_file = Path('logs/pta_bidirectional_accurate_trades.csv')
    output_file.parent.mkdir(exist_ok=True)
    trades_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n交易记录已保存到: {output_file}")

    # ========== 与只做多优化结果对比 ==========
    print(f"\n{'='*120}")
    print("与之前只做多优化结果对比")
    print(f"{'='*120}")

    print(f"\n{'项目':<25} {'只做多(优化)':<20} {'做多(双向)':<20} {'做空(双向)':<20} {'双向合计':<20}")
    print(f"{'-'*100}")
    print(f"{'交易次数':<25} {'13笔':<20} {f'{long_total}笔':<20} {f'{short_total}笔':<20} {f'{total_trades}笔':<20}")
    print(f"{'胜率':<25} {'84.6%':<20} {f'{long_win_rate:.1f}%':<20} {f'{short_win_rate:.1f}%':<20} {f'{total_win_rate:.1f}%':<20}")
    print(f"{'收益率':<25} {'+167.47%':<20} {f'{long_total_pnl_pct:+.2f}%':<20} {f'{short_total_pnl_pct:+.2f}%':<20} {f'{total_pnl_pct:+.2f}%':<20}")

    return trades_df

if __name__ == '__main__':
    backtest_bidirectional_accurate()
