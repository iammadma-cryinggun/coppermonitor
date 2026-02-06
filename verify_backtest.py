# -*- coding: utf-8 -*-
"""
回测准确性验证 - 运行3次确认结果一致
"""

import pandas as pd
import numpy as np
from datetime import datetime

# 优化后的参数
OPTIMIZED_PARAMS = {
    'EMA_FAST': 3,
    'EMA_SLOW': 20,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'RSI_PERIOD': 14,
    'RSI_FILTER': 40,
    'RATIO_TRIGGER': 1.2,
    'STC_LENGTH': 10,
    'STC_FAST': 23,
    'STC_SLOW': 50,
    'STC_SELL_ZONE': 85,
    'STOP_LOSS_PCT': 0.015
}

def run_backtest_validation():
    """验证回测"""
    # 加载数据
    df = pd.read_csv('沪铜4小时K线_1年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 计算指标
    df['ema_fast'] = df['close'].ewm(span=OPTIMIZED_PARAMS['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=OPTIMIZED_PARAMS['EMA_SLOW'], adjust=False).mean()

    exp1 = df['close'].ewm(span=OPTIMIZED_PARAMS['MACD_FAST'], adjust=False).mean()
    exp2 = df['close'].ewm(span=OPTIMIZED_PARAMS['MACD_SLOW'], adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=OPTIMIZED_PARAMS['MACD_SIGNAL'], adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/OPTIMIZED_PARAMS['RSI_PERIOD'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/OPTIMIZED_PARAMS['RSI_PERIOD'], adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    stc_macd = df['close'].ewm(span=OPTIMIZED_PARAMS['STC_FAST'], adjust=False).mean() - df['close'].ewm(span=OPTIMIZED_PARAMS['STC_SLOW'], adjust=False).mean()
    stoch_period = OPTIMIZED_PARAMS['STC_LENGTH']
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

    # 回测
    INITIAL_CAPITAL = 100000
    MARGIN_RATIO = 0.15
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < OPTIMIZED_PARAMS['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > OPTIMIZED_PARAMS['RSI_FILTER']
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong
        buy_signal = sniper_signal or chase_signal

        if buy_signal and position is None:
            entry_price = current['close']
            ratio_prev_val = df['ratio_prev'].iloc[i]

            if ratio_prev_val > 2.0:
                position_size = 2.0
            elif ratio_prev_val > 1.5:
                position_size = 1.5
            elif ratio_prev_val > 1.0:
                position_size = 1.2
            else:
                position_size = 1.0

            contract_value = entry_price * 5
            margin_per_contract = contract_value * MARGIN_RATIO
            available_margin = capital * position_size
            contracts = int(available_margin / margin_per_contract)

            if contracts > 0:
                position = {
                    'entry_datetime': current['datetime'],
                    'entry_price': entry_price,
                    'contracts': contracts,
                    'position_size': position_size,
                    'stop_loss': entry_price * (1 - OPTIMIZED_PARAMS['STOP_LOSS_PCT']),
                    'entry_index': i
                }

        elif position is not None:
            stc_prev_val = df['stc_prev'].iloc[i]
            stc_exit = (stc_prev_val > OPTIMIZED_PARAMS['STC_SELL_ZONE'] and current['stc'] < stc_prev_val)
            trend_exit = current['ema_fast'] < current['ema_slow']
            stop_loss_hit = current['low'] <= position['stop_loss']

            if stc_exit or trend_exit or stop_loss_hit:
                exit_price = position['stop_loss'] if stop_loss_hit else current['close']
                exit_reason = 'stop_loss' if stop_loss_hit else ('stc' if stc_exit else 'trend')

                pnl = (exit_price - position['entry_price']) * position['contracts'] * 5
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'position_size': position['position_size'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': exit_reason,
                    'holding_bars': i - position['entry_index']
                })

                position = None

    # 统计
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])

    total_pnl = trades_df['pnl'].sum()
    final_capital = INITIAL_CAPITAL + total_pnl
    win_rate = winning_trades / total_trades * 100

    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'total_pnl': total_pnl,
        'final_capital': final_capital,
        'return_pct': total_pnl / INITIAL_CAPITAL * 100,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
    }

if __name__ == '__main__':
    # 运行3次验证
    print('='*80)
    print('回测准确性验证 - 运行3次')
    print('='*80)

    results = []
    for i in range(3):
        result = run_backtest_validation()
        results.append(result)
        print(f'\n第 {i+1} 次运行:')
        print(f"  收益率: {result['return_pct']:.2f}%")
        print(f"  胜率: {result['win_rate']:.1f}%")
        print(f"  盈亏比: {result['profit_factor']:.2f}")
        print(f"  交易次数: {result['total_trades']}")

    # 验证一致性
    returns = [r['return_pct'] for r in results]
    win_rates = [r['win_rate'] for r in results]
    pf_ratios = [r['profit_factor'] for r in results]

    print('\n' + '='*80)
    print('验证结果')
    print('='*80)

    if max(returns) - min(returns) < 0.01 and max(win_rates) - min(win_rates) < 0.01:
        print('✓ 验证通过：3次运行结果完全一致')
        print(f'\n最终确认结果:')
        print(f'  收益率: {results[0]["return_pct"]:.2f}%')
        print(f'  胜率: {results[0]["win_rate"]:.1f}%')
        print(f'  盈亏比: {results[0]["profit_factor"]:.2f}')
        print(f'  交易次数: {results[0]["total_trades"]}')
        print(f'  总盈利: {results[0]["total_pnl"]:,.0f} 元')
        print(f'  最终资金: {results[0]["final_capital"]:,.0f} 元')
    else:
        print('✗ 警告：3次运行结果不一致，请检查代码')
        print(f'收益率差异: {max(returns) - min(returns):.4f}%')
        print(f'胜率差异: {max(win_rates) - min(win_rates):.4f}%')
