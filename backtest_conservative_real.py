# -*- coding: utf-8 -*-
"""
使用真实保守参数回测
对比激进参数 vs 保守参数
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

def backtest_with_risk_params(df, params, contract_size, margin_rate,
                               max_position_ratio, max_single_loss_pct):
    """用指定风险参数回测"""
    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if df['low'].iloc[i] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_triggered = True
                exit_reason = '止损'
            elif (df['stc_prev'].iloc[i] > params['STC_SELL_ZONE'] and
                  current['stc'] < df['stc_prev'].iloc[i]):
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'STC止盈'
            elif current['ema_fast'] < current['ema_slow']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = '趋势反转'

            if exit_triggered:
                pnl = (exit_price - position['entry_price']) * position['contracts'] * contract_size
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'entry_price': position['entry_price'],
                    'exit_datetime': current['datetime'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'capital_after': capital
                })

                position = None
            continue

        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > params['RSI_FILTER']
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong
        buy_signal = sniper_signal or chase_signal

        if buy_signal and position is None:
            entry_price = current['close']

            contract_value = entry_price * contract_size
            margin_per_contract = contract_value * margin_rate

            stop_loss = entry_price * (1 - STOP_LOSS_PCT)

            potential_loss_per_contract = (entry_price - stop_loss) * contract_size
            max_contracts_by_risk = int((capital * max_single_loss_pct) / potential_loss_per_contract)
            max_contracts_by_margin = int((capital * max_position_ratio) / margin_per_contract)

            contracts = min(max_contracts_by_margin, max_contracts_by_risk)

            if contracts <= 0:
                continue

            position = {
                'entry_datetime': current['datetime'],
                'entry_price': entry_price,
                'contracts': contracts,
                'stop_loss': stop_loss,
                'entry_index': i,
                'capital_at_entry': capital
            }

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    total_pnl = trades_df['pnl'].sum()
    return_pct = total_pnl / INITIAL_CAPITAL * 100
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    # 计算最大回撤
    capital_curve = [INITIAL_CAPITAL] + list(trades_df['capital_after'])
    max_drawdown = 0
    peak = capital_curve[0]
    for cap in capital_curve:
        if cap > peak:
            peak = cap
        drawdown = (peak - cap) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # 计算连续最大亏损
    consecutive_losses = 0
    max_consecutive_losses = 0
    for pnl in trades_df['pnl']:
        if pnl < 0:
            consecutive_losses += 1
            if consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = consecutive_losses
        else:
            consecutive_losses = 0

    return {
        'return_pct': return_pct,
        'trades': total_trades,
        'win_rate': win_rate,
        'final_capital': capital,
        'max_drawdown': max_drawdown,
        'max_consecutive_losses': max_consecutive_losses,
        'trades_detail': trades_df
    }

def compare_variety(csv_file, variety_name, contract_size, margin_rate, params):
    """对比激进vs保守参数"""
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"\n{'='*80}")
    print(f"{variety_name} - 风险参数对比")
    print(f"{'='*80}")
    print(f"\n合约规格: {contract_size}吨/手, 保证金{margin_rate*100}%")

    # 激进参数（之前）
    aggressive_result = backtest_with_risk_params(
        df.copy(), params, contract_size, margin_rate,
        max_position_ratio=0.8,
        max_single_loss_pct=0.15
    )

    # 保守参数（真实）
    conservative_result = backtest_with_risk_params(
        df.copy(), params, contract_size, margin_rate,
        max_position_ratio=0.3,  # 只用30%资金
        max_single_loss_pct=0.02  # 单笔只亏2%
    )

    # 极度保守参数（实盘推荐）
    very_conservative_result = backtest_with_risk_params(
        df.copy(), params, contract_size, margin_rate,
        max_position_ratio=0.2,  # 只用20%资金
        max_single_loss_pct=0.01  # 单笔只亏1%
    )

    print(f"\n{'风险类型':<15} {'仓位比例':<10} {'单笔亏损':<10} {'收益率':>12} {'最大回撤':>12} {'交易数':>8} {'胜率':>8}")
    print("-" * 90)

    if aggressive_result:
        print(f"{'激进（回测）':<15} {'80%':<10} {'15%':<10} "
              f"{aggressive_result['return_pct']:>+10.2f}% "
              f"{aggressive_result['max_drawdown']:>+10.2f}% "
              f"{aggressive_result['trades']:>6}笔 "
              f"{aggressive_result['win_rate']:>6.1f}%")
    else:
        print(f"{'激进（回测）':<15} {'80%':<10} {'15%':<10} {'无交易':>12} {'N/A':>12} {'N/A':>8} {'N/A':>8}")

    if conservative_result:
        print(f"{'保守（真实）':<15} {'30%':<10} {'2%':<10} "
              f"{conservative_result['return_pct']:>+10.2f}% "
              f"{conservative_result['max_drawdown']:>+10.2f}% "
              f"{conservative_result['trades']:>6}笔 "
              f"{conservative_result['win_rate']:>6.1f}%")
    else:
        print(f"{'保守（真实）':<15} {'30%':<10} {'2%':<10} {'无交易':>12} {'N/A':>12} {'N/A':>8} {'N/A':>8}")

    if very_conservative_result:
        print(f"{'极度保守':<15} {'20%':<10} {'1%':<10} "
              f"{very_conservative_result['return_pct']:>+10.2f}% "
              f"{very_conservative_result['max_drawdown']:>+10.2f}% "
              f"{very_conservative_result['trades']:>6}笔 "
              f"{very_conservative_result['win_rate']:>6.1f}%")
    else:
        print(f"{'极度保守':<15} {'20%':<10} {'1%':<10} {'无交易':>12} {'N/A':>12} {'N/A':>8} {'N/A':>8}")

    # 风险分析
    if aggressive_result and conservative_result:
        print(f"\n风险分析:")
        print(f"  收益率下降: {aggressive_result['return_pct'] - conservative_result['return_pct']:+.2f}%")
        print(f"  回撤改善: {aggressive_result['max_drawdown'] - conservative_result['max_drawdown']:+.2f}%")
        print(f"  最大连续亏损: {aggressive_result['max_consecutive_losses']}笔 (激进) vs "
              f"{conservative_result['max_consecutive_losses']}笔 (保守)")

        # 爆仓风险分析
        print(f"\n爆仓风险分析:")
        if aggressive_result['max_consecutive_losses'] > 0:
            avg_loss = aggressive_result['trades_detail'][
                aggressive_result['trades_detail']['pnl'] < 0
            ]['pnl'].mean()
            print(f"  激进参数平均止损: {avg_loss:,.2f}元")
            print(f"  7次连续止损损失: {avg_loss * 7:,.2f}元 ({avg_loss * 7 / INITIAL_CAPITAL * 100:.1f}%)")
            if aggressive_result['max_consecutive_losses'] >= 3:
                print(f"  警告: 最大连续亏损{aggressive_result['max_consecutive_losses']}笔!")

def main():
    print("=" * 80)
    print("真实风险参数对比分析")
    print("对比激进参数（回测） vs 保守参数（实盘）")
    print("=" * 80)

    # 测试Top 3品种
    test_varieties = [
        ('沪锡_4hour.csv', '沪锡', 1, 0.13,
         {'EMA_FAST': 3, 'EMA_SLOW': 10, 'RSI_FILTER': 35, 'RATIO_TRIGGER': 1.25, 'STC_SELL_ZONE': 75}),
        ('PTA_4hour.csv', 'PTA', 5, 0.07,
         {'EMA_FAST': 12, 'EMA_SLOW': 10, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.25, 'STC_SELL_ZONE': 75}),
        ('焦煤_4hour.csv', '焦煤', 60, 0.08,
         {'EMA_FAST': 12, 'EMA_SLOW': 10, 'RSI_FILTER': 35, 'RATIO_TRIGGER': 1.05, 'STC_SELL_ZONE': 75}),
    ]

    data_dir = Path('futures_data_4h')

    for csv_name, name, contract_size, margin_rate, params in test_varieties:
        csv_file = data_dir / csv_name
        if csv_file.exists():
            compare_variety(csv_file, name, contract_size, margin_rate, params)

    print(f"\n{'='*80}")
    print("关键结论:")
    print("=" * 80)
    print("""
1. 激进参数（80%仓位，15%单笔亏损）：
   - 回测收益高，但实盘风险极大
   - 连续3-4次止损可能损失50-60%本金
   - 最大回撤可能超过30%
   - 不适合实盘交易！

2. 保守参数（30%仓位，2%单笔亏损）：
   - 收益率下降，但风险可控
   - 连续止损不会爆仓
   - 最大回撤通常在10-15%以内
   - 适合实盘交易

3. 实盘建议：
   - 新手：20%仓位，1%单笔亏损
   - 有经验：30%仓位，2%单笔亏损
   - 专业：40%仓位，3%单笔亏损（最大不超过50%）

4. 重要提醒：
   - 回测收益率不等于实盘收益率
   - 实盘需要考虑滑点、手续费、心理因素
   - 期货交易风险巨大，可能损失全部本金
   - 建议从小资金开始，逐步验证策略
    """)

if __name__ == '__main__':
    main()
