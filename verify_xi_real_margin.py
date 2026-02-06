# -*- coding: utf-8 -*-
"""
验证沪锡使用真实保证金比例（13%）的结果
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

# 真实沪锡合约规格
CONTRACT_SIZE = 1  # 1吨/手
MARGIN_RATE = 0.13  # 13%保证金

# 风险控制参数
MAX_POSITION_RATIO = 0.8  # 最大仓位使用率
MAX_SINGLE_LOSS_PCT = 0.15  # 单笔最大亏损

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

def backtest_realistic(df, params):
    """真实沪锡回测"""
    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    position = None  # 一次只持有一个仓位
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 只有在没有持仓时才考虑开仓
        if position is not None:
            # 检查止损止盈
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
                pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
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

        # 检查开仓信号
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

            # 计算合约价值和保证金（使用真实保证金比例13%）
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATE

            # 计算止损价格
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)

            # 检查单笔最大亏损
            potential_loss_per_contract = (entry_price - stop_loss) * CONTRACT_SIZE
            max_contracts_by_risk = int((capital * MAX_SINGLE_LOSS_PCT) / potential_loss_per_contract)

            # 计算基于保证金的最大手数
            max_contracts_by_margin = int((capital * MAX_POSITION_RATIO) / margin_per_contract)

            # 取较小值
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

    return {
        'return_pct': return_pct,
        'trades': total_trades,
        'win_rate': win_rate,
        'final_capital': capital,
        'trades_detail': trades_df
    }

def main():
    print("=" * 80)
    print("沪锡真实保证金比例验证")
    print("=" * 80)

    # 加载数据
    csv_file = Path('futures_data_4h/沪锡_4hour.csv')
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"\n合约规格：")
    print(f"  合约单位: {CONTRACT_SIZE} 吨/手")
    print(f"  保证金比例: {MARGIN_RATE*100}%")
    print(f"  初始资金: {INITIAL_CAPITAL:,.0f} 元")

    print(f"\n数据信息：")
    print(f"  品种: 沪锡")
    print(f"  数据量: {len(df)}条")
    print(f"  时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 最优参数（来自优化结果）
    params = {
        'EMA_FAST': 3,
        'EMA_SLOW': 10,
        'RSI_FILTER': 35,
        'RATIO_TRIGGER': 1.25,
        'STC_SELL_ZONE': 75
    }

    print(f"\n优化参数：")
    print(f"  EMA({params['EMA_FAST']}, {params['EMA_SLOW']})")
    print(f"  RSI_FILTER: {params['RSI_FILTER']}")
    print(f"  RATIO_TRIGGER: {params['RATIO_TRIGGER']:.2f}")
    print(f"  STC_SELL_ZONE: {params['STC_SELL_ZONE']}")

    print(f"\n预期结果（12%保证金）：")
    print(f"  收益率: +458.51%")
    print(f"  交易数: 18笔")
    print(f"  胜率: 55.6%")

    # 回测
    result = backtest_realistic(df.copy(), params)

    if result is None:
        print("\n[ERROR] 无交易记录")
        return

    trades_df = result['trades_detail']

    print(f"\n实际回测结果（13%保证金）：")
    print(f"  收益率: {result['return_pct']:+.2f}%")
    print(f"  交易数: {result['trades']}笔")
    print(f"  胜率: {result['win_rate']:.1f}%")
    print(f"  最终资金: {result['final_capital']:,.2f}")

    # 对比
    print(f"\n对比分析：")
    print(f"  12%保证金: +458.51% (18笔)")
    print(f"  13%保证金: {result['return_pct']:+.2f}% ({result['trades']}笔)")
    diff = result['return_pct'] - 458.51
    print(f"  差异: {diff:+.2f}% ({(diff/458.51*100):+.1f}%)")

    # 详细交易记录
    print(f"\n{'='*80}")
    print("详细交易记录：")
    print(f"{'='*80}")

    print(f"\n{'序号':<6} {'入场时间':>20} {'入场价格':>10} {'出场时间':>20} {'出场价格':>10} "
          f"{'手数':>6} {'出场原因':>10} {'盈亏':>12} {'收益率%':>10} {'资金后':>15}")

    print("-" * 150)

    for i, trade in trades_df.iterrows():
        pnl_str = f"{trade['pnl']:+,.2f}" if trade['pnl'] >= 0 else f"{trade['pnl']:,.2f}"
        return_pct = trade['pnl'] / trade['capital_after'] * 100 if trade['capital_after'] > 0 else 0
        return_str = f"{return_pct:+.2f}%" if trade['pnl'] >= 0 else f"{return_pct:.2f}%"
        capital_str = f"{trade['capital_after']:,.2f}"

        print(f"{i+1:<6} {str(trade['entry_datetime']):>20} {trade['entry_price']:>10.2f} "
              f"{str(trade['exit_datetime']):>20} {trade['exit_price']:>10.2f} "
              f"{trade['contracts']:>6} {trade['exit_reason']:>10} "
              f"{pnl_str:>12} {return_str:>10} {capital_str:>15}")

    # 盈亏分析
    print(f"\n{'='*80}")
    print("盈亏分析：")
    print(f"{'='*80}")

    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    print(f"\n盈利交易 ({len(wins)}笔):")
    if len(wins) > 0:
        avg_win = wins['pnl'].mean()
        max_win = wins['pnl'].max()
        total_win = wins['pnl'].sum()
        print(f"  平均盈利: {avg_win:,.2f}")
        print(f"  最大盈利: {max_win:,.2f}")
        print(f"  盈利总额: {total_win:,.2f}")

    print(f"\n亏损交易 ({len(losses)}笔):")
    if len(losses) > 0:
        avg_loss = losses['pnl'].mean()
        max_loss = losses['pnl'].min()
        total_loss = losses['pnl'].sum()
        print(f"  平均亏损: {avg_loss:,.2f}")
        print(f"  最大亏损: {max_loss:,.2f}")
        print(f"  亏损总额: {total_loss:,.2f}")

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = abs(total_win / total_loss)
        print(f"\n盈亏比: {profit_factor:.2f}")

    # 出场原因统计
    print(f"\n出场原因统计:")
    exit_reasons = trades_df['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        pct = count / len(trades_df) * 100
        print(f"  {reason}: {count}笔 ({pct:.1f}%)")

    # 仓位分析
    print(f"\n仓位分析:")
    max_contracts = trades_df['contracts'].max()
    min_contracts = trades_df['contracts'].min()
    avg_contracts = trades_df['contracts'].mean()
    print(f"  最大手数: {max_contracts}手")
    print(f"  最小手数: {min_contracts}手")
    print(f"  平均手数: {avg_contracts:.1f}手")

    # 保证金使用分析
    print(f"\n保证金使用分析（以第一笔交易为例）:")
    if len(trades_df) > 0:
        first_trade = trades_df.iloc[0]
        entry_price = first_trade['entry_price']
        contracts = first_trade['contracts']
        capital_at_entry = first_trade['capital_after'] - first_trade['pnl']

        contract_value = entry_price * CONTRACT_SIZE
        margin_per_contract = contract_value * MARGIN_RATE
        total_margin = margin_per_contract * contracts
        margin_usage = total_margin / capital_at_entry * 100

        print(f"  入场价格: {entry_price:,.2f} 元/吨")
        print(f"  合约价值: {contract_value:,.2f} 元/手")
        print(f"  保证金/手: {margin_per_contract:,.2f} 元")
        print(f"  开仓手数: {contracts}手")
        print(f"  总保证金: {total_margin:,.2f} 元")
        print(f"  当时资金: {capital_at_entry:,.2f} 元")
        print(f"  保证金使用率: {margin_usage:.2f}%")

if __name__ == '__main__':
    main()
