# -*- coding: utf-8 -*-
"""
调试白糖回测 - 用铜参数
"""

import pandas as pd
import numpy as np

# 铜参数
params = {
    'EMA_FAST': 5,
    'EMA_SLOW': 15,
    'RSI_FILTER': 45,
    'RATIO_TRIGGER': 1.15,
    'STC_SELL_ZONE': 85
}

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50
STOP_LOSS_PCT = 0.02
INITIAL_CAPITAL = 100000
MARGIN_RATIO = 0.15
CONTRACT_SIZE = 5

df = pd.read_csv('futures_data_4h/白糖_4hour.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"白糖数据: {len(df)}条")
print(f"时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
print(f"参数: EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']}, RATIO={params['RATIO_TRIGGER']}, STC={params['STC_SELL_ZONE']}")

# 计算指标
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

stc_macd = df['close'].ewm(span=STC_FAST, adjust=False).mean() - df['close'].ewm(span=STC_SLOW, adjust=False).mean()
min_macd = stc_macd.rolling(window=STC_LENGTH).min()
max_macd = stc_macd.rolling(window=STC_LENGTH).max()
stoch_k = 100 * (stc_macd - min_macd) / (max_macd - min_macd).replace(0, np.nan)
stoch_k = stoch_k.fillna(50)
stoch_d = stoch_k.rolling(window=3).mean()
min_stoch_d = stoch_d.rolling(window=STC_LENGTH).min()
max_stoch_d = stoch_d.rolling(window=STC_LENGTH).max()
stc_raw = 100 * (stoch_d - min_stoch_d) / (max_stoch_d - min_stoch_d).replace(0, np.nan)
stc_raw = stc_raw.fillna(50)
df['stc'] = stc_raw.rolling(window=3).mean()
df['ratio_prev'] = df['ratio'].shift(1)
df['stc_prev'] = df['stc'].shift(1)

# 回测
capital = INITIAL_CAPITAL
position = None
trades = []

for i in range(200, len(df)):
    current = df.iloc[i]
    prev = df.iloc[i-1]

    # 入场条件
    trend_up = current['ema_fast'] > current['ema_slow']
    ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
    ratio_shrinking = current['ratio'] < prev['ratio']
    turning_up = current['macd_dif'] > prev['macd_dif']
    is_strong = current['rsi'] > params['RSI_FILTER']
    ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

    sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
    chase_signal = ema_cross and is_strong
    buy_signal = sniper_signal or chase_signal

    # 仓位计算
    if df['ratio_prev'].iloc[i] > 0:
        if df['ratio_prev'].iloc[i] > 2.0:
            position_size = 2.0
        elif df['ratio_prev'].iloc[i] > 1.5:
            position_size = 1.5
        elif df['ratio_prev'].iloc[i] > 1.0:
            position_size = 1.2
        else:
            position_size = 1.0
    else:
        position_size = 1.0

    stop_loss = current['close'] * (1 - STOP_LOSS_PCT)

    # 买入
    if buy_signal and position is None:
        entry_price = current['close']
        contract_value = entry_price * CONTRACT_SIZE
        margin_per_contract = contract_value * MARGIN_RATIO
        available_margin = capital * position_size
        contracts = int(available_margin / margin_per_contract)

        if contracts <= 0:
            continue

        position = {
            'entry_datetime': current['datetime'],
            'entry_price': entry_price,
            'contracts': contracts,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'entry_index': i,
            'signal_type': 'sniper' if sniper_signal else 'chase'
        }

    # 卖出
    elif position is not None:
        for j in range(position['entry_index'] + 1, len(df)):
            bar = df.iloc[j]

            exit_triggered = False
            exit_price = None
            exit_reason = None

            if bar['low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_reason = 'stop_loss'
                exit_triggered = True
            elif (df['stc_prev'].iloc[j] > params['STC_SELL_ZONE'] and
                  bar['stc'] < df['stc_prev'].iloc[j]):
                exit_price = bar['close']
                exit_reason = 'stc'
                exit_triggered = True
            elif bar['ema_fast'] < bar['ema_slow']:
                exit_price = bar['close']
                exit_reason = 'trend'
                exit_triggered = True

            if exit_triggered:
                pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': bar['datetime'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'position_size': position['position_size'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'holding_bars': j - position['entry_index'],
                    'signal_type': position['signal_type'],
                    'exit_reason': exit_reason
                })

                position = None
                break

if not trades:
    print("\n无交易记录")
else:
    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])

    total_pnl = trades_df['pnl'].sum()
    return_pct = total_pnl / INITIAL_CAPITAL * 100
    win_rate = winning_trades / total_trades * 100

    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    print(f"\n{'='*80}")
    print("白糖回测结果（铜参数）")
    print(f"{'='*80}")
    print(f"\n【资金】")
    print(f"  初始资金: {INITIAL_CAPITAL:,.0f}")
    print(f"  最终资金: {capital:,.0f}")
    print(f"  总收益: {total_pnl:,.0f} ({return_pct:+.2f}%)")

    print(f"\n【交易统计】")
    print(f"  总交易次数: {total_trades}")
    print(f"  盈利次数: {winning_trades} ({win_rate:.1f}%)")
    print(f"  亏损次数: {losing_trades}")

    sniper_trades = trades_df[trades_df['signal_type'] == 'sniper']
    chase_trades = trades_df[trades_df['signal_type'] == 'chase']
    print(f"  狙击信号: {len(sniper_trades)}")
    print(f"  追击信号: {len(chase_trades)}")

    print(f"\n【盈亏分析】")
    print(f"  平均盈利: {avg_win:+,.2f}")
    print(f"  平均亏损: {avg_loss:+,.2f}")
    print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  盈亏比: N/A")

    print(f"\n【对比】")
    print(f"  原始批量回测: +188.15%, 16笔交易")
    print(f"  当前验证结果: {return_pct:+.2f}%, {total_trades}笔交易")
