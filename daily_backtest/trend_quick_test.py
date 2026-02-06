# -*- coding: utf-8 -*-
"""
快速测试：ADX趋势指标效果
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

from futures_monitor import calculate_indicators

# 加载数据
df = pd.read_csv('SN_沪锡_日线_10年.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# 转换列名
df = df.rename(columns={
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low',
    '收盘价': 'close',
    '成交量': 'volume',
    '持仓量': 'hold'
})

# 计算ADX
def calculate_ADX_simple(df, period=14):
    df = df.copy()

    # +DM和-DM
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']

    df['plus_dm'] = df.apply(
        lambda x: x['up_move'] if (x['up_move'] > x['down_move']) and (x['up_move'] > 0) else 0,
        axis=1
    )
    df['minus_dm'] = df.apply(
        lambda x: x['down_move'] if (x['down_move'] > x['up_move']) and (x['down_move'] > 0) else 0,
        axis=1
    )

    # TR
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)

    # 平滑
    df['plus_dm_smooth'] = df['plus_dm'].rolling(period).sum()
    df['minus_dm_smooth'] = df['minus_dm'].rolling(period).sum()
    df['tr_smooth'] = df['tr'].rolling(period).sum()

    # +DI和-DI
    df['plus_di'] = 100 * df['plus_dm_smooth'] / df['tr_smooth']
    df['minus_di'] = 100 * df['minus_dm_smooth'] / df['tr_smooth']

    # DX和ADX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(period).mean()

    return df

df = calculate_ADX_simple(df)

print("="*80)
print("ADX趋势指标分析")
print("="*80)
print(f"\nADX统计:")
print(f"  平均值: {df['adx'].mean():.2f}")
print(f"  中位数: {df['adx'].median():.2f}")
print(f"  最小值: {df['adx'].min():.2f}")
print(f"  最大值: {df['adx'].max():.2f}")

# 统计不同ADX范围的天数
print(f"\nADX分布:")
print(f"  ADX < 20 (弱趋势): {len(df[df['adx'] < 20])}天 ({len(df[df['adx'] < 20])/len(df)*100:.1f}%)")
print(f"  20 ≤ ADX < 25 (震荡): {len(df[(df['adx'] >= 20) & (df['adx'] < 25)])}天 ({len(df[(df['adx'] >= 20) & (df['adx'] < 25)])/len(df)*100:.1f}%)")
print(f"  25 ≤ ADX < 30 (强趋势): {len(df[(df['adx'] >= 25) & (df['adx'] < 30)])}天 ({len(df[(df['adx'] >= 25) & (df['adx'] < 30)])/len(df)*100:.1f}%)")
print(f"  ADX ≥ 30 (超强趋势): {len(df[df['adx'] >= 30])}天 ({len(df[df['adx'] >= 30])/len(df)*100:.1f}%)")

# 对比原始策略交易日的ADX
print(f"\n加载原始策略交易记录...")
original_trades = []
capital = 100000
position = None

BEST_PARAMS = {
    'EMA_FAST': 3,
    'EMA_SLOW': 15,
    'RSI_FILTER': 30,
    'RATIO_TRIGGER': 1.05,
    'STC_SELL_ZONE': 65,
    'STOP_LOSS_PCT': 0.02
}

INITIAL_CAPITAL = 100000
MAX_POSITION_RATIO = 0.8
STOP_LOSS_PCT = 0.02
CONTRACT_SIZE = 1
MARGIN_RATE = 0.13

df = calculate_indicators(df, BEST_PARAMS)

for i in range(200, len(df)):
    current = df.iloc[i]
    prev = df.iloc[i-1]

    # 平仓
    if position is not None:
        exit_triggered = False
        exit_price = None

        if current['low'] <= position['stop_loss']:
            exit_price = position['stop_loss']
            exit_triggered = True
        elif (prev['stc'] > BEST_PARAMS['STC_SELL_ZONE'] and
              current['stc'] < prev['stc']):
            exit_price = current['close']
            exit_triggered = True
        elif current['ema_fast'] < current['ema_slow']:
            exit_price = current['close']
            exit_triggered = True

        if exit_triggered:
            pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
            capital += pnl
            original_trades.append({
                'entry_datetime': position['entry_datetime'],
                'exit_datetime': current['datetime'],
                'pnl': pnl,
                'entry_adx': position['adx']
            })
            position = None
            continue

    # 开仓
    if position is None:
        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < BEST_PARAMS['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < current['ratio_prev']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > BEST_PARAMS['RSI_FILTER']

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
        buy_signal = sniper_signal or (ema_cross and is_strong)

        if buy_signal:
            entry_price = current['close']
            margin_per_contract = entry_price * CONTRACT_SIZE * MARGIN_RATE
            available_capital = capital * MAX_POSITION_RATIO
            max_contracts = int(available_capital / margin_per_contract)

            if max_contracts > 0:
                stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                position = {
                    'entry_datetime': current['datetime'],
                    'entry_price': entry_price,
                    'contracts': max_contracts,
                    'stop_loss': stop_loss_price,
                    'adx': current['adx']
                }

trades_df = pd.DataFrame(original_trades)

print(f"\n原始策略交易日的ADX分析:")
print(f"  总交易: {len(trades_df)}笔")
print(f"  平均ADX: {trades_df['entry_adx'].mean():.2f}")

# 按ADX分组分析
trades_df['adx_group'] = pd.cut(trades_df['entry_adx'],
                                 bins=[0, 20, 25, 30, 100],
                                 labels=['弱趋势(<20)', '震荡(20-25)', '强趋势(25-30)', '超强趋势(>30)'])

print(f"\n按ADX分组统计:")
print(f"{'ADX组':<20} {'交易数':<8} {'盈利':<8} {'亏损':<8} {'胜率':<10} {'平均盈亏':<15}")
print("-"*80)

for name, group in trades_df.groupby('adx_group'):
    wins = len(group[group['pnl'] > 0])
    win_rate = wins / len(group) * 100
    avg_pnl = group['pnl'].mean()
    print(f"{str(name):<20} {len(group):<8} {wins:<8} {len(group)-wins:<8} {win_rate:<10.1f}% {avg_pnl:>13,.0f}元")

print("\n" + "="*80)
print("结论")
print("="*80)

# 判断ADX是否有效
weak_trend = trades_df[trades_df['entry_adx'] < 20]
strong_trend = trades_df[trades_df['entry_adx'] >= 25]

if len(weak_trend) > 0 and len(strong_trend) > 0:
    weak_wr = len(weak_trend[weak_trend['pnl'] > 0]) / len(weak_trend) * 100
    strong_wr = len(strong_trend[strong_trend['pnl'] > 0]) / len(strong_trend) * 100

    print(f"\nADX有效性验证:")
    print(f"  弱趋势期(ADX<25)胜率: {weak_wr:.1f}%")
    print(f"  强趋势期(ADX≥25)胜率: {strong_wr:.1f}%")

    if strong_wr > weak_wr + 5:
        print(f"\n[OK] ADX能有效识别趋势！")
        print(f"  强趋势期胜率比弱趋势期高{strong_wr - weak_wr:.1f}个百分点")
        print(f"  建议：只在ADX≥25时开仓")
    else:
        print(f"\n[!] ADX对沪锡日线趋势识别效果不明显")
        print(f"  原因：沪锡10年是大牛市，任何时期都有交易机会")
else:
    print(f"\n[!] 数据不足，无法验证ADX效果")

print("\n" + "="*80)
