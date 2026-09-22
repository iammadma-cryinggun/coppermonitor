# -*- coding: utf-8 -*-
"""
调试V2交易逻辑 - 打印详细交易记录
"""
import pandas as pd
import numpy as np

# 复制V2的核心代码
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
STRATEGY_PARAMS = {
    'stop_loss': 1.5,
    'take_profit': 3.0,
    'stc_level': 80,
    'cci_trigger': 50,
}

def calculate_tr(df):
    data = df.copy()
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    return data['tr']

def calculate_cci(df, cci_length=22):
    data = df.copy()
    data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
    tp_sma = data['hlc3'].rolling(window=cci_length).mean()
    mad = data['hlc3'].rolling(window=cci_length).apply(
        lambda x: np.abs(x - x.mean()).mean(),
        raw=False
    )
    data['cci'] = (data['hlc3'] - tp_sma) / (0.015 * mad)
    data['cci'] = data['cci'].fillna(0)
    return data['cci']

def calculate_stc(df, length=10, fast=23, slow=50, aaa=0.5):
    data = df.copy()
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    lowest_macd = macd.rolling(window=length).min()
    highest_macd = macd.rolling(window=length).max()
    k1 = 100 * (macd - lowest_macd) / (highest_macd - lowest_macd)
    k1 = k1.fillna(0)
    d1 = k1.ewm(span=3, adjust=False).mean()
    lowest_d1 = d1.rolling(window=length).min()
    highest_d1 = d1.rolling(window=length).max()
    k2 = 100 * (d1 - lowest_d1) / (highest_d1 - lowest_d1)
    k2 = k2.fillna(0)
    smooth_len = max(1, int(aaa * 10))
    stc = k2.ewm(span=smooth_len, adjust=False).mean()
    return stc

def prepare_data(df):
    df = df.copy()
    df['tr'] = calculate_tr(df)
    df['cci'] = calculate_cci(df)
    df['stc'] = calculate_stc(df)
    df['atr'] = df['tr'].rolling(14).mean()
    df['ma60'] = df['close'].ewm(span=60).mean()
    df['vol_ratio'] = (df['atr'].rolling(20).mean() / df['close'].replace(0, np.nan)).fillna(0) * 100
    df['trend_strength'] = abs(df['close'].ewm(span=20).mean() - df['close'].ewm(span=60).mean()) / df['close'].ewm(span=60).mean().replace(0, np.nan).fillna(1) * 100
    df['is_bullish'] = df['close'].ewm(span=20).mean() > df['close'].ewm(span=60).mean()
    df['vol_ratio'].fillna(0, inplace=True)
    df['trend_strength'].fillna(0, inplace=True)
    df['is_bullish'].fillna(False, inplace=True)
    return df

# 读取数据
df_all = pd.read_csv(r'D:\期货数据\铜期货监控\CCI策略系统\sa0_2020-2025.csv')
df_all['date'] = pd.to_datetime(df_all['date'])
df_all = df_all.sort_values('date')
df_all.set_index('date', inplace=True)
df_all = prepare_data(df_all)

# 测试2024年
year = 2024
start_date = pd.to_datetime(f'{year}-01-01')
end_date = pd.to_datetime(f'{year}-12-31')
df_year = df_all[(df_all.index >= start_date) & (df_all.index <= end_date)].copy()

print(f"调试 2024年 V2交易逻辑")
print("="*80)

year_start_price = df_year['close'].iloc[0]
balance = 100000
position = 0
entry_price = 0.0
entry_date = None
trades = []

for i in range(1, len(df_year)):
    row = df_year.iloc[i]
    current_date = df_year.index[i]

    if pd.isna(row['close']) or pd.isna(row['atr']) or row['atr'] <= 0:
        continue

    # 平仓逻辑
    if position != 0:
        profit_raw = (row['close'] - entry_price) if position == 1 else (entry_price - row['close'])
        atr_value = row['atr']
        stop_loss_distance = atr_value * STRATEGY_PARAMS['stop_loss']
        take_profit_distance = atr_value * STRATEGY_PARAMS['take_profit']

        exit = False
        exit_reason = ''

        # 止盈（原始逻辑）
        if abs(profit_raw) >= take_profit_distance and profit_raw < 0:
            exit = True
            exit_reason = '止盈'

        # ATR止损
        if not exit:
            if profit_raw < -stop_loss_distance:
                exit = True
                exit_reason = '止损'

        # 强制止损
        if not exit:
            max_loss = balance * 0.03
            if position == -1:
                potential_loss = (row['close'] - entry_price) * 20 * MULTIPLIER
            else:
                potential_loss = (entry_price - row['close']) * 20 * MULTIPLIER

            if potential_loss < -max_loss:
                exit = True
                exit_reason = '强制止损'

        if exit:
            pnl = profit_raw * 20 * MULTIPLIER
            commission = abs(pnl) * COMMISSION_RATE * 2
            net_pnl = pnl - commission

            # 打印详细交易信息
            print(f"\n[平仓] {current_date.strftime('%Y-%m-%d')}")
            print(f"  开仓: {entry_date.strftime('%Y-%m-%d')} @ {entry_price:.2f}")
            print(f"  平仓: @ {row['close']:.2f}")
            print(f"  盈亏: {profit_raw:.2f} ({profit_raw/entry_price*100:.2f}%)")
            print(f"  净PnL: {net_pnl:.2f}")
            print(f"  原因: {exit_reason}")
            print(f"  ATR: {atr_value:.2f}, 止损距: {stop_loss_distance:.2f}, 止盈距: {take_profit_distance:.2f}")
            print(f"  条件检查: abs({profit_raw:.2f}) >= {take_profit_distance:.2f}? {abs(profit_raw) >= take_profit_distance}")
            print(f"  条件检查: {profit_raw:.2f} < 0? {profit_raw < 0}")

            balance = balance + net_pnl

            trades.append({
                'date': current_date,
                'entry_date': entry_date,
                'type': exit_reason,
                'pnl': net_pnl,
                'profit_raw': profit_raw,
                'exit_price': row['close'],
                'entry_price': entry_price,
            })

            position = 0

    # 开仓逻辑
    if position == 0:
        ytd_return = (row['close'] - year_start_price) / year_start_price * 100
        if ytd_return > 10:
            continue

        current_vol = row.get('vol_ratio', 0)
        current_stc = row.get('stc', 0)
        current_cci = row.get('cci', 0)

        if (current_stc < STRATEGY_PARAMS['stc_level'] and
            current_cci < STRATEGY_PARAMS['cci_trigger'] and
            current_vol > 0.3 and
            not pd.isna(current_stc) and
            not pd.isna(current_cci)):
            position = -1
            entry_price = row['close']
            entry_date = current_date
            print(f"\n[开仓] {current_date.strftime('%Y-%m-%d')} @ {entry_price:.2f}")

print(f"\n{'='*80}")
print(f"总计: {len(trades)}笔交易")
total_pnl = sum([t['pnl'] for t in trades])
total_return = (balance + total_pnl - 100000) / 100000 * 100
print(f"总收益: {total_return:.2f}%")
