# -*- coding: utf-8 -*-
"""
交易详情分析工具
分析2021-2022年亏损原因
"""
import pandas as pd
import numpy as np
from datetime import datetime

# 复制计算函数
def calculate_tr(df):
    """计算 True Range (TR)"""
    data = df.copy()
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    return data['tr']

def calculate_cci(df, cci_length=22):
    """计算 CCI 指标"""
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
    """计算 STC 指标"""
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

def analyze_year(df_all, year):
    """分析单年交易"""
    print(f"\n{'='*100}")
    print(f"详细分析: {year}年")
    print('='*100)

    start_date = pd.to_datetime(f'{year}-01-01')
    end_date = pd.to_datetime(f'{year}-12-31')
    df_year = df_all[(df_all.index >= start_date) & (df_all.index <= end_date)].copy()

    # 市场特征
    year_start_price = df_year['close'].iloc[0]
    year_end_price = df_year['close'].iloc[-1]
    year_return = (year_end_price - year_start_price) / year_start_price * 100

    print(f"\n【市场环境】")
    print(f"  年初价格: {year_start_price:.2f}")
    print(f"  年末价格: {year_end_price:.2f}")
    print(f"  年度涨跌: {year_return:+.2f}%")
    print(f"  交易天数: {len(df_year)}")

    # 趋势分析
    df_year['ma20'] = df_year['close'].ewm(span=20).mean()
    df_year['ma60'] = df_year['close'].ewm(span=60).mean()
    is_bullish = (df_year['ma20'].iloc[-1] > df_year['ma60'].iloc[-1])

    print(f"  趋势状态: {'牛市' if is_bullish else '熊市'} (MA20={'>' if is_bullish else '<'}MA60)")

    # 回测交易
    MULTIPLIER = 20
    COMMISSION_RATE = 0.0001
    INITIAL_CAPITAL = 100000

    params = {
        'stop_loss': 1.5,
        'take_profit': 3.0,
        'stc_level': 80,
        'cci_trigger': 50,
    }

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    trades = []

    for i in range(1, len(df_year)):
        row = df_year.iloc[i]

        if pd.isna(row['close']) or pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # 平仓
        if position != 0:
            profit_raw = (row['close'] - entry_price) if position == 1 else (entry_price - row['close'])
            profit_atr = profit_raw / row['atr']

            exit = False
            exit_reason = ''

            if profit_atr > params['take_profit']:
                exit = True
                exit_reason = '止盈'
            elif profit_atr < -params['stop_loss']:
                exit = True
                exit_reason = '止损'

            if exit:
                pnl = profit_raw * 20 * MULTIPLIER
                commission = abs(pnl) * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance = balance + net_pnl

                trades.append({
                    'date': df_year.index[i],
                    'entry_date': entry_date,
                    'direction': '做多' if position == 1 else '做空',
                    'entry_price': entry_price,
                    'exit_price': row['close'],
                    'pnl': net_pnl,
                    'return_pct': net_pnl / INITIAL_CAPITAL * 100,
                    'reason': exit_reason,
                    'hold_days': (df_year.index[i] - entry_date).days,
                })

                position = 0

        # 开仓（SNIPER模式：做空）
        if position == 0:
            current_vol = row.get('vol_ratio', 0)
            current_stc = row.get('stc', 0)
            current_cci = row.get('cci', 0)

            if (current_stc < params['stc_level'] and
                current_cci < params['cci_trigger'] and
                current_vol > 0.3):
                position = -1
                entry_price = row['close']
                entry_date = df_year.index[i]

    # 统计分析
    if len(trades) == 0:
        print(f"\n【交易记录】")
        print(f"  无交易")
        return

    trades_df = pd.DataFrame(trades)

    total_pnl = trades_df['pnl'].sum()
    total_return = total_pnl / INITIAL_CAPITAL * 100

    win_trades = trades_df[trades_df['pnl'] > 0]
    lose_trades = trades_df[trades_df['pnl'] < 0]

    print(f"\n【交易统计】")
    print(f"  总交易: {len(trades_df)}笔")
    print(f"  盈利交易: {len(win_trades)}笔")
    print(f"  亏损交易: {len(lose_trades)}笔")
    print(f"  总收益: {total_pnl:.2f}")
    print(f"  收益率: {total_return:.2f}%")
    print(f"  胜率: {len(win_trades)/len(trades_df)*100:.1f}%")

    if len(win_trades) > 0:
        avg_win = win_trades['pnl'].mean()
        max_win = win_trades['pnl'].max()
        print(f"\n【盈利交易】")
        print(f"  平均盈利: {avg_win:.2f}")
        print(f"  最大盈利: {max_win:.2f}")
        print(f"  平均持仓: {win_trades['hold_days'].mean():.1f}天")

    if len(lose_trades) > 0:
        avg_lose = lose_trades['pnl'].mean()
        max_lose = lose_trades['pnl'].min()
        print(f"\n【亏损交易】")
        print(f"  平均亏损: {avg_lose:.2f}")
        print(f"  最大亏损: {max_lose:.2f}")
        print(f"  平均持仓: {lose_trades['hold_days'].mean():.1f}天")

        # 盈亏比
        if len(win_trades) > 0:
            profit_factor = abs(win_trades['pnl'].sum() / lose_trades['pnl'].sum())
            print(f"\n【风险指标】")
            print(f"  盈亏比: {profit_factor:.2f}")

    # 分析亏损原因
    if len(lose_trades) > 0:
        print(f"\n【亏损原因分析】")
        lose_by_reason = lose_trades.groupby('reason').size()
        for reason, count in lose_by_reason.items():
            print(f"  {reason}: {count}笔 ({count/len(lose_trades)*100:.1f}%)")

    # 显示最大亏损交易
    if len(lose_trades) > 0:
        worst_trade = lose_trades.loc[lose_trades['pnl'].idxmin()]
        print(f"\n【最大亏损交易】")
        print(f"  开仓: {worst_trade['entry_date'].strftime('%Y-%m-%d')}")
        print(f"  平仓: {worst_trade['date'].strftime('%Y-%m-%d')}")
        print(f"  方向: {worst_trade['direction']}")
        print(f"  开仓价: {worst_trade['entry_price']:.2f}")
        print(f"  平仓价: {worst_trade['exit_price']:.2f}")
        print(f"  亏损额: {worst_trade['pnl']:.2f}")
        print(f"  亏损率: {worst_trade['return_pct']:.2f}%")
        print(f"  持仓: {worst_trade['hold_days']}天")

def main():
    print("=" * 100)
    print("交易详情分析工具".center(100))
    print("=" * 100)

    # 读取数据
    df_all = pd.read_csv(r'D:\期货数据\铜期货监控\CCI策略系统\sa0_2020-2025.csv')
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values('date')
    df_all.set_index('date', inplace=True)

    # 计算指标
    print("计算技术指标...")
    df_all['tr'] = calculate_tr(df_all)
    df_all['cci'] = calculate_cci(df_all)
    df_all['stc'] = calculate_stc(df_all)
    df_all['atr'] = df_all['tr'].rolling(14).mean()
    df_all['vol_ratio'] = (df_all['atr'].rolling(20).mean() / df_all['close'].replace(0, np.nan)).fillna(0) * 100

    # 分析重点年份
    for year in [2020, 2021, 2022, 2023]:
        analyze_year(df_all, year)

    print(f"\n{'='*100}")
    print("分析完成")
    print("=" * 100)

if __name__ == "__main__":
    main()
