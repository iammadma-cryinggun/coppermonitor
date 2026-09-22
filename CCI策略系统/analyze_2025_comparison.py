# -*- coding: utf-8 -*-
"""
2025年详细对比分析工具
对比v2和v3在2025年的具体交易差异
"""
import pandas as pd
import numpy as np
from datetime import datetime

# 全局参数
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
INITIAL_CAPITAL = 100000


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


def prepare_data(df):
    """计算指标"""
    df = df.copy()
    df['tr'] = calculate_tr(df)
    df['cci'] = calculate_cci(df)
    df['stc'] = calculate_stc(df)
    df['atr'] = df['tr'].rolling(14).mean()
    df['ma20'] = df['close'].ewm(span=20).mean()
    df['ma60'] = df['close'].ewm(span=60).mean()
    df['vol_ratio'] = (df['atr'].rolling(20).mean() / df['close'].replace(0, np.nan)).fillna(0) * 100
    df['trend_strength'] = abs(df['close'].ewm(span=20).mean() - df['close'].ewm(span=60).mean()) / df['close'].ewm(span=60).mean().replace(0, np.nan).fillna(1) * 100
    df['is_bullish'] = df['ma20'] > df['ma60']
    df['vol_ratio'].fillna(0, inplace=True)
    df['trend_strength'].fillna(0, inplace=True)
    df['is_bullish'].fillna(False, inplace=True)
    return df


def backtest_v2_2025(df_all, symbol):
    """v2版本回测2025年"""
    print("\n" + "="*100)
    print("v2版本 - 2025年回测")
    print("="*100)

    # 识别市场状态
    vol = df_all[(df_all.index >= pd.to_datetime('2025-01-01')) &
              (df_all.index <= pd.to_datetime('2025-12-31'))]['vol_ratio'].iloc[-1]
    trend = df_all[(df_all.index >= pd.to_datetime('2025-01-01')) &
               (df_all.index <= pd.to_datetime('2025-12-31'))]['trend_strength'].iloc[-1]
    is_bullish = df_all[(df_all.index >= pd.to_datetime('2025-01-01')) &
                    (df_all.index <= pd.to_datetime('2025-12-31'))]['is_bullish'].iloc[-1]

    regime = ''
    if vol > 1.5:
        if not is_bullish:
            regime = 'SNIPER'

    start_date = pd.to_datetime('2025-01-01')
    end_date = pd.to_datetime('2025-12-31')
    df_year = df_all[(df_all.index >= start_date) & (df_all.index <= end_date)].copy()

    year_start_price = df_year['close'].iloc[0]
    year_end_price = df_year['close'].iloc[-1]
    ytd_return_total = (year_end_price - year_start_price) / year_start_price * 100

    print(f"2025年市场概况:")
    print(f"  年初价格: {year_start_price:.2f}")
    print(f"  年末价格: {year_end_price:.2f}")
    print(f"  年度涨跌: {ytd_return_total:.2f}%")

    params = {
        'stop_loss': 1.5,
        'take_profit': 3.0,
        'stc_level': 80,
        'cci_trigger': 50,
    }

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_date = None
    trades = []

    for i in range(1, len(df_year)):
        row = df_year.iloc[i]
        current_date = df_year.index[i]

        if pd.isna(row['close']) or pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # YTD趋势
        ytd_return = (row['close'] - year_start_price) / year_start_price * 100

        # 平仓
        if position != 0:
            profit_raw = (row['close'] - entry_price) if position == 1 else (entry_price - row['close'])
            atr_value = row['atr']
            stop_loss_distance = atr_value * params['stop_loss']
            take_profit_distance = atr_value * params['take_profit']

            exit = False
            exit_reason = ''

            if abs(profit_raw) >= take_profit_distance and profit_raw < 0:
                exit = True
                exit_reason = '止盈'

            if profit_raw < -stop_loss_distance:
                exit = True
                exit_reason = '止损'

            if exit:
                pnl = profit_raw * 20 * MULTIPLIER
                commission = abs(pnl) * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance = balance + net_pnl

                trades.append({
                    'date': current_date,
                    'entry_date': entry_date,
                    'pnl': net_pnl,
                    'ytd_trend': ytd_return,
                    'exit_reason': exit_reason,
                })

                position = 0

        # 开仓（v2逻辑）
        if position == 0:
            if ytd_return > 10:  # 牛市过滤
                continue

            current_vol = row.get('vol_ratio', 0)
            current_stc = row.get('stc', 0)
            current_cci = row.get('cci', 0)

            if (current_stc < params['stc_level'] and
                current_cci < params['cci_trigger'] and
                current_vol > 0.3 and
                not pd.isna(current_stc) and
                not pd.isna(current_cci)):
                position = -1
                entry_price = row['close']
                entry_date = current_date

    # 统计
    total_pnl = sum([t['pnl'] for t in trades])
    total_return = (balance + total_pnl - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    if len(trades) == 0:
        print("无交易")
        return None

    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100

    print(f"\nv2版本2025年结果:")
    print(f"  交易数: {len(trades)}笔")
    print(f"  收益率: {total_return:.2f}%")
    print(f"  胜率: {win_rate:.1f}%")

    if len(win_trades) > 0:
        print(f"  平均盈利: {np.mean([t['pnl'] for t in win_trades]):.2f}")
        print(f"  最大盈利: {max([t['pnl'] for t in win_trades]):.2f}")

    lose_trades = [t for t in trades if t['pnl'] < 0]
    if len(lose_trades) > 0:
        print(f"  平均亏损: {np.mean([t['pnl'] for t in lose_trades]):.2f}")
        print(f"  最大亏损: {min([t['pnl'] for t in lose_trades]):.2f}")

    return {
        'version': 'v2',
        'total_return': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'trades_df': pd.DataFrame(trades),
    }


def backtest_v3_2025(df_all, symbol):
    """v3版本回测2025年"""
    print("\n" + "="*100)
    print("v3版本 - 2025年回测")
    print("="*100)

    # 识别市场状态
    vol = df_all[(df_all.index >= pd.to_datetime('2025-01-01')) &
              (df_all.index <= pd.to_datetime('2025-12-31'))]['vol_ratio'].iloc[-1]
    trend = df_all[(df_all.index >= pd.to_datetime('2025-01-01')) &
               (df_all.index <= pd.to_datetime('2025-12-31'))]['trend_strength'].iloc[-1]
    is_bullish = df_all[(df_all.index >= pd.to_datetime('2025-01-01')) &
                    (df_all.index <= pd.to_datetime('2025-12-31'))]['is_bullish'].iloc[-1]

    regime = ''
    if vol > 1.5:
        if not is_bullish:
            regime = 'SNIPER'

    start_date = pd.to_datetime('2025-01-01')
    end_date = pd.to_datetime('2025-12-31')
    df_year = df_all[(df_all.index >= start_date) & (df_all.index <= end_date)].copy()

    year_start_price = df_year['close'].iloc[0]
    year_end_price = df_year['close'].iloc[-1]
    ytd_return_total = (year_end_price - year_start_price) / year_start_price * 100

    print(f"2025年市场概况:")
    print(f"  年初价格: {year_start_price:.2f}")
    print(f"  年末价格: {year_end_price:.2f}")
    print(f"  年度涨跌: {ytd_return_total:.2f}%")

    params = {
        'stop_loss': 1.5,
        'take_profit': 3.0,
        'stc_level': 80,
        'cci_trigger': 50,
        'min_trade_interval': 5,
        'trend_lookback': 10,
        'max_trend_threshold': 0.05,
        'partial_close_atr': 2.5,
        'full_close_atr': 4.5,
    }

    balance = INITIAL_CAPITAL
    position = 0
    position_size = 0
    entry_price = 0.0
    entry_date = None
    last_exit_date = None
    trades = []

    for i in range(1, len(df_year)):
        row = df_year.iloc[i]
        current_date = df_year.index[i]

        if pd.isna(row['close']) or pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # YTD趋势
        ytd_return = (row['close'] - year_start_price) / year_start_price * 100

        # 短期趋势
        lookback = params['trend_lookback']
        if i < lookback:
            short_term_trend = 0
        else:
            start_price = df_year['close'].iloc[i - lookback]
            current_price = df_year['close'].iloc[i]
            short_term_trend = (current_price - start_price) / start_price

        # 平仓
        if position != 0:
            atr_value = row['atr']

            if position == -1:  # 做空
                price_change = entry_price - row['close']
            else:  # 做多
                price_change = row['close'] - entry_price

            profit_atr = price_change / atr_value

            exit = False
            exit_reason = ''
            close_size = 0

            # 分批止盈
            if regime == 'SNIPER' and position == -1 and position_size == 20:
                partial_threshold = params.get('partial_close_atr', 2.5)
                full_threshold = params.get('full_close_atr', 4.5)

                if profit_atr > partial_threshold:
                    close_size = position_size // 2
                    pnl = price_change * close_size * MULTIPLIER
                    commission = abs(pnl) * COMMISSION_RATE * 2
                    net_pnl = pnl - commission

                    balance = balance + net_pnl

                    trades.append({
                        'date': current_date,
                        'entry_date': entry_date,
                        'pnl': net_pnl,
                        'ytd_trend': ytd_return,
                        'exit_reason': '止盈(半仓)',
                    })

                    position_size = position_size - close_size

                if profit_atr > full_threshold and position_size > 0:
                    close_size = position_size
                    pnl = price_change * close_size * MULTIPLIER
                    commission = abs(pnl) * COMMISSION_RATE * 2
                    net_pnl = pnl - commission

                    balance = balance + net_pnl

                    trades.append({
                        'date': current_date,
                        'entry_date': entry_date,
                        'pnl': net_pnl,
                        'ytd_trend': ytd_return,
                        'exit_reason': '止盈(全仓)',
                    })

                    exit = True
                    exit_reason = '止盈(全仓)'

            # ATR止损
            if not exit:
                stop_loss_distance = atr_value * params['stop_loss']

                if price_change < -stop_loss_distance:
                    exit = True
                    close_size = position_size if close_size == 0 else position_size
                    exit_reason = '止损'

            # 硬止损（3%本金）
            if not exit:
                max_loss = balance * 0.03
                remaining_size = position_size if close_size == 0 else position_size

                if position == -1:
                    potential_loss = (row['close'] - entry_price) * remaining_size * MULTIPLIER
                else:
                    potential_loss = (entry_price - row['close']) * remaining_size * MULTIPLIER

                if potential_loss < -max_loss:
                    exit = True
                    close_size = remaining_size
                    exit_reason = '强制止损'

            if exit:
                final_close_size = position_size if close_size == 0 else close_size
                if position == -1:
                    final_price_change = entry_price - row['close']
                else:
                    final_price_change = row['close'] - entry_price

                pnl = final_price_change * final_close_size * MULTIPLIER
                commission = abs(pnl) * COMMISSION_RATE * 2
                net_pnl = pnl - commission

                balance = balance + net_pnl

                trades.append({
                    'date': current_date,
                    'entry_date': entry_date,
                    'pnl': net_pnl,
                    'ytd_trend': ytd_return,
                    'exit_reason': exit_reason,
                })

                position = 0
                position_size = 0
                last_exit_date = current_date

        # 开仓（v3逻辑）
        if position == 0:
            if ytd_return > 10:
                continue

            if last_exit_date:
                days_since_last_exit = (current_date - last_exit_date).days
                if days_since_last_exit < params['min_trade_interval']:
                    continue

            current_vol = row.get('vol_ratio', 0)
            current_stc = row.get('stc', 0)
            current_cci = row.get('cci', 0)

            if short_term_trend > params['max_trend_threshold']:
                continue

            if (current_stc < params['stc_level'] and
                current_cci < params['cci_trigger'] and
                current_vol > 0.3 and
                not pd.isna(current_stc) and
                not pd.isna(current_cci)):
                position = -1
                position_size = 20
                entry_price = row['close']
                entry_date = current_date

    # 统计
    total_pnl = sum([t['pnl'] for t in trades])
    total_return = (balance + total_pnl - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    if len(trades) == 0:
        print("无交易")
        return None

    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100

    print(f"\nv3版本2025年结果:")
    print(f"  交易数: {len(trades)}笔")
    print(f"  收益率: {total_return:.2f}%")
    print(f"  胜率: {win_rate:.1f}%")

    if len(win_trades) > 0:
        print(f"  平均盈利: {np.mean([t['pnl'] for t in win_trades]):.2f}")
        print(f"  最大盈利: {max([t['pnl'] for t in win_trades]):.2f}")

    lose_trades = [t for t in trades if t['pnl'] < 0]
    if len(lose_trades) > 0:
        print(f"  平均亏损: {np.mean([t['pnl'] for t in lose_trades]):.2f}")
        print(f"  最大亏损: {min([t['pnl'] for t in lose_trades]):.2f}")

    # 分析分批止盈
    partial_close_trades = [t for t in trades if 'exit_reason' in t and t['exit_reason'] == '止盈(半仓)']
    if len(partial_close_trades) > 0:
        partial_pnl = sum([t['pnl'] for t in partial_close_trades])
        print(f"  分批止盈: {len(partial_close_trades)}笔, 总计: {partial_pnl:.2f}")

    return {
        'version': 'v3',
        'total_return': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'trades_df': pd.DataFrame(trades),
    }


def analyze_detailed(v2_result, v3_result):
    """详细分析两个版本的差异"""
    print("\n" + "="*100)
    print("v2 vs v3 详细对比 - 2025年")
    print("="*100)

    v2_trades = v2_result['trades_df']
    v3_trades = v3_result['trades_df']

    print(f"\n基础对比:")
    print(f"  v2交易数: {v2_result['total_trades']}笔")
    print(f"  v3交易数: {v3_result['total_trades']}笔")
    print(f"  v2收益率: {v2_result['total_return']:.2f}%")
    print(f"  v3收益率: {v3_result['total_return']:.2f}%")
    print(f"  收益差异: {v3_result['total_return'] - v2_result['total_return']:.2f}%")
    print(f"  交易数差异: {v3_result['total_trades'] - v2_result['total_trades']}笔")

    # 月度交易分布
    v2_trades_copy = v2_trades.copy()
    v3_trades_copy = v3_trades.copy()

    v2_trades_copy['month'] = pd.to_datetime(v2_trades_copy['date']).dt.month
    v3_trades_copy['month'] = pd.to_datetime(v3_trades_copy['date']).dt.month

    print(f"\n月度交易分布:")
    print(f"  {'月份':<6} {'v2交易数':>10} {'v3交易数':>10}")
    print("-" * 30)

    for month in range(1, 13):
        v2_count = len(v2_trades_copy[v2_trades_copy['month'] == month])
        v3_count = len(v3_trades_copy[v3_trades_copy['month'] == month])
        if v2_count > 0 or v3_count > 0:
            print(f"  {month:>4}月    {v2_count:>8}笔    {v3_count:>8}笔")

    # 平仓原因分析
    print(f"\nv2平仓原因分析:")
    v2_by_reason = v2_trades.groupby('exit_reason').size()
    for reason, count in v2_by_reason.items():
        avg_pnl = v2_trades[v2_trades['exit_reason'] == reason]['pnl'].mean()
        print(f"  {reason}: {count}笔, 平均盈亏: {avg_pnl:.2f}")

    print(f"\nv3平仓原因分析:")
    v3_by_reason = v3_trades.groupby('exit_reason').size()
    for reason, count in v3_by_reason.items():
        avg_pnl = v3_trades[v3_trades['exit_reason'] == reason]['pnl'].mean()
        print(f"  {reason}: {count}笔, 平均盈亏: {avg_pnl:.2f}")

    # 关键发现
    print("\n关键发现:")

    # 找出v3有但v2没有的交易
    v2_dates = set(pd.to_datetime(v2_trades['date']).dt.date)
    v3_dates = set(pd.to_datetime(v3_trades['date']).dt.date)

    v3_only_dates = v3_dates - v2_dates
    v2_only_dates = v2_dates - v3_dates

    print(f"  1. v3独有交易日期: {len(v3_only_dates)}个")
    print(f"  2. v2独有交易日期: {len(v2_only_dates)}个")
    print(f"  3. 共同交易日期: {len(v2_dates & v3_dates)}个")

    if len(v3_only_dates) > 0:
        print(f"\n  v3独有交易示例（前5个）:")
        for date in list(v3_only_dates)[:5]:
            v3_trade = v3_trades[pd.to_datetime(v3_trades['date']).dt.date == date].iloc[0]
            print(f"    {date}: {v3_trade['pnl']:.2f} ({v3_trade['exit_reason']})")


def main():
    print("=" * 100)
    print("2025年详细对比分析".center(100))
    print("=" * 100)

    # 读取数据
    df_all = pd.read_csv(r'D:\期货数据\铜期货监控\CCI策略系统\sa0_2020-2025.csv')
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values('date')
    df_all.set_index('date', inplace=True)

    # 计算指标
    df_all = prepare_data(df_all)

    # 回测v2和v3
    v2_result = backtest_v2_2025(df_all, 'sa0')
    v3_result = backtest_v3_2025(df_all, 'sa0')

    # 详细分析
    if v2_result and v3_result:
        analyze_detailed(v2_result, v3_result)

    print("\n" + "="*100)
    print("分析完成")
    print("="*100)


if __name__ == "__main__":
    main()
