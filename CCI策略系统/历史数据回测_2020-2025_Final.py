# -*- coding: utf-8 -*-
"""
历史数据回测系统 - 往年验证
最终简化版，确保稳定运行
"""
import pandas as pd
import numpy as np
from datetime import datetime

# 全局参数
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
INITIAL_CAPITAL = 100000

# 策略参数
STRATEGY_PARAMS = {
    'SNIPER': {
        'stop_loss': 1.5,
        'take_profit': 3.0,
        'stc_level': 80,
        'cci_trigger': 50,
    },
    'TREND': {
        'stop_loss': 2.0,
        'stc_level': 60,
        'cci_trigger': 100,
    }
}


def prepare_data(df):
    """计算指标"""
    df['atr'] = df['tr'].rolling(14).mean()
    df['ma60'] = df['close'].ewm(span=60).mean()
    df['vol_ratio'] = (df['atr'].rolling(20).mean() / df['close']) * 100
    df['trend_strength'] = abs(df['close'].ewm(span=20).mean() - df['close'].ewm(span=60).mean()) / df['close'].ewm(span=60).mean() * 100
    df['is_bullish'] = df['close'].ewm(span=20).mean() > df['close'].ewm(span=60).mean()
    return df


def backtest_year(df_all, year, symbol):
    """单年份回测"""
    print(f"\n正在回测: {year}年")
    print('-'*80)

    start_date = pd.to_datetime(f'{year}-01-01')
    end_date = pd.to_datetime(f'{year}-12-31')

    df_year = df_all[(df_all.index >= start_date) & (df_all.index <= end_date)]

    if len(df_year) < 200:
        print(f"    [SKIP] 数据不足: {len(df_year)}条")
        return None

    # 识别市场状态
    vol = df_year['vol_ratio'].iloc[-1]
    trend = df_year['trend_strength'].iloc[-1]
    is_bullish = df_year['is_bullish'].iloc[-1]

    params = None
    regime = ''

    if vol > 2.0:
        regime = 'SNIPER'
        params = STRATEGY_PARAMS['SNIPER']
    elif trend > 2.0 and is_bullish:
        regime = 'TREND'
        params = STRATEGY_PARAMS['TREND']
    else:
        print(f"    [识别] 休眠模式 (波动:{vol:.2f}% 趋势:{trend:.2f}%)")
        return None

    print(f"    [识别] {regime}模式")

    # 回测
    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    trades = []

    for i in range(1, len(df_year)):
        row = df_year.iloc[i]

        # 平仓
        if position != 0:
            profit_raw = (row['close'] - entry_price) if position == 1 else (entry_price - row['close'])
            profit_atr = profit_raw / row['atr']

            exit = False
            exit_reason = ''

            # 止盈
            if regime == 'SNIPER' and profit_atr > params['take_profit']:
                exit = True
                exit_reason = '止盈'

            # 移动止损
            if not exit and regime == 'TREND' and position == 1 and row['close'] < row['ma60']:
                exit = True
                exit_reason = '趋势破'

            # ATR止损
            if not exit:
                if profit_atr < -params['stop_loss']:
                    exit = True
                    exit_reason = '止损'

            if exit:
                pnl = profit_raw * 20 * MULTIPLIER
                commission = abs(pnl) * COMMISSION_RATE * 2
                net_pnl = pnl - commission

                balance = balance + net_pnl

                trades.append({
                    'date': df_year.index[i],
                    'type': exit_reason,
                    'pnl': net_pnl,
                })

                position = 0

        # 开仓
        if position == 0:
            # 做空
            if regime == 'SNIPER':
                # STC回落
                if row['stc'] < params['stc_level'] and row['cci'] < params['cci_trigger'] and row['vol_ratio'] > 0.3:
                    position = -1
                    entry_price = row['close']
                    print(f"    [开空] {df_year.index[i]} 价格:{entry_price:.2f}")

            # 做多
            if regime == 'TREND':
                # STC突破
                if row['stc'] > params['stc_level'] and row['cci'] > params['cci_trigger'] - 10 and row['vol_ratio'] > 1.0 and row['is_bullish'] and row['close'] > row['ma60']:
                    position = 1
                    entry_price = row['close']
                    print(f"    [开多] {df_year.index[i]} 价格:{entry_price:.2f}")

    # 计算结果
    total_pnl = sum([t['pnl'] for t in trades])
    total_return = (balance + total_pnl - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    if len(trades) == 0:
        print(f"    [结果] 无交易")
        return None

    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100

    return {
        'year': year,
        'regime': regime,
        'total_return': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'trades_df': pd.DataFrame(trades),
    }


def backtest_by_year(symbol, start_year, end_year):
    """按年份回测并对比"""
    print(f"\n{'='*100}")
    print(f"年份回测: {symbol} ({start_year}-{end_year})")
    print('='*100)

    # 读取数据
    df_all = pd.read_csv(r'D:\期货数据\铜期货监控\CCI策略系统\sa0_2020-2025.csv')
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values('date')
    df_all.set_index('date', inplace=True)

    yearly_results = []

    for year in range(start_year, end_year + 1):
        result = backtest_year(df_all, year, symbol)

        if result:
            yearly_results.append({
                'year': year,
                'regime': result['regime'],
                'total_return': result['total_return'],
                'total_trades': result['total_trades'],
                'win_rate': result['win_rate'],
            })

            print(f"    [结果] 交易:{result['total_trades']}笔 | 收益:{result['total_return']:.2f}% | 胜率:{result['win_rate']:.1f}%")

    # 输出总结
    print(f"\n{'='*100}")
    print(f"年份回测总结: {symbol} ({start_year}-{end_year})")
    print('='*100)

    if yearly_results:
        results_df = pd.DataFrame(yearly_results)

        print(f"\n {'年份':<8} {'模式':<12} {'收益率':>10} {'交易数':>8} {'胜率':>8}")
        print("-" * 100)

        for _, r in results_df.iterrows():
            print(f"  {r['year']:<8} {r['regime']:<12} {r['total_return']:>9.2f}%   {r['total_trades']:>6}笔   {r['win_rate']:>7.1f}%")

        # 统计
        avg_return = results_df['total_return'].mean()
        std_return = results_df['total_return'].std()
        up_years = len(results_df[results_df['total_return'] > 0])
        total_years = len(results_df)

        print(f"\n 平均收益率: {avg_return:.2f}%")
        print(f" 收益标准差: {std_return:.2f}%")
        print(f" 盈利年份比例: {up_years}/{total_years} ({up_years/total_years*100:.1f}%)")

    return results_df


def main():
    print("=" * 100)
    print("历史数据回测系统 - 往年验证".center(100))
    print("=" * 100)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # 测试品种
    symbols = ['sa0']
    start_year = 2020
    end_year = 2025

    all_results = {}

    for s in symbols:
        print(f"\n{'='*100}")
        print(f"测试品种: {s}")
        print('='*100)

        results_df = backtest_by_year(s, start_year, end_year)

        if results_df is not None:
            all_results[s] = results_df

    # 横向对比
    print(f"\n{'='*100}")
    print("品种表现对比".center(100))
    print("=" * 100)

    for s, results_df in all_results.items():
        if results_df is None or len(results_df) == 0:
            continue
        print(f"\n{s} ({start_year}-{end_year})")
        print("-" * 60)

        print(f" 年份   收益率    交易数  胜率")
        print("-" * 60)

        for _, r in results_df.iterrows():
            trend_arrow = "↑" if r['total_return'] > 0 else "↓"
            print(f"  {r['year']:<8} {r['total_return']:>9.2f}%   {r['total_trades']:>6}笔   {r['win_rate']:>7.1f}%")

    print(f"\n{'='*100}")
    print(f"系统运行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
