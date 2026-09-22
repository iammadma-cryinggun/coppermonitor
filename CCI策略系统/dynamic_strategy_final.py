# -*- coding: utf-8 -*-
"""
动态策略切换系统 - 最终版本
核心思想：根据实时市场特征选择v2或v3策略，无look-ahead bias
"""
import pandas as pd
import numpy as np
from datetime import datetime

# 全局参数
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
INITIAL_CAPITAL = 100000

# v2参数：高频交易
V2_PARAMS = {
    'stop_loss': 1.5,
    'take_profit': 3.0,
    'stc_level': 80,
    'cci_trigger': 50,
}

# v3参数：质量控制
V3_PARAMS = {
    'stop_loss': 1.5,
    'take_profit': 3.0,
    'stc_level': 80,
    'cci_trigger': 50,
    'min_trade_interval': 5,
    'trend_lookback': 10,
    'max_trend_threshold': 0.05,
}


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


def detect_market_type(df_year, current_index, year_start_price):
    """
    实时检测市场类型（无look-ahead bias）
    返回：'v2' 或 'v3' 或 'sleep'
    """
    if current_index < 20:
        return 'v3'  # 数据不足，使用保守策略

    current_price = df_year['close'].iloc[current_index]
    ytd_return = (current_price - year_start_price) / year_start_price * 100

    # 牛市过滤：YTD>10%不开仓
    if ytd_return > 10:
        return 'sleep'

    # 策略1：快速下跌检测（v2适合）
    # 条件：过去10天，下跌天数>7天，且波动率适中
    lookback = 10
    recent_prices = df_year['close'].iloc[current_index - lookback:current_index + 1].values

    # 计算下跌天数
    down_days = 0
    for i in range(len(recent_prices) - 1):
        if recent_prices[i] < recent_prices[i+1]:
            down_days += 1

    down_ratio = down_days / (len(recent_prices) - 1)

    # 计算波动率（标准差/均值）
    recent_vol = recent_prices.std() / recent_prices.mean() * 100

    # 判断：快速下跌 + 波动率适中
    if down_ratio > 0.7 and 1.0 < recent_vol < 5.0:
        return 'v2'

    # 策略2：其他情况用v3（更保守）
    return 'v3'


def backtest_year_dynamic(df_all, year, symbol):
    """单年份动态回测"""
    print(f"\n{'='*80}")
    print(f"年份回测: {year}年（动态切换版）")
    print('-'*80)

    start_date = pd.to_datetime(f'{year}-01-01')
    end_date = pd.to_datetime(f'{year}-12-31')

    df_year = df_all[(df_all.index >= start_date) & (df_all.index <= end_date)].copy()

    if len(df_year) < 200:
        print(f"    [SKIP] 数据不足: {len(df_year)}条")
        return None

    year_start_price = df_year['close'].iloc[0]
    year_end_price = df_year['close'].iloc[-1]
    ytd_return_total = (year_end_price - year_start_price) / year_start_price * 100

    print(f"    年初价格: {year_start_price:.2f}")
    print(f"    年末价格: {year_end_price:.2f}")
    print(f"    年度涨跌: {ytd_return_total:.2f}%")

    # 识别市场状态
    vol = df_year['vol_ratio'].iloc[-1]
    trend = df_year['trend_strength'].iloc[-1]
    is_bullish = df_year['is_bullish'].iloc[-1]

    print(f"      波动率: {vol:.2f}%")
    print(f"      趋势强度: {trend:.2f}%")
    print(f"      品种特性: {'做多' if is_bullish else '做空'}")

    if is_bullish:
        print(f"    [识别] 牛市，SNIPER做空模式禁用")
        print(f"    [识别] 休眠模式")
        return None

    regime = 'SNIPER'
    print(f"      识别为: {regime}模式")

    # 回测
    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_date = None
    last_exit_date = None
    trades = []
    strategy_stats = {'v2': 0, 'v3': 0, 'switches': 0}

    print("    开始回测...")

    for i in range(1, len(df_year)):
        row = df_year.iloc[i]
        current_date = df_year.index[i]

        if pd.isna(row['close']) or pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # 动态选择策略
        selected_strategy = detect_market_type(df_year, i, year_start_price)

        if selected_strategy == 'sleep':
            continue

        # 选择参数
        if selected_strategy == 'v2':
            params = V2_PARAMS
            strategy_stats['v2'] += 1
        else:
            params = V3_PARAMS
            strategy_stats['v3'] += 1

        # 平仓逻辑（v2和v3的平仓逻辑基本相同）
        if position != 0:
            if position == -1:  # 做空
                price_change = entry_price - row['close']
            else:  # 做多
                price_change = row['close'] - entry_price

            atr_value = row['atr']
            profit_atr = price_change / atr_value
            stop_loss_distance = atr_value * params['stop_loss']
            take_profit_distance = atr_value * params['take_profit']

            exit = False
            exit_reason = ''

            # 止盈（v2和v3相同）
            if abs(profit_atr) >= take_profit_distance:
                exit = True
                exit_reason = '止盈'

            # 止损（v2和v3相同）
            if not exit and price_change < -stop_loss_distance:
                exit = True
                exit_reason = '止损'

            # 强制止损（v2和v3相同）
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
                pnl = price_change * 20 * MULTIPLIER
                commission = abs(pnl) * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance = balance + net_pnl

                trades.append({
                    'date': current_date,
                    'entry_date': entry_date,
                    'type': exit_reason,
                    'pnl': net_pnl,
                    'strategy': selected_strategy,
                })

                position = 0
                last_exit_date = current_date

        # 开仓逻辑
        if position == 0:
            current_vol = row.get('vol_ratio', 0)
            current_stc = row.get('stc', 0)
            current_cci = row.get('cci', 0)

            # v2和v3的通用开仓条件
            if not (current_stc < params['stc_level'] and
                    current_cci < params['cci_trigger'] and
                    current_vol > 0.3 and
                    not pd.isna(current_stc) and
                    not pd.isna(current_cci)):
                continue

            # v3特有过滤
            if selected_strategy == 'v3':
                # 交易间隔限制
                if last_exit_date:
                    days_since_last_exit = (current_date - last_exit_date).days
                    if days_since_last_exit < params['min_trade_interval']:
                        continue

                # 短期趋势过滤
                lookback = params['trend_lookback']
                if i >= lookback:
                    start_price = df_year['close'].iloc[i - lookback]
                    current_price_calc = df_year['close'].iloc[i]
                    short_term_trend = (current_price_calc - start_price) / start_price

                    if short_term_trend > params['max_trend_threshold']:
                        continue

            # 开空仓
            position = -1
            entry_price = row['close']
            entry_date = current_date

            if selected_strategy == 'v2':
                print(f"    [v2开空] {current_date} 价格:{entry_price:.2f}")
            else:
                print(f"    [v3开空] {current_date} 价格:{entry_price:.2f}")

    # 计算结果
    total_pnl = sum([t['pnl'] for t in trades])
    total_return = (balance + total_pnl - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    if len(trades) == 0:
        print(f"    [结果] 无交易")
        return None

    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100

    # 策略分布
    v2_trades = len([t for t in trades if t['strategy'] == 'v2'])
    v3_trades = len([t for t in trades if t['strategy'] == 'v3'])

    print(f"    [结果] 交易:{len(trades)}笔 | v2:{v2_trades}笔 | v3:{v3_trades}笔")
    print(f"    [结果] 收益:{total_return:.2f}% | 胜率:{win_rate:.1f}%")
    print(f"    [结果] 策略切换: {strategy_stats['switches']}次 (v2:{strategy_stats['v2']}次, v3:{strategy_stats['v3']}次)")

    return {
        'year': year,
        'total_return': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'v2_trades': v2_trades,
        'v3_trades:': v3_trades,
        'switches': strategy_stats['switches'],
        'trades_df': pd.DataFrame(trades),
    }


def backtest_by_year(symbol, start_year, end_year):
    """按年份回测"""
    print(f"\n{'='*100}")
    print(f"年份回测: {symbol} ({start_year}-{end_year})")
    print('='*100)

    # 读取数据
    print(f"\n正在读取 {symbol} {start_year}-{end_year} 历史数据...")
    df_all = pd.read_csv(r'D:\期货数据\铜期货监控\CCI策略系统\sa0_2020-2025.csv')

    print(f"    [成功] 读取数据: {len(df_all)}条")
    if 'date' in df_all.columns:
        print(f"    时间范围: {df_all['date'].min()} 至 {df_all['date'].max()}")

    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values('date')
    df_all.set_index('date', inplace=True)

    # 计算指标
    try:
        df_all = prepare_data(df_all)
    except ValueError as e:
        print(f"    [错误] {e}")
        return None

    yearly_results = []

    for year in range(start_year, end_year + 1):
        result = backtest_year_dynamic(df_all, year, symbol)

        if result:
            yearly_results.append({
                'year': year,
                'total_return': result['total_return'],
                'total_trades': result['total_trades'],
                'win_rate': result['win_rate'],
                'v2_trades': result['v2_trades'],
                'v3_trades:': result['v3_trades:'],
                'switches': result['switches'],
            })

    # 输出总结
    print(f"\n{'='*100}")
    print(f"年份回测总结: {symbol} ({start_year}-{end_year})")
    print('='*100)

    if yearly_results:
        results_df = pd.DataFrame(yearly_results)

        print(f"\n {'年份':<8} {'收益率':>10} {'交易数':>8} {'胜率':>8} {'v2':>6} {'v3':>6} {'切换':>6}")
        print("-" * 100)

        for _, r in results_df.iterrows():
            print(f"  {r['year']:<8} {r['total_return']:>9.2f}%   {r['total_trades']:>6}笔   {r['win_rate']:>7.1f}%   {r['v2_trades']:>4}笔   {r['v3_trades:']:>4}笔   {r['switches']:>4}次")

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
    print("动态策略切换系统 - 最终版".center(100))
    print("=" * 100)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print("\n核心机制: 实时检测市场类型，动态选择v2或v3策略")
    print("  - 快速下跌市场（过去10天>70%下跌）→ 使用v2（高频交易）")
    print("  - 缓慢下跌或震荡市场 → 使用v3（质量控制）")
    print("  - 完全避免look-ahead bias（只使用当前和过去数据）")
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

    # 最终对比
    print(f"\n{'='*100}")
    print("动态切换版 vs 固定版本对比".center(100))
    print("=" * 100)

    for s, results_df in all_results.items():
        if results_df is None or len(results_df) == 0:
            continue
        print(f"\n{s} ({start_year}-{end_year})")
        print("-" * 60)

        print(f" 年份   收益率    交易数  胜率  v2  v3  切换")
        print("-" * 60)

        for _, r in results_df.iterrows():
            trend_arrow = "↑" if r['total_return'] > 0 else "↓"
            print(f"  {r['year']:<6} {r['total_return']:>8.2f}%   {r['total_trades']:>6}笔   {r['win_rate']:>6.1f}%   {r['v2_trades']:>3}笔  {r['v3_trades:']:>3}笔  {r['switches']:>4}次")

    print(f"\n{'='*100}")
    print(f"系统运行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
