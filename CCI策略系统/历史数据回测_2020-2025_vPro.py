# -*- coding: utf-8 -*-
"""
历史数据回测系统 - 往年期货数据测试
获取2020-2025年的历史数据进行策略验证
使用全自动S.C.C.系统 v3.0 Pro版逻辑
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import akshare as ak

# ========================================================
# 1. 策略参数仓库
# ========================================================
STRATEGY_REPO = {
    # 【模式A：空头狙击 (SNIPER_SHORT)】
    'SNIPER_SHORT': {
        'desc': '高波-空头狙击',
        'long_enabled': False,
        'short_enabled': True,
        'stc_entry': 80,
        'stc_slope': -1.0,
        'cci_entry': 50,
        'cci_period': 14,
        'vol_filter': 0.5,
        'stop_loss': 1.5,
        'take_profit': 2.5,
        'use_ma_filter': False,
    },

    # 【模式B：多头突击 (SNIPER_LONG)】
    'SNIPER_LONG': {
        'desc': '高波-多头突击',
        'long_enabled': True,
        'short_enabled': False,
        'stc_entry': 20,
        'stc_slope': 1.0,
        'cci_entry': -50,
        'cci_period': 14,
        'vol_filter': 0.8,
        'stop_loss': 1.5,
        'take_profit': 3.0,
        'use_ma_filter': False,
    },

    # 【模式C：稳健趋势 (TREND)】
    'TREND': {
        'desc': '稳健趋势跟随',
        'long_enabled': True,
        'short_enabled': True,
        'stc_entry': 25,
        'stc_short_entry': 75,
        'stc_slope': -1.5,
        'cci_entry': 100,
        'cci_short_entry': -100,
        'cci_period': 20,
        'vol_filter': 1.0,
        'stop_loss': 2.0,
        'take_profit': 999,
        'use_ma_filter': True,
    },

    # 【模式D：休眠 (TRASH)】
    'TRASH': {
        'desc': '休眠模式',
        'long_enabled': False,
        'short_enabled': False,
    }
}

# 全局交易参数
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
INITIAL_CAPITAL = 100000


# ========================================================
# 2. 标准化 STC 计算
# ========================================================
def calculate_stc_standard(df, period_fast=23, period_slow=50, k_period=10, d_period=3):
    """
    标准化 Schaff Trend Cycle 计算
    """
    # 1. MACD
    ema_fast = df['close'].ewm(span=period_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=period_slow, adjust=False).mean()
    macd = ema_fast - ema_slow

    # 2. First Stochastic (%K1)
    min_macd = macd.rolling(window=k_period).min()
    max_macd = macd.rolling(window=k_period).max()
    stoch_k1 = 100 * (macd - min_macd) / (max_macd - min_macd)
    stoch_k1 = stoch_k1.fillna(50)

    # 3. First Smoothing (%D1 of %K1)
    stoch_d1 = stoch_k1.ewm(span=d_period, adjust=False).mean()

    # 4. Second Stochastic (%K2 of %D1)
    min_stoch_d1 = stoch_d1.rolling(window=k_period).min()
    max_stoch_d1 = stoch_d1.rolling(window=k_period).max()
    stoch_k2 = 100 * (stoch_d1 - min_stoch_d1) / (max_stoch_d1 - min_stoch_d1)
    stoch_k2 = stoch_k2.fillna(50)

    # 5. Second Smoothing (%D2 of %K2) → Final STC
    stc = stoch_k2.ewm(span=d_period, adjust=False).mean()

    return stc


# ========================================================
# 3. 市场状态识别
# ========================================================
def detect_market_regime(df):
    """自动识别市场状态"""
    if len(df) < 60:
        return 'TRASH'

    # 计算ATR（用于波动率）
    tr = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1)))
    atr20 = tr.rolling(window=20).mean().iloc[-1]
    current_price = df['close'].iloc[-1]
    vol_ratio = (atr20 / current_price) * 100

    # 计算趋势方向和强度
    ema20 = df['close'].ewm(span=20).mean().iloc[-1]
    ema60 = df['close'].ewm(span=60).mean().iloc[-1]
    trend_strength = abs(ema20 - ema60) / ema60 * 100
    is_bullish = ema20 > ema60

    is_strong_trend = trend_strength > 2.0

    print(f"  [系统自检] 波动:{vol_ratio:.2f}% | 强度:{trend_strength:.2f}% | 多头:{is_bullish}")

    # 自动判定
    if vol_ratio > 2.0:
        return 'SNIPER_LONG' if is_bullish else 'SNIPER_SHORT'
    elif vol_ratio > 0.8 and is_strong_trend:
        return 'TREND'
    else:
        return 'TRASH'


def get_historical_data(symbol, start_year, end_year):
    """获取指定年份范围的期货历史数据"""
    print(f"\n正在获取 {symbol} {start_year}-{end_year} 年的历史数据...")

    try:
        df = ak.futures_main_sina(symbol=symbol)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()

        # 过滤时间范围
        start_date = pd.to_datetime(f'{start_year}-01-01')
        end_date = pd.to_datetime(f'{end_year}-12-31')

        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        if len(df) == 0:
            print(f"    [WARNING] 该年份范围内无数据")
            return None

        df.set_index('date', inplace=True)

        print(f"    [成功] 获取数据量: {len(df)}条")
        print(f"    时间范围: {df.index[0]} 至 {df.index[-1]}")

        return df

    except Exception as e:
        print(f"    [ERROR] 数据获取失败: {e}")
        return None


def backtest_by_year(symbol, start_year, end_year):
    """按年份回测并对比"""
    print(f"\n{'='*100}")
    print(f"年份回测: {symbol} ({start_year}-{end_year})")
    print('='*100)

    # 获取数据
    df = get_historical_data(symbol, start_year, end_year)

    if df is None:
        print(f"\n[FAILED] 无法获取 {symbol} 历史数据")
        return None

    # 分年份回测
    yearly_results = []

    for year in range(start_year, end_year + 1):
        print(f"\n{'='*80}")
        print(f"正在回测: {year}年")
        print('-'*80)

        start_date = pd.to_datetime(f'{year}-01-01')
        end_date = pd.to_datetime(f'{year}-12-31')

        df_year = df[(df.index >= start_date) & (df.index <= end_date)].copy()

        if len(df_year) < 200:
            print(f"    [SKIP] 数据量不足: {len(df_year)}条")
            continue

        # 识别市场状态
        regime = detect_market_regime(df_year)
        params = STRATEGY_REPO[regime]

        if regime == 'TRASH':
            print(f"    [SKIP] 市场状态不适合交易")
            continue

        # 计算指标
        print(f"    计算技术指标...")

        # CCI
        tp = (df_year['high'] + df_year['low'] + df_year['close']) / 3
        sma = tp.rolling(window=params['cci_period']).mean()
        mad = tp.rolling(window=params['cci_period']).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
        df_year['cci'] = (tp - sma) / (0.015 * mad)
        df_year['cci_prev'] = df_year['cci'].shift(1)

        # STC（标准版）
        df_year['stc'] = calculate_stc_standard(df_year)

        # ATR
        df_year['tr'] = np.maximum(df_year['high'] - df_year['low'],
                                 np.abs(df_year['high'] - df_year['close'].shift(1)))
        df_year['atr'] = df_year['tr'].rolling(14).mean()

        # 成交量
        df_year['vol_ma'] = df_year['volume'].rolling(window=5).mean()
        df_year['vol_ratio'] = df_year['volume'] / df_year['vol_ma']

        # 趋势过滤
        df_year['is_bullish_candle'] = df_year['close'] > df_year['open']
        if 'ma60' not in df_year.columns:
            df_year['ma60'] = df_year['close'].ewm(span=60).mean()

        print(f"    识别为: 【{params['desc']}】")

        # 回测
        print(f"    开始回测...")
        balance = INITIAL_CAPITAL
        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        position_type = None

        trades = []

        for i in range(50, len(df_year) - 1):
            row = df_year.iloc[i]
            prev = df_year.iloc[i-1]

            # 平仓逻辑
            if position != 0:
                profit_raw = (entry_price - row['close']) * position
                profit_atr = profit_raw / entry_atr

                exit_triggered = False

                # 固定止盈
                if params['take_profit'] < 999:
                    if profit_atr > params['take_profit']:
                        exit_price = row['close']
                        pnl = (exit_price - entry_price) * position * MULTIPLIER
                        commission = exit_price * abs(position) * MULTIPLIER * COMMISSION_RATE * 2
                        net_pnl = pnl - commission
                        balance += net_pnl

                        trades.append({
                            'date': df_year.index[i],
                            'type': 'take_profit',
                            'pnl': net_pnl,
                        })

                        position = 0
                        exit_triggered = True

                # 移动止损（TREND模式：跌破MA60）
                if not exit_triggered and params.get('use_ma_filter', False):
                    if position > 0 and row['close'] < row['ma60']:
                        exit_price = row['close']
                        pnl = (exit_price - entry_price) * position * MULTIPLIER
                        commission = exit_price * abs(position) * MULTIPLIER * COMMISSION_RATE * 2
                        net_pnl = pnl - commission
                        balance += net_pnl

                        trades.append({
                            'date': df_year.index[i],
                            'type': 'trend_stop',
                            'pnl': net_pnl,
                        })

                        position = 0
                        exit_triggered = True

                # ATR止损（所有模式通用）
                if not exit_triggered:
                    if profit_atr < -params['stop_loss']:
                        stop_price = entry_price * (1 - params['stop_loss']) if position > 0 else \
                                     entry_price * (1 + params['stop_loss'])

                        pnl = (stop_price - entry_price) * position * MULTIPLIER
                        commission = stop_price * abs(position) * MULTIPLIER * COMMISSION_RATE * 2
                        net_pnl = pnl - commission
                        balance += net_pnl

                        trades.append({
                            'date': df_year.index[i],
                            'type': 'atr_stop',
                            'pnl': net_pnl,
                        })

                        position = 0
                        exit_triggered = True

            # 开仓逻辑
            if position == 0:
                # 做空信号
                if params['short_enabled']:
                    threshold = params['stc_entry']
                    cond_stc = (prev['stc'] > threshold) and (row['stc'] < 0)
                    cond_slope = row['stc'] < params['stc_slope']

                    cci_threshold = params.get('stc_short_entry', params['cci_entry'])
                    cond_cci = (prev['cci'] > cci_threshold) and (row['cci'] < cci_threshold)

                    cond_vol = row['vol_ratio'] > params['vol_filter']

                    if cond_stc and cond_slope and cond_cci and cond_vol:
                        position = -1
                        entry_price = row['close']
                        entry_atr = row['atr']

                # 做多信号
                if params['long_enabled']:
                    threshold = params['stc_entry']
                    cond_stc = (prev['stc'] < -threshold) and (row['stc'] > 0)
                    cond_slope = row['stc'] > abs(params['stc_slope'])

                    cci_threshold = params.get('stc_short_entry', params['cci_entry'])
                    cond_cci = (prev['cci'] < -cci_threshold) and (row['cci'] > -cci_threshold)

                    cond_vol = row['vol_ratio'] > params['vol_filter']
                    cond_ma = True
                    if params.get('use_ma_filter', False):
                        cond_ma = row['close'] > row['ma60']

                    if cond_stc and cond_slope and cond_cci and cond_vol and cond_ma:
                        position = 1
                        entry_price = row['close']
                        entry_atr = row['atr']

        # 计算结果
        if len(trades) == 0:
            print(f"    [结果] 无交易")
            continue

        trades_df = pd.DataFrame(trades)
        total_return = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        win_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(win_trades) / len(trades_df) * 100

        print(f"    [结果] 交易:{len(trades_df)}笔 | 收益:{total_return:.2f}% | 胜率:{win_rate:.1f}%")

        yearly_results.append({
            'year': year,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
        })

    # 输出总结
    print(f"\n{'='*100}")
    print(f"年份回测总结: {symbol} ({start_year}-{end_year})")
    print('='*100)

    if yearly_results:
        results_df = pd.DataFrame(yearly_results)

        print(f"\n  {'年份':<8} {'收益率':>10} {'交易数':>8} {'胜率':>8}")
        print("-" * 40)

        for _, r in results_df.iterrows():
            print(f"  {r['year']:<8} {r['total_return']:>9.2f}%   {r['total_trades']:>6}笔   "
                  f"{r['win_rate']:>7.1f}%")

        # 统计
        avg_return = results_df['total_return'].mean()
        std_return = results_df['total_return'].std()
        up_years = len(results_df[results_df['total_return'] > 0])
        total_years = len(results_df)

        print(f"\n  平均收益率: {avg_return:.2f}%")
        print(f"  收益标准差: {std_return:.2f}%")
        print(f"  盈利年份比例: {up_years}/{total_years} ({up_years/total_years*100:.1f}%)")

    return results_df if yearly_results else None


def main():
    print("=" * 100)
    print("历史数据回测系统 - 往年验证".center(100))
    print("=" * 100)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # 测试品种
    symbols = ['sa0', 'cu0']
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

    # 横向对比总结
    print("\n" + "=" * 100)
    print("品种表现对比".center(100))
    print("=" * 100)

    for s, results_df in all_results.items():
        print(f"\n{s} ({start_year}-{end_year})")
        print("-" * 60)

        print(f"  年份   收益率    交易数  胜率")
        print("-" * 60)

        for _, r in results_df.iterrows():
            trend_arrow = "↑" if r['total_return'] > 0 else "↓"
            print(f"  {r['year']:<8} {r['total_return']:>8.2f}%   "
                  f"{r['total_trades']:>6}笔   "
                  f"{r['win_rate']:>6.1f}%")

    print("\n" + "=" * 100)
    print(f"系统运行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
