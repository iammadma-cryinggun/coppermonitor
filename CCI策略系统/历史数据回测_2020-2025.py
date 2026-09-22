# -*- coding: utf-8 -*-
"""
历史数据回测系统 - 往年期货数据测试
获取2020-2025年的历史数据进行策略验证
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import akshare as ak

# 复用策略代码
import sys
sys.path.insert(0, r'D:\期货数据\铜期货监控\CCI策略系统')

# 从v3.0 Pro版导入核心组件
# 我们需要先创建一个精简版


# ========================================================
# 历史数据获取器
# ========================================================
def get_historical_data(symbol, start_year, end_year):
    """
    获取指定年份范围的期货历史数据
    使用主力合约连续数据
    """
    print(f"\n正在获取 {symbol} {start_year}-{end_year} 年的历史数据...")

    try:
        # 获取当前主力合约数据（包含历史）
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


def get_historical_data_by_contracts(symbol, start_year, end_year):
    """
    通过历史合约获取数据（更完整）
    例如：SA2401 (2020年), SA2601 (2026年)
    """
    print(f"\n正在通过历史合约获取 {symbol} {start_year}-{end_year} 年数据...")

    all_data = []

    for year in range(start_year, end_year + 1):
        # 构造合约代码（纯碱：SA + 年份后2位 + 1）
        contract = f"{symbol[:2].upper()}{year}{1}"
        print(f"  正在获取合约: {contract} ...")

        try:
            # 获取历史合约数据
            # Akshare的历史数据接口
            # 注意：需要使用正确的接口
            df = ak.futures_main_sina(symbol=symbol)

            # 过滤当年数据
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df.dropna()

            start_date = pd.to_datetime(f'{year}-01-01')
            end_date = pd.to_datetime(f'{year}-12-31')

            df_year = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            if len(df_year) > 0:
                all_data.append(df_year)
                print(f"    [成功] {year}年: {len(df_year)}条")
            else:
                print(f"    [SKIP] {year}年: 无数据")

        except Exception as e:
            print(f"    [ERROR] {year}年: {e}")

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        print(f"  [总计] 合并后数据: {len(df)}条")
        return df
    else:
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

        if len(df_year) < 200:  # 数据量太少
            print(f"    [SKIP] 数据量不足: {len(df_year)}条")
            continue

        # 计算指标
        print(f"    计算技术指标...")

        # CCI
        tp = (df_year['high'] + df_year['low'] + df_year['close']) / 3
        sma = tp.rolling(window=14).mean()
        mad = tp.rolling(window=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
        df_year['cci'] = (tp - sma) / (0.015 * mad)

        # STC (简化版)
        ema_fast = df_year['close'].ewm(span=23, adjust=False).mean()
        ema_slow = df_year['close'].ewm(span=50, adjust=False).mean()
        macd = ema_fast - ema_slow

        min_macd = macd.rolling(window=10).min()
        max_macd = macd.rolling(window=10).max()

        stc_k1 = 100 * (macd - min_macd) / (max_macd - min_macd)
        stc_k1 = stc_k1.fillna(50)

        stc_d1 = stc_k1.ewm(span=3, adjust=False).mean()

        stc_k2 = 100 * (stc_d1.rolling(window=10).min() - stc_d1.rolling(window=10).max()) / \
                  (stc_d1.rolling(window=10).max() - stc_d1.rolling(window=10).min())
        stc_k2 = stc_k2.fillna(50)

        df_year['stc'] = stc_k2.ewm(span=3, adjust=False).mean()

        df_year['stc_prev'] = df_year['stc'].shift(1)
        df_year['stc_slope'] = df_year['stc'] - df_year['stc_prev']

        # ATR
        df_year['tr'] = np.maximum(df_year['high'] - df_year['low'],
                                 np.abs(df_year['high'] - df_year['close'].shift(1)))
        df_year['atr'] = df_year['tr'].rolling(14).mean()

        # 成交量
        df_year['vol_ma'] = df_year['volume'].rolling(window=5).mean()
        df_year['vol_ratio'] = df_year['volume'] / df_year['vol_ma']

        # 识别市场状态
        print(f"    识别市场状态...")

        atr20 = df_year['atr'].rolling(20).mean()
        current_price = df_year['close'].iloc[-1]
        vol_ratio = (atr20.iloc[-1] / current_price) * 100 if len(atr20) > 0 else 0

        ema20 = df_year['close'].ewm(span=20).mean().iloc[-1]
        ema60 = df_year['close'].ewm(span=60).mean().iloc[-1]
        trend_strength = abs(ema20 - ema60) / ema60 * 100
        is_bullish = ema20 > ema60

        print(f"      波动率: {vol_ratio:.2f}%")
        print(f"      趋势强度: {trend_strength:.2f}%")
        print(f"      趋势方向: {'上升' if is_bullish else '下跌'}")

        # 判定策略模式
        if vol_ratio > 2.0:
            regime = 'SNIPER_LONG' if is_bullish else 'SNIPER_SHORT'
            print(f"      识别为: 高波{'多头' if is_bullish else '空头'}模式")
        elif vol_ratio > 0.8 and trend_strength > 2.0:
            regime = 'TREND'
            print(f"      识别为: 趋势跟随模式")
        else:
            regime = 'TRASH'
            print(f"      识别为: 休眠模式")

        # 参数映射
        params = {
            'SNIPER_SHORT': {
                'long_enabled': False,
                'short_enabled': True,
                'stc_entry': 80,
                'stc_slope': -1.0,
                'cci_entry': 50,
                'cci_period': 14,
                'vol_filter': 0.5,
                'stop_loss': 1.5,
                'take_profit': 2.5,
            },
            'SNIPER_LONG': {
                'long_enabled': True,
                'short_enabled': False,
                'stc_entry': 20,
                'stc_slope': 1.0,
                'cci_entry': -50,
                'cci_period': 14,
                'vol_filter': 0.8,
                'stop_loss': 1.5,
                'take_profit': 3.0,
            },
            'TREND': {
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
            'TRASH': {
                'long_enabled': False,
                'short_enabled': False,
            }
        }[regime]

        if regime == 'TRASH':
            print(f"    [SKIP] 市场状态不适合交易")
            continue

        # 简化回测
        print(f"    开始回测...")
        balance = 100000
        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        position_type = None

        trades = []
        equity = []

        for i in range(10, len(df_year) - 1):
            row = df_year.iloc[i]
            prev = df_year.iloc[i-1]

            # 计算权益
            if position != 0:
                unrealized_pnl = (row['close'] - entry_price) * position * 20
                current_equity = balance + unrealized_pnl
            else:
                current_equity = balance
            equity.append(current_equity)

            # 平仓
            if position != 0:
                profit_raw = (entry_price - row['close']) * position
                profit_atr = profit_raw / entry_atr

                exit_triggered = False

                # 固定止盈
                if params['take_profit'] < 999:
                    if profit_atr > params['take_profit']:
                        exit_price = row['close']
                        pnl = (exit_price - entry_price) * position * 20
                        commission = exit_price * abs(position) * 20 * 0.0001 * 2
                        net_pnl = pnl - commission
                        balance += net_pnl

                        trades.append({
                            'date': row.name,
                            'type': 'take_profit',
                            'pnl': net_pnl,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                        })
                        print(f"    [交易止盈] 日期:{row.name.date()} | 进场:{entry_price:.2f} | 出场:{exit_price:.2f} | 盈亏:{net_pnl:.2f}")

                        position = 0
                        exit_triggered = True

                # ATR止损
                if not exit_triggered:
                    if profit_atr < -params['stop_loss']:
                        stop_price = entry_price * (1 - params['stop_loss']) if position > 0 else \
                                     entry_price * (1 + params['stop_loss'])
                        pnl = (stop_price - entry_price) * position * 20
                        commission = stop_price * abs(position) * 20 * 0.0001 * 2
                        net_pnl = pnl - commission
                        balance += net_pnl

                        trades.append({
                            'date': row.name,
                            'type': 'atr_stop',
                            'pnl': net_pnl,
                            'entry_price': entry_price,
                            'stop_price': stop_price,
                            'position': position,
                        })
                        print(f"    [ATR止损] 日期:{row.name.date()} | 进场:{entry_price:.2f} | 出场:{stop_price:.2f} | 盈亏:{net_pnl:.2f}")

                        position = 0
                        exit_triggered = True

            # 开仓
            if position == 0:
                # 做空
                if params['short_enabled']:
                    cond_stc = (prev['stc'] > params['stc_entry']) and (row['stc'] < params['stc_entry'])
                    cond_slope = row['stc_slope'] < params['stc_slope']
                    cci_threshold = params.get('stc_short_entry', params['cci_entry'])
                    cond_cci = (prev['cci'] > cci_threshold) and (row['cci'] < cci_threshold)

                    if cond_stc and cond_slope and cond_cci:
                        position = -1
                        entry_price = row['close']
                        entry_atr = row['atr']

                # 做多
                if params['long_enabled']:
                    cond_stc = (prev['stc'] < -params['stc_entry']) and (row['stc'] > -params['stc_entry'])
                    cond_slope = row['stc_slope'] > abs(params['stc_slope'])
                    cci_threshold = params.get('stc_short_entry', params['cci_entry'])
                    cond_cci = (prev['cci'] < -cci_threshold) and (row['cci'] > -cci_threshold)

                    if cond_stc and cond_slope and cond_cci:
                        position = 1
                        entry_price = row['close']
                        entry_atr = row['atr']

        # 计算结果
        if len(trades) == 0:
            print(f"    [结果] 无交易")
            continue

        trades_df = pd.DataFrame(trades)
        total_return = (balance - 100000) / 100000 * 100
        win_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(win_trades) / len(trades_df) * 100

        print(f"    [结果] 交易:{len(trades_df)}笔 | 收益:{total_return:.2f}% | 胜率:{win_rate:.1f}%")

        yearly_results.append({
            'year': year,
            'total_return': total_return,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'vol_ratio': vol_ratio,
            'trend_strength': trend_strength,
            'trend_direction': 'up' if is_bullish else 'down',
        })

    # 输出总结
    print(f"\n{'='*100}")
    print(f"年份回测总结: {symbol} ({start_year}-{end_year})")
    print('='*100)

    if yearly_results:
        results_df = pd.DataFrame(yearly_results)

        print(f"\n  {'年份':<8} {'收益率':>10} {'交易数':>8} {'胜率':>8} {'波动率':>10} {'趋势':>8}")
        print("-" * 70)

        for _, r in results_df.iterrows():
            print(f"  {r['year']:<8} {r['total_return']:>9.2f}% {r['total_trades']:>10}笔 "
                  f"{r['win_rate']:>7.1f}% {r['vol_ratio']:>8.2f}% {r['trend_direction']:<6}")

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

        print(f"  年份   收益率    交易数  胜率  波动率  趋势")
        print("-" * 60)

        for _, r in results_df.iterrows():
            trend_arrow = "↑" if r['trend_direction'] == 'up' else "↓"
            print(f"  {r['year']:<8} {r['total_return']:>8.2f}%   "
                  f"{r['total_trades']:>8}笔   "
                  f"{r['win_rate']:>6.1f}%   "
                  f"{r['vol_ratio']:>6.2f}%   "
                  f"{trend_arrow:>6}")

    print("\n" + "=" * 100)
    print(f"系统运行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
