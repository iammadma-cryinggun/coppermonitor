# -*- coding: utf-8 -*-
"""
全自动 S.C.C. 系统 v3.1 (复苏版)
核心修复：
1. 动态识别 (每根K线独立判断)
2. 解耦信号 (STC环境+ CCI扳机)
3. 修正收益率计算
"""
import pandas as pd
import numpy as np
from datetime import datetime

# ========================================================
# 1. 策略参数库
# ========================================================
STRATEGY_REPO = {
    'SNIPER': {  # 狙击模式
        'stc_level': 80,
        'cci_trigger': 50,
        'stop_loss': 1.5,
        'take_profit': 3.0,
    },
    'TREND': {  # 趋势模式
        'stc_level': 60,
        'cci_trigger': 100,
        'stop_loss': 2.0,
        'take_profit': 999,  # 移动止损
    }
}

# 全局参数
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
INITIAL_CAPITAL = 100000


# ========================================================
# 2. 核心计算模块
# ========================================================
def prepare_data(df):
    """一次性计算所有指标"""
    df['tr'] = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1)))
    df['atr'] = df['tr'].rolling(14).mean()
    df['ma60'] = df['close'].ewm(span=60).mean()

    # 动态波动率
    df['vol_ratio'] = (df['atr'].rolling(20).mean() / df['close']) * 100

    # 动态趋势强度
    ma20 = df['close'].ewm(span=20).mean()
    ma60 = df['close'].ewm(span=60).mean()
    df['trend_strength'] = abs(ma20 - ma60) / ma60 * 100
    df['is_bullish'] = ma20 > ma60

    # STC标准算法
    ema_fast = df['close'].ewm(span=23, adjust=False).mean()
    ema_slow = df['close'].ewm(span=50, adjust=False).mean()
    macd = ema_fast - ema_slow

    min_macd = macd.rolling(window=10).min()
    max_macd = macd.rolling(window=10).max()
    stc_k1 = 100 * (macd - min_macd) / (max_macd - min_macd)
    stc_k1 = stc_k1.fillna(50)

    stc_d1 = stc_k1.ewm(span=3, adjust=False).mean()

    min_stc_d1 = stc_d1.rolling(window=10).min()
    max_stc_d1 = stc_d1.rolling(window=10).max()
    stc_k2 = 100 * (stc_d1 - min_stc_d1) / (max_stc_d1 - min_stc_d1)
    stc_k2 = stc_k2.fillna(50)

    stc = stc_k2.ewm(span=3, adjust=False).mean()

    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=14).mean()
    tp_mean_abs_diff = tp - sma
    tp_abs_diff = tp_mean_abs_diff.abs()
    mad = tp_abs_diff.rolling(window=14).mean()
    df['cci'] = (tp - sma) / (0.015 * mad)

    return df


# ========================================================
# 3. 动态回测引擎
# ========================================================
def run_dynamic_backtest(df, symbol):
    df = prepare_data(df).dropna()

    position = 0
    entry_price = 0.0
    entry_mode = ''
    trades = []

    print(f"  [引擎启动] 数据处理完成，开始逐日扫描 {len(df)} 根K线...")

    for i in range(1, len(df)):
        row = df.iloc[i]

        # === 步骤A: 动态识别 ===
        current_mode = 'TRASH'  # 默认休眠

        if row['vol_ratio'] > 2:
            current_mode = 'SNIPER'  # 高波动
        elif row['trend_strength'] > 2 and row['is_bullish']:
            current_mode = 'TREND'  # 趋势
        elif row['trend_strength'] > 2 and not row['is_bullish']:
            current_mode = 'TREND'  # 趋势
        else:
            current_mode = 'TRASH'  # 低波动

        # === 步骤B: 信号执行 ===
        params = STRATEGY_REPO.get(current_mode)
        if not params:
            continue

        # 平仓逻辑
        if position != 0:
            pnl_raw = (row['close'] - entry_price) if position == 1 else (entry_price - row['close'])
            pnl_atr = pnl_raw / row['atr']

            exit = False
            exit_reason = ''

            # 固定止盈
            if params.get('take_profit', 999) > 10:
                if pnl_atr > params['take_profit']:
                    exit = True
                    exit_reason = '止盈'

            # 移动止损
            if not exit and current_mode == 'TREND' and position == 1 and row['close'] < row['ma60']:
                exit = True
                exit_reason = '趋势破'

            # ATR止损
            if not exit:
                if pnl_atr < -params['stop_loss']:
                    exit = True
                    exit_reason = '止损'

            if exit:
                pnl = pnl_raw * 20 * MULTIPLIER
                commission = abs(pnl) * COMMISSION_RATE * 2
                net_pnl = pnl - commission

                balance = balance + net_pnl

                trades.append({
                    'date': df.index[i],
                    'mode': entry_mode,
                    'type': exit_reason,
                    'pnl': net_pnl,
                })

                position = 0
                entry_mode = ''
                continue

        # 开仓逻辑
        if position == 0:
            # 做空
            if current_mode == 'SNIPER':
                params_sni = STRATEGY_REPO['SNIPER']

                # 条件1: STC回落
                cond1 = row['stc'] < params_sni['stc_level']

                # 条件2: CCI跌破
                cond2 = row['cci'] < params_sni['cci_trigger']

                # 条件3: 有量
                cond3 = row['vol_ratio'] > 0.3

                if cond1 and cond2 and cond3:
                    position = -1
                    entry_price = row['close']
                    entry_mode = 'SNIPER'
                    print(f"    [开空] {df.index[i]} 价格:{entry_price:.2f}")

            # 做多
            if current_mode == 'TREND':
                params_trend = STRATEGY_REPO['TREND']

                # 条件1: STC突破 (阳线)
                cond1 = row['stc'] > params_trend['stc_level']

                # 条件2: CCI突破
                cond2 = row['cci'] > params_trend['cci_trigger'] - 10

                # 条件3: 有量
                cond3 = row['vol_ratio'] > 1.0

                # 条件4: 趋势向上
                cond4 = row['is_bullish']

                # 条件5: 均线之上
                cond5 = row['close'] > row['ma60']

                if cond1 and cond2 and cond3 and cond4 and cond5:
                    position = 1
                    entry_price = row['close']
                    entry_mode = 'TREND'
                    print(f"    [开多] {df.index[i]} 价格:{entry_price:.2f}")

    # 计算结果
    total_pnl = sum([t['pnl'] for t in trades])
    total_return = (INITIAL_CAPITAL + total_pnl - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    if len(trades) == 0:
        print(f"    [结果] 无交易")
        return None

    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100

    return {
        'symbol': symbol,
        'mode': entry_mode,
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

    # 获取数据
    try:
        df = pd.read_csv(r'D:\期货数据\铜期货监控\CCI策略系统\sa0_2020-2025.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
    except:
        print(f"    [ERROR] 无法读取数据文件")
        return None

    yearly_results = []

    for year in range(start_year, end_year + 1):
        print(f"\n{'='*80}")
        print(f"正在回测: {year}年")
        print('-'*80)

        start_date = pd.to_datetime(f'{year}-01-01')
        end_date = pd.to_datetime(f'{year}-12-31')

        df_year = df[(df.index >= start_date) & (df.index <= end_date)].copy()

        if len(df_year) < 200:
            print(f"    [SKIP] 数据不足: {len(df_year)}条")
            continue

        result = run_dynamic_backtest(df_year, symbol)

        if result:
            yearly_results.append({
                'year': year,
                'mode': result['mode'],
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
            print(f"  {r['year']:<8} {r['mode']:<12} {r['total_return']:>9.2f}%   {r['total_trades']:>6}笔   {r['win_rate']:>7.1f}%")

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
    print("全自动 S.C.C. 系统 v3.1 (复苏版)".center(100))
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
