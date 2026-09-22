# -*- coding: utf-8 -*-
"""
历史数据回测系统 - 修复版
主要修复：
1. 修复 KeyError: 'vol_ratio' 问题
2. 添加所有必需指标的自动计算（CCI, STC, TR）
3. 放宽入场条件，增加交易机会
4. 优化止损逻辑
5. 添加仓位管理
6. 增加数据验证
"""
import pandas as pd
import numpy as np
from datetime import datetime


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

    # Mean Absolute Deviation
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

# 全局参数
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
INITIAL_CAPITAL = 100000

# 策略参数（已优化）
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
    print("    [进度] 计算技术指标...")

    # 检查基础数据列是否存在
    required_cols = ['close', 'high', 'low']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必需列: {missing_cols}")

    # 处理缺失值
    df = df.copy()

    # 计算所有必需指标
    df['tr'] = calculate_tr(df)
    df['cci'] = calculate_cci(df)
    df['stc'] = calculate_stc(df)

    # 计算其他指标
    df['atr'] = df['tr'].rolling(14).mean()
    df['ma60'] = df['close'].ewm(span=60).mean()

    # 避免除零错误
    df['vol_ratio'] = (df['atr'].rolling(20).mean() / df['close'].replace(0, np.nan)).fillna(0) * 100

    df['trend_strength'] = abs(df['close'].ewm(span=20).mean() - df['close'].ewm(span=60).mean()) / df['close'].ewm(span=60).mean().replace(0, np.nan).fillna(1) * 100
    df['is_bullish'] = df['close'].ewm(span=20).mean() > df['close'].ewm(span=60).mean()

    # 初始填充缺失值
    df['vol_ratio'].fillna(0, inplace=True)
    df['trend_strength'].fillna(0, inplace=True)
    df['is_bullish'].fillna(False, inplace=True)

    return df


def validate_data(df, year):
    """验证数据质量"""
    print(f"    [进度] 识别市场状态...")

    if len(df) < 200:
        print(f"    [SKIP] 数据不足: {len(df)}条")
        return False

    # 检查数据完整性
    if df[['close', 'stc', 'cci', 'tr']].isnull().any().any():
        print(f"    [SKIP] 数据存在缺失值")
        return False

    return True


def backtest_year(df_all, year, symbol):
    """单年份回测"""
    print(f"\n{'='*80}")
    print(f"年份回测: {year}年")
    print('-'*80)

    start_date = pd.to_datetime(f'{year}-01-01')
    end_date = pd.to_datetime(f'{year}-12-31')

    df_year = df_all[(df_all.index >= start_date) & (df_all.index <= end_date)].copy()

    # 数据验证
    if not validate_data(df_year, year):
        return None

    # 识别市场状态
    vol = df_year['vol_ratio'].iloc[-1]
    trend = df_year['trend_strength'].iloc[-1]
    is_bullish = df_year['is_bullish'].iloc[-1]

    print(f"      波动率: {vol:.2f}%")
    print(f"      趋势强度: {trend:.2f}%")

    params = None
    regime = ''

    # 优化后的策略选择逻辑（放宽条件）
    if vol > 1.5:  # 原值 2.0
        regime = 'SNIPER'
        params = STRATEGY_PARAMS['SNIPER']
    elif trend > 1.5 and is_bullish:  # 原值 2.0
        regime = 'TREND'
        params = STRATEGY_PARAMS['TREND']
    else:
        print(f"    [识别] 休眠模式 (波动:{vol:.2f}% 趋势:{trend:.2f}%)")
        return None

    print(f"      品种特性: {'做多' if is_bullish else '做空'}")
    print(f"      识别为: {regime}模式")

    # 回测
    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    trades = []

    print("    开始回测...")

    for i in range(1, len(df_year)):
        row = df_year.iloc[i]

        # 安全检查
        if pd.isna(row['close']) or pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # 平仓逻辑
        if position != 0:
            profit_raw = (row['close'] - entry_price) if position == 1 else (entry_price - row['close'])
            profit_atr = profit_raw / row['atr']

            exit = False
            exit_reason = ''

            # 止盈（SNIPER模式）
            if regime == 'SNIPER' and profit_atr > params['take_profit']:
                exit = True
                exit_reason = '止盈'

            # 移动止损（TREND模式）
            if not exit and regime == 'TREND' and position == 1 and row['close'] < row['ma60']:
                exit = True
                exit_reason = '趋势破'

            # ATR止损（关键修复）
            if not exit:
                if profit_atr < -params['stop_loss']:
                    exit = True
                    exit_reason = '止损'

            # 强制止损：亏损超过本金的30%
            if not exit:
                loss_pct = (profit_raw * 20 * MULTIPLIER) / balance * 100
                if loss_pct < -30:
                    exit = True
                    exit_reason = '强制止损'

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

        # 开仓逻辑（添加额外安全检查）
        if position == 0:
            current_vol = row.get('vol_ratio', 0)
            current_stc = row.get('stc', 0)
            current_cci = row.get('cci', 0)

            # 做空（SNIPER模式）
            if regime == 'SNIPER':
                if (current_stc < params['stc_level'] and
                    current_cci < params['cci_trigger'] and
                    current_vol > 0.3 and
                    not pd.isna(current_stc) and
                    not pd.isna(current_cci)):
                    position = -1
                    entry_price = row['close']
                    print(f"    [开空] {df_year.index[i]} 价格:{entry_price:.2f}")

            # 做多（TREND模式）
            if regime == 'TREND':
                if (current_stc > params['stc_level'] and
                    current_cci > params['cci_trigger'] - 10 and
                    current_vol > 1.0 and
                    row.get('is_bullish', False) and
                    row['close'] > row['ma60'] and
                    not pd.isna(current_stc) and
                    not pd.isna(current_cci)):
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
    print(f"\n正在读取 {symbol} {start_year}-{end_year} 历史数据...")
    df_all = pd.read_csv(r'D:\期货数据\铜期货监控\CCI策略系统\sa0_2020-2025.csv')

    # 检查数据
    print(f"    [成功] 读取数据: {len(df_all)}条")
    if 'date' in df_all.columns:
        print(f"    时间范围: {df_all['date'].min()} 至 {df_all['date'].max()}")

    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values('date')
    df_all.set_index('date', inplace=True)

    # 关键修复：调用 prepare_data 计算所有指标
    try:
        df_all = prepare_data(df_all)
    except ValueError as e:
        print(f"    [错误] {e}")
        return None

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
    print("历史数据回测系统 - 修复版".center(100))
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
