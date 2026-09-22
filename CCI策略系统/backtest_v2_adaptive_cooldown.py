# -*- coding: utf-8 -*-
"""
历史数据回测系统 - V2.2 (自适应冷却版)
核心思想：
1. V2 激进开仓 (保留狼性)
2. 只有在"止损"后才触发冷却 (Smart Cooldown)
   - 止盈 -> 立即再战 (保住2024暴跌利润)
   - 止损 -> 强制休息3天 (规避2025震荡磨损)
"""
import pandas as pd
import numpy as np
from datetime import datetime

# ========================================================
# 全局配置
# ========================================================
CONFIG = {
    # 冷却参数 (核心修改)
    'cooldown_days_loss': 3,  # 止损后的惩罚冷却天数
    'cooldown_days_win': 0,   # 止盈后不冷却，乘胜追击

    # V2 核心参数
    'stc_entry': 80,
    'cci_entry': 50,
    'stop_loss_atr': 1.5,
    'take_profit_atr': 3.0,

    # 回测设置
    'multiplier': 20,
    'commission': 0.0001,
    'initial_capital': 100000
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


def calculate_indicators(df):
    """计算所有指标"""
    data = df.copy()

    # 1. TR 和 ATR
    data['tr'] = calculate_tr(data)
    data['atr'] = data['tr'].rolling(14).mean()

    # 2. CCI
    data['cci'] = calculate_cci(data)

    # 3. STC
    data['stc'] = calculate_stc(data)

    # 4. YTD (Year to Date) for filter
    data['year'] = data.index.year
    year_start_prices = data.groupby('year')['close'].transform('first')
    data['ytd_return'] = (data['close'] - year_start_prices) / year_start_prices * 100

    # 5. 其他辅助指标
    data['ma20'] = data['close'].ewm(span=20).mean()
    data['ma60'] = data['close'].ewm(span=60).mean()
    data['vol_ratio'] = (data['atr'].rolling(20).mean() / data['close']).fillna(0) * 100
    data['trend_strength'] = abs(data['close'].ewm(span=20).mean() - data['close'].ewm(span=60).mean()) / data['close'].ewm(span=60).mean().fillna(1) * 100
    data['is_bullish'] = data['ma20'] > data['ma60']

    data['vol_ratio'].fillna(0, inplace=True)
    data['trend_strength'].fillna(0, inplace=True)
    data['is_bullish'].fillna(False, inplace=True)

    return data.dropna()


def backtest_year(df_all, year, symbol):
    """单年份回测 - V2.2自适应冷却"""
    print(f"\n{'='*80}")
    print(f"年份回测: {year}年 (V2.2自适应冷却)")
    print('-'*80)

    start_date = pd.to_datetime(f'{year}-01-01')
    end_date = pd.to_datetime(f'{year}-12-31')

    df_year = df_all[(df_all.index >= start_date) & (df_all.index <= end_date)].copy()

    if len(df_year) < 200:
        print(f"    [SKIP] 数据不足: {len(df_year)}条")
        return None

    # 识别市场状态
    vol = df_year['vol_ratio'].iloc[-1]
    trend = df_year['trend_strength'].iloc[-1]
    is_bullish = df_year['is_bullish'].iloc[-1]
    year_start_price = df_year['close'].iloc[0]

    print(f"      波动率: {vol:.2f}%")
    print(f"      趋势强度: {trend:.2f}%")
    print(f"      年初价格: {year_start_price:.2f}")
    print(f"      年末价格: {df_year['close'].iloc[-1]:.2f}")

    if is_bullish:
        print(f"    [识别] 牛市，SNIPER做空模式禁用")
        print(f"    [识别] 休眠模式")
        return None

    regime = 'SNIPER'
    print(f"      识别为: {regime}模式")
    print(f"      自适应冷却: 止盈不冷却, 止损冷却{CONFIG['cooldown_days_loss']}天")

    # 回测
    balance = CONFIG['initial_capital']
    position = 0
    entry_price = 0.0
    entry_date = None

    # 冷却相关变量
    last_exit_date = None
    last_exit_type = None  # 'Win' 或 'Loss'

    trades = []

    print("    开始回测...")

    for i in range(1, len(df_year)):
        row = df_year.iloc[i]
        current_date = df_year.index[i]

        # 安全检查
        if pd.isna(row['close']) or pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # 平仓逻辑
        if position != 0:
            profit_raw = (row['close'] - entry_price) if position == 1 else (entry_price - row['close'])
            atr_value = row['atr']
            stop_loss_distance = atr_value * CONFIG['stop_loss_atr']
            take_profit_distance = atr_value * CONFIG['take_profit_atr']

            exit = False
            exit_reason = ''

            # 止盈 (Good Exit) - 原始V2逻辑：反向突破保护
            if abs(profit_raw) >= take_profit_distance and profit_raw < 0:
                exit = True
                exit_reason = '反向突破'

            # ATR止损 (Bad Exit)
            if not exit:
                if profit_raw < -stop_loss_distance:
                    exit = True
                    exit_reason = 'ATR止损'

            # 强制止损 (单笔最大亏损3%本金)
            if not exit:
                max_loss = balance * 0.03
                if position == -1:
                    potential_loss = (row['close'] - entry_price) * 20 * CONFIG['multiplier']
                else:
                    potential_loss = (entry_price - row['close']) * 20 * CONFIG['multiplier']

                if potential_loss < -max_loss:
                    exit = True
                    exit_reason = '强制止损'

            if exit:
                pnl = profit_raw * 20 * CONFIG['multiplier']
                commission = abs(pnl) * CONFIG['commission'] * 2
                net_pnl = pnl - commission

                balance = balance + net_pnl

                # 关键修复：按实际盈亏分类，而不是按退出原因
                exit_type_tag = 'Win' if net_pnl > 0 else 'Loss'

                trades.append({
                    'date': current_date,
                    'entry_date': entry_date,
                    'type': exit_reason,
                    'pnl': net_pnl,
                    'exit_type': exit_type_tag,  # 关键：记录实际盈亏类型
                })

                position = 0
                last_exit_date = current_date
                last_exit_type = exit_type_tag  # 关键：记录实际是赚钱还是亏钱

        # 开仓逻辑 (Smart Cooldown)
        if position == 0:
            # 1. 智能冷却检查
            if last_exit_date:
                days_since_exit = (current_date - last_exit_date).days

                # 动态决定冷却天数
                required_cooldown = 0
                if last_exit_type == 'Loss':
                    required_cooldown = CONFIG['cooldown_days_loss']
                elif last_exit_type == 'Win':
                    required_cooldown = CONFIG['cooldown_days_win']

                # 检查是否满足冷却
                if days_since_exit < required_cooldown:
                    continue

            # 2. YTD趋势过滤 (牛市过滤)
            ytd_return = (row['close'] - year_start_price) / year_start_price * 100
            if ytd_return > 10:
                continue

            current_vol = row.get('vol_ratio', 0)
            current_stc = row.get('stc', 0)
            current_cci = row.get('cci', 0)

            # V2 激进开仓条件
            if (current_stc < CONFIG['stc_entry'] and
                current_cci < CONFIG['cci_entry'] and
                current_vol > 0.3 and
                not pd.isna(current_stc) and
                not pd.isna(current_cci)):
                position = -1  # SNIPER做空
                entry_price = row['close']
                entry_date = current_date

    # 计算结果
    total_pnl = sum([t['pnl'] for t in trades])
    total_return = (balance + total_pnl - CONFIG['initial_capital']) / CONFIG['initial_capital'] * 100

    if len(trades) == 0:
        print(f"    [结果] 无交易")
        return None

    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100

    # 统计止盈和止损次数
    win_exits = len([t for t in trades if t['exit_type'] == 'Win'])
    loss_exits = len([t for t in trades if t['exit_type'] == 'Loss'])

    print(f"    [结果] 交易:{len(trades)}笔 | 止盈:{win_exits}笔 | 止损:{loss_exits}笔")
    print(f"    [结果] 收益:{total_return:.2f}% | 胜率:{win_rate:.1f}%")

    return {
        'year': year,
        'regime': regime,
        'total_return': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'win_exits': win_exits,
        'loss_exits': loss_exits,
    }


def backtest_by_year(symbol, start_year, end_year):
    """按年份回测"""
    print(f"\n{'='*100}")
    print(f"年份回测: {symbol} ({start_year}-{end_year}) - V2.2自适应冷却版")
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
        df_all = calculate_indicators(df_all)
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
                'win_exits': result['win_exits'],
                'loss_exits': result['loss_exits'],
            })

            print(f"    [结果] 交易:{result['total_trades']}笔 | 收益:{result['total_return']:.2f}% | 胜率:{result['win_rate']:.1f}%")

    # 输出总结
    print(f"\n{'='*100}")
    print(f"年份回测总结: {symbol} ({start_year}-{end_year}) - V2.2自适应冷却版")
    print('='*100)

    if yearly_results:
        results_df = pd.DataFrame(yearly_results)

        print(f"\n {'年份':<8} {'模式':<12} {'收益率':>10} {'交易数':>8} {'胜率':>8} {'止盈':>6} {'止损':>6}")
        print("-" * 100)

        for _, r in results_df.iterrows():
            print(f"  {r['year']:<8} {r['regime']:<12} {r['total_return']:>9.2f}%   {r['total_trades']:>6}笔   {r['win_rate']:>7.1f}%   {r['win_exits']:>4}笔  {r['loss_exits']:>4}笔")

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
    print("历史数据回测系统 - V2.2 (自适应冷却版)")
    print("=" * 100)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print("\n核心机制: Smart Cooldown (自适应冷却)")
    print("  - 止盈 (Win) -> 冷却0天 (乘胜追击)")
    print("  - 止损 (Loss) -> 冷却3天 (惩罚性休息)")
    print("  - 保留V2的狼性，增加V3的防弹衣")
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
    print("V2.2自适应冷却版 - 最终表现".center(100))
    print("=" * 100)

    for s, results_df in all_results.items():
        if results_df is None or len(results_df) == 0:
            continue
        print(f"\n{s} ({start_year}-{end_year})")
        print("-" * 60)

        print(f" 年份   收益率    交易数  胜率  止盈  止损")
        print("-" * 60)

        for _, r in results_df.iterrows():
            trend_arrow = "↑" if r['total_return'] > 0 else "↓"
            print(f"  {r['year']:<8} {r['total_return']:>9.2f}%   {r['total_trades']:>6}笔   {r['win_rate']:>6.1f}%   {r['win_exits']:>4}笔  {r['loss_exits']:>4}笔")

    print(f"\n{'='*100}")
    print(f"系统运行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
