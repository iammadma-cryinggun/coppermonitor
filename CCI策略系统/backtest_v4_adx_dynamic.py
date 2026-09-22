# -*- coding: utf-8 -*-
"""
冰火双刀流 - V4.0 (ADX 动态红绿灯版)
核心逻辑：
不再是 50/50 配比，而是基于 ADX 指标实时切换 V2.1 和 V2.2
- ADX > 25 (趋势强) -> 切换为 V2.2 (赢了不冷，输了冷3天)
- ADX < 25 (震荡市) -> 切换为 V2.1 (无论盈亏，强制冷3天)
"""
import pandas as pd
import numpy as np
from datetime import datetime

# ========================================================
# 1. 全局配置
# ========================================================
CONFIG = {
    'symbol': 'sa0',
    'initial_capital': 100000,
    'multiplier': 20,

    # 核心阈值
    'adx_threshold': 25,  # 【关键】趋势/震荡的分界线

    # 策略参数
    'stc_entry': 80,
    'cci_entry': 50,
    'stop_loss_atr': 1.5,
    'take_profit_atr': 3.0,
    'vol_filter': 0.3,

    # 冷却参数
    'cooldown_defensive': 3,      # 防守模式强制冷却
    'cooldown_aggressive_loss': 3, # 进攻模式输了冷却
    'cooldown_aggressive_win': 0,   # 进攻模式赢了不冷却
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


def calculate_adx(df, period=14):
    """计算 ADX (Average Directional Index) - 趋势强度指标"""
    data = df.copy()

    # 计算方向移动
    data['up_move'] = data['high'] - data['high'].shift(1)
    data['down_move'] = data['low'].shift(1) - data['low']

    # 计算 +DM 和 -DM
    data['plus_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0),
                                data['up_move'], 0)
    data['minus_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0),
                                 data['down_move'], 0)

    # 平滑
    data['atr_smooth'] = data['tr'].ewm(alpha=1/period, adjust=False).mean()
    data['plus_di'] = 100 * data['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / data['atr_smooth']
    data['minus_di'] = 100 * data['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / data['atr_smooth']

    # 计算 DX
    data['dx'] = 100 * np.abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
    data['dx'] = data['dx'].fillna(0)

    # 计算 ADX
    data['adx'] = data['dx'].ewm(alpha=1/period, adjust=False).mean()
    data['adx'] = data['adx'].fillna(0)

    return data['adx']


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

    # 4. ADX (核心新增)
    data['adx'] = calculate_adx(data)

    # 5. YTD (Year to Date) for filter
    data['year'] = data.index.year
    year_start_prices = data.groupby('year')['close'].transform('first')
    data['ytd_return'] = (data['close'] - year_start_prices) / year_start_prices * 100

    # 6. 其他辅助指标
    data['ma20'] = data['close'].ewm(span=20).mean()
    data['ma60'] = data['close'].ewm(span=60).mean()
    data['vol_ratio'] = (data['atr'].rolling(20).mean() / data['close']).fillna(0) * 100
    data['trend_strength'] = abs(data['close'].ewm(span=20).mean() - data['close'].ewm(span=60).mean()) / data['close'].ewm(span=60).mean().fillna(1) * 100
    data['is_bullish'] = data['ma20'] > data['ma60']

    data['vol_ratio'].fillna(0, inplace=True)
    data['trend_strength'].fillna(0, inplace=True)
    data['is_bullish'].fillna(False, inplace=True)

    return data.dropna()


def backtest_year_v4(df_all, year, symbol):
    """单年份回测 - V4.0 ADX动态切换版"""
    print(f"\n{'='*80}")
    print(f"年份回测: {year}年 (V4.0 ADX动态切换版)")
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

    # ADX 统计
    avg_adx = df_year['adx'].mean()
    high_adx_days = len(df_year[df_year['adx'] > CONFIG['adx_threshold']])

    print(f"      波动率: {vol:.2f}%")
    print(f"      趋势强度: {trend:.2f}%")
    print(f"      年初价格: {year_start_price:.2f}")
    print(f"      年末价格: {df_year['close'].iloc[-1]:.2f}")
    print(f"      平均ADX: {avg_adx:.2f} (阈值:{CONFIG['adx_threshold']})")
    print(f"      高ADX天数: {high_adx_days}天 ({high_adx_days/len(df_year)*100:.1f}%)")

    if is_bullish:
        print(f"    [识别] 牛市，SNIPER做空模式禁用")
        print(f"    [识别] 休眠模式")
        return None

    regime = 'SNIPER'
    print(f"      识别为: {regime}模式")
    print(f"      动态切换: ADX>{CONFIG['adx_threshold']}用进攻(V2.2), ADX<{CONFIG['adx_threshold']}用防守(V2.1)")

    # 回测
    balance = CONFIG['initial_capital']
    position = 0
    entry_price = 0.0
    entry_date = None
    last_exit_date = None
    last_exit_type = None

    trades = []

    # 状态监控
    regime_history = []  # 记录每天判断出的市场状态

    print("    开始回测...")

    for i in range(1, len(df_year)):
        row = df_year.iloc[i]
        prev = df_year.iloc[i-1]
        current_date = df_year.index[i]

        # 安全检查
        if pd.isna(row['close']) or pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # ==================== 1. 市场状态判断 (Regime Check) ====================
        # 使用前一天的 ADX，避免 look-ahead bias
        current_adx = prev['adx']

        # 判断当前模式
        if current_adx > CONFIG['adx_threshold']:
            current_mode = 'Aggressive'  # 趋势模式 (V2.2)
        else:
            current_mode = 'Defensive'  # 震荡模式 (V2.1)

        regime_history.append({
            'date': current_date,
            'adx': current_adx,
            'mode': current_mode
        })

        # ==================== 2. 平仓逻辑 (通用) ====================
        if position != 0:
            profit_raw = (row['close'] - entry_price) if position == 1 else (entry_price - row['close'])
            atr_value = row['atr']
            stop_loss_distance = atr_value * CONFIG['stop_loss_atr']
            take_profit_distance = atr_value * CONFIG['take_profit_atr']

            exit = False
            exit_reason = ''

            # ATR止损 (Bad Exit)
            if profit_raw < -stop_loss_distance:
                exit = True
                exit_reason = '止损'

            # 强制止损 (单笔最大亏损3%本金)
            if not exit:
                max_loss = balance * 0.03
                if position == -1:
                    potential_loss = (row['close'] - entry_price) * CONFIG['multiplier']
                else:
                    potential_loss = (entry_price - row['close']) * CONFIG['multiplier']

                if potential_loss < -max_loss:
                    exit = True
                    exit_reason = '强制止损'

            if exit:
                real_pnl = profit_raw * CONFIG['multiplier']
                commission = abs(real_pnl) * 0.0001 * 2
                net_pnl = real_pnl - commission

                balance = balance + net_pnl

                trades.append({
                    'date': current_date,
                    'entry_date': entry_date,
                    'type': exit_reason,
                    'pnl': net_pnl,
                    'mode_at_exit': current_mode,  # 记录平仓时的市场模式
                })

                position = 0
                last_exit_date = current_date
                last_exit_type = 'Loss' if exit_reason == '止损' else 'Win'
                continue

        # ==================== 3. 开仓逻辑 (基于模式动态调整冷却) ====================
        if position == 0:
            # A. 计算所需的冷却天数
            required_cooldown = 999

            if current_mode == 'Defensive':
                # 防守模式：无论上次盈亏，统统冷3天
                required_cooldown = CONFIG['cooldown_defensive']
            else:
                # 进攻模式：赢了冷0天，输了冷3天
                if last_exit_type == 'Loss':
                    required_cooldown = CONFIG['cooldown_aggressive_loss']
                else:
                    required_cooldown = CONFIG['cooldown_aggressive_win']

            # B. 检查冷却
            if last_exit_date:
                days_since = (current_date - last_exit_date).days
                if days_since < required_cooldown:
                    continue

            # C. 信号检查 (V2 核心信号)
            ytd_return = (row['close'] - year_start_price) / year_start_price * 100
            is_bull = ytd_return > 10
            if is_bull:
                continue

            current_vol = row.get('vol_ratio', 0)
            current_stc = row.get('stc', 0)
            current_cci = row.get('cci', 0)

            cond_stc = (current_stc < CONFIG['stc_entry']) and (current_stc > 20)
            cond_cci = current_cci < CONFIG['cci_entry']

            if cond_stc and cond_cci and current_vol > CONFIG['vol_filter']:
                position = -1
                entry_price = row['close']
                entry_date = current_date

    # 计算结果
    total_pnl = sum([t['pnl'] for t in trades])
    total_return = total_pnl / CONFIG['initial_capital'] * 100

    if len(trades) == 0:
        print(f"    [结果] 无交易")
        return None

    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100

    # 统计不同模式的交易数
    mode_counts = {}
    for t in trades:
        mode = t['mode_at_exit']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    mode_str = ", ".join([f"{k}:{v}" for k, v in mode_counts.items()])

    print(f"    [结果] 交易:{len(trades)}笔 | 收益:{total_return:.2f}% | 胜率:{win_rate:.1f}%")
    print(f"           模式分布: {mode_str}")

    return {
        'year': year,
        'regime': regime,
        'total_return': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_adx': avg_adx,
        'high_adx_ratio': high_adx_days / len(df_year) * 100,
    }


def backtest_by_year(symbol, start_year, end_year):
    """按年份回测"""
    print(f"\n{'='*100}")
    print(f"年份回测: {symbol} ({start_year}-{end_year}) - V4.0 ADX动态切换版")
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
        result = backtest_year_v4(df_all, year, symbol)

        if result:
            yearly_results.append(result)

    # 输出总结
    print(f"\n{'='*100}")
    print(f"年份回测总结: {symbol} ({start_year}-{end_year}) - V4.0 ADX动态切换版")
    print('='*100)

    if yearly_results:
        results_df = pd.DataFrame(yearly_results)

        print(f"\n {'年份':<8} {'收益率':>10} {'交易数':>8} {'胜率':>8} {'平均ADX':>10} {'高ADX比例':>12}")
        print("-" * 100)

        for _, r in results_df.iterrows():
            print(f"  {r['year']:<8} {r['total_return']:>9.2f}%   {r['total_trades']:>6}笔   {r['win_rate']:>7.1f}%   {r['avg_adx']:>8.2f}     {r['high_adx_ratio']:>8.1f}%")

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
    print("冰火双刀流 - V4.0 (ADX 动态红绿灯版)")
    print("=" * 100)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print("\n核心机制: ADX 动态切换 (实时判断市场状态)")
    print("  - ADX > 25 (趋势强) -> 切换为 V2.2 (进攻)")
    print("  - ADX < 25 (震荡市) -> 切换为 V2.1 (防守)")
    print("  - 目标: 智能判断，而不是傻瓜平摊")
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
    print("V4.0 ADX动态切换版 - 最终表现".center(100))
    print("=" * 100)

    for s, results_df in all_results.items():
        if results_df is None or len(results_df) == 0:
            continue
        print(f"\n{s} ({start_year}-{end_year})")
        print("-" * 80)

        print(f" 年份   收益率    交易数  胜率  平均ADX  高ADX比例")
        print("-" * 80)

        for _, r in results_df.iterrows():
            trend_arrow = "↑" if r['total_return'] > 0 else "↓"
            print(f"  {r['year']:<6} {r['total_return']:>10.2f}%   {r['total_trades']:>6}笔  {r['win_rate']:>6.1f}%  {r['avg_adx']:>8.2f}   {r['high_adx_ratio']:>8.1f}%")

    print(f"\n{'='*100}")
    print(f"系统运行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
