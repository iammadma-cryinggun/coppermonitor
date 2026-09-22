# -*- coding: utf-8 -*-
"""
冰火双刀流 - 终极策略 V3.0 (Composite Strategy - Complete)
核心思想：资产配置理论 (Portfolio Theory)
机制：
1. 激进轨(V2.2): 止盈冷0天，止损冷3天 → 捕捉快速下跌暴利
2. 保守轨(V2.1): 无论盈亏，强制冷3天 → 应对缓慢震荡保护
3. 资金管理: 50/50配比 + 杠杆乘数修正(20元/点)
结果：平滑资金曲线，穿越牛熊
"""
import pandas as pd
import numpy as np
from datetime import datetime

# ========================================================
# 1. 全局配置 (实盘参数)
# ========================================================
CONFIG = {
    'symbol': 'sa0',
    'initial_capital': 100000,
    'multiplier': 20,          # 【修正】纯碱合约乘数 20元/点

    # 核心参数
    'stc_entry': 80,
    'cci_entry': 50,
    'vol_filter': 0.3,
    'stop_loss_atr': 1.5,
    'take_profit_atr': 3.0,

    # 资金分配
    'allocation_ratio': 0.5,   # 各占50%

    # 策略 A (激进)
    'agg_cooldown_loss': 3,
    'agg_cooldown_win': 0,

    # 策略 B (保守)
    'def_cooldown': 3
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


def backtest_year_composite(df_all, year, symbol):
    """单年份回测 - 冰火双刀流 V3.0 (带杠杆修正)"""
    print(f"\n{'='*80}")
    print(f"年份回测: {year}年 (冰火双刀流 V3.0 - 杠杆修正版)")
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
    print(f"      资金配置: 50%激进轨(V2.2) + 50%保守轨(V2.1)")
    print(f"      杠杆修正: {CONFIG['multiplier']}元/点")

    # 初始化两轨的资金
    balance_agg = CONFIG['initial_capital'] * CONFIG['allocation_ratio']  # 激进轨 50%
    balance_def = CONFIG['initial_capital'] * CONFIG['allocation_ratio']  # 保守轨 50%

    # 激进轨状态
    pos_agg = 0
    entry_price_agg = 0.0
    entry_date_agg = None
    last_exit_date_agg = None
    last_exit_type_agg = None

    # 保守轨状态
    pos_def = 0
    entry_price_def = 0.0
    entry_date_def = None
    last_exit_date_def = None

    trades_agg = []
    trades_def = []

    print("    开始回测...")

    for i in range(1, len(df_year)):
        row = df_year.iloc[i]
        current_date = df_year.index[i]

        # 安全检查
        if pd.isna(row['close']) or pd.isna(row['atr']) or row['atr'] <= 0:
            continue

        # ==================== 公共信号计算 ====================
        # YTD趋势（牛市过滤）
        ytd_return = (row['close'] - year_start_price) / year_start_price * 100
        if ytd_return > 10:
            continue

        current_vol = row.get('vol_ratio', 0)
        current_stc = row.get('stc', 0)
        current_cci = row.get('cci', 0)

        raw_signal = (current_stc < CONFIG['stc_entry'] and
                     current_cci < CONFIG['cci_entry'] and
                     current_vol > CONFIG['vol_filter'] and
                     not pd.isna(current_stc) and
                     not pd.isna(current_cci))

        # ==================== 激进轨 A: V2.2 (自适应冷却) ====================
        # 平仓 A
        if pos_agg != 0:
            profit_raw = (row['close'] - entry_price_agg) if pos_agg == 1 else (entry_price_agg - row['close'])
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
                max_loss = balance_agg * 0.03
                if pos_agg == -1:
                    potential_loss = (row['close'] - entry_price_agg) * CONFIG['multiplier']
                else:
                    potential_loss = (entry_price_agg - row['close']) * CONFIG['multiplier']

                if potential_loss < -max_loss:
                    exit = True
                    exit_reason = '强制止损'

            if exit:
                # 【修正】计算真实金额：点数 * 杠杆倍数 * 50%仓位
                real_pnl = profit_raw * CONFIG['multiplier'] * CONFIG['allocation_ratio']
                commission = abs(real_pnl) * 0.0001 * 2
                net_pnl = real_pnl - commission

                balance_agg = balance_agg + net_pnl

                # 关键：按实际盈亏分类
                exit_type_tag = 'Win' if net_pnl > 0 else 'Loss'

                trades_agg.append({
                    'date': current_date,
                    'entry_date': entry_date_agg,
                    'type': exit_reason,
                    'pnl': net_pnl,
                    'exit_type': exit_type_tag,
                })

                pos_agg = 0
                last_exit_date_agg = current_date
                last_exit_type_agg = exit_type_tag

        # 开仓 A
        if pos_agg == 0 and raw_signal:
            # 自适应冷却检查
            if last_exit_date_agg:
                days_since = (current_date - last_exit_date_agg).days
                required_cooldown = CONFIG['agg_cooldown_loss'] if last_exit_type_agg == 'Loss' else CONFIG['agg_cooldown_win']

                if days_since < required_cooldown:
                    pass  # 还在冷却期
                else:
                    pos_agg = -1
                    entry_price_agg = row['close']
                    entry_date_agg = current_date
            else:
                pos_agg = -1
                entry_price_agg = row['close']
                entry_date_agg = current_date

        # ==================== 保守轨 B: V2.1 (固定冷却) ====================
        # 平仓 B
        if pos_def != 0:
            profit_raw = (row['close'] - entry_price_def) if pos_def == 1 else (entry_price_def - row['close'])
            atr_value = row['atr']
            stop_loss_distance = atr_value * CONFIG['stop_loss_atr']
            take_profit_distance = atr_value * CONFIG['take_profit_atr']

            exit = False
            exit_reason = ''

            # ATR止损
            if profit_raw < -stop_loss_distance:
                exit = True
                exit_reason = '止损'

            # 强制止损
            if not exit:
                max_loss = balance_def * 0.03
                if pos_def == -1:
                    potential_loss = (row['close'] - entry_price_def) * CONFIG['multiplier']
                else:
                    potential_loss = (entry_price_def - row['close']) * CONFIG['multiplier']

                if potential_loss < -max_loss:
                    exit = True
                    exit_reason = '强制止损'

            if exit:
                # 【修正】计算真实金额
                real_pnl = profit_raw * CONFIG['multiplier'] * CONFIG['allocation_ratio']
                commission = abs(real_pnl) * 0.0001 * 2
                net_pnl = real_pnl - commission

                balance_def = balance_def + net_pnl

                trades_def.append({
                    'date': current_date,
                    'entry_date': entry_date_def,
                    'type': exit_reason,
                    'pnl': net_pnl,
                })

                pos_def = 0
                last_exit_date_def = current_date

        # 开仓 B
        if pos_def == 0 and raw_signal:
            # 固定3天冷却检查
            if last_exit_date_def:
                days_since = (current_date - last_exit_date_def).days

                if days_since >= CONFIG['def_cooldown']:
                    pos_def = -1
                    entry_price_def = row['close']
                    entry_date_def = current_date
            else:
                pos_def = -1
                entry_price_def = row['close']
                entry_date_def = current_date

    # 计算结果
    total_pnl_agg = sum([t['pnl'] for t in trades_agg])
    total_pnl_def = sum([t['pnl'] for t in trades_def])

    return_agg = total_pnl_agg / (CONFIG['initial_capital'] * CONFIG['allocation_ratio']) * 100
    return_def = total_pnl_def / (CONFIG['initial_capital'] * CONFIG['allocation_ratio']) * 100

    # 复合收益率 (两轨加权平均)
    total_return = (return_agg + return_def) / 2

    if len(trades_agg) == 0 and len(trades_def) == 0:
        print(f"    [结果] 无交易")
        return None

    win_trades_agg = [t for t in trades_agg if t['pnl'] > 0]
    win_trades_def = [t for t in trades_def if t['pnl'] > 0]

    win_rate_agg = len(win_trades_agg) / len(trades_agg) if len(trades_agg) > 0 else 0
    win_rate_def = len(win_trades_def) / len(trades_def) if len(trades_def) > 0 else 0

    # 统计止盈和止损次数（激进轨）
    win_exits_agg = len([t for t in trades_agg if t.get('exit_type') == 'Win'])
    loss_exits_agg = len([t for t in trades_agg if t.get('exit_type') == 'Loss'])

    print(f"    [结果] 激进轨: {len(trades_agg)}笔 | 收益:{return_agg:.2f}% | 胜率:{win_rate_agg:.1f}%")
    print(f"           保守轨: {len(trades_def)}笔 | 收益:{return_def:.2f}% | 胜率:{win_rate_def:.1f}%")
    print(f"           复合收益: {total_return:.2f}%")

    return {
        'year': year,
        'regime': regime,
        'total_return': total_return,
        'agg_return': return_agg,
        'def_return': return_def,
        'total_trades': len(trades_agg) + len(trades_def),
        'agg_trades': len(trades_agg),
        'def_trades': len(trades_def),
        'win_rate': (win_rate_agg + win_rate_def) / 2,
        'win_exits_agg': win_exits_agg,
        'loss_exits_agg': loss_exits_agg,
    }


def backtest_by_year(symbol, start_year, end_year):
    """按年份回测"""
    print(f"\n{'='*100}")
    print(f"年份回测: {symbol} ({start_year}-{end_year}) - 冰火双刀流 V3.0")
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
        result = backtest_year_composite(df_all, year, symbol)

        if result:
            yearly_results.append(result)

    # 输出总结
    print(f"\n{'='*100}")
    print(f"年份回测总结: {symbol} ({start_year}-{end_year}) - 冰火双刀流 V3.0")
    print('='*100)

    if yearly_results:
        results_df = pd.DataFrame(yearly_results)

        print(f"\n {'年份':<8} {'复合收益':>12} {'激进轨':>10} {'保守轨':>10} {'总交易':>8} {'胜率':>8} {'止盈':>6} {'止损':>6}")
        print("-" * 100)

        for _, r in results_df.iterrows():
            print(f"  {r['year']:<8} {r['total_return']:>11.2f}%   {r['agg_return']:>9.2f}%  {r['def_return']:>9.2f}%  {r['total_trades']:>6}笔   {r['win_rate']:>7.1f}%   {r['win_exits_agg']:>4}笔  {r['loss_exits_agg']:>4}笔")

        # 统计
        avg_return = results_df['total_return'].mean()
        std_return = results_df['total_return'].std()
        up_years = len(results_df[results_df['total_return'] > 0])
        total_years = len(results_df)

        print(f"\n 平均复合收益率: {avg_return:.2f}%")
        print(f" 收益标准差: {std_return:.2f}%")
        print(f" 盈利年份比例: {up_years}/{total_years} ({up_years/total_years*100:.1f}%)")

    return results_df


def main():
    print("=" * 100)
    print("冰火双刀流 - 终极策略 V3.0 (Composite Strategy - Complete)")
    print("=" * 100)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print("\n核心机制: 资产配置理论 (Portfolio Theory)")
    print("  - 激进轨(V2.2自适应冷却): 止盈不冷却，乘胜追击")
    print("  - 保守轨(V2.1固定冷却): 强制休息3天，规避噪音")
    print("  - 资金分配: 50% + 50%")
    print("  - 杠杆修正: 20元/点 (实盘标准)")
    print("  - 目标: 平滑资金曲线，穿越牛熊")
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
    print("冰火双刀流 V3.0 - 最终表现".center(100))
    print("=" * 100)

    for s, results_df in all_results.items():
        if results_df is None or len(results_df) == 0:
            continue
        print(f"\n{s} ({start_year}-{end_year})")
        print("-" * 80)

        print(f" 年份   复合收益    激进轨   保守轨   总交易  胜率  止盈  止损")
        print("-" * 80)

        for _, r in results_df.iterrows():
            trend_arrow = "↑" if r['total_return'] > 0 else "↓"
            print(f"  {r['year']:<6} {r['total_return']:>10.2f}%   {r['agg_return']:>8.2f}%  {r['def_return']:>8.2f}%  {r['total_trades']:>6}笔  {r['win_rate']:>6.1f}%   {r['win_exits_agg']:>4}笔  {r['loss_exits_agg']:>4}笔")

    print(f"\n{'='*100}")
    print(f"系统运行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
