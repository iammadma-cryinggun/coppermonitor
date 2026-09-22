# -*- coding: utf-8 -*-
"""
冰火双刀流 - V5.0 (Smart Trader - 智能交易逻辑版)
核心升级：
1. 移除固定止盈，改为 ATR 移动止损 (让利润奔跑)
2. 引入保本机制 (Breakeven)，浮盈 > 1 ATR 后触发
3. 动态风控：只有真正趋势出来时才贪婪
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

    # 信号参数 (保持 V3 不变)
    'stc_entry': 80,
    'cci_entry': 50,
    'vol_filter': 0.3,

    # --- 交易逻辑参数 (核心升级) ---
    'initial_stop_atr': 1.5,   # 初始硬止损
    'breakeven_trigger': 1.0,  # 浮盈多少ATR后，移动止损到成本价
    'trailing_stop_atr': 2.5,  # 移动止损回撤幅度 (吊灯止损)

    # 资金分配
    'allocation_ratio': 0.5,

    # 冷却参数
    'agg_cooldown_loss': 3,
    'agg_cooldown_win': 0,
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


# ========================================================
# 3. 智能交易管理器 (Trade Manager)
# ========================================================
class TradeManager:
    """
    智能交易管理器
    核心功能：
    1. 保本逻辑 (Breakeven)
    2. 移动止损 (Trailing Stop)
    3. 动态风控
    """
    def __init__(self, entry_price, entry_atr, entry_date, strategy_type):
        self.entry_price = entry_price
        self.entry_atr = entry_atr
        self.entry_date = entry_date
        self.strategy_type = strategy_type

        # 初始止损 (Hard Stop)
        # 做空：止损价 = 入场价 + 1.5 ATR
        self.current_stop = entry_price + (entry_atr * CONFIG['initial_stop_atr'])

        self.highest_profit_atr = -999  # 记录最大浮盈
        self.lowest_price = entry_price  # 记录最低价格(对于做空)
        self.status = 'Open'
        self.exit_reason = ''
        self.breakeven_triggered = False

    def update(self, row):
        """
        更新交易状态
        返回：True = 触发平仓，False = 继续持有
        """
        current_price = row['close']
        current_atr = row['atr']

        # 1. 计算当前浮盈 (ATR倍数)
        # 做空：(入场 - 当前) / ATR
        pnl_atr = (self.entry_price - current_price) / self.entry_atr

        # 更新最大浮盈
        if pnl_atr > self.highest_profit_atr:
            self.highest_profit_atr = pnl_atr

        # 更新最低价（做空时价格越低越有利）
        if current_price < self.lowest_price:
            self.lowest_price = current_price

        # 2. 检查是否触发止损/止盈退出
        if current_price >= self.current_stop:
            self.status = 'Closed'
            self.exit_reason = '移动止损'
            return True  # Signal to exit

        if row['cci'] > 100:  # CCI 强制反转信号
            self.status = 'Closed'
            self.exit_reason = 'CCI反转'
            return True

        # 3. === 核心交易逻辑升级 ===

        # A. 保本逻辑 (Breakeven)
        # 如果浮盈超过 1.0 ATR，且止损还在入场价上方，强行把止损拉到入场价
        if pnl_atr > CONFIG['breakeven_trigger']:
            if self.current_stop > self.entry_price:
                self.current_stop = self.entry_price
                self.breakeven_triggered = True

        # B. 移动止损 (Trailing Stop / Chandelier Exit)
        # 吊灯止损：止损线 = 历史最低价 + 2.5 ATR
        # 随着价格下跌，最低价更新，止损线不断下移。价格反弹如果超过 2.5 ATR 则出场
        trailing_level = self.lowest_price + (current_atr * CONFIG['trailing_stop_atr'])

        # 止损线只能下移(做空时)，不能上移
        if trailing_level < self.current_stop:
            self.current_stop = trailing_level

        return False  # Hold


def backtest_year_v5(df_all, year, symbol):
    """单年份回测 - V5.0 智能交易版"""
    print(f"\n{'='*80}")
    print(f"年份回测: {year}年 (V5.0 智能交易逻辑版)")
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
    print(f"      智能交易: 移动止损({CONFIG['trailing_stop_atr']}ATR) + 保本({CONFIG['breakeven_trigger']}ATR)")

    # 回测
    balance_agg = CONFIG['initial_capital'] * CONFIG['allocation_ratio']  # 激进轨 50%
    balance_def = CONFIG['initial_capital'] * CONFIG['allocation_ratio']  # 保守轨 50%

    active_trade_agg = None
    active_trade_def = None

    last_exit_date_agg = None
    last_exit_type_agg = None
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
        # 平仓 A (使用智能交易管理器)
        if active_trade_agg:
            should_exit = active_trade_agg.update(row)

            if should_exit:
                real_pnl = (active_trade_agg.entry_price - row['close']) * CONFIG['multiplier'] * CONFIG['allocation_ratio']
                commission = abs(real_pnl) * 0.0001 * 2
                net_pnl = real_pnl - commission

                balance_agg = balance_agg + net_pnl

                trades_agg.append({
                    'date': current_date,
                    'entry_date': active_trade_agg.entry_date,
                    'type': active_trade_agg.exit_reason,
                    'pnl': net_pnl,
                    'max_profit_atr': active_trade_agg.highest_profit_atr,
                    'breakeven_triggered': active_trade_agg.breakeven_triggered,
                })

                last_exit_date_agg = current_date
                last_exit_type_agg = 'Loss' if net_pnl <= 0 else 'Win'
                active_trade_agg = None

        # 开仓 A
        if active_trade_agg is None and raw_signal:
            if last_exit_date_agg:
                days_since = (current_date - last_exit_date_agg).days
                required_cooldown = CONFIG['agg_cooldown_loss'] if last_exit_type_agg == 'Loss' else CONFIG['agg_cooldown_win']

                if days_since >= required_cooldown:
                    active_trade_agg = TradeManager(row['close'], row['atr'], current_date, 'Aggressive')
            else:
                active_trade_agg = TradeManager(row['close'], row['atr'], current_date, 'Aggressive')

        # ==================== 保守轨 B: V2.1 (固定冷却) ====================
        # 平仓 B (使用智能交易管理器)
        if active_trade_def:
            should_exit = active_trade_def.update(row)

            if should_exit:
                real_pnl = (active_trade_def.entry_price - row['close']) * CONFIG['multiplier'] * CONFIG['allocation_ratio']
                commission = abs(real_pnl) * 0.0001 * 2
                net_pnl = real_pnl - commission

                balance_def = balance_def + net_pnl

                trades_def.append({
                    'date': current_date,
                    'entry_date': active_trade_def.entry_date,
                    'type': active_trade_def.exit_reason,
                    'pnl': net_pnl,
                    'max_profit_atr': active_trade_def.highest_profit_atr,
                    'breakeven_triggered': active_trade_def.breakeven_triggered,
                })

                last_exit_date_def = current_date
                active_trade_def = None

        # 开仓 B
        if active_trade_def is None and raw_signal:
            if last_exit_date_def:
                days_since = (current_date - last_exit_date_def).days

                if days_since >= CONFIG['def_cooldown']:
                    active_trade_def = TradeManager(row['close'], row['atr'], current_date, 'Defensive')
            else:
                active_trade_def = TradeManager(row['close'], row['atr'], current_date, 'Defensive')

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

    # 统计保本触发次数
    breakeven_count_agg = len([t for t in trades_agg if t['breakeven_triggered']])
    breakeven_count_def = len([t for t in trades_def if t['breakeven_triggered']])

    # 统计平均最大浮盈
    avg_max_profit_agg = np.mean([t['max_profit_atr'] for t in trades_agg]) if len(trades_agg) > 0 else 0
    avg_max_profit_def = np.mean([t['max_profit_atr'] for t in trades_def]) if len(trades_def) > 0 else 0

    print(f"    [结果] 激进轨: {len(trades_agg)}笔 | 收益:{return_agg:.2f}% | 胜率:{win_rate_agg:.1f}%")
    print(f"           保守轨: {len(trades_def)}笔 | 收益:{return_def:.2f}% | 胜率:{win_rate_def:.1f}%")
    print(f"           复合收益: {total_return:.2f}%")
    print(f"           智能交易统计:")
    print(f"             - 激进轨保本触发:{breakeven_count_agg}笔, 平均最大浮盈:{avg_max_profit_agg:.2f}ATR")
    print(f"             - 保守轨保本触发:{breakeven_count_def}笔, 平均最大浮盈:{avg_max_profit_def:.2f}ATR")

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
        'breakeven_agg': breakeven_count_agg,
        'breakeven_def': breakeven_count_def,
        'avg_max_profit': (avg_max_profit_agg + avg_max_profit_def) / 2,
    }


def backtest_by_year(symbol, start_year, end_year):
    """按年份回测"""
    print(f"\n{'='*100}")
    print(f"年份回测: {symbol} ({start_year}-{end_year}) - V5.0 智能交易逻辑版")
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
        result = backtest_year_v5(df_all, year, symbol)

        if result:
            yearly_results.append(result)

    # 输出总结
    print(f"\n{'='*100}")
    print(f"年份回测总结: {symbol} ({start_year}-{end_year}) - V5.0 智能交易逻辑版")
    print('='*100)

    if yearly_results:
        results_df = pd.DataFrame(yearly_results)

        print(f"\n {'年份':<8} {'复合收益':>12} {'激进轨':>10} {'保守轨':>10} {'总交易':>8} {'胜率':>8} {'保本触发':>10} {'平均最大浮盈':>12}")
        print("-" * 100)

        for _, r in results_df.iterrows():
            print(f"  {r['year']:<8} {r['total_return']:>11.2f}%   {r['agg_return']:>9.2f}%  {r['def_return']:>9.2f}%  {r['total_trades']:>6}笔   {r['win_rate']:>7.1f}%   {r['breakeven_agg']+r['breakeven_def']:>8}笔    {r['avg_max_profit']:>10.2f}ATR")

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
    print("冰火双刀流 - V5.0 (Smart Trader - 智能交易逻辑版)")
    print("=" * 100)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print("\n核心升级: 智能交易逻辑 (Trading Logic)")
    print("  - 移动止损 (Trailing Stop): 让利润奔跑")
    print("  - 保本机制 (Breakeven): 浮盈>1ATR后，止损移至成本价")
    print("  - 动态风控: 只有真正趋势出来时才贪婪")
    print("  - 目标: 截断亏损，让利润奔跑")
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
    print("V5.0 智能交易版 - 最终表现".center(100))
    print("=" * 100)

    for s, results_df in all_results.items():
        if results_df is None or len(results_df) == 0:
            continue
        print(f"\n{s} ({start_year}-{end_year})")
        print("-" * 80)

        print(f" 年份   复合收益    激进轨   保守轨   总交易  胜率  保本触发  平均最大浮盈")
        print("-" * 80)

        for _, r in results_df.iterrows():
            trend_arrow = "↑" if r['total_return'] > 0 else "↓"
            print(f"  {r['year']:<6} {r['total_return']:>10.2f}%   {r['agg_return']:>8.2f}%  {r['def_return']:>8.2f}%  {r['total_trades']:>6}笔  {r['win_rate']:>6.1f}%   {r['breakeven_agg']+r['breakeven_def']:>8}笔   {r['avg_max_profit']:>10.2f}ATR")

    print(f"\n{'='*100}")
    print(f"系统运行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
