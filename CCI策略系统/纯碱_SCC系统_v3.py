# -*- coding: utf-8 -*-
"""
纯碱SA0 - S.C.C.系统 v3.0 结构猎人完整版
注入3个灵魂：STC斜率 + 阳线放量 + 2日确认
"""
import pandas as pd
import numpy as np
from datetime import datetime

SYMBOL = 'sa0'
MULTIPLIER = 20
COMMISSION_RATE = 0.0003
SLIPPAGE_RATE = 0.0002
INITIAL_CAPITAL = 100000
LEVERAGE = 2

# ==========================================
# 战术B参数：结构猎人 v3.0
# ==========================================
TACTICS_B_PARAMS = {
    # STC参数（趋势指南针）- 第一层过滤
    'stc_oversold': -30,         # STC超卖区（底部趴窝）
    'stc_awake': -20,            # STC抬头阈值（脱离底部）
    'stc_slope_min': 2,          # STC斜率最小值（必须向上抬头2点以上）

    # CCI参数（爆发油门）- 第二层确认
    'cci_length': 12,
    'cci_ma_length': 5,
    'cci_oversold': -100,        # CCI超卖区
    'cci_zero_cross': -80,        # CCI向0轴突破
    'cci_confirm_days': 2,        # CCI连续确认天数（避免一日游）

    # 量能+阳线过滤（机构动量）
    'vol_ma_period': 5,
    'vol_threshold': 0.9,         # 量比阈值（严格要求0.9）

    # 风险管理
    'atr_length': 14,
    'atr_stop_multiplier': 2.0,
    'risk_per_trade': 0.02,
    'multiplier': 20,

    # 止盈条件
    'cci_overbought': 130,        # CCI超买平仓（降低避免太晚）
}


def get_data():
    import akshare as ak
    df = ak.futures_main_sina(symbol=SYMBOL)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.dropna()
    df.set_index('date', inplace=True)
    return df


def calc_stc(df, fast_period=23, slow_period=50, cycle_period=10):
    """计算STC"""
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    lowest_macd = macd.rolling(window=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period).max()
    range_macd = highest_macd - lowest_macd

    stc = pd.Series(index=df.index, dtype=float)
    stc.iloc[:cycle_period] = 0

    for i in range(cycle_period, len(macd)):
        if range_macd.iloc[i] != 0:
            stc.iloc[i] = ((macd.iloc[i] - lowest_macd.iloc[i]) / range_macd.iloc[i] - 0.5) * 200
        else:
            stc.iloc[i] = stc.iloc[i-1] if i > cycle_period else 0

    stc_normalized = pd.Series(index=stc.index, dtype=float)
    lookback = 20
    for i in range(len(stc)):
        if i < lookback:
            stc_normalized.iloc[i] = stc.iloc[i]
        else:
            window = stc.iloc[i-lookback+1:i+1]
            low = window.min()
            high = window.max()
            if high != low:
                stc_normalized.iloc[i] = (stc.iloc[i] - low) / (high - low) * 100 - 50
            else:
                stc_normalized.iloc[i] = stc_normalized.iloc[i-1] if i > 0 else 0

    return stc_normalized


def calc_atr(df, period=14):
    """计算ATR"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def run_backtest_v3(df, params):
    """
    v3.0 完整版：STC斜率 + 阳线放量 + 2日确认
    """

    # 计算指标
    print("计算指标...")

    # STC + 斜率
    df['stc'] = calc_stc(df)
    df['stc_prev'] = df['stc'].shift(1)
    df['stc_slope'] = df['stc'] - df['stc_prev']  # STC变化率

    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=params['cci_length']).mean()
    mad = tp.rolling(window=params['cci_length']).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=params['cci_ma_length']).mean()
    df['cci_prev'] = df['cci'].shift(1)

    # ATR
    df['atr'] = calc_atr(df, params['atr_length'])

    # 量比 + 阳线判断
    df['vol_ma'] = df['volume'].rolling(window=params['vol_ma_period']).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['is_bullish_candle'] = df['close'] > df['open']  # 阳线

    # 交易循环
    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_date = None
    entry_atr_stop = 0.0
    trades = []
    equity = []
    cci_confirm_count = 0  # CCI连续确认计数器

    print("开始回测...")

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        # 计算当前权益
        if position > 0:
            unrealized_pnl = (current['close'] - entry_price) * position * params['multiplier']
            current_equity = balance + unrealized_pnl
        else:
            current_equity = balance
        equity.append(current_equity)

        # 平仓逻辑
        if position > 0:
            exit_triggered = False

            # ATR止损
            if next_day['low'] <= entry_atr_stop:
                stop_price = max(next_day['open'], entry_atr_stop)
                exit_price = stop_price * (1 - SLIPPAGE_RATE)
                pnl = (exit_price - entry_price) * position * params['multiplier']
                commission = exit_price * position * params['multiplier'] * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'atr_stop',
                    'hold_days': (next_day.name - entry_date).days,
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # CCI超买平仓（降低阈值到130）
            if not exit_triggered and current['cci'] > params['cci_overbought']:
                exit_price = next_day['close'] * (1 - SLIPPAGE_RATE)
                pnl = (exit_price - entry_price) * position * params['multiplier']
                commission = exit_price * position * params['multiplier'] * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'cci_overbought',
                    'hold_days': (next_day.name - entry_date).days,
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # CCI死叉平仓
            if not exit_triggered and i > 0:
                cci_prev = df.iloc[i-1]['cci']
                cci_ma_prev = df.iloc[i-1]['cci_ma']
                if cci_prev >= cci_ma_prev and current['cci'] < current['cci_ma']:
                    exit_price = next_day['close'] * (1 - SLIPPAGE_RATE)
                    pnl = (exit_price - entry_price) * position * params['multiplier']
                    commission = exit_price * position * params['multiplier'] * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': next_day.name,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'death_cross',
                        'hold_days': (next_day.name - entry_date).days,
                    })
                    position = 0
                    entry_price = 0.0
                    exit_triggered = True

        # 开仓逻辑（v3.0 完整逻辑）
        if position == 0:
            # ========================================
            # 灵魂1：STC斜率检查（拒绝半山腰）
            # ========================================
            # STC不仅>阈值，且必须>前一根 + 2（明确向上抬头）
            stc_prev_oversold = (df.iloc[i-1]['stc'] < params['stc_oversold']) if i > 0 else False
            stc_current_awake = current['stc'] > params['stc_awake']
            stc_slope_strong = current['stc_slope'] > params['stc_slope_min']

            condition_stc = stc_prev_oversold and stc_current_awake and stc_slope_strong

            # ========================================
            # 灵魂2：阳线放量（拒绝阴线放量）
            # ========================================
            # 成交量 > 0.9倍均量 且 收盘价 > 开盘价（必须是阳线）
            vol_sufficient = current['vol_ratio'] > params['vol_threshold']
            is_bullish = current['is_bullish_candle']

            condition_vol = vol_sufficient and is_bullish

            # ========================================
            # 灵魂3：2日确认（拒绝一日游）
            # ========================================
            # CCI连续2天都在回升
            if i > 0:
                cci_rising_prev = df.iloc[i-1]['cci'] > df.iloc[i-1]['cci_prev'] if i > 1 else False
                cci_rising_current = current['cci'] > current['cci_prev']
                cci_consecutive = cci_rising_prev and cci_rising_current
            else:
                cci_consecutive = False

            # CCI必须突破阈值
            cci_breakout = (current['cci'] > params['cci_zero_cross'] and
                          current['cci'] > current['cci_ma'])

            condition_cci = cci_breakout and cci_consecutive

            # ========================================
            # 最终开仓指令：3个灵魂全部满足
            # ========================================
            if condition_stc and condition_vol and condition_cci:
                entry_price = next_day['open'] * (1 + SLIPPAGE_RATE)

                # ATR止损
                entry_atr_stop = entry_price - current['atr'] * params['atr_stop_multiplier']

                # 仓位计算
                risk_amount = balance * params['risk_per_trade']
                stop_distance = entry_price - entry_atr_stop
                max_value = balance * 0.9 * LEVERAGE

                qty_by_risk = int(risk_amount / (stop_distance * params['multiplier']))
                qty_by_capital = int(max_value / (entry_price * params['multiplier']))
                qty = min(max(1, qty_by_risk), qty_by_capital)

                commission = entry_price * qty * params['multiplier'] * COMMISSION_RATE
                balance -= commission

                position = qty
                entry_date = next_day.name

                print(f"开仓: {entry_date.date()} 价格:{entry_price:.2f} "
                      f"ATR止损:{entry_atr_stop:.2f} CCI:{current['cci']:.2f} "
                      f"STC:{current['stc']:.2f}(斜:{current['stc_slope']:.2f}) "
                      f"量比:{current['vol_ratio']:.2f} 阳:{is_bullish}")

    # 计算最终指标
    trades_df = pd.DataFrame(trades)
    total_return = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    if len(trades_df) > 0:
        equity_series = pd.Series(equity)
        max_equity = equity_series.cummax()
        drawdown = (equity_series - max_equity) / max_equity * 100
        max_dd = drawdown.min()

        win_trades = trades_df[trades_df['pnl'] > 0]
        lose_trades = trades_df[trades_df['pnl'] < 0]

        win_rate = len(win_trades) / len(trades_df) * 100
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = lose_trades['pnl'].mean() if len(lose_trades) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        profit_factor = abs(win_trades['pnl'].sum() / lose_trades['pnl'].sum()) if lose_trades['pnl'].sum() != 0 else 0
        expected_value = trades_df['pnl'].mean()

        avg_hold_days = trades_df['hold_days'].mean()

        exit_stats = trades_df['type'].value_counts()

        return {
            'total_return': total_return,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expected_value': expected_value,
            'total_trades': len(trades_df),
            'avg_hold_days': avg_hold_days,
            'exit_stats': exit_stats,
            'trades_df': trades_df,
            'final_balance': balance,
        }
    else:
        return None


def print_results(result, params):
    """打印详细回测结果"""

    print("\n" + "=" * 100)
    print("纯碱SA0 - S.C.C.系统 v3.0 结构猎人完整版".center(100))
    print("=" * 100)

    if result is None:
        print("\n[FAILED] 回测失败或无交易")
        return

    print(f"\n数据时间范围: {result['trades_df']['entry_date'].min()} 至 {result['trades_df']['exit_date'].max()}")

    print("\n" + "=" * 100)
    print("整体表现")
    print("=" * 100)
    print(f"  总收益率:     {result['total_return']:>10.2f}%")
    print(f"  最大回撤:     {result['max_dd']:>10.2f}%")
    if result['max_dd'] != 0:
        print(f"  收益回撤比:   {result['total_return'] / abs(result['max_dd']):>10.2f}")
    print(f"  胜率:         {result['win_rate']:>10.2f}%")
    print(f"  盈亏比:       {result['win_loss_ratio']:>10.2f}")
    print(f"  盈亏因子:     {result['profit_factor']:>10.2f}")
    print(f"  期望收益:     ${result['expected_value']:>10.2f}/笔")
    print(f"  总交易次数:   {result['total_trades']:>10}笔")
    print(f"  平均持仓天数: {result['avg_hold_days']:>10.1f}天")

    print("\n" + "=" * 100)
    print("平仓类型统计")
    print("=" * 100)
    print(f"  {'类型':<20} {'次数':>8} {'占比':>10} {'平均盈亏':>12} {'总盈亏':>14}")
    print("-" * 80)

    type_map = {
        'atr_stop': 'ATR动态止损',
        'cci_overbought': 'CCI超买平仓',
        'death_cross': 'CCI死叉平仓'
    }

    for exit_type, count in result['exit_stats'].items():
        pct = count / result['total_trades'] * 100
        type_trades = result['trades_df'][result['trades_df']['type'] == exit_type]
        avg_pnl = type_trades['pnl'].mean()
        total_pnl = type_trades['pnl'].sum()
        print(f"  {type_map.get(exit_type, exit_type):<20} {count:>8} {pct:>9.1f}% {avg_pnl:>11.0f} {total_pnl:>13.0f}")

    print("\n" + "=" * 100)
    print("注入的3个灵魂")
    print("=" * 100)
    print("  1. STC斜率检查: STC > 前一根 + 2 (拒绝半山腰)")
    print("  2. 阳线放量: vol_ratio > 0.9 且 close > open (拒绝阴线放量)")
    print("  3. 2日确认: CCI连续2天回升 (拒绝一日游)")
    print(f"\n  当前参数:")
    print(f"    STC抬头阈值: > {params['stc_awake']}")
    print(f"    STC斜率最小值: > {params['stc_slope_min']}")
    print(f"    量比阈值: > {params['vol_threshold']}")
    print(f"    CCI确认天数: {params['cci_confirm_days']}天")

    print("\n" + "=" * 100)


def main():
    print("=" * 100)
    print("纯碱SA0 - S.C.C.系统 v3.0 结构猎人".center(100))
    print("注入3个灵魂：STC斜率 + 阳线放量 + 2日确认".center(100))
    print("=" * 100)
    print(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # 获取数据
    print(f"\n[1/2] 获取数据...")
    df = get_data()

    if df is None:
        print("[FAILED] 数据获取失败")
        return

    print(f"数据量: {len(df)}条")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")

    # 运行回测
    print(f"\n[2/2] 运行回测...")
    result = run_backtest_v3(df, TACTICS_B_PARAMS)

    if result:
        # 输出结果
        print(f"\n[2/2] 输出结果...")
        print_results(result, TACTICS_B_PARAMS)
    else:
        print("\n[FAILED] 回测失败或无交易")

    print("\n" + "=" * 100)
    print(f"回测完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
