# -*- coding: utf-8 -*-
"""
纯碱SA0 - S.C.C.系统 多空不对称v3
基于v3.1（验证可用）添加做空逻辑
"""
import pandas as pd
import numpy as np
from datetime import datetime

SYMBOL = 'sa0'
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
SLIPPAGE_RATE = 0.0002
INITIAL_CAPITAL = 100000
MARGIN_RATE = 0.10
LEVERAGE = 2

# ==========================================
# 参数配置
# ==========================================
PARAMS = {
    # 共享参数
    'cci_length': 12,
    'cci_ma_length': 5,
    'atr_length': 14,
    'vol_ma_period': 5,
    'risk_per_trade': 0.02,
    'multiplier': 20,

    # 做多参数（基于v3.1验证可用的参数）
    'long_stc_oversold': -30,
    'long_stc_awake': -20,
    'long_stc_slope_min': 2,
    'long_cci_oversold': -100,
    'long_cci_zero_cross': -80,
    'long_vol_threshold': 0.9,
    'long_atr_stop_multiplier': 2.0,
    'long_cci_overbought': 130,

    # 做空参数（相对宽松，确保能触发）
    'short_stc_overbought': 30,      # STC>30为高位
    'short_stc_breakdown': -20,       # STC跌破-20为确认
    'short_stc_slope_max': -2,         # STC斜率<-2（快速掉头）
    'short_cci_overbought': 100,       # CCI>100为超买
    'short_cci_falling_threshold': -50, # CCI跌破-50
    'short_cci_zero_cross': 0,         # CCI跌破0轴
    'short_vol_min': 0.3,             # 量比>0.3（不要求放量）
    'short_vol_max': 5.0,             # 量比<5.0（防止极端值）
    'short_atr_stop_multiplier': 1.5,  # 更紧的止损
    'short_cci_oversold': -150,        # CCI<-150平空
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


def run_backtest(df, params):
    """多空不对称回测"""

    # 计算指标
    print("计算指标...")

    df['stc'] = calc_stc(df)
    df['stc_prev'] = df['stc'].shift(1)
    df['stc_slope'] = df['stc'] - df['stc_prev']

    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=params['cci_length']).mean()
    mad = tp.rolling(window=params['cci_length']).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=params['cci_ma_length']).mean()
    df['cci_prev'] = df['cci'].shift(1)

    df['atr'] = calc_atr(df, params['atr_length'])

    df['vol_ma'] = df['volume'].rolling(window=params['vol_ma_period']).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['is_bullish_candle'] = df['close'] > df['open']

    # 交易状态变量
    balance = INITIAL_CAPITAL
    position_long = 0
    position_short = 0
    entry_price_long = 0.0
    entry_price_short = 0.0
    entry_date_long = None
    entry_date_short = None
    entry_atr_stop_long = 0.0
    entry_atr_stop_short = 0.0
    entry_atr_long = 0.0
    entry_atr_short = 0.0
    has_taken_half_profit_long = False
    has_taken_half_profit_short = False

    trades = []
    equity = []

    print("开始回测...")

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        # 计算当前权益
        unrealized_pnl = 0
        if position_long > 0:
            unrealized_pnl += (current['close'] - entry_price_long) * position_long * params['multiplier']
        if position_short > 0:
            unrealized_pnl += (entry_price_short - current['close']) * position_short * params['multiplier']
        current_equity = balance + unrealized_pnl
        equity.append(current_equity)

        # ========================================================
        # 做多平仓逻辑
        # ========================================================
        if position_long > 0:
            exit_triggered = False

            # 计算浮盈（ATR倍数）
            current_profit_atr = (current['close'] - entry_price_long) / entry_atr_long

            # 50%分批止盈
            if not has_taken_half_profit_long:
                if current['cci'] > params['long_cci_first_target'] or current_profit_atr > params['long_profit_atr_multiple']:
                    exit_qty = int(position_long * params['long_half_exit_ratio'])
                    exit_qty = max(1, exit_qty)

                    exit_price = current['close'] * (1 - SLIPPAGE_RATE)
                    pnl = (exit_price - entry_price_long) * exit_qty * params['multiplier']
                    commission = exit_price * exit_qty * params['multiplier'] * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl

                    trades.append({
                        'direction': 'long',
                        'entry_date': entry_date_long,
                        'exit_date': current.name,
                        'entry_price': entry_price_long,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'half_profit_long',
                        'hold_days': (current.name - entry_date_long).days,
                        'exit_qty': exit_qty,
                    })

                    print(f"多50%止盈: {current.name.date()} 价格:{exit_price:.2f} CCI:{current['cci']:.2f} 盈利:{net_pnl:.0f}")

                    position_long -= exit_qty
                    has_taken_half_profit_long = True

                    # 止损移到成本价
                    entry_atr_stop_long = entry_price_long

                    if position_long == 0:
                        entry_price_long = 0.0
                        has_taken_half_profit_long = False
                        exit_triggered = True

            # 剩余仓位平仓
            if not exit_triggered and position_long > 0:
                # ATR止损
                if next_day['low'] <= entry_atr_stop_long:
                    stop_price = max(next_day['open'], entry_atr_stop_long)
                    exit_price = stop_price * (1 - SLIPPAGE_RATE)
                    pnl = (exit_price - entry_price_long) * position_long * params['multiplier']
                    commission = exit_price * position_long * params['multiplier'] * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'direction': 'long',
                        'entry_date': entry_date_long,
                        'exit_date': next_day.name,
                        'entry_price': entry_price_long,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'atr_stop_long',
                        'hold_days': (next_day.name - entry_date_long).days,
                    })
                    position_long = 0
                    entry_price_long = 0.0
                    has_taken_half_profit_long = False
                    exit_triggered = True

                # CCI超买平仓
                if not exit_triggered and current['cci'] > params['long_cci_overbought']:
                    exit_price = next_day['close'] * (1 - SLIPPAGE_RATE)
                    pnl = (exit_price - entry_price_long) * position_long * params['multiplier']
                    commission = exit_price * position_long * params['multiplier'] * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'direction': 'long',
                        'entry_date': entry_date_long,
                        'exit_date': next_day.name,
                        'entry_price': entry_price_long,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'cci_overbought_long',
                        'hold_days': (next_day.name - entry_date_long).days,
                    })
                    position_long = 0
                    entry_price_long = 0.0
                    has_taken_half_profit_long = False
                    exit_triggered = True

                # CCI死叉平仓
                if not exit_triggered and i > 0:
                    cci_prev = df.iloc[i-1]['cci']
                    cci_ma_prev = df.iloc[i-1]['cci_ma']
                    if cci_prev >= cci_ma_prev and current['cci'] < current['cci_ma']:
                        exit_price = next_day['close'] * (1 - SLIPPAGE_RATE)
                        pnl = (exit_price - entry_price_long) * position_long * params['multiplier']
                        commission = exit_price * position_long * params['multiplier'] * COMMISSION_RATE * 2
                        net_pnl = pnl - commission
                        balance += net_pnl
                        trades.append({
                            'direction': 'long',
                            'entry_date': entry_date_long,
                            'exit_date': next_day.name,
                            'entry_price': entry_price_long,
                            'exit_price': exit_price,
                            'pnl': net_pnl,
                            'type': 'death_cross_long',
                            'hold_days': (next_day.name - entry_date_long).days,
                        })
                        position_long = 0
                        entry_price_long = 0.0
                        has_taken_half_profit_long = False
                        exit_triggered = True

        # ========================================================
        # 做空平仓逻辑
        # ========================================================
        if position_short > 0:
            exit_triggered = False

            # 计算浮盈（ATR倍数）
            current_profit_atr = (entry_price_short - current['close']) / entry_atr_short

            # 50%分批止盈（更敏感）
            if not has_taken_half_profit_short:
                if current['cci'] < params['short_cci_first_target'] or current_profit_atr > params['short_profit_atr_multiple']:
                    exit_qty = int(position_short * params['short_half_exit_ratio'])
                    exit_qty = max(1, exit_qty)

                    exit_price = current['close'] * (1 + SLIPPAGE_RATE)
                    pnl = (entry_price_short - exit_price) * exit_qty * params['multiplier']
                    commission = exit_price * exit_qty * params['multiplier'] * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl

                    trades.append({
                        'direction': 'short',
                        'entry_date': entry_date_short,
                        'exit_date': current.name,
                        'entry_price': entry_price_short,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'half_profit_short',
                        'hold_days': (current.name - entry_date_short).days,
                        'exit_qty': exit_qty,
                    })

                    print(f"空50%止盈: {current.name.date()} 价格:{exit_price:.2f} CCI:{current['cci']:.2f} 盈利:{net_pnl:.0f}")

                    position_short -= exit_qty
                    has_taken_half_profit_short = True

                    # 止损移到成本价
                    entry_atr_stop_short = entry_price_short

                    if position_short == 0:
                        entry_price_short = 0.0
                        has_taken_half_profit_short = False
                        exit_triggered = True

            # 剩余仓位平仓
            if not exit_triggered and position_short > 0:
                # ATR止损
                if next_day['high'] >= entry_atr_stop_short:
                    stop_price = min(next_day['open'], entry_atr_stop_short)
                    exit_price = stop_price * (1 + SLIPPAGE_RATE)
                    pnl = (entry_price_short - exit_price) * position_short * params['multiplier']
                    commission = exit_price * position_short * params['multiplier'] * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'direction': 'short',
                        'entry_date': entry_date_short,
                        'exit_date': next_day.name,
                        'entry_price': entry_price_short,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'atr_stop_short',
                        'hold_days': (next_day.name - entry_date_short).days,
                    })
                    position_short = 0
                    entry_price_short = 0.0
                    has_taken_half_profit_short = False
                    exit_triggered = True

                # CCI超卖平仓（防止逼空）
                if not exit_triggered and current['cci'] < params['short_cci_oversold']:
                    exit_price = next_day['close'] * (1 + SLIPPAGE_RATE)
                    pnl = (entry_price_short - exit_price) * position_short * params['multiplier']
                    commission = exit_price * position_short * params['multiplier'] * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'direction': 'short',
                        'entry_date': entry_date_short,
                        'exit_date': next_day.name,
                        'entry_price': entry_price_short,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'cci_oversold_short',
                        'hold_days': (next_day.name - entry_date_short).days,
                    })
                    position_short = 0
                    entry_price_short = 0.0
                    has_taken_half_profit_short = False
                    exit_triggered = True

                # CCI金叉平仓
                if not exit_triggered and i > 0:
                    cci_prev = df.iloc[i-1]['cci']
                    cci_ma_prev = df.iloc[i-1]['cci_ma']
                    if cci_prev <= cci_ma_prev and current['cci'] > current['cci_ma']:
                        exit_price = next_day['close'] * (1 + SLIPPAGE_RATE)
                        pnl = (entry_price_short - exit_price) * position_short * params['multiplier']
                        commission = exit_price * position_short * params['multiplier'] * COMMISSION_RATE * 2
                        net_pnl = pnl - commission
                        balance += net_pnl
                        trades.append({
                            'direction': 'short',
                            'entry_date': entry_date_short,
                            'exit_date': next_day.name,
                            'entry_price': entry_price_short,
                            'exit_price': exit_price,
                            'pnl': net_pnl,
                            'type': 'golden_cross_short',
                            'hold_days': (next_day.name - entry_date_short).days,
                        })
                        position_short = 0
                        entry_price_short = 0.0
                        has_taken_half_profit_short = False
                        exit_triggered = True

        # ========================================================
        # 做多开仓逻辑（基于v3.1）
        # ========================================================
        if position_long == 0:
            stc_prev_oversold = (df.iloc[i-1]['stc'] < params['long_stc_oversold']) if i > 0 else False
            stc_current_awake = current['stc'] > params['long_stc_awake']
            stc_slope_strong = current['stc_slope'] > params['long_stc_slope_min']

            condition_stc = stc_prev_oversold and stc_current_awake and stc_slope_strong

            vol_sufficient = current['vol_ratio'] > params['long_vol_threshold']
            is_bullish = current['is_bullish_candle']
            condition_vol = vol_sufficient and is_bullish

            if i > 0:
                cci_rising_prev = df.iloc[i-1]['cci'] > df.iloc[i-1]['cci_prev'] if i > 1 else False
                cci_rising_current = current['cci'] > current['cci_prev']
                cci_consecutive = cci_rising_prev and cci_rising_current
            else:
                cci_consecutive = False

            cci_breakout = (current['cci'] > params['long_cci_zero_cross'] and
                          current['cci'] > current['cci_ma'])

            condition_cci = cci_breakout and cci_consecutive

            if condition_stc and condition_vol and condition_cci:
                entry_price = next_day['open'] * (1 + SLIPPAGE_RATE)
                entry_atr_long = current['atr']
                entry_atr_stop_long = entry_price - entry_atr_long * params['long_atr_stop_multiplier']

                risk_amount = balance * params['risk_per_trade']
                stop_distance = entry_price - entry_atr_stop_long

                margin_required = entry_price * 1 * params['multiplier'] * MARGIN_RATE
                max_value = balance / MARGIN_RATE

                qty_by_risk = int(risk_amount / (stop_distance * params['multiplier']))
                qty_by_capital = int(max_value / (entry_price * params['multiplier']))
                qty = min(max(1, qty_by_risk), qty_by_capital)

                commission = entry_price * qty * params['multiplier'] * COMMISSION_RATE
                balance -= commission

                position_long = qty
                entry_price_long = entry_price
                entry_date_long = next_day.name
                has_taken_half_profit_long = False

                print(f"开多: {entry_date_long.date()} 价格:{entry_price:.2f} "
                      f"ATR止损:{entry_atr_stop_long:.2f} CCI:{current['cci']:.2f} "
                      f"STC:{current['stc']:.2f}(斜:{current['stc_slope']:.2f}) "
                      f"量比:{current['vol_ratio']:.2f}")

        # ========================================================
        # 做空开仓逻辑（宽松条件，确保能触发）
        # ========================================================
        if position_short == 0:
            # STC: 从高位掉头
            stc_was_high = (df.iloc[i-1]['stc'] > params['short_stc_overbought']) if i > 0 else False
            stc_breakdown = current['stc'] < params['short_stc_breakdown']
            stc_slope_down = current['stc_slope'] < params['short_stc_slope_max']
            condition_stc = stc_was_high and stc_breakdown and stc_slope_down

            # 量能: 不需要放量，只要不是极端值
            vol_ok = (params['short_vol_min'] < current['vol_ratio'] < params['short_vol_max'])

            # CCI: 超买后回落或跌破关键位
            cci_was_high = (df.iloc[i-1]['cci'] > params['short_cci_overbought']) if i > 0 else False
            cci_falling = current['cci'] < df.iloc[i-1]['cci'] if i > 0 else False
            cci_break_threshold = current['cci'] < params['short_cci_falling_threshold']
            cci_break_zero = current['cci'] < params['short_cci_zero_cross']
            condition_cci = (cci_was_high and cci_falling) or cci_break_threshold or cci_break_zero

            if condition_stc and vol_ok and condition_cci:
                entry_price = next_day['open'] * (1 - SLIPPAGE_RATE)
                entry_atr_short = current['atr']
                entry_atr_stop_short = entry_price + entry_atr_short * params['short_atr_stop_multiplier']

                risk_amount = balance * params['risk_per_trade']
                stop_distance = entry_atr_stop_short - entry_price

                margin_required = entry_price * 1 * params['multiplier'] * MARGIN_RATE
                max_value = balance / MARGIN_RATE

                qty_by_risk = int(risk_amount / (stop_distance * params['multiplier']))
                qty_by_capital = int(max_value / (entry_price * params['multiplier']))
                qty = min(max(1, qty_by_risk), qty_by_capital)

                commission = entry_price * qty * params['multiplier'] * COMMISSION_RATE
                balance -= commission

                position_short = qty
                entry_price_short = entry_price
                entry_date_short = next_day.name
                has_taken_half_profit_short = False

                print(f"开空: {entry_date_short.date()} 价格:{entry_price:.2f} "
                      f"ATR止损:{entry_atr_stop_short:.2f} CCI:{current['cci']:.2f} "
                      f"STC:{current['stc']:.2f}(斜:{current['stc_slope']:.2f}) "
                      f"量比:{current['vol_ratio']:.2f}")

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

        long_trades = trades_df[trades_df['direction'] == 'long']
        short_trades = trades_df[trades_df['direction'] == 'short']

        long_pnl = long_trades['pnl'].sum() if len(long_trades) > 0 else 0
        short_pnl = short_trades['pnl'].sum() if len(short_trades) > 0 else 0

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
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'long_win_rate': len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0,
            'short_win_rate': len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0,
        }
    else:
        return None


def print_results(result, params):
    """打印详细回测结果"""

    print("\n" + "=" * 100)
    print("纯碱SA0 - S.C.C.系统 多空不对称v3".center(100))
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
    print("多空不对称对比")
    print("=" * 100)
    print(f"  {'方向':<8} {'交易次数':>10} {'胜率':>10} {'总盈亏':>15}")
    print("-" * 50)
    print(f"  {'多头':<8} {result['long_trades']:>10}笔 {result['long_win_rate']:>9.2f}% {result['long_pnl']:>14.0f}元")
    print(f"  {'空头':<8} {result['short_trades']:>10}笔 {result['short_win_rate']:>9.2f}% {result['short_pnl']:>14.0f}元")

    print("\n" + "=" * 100)
    print("平仓类型统计")
    print("=" * 100)
    print(f"  {'类型':<25} {'次数':>8} {'占比':>10} {'平均盈亏':>12} {'总盈亏':>14}")
    print("-" * 80)

    type_map = {
        'half_profit_long': '多50%分批止盈',
        'half_profit_short': '空50%分批止盈',
        'atr_stop_long': '多ATR保本止损',
        'atr_stop_short': '空ATR止损',
        'cci_overbought_long': '多CCI超买平仓',
        'cci_oversold_short': '空CCI超卖平仓',
        'death_cross_long': '多CCI死叉平仓',
        'golden_cross_short': '空CCI金叉平仓',
    }

    for exit_type, count in result['exit_stats'].items():
        pct = count / result['total_trades'] * 100
        type_trades = result['trades_df'][result['trades_df']['type'] == exit_type]
        avg_pnl = type_trades['pnl'].mean()
        total_pnl = type_trades['pnl'].sum()
        print(f"  {type_map.get(exit_type, exit_type):<25} {count:>8} {pct:>9.1f}% {avg_pnl:>11.0f} {total_pnl:>13.0f}")

    print("\n" + "=" * 100)
    print("多空不对称核心逻辑")
    print("=" * 100)
    print("  做多（结构猎人）：")
    print("    STC: 从底部抬头(<-30 → >-20) + 斜率>2")
    print("    量能: 必须放量(>0.9) + 必须阳线")
    print("    CCI: 突破-80 + 2日确认")
    print("    止盈: CCI>100 或 浮盈>1.5ATR → 50%止盈")
    print("\n  做空（重力加速器）：")
    print("    STC: 从高位掉头(>30 → <-20) + 斜率<-2")
    print("    量能: 不需要放量(0.3 < vol < 5.0)")
    print("    CCI: 超买后回落 或 跌破-50/0")
    print("    止盈: CCI<-100 或 浮盈>1.0ATR → 50%止盈（更敏感）")

    print("\n" + "=" * 100)


def main():
    print("=" * 100)
    print("纯碱SA0 - S.C.C.系统 多空不对称v3".center(100))
    print("基于v3.1做多 + 宽松做空条件".center(100))
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
    result = run_backtest(df, PARAMS)

    if result:
        # 输出结果
        print(f"\n[2/2] 输出结果...")
        print_results(result, PARAMS)
    else:
        print("\n[FAILED] 回测失败或无交易")

    print("\n" + "=" * 100)
    print(f"回测完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
