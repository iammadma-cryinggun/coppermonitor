# -*- coding: utf-8 -*-
"""
纯碱SA0日线回测 V2 - 使用日线数据
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

def get_daily_data():
    import akshare as ak
    df = ak.futures_main_sina(symbol=SYMBOL)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.dropna()

    df_daily = df.resample('D', on='date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # resample后date是index，不需要set_index
    return df_daily

def calc_indicators(df, params):
    tp = (df['high'] + df['low'] + df['close']) / 3
    cci_length = params['cci_length']
    ma_length = params['ma_length']
    sma = tp.rolling(window=cci_length).mean()
    mad = tp.rolling(window=cci_length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()

    ema_fast = df['close'].ewm(span=23, adjust=False).mean()
    ema_slow = df['close'].ewm(span=50, adjust=False).mean()
    macd = ema_fast - ema_slow
    lowest_macd = macd.rolling(window=10).min()
    highest_macd = macd.rolling(window=10).max()
    range_macd = highest_macd - lowest_macd
    stoch_k = 100 * (macd - lowest_macd) / (highest_macd - lowest_macd)
    stc_raw = stoch_k.rolling(window=10).mean()
    df['stc'] = stc_raw - 50

    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    return df

def run_backtest(df, params):
    df = calc_indicators(df.copy(), params)

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    for i in range(len(df) - 1):
        curr = df.iloc[i]
        nextd = df.iloc[i + 1]
        cci = curr['cci']
        cci_ma = curr['cci_ma']
        stc = curr['stc']
        vol_ratio = curr['vol_ratio']
        cci_prev = df.iloc[i-1]['cci'] if i > 0 else cci
        cci_ma_prev = df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

        if position > 0:
            unrealized = (curr['close'] - entry_price) * position * MULTIPLIER
            current_equity = balance + unrealized
        else:
            current_equity = balance
        equity.append(current_equity)

        if position > 0:
            exit_triggered = False

            if nextd['low'] <= entry_price * 0.96:
                stop_p = max(nextd['open'], entry_price * 0.96)
                exit_p = stop_p * (1 - SLIPPAGE_RATE)
                pnl = (exit_p - entry_price) * position * MULTIPLIER
                comm = exit_p * position * MULTIPLIER * COMMISSION_RATE * 2
                net = pnl - comm
                balance += net
                trades.append({'pnl': net, 'type': 'stop_loss'})
                position = 0
                entry_price = 0.0
                exit_triggered = True

            if not exit_triggered and nextd['high'] >= entry_price * 1.20:
                tp_p = min(nextd['open'], entry_price * 1.20)
                exit_p = tp_p * (1 - SLIPPAGE_RATE)
                pnl = (exit_p - entry_price) * position * MULTIPLIER
                comm = exit_p * position * MULTIPLIER * COMMISSION_RATE * 2
                net = pnl - comm
                balance += net
                trades.append({'pnl': net, 'type': 'take_profit'})
                position = 0
                entry_price = 0.0
                exit_triggered = True

            if not exit_triggered and cci > params['cci_overbought']:
                exit_p = nextd['close'] * (1 - SLIPPAGE_RATE)
                pnl = (exit_p - entry_price) * position * MULTIPLIER
                comm = exit_p * position * MULTIPLIER * COMMISSION_RATE * 2
                net = pnl - comm
                balance += net
                trades.append({'pnl': net, 'type': 'overbought'})
                position = 0
                entry_price = 0.0
                exit_triggered = True

            if not exit_triggered and (cci_prev >= cci_ma_prev and cci < cci_ma):
                exit_p = nextd['close'] * (1 - SLIPPAGE_RATE)
                pnl = (exit_p - entry_price) * position * MULTIPLIER
                comm = exit_p * position * MULTIPLIER * COMMISSION_RATE * 2
                net = pnl - comm
                balance += net
                trades.append({'pnl': net, 'type': 'death_cross'})
                position = 0
                entry_price = 0.0
                exit_triggered = True

        if position == 0:
            open_signal = False
            if (cci < params['cci_oversold'] and stc >= params['stc_oversold']):
                open_signal = True
            elif (cci_prev <= cci_ma_prev and cci > cci_ma and cci <= params['cci_cross_max'] and stc >= params['stc_cross']):
                open_signal = True

            if open_signal:
                if not (params['vol_threshold'] > 0 and not np.isnan(vol_ratio) and vol_ratio < params['vol_threshold']):
                    entry_p = nextd['open'] * (1 + SLIPPAGE_RATE)
                    max_val = balance * 0.9 * LEVERAGE
                    qty = int(max_val / (entry_p * MULTIPLIER))
                    qty = max(1, qty)
                    comm = entry_p * qty * MULTIPLIER * COMMISSION_RATE
                    balance -= comm
                    position = qty

    trades_df = pd.DataFrame(trades)
    total_return = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    if len(trades_df) > 0:
        eq_series = pd.Series(equity)
        max_eq = eq_series.cummax()
        dd = (eq_series - max_eq) / max_eq * 100
        max_dd = dd.min()
        win_trades = trades_df[trades_df['pnl'] > 0]
        lose_trades = trades_df[trades_df['pnl'] < 0]
        win_rate = len(win_trades) / len(trades_df) * 100
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = lose_trades['pnl'].mean() if len(lose_trades) > 0 else 0
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        pf = abs(win_trades['pnl'].sum() / lose_trades['pnl'].sum()) if lose_trades['pnl'].sum() != 0 else 0
        ev = trades_df['pnl'].mean()
        return {
            'total_return': total_return, 'max_dd': max_dd, 'win_rate': win_rate,
            'win_loss_ratio': wl_ratio, 'profit_factor': pf,
            'expected_value': ev, 'total_trades': len(trades_df),
        }
    return None

def main():
    print("=" * 100)
    print("纯碱SA0日线回测 V2".center(100))
    print("=" * 100)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = get_daily_data()
    print(f"\n原始数据: N/A")
    print(f"日线数据: {len(df)}条")
    print(f"时间范围: {df.index[0].date()} 至 {df.index[-1].date()}")

    old_params = {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 190,
        'cci_cross_max': 180,
        'stc_oversold': -50,
        'stc_cross': -130,
        'vol_threshold': 0.50,
    }

    print("\n" + "=" * 100)
    print("使用老参数测试日线数据")
    print("=" * 100)

    result = run_backtest(df.copy(), old_params)

    if result:
        print(f"\n收益率: {result['total_return']:.2f}%")
        print(f"最大回撤: {result['max_dd']:.2f}%")
        print(f"收益回撤比: {result['total_return'] / abs(result['max_dd']):.2f}")
        print(f"胜率: {result['win_rate']:.2f}%")
        print(f"盈亏比: {result['win_loss_ratio']:.2f}")
        print(f"盈亏因子: {result['profit_factor']:.2f}")
        print(f"交易次数: {result['total_trades']}")
        print(f"期望收益: ${result['expected_value']:.2f}/笔")

        print("\n与小时数据对比:")
        print(f"  小时: -25.83% (1502条，6.2年)")
        print(f"  日线: {result['total_return']:.2f}% ({len(df)}条，{(df.index[-1] - df.index[0]).days / 365:.1f}年)")

        print("\n" + "=" * 100)
        print("结论")
        print("=" * 100)
        if result['total_return'] > -10:
            print("✓ 日线数据改善明显，建议实盘使用日线信号")
            print("  原因：日线过滤了日内噪音")
        elif result['total_return'] > -25:
            print("~ 日线略有改善但仍亏损")
            print("  建议：继续优化或增加确认条件")
        else:
            print("✗ 日线数据仍然很糟糕")
            print("  建议：策略可能已失效，需要重新设计")

    print("\n" + "=" * 100)
    print(f"完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

if __name__ == "__main__":
    main()
