"""
纯碱(SA0) 分段回测 - 找出策略失效时间点
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from datetime import datetime

# 纯碱参数
SA_PARAMS = {
    'code': 'sa0',
    'exchange': 'CZCE',
    'multiplier': 20,
    'cci_length': 12,
    'ma_length': 5,
    'cci_oversold': -40,
    'cci_overbought': 190,
    'cci_cross_max': 180,
    'stc_oversold': -50,
    'stc_cross': -130,
}

COMMISSION_RATE = 0.0003
SLIPPAGE_RATE = 0.0002
INITIAL_CAPITAL = 100000
LEVERAGE = 2
VOL_THRESHOLD = 0.50


def get_akshare_data(code):
    """从akshare获取完整历史数据"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def calculate_stc(close_prices, fast_period=23, slow_period=50, cycle_period=10):
    """计算STC指标"""
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    lowest_macd = macd.rolling(window=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period).max()
    range_macd = highest_macd - lowest_macd

    stc = pd.Series(index=close_prices.index, dtype=float)
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


def run_backtest(df, params, vol_threshold, start_year=None, end_year=None):
    """运行老版本回测"""

    # 筛选时间范围
    if start_year:
        df = df[df.index.year >= start_year].copy()
    if end_year:
        df = df[df.index.year <= end_year].copy()

    print(f"\n数据时间范围: {df.index[0].date()} 至 {df.index[-1].date()} ({len(df)}天)")

    # 计算指标
    tp = (df['high'] + df['low'] + df['close']) / 3
    cci_length = params['cci_length']
    ma_length = params['ma_length']

    sma = tp.rolling(window=cci_length).mean()
    mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()

    df['stc'] = calculate_stc(df['close'])

    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_date = None
    entry_cci = None
    trades = []
    equity = []
    skipped_low_vol = 0

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        cci = current['cci']
        cci_ma = current['cci_ma']
        stc = current['stc']
        vol_ratio = current['vol_ratio']

        cci_prev = df.iloc[i-1]['cci'] if i > 0 else cci
        cci_ma_prev = df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

        if position > 0:
            unrealized_pnl = (current['close'] - entry_price) * position * params['multiplier']
            current_equity = balance + unrealized_pnl
        else:
            current_equity = balance
        equity.append(current_equity)

        # 平仓条件检查
        if position > 0:
            exit_triggered = False

            # 1. 止损(-4%)
            if next_day['low'] <= entry_price * 0.96:
                stop_price = max(next_day['open'], entry_price * 0.96)
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
                    'type': 'stop_loss',
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # 2. 止盈(20%)
            if not exit_triggered and next_day['high'] >= entry_price * 1.20:
                take_profit_price = min(next_day['open'], entry_price * 1.20)
                exit_price = take_profit_price * (1 - SLIPPAGE_RATE)
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
                    'type': 'take_profit',
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # 3. CCI超买平仓
            if not exit_triggered and cci > params['cci_overbought']:
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
                    'type': 'overbought',
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # 4. CCI死叉平仓
            if not exit_triggered and (cci_prev >= cci_ma_prev and cci < cci_ma):
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
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

        # 开仓逻辑
        if position == 0:
            open_signal = False

            if (cci < params['cci_oversold'] and
                stc >= params['stc_oversold']):
                open_signal = True

            elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                  cci <= params['cci_cross_max'] and
                  stc >= params['stc_cross']):
                open_signal = True

            if open_signal:
                if vol_threshold > 0 and not np.isnan(vol_ratio) and vol_ratio < vol_threshold:
                    skipped_low_vol += 1
                else:
                    entry_price = next_day['open'] * (1 + SLIPPAGE_RATE)
                    max_value = balance * 0.9 * LEVERAGE
                    qty = int(max_value / (entry_price * params['multiplier']))
                    qty = max(1, qty)
                    commission = entry_price * qty * params['multiplier'] * COMMISSION_RATE
                    balance -= commission

                    position = qty
                    entry_date = next_day.name
                    entry_cci = cci

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

        avg_hold_days = (trades_df['exit_date'] - trades_df['entry_date']).dt.days.mean()

        return {
            'total_return': total_return,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expected_value': expected_value,
            'total_trades': len(trades_df),
            'avg_hold_days': avg_hold_days,
            'final_balance': balance,
            'start_date': df.index[0],
            'end_date': df.index[-1],
        }
    else:
        return None


def main():
    print("=" * 100)
    print("纯碱(SA0) 分段回测 - 找出策略失效时间点".center(100))
    print("=" * 100)

    df = get_akshare_data('sa0')

    if df is None:
        print("[FAILED] 数据获取失败")
        return

    # 测试不同时间段
    periods = [
        (2019, 2022, "2019-2022（早期数据）"),
        (2023, 2025, "2023-2025（近期数据）"),
        (2019, None, "2019-至今（全部数据）"),
    ]

    print(f"\n{'时间段':<20} {'收益率':>12} {'最大回撤':<12} {'交易数':>8} {'胜率':<10}")
    print("-" * 70)

    for start_year, end_year, label in periods:
        result = run_backtest(df.copy(), SA_PARAMS, VOL_THRESHOLD, start_year, end_year)

        if result:
            print(f"{label:<20} {result['total_return']:>10.2f}% {result['max_dd']:>10.2f}% {result['total_trades']:>8}笔 {result['win_rate']:>8.1f}%")
        else:
            print(f"{label:<20} 无交易")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
