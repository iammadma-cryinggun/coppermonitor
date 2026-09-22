"""分析纯碱的详细回测情况"""
import pandas as pd
import numpy as np
import sys
sys.path.append('D:/期货数据/铜期货监控/CCI策略系统')

# 纯碱参数
params = {
    'cci_length': 12, 'ma_length': 5,
    'cci_oversold': -40, 'cci_overbought': 190,
    'cci_cross_max': 180, 'stc_oversold': -50,
    'stc_cross': -130, 'multiplier': 20,
}

def get_akshare_data(code):
    """从akshare获取数据"""
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

    # 归一化
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

def run_backtest(df, params):
    """运行回测"""
    commission_rate = 0.0003
    slippage_rate = 0.0002
    multiplier = params['multiplier']
    initial_capital = 100000
    leverage = 2

    # 计算指标
    tp = (df['high'] + df['low'] + df['close']) / 3
    cci_length = params['cci_length']
    ma_length = params['ma_length']
    sma = tp.rolling(window=cci_length).mean()
    mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()
    df['stc'] = calculate_stc(df['close'])

    balance = initial_capital
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        cci = current['cci']
        cci_ma = current['cci_ma']
        stc = current['stc']

        cci_prev = df.iloc[i-1]['cci'] if i > 0 else cci
        cci_ma_prev = df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

        equity.append(balance)

        if position > 0:
            # 止损（-4%）
            if next_day['low'] <= entry_price * 0.96:
                stop_price = max(next_day['open'], entry_price * 0.96)
                exit_price = stop_price * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'stop_loss',
                    'hold_days': (next_day.name - entry_date).days
                })
                position = 0
                entry_price = 0.0
                continue

            # CCI超买平仓（主要止盈）
            if cci > params['cci_overbought']:
                exit_price = next_day['close'] * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'overbought',
                    'hold_days': (next_day.name - entry_date).days
                })
                position = 0
                entry_price = 0.0
                continue

            # CCI死叉平仓
            if cci_prev >= cci_ma_prev and cci < cci_ma:
                exit_price = next_day['close'] * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'death_cross',
                    'hold_days': (next_day.name - entry_date).days
                })
                position = 0
                entry_price = 0.0
                continue

        # 开仓
        if position == 0:
            open_signal = False

            # 超卖开仓
            if (cci < params['cci_oversold'] and
                stc >= params['stc_oversold']):
                open_signal = True

            # 金叉开仓
            elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                  cci <= params['cci_cross_max'] and
                  stc >= params['stc_cross']):
                open_signal = True

            if open_signal:
                entry_price = next_day['open'] * (1 + slippage_rate)
                max_value = balance * 0.9 * leverage
                qty = int(max_value / (entry_price * multiplier))
                qty = max(1, qty)
                position = qty
                entry_date = next_day.name

    # 计算最终指标
    trades_df = pd.DataFrame(trades)
    total_return = (balance - initial_capital) / initial_capital * 100

    if len(trades_df) > 0:
        equity_series = pd.Series(equity + [balance])
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

        # 平仓类型统计
        exit_stats = trades_df['type'].value_counts()

        return {
            'total_return': total_return,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expected_value': expected_value,
            'total_trades': len(trades_df),
            'exit_stats': exit_stats,
            'trades_df': trades_df
        }
    else:
        return None

print("=" * 100)
print("纯碱(SA0) 详细回测分析".center(100))
print("=" * 100)

# 获取数据
df = get_akshare_data('sa0')
if df is not None:
    print(f"数据范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"数据量: {len(df)} 条\n")

    # 运行回测
    result = run_backtest(df, params)

    if result:
        print("\n【整体表现】")
        print(f"  总收益率:     {result['total_return']:.2f}%")
        print(f"  最大回撤:     {result['max_dd']:.2f}%")
        print(f"  胜率:         {result['win_rate']:.1f}%")
        print(f"  盈亏比:       {result['win_loss_ratio']:.2f}")
        print(f"  盈亏因子:     {result['profit_factor']:.2f}")
        print(f"  期望收益:     ${result['expected_value']:.0f}/笔")
        print(f"  总交易次数:   {result['total_trades']}笔")

        print("\n【平仓类型统计】")
        print(f"  {'类型':<15} {'次数':>8} {'占比':>10}")
        print("-" * 40)

        type_map = {
            'stop_loss': '止损(-4%)',
            'overbought': 'CCI超买平仓',
            'death_cross': 'CCI死叉平仓'
        }

        for exit_type, count in result['exit_stats'].items():
            pct = count / result['total_trades'] * 100
            print(f"  {type_map.get(exit_type, exit_type):<15} {count:>8} {pct:>9.1f}%")

        print("\n【各平仓类型盈亏分析】")
        trades_df = result['trades_df']
        print(f"  {'类型':<15} {'次数':>6} {'平均盈亏':>12} {'总盈亏':>14}")
        print("-" * 50)

        for exit_type in ['stop_loss', 'overbought', 'death_cross']:
            if exit_type in trades_df['type'].values:
                type_trades = trades_df[trades_df['type'] == exit_type]
                avg_pnl = type_trades['pnl'].mean()
                total_pnl = type_trades['pnl'].sum()
                print(f"  {type_map.get(exit_type, exit_type):<15} {len(type_trades):>6} {avg_pnl:>11.0f} {total_pnl:>13.0f}")

        print("\n【持仓时间分析】")
        avg_hold = trades_df['hold_days'].mean()
        print(f"  平均持仓天数: {avg_hold:.1f}天")

print("\n" + "=" * 100)
