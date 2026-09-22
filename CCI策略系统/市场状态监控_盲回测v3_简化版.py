"""
盲回测v3简化版 - 只在极端情况下暂停交易
测试品种：铜、白银、纯碱、糖（代表不同特性）
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from 最优参数配置 import OPTIMAL_PARAMS

CODE_MAP = {
    '铜': 'cu0', '白银': 'ag0', '纯碱': 'sa0', '糖': 'sr0'
}

VOL_THRESHOLDS = {
    '铜': 0.50, '白银': 0.65, '纯碱': 0.50, '糖': 0.70
}

# 预加载数据缓存
DATA_CACHE = {}

def get_akshare_data(code):
    if code in DATA_CACHE:
        return DATA_CACHE[code].copy()
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()
        df.set_index('date', inplace=True)
        DATA_CACHE[code] = df.copy()
        return df
    except:
        return None


def calculate_stc(close_prices, fast_period=23, slow_period=50, cycle_period=10):
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
            low, high = window.min(), window.max()
            if high != low:
                stc_normalized.iloc[i] = (stc.iloc[i] - low) / (high - low) * 100 - 50
            else:
                stc_normalized.iloc[i] = stc_normalized.iloc[i-1] if i > 0 else 0
    return stc_normalized


def is_extreme_condition(df, idx):
    """检测极端情况"""
    if idx < 20:
        return False, ""

    # 1. 连续下跌天数
    returns = df['close'].pct_change()
    consecutive_down = 0
    for i in range(idx, max(0, idx-15), -1):
        if returns.iloc[i] < 0:
            consecutive_down += 1
        else:
            break
    if consecutive_down >= 10:
        return True, f"连续下跌{consecutive_down}天"

    # 2. 20天跌幅
    if idx >= 19:
        drop_20d = (df['close'].iloc[idx-19] - df['close'].iloc[idx]) / df['close'].iloc[idx-19]
        if drop_20d > 0.20:
            return True, f"20天跌{drop_20d*100:.1f}%"

    # 3. 5天跌幅
    if idx >= 4:
        drop_5d = (df['close'].iloc[idx-4] - df['close'].iloc[idx]) / df['close'].iloc[idx-4]
        if drop_5d > 0.15:
            return True, f"5天跌{drop_5d*100:.1f}%"

    return False, ""


def run_backtest(df, params, vol_threshold, use_extreme_filter=False):
    """运行回测"""
    df = df.copy()

    # 预计算指标
    cci_length = params['cci_length']
    ma_length = params['ma_length']
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=cci_length).mean()
    mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()
    df['stc'] = calculate_stc(df['close'])
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    balance = 100000
    position = 0
    entry_price = 0.0
    trades = []
    equity = [100000]

    commission_rate = 0.0003
    slippage_rate = 0.0002
    multiplier = params.get('multiplier', 5)

    total_signals = 0
    filtered_signals = 0
    filter_reasons = {}

    for i in range(100, len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]
        prev = df.iloc[i-1]

        cci = current['cci']
        cci_ma = current['cci_ma']
        cci_prev = prev['cci']
        cci_ma_prev = prev['cci_ma']
        stc = current['stc']
        vol_ratio = current['vol_ratio']

        # 持仓处理
        if position > 0:
            # 止损
            if next_day['low'] <= entry_price * 0.96:
                stop_price = max(next_day['open'], entry_price * 0.96)
                exit_price = stop_price * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                balance += pnl - commission
                trades.append({'type': 'stop_loss', 'pnl': pnl - commission})
                position = 0
                continue

            # 止盈
            if next_day['high'] >= entry_price * 1.20:
                exit_price = min(next_day['open'], entry_price * 1.20) * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                balance += pnl - commission
                trades.append({'type': 'take_profit', 'pnl': pnl - commission})
                position = 0
                continue

            # 超买平仓
            if cci > params['cci_overbought']:
                exit_price = next_day['close'] * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                balance += pnl - commission
                trades.append({'type': 'overbought', 'pnl': pnl - commission})
                position = 0
                continue

            # 死叉平仓
            if cci_prev >= cci_ma_prev and cci < cci_ma:
                exit_price = next_day['close'] * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                balance += pnl - commission
                trades.append({'type': 'death_cross', 'pnl': pnl - commission})
                position = 0
                continue

        # 开仓
        if position == 0:
            open_signal = False

            if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
                open_signal = True
            elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                  cci <= params['cci_cross_max'] and stc >= params['stc_cross']):
                open_signal = True

            if open_signal:
                total_signals += 1

                # 量能过滤
                if vol_threshold > 0 and not np.isnan(vol_ratio) and vol_ratio < vol_threshold:
                    filtered_signals += 1
                    filter_reasons['量能不足'] = filter_reasons.get('量能不足', 0) + 1
                    continue

                # 极端情况过滤
                if use_extreme_filter:
                    is_extreme, reason = is_extreme_condition(df, i)
                    if is_extreme:
                        filtered_signals += 1
                        filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
                        continue

                entry_price = next_day['open'] * (1 + slippage_rate)
                max_value = balance * 0.9 * 2
                qty = max(1, int(max_value / (entry_price * multiplier)))
                commission = entry_price * qty * multiplier * commission_rate
                balance -= commission
                position = qty

        # 净值
        val = balance
        if position > 0:
            val += (next_day['close'] - entry_price) * position * multiplier
        equity.append(val)

    if len(trades) < 5:
        return None

    trades_df = pd.DataFrame(trades)
    total_return = (equity[-1] - 100000) / 100000 * 100
    equity_series = pd.Series(equity)
    running_max = equity_series.expanding().max()
    max_drawdown = ((equity_series - running_max) / running_max * 100).min()

    winning = trades_df[trades_df['pnl'] > 0]
    win_rate = len(winning) / len(trades_df) * 100

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'total_signals': total_signals,
        'filtered_signals': filtered_signals,
        'filter_reasons': filter_reasons,
    }


def main():
    print("=" * 100)
    print("盲回测v3 - 只在极端情况下暂停交易".center(100))
    print("=" * 100)
    print("""
【方案】
- 不使用MA60趋势过滤
- 只在极端情况暂停：
  · 连续下跌 >= 10天
  · 20天跌幅 > 20%
  · 5天跌幅 > 15%
""")

    results = []

    for symbol in ['铜', '白银', '纯碱', '糖']:
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            continue

        code = CODE_MAP[symbol]
        vol_threshold = VOL_THRESHOLDS.get(symbol, 0.0)

        print(f"\n[{symbol}] 获取数据...")
        df = get_akshare_data(code)
        if df is None:
            continue

        print(f"  数据: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')} ({len(df)}天)")

        # 原始回测
        print(f"  原始回测...")
        original = run_backtest(df, params, vol_threshold, use_extreme_filter=False)

        # v3回测
        print(f"  v3回测...")
        v3 = run_backtest(df, params, vol_threshold, use_extreme_filter=True)

        if original and v3:
            print(f"\n  【结果对比】")
            print(f"  {'指标':<15}{'原始':>15}{'v3':>15}{'变化':>15}")
            print(f"  {'-'*60}")
            print(f"  {'收益率':.<15}{original['total_return']:>14.1f}%{v3['total_return']:>14.1f}%{v3['total_return']-original['total_return']:>+14.1f}%")
            print(f"  {'最大回撤':.<15}{original['max_drawdown']:>14.1f}%{v3['max_drawdown']:>14.1f}%{v3['max_drawdown']-original['max_drawdown']:>+14.1f}%")
            print(f"  {'交易次数':.<15}{original['total_trades']:>14}{v3['total_trades']:>14}{v3['total_trades']-original['total_trades']:>+14}")
            print(f"  {'胜率':.<15}{original['win_rate']:>13.1f}%{v3['win_rate']:>13.1f}%{v3['win_rate']-original['win_rate']:>+13.1f}%")

            print(f"\n  【信号过滤】")
            print(f"  总信号: {v3['total_signals']}")
            print(f"  被过滤: {v3['filtered_signals']} ({v3['filtered_signals']/v3['total_signals']*100:.1f}%)")
            if v3['filter_reasons']:
                for reason, count in sorted(v3['filter_reasons'].items(), key=lambda x: -x[1]):
                    print(f"    {reason}: {count}次")

            results.append({
                'symbol': symbol,
                'orig_ret': original['total_return'],
                'v3_ret': v3['total_return'],
                'ret_diff': v3['total_return'] - original['total_return'],
                'orig_dd': original['max_drawdown'],
                'v3_dd': v3['max_drawdown'],
                'dd_improve': original['max_drawdown'] - v3['max_drawdown'],
                'filter_rate': v3['filtered_signals']/v3['total_signals']*100 if v3['total_signals'] > 0 else 0,
            })

    # 汇总
    print("\n" + "=" * 100)
    print("汇总".center(100))
    print("=" * 100)

    print(f"\n{'品种':<8}{'原始收益':>12}{'v3收益':>12}{'收益变化':>12}{'原始回撤':>12}{'v3回撤':>12}{'回撤改善':>12}{'过滤率':>10}")
    print("-" * 90)

    for r in results:
        print(f"{r['symbol']:<8}{r['orig_ret']:>11.1f}%{r['v3_ret']:>11.1f}%{r['ret_diff']:>+11.1f}%{r['orig_dd']:>11.1f}%{r['v3_dd']:>11.1f}%{r['dd_improve']:>+11.1f}%{r['filter_rate']:>9.1f}%")

    if results:
        avg_ret_diff = np.mean([r['ret_diff'] for r in results])
        avg_dd_improve = np.mean([r['dd_improve'] for r in results])
        avg_filter = np.mean([r['filter_rate'] for r in results])
        print("-" * 90)
        print(f"{'平均':<8}{'':>12}{'':>12}{avg_ret_diff:>+11.1f}%{'':>12}{'':>12}{avg_dd_improve:>+11.1f}%{avg_filter:>9.1f}%")

    print(f"\n【结论】")
    improved = [r for r in results if r['dd_improve'] > 0]
    print(f"回撤改善: {len(improved)}/{len(results)} 个品种")

    return results


if __name__ == "__main__":
    main()
