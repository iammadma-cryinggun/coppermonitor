"""
盲回测v3 - 全品种测试
只在极端情况下暂停交易
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from 最优参数配置 import OPTIMAL_PARAMS

CODE_MAP = {
    '铜': 'cu0', '铝': 'al0', '锌': 'zn0', '铅': 'pb0', '镍': 'ni0', '锡': 'sn0',
    '黄金': 'au0', '白银': 'ag0', '玻璃': 'fg0', '纯碱': 'sa0', '糖': 'sr0', '棉花': 'cf0'
}

VOL_THRESHOLDS = {
    '棉花': 0.90, '玻璃': 0.90, '白银': 0.65, '糖': 0.70,
    '纯碱': 0.50, '铅': 0.70, '铜': 0.50, '铝': 0.75,
    '锌': 0.60, '镍': 0.00, '锡': 0.00, '黄金': 0.80,
}

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
    if idx < 20:
        return False, ""

    returns = df['close'].pct_change()
    consecutive_down = 0
    for i in range(idx, max(0, idx-15), -1):
        if returns.iloc[i] < 0:
            consecutive_down += 1
        else:
            break
    if consecutive_down >= 10:
        return True, f"连续下跌{consecutive_down}天"

    if idx >= 19:
        drop_20d = (df['close'].iloc[idx-19] - df['close'].iloc[idx]) / df['close'].iloc[idx-19]
        if drop_20d > 0.20:
            return True, f"20天跌{drop_20d*100:.1f}%"

    if idx >= 4:
        drop_5d = (df['close'].iloc[idx-4] - df['close'].iloc[idx]) / df['close'].iloc[idx-4]
        if drop_5d > 0.15:
            return True, f"5天跌{drop_5d*100:.1f}%"

    return False, ""


def run_backtest(df, params, vol_threshold, use_extreme_filter=False):
    df = df.copy()

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

        if position > 0:
            if next_day['low'] <= entry_price * 0.96:
                stop_price = max(next_day['open'], entry_price * 0.96)
                exit_price = stop_price * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                balance += pnl - commission
                trades.append({'type': 'stop_loss', 'pnl': pnl - commission})
                position = 0
                continue

            if next_day['high'] >= entry_price * 1.20:
                exit_price = min(next_day['open'], entry_price * 1.20) * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                balance += pnl - commission
                trades.append({'type': 'take_profit', 'pnl': pnl - commission})
                position = 0
                continue

            if cci > params['cci_overbought']:
                exit_price = next_day['close'] * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                balance += pnl - commission
                trades.append({'type': 'overbought', 'pnl': pnl - commission})
                position = 0
                continue

            if cci_prev >= cci_ma_prev and cci < cci_ma:
                exit_price = next_day['close'] * (1 - slippage_rate)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * commission_rate * 2
                balance += pnl - commission
                trades.append({'type': 'death_cross', 'pnl': pnl - commission})
                position = 0
                continue

        if position == 0:
            open_signal = False

            if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
                open_signal = True
            elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                  cci <= params['cci_cross_max'] and stc >= params['stc_cross']):
                open_signal = True

            if open_signal:
                total_signals += 1

                if vol_threshold > 0 and not np.isnan(vol_ratio) and vol_ratio < vol_threshold:
                    filtered_signals += 1
                    filter_reasons['量能不足'] = filter_reasons.get('量能不足', 0) + 1
                    continue

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
    print("盲回测v3 - 全品种测试（只暂停极端情况）".center(100))
    print("=" * 100)

    results = []
    good_results = []  # 收益提升的品种

    symbols = ['铜', '白银', '纯碱', '糖', '黄金', '镍', '锡', '玻璃', '锌', '棉花', '铝', '铅']

    for symbol in symbols:
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            continue

        code = CODE_MAP[symbol]
        vol_threshold = VOL_THRESHOLDS.get(symbol, 0.0)

        print(f"\n[{symbol}]...", end=" ", flush=True)
        df = get_akshare_data(code)
        if df is None:
            print("数据获取失败")
            continue

        original = run_backtest(df, params, vol_threshold, use_extreme_filter=False)
        v3 = run_backtest(df, params, vol_threshold, use_extreme_filter=True)

        if original and v3:
            ret_diff = v3['total_return'] - original['total_return']
            dd_improve = original['max_drawdown'] - v3['max_drawdown']  # 正数=改善
            filter_rate = v3['filtered_signals']/v3['total_signals']*100 if v3['total_signals'] > 0 else 0

            status = "[UP]" if ret_diff > 0 else "[DN]"
            print(f"原始{original['total_return']:.1f}% -> v3={v3['total_return']:.1f}% ({ret_diff:+.1f}%) {status}")

            results.append({
                'symbol': symbol,
                'orig_ret': original['total_return'],
                'v3_ret': v3['total_return'],
                'ret_diff': ret_diff,
                'orig_dd': original['max_drawdown'],
                'v3_dd': v3['max_drawdown'],
                'dd_improve': dd_improve,
                'orig_trades': original['total_trades'],
                'v3_trades': v3['total_trades'],
                'win_rate': v3['win_rate'],
                'filter_rate': filter_rate,
                'filter_reasons': v3['filter_reasons'],
            })

            if ret_diff > 0:
                good_results.append(results[-1])

    # 汇总
    print("\n" + "=" * 100)
    print("全品种汇总".center(100))
    print("=" * 100)

    print(f"\n{'品种':<6}{'原始收益':>10}{'v3收益':>10}{'收益变化':>10}{'原始回撤':>10}{'v3回撤':>10}{'回撤改善':>10}{'过滤率':>8}{'评价':<8}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: -x['ret_diff']):
        status = "[REC]" if r['ret_diff'] > 0 else ""
        print(f"{r['symbol']:<6}{r['orig_ret']:>9.1f}%{r['v3_ret']:>9.1f}%{r['ret_diff']:>+9.1f}%{r['orig_dd']:>9.1f}%{r['v3_dd']:>9.1f}%{r['dd_improve']:>+9.1f}%{r['filter_rate']:>7.1f}% {status}")

    # 统计
    improved = [r for r in results if r['ret_diff'] > 0]
    avg_ret_diff = np.mean([r['ret_diff'] for r in results])
    avg_dd_improve = np.mean([r['dd_improve'] for r in results])

    print("-" * 90)
    print(f"平均{'':<6}{'':>10}{'':>10}{avg_ret_diff:>+9.1f}%{'':>10}{'':>10}{avg_dd_improve:>+9.1f}%")
    print(f"\n收益提升品种: {len(improved)}/{len(results)}")

    # 推荐品种详情
    if improved:
        print("\n" + "=" * 100)
        print("推荐使用v3极端过滤的品种".center(100))
        print("=" * 100)

        for r in sorted(improved, key=lambda x: -x['ret_diff']):
            print(f"\n【{r['symbol']}】")
            print(f"  收益: {r['orig_ret']:.1f}% -> {r['v3_ret']:.1f}% ({r['ret_diff']:+.1f}%)")
            print(f"  回撤: {r['orig_dd']:.1f}% -> {r['v3_dd']:.1f}% ({r['dd_improve']:+.1f}%)")
            print(f"  交易: {r['orig_trades']} -> {r['v3_trades']}笔")
            print(f"  胜率: {r['win_rate']:.1f}%")
            print(f"  信号过滤: {r['filter_rate']:.1f}%")
            if r['filter_reasons']:
                reasons = sorted(r['filter_reasons'].items(), key=lambda x: -x[1])
                print(f"  过滤原因: {dict(reasons[:3])}")

    # 不推荐品种
    not_recommended = [r for r in results if r['ret_diff'] <= 0]
    if not_recommended:
        print("\n" + "=" * 100)
        print("不建议使用v3的品种（保持原始策略）".center(100))
        print("=" * 100)

        for r in sorted(not_recommended, key=lambda x: x['ret_diff']):
            print(f"  {r['symbol']}: {r['orig_ret']:.1f}% -> {r['v3_ret']:.1f}% ({r['ret_diff']:+.1f}%)")

    return results


if __name__ == "__main__":
    main()
