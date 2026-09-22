"""
糖和玻璃参数优化 - 提高稳定性
针对不稳定的两个品种进行参数网格搜索
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from itertools import product

# 品种代码映射
CODE_MAP = {
    '糖': 'sr0',
    '玻璃': 'fg0',
}


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


class Backtest:
    """回测引擎"""

    def __init__(self, df, params):
        self.df = df.copy()
        self.params = params
        self.commission_rate = 0.0003
        self.slippage_rate = 0.0002
        self.multiplier = params.get('multiplier', 5)
        self.initial_capital = 100000
        self.leverage = 2
        self._calculate_indicators()

    def _calculate_indicators(self):
        cci_length = self.params['cci_length']
        ma_length = self.params['ma_length']

        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma = tp.rolling(window=cci_length).mean()
        mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
        self.df['cci'] = (tp - sma) / (0.015 * mad)
        self.df['cci_ma'] = self.df['cci'].rolling(window=ma_length).mean()

        self.df['stc'] = calculate_stc(self.df['close'])

    def run(self):
        balance = self.initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity = []

        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]
            next_day = self.df.iloc[i + 1]

            cci = current['cci']
            cci_ma = current['cci_ma']
            stc = current['stc']

            cci_prev = self.df.iloc[i-1]['cci'] if i > 0 else cci
            cci_ma_prev = self.df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

            if position > 0:
                if next_day['low'] <= entry_price * 0.96:
                    stop_price = max(next_day['open'], entry_price * 0.96)
                    exit_price = stop_price * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl})
                    position = 0
                    entry_price = 0.0
                    continue

                if next_day['high'] >= entry_price * 1.20:
                    take_profit_price = min(next_day['open'], entry_price * 1.20)
                    exit_price = take_profit_price * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl})
                    position = 0
                    entry_price = 0.0
                    continue

                if cci > self.params['cci_overbought']:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl})
                    position = 0
                    entry_price = 0.0
                    continue

                if cci_prev >= cci_ma_prev and cci < cci_ma:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl})
                    position = 0
                    entry_price = 0.0
                    continue

            if position == 0:
                open_signal = False

                if (cci < self.params['cci_oversold'] and
                    stc >= self.params['stc_oversold']):
                    open_signal = True

                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max'] and
                      stc >= self.params['stc_cross']):
                    open_signal = True

                if open_signal:
                    entry_price = next_day['open'] * (1 + self.slippage_rate)
                    max_value = balance * 0.9 * self.leverage
                    qty = int(max_value / (entry_price * self.multiplier))
                    qty = max(1, qty)
                    commission = entry_price * qty * self.multiplier * self.commission_rate
                    balance -= commission
                    position = qty

            val = balance
            if position > 0:
                unrealized_pnl = (next_day['close'] - entry_price) * position * self.multiplier
                val += unrealized_pnl
            equity.append(val)

        if len(trades) < 5:
            return None

        trades_df = pd.DataFrame(trades)
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100

        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]

        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        profit_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        equity_series = pd.Series(equity)
        peak = equity_series.cummax()
        dd = (equity_series - peak) / peak * 100
        max_dd = dd.min()

        return {
            'total_return': total_return,
            'max_dd': max_dd,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_ratio': profit_ratio,
        }


def split_into_periods(df, years=2):
    """将数据按年分段"""
    periods = []
    start_date = df.index[0]
    end_date = df.index[-1]

    period_start = start_date
    while period_start < end_date:
        period_end = period_start + pd.DateOffset(years=years)
        if period_end > end_date:
            period_end = end_date

        mask = (df.index >= period_start) & (df.index < period_end)
        period_df = df[mask]

        if len(period_df) > 250:
            periods.append({
                'start': period_start,
                'end': period_df.index[-1],
                'df': period_df,
            })

        period_start = period_end

    return periods


def evaluate_stability(periods, params):
    """评估参数在多个周期的稳定性"""
    results = []
    for period in periods:
        bt = Backtest(period['df'], params)
        result = bt.run()
        if result:
            results.append(result)

    if not results:
        return None

    profitable = sum(1 for r in results if r['total_return'] > 0)
    avg_return = np.mean([r['total_return'] for r in results])
    std_return = np.std([r['total_return'] for r in results])
    min_return = min(r['total_return'] for r in results)

    # 稳定性得分：盈利周期比例 - 收益标准差/100
    stability_score = profitable / len(results) * 100 - std_return / 10

    return {
        'profitable_periods': profitable,
        'total_periods': len(results),
        'win_rate': profitable / len(results) * 100,
        'avg_return': avg_return,
        'std_return': std_return,
        'min_return': min_return,
        'stability_score': stability_score,
    }


def optimize_params(symbol, df):
    """参数网格搜索优化"""
    print(f"\n{'='*100}")
    print(f"[{symbol}] 参数优化".center(100))
    print("="*100)

    # 按两年分段
    periods = split_into_periods(df, years=2)
    print(f"数据分为 {len(periods)} 个周期进行稳定性测试")

    # 参数网格
    cci_lengths = [10, 12, 14, 16, 18, 20]
    ma_lengths = [5, 8, 10, 13, 15, 20]
    cci_oversolds = [-40, -50, -60, -70]
    cci_overboughts = [140, 150, 160, 170, 180, 190]
    cci_cross_maxs = [120, 130, 140, 150, 160]
    stc_oversolds = [-40, -50, -60]
    stc_crosss = [-100, -110, -120, -130]

    # 固定multiplier
    multiplier = 10 if symbol == '糖' else 20

    best_result = None
    best_params = None
    best_score = -999

    total = len(cci_lengths) * len(ma_lengths) * len(cci_oversolds) * len(cci_overboughts) * \
            len(cci_cross_maxs) * len(stc_oversolds) * len(stc_crosss)

    print(f"共 {total} 种参数组合，开始搜索...")

    count = 0
    for cci_len, ma_len, oversold, overbought, cross_max, stc_over, stc_cr in product(
        cci_lengths, ma_lengths, cci_oversolds, cci_overboughts, cci_cross_maxs, stc_oversolds, stc_crosss
    ):
        count += 1
        if count % 1000 == 0:
            print(f"  已测试 {count}/{total} 组合...")

        params = {
            'cci_length': cci_len,
            'ma_length': ma_len,
            'cci_oversold': oversold,
            'cci_overbought': overbought,
            'cci_cross_max': cross_max,
            'stc_oversold': stc_over,
            'stc_cross': stc_cr,
            'multiplier': multiplier,
        }

        stability = evaluate_stability(periods, params)
        if stability and stability['stability_score'] > best_score:
            best_score = stability['stability_score']
            best_result = stability
            best_params = params

    print(f"\n搜索完成！共测试 {count} 组合")

    if best_params:
        print(f"\n最优参数:")
        print(f"  CCI周期: {best_params['cci_length']}")
        print(f"  MA周期: {best_params['ma_length']}")
        print(f"  CCI超卖: {best_params['cci_oversold']}")
        print(f"  CCI超买: {best_params['cci_overbought']}")
        print(f"  CCI金叉上限: {best_params['cci_cross_max']}")
        print(f"  STC超卖: {best_params['stc_oversold']}")
        print(f"  STC金叉: {best_params['stc_cross']}")

        print(f"\n稳定性表现:")
        print(f"  盈利周期: {best_result['profitable_periods']}/{best_result['total_periods']} ({best_result['win_rate']:.0f}%)")
        print(f"  平均收益: {best_result['avg_return']:+.2f}%")
        print(f"  收益标准差: {best_result['std_return']:.2f}%")
        print(f"  最差周期: {best_result['min_return']:+.2f}%")
        print(f"  稳定性得分: {best_result['stability_score']:.2f}")

        # 显示每个周期的表现
        print(f"\n各周期表现:")
        for i, period in enumerate(periods):
            bt = Backtest(period['df'], best_params)
            result = bt.run()
            if result:
                label = f"{period['start'].strftime('%Y-%m')} ~ {period['end'].strftime('%Y-%m')}"
                effect = "盈利" if result['total_return'] > 0 else "亏损"
                print(f"  {label}: {result['total_return']:+.2f}% ({effect})")

        return best_params, best_result

    return None, None


def main():
    print("="*120)
    print("糖和玻璃参数优化 - 提高稳定性".center(120))
    print("="*120)

    print("""
【优化目标】
提高周期胜率（目标>=70%），降低收益波动

【优化方法】
1. 网格搜索参数空间
2. 在多个2年周期上评估稳定性
3. 选择稳定性得分最高的参数
""")

    all_results = {}

    for symbol, code in CODE_MAP.items():
        df = get_akshare_data(code)
        if df is None:
            continue

        print(f"\n数据范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} ({len(df)}天)")

        best_params, best_result = optimize_params(symbol, df)

        if best_params:
            all_results[symbol] = {
                'params': best_params,
                'result': best_result
            }

    # 汇总对比
    print("\n" + "="*120)
    print("优化结果汇总".center(120))
    print("="*120)

    # 原版参数
    from 最优参数配置 import OPTIMAL_PARAMS

    print(f"\n{'品种':<8}{'原版周期胜率':<15}{'原版平均收益':<15}{'优化周期胜率':<15}{'优化平均收益':<15}{'改善':<10}")
    print("-"*85)

    for symbol in all_results:
        old_params = OPTIMAL_PARAMS.get(symbol, {})
        df = get_akshare_data(CODE_MAP[symbol])
        periods = split_into_periods(df, years=2)

        # 原版
        old_stability = evaluate_stability(periods, old_params)

        # 优化版
        new_result = all_results[symbol]['result']

        if old_stability:
            improvement = "改善" if new_result['win_rate'] > old_stability['win_rate'] else "持平"
            print(f"{symbol:<8}{old_stability['win_rate']:>10.0f}%{old_stability['avg_return']:>+12.2f}%"
                  f"{new_result['win_rate']:>12.0f}%{new_result['avg_return']:>+12.2f}%  {improvement}")

    # 生成配置代码
    print("\n" + "="*120)
    print("新参数配置".center(120))
    print("="*120)

    for symbol, data in all_results.items():
        p = data['params']
        print(f"\n'{symbol}': {{")
        print(f"    'cci_length': {p['cci_length']},")
        print(f"    'ma_length': {p['ma_length']},")
        print(f"    'cci_oversold': {p['cci_oversold']},")
        print(f"    'cci_overbought': {p['cci_overbought']},")
        print(f"    'cci_cross_max': {p['cci_cross_max']},")
        print(f"    'stc_oversold': {p['stc_oversold']},")
        print(f"    'stc_cross': {p['stc_cross']},")
        print(f"    'multiplier': {p['multiplier']},")
        print(f"}},")

    return all_results


if __name__ == "__main__":
    main()
