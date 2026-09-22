"""
论文因子补充测试 - 糖、锌、棉花、黄金
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from 最优参数配置 import OPTIMAL_PARAMS

CODE_MAP = {
    '糖': 'sr0', '锌': 'zn0', '棉花': 'cf0', '黄金': 'au0'
}

VOL_THRESHOLDS = {
    '糖': 0.70, '锌': 0.60, '棉花': 0.90, '黄金': 0.80,
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


def calc_overnight_gap_factor(df, lookback=10):
    gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    high_max = df['high'].rolling(lookback).max()
    low_min = df['low'].rolling(lookback).min()
    true_range = high_max - low_min
    normalized_gap = gap / (true_range / df['close'])
    return normalized_gap


def calc_gap_acceptance_factor(df, lookback=20):
    gap = df['open'] - df['close'].shift(1)
    gap_direction = gap > 0
    intraday_direction = df['close'] > df['open']
    accepted = (gap_direction == intraday_direction).astype(int)
    acceptance_score = accepted.rolling(lookback).mean()
    gap_size = abs(gap) / df['close']
    weighted_score = acceptance_score * gap_size.rolling(lookback).mean()
    return weighted_score


def calc_trend_quality_factor(df, trend_lookback=10, vol_lookback=60):
    direction = (df['close'].diff() > 0).astype(int)
    direction_consistency = direction.rolling(trend_lookback).mean()

    atr = (df['high'] - df['low']).rolling(14).mean()
    atr_ma = atr.rolling(vol_lookback).mean()
    atr_ratio = atr / atr_ma
    volatility_quality = 1 / (atr_ratio + 0.1)

    vol_ratio = df['volume'] / df['volume'].rolling(5).mean()
    liquidity_quality = np.minimum(vol_ratio, 2) / 2

    trend_quality = direction_consistency * volatility_quality * liquidity_quality
    return trend_quality


def calc_orderly_trend_factor(df, price_lookback=10, vol_lookback=5):
    price_change = df['close'].pct_change(price_lookback)
    vol_change = df['volume'].rolling(vol_lookback).sum() / df['volume'].rolling(vol_lookback * 4).sum()
    orderly_up = (price_change > 0) & (vol_change < 1.5) & (vol_change > 0.5)
    absorption = (vol_change > 1.2) & (abs(price_change) < 0.02)
    factor = orderly_up.astype(int) * 2 + absorption.astype(int)
    return factor


class CompleteFactorBacktest:
    """完整因子回测"""

    def __init__(self, df, params, vol_threshold=0.0):
        self.df = df.copy()
        self.params = params
        self.vol_threshold = vol_threshold
        self.commission_rate = 0.0003
        self.slippage_rate = 0.0002
        self.multiplier = params.get('multiplier', 5)
        self.initial_capital = 100000
        self._prepare_indicators()

    def _prepare_indicators(self):
        cci_length = self.params['cci_length']
        ma_length = self.params['ma_length']

        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma = tp.rolling(window=cci_length).mean()
        mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
        self.df['cci'] = (tp - sma) / (0.015 * mad)
        self.df['cci_ma'] = self.df['cci'].rolling(window=ma_length).mean()
        self.df['stc'] = calculate_stc(self.df['close'])
        self.df['vol_ma5'] = self.df['volume'].rolling(5).mean()
        self.df['vol_ratio'] = self.df['volume'] / self.df['vol_ma5']

        self.df['gap_factor'] = calc_overnight_gap_factor(self.df)
        self.df['gap_acceptance'] = calc_gap_acceptance_factor(self.df)
        self.df['trend_quality'] = calc_trend_quality_factor(self.df)
        self.df['orderly_trend'] = calc_orderly_trend_factor(self.df)

    def run_with_factor(self, factor_name, factor_threshold=None):
        """使用指定因子回测"""
        return self._run_backtest(factor_name=factor_name, factor_threshold=factor_threshold)

    def _run_backtest(self, factor_name=None, factor_threshold=None):
        balance = self.initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity = [self.initial_capital]

        for i in range(100, len(self.df) - 1):
            current = self.df.iloc[i]
            next_day = self.df.iloc[i + 1]
            prev = self.df.iloc[i - 1]

            cci = current['cci']
            cci_ma = current['cci_ma']
            cci_prev = prev['cci']
            cci_ma_prev = prev['cci_ma']
            stc = current['stc']
            vol_ratio = current['vol_ratio']

            # 持仓管理
            if position > 0:
                if next_day['low'] <= entry_price * 0.96:
                    stop_price = max(next_day['open'], entry_price * 0.96)
                    exit_price = stop_price * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    balance += pnl - commission
                    trades.append({'pnl': pnl - commission})
                    position = 0
                    continue

                if next_day['high'] >= entry_price * 1.20:
                    exit_price = min(next_day['open'], entry_price * 1.20) * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    balance += pnl - commission
                    trades.append({'pnl': pnl - commission})
                    position = 0
                    continue

                if cci > self.params['cci_overbought']:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    balance += pnl - commission
                    trades.append({'pnl': pnl - commission})
                    position = 0
                    continue

                if cci_prev >= cci_ma_prev and cci < cci_ma:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    balance += pnl - commission
                    trades.append({'pnl': pnl - commission})
                    position = 0
                    continue

            # 开仓信号
            if position == 0:
                open_signal = False

                if cci < self.params['cci_oversold'] and stc >= self.params['stc_oversold']:
                    open_signal = True
                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max'] and stc >= self.params['stc_cross']):
                    open_signal = True

                # 论文因子过滤
                if open_signal and factor_name:
                    factor_value = current[factor_name]

                    if factor_name == 'gap_factor':
                        if factor_threshold and factor_value < factor_threshold:
                            open_signal = False

                    elif factor_name == 'gap_acceptance':
                        if factor_threshold and factor_value < factor_threshold:
                            open_signal = False

                    elif factor_name == 'trend_quality':
                        if factor_threshold and factor_value < factor_threshold:
                            open_signal = False

                    elif factor_name == 'orderly_trend':
                        if factor_value < 1:
                            open_signal = False

                # 量能过滤
                if open_signal and self.vol_threshold > 0:
                    if not np.isnan(vol_ratio) and vol_ratio < self.vol_threshold:
                        open_signal = False

                # 开仓
                if open_signal:
                    entry_price = next_day['open'] * (1 + self.slippage_rate)
                    max_value = balance * 0.9 * 2
                    qty = max(1, int(max_value / (entry_price * self.multiplier)))
                    commission = entry_price * qty * self.multiplier * self.commission_rate
                    balance -= commission
                    position = qty

            # 净值计算
            val = balance
            if position > 0:
                val += (next_day['close'] - entry_price) * position * self.multiplier
            equity.append(val)

        if len(trades) < 5:
            return None

        trades_df = pd.DataFrame(trades)
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100
        equity_series = pd.Series(equity)
        running_max = equity_series.expanding().max()
        max_drawdown = ((equity_series - running_max) / running_max * 100).min()

        winning = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
        }


def main():
    print("=" * 140)
    print("论文因子补充测试 - 糖、锌、棉花、黄金".center(140))
    print("=" * 140)

    symbols = ['糖', '锌', '棉花', '黄金']

    # 测试配置
    test_configs = [
        ('原始', None, None),
        ('隔夜缺口(-0.1)', 'gap_factor', -0.1),
        ('隔夜缺口(-0.15)', 'gap_factor', -0.15),
        ('隔夜缺口(-0.2)', 'gap_factor', -0.2),
        ('隔夜缺口(-0.25)', 'gap_factor', -0.25),
        ('隔夜缺口(-0.3)', 'gap_factor', -0.3),
        ('缺口接受(0.001)', 'gap_acceptance', 0.001),
        ('缺口接受(0.002)', 'gap_acceptance', 0.002),
        ('缺口接受(0.003)', 'gap_acceptance', 0.003),
        ('趋势质量(0.2)', 'trend_quality', 0.20),
        ('趋势质量(0.25)', 'trend_quality', 0.25),
        ('趋势质量(0.3)', 'trend_quality', 0.30),
        ('趋势质量(0.35)', 'trend_quality', 0.35),
        ('趋势质量(0.4)', 'trend_quality', 0.40),
        ('有序趋势', 'orderly_trend', None),
    ]

    all_results = {}

    for symbol in symbols:
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            print(f"\n[{symbol}] 无配置")
            continue

        code = CODE_MAP[symbol]
        vol_threshold = VOL_THRESHOLDS.get(symbol, 0.0)

        print(f"\n[{symbol}] 获取数据...", end=" ", flush=True)
        df = get_akshare_data(code)
        if df is None:
            print("失败")
            continue

        bt = CompleteFactorBacktest(df, params, vol_threshold)

        results = {}
        for name, factor, threshold in test_configs:
            result = bt.run_with_factor(factor, threshold)
            if result:
                results[name] = result

        if results:
            original_ret = results.get('原始', {}).get('total_return', 0)
            best_name = max(results.keys(), key=lambda x: results[x]['total_return'])
            best_ret = results[best_name]['total_return']
            print(f"原始={original_ret:.1f}% | 最佳={best_name}({best_ret:.1f}%)")
            all_results[symbol] = results
        else:
            print("无有效结果")

    # 汇总
    print("\n" + "=" * 140)
    print("补充品种因子效果对比".center(140))
    print("=" * 140)

    print(f"\n{'品种':<8}", end="")
    for name, _, _ in test_configs:
        print(f"{name:>18}", end="")
    print()
    print("-" * 140)

    for symbol, results in all_results.items():
        print(f"{symbol:<8}", end="")
        original_ret = results.get('原始', {}).get('total_return', 0)
        for name, _, _ in test_configs:
            if name in results:
                ret = results[name]['total_return']
                diff = ret - original_ret
                if ret > original_ret and diff > 20:
                    print(f"{ret:>16.1f}%++", end="")
                elif ret > original_ret and diff > 0:
                    print(f"{ret:>16.1f}+", end="")
                elif ret < original_ret and diff < -20:
                    print(f"{ret:>16.1f}%--", end="")
                else:
                    print(f"{ret:>16.1f}%", end="")
            else:
                print(f"{'N/A':>18}", end="")
        print()

    # 最终汇总
    print("\n" + "=" * 140)
    print("补充品种推荐配置".center(140))
    print("=" * 140)

    print(f"\n{'品种':<8}{'推荐配置':<25}{'收益':>12}{'回撤':>10}{'交易数':>8}{'胜率':>8}")
    print("-" * 75)

    for symbol, results in all_results.items():
        best_name = max(results.keys(), key=lambda x: results[x]['total_return'])
        best = results[best_name]
        original_ret = results.get('原始', {}).get('total_return', 0)
        diff = best['total_return'] - original_ret

        print(f"{symbol:<8}{best_name:<25}{best['total_return']:>10.1f}%{best['max_drawdown']:>9.1f}%{best['total_trades']:>8}{best['win_rate']:>7.1f}% ({diff:+.1f}%)")

    return all_results


if __name__ == "__main__":
    main()
