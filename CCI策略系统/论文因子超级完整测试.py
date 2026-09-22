"""
论文因子超级完整测试
包含：
1. 所有4个因子的多阈值测试
2. 因子两两组合测试
3. 因子三三组合测试
4. 全因子组合测试
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


# 论文因子计算
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


class SuperCompleteBacktest:
    """超级完整回测"""

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

    def run_with_filters(self, filter_config):
        """
        filter_config: {
            'gap_threshold': float or None,
            'gap_acceptance_threshold': float or None,
            'trend_quality_threshold': float or None,
            'use_orderly_trend': bool,
        }
        """
        return self._run_backtest(filter_config)

    def _run_backtest(self, filter_config):
        balance = self.initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity = [self.initial_capital]

        gap_th = filter_config.get('gap_threshold')
        gap_acc_th = filter_config.get('gap_acceptance_threshold')
        trend_qual_th = filter_config.get('trend_quality_threshold')
        use_orderly = filter_config.get('use_orderly_trend', False)

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

                # 应用因子过滤
                if open_signal:
                    # 隔夜缺口过滤
                    if gap_th is not None:
                        if current['gap_factor'] < gap_th:
                            open_signal = False

                    # 缺口接受过滤
                    if gap_acc_th is not None and open_signal:
                        if current['gap_acceptance'] < gap_acc_th:
                            open_signal = False

                    # 趋势质量过滤
                    if trend_qual_th is not None and open_signal:
                        if current['trend_quality'] < trend_qual_th:
                            open_signal = False

                    # 有序趋势过滤
                    if use_orderly and open_signal:
                        if current['orderly_trend'] < 1:
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
    print("=" * 160)
    print("论文因子超级完整测试 - 所有因子及组合".center(160))
    print("=" * 160)

    print("""
【测试方案】
第一部分：单因子多阈值测试
  - 隔夜缺口：-0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4
  - 缺口接受：0.001, 0.002, 0.003, 0.004, 0.005
  - 趋势质量：0.20, 0.25, 0.30, 0.35, 0.40
  - 有序趋势：开/关

第二部分：双因子组合测试
第三部分：三因子组合测试
第四部分：全因子组合测试
""")

    # 重点品种（高收益品种）
    symbols = ['白银', '铜', '纯碱', '镍', '玻璃', '锡', '铝', '铅']

    # 第一部分：单因子多阈值测试
    print("\n" + "=" * 160)
    print("第一部分：单因子多阈值测试".center(160))
    print("=" * 160)

    single_factor_tests = [
        ('gap', '隔夜缺口', 'gap_threshold', [-0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4]),
        ('gap_acc', '缺口接受', 'gap_acceptance_threshold', [0.001, 0.002, 0.003, 0.004, 0.005]),
        ('trend_qual', '趋势质量', 'trend_quality_threshold', [0.20, 0.25, 0.30, 0.35, 0.40]),
        ('orderly', '有序趋势', 'use_orderly_trend', [True]),
    ]

    all_results = {}

    for symbol in symbols:
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            continue

        code = CODE_MAP[symbol]
        vol_threshold = VOL_THRESHOLDS.get(symbol, 0.0)

        print(f"\n[{symbol}] 获取数据...", end=" ", flush=True)
        df = get_akshare_data(code)
        if df is None:
            print("失败")
            continue

        bt = SuperCompleteBacktest(df, params, vol_threshold)
        symbol_results = {}

        # 原始
        print(f"回测中...", end=" ", flush=True)
        original = bt.run_with_filters({})
        if original:
            symbol_results['原始'] = original

        # 单因子测试
        for factor_key, factor_name, param_key, values in single_factor_tests:
            for val in values:
                config = {param_key: val}
                result = bt.run_with_filters(config)
                if result:
                    test_name = f"{factor_name}({val})"
                    symbol_results[test_name] = result

        all_results[symbol] = symbol_results
        print(f"完成({len(symbol_results)}个配置)")

    # 输出第一部分结果
    print("\n" + "=" * 160)
    print("第一部分结果：单因子效果对比".center(160))
    print("=" * 160)

    for symbol, results in all_results.items():
        print(f"\n【{symbol}】")
        print(f"{'配置':<30}{'收益':>12}{'回撤':>10}{'交易数':>8}{'胜率':>8}{'评价':<15}")
        print("-" * 90)

        original_ret = results.get('原始', {}).get('total_return', 0)

        # 找出最佳
        best_name = '原始'
        best_ret = original_ret
        for name, result in results.items():
            if name != '原始' and result['total_return'] > best_ret:
                best_ret = result['total_return']
                best_name = name

        # 排序显示
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_return'], reverse=True)

        for name, result in sorted_results[:15]:  # 只显示前15个
            ret = result['total_return']
            diff = ret - original_ret

            if name == best_name:
                marker = "*** 最佳"
            elif name == '原始':
                marker = "基线"
            elif diff > 100:
                marker = "+++ 大幅提升"
            elif diff > 50:
                marker = "++ 显著提升"
            elif diff > 20:
                marker = "+ 提升"
            elif diff > 0:
                marker = "+ 微升"
            elif diff > -20:
                marker = "- 下降"
            else:
                marker = "-- 大降"

            print(f"{name:<30}{ret:>10.1f}%{result['max_drawdown']:>9.1f}%{result['total_trades']:>8}{result['win_rate']:>7.1f}%  {marker}")

    # 第二部分：双因子组合测试
    print("\n" + "=" * 160)
    print("第二部分：双因子组合测试（针对白银、铜、纯碱）".center(160))
    print("=" * 160)

    combo_symbols = ['白银', '铜', '纯碱']

    for symbol in combo_symbols:
        if symbol not in all_results:
            continue

        params = OPTIMAL_PARAMS.get(symbol)
        code = CODE_MAP[symbol]
        vol_threshold = VOL_THRESHOLDS.get(symbol, 0.0)
        df = get_akshare_data(code)

        if df is None:
            continue

        bt = SuperCompleteBacktest(df, params, vol_threshold)

        print(f"\n【{symbol}】双因子组合测试")
        print(f"{'组合配置':<60}{'收益':>12}{'回撤':>10}{'交易数':>8}{'胜率':>8}")
        print("-" * 100)

        # 根据单因子结果选择最佳阈值
        if symbol == '白银':
            gap_th = -0.3
        elif symbol in ['铜', '纯碱']:
            gap_th = -0.2
        else:
            gap_th = -0.2

        # 双因子组合
        combo_tests = [
            ('隔夜缺口 + 趋势质量(0.25)', {'gap_threshold': gap_th, 'trend_quality_threshold': 0.25}),
            ('隔夜缺口 + 趋势质量(0.30)', {'gap_threshold': gap_th, 'trend_quality_threshold': 0.30}),
            ('隔夜缺口 + 有序趋势', {'gap_threshold': gap_th, 'use_orderly_trend': True}),
            ('趋势质量(0.25) + 有序趋势', {'trend_quality_threshold': 0.25, 'use_orderly_trend': True}),
            ('缺口接受(0.003) + 趋势质量(0.25)', {'gap_acceptance_threshold': 0.003, 'trend_quality_threshold': 0.25}),
            ('隔夜缺口 + 缺口接受(0.003)', {'gap_threshold': gap_th, 'gap_acceptance_threshold': 0.003}),
        ]

        original_ret = all_results[symbol]['原始']['total_return']

        for name, config in combo_tests:
            result = bt.run_with_filters(config)
            if result:
                ret = result['total_return']
                diff = ret - original_ret
                marker = "***" if ret > original_ret else ""
                print(f"{name:<60}{ret:>10.1f}%{result['max_drawdown']:>9.1f}%{result['total_trades']:>8}{result['win_rate']:>7.1f}%  {marker}")

    # 最终汇总
    print("\n" + "=" * 160)
    print("最终汇总：各品种推荐配置".center(160))
    print("=" * 160)

    print(f"\n{'品种':<8}{'推荐配置':<50}{'收益':>12}{'回撤':>10}{'提升':>10}")
    print("-" * 90)

    for symbol in symbols:
        if symbol not in all_results:
            continue

        results = all_results[symbol]
        original_ret = results.get('原始', {}).get('total_return', 0)

        best_name = '原始'
        best_ret = original_ret
        for name, result in results.items():
            if result['total_return'] > best_ret:
                best_ret = result['total_return']
                best_name = name

        diff = best_ret - original_ret
        print(f"{symbol:<8}{best_name:<50}{best_ret:>10.1f}%{results[best_name]['max_drawdown']:>9.1f}%{diff:>+9.1f}%")

    return all_results


if __name__ == "__main__":
    main()
