"""
论文因子综合测试
包含：
1. 隔夜缺口因子 (GapZ10_Overnight_vs_TR) - 论文Rank IC: 0.0793
2. 趋势质量因子 (CleanTrend_Continuation_Score) - 论文Rank IC: 0.0590
3. 波动率自适应阈值 (ATR Adaptive Threshold)

与原有CCI+STC策略结合测试
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


def calculate_atr(df, period=14):
    """计算ATR"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


# ============================================================
# 论文因子计算
# ============================================================

def calc_overnight_gap_factor(df, lookback=10):
    """
    论文因子: GapZ10_Overnight_vs_TR
    隔夜缺口 vs 近期真实波动范围

    论文Rank IC: 0.0793 (最强因子)
    """
    # 隔夜缺口
    gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    # 近期真实波动范围
    high_max = df['high'].rolling(lookback).max()
    low_min = df['low'].rolling(lookback).min()
    true_range = high_max - low_min

    # 标准化缺口
    normalized_gap = gap / (true_range / df['close'])

    # 缺口方向信号：正缺口=看多，负缺口=看空
    gap_signal = normalized_gap

    return gap_signal


def calc_gap_acceptance_factor(df, lookback=20):
    """
    论文因子: Gap_IntradayAcceptanceScore_20D
    缺口是否被日内价格接受

    论文Rank IC: 0.0744
    """
    # 隔夜缺口方向
    gap = df['open'] - df['close'].shift(1)
    gap_direction = gap > 0

    # 日内方向
    intraday_direction = df['close'] > df['open']

    # 缺口是否被接受（方向一致）
    accepted = (gap_direction == intraday_direction).astype(int)

    # 近期接受率
    acceptance_score = accepted.rolling(lookback).mean()

    # 结合缺口大小加权
    gap_size = abs(gap) / df['close']
    weighted_score = acceptance_score * gap_size.rolling(lookback).mean()

    return weighted_score


def calc_trend_quality_factor(df, trend_lookback=10, vol_lookback=60):
    """
    论文因子: CleanTrend_Continuation_Score
    趋势质量：方向一致性 * 低波动 * 流动性

    论文Rank IC: 0.0590
    """
    # 1. 趋势方向一致性
    direction = (df['close'].diff() > 0).astype(int)
    direction_consistency = direction.rolling(trend_lookback).mean()

    # 2. 波动率因子（低波动=高质量）
    atr = calculate_atr(df, 14)
    atr_ma = atr.rolling(vol_lookback).mean()
    atr_ratio = atr / atr_ma
    volatility_quality = 1 / (atr_ratio + 0.1)  # 低波动高质量

    # 3. 流动性因子
    vol_ratio = df['volume'] / df['volume'].rolling(5).mean()
    liquidity_quality = np.minimum(vol_ratio, 2) / 2  # 限制在0-1

    # 综合趋势质量分数
    trend_quality = direction_consistency * volatility_quality * liquidity_quality

    return trend_quality


def calc_orderly_trend_factor(df, price_lookback=10, vol_lookback=5):
    """
    论文因子: OrderlyTrend_x_Absorption
    有序趋势 + 流动性吸收

    论文Rank IC: 0.0465
    """
    # 价格变化
    price_change = df['close'].pct_change(price_lookback)

    # 成交量变化
    vol_change = df['volume'].rolling(vol_lookback).sum() / df['volume'].rolling(vol_lookback * 4).sum()

    # 有序趋势：价格上涨且成交量正常
    orderly_up = (price_change > 0) & (vol_change < 1.5) & (vol_change > 0.5)

    # 流动性吸收：成交量增加但价格稳定
    absorption = (vol_change > 1.2) & (abs(price_change) < 0.02)

    factor = orderly_up.astype(int) * 2 + absorption.astype(int)

    return factor


# ============================================================
# 回测引擎
# ============================================================

class PaperFactorBacktest:
    """论文因子回测"""

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
        """计算所有指标"""
        # 原有CCI+STC指标
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

        # ATR
        self.df['atr'] = calculate_atr(self.df, 14)
        self.df['atr_ma60'] = self.df['atr'].rolling(60).mean()

        # 论文因子
        self.df['gap_factor'] = calc_overnight_gap_factor(self.df)
        self.df['gap_acceptance'] = calc_gap_acceptance_factor(self.df)
        self.df['trend_quality'] = calc_trend_quality_factor(self.df)
        self.df['orderly_trend'] = calc_orderly_trend_factor(self.df)

    def run_original(self):
        """原始策略"""
        return self._run_backtest(use_paper_factors=False, use_atr_adaptive=False, use_extreme_filter=False)

    def run_with_extreme_filter(self):
        """v3: 原始 + 极端过滤"""
        return self._run_backtest(use_paper_factors=False, use_atr_adaptive=False, use_extreme_filter=True)

    def run_with_atr_adaptive(self):
        """ATR自适应阈值"""
        return self._run_backtest(use_paper_factors=False, use_atr_adaptive=True, use_extreme_filter=False)

    def run_with_gap_factor(self):
        """隔夜缺口因子"""
        return self._run_backtest(use_paper_factors=True, paper_factor='gap', use_extreme_filter=False)

    def run_with_trend_quality(self):
        """趋势质量因子"""
        return self._run_backtest(use_paper_factors=True, paper_factor='trend_quality', use_extreme_filter=False)

    def run_combined(self):
        """组合策略：CCI + 趋势质量 + ATR自适应 + 极端过滤"""
        return self._run_backtest(use_paper_factors=True, paper_factor='combined',
                                   use_atr_adaptive=True, use_extreme_filter=True)

    def _is_extreme_condition(self, idx):
        """极端情况检测"""
        if idx < 20:
            return False

        returns = self.df['close'].pct_change()
        consecutive_down = 0
        for i in range(idx, max(0, idx - 15), -1):
            if returns.iloc[i] < 0:
                consecutive_down += 1
            else:
                break
        if consecutive_down >= 10:
            return True

        if idx >= 19:
            drop_20d = (self.df['close'].iloc[idx - 19] - self.df['close'].iloc[idx]) / self.df['close'].iloc[idx - 19]
            if drop_20d > 0.20:
                return True

        if idx >= 4:
            drop_5d = (self.df['close'].iloc[idx - 4] - self.df['close'].iloc[idx]) / self.df['close'].iloc[idx - 4]
            if drop_5d > 0.15:
                return True

        return False

    def _run_backtest(self, use_paper_factors=False, paper_factor=None,
                      use_atr_adaptive=False, use_extreme_filter=False):
        """核心回测逻辑"""
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

            # ATR自适应阈值
            if use_atr_adaptive and not np.isnan(current['atr_ma60']):
                atr_ratio = current['atr'] / current['atr_ma60']
                if atr_ratio > 1.5:
                    oversold_threshold = -150  # 高波动时更保守
                    stop_loss_pct = 0.06
                else:
                    oversold_threshold = self.params['cci_oversold']
                    stop_loss_pct = 0.04
            else:
                oversold_threshold = self.params['cci_oversold']
                stop_loss_pct = 0.04

            # 持仓管理
            if position > 0:
                # 止损
                if next_day['low'] <= entry_price * (1 - stop_loss_pct):
                    stop_price = max(next_day['open'], entry_price * (1 - stop_loss_pct))
                    exit_price = stop_price * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    balance += pnl - commission
                    trades.append({'pnl': pnl - commission})
                    position = 0
                    continue

                # 止盈
                if next_day['high'] >= entry_price * 1.20:
                    exit_price = min(next_day['open'], entry_price * 1.20) * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    balance += pnl - commission
                    trades.append({'pnl': pnl - commission})
                    position = 0
                    continue

                # 超买平仓
                if cci > self.params['cci_overbought']:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    balance += pnl - commission
                    trades.append({'pnl': pnl - commission})
                    position = 0
                    continue

                # 死叉平仓
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

                # 原始CCI信号
                if cci < oversold_threshold and stc >= self.params['stc_oversold']:
                    open_signal = True
                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max'] and stc >= self.params['stc_cross']):
                    open_signal = True

                # 论文因子过滤/增强
                if use_paper_factors and open_signal:
                    gap_factor = current['gap_factor']
                    trend_quality = current['trend_quality']
                    gap_acceptance = current['gap_acceptance']

                    if paper_factor == 'gap':
                        # 隔夜缺口：正缺口增强信号，负缺口削弱
                        if gap_factor < -0.3:  # 负缺口太大
                            open_signal = False

                    elif paper_factor == 'trend_quality':
                        # 趋势质量：低质量时不开
                        if trend_quality < 0.3:
                            open_signal = False

                    elif paper_factor == 'combined':
                        # 组合：多重过滤
                        if trend_quality < 0.25:
                            open_signal = False
                        if gap_factor < -0.5:
                            open_signal = False

                # 量能过滤
                if open_signal and self.vol_threshold > 0:
                    if not np.isnan(vol_ratio) and vol_ratio < self.vol_threshold:
                        open_signal = False

                # 极端情况过滤
                if open_signal and use_extreme_filter:
                    if self._is_extreme_condition(i):
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
    print("=" * 120)
    print("论文因子综合测试".center(120))
    print("=" * 120)

    print("""
【测试方案】
1. 原始策略 (Original)
2. v3极端过滤 (Extreme Filter)
3. ATR自适应阈值 (ATR Adaptive)
4. 隔夜缺口因子 (Gap Factor) - 论文Rank IC: 0.0793
5. 趋势质量因子 (Trend Quality) - 论文Rank IC: 0.0590
6. 组合策略 (Combined: CCI + 趋势质量 + ATR自适应 + 极端过滤)
""")

    symbols = ['铜', '白银', '纯碱', '糖', '黄金', '镍', '锡', '玻璃', '锌', '棉花', '铝', '铅']
    all_results = {}

    for symbol in symbols:
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            continue

        code = CODE_MAP[symbol]
        vol_threshold = VOL_THRESHOLDS.get(symbol, 0.0)

        print(f"\n[{symbol}]...", end=" ", flush=True)
        df = get_akshare_data(code)
        if df is None:
            print("数据失败")
            continue

        bt = PaperFactorBacktest(df, params, vol_threshold)

        results = {}
        strategies = [
            ('原始', bt.run_original),
            ('v3极端', bt.run_with_extreme_filter),
            ('ATR自适应', bt.run_with_atr_adaptive),
            ('隔夜缺口', bt.run_with_gap_factor),
            ('趋势质量', bt.run_with_trend_quality),
            ('组合', bt.run_combined),
        ]

        for name, func in strategies:
            result = func()
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
    print("\n" + "=" * 120)
    print("各策略效果对比".center(120))
    print("=" * 120)

    print(f"\n{'品种':<6}", end="")
    for name in ['原始', 'v3极端', 'ATR自适应', '隔夜缺口', '趋势质量', '组合']:
        print(f"{name:>12}", end="")
    print(f"{'最佳策略':<12}")
    print("-" * 90)

    for symbol, results in all_results.items():
        print(f"{symbol:<6}", end="")
        best_name = ""
        best_ret = -999
        for name in ['原始', 'v3极端', 'ATR自适应', '隔夜缺口', '趋势质量', '组合']:
            if name in results:
                ret = results[name]['total_return']
                print(f"{ret:>11.1f}%", end="")
                if ret > best_ret:
                    best_ret = ret
                    best_name = name
            else:
                print(f"{'N/A':>12}", end="")
        print(f"{best_name:<12}")

    # 统计各策略胜出次数
    print("\n" + "=" * 120)
    print("策略胜出统计".center(120))
    print("=" * 120)

    strategy_wins = {'原始': 0, 'v3极端': 0, 'ATR自适应': 0, '隔夜缺口': 0, '趋势质量': 0, '组合': 0}

    for symbol, results in all_results.items():
        best_name = max(results.keys(), key=lambda x: results[x]['total_return'])
        strategy_wins[best_name] += 1

    print(f"\n{'策略':<15}{'胜出品种数':>10}")
    print("-" * 30)
    for name, wins in sorted(strategy_wins.items(), key=lambda x: -x[1]):
        print(f"{name:<15}{wins:>10}")

    # 推荐配置
    print("\n" + "=" * 120)
    print("各品种推荐配置".center(120))
    print("=" * 120)

    print(f"\n{'品种':<8}{'推荐策略':<12}{'收益':>10}{'回撤':>10}{'交易数':>8}{'胜率':>8}")
    print("-" * 60)

    for symbol, results in all_results.items():
        best_name = max(results.keys(), key=lambda x: results[x]['total_return'])
        best = results[best_name]
        print(f"{symbol:<8}{best_name:<12}{best['total_return']:>9.1f}%{best['max_drawdown']:>9.1f}%{best['total_trades']:>8}{best['win_rate']:>7.1f}%")

    return all_results


if __name__ == "__main__":
    main()
