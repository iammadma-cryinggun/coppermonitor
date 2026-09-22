"""
量能专属参数优化 - 完整版（CCI+STC双指标）
使用最优参数配置中的所有参数，不丢失任何数据
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from 最优参数配置 import OPTIMAL_PARAMS

# 品种代码映射
CODE_MAP = {
    '铜': 'cu0', '铝': 'al0', '锌': 'zn0', '铅': 'pb0', '镍': 'ni0', '锡': 'sn0',
    '黄金': 'au0', '白银': 'ag0', '玻璃': 'fg0', '纯碱': 'sa0', '糖': 'sr0', '棉花': 'cf0'
}


def get_full_data(code):
    """获取完整历史数据"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def calculate_stc(close_prices, fast_period=23, slow_period=50, cycle_period=10):
    """计算STC指标"""
    # MACD
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    # Schaff Trend Cycle
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

    return stc


class CompleteBacktest:
    """完整回测引擎（CCI+STC双指标）"""

    def __init__(self, df, params, min_vol_ratio=0.0):
        self.df = df.copy()
        self.params = params
        self.min_vol_ratio = min_vol_ratio
        self.commission_rate = 0.0003
        self.slippage_rate = 0.0002
        self.multiplier = params.get('multiplier', 5)
        self.initial_capital = 100000
        self._calculate_indicators()

    def _calculate_indicators(self):
        cci_length = self.params['cci_length']
        ma_length = self.params['ma_length']

        # CCI
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma = tp.rolling(window=cci_length).mean()
        mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
        self.df['cci'] = (tp - sma) / (0.015 * mad)
        self.df['cci_ma'] = self.df['cci'].rolling(window=ma_length).mean()

        # STC
        self.df['stc'] = calculate_stc(self.df['close'])

        # 量比
        self.df['vol_ma5'] = self.df['volume'].rolling(5).mean()
        self.df['vol_ratio'] = self.df['volume'] / self.df['vol_ma5']

    def run(self):
        balance = self.initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity = []
        skipped_low_vol = 0

        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]
            next_day = self.df.iloc[i + 1]

            cci = current['cci']
            cci_ma = current['cci_ma']
            stc = current['stc']
            vol_ratio = current['vol_ratio']

            cci_prev = self.df.iloc[i-1]['cci'] if i > 0 else cci
            cci_ma_prev = self.df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

            if position > 0:
                # 止损（4%）
                if next_day['low'] <= entry_price * 0.96:
                    stop_price = max(next_day['open'], entry_price * 0.96)
                    exit_price = stop_price * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'stop_loss'})
                    position = 0
                    entry_price = 0.0
                    continue

                # 止盈（20%）
                if next_day['high'] >= entry_price * 1.20:
                    take_profit_price = min(next_day['open'], entry_price * 1.20)
                    exit_price = take_profit_price * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'take_profit'})
                    position = 0
                    entry_price = 0.0
                    continue

                # CCI超买平仓
                if cci > self.params['cci_overbought']:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'overbought'})
                    position = 0
                    entry_price = 0.0
                    continue

                # CCI死叉平仓
                if cci_prev >= cci_ma_prev and cci < cci_ma:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'death_cross'})
                    position = 0
                    entry_price = 0.0
                    continue

            # 开仓
            if position == 0:
                open_signal = False

                # 超卖开仓（CCI + STC 双条件）
                if (cci < self.params['cci_oversold'] and
                    stc >= self.params['stc_oversold']):
                    open_signal = True

                # 金叉开仓（CCI + STC 双条件）
                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max'] and
                      stc >= self.params['stc_cross']):
                    open_signal = True

                if open_signal:
                    # 量能过滤
                    if self.min_vol_ratio > 0 and not np.isnan(vol_ratio) and vol_ratio < self.min_vol_ratio:
                        skipped_low_vol += 1
                    else:
                        entry_price = next_day['open'] * (1 + self.slippage_rate)
                        max_value = balance * 0.9 * 2
                        qty = int(max_value / (entry_price * self.multiplier))
                        qty = max(1, qty)
                        commission = entry_price * qty * self.multiplier * self.commission_rate
                        balance -= commission
                        position = qty

            # 更新权益
            val = balance
            if position > 0:
                unrealized_pnl = (next_day['close'] - entry_price) * position * self.multiplier
                val += unrealized_pnl
            equity.append(val)

        if len(trades) < 10:
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

        total_profit = winning_trades['pnl'].sum()
        total_loss = abs(losing_trades['pnl'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        expectancy = (win_rate/100 * avg_win - (1-win_rate/100) * avg_loss)

        return {
            'total_return': total_return,
            'max_dd': max_dd,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_ratio': profit_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'skipped_low_vol': skipped_low_vol
        }


def main():
    print("="*120)
    print("量能专属参数优化 - 完整版（CCI+STC双指标）".center(120))
    print("="*120)

    print("""
【测试说明】
使用最优参数配置中的所有参数：
- CCI参数: cci_length, ma_length, cci_oversold, cci_overbought, cci_cross_max
- STC参数: stc_oversold, stc_cross
- 风控参数: 止损4%, 止盈20%
- 交易成本: 滑点0.02%, 手续费0.03%

【量能阈值测试范围】
0.0 = 无过滤
0.5 ~ 1.0 = 逐步提高过滤标准
""")

    test_symbols = ['黄金', '镍', '锡', '白银', '玻璃', '纯碱', '铜', '糖', '锌', '棉花', '铝', '铅']
    thresholds = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    all_optimal = {}

    for symbol in test_symbols:
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            continue

        code = CODE_MAP[symbol]
        df = get_full_data(code)
        if df is None:
            continue

        print(f"\n{'='*100}")
        print(f"[{symbol}] 配置收益: {params['expected_return']:+.2f}%".center(100))
        print("="*100)

        results = []

        for threshold in thresholds:
            bt = CompleteBacktest(df, params, min_vol_ratio=threshold)
            result = bt.run()

            if result:
                results.append({
                    'threshold': threshold,
                    'return': result['total_return'],
                    'max_dd': result['max_dd'],
                    'trades': result['total_trades'],
                    'win_rate': result['win_rate'],
                    'profit_ratio': result['profit_ratio'],
                    'profit_factor': result['profit_factor'],
                    'skipped': result['skipped_low_vol']
                })

        if not results:
            continue

        results_df = pd.DataFrame(results)

        # 找出最优
        best_by_return = results_df.loc[results_df['return'].idxmax()]
        baseline = results_df[results_df['threshold'] == 0.0].iloc[0]

        # 显示结果
        print(f"\n{'阈值':<8}{'收益率':<15}{'最大回撤':<12}{'交易次数':<10}{'胜率':<10}{'盈亏比':<10}{'盈亏因子':<10}{'评价':<15}")
        print("-"*100)

        for _, row in results_df.iterrows():
            ret_change = row['return'] - baseline['return']
            if row['threshold'] == best_by_return['threshold']:
                marker = "*** 最优"
            elif row['threshold'] == 0.0:
                marker = "基线"
            elif ret_change > 10:
                marker = "+ 大幅改善"
            elif ret_change > 0:
                marker = "+ 改善"
            elif ret_change > -20:
                marker = ""
            else:
                marker = "- 下降"

            print(f"{row['threshold']:<8.2f}{row['return']:>+12.2f}%{row['max_dd']:>10.2f}%"
                  f"{row['trades']:>10}{row['win_rate']:>8.1f}%{row['profit_ratio']:>8.2f}"
                  f"{row['profit_factor']:>9.2f}  {marker}")

        all_optimal[symbol] = {
            'best_threshold': best_by_return['threshold'],
            'best_return': best_by_return['return'],
            'best_dd': best_by_return['max_dd'],
            'baseline_return': baseline['return'],
            'baseline_dd': baseline['max_dd'],
            'config_return': params['expected_return'],
            'skipped': best_by_return['skipped'] if best_by_return['threshold'] > 0 else 0
        }

        print(f"\n最优: 阈值 {best_by_return['threshold']:.2f}, 收益 {best_by_return['return']:+.2f}%")

    # 汇总结果
    print("\n" + "="*120)
    print("汇总 - 各品种最优量能参数".center(120))
    print("="*120)

    print(f"\n{'品种':<8}{'配置收益':<12}{'回测收益':<12}{'最优阈值':<10}{'优化收益':<12}{'收益变化':<12}{'回撤变化':<12}{'效果':<10}")
    print("-"*100)

    optimal_thresholds = {}

    for symbol, data in all_optimal.items():
        ret_change = data['best_return'] - data['baseline_return']
        dd_change = data['best_dd'] - data['baseline_dd']

        if ret_change > 50:
            effect = "大幅改善"
        elif ret_change > 10:
            effect = "明显改善"
        elif ret_change > 0:
            effect = "改善"
        elif ret_change > -10:
            effect = "持平"
        else:
            effect = "下降"

        print(f"{symbol:<8}{data['config_return']:>+10.2f}%{data['baseline_return']:>+10.2f}%"
              f"{data['best_threshold']:>8.2f}{data['best_return']:>+10.2f}%"
              f"{ret_change:>+10.2f}%{dd_change:>+10.2f}%  {effect}")

        optimal_thresholds[symbol] = data['best_threshold']

    # 生成配置代码
    print("\n" + "="*120)
    print("最优配置代码".center(120))
    print("="*120)

    print("\n# 各品种最优量比阈值（基于CCI+STC完整回测）")
    print("OPTIMAL_VOLUME_THRESHOLDS = {")
    for symbol in sorted(optimal_thresholds.keys()):
        print(f"    '{symbol}': {optimal_thresholds[symbol]:.2f},")
    print("}")

    # 分类建议
    print("\n" + "="*120)
    print("分类建议".center(120))
    print("="*120)

    categories = {
        '强烈推荐': [],
        '推荐': [],
        '可选': [],
        '不推荐': []
    }

    for symbol, data in all_optimal.items():
        ret_change = data['best_return'] - data['baseline_return']
        if ret_change > 50:
            categories['强烈推荐'].append((symbol, data['best_threshold'], ret_change))
        elif ret_change > 10:
            categories['推荐'].append((symbol, data['best_threshold'], ret_change))
        elif ret_change > -10:
            categories['可选'].append((symbol, data['best_threshold'], ret_change))
        else:
            categories['不推荐'].append((symbol, data['best_threshold'], ret_change))

    for category, items in categories.items():
        if items:
            print(f"\n【{category}启用量能过滤】")
            for symbol, threshold, change in sorted(items, key=lambda x: -x[2]):
                print(f"  {symbol}: 阈值 {threshold:.2f}, 收益提升 {change:+.2f}%")

    # 数据验证
    print("\n" + "="*120)
    print("数据验证".center(120))
    print("="*120)

    print("\n对比配置文件与回测结果（阈值=0.0时）:")
    print(f"{'品种':<10}{'配置收益':<15}{'回测收益':<15}{'差异':<12}{'状态':<10}")
    print("-"*65)

    for symbol, data in all_optimal.items():
        diff = data['baseline_return'] - data['config_return']
        status = "一致" if abs(diff) < 100 else "有差异"
        print(f"{symbol:<10}{data['config_return']:>+12.2f}%{data['baseline_return']:>+12.2f}%{diff:>+10.2f}%  {status}")

    return optimal_thresholds


if __name__ == "__main__":
    main()
