"""
完整回测 + 量能优化（akshare 18年数据）
使用完整的CCI+STC双指标，测试量能优化效果
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


def get_akshare_data(code):
    """从akshare获取完整历史数据（18年）"""
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
    """计算STC指标（与原始代码一致）"""
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

    # 归一化到STC范围
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


class CompleteBacktest:
    """完整回测引擎（CCI+STC双指标 + 量能过滤）"""

    def __init__(self, df, params, min_vol_ratio=0.0):
        self.df = df.copy()
        self.params = params
        self.min_vol_ratio = min_vol_ratio
        self.commission_rate = 0.0003
        self.slippage_rate = 0.0002
        self.multiplier = params.get('multiplier', 5)
        self.initial_capital = 100000
        self.leverage = 2
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
                        max_value = balance * 0.9 * self.leverage
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
    print("完整回测 + 量能优化（akshare 18年数据）".center(120))
    print("="*120)

    print("""
【数据源】akshare - 2008年至2026年（约18年历史）
【指标】CCI + STC 双指标
【成本】滑点0.02% + 手续费0.03%
【风控】止损4%, 止盈20%

【量能阈值测试】0.0 ~ 1.0（0=不过滤，1=必须放量）
""")

    test_symbols = ['黄金', '镍', '锡', '白银', '玻璃', '纯碱', '铜', '糖', '锌', '棉花', '铝', '铅']
    thresholds = [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    all_optimal = {}
    all_data = {}

    # 先获取所有数据
    print("正在获取数据...")
    for symbol in test_symbols:
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            continue
        code = CODE_MAP[symbol]
        df = get_akshare_data(code)
        if df is not None:
            all_data[symbol] = df
            print(f"  {symbol}: {len(df)}行, {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

    print("\n开始回测...")

    for symbol in test_symbols:
        if symbol not in all_data:
            continue

        params = OPTIMAL_PARAMS[symbol]
        df = all_data[symbol]

        print(f"\n{'='*100}")
        print(f"[{symbol}] 数据: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} ({len(df)}天)".center(100))
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
        print(f"\n{'阈值':<8}{'收益率':<15}{'最大回撤':<12}{'交易':<8}{'胜率':<10}{'盈亏比':<10}{'盈亏因子':<10}{'评价':<15}")
        print("-"*90)

        for _, row in results_df.iterrows():
            ret_change = row['return'] - baseline['return']
            if row['threshold'] == best_by_return['threshold']:
                marker = "*** 最优"
            elif row['threshold'] == 0.0:
                marker = "原版"
            elif ret_change > 50:
                marker = "+ 大幅改善"
            elif ret_change > 10:
                marker = "+ 明显改善"
            elif ret_change > 0:
                marker = "+ 改善"
            elif ret_change > -10:
                marker = ""
            else:
                marker = "- 下降"

            print(f"{row['threshold']:<8.2f}{row['return']:>+12.2f}%{row['max_dd']:>10.2f}%"
                  f"{row['trades']:>6}{row['win_rate']:>8.1f}%{row['profit_ratio']:>8.2f}"
                  f"{row['profit_factor']:>8.2f}  {marker}")

        all_optimal[symbol] = {
            'best_threshold': best_by_return['threshold'],
            'best_return': best_by_return['return'],
            'best_dd': best_by_return['max_dd'],
            'best_trades': best_by_return['trades'],
            'baseline_return': baseline['return'],
            'baseline_dd': baseline['max_dd'],
            'baseline_trades': baseline['trades'],
        }

        print(f"\n最优: 阈值 {best_by_return['threshold']:.2f}, 收益 {best_by_return['return']:+.2f}%")

    # 汇总结果
    print("\n" + "="*120)
    print("汇总 - 各品种回测结果与最优量能参数".center(120))
    print("="*120)

    print(f"\n{'品种':<8}{'原版收益':<12}{'原版回撤':<10}{'最优阈值':<10}{'优化收益':<12}{'优化回撤':<10}{'收益变化':<12}{'效果':<10}")
    print("-"*90)

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

        print(f"{symbol:<8}{data['baseline_return']:>+10.2f}%{data['baseline_dd']:>8.2f}%"
              f"{data['best_threshold']:>8.2f}{data['best_return']:>+10.2f}%"
              f"{data['best_dd']:>8.2f}%{ret_change:>+10.2f}%  {effect}")

        optimal_thresholds[symbol] = data['best_threshold']

    # 生成配置代码
    print("\n" + "="*120)
    print("最优配置代码".center(120))
    print("="*120)

    print("\n# 各品种回测结果（akshare 18年数据，CCI+STC）")
    print("BACKTEST_RESULTS = {")
    for symbol in all_optimal:
        d = all_optimal[symbol]
        print(f"    '{symbol}': {{'return': {d['best_return']:.2f}, 'max_dd': {d['best_dd']:.2f}, 'trades': {d['best_trades']}}},")
    print("}")

    print("\n# 各品种最优量比阈值")
    print("OPTIMAL_VOLUME_THRESHOLDS = {")
    for symbol in sorted(optimal_thresholds.keys()):
        print(f"    '{symbol}': {optimal_thresholds[symbol]:.2f},")
    print("}")

    # 统计
    print("\n" + "="*120)
    print("统计".center(120))
    print("="*120)

    improved = sum(1 for d in all_optimal.values() if d['best_return'] > d['baseline_return'])
    total = len(all_optimal)

    print(f"\n量能优化效果:")
    print(f"  改善品种数: {improved}/{total}")
    print(f"  平均收益变化: {sum(d['best_return']-d['baseline_return'] for d in all_optimal.values())/total:.2f}%")

    # 按收益排序
    sorted_results = sorted(all_optimal.items(), key=lambda x: x[1]['best_return'], reverse=True)

    print(f"\n按收益率排序:")
    print(f"{'排名':<6}{'品种':<10}{'收益率':<15}{'最大回撤':<12}{'交易次数':<10}{'评级':<10}")
    print("-"*65)

    for rank, (symbol, d) in enumerate(sorted_results, 1):
        return_dd_ratio = d['best_return'] / abs(d['best_dd']) if d['best_dd'] != 0 else 0

        if return_dd_ratio >= 30 and d['best_return'] > 500:
            rating = "优秀"
        elif return_dd_ratio >= 15 and d['best_return'] > 200:
            rating = "良好"
        elif d['best_return'] > 0:
            rating = "一般"
        else:
            rating = "较差"

        print(f"{rank:<6}{symbol:<10}{d['best_return']:>+12.2f}%{d['best_dd']:>10.2f}%{d['best_trades']:>10}  {rating}")

    return all_optimal, optimal_thresholds


if __name__ == "__main__":
    main()
