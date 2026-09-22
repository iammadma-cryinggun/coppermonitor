"""
参数稳定性分析 - 2年滚动回测
将历史数据按2年分段，测试参数在不同时期的表现
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


class CompleteBacktest:
    """完整回测引擎"""

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

        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma = tp.rolling(window=cci_length).mean()
        mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
        self.df['cci'] = (tp - sma) / (0.015 * mad)
        self.df['cci_ma'] = self.df['cci'].rolling(window=ma_length).mean()

        self.df['stc'] = calculate_stc(self.df['close'])

        self.df['vol_ma5'] = self.df['volume'].rolling(5).mean()
        self.df['vol_ratio'] = self.df['volume'] / self.df['vol_ma5']

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
            vol_ratio = current['vol_ratio']

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
                    trades.append({'pnl': net_pnl, 'type': 'stop_loss'})
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
                    trades.append({'pnl': net_pnl, 'type': 'take_profit'})
                    position = 0
                    entry_price = 0.0
                    continue

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
                    if self.min_vol_ratio > 0 and not np.isnan(vol_ratio) and vol_ratio < self.min_vol_ratio:
                        pass
                    else:
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

        total_profit = winning_trades['pnl'].sum()
        total_loss = abs(losing_trades['pnl'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        return {
            'total_return': total_return,
            'max_dd': max_dd,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_ratio': profit_ratio,
            'profit_factor': profit_factor,
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

        if len(period_df) > 250:  # 至少250个交易日
            periods.append({
                'start': period_start,
                'end': period_df.index[-1],
                'df': period_df,
                'days': len(period_df)
            })

        period_start = period_end

    return periods


def main():
    print("="*120)
    print("参数稳定性分析 - 2年滚动回测".center(120))
    print("="*120)

    print("""
【分析方法】
将历史数据按2年分段，测试参数在不同时期的表现
评估参数的时间稳定性

【评估标准】
- 盈利周期数/总周期数 = 胜率
- 收益率标准差 = 波动性
- 最差周期表现 = 风险
""")

    test_symbols = ['黄金', '镍', '锡', '白银', '玻璃', '纯碱', '铜', '糖', '锌', '棉花', '铝', '铅']

    # 最优量能阈值
    vol_thresholds = {
        '棉花': 0.90, '玻璃': 0.90, '白银': 0.65, '糖': 0.70,
        '纯碱': 0.50, '铅': 0.70, '铜': 0.50, '铝': 0.75,
        '锌': 0.60, '镍': 0.00, '锡': 0.00, '黄金': 0.80,
    }

    all_results = {}

    for symbol in test_symbols:
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            continue

        code = CODE_MAP[symbol]
        df = get_akshare_data(code)
        if df is None:
            continue

        # 按两年分段
        periods = split_into_periods(df, years=2)

        if len(periods) < 2:
            continue

        print(f"\n{'='*100}")
        print(f"[{symbol}] 共{len(periods)}个周期".center(100))
        print("="*100)

        vol_threshold = vol_thresholds.get(symbol, 0.0)
        period_results = []

        print(f"\n{'周期':<25}{'交易日':<8}{'收益率':<12}{'最大回撤':<10}{'交易次数':<8}{'胜率':<8}{'盈亏比':<8}{'效果':<10}")
        print("-"*95)

        for i, period in enumerate(periods):
            bt = CompleteBacktest(period['df'], params, min_vol_ratio=vol_threshold)
            result = bt.run()

            if result:
                period_label = f"{period['start'].strftime('%Y-%m')} ~ {period['end'].strftime('%Y-%m')}"
                effect = "盈利" if result['total_return'] > 0 else "亏损"

                print(f"{period_label:<25}{period['days']:<8}{result['total_return']:>+10.2f}%"
                      f"{result['max_dd']:>8.2f}%{result['total_trades']:>8}"
                      f"{result['win_rate']:>6.1f}%{result['profit_ratio']:>6.2f}  {effect}")

                period_results.append({
                    'period': period_label,
                    'start': period['start'],
                    'return': result['total_return'],
                    'max_dd': result['max_dd'],
                    'trades': result['total_trades'],
                    'win_rate': result['win_rate'],
                    'profit_ratio': result['profit_ratio'],
                    'profit_factor': result['profit_factor']
                })

        if period_results:
            results_df = pd.DataFrame(period_results)

            # 统计
            profitable = (results_df['return'] > 0).sum()
            total = len(results_df)
            avg_return = results_df['return'].mean()
            std_return = results_df['return'].std()
            min_return = results_df['return'].min()
            max_return = results_df['return'].max()
            avg_dd = results_df['max_dd'].mean()

            print("\n" + "-"*95)
            print(f"{'统计':<25}{'':8}{avg_return:>+10.2f}%{avg_dd:>8.2f}%")
            print(f"\n盈利周期: {profitable}/{total} ({profitable/total*100:.0f}%)")
            print(f"平均收益: {avg_return:+.2f}%")
            print(f"收益标准差: {std_return:.2f}%")
            print(f"最差周期: {min_return:+.2f}%")
            print(f"最好周期: {max_return:+.2f}%")

            all_results[symbol] = {
                'periods': period_results,
                'profitable_periods': profitable,
                'total_periods': total,
                'win_rate': profitable/total*100,
                'avg_return': avg_return,
                'std_return': std_return,
                'min_return': min_return,
                'max_return': max_return,
                'avg_dd': avg_dd
            }

    # 汇总结果
    print("\n" + "="*120)
    print("汇总 - 参数稳定性评估".center(120))
    print("="*120)

    print(f"\n{'品种':<8}{'总周期':<8}{'盈利周期':<10}{'周期胜率':<10}{'平均收益':<12}{'收益标准差':<12}{'最差周期':<12}{'最好周期':<12}{'稳定性':<10}")
    print("-"*100)

    stability_ranking = []

    for symbol, data in all_results.items():
        stability = "稳定" if data['win_rate'] >= 80 and data['std_return'] < 100 else \
                   "较稳定" if data['win_rate'] >= 60 and data['std_return'] < 150 else \
                   "一般" if data['win_rate'] >= 50 else "不稳定"

        print(f"{symbol:<8}{data['total_periods']:<8}{data['profitable_periods']:<10}"
              f"{data['win_rate']:>6.0f}%{data['avg_return']:>+10.2f}%"
              f"{data['std_return']:>10.2f}%{data['min_return']:>+10.2f}%"
              f"{data['max_return']:>+10.2f}%  {stability}")

        stability_ranking.append({
            'symbol': symbol,
            'win_rate': data['win_rate'],
            'std_return': data['std_return'],
            'avg_return': data['avg_return'],
            'stability': stability
        })

    # 稳定性排名
    print("\n" + "="*120)
    print("稳定性排名".center(120))
    print("="*120)

    # 按周期胜率排序
    stability_ranking.sort(key=lambda x: (-x['win_rate'], x['std_return']))

    print(f"\n{'排名':<6}{'品种':<10}{'周期胜率':<12}{'收益标准差':<12}{'平均收益':<12}{'评级':<10}")
    print("-"*65)

    for rank, item in enumerate(stability_ranking, 1):
        rating = "优秀" if item['win_rate'] >= 80 else \
                "良好" if item['win_rate'] >= 60 else \
                "一般" if item['win_rate'] >= 50 else "较差"

        print(f"{rank:<6}{item['symbol']:<10}{item['win_rate']:>8.0f}%"
              f"{item['std_return']:>10.2f}%{item['avg_return']:>+10.2f}%  {rating}")

    # 结论
    print("\n" + "="*120)
    print("结论与建议".center(120))
    print("="*120)

    stable = [s for s in stability_ranking if s['win_rate'] >= 80]
    moderate = [s for s in stability_ranking if 60 <= s['win_rate'] < 80]
    unstable = [s for s in stability_ranking if s['win_rate'] < 60]

    print(f"""
【稳定性分析】

高度稳定（周期胜率>=80%）: {len(stable)}个品种
  {', '.join([s['symbol'] for s in stable])}

中等稳定（周期胜率60-80%）: {len(moderate)}个品种
  {', '.join([s['symbol'] for s in moderate])}

不稳定（周期胜率<60%）: {len(unstable)}个品种
  {', '.join([s['symbol'] for s in unstable])}

【建议】

1. 高度稳定的品种: 可放心使用当前参数
2. 中等稳定的品种: 建议小仓位或组合使用
3. 不稳定的品种: 需要谨慎，考虑是否调整参数
""")

    return all_results


if __name__ == "__main__":
    main()
