"""
所有品种严格回测验证
==================
测试全部12个品种的真实表现
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')
from cci_calculations import calculate_cci_tv, calculate_stc, f_normalize


class StrictBacktest:
    """严格的回测引擎"""

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

        stc_raw = calculate_stc(self.df, length=10, fast=23, slow=50, aaa=0.5)
        self.df['stc'] = f_normalize(stc_raw, 20, 80)

    def apply_slippage(self, price, direction='buy'):
        if direction == 'buy':
            return price * (1 + self.slippage_rate)
        else:
            return price * (1 - self.slippage_rate)

    def calculate_commission(self, price, qty):
        return price * qty * self.multiplier * self.commission_rate

    def run(self):
        balance = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_date = None
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
                # 止损
                if next_day['low'] <= entry_price * 0.96:
                    stop_price = max(next_day['open'], entry_price * 0.96)
                    exit_price = self.apply_slippage(stop_price, 'sell')
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = self.calculate_commission(exit_price, position) * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'stop_loss'})
                    position = 0
                    entry_price = 0.0
                    continue

                # 止盈
                if next_day['high'] >= entry_price * 1.20:
                    take_profit_price = min(next_day['open'], entry_price * 1.20)
                    exit_price = self.apply_slippage(take_profit_price, 'sell')
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = self.calculate_commission(exit_price, position) * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'take_profit'})
                    position = 0
                    entry_price = 0.0
                    continue

                # CCI超买平仓
                if cci > self.params['cci_overbought']:
                    exit_price = self.apply_slippage(next_day['close'], 'sell')
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = self.calculate_commission(exit_price, position) * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'overbought'})
                    position = 0
                    entry_price = 0.0
                    continue

                # CCI死叉平仓
                if cci_prev >= cci_ma_prev and cci < cci_ma:
                    exit_price = self.apply_slippage(next_day['close'], 'sell')
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = self.calculate_commission(exit_price, position) * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'death_cross'})
                    position = 0
                    entry_price = 0.0
                    continue

            # 开仓
            if position == 0:
                open_signal = False

                if cci < self.params['cci_oversold'] and stc >= self.params['stc_oversold']:
                    open_signal = True
                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max'] and
                      stc >= self.params['stc_cross']):
                    open_signal = True

                if open_signal:
                    entry_price = self.apply_slippage(next_day['open'], 'buy')
                    max_value = balance * 0.9 * self.leverage
                    qty = int(max_value / (entry_price * self.multiplier))
                    qty = max(1, qty)
                    commission = self.calculate_commission(entry_price, qty)
                    balance -= commission
                    position = qty
                    entry_date = next_day.name

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
        total_trades = len(trades)

        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
        profit_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        expectancy = (win_rate/100 * avg_win - (1-win_rate/100) * avg_loss)

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
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'profit_ratio': profit_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }


def main():
    from 最优参数配置 import OPTIMAL_PARAMS

    print("="*120)
    print("所有品种严格回测验证".center(120))
    print("="*120)
    print("\n包含: 滑点(0.02%) + 手续费(0.03%)")

    data_dir = "D:\\期货数据\\铜期货监控\\global_futures_daily"

    all_results = {}

    for symbol, params in OPTIMAL_PARAMS.items():
        file_name = f"{symbol}_daily.csv"
        if symbol == '铜':
            file_name = "铜_LME_daily.csv"

        file_path = os.path.join(data_dir, file_name)

        print(f"\n[{symbol}]", end=" ")

        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.dropna(subset=['open', 'high', 'low', 'close'])

            bt = StrictBacktest(df, params)
            result = bt.run()

            if result:
                all_results[symbol] = result
                print(f"收益率: {result['total_return']:+.2f}%, 胜率: {result['win_rate']:.1f}%, 盈亏比: {result['profit_ratio']:.2f}")
            else:
                print("[交易次数不足]")

        except Exception as e:
            print(f"[ERROR] {e}")

    # 排序并输出
    print("\n" + "="*120)
    print("严格回测结果汇总（按收益率排序）".center(120))
    print("="*120)

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['total_return'], reverse=True)

    print(f"\n{'排名':<6}{'品种':<10}{'收益率':<12}{'最大回撤':<12}{'胜率':<10}{'盈亏比':<10}{'盈亏因子':<10}{'期望收益':<12}{'评级':<10}")
    print("-"*120)

    for rank, (symbol, r) in enumerate(sorted_results, 1):
        return_dd = r['total_return'] / abs(r['max_dd']) if r['max_dd'] != 0 else 0

        if r['profit_ratio'] >= 2.0 and r['win_rate'] >= 50:
            rating = "优秀"
        elif r['profit_ratio'] >= 1.0 and r['win_rate'] >= 45:
            rating = "良好"
        elif r['profit_ratio'] >= 0.8 and r['win_rate'] >= 40:
            rating = "一般"
        else:
            rating = "较差"

        print(f"{rank:<6}{symbol:<10}{r['total_return']:>+10.2f}%{r['max_dd']:>10.2f}%"
              f"{r['win_rate']:>8.1f}%{r['profit_ratio']:>8.2f}{r['profit_factor']:>8.2f}"
              f"${r['expectancy']:>10.0f}{rating:<10}")

    print("\n" + "="*120)
    print("完成!".center(120))
    print("="*120)

    # 保存结果
    output_file = os.path.join(data_dir, "..", "CCI策略系统", "全品种严格回测结果.csv")
    results_list = []
    for symbol, r in sorted_results:
        results_list.append({
            '品种': symbol,
            '收益率%': r['total_return'],
            '最大回撤%': r['max_dd'],
            '胜率%': r['win_rate'],
            '盈亏比': r['profit_ratio'],
            '盈亏因子': r['profit_factor'],
            '期望收益': r['expectancy'],
            '交易次数': r['total_trades']
        })

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
