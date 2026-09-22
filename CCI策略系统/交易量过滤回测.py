"""
交易量过滤回测 - 验证量比过滤的效果
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
    """获取完整历史数据（含交易量）"""
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


class VolumeFilteredBacktest:
    """带交易量过滤的回测引擎"""

    def __init__(self, df, params, min_vol_ratio=0.0):
        self.df = df.copy()
        self.params = params
        self.min_vol_ratio = min_vol_ratio  # 最小量比阈值
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

        # 量比
        self.df['vol_ma5'] = self.df['volume'].rolling(5).mean()
        self.df['vol_ratio'] = self.df['volume'] / self.df['vol_ma5']

    def run(self):
        balance = self.initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity = []
        skipped_low_vol = 0  # 因量比过低跳过的次数

        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]
            next_day = self.df.iloc[i + 1]

            cci = current['cci']
            cci_ma = current['cci_ma']
            vol_ratio = current['vol_ratio']

            cci_prev = self.df.iloc[i-1]['cci'] if i > 0 else cci
            cci_ma_prev = self.df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

            if position > 0:
                # 止损
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

                # 止盈
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

                # 超卖开仓
                if cci < self.params['cci_oversold']:
                    open_signal = True
                # 金叉开仓
                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max']):
                    open_signal = True

                if open_signal:
                    # 交易量过滤
                    if not np.isnan(vol_ratio) and vol_ratio < self.min_vol_ratio:
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
            'profit_factor': profit_factor,
            'profit_ratio': profit_ratio,
            'expectancy': expectancy,
            'skipped_low_vol': skipped_low_vol
        }


def main():
    print("="*120)
    print("交易量过滤回测对比".center(120))
    print("="*120)

    print("""
【测试说明】

对比三种情况：
1. 无过滤（原版）
2. 量比 > 0.5
3. 量比 > 0.6（推荐）
""")

    test_symbols = ['铜', '纯碱', '玻璃', '白银', '镍']
    vol_thresholds = [0.0, 0.5, 0.6]

    all_results = {}

    for symbol in test_symbols:
        params = OPTIMAL_PARAMS[symbol]
        code = CODE_MAP[symbol]

        df = get_full_data(code)
        if df is None:
            continue

        print(f"\n{'='*100}")
        print(f"[{symbol}]".center(100))
        print("="*100)

        all_results[symbol] = {}

        for min_vol in vol_thresholds:
            bt = VolumeFilteredBacktest(df, params, min_vol_ratio=min_vol)
            result = bt.run()

            if result:
                all_results[symbol][min_vol] = result
                label = f"量比>{min_vol}" if min_vol > 0 else "无过滤"
                print(f"\n{label}:")
                print(f"  收益率: {result['total_return']:+.2f}%")
                print(f"  最大回撤: {result['max_dd']:.2f}%")
                print(f"  胜率: {result['win_rate']:.1f}%")
                print(f"  盈亏比: {result['profit_ratio']:.2f}")
                print(f"  交易次数: {result['total_trades']}")
                if min_vol > 0:
                    print(f"  因低量比跳过: {result['skipped_low_vol']}次")

    # 汇总对比
    print("\n" + "="*120)
    print("汇总对比".center(120))
    print("="*120)

    print(f"\n{'品种':<10}{'无过滤收益':<15}{'无过滤回撤':<12}{'量比>0.5收益':<15}{'量比>0.5回撤':<12}{'量比>0.6收益':<15}{'量比>0.6回撤':<12}")
    print("-"*120)

    for symbol in test_symbols:
        if symbol in all_results:
            r0 = all_results[symbol].get(0.0, {})
            r5 = all_results[symbol].get(0.5, {})
            r6 = all_results[symbol].get(0.6, {})

            ret0 = f"{r0.get('total_return', 0):+.2f}%" if r0 else "N/A"
            dd0 = f"{r0.get('max_dd', 0):.2f}%" if r0 else "N/A"
            ret5 = f"{r5.get('total_return', 0):+.2f}%" if r5 else "N/A"
            dd5 = f"{r5.get('max_dd', 0):.2f}%" if r5 else "N/A"
            ret6 = f"{r6.get('total_return', 0):+.2f}%" if r6 else "N/A"
            dd6 = f"{r6.get('max_dd', 0):.2f}%" if r6 else "N/A"

            print(f"{symbol:<10}{ret0:<15}{dd0:<12}{ret5:<15}{dd5:<12}{ret6:<15}{dd6:<12}")

    # 详细对比（收益率和回撤变化）
    print("\n" + "="*120)
    print("效果分析".center(120))
    print("="*120)

    print(f"\n{'品种':<10}{'过滤掉信号数':<15}{'收益率变化':<15}{'回撤变化':<15}{'胜率变化':<15}")
    print("-"*80)

    for symbol in test_symbols:
        if symbol in all_results:
            r0 = all_results[symbol].get(0.0, {})
            r6 = all_results[symbol].get(0.6, {})

            if r0 and r6:
                skipped = r0['total_trades'] - r6['total_trades']
                ret_change = r6['total_return'] - r0['total_return']
                dd_change = r6['max_dd'] - r0['max_dd']
                wr_change = r6['win_rate'] - r0['win_rate']

                print(f"{symbol:<10}{skipped:<15}{ret_change:>+12.2f}%{dd_change:>+12.2f}%{wr_change:>+12.1f}%")

    # 结论
    print("\n" + "="*120)
    print("结论".center(120))
    print("="*120)

    print("""
【交易量过滤效果评估】

1. 优点：
   - 过滤掉极端缩量时的信号（市场没有方向）
   - 可能提高胜率
   - 减少无效交易

2. 缺点：
   - 可能错过一些有效信号
   - 收益率可能略有下降
   - 增加了策略复杂度

3. 建议：
   - 如果胜率提升明显 → 建议使用
   - 如果收益率下降不多 → 可以使用
   - 如果回撤改善 → 推荐使用
""")


if __name__ == "__main__":
    main()
