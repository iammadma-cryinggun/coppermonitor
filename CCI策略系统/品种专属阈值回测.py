"""
品种专属量比阈值回测 - 验证个性化阈值的效果
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

# 各品种专属量比阈值（基于超卖信号15%分位数）
VOLUME_RATIO_THRESHOLDS = {
    '棉花': 0.67,
    '锌': 0.69,
    '白银': 0.70,
    '铅': 0.70,
    '黄金': 0.73,
    '锡': 0.74,
    '玻璃': 0.74,
    '铝': 0.75,
    '纯碱': 0.76,
    '铜': 0.78,
    '镍': 0.79,
    '糖': 0.74,  # 估计值
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


class PersonalizedVolumeBacktest:
    """带品种专属量比阈值的回测引擎"""

    def __init__(self, df, params, vol_threshold=0.0):
        self.df = df.copy()
        self.params = params
        self.vol_threshold = vol_threshold
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
        skipped_low_vol = 0

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

                if cci < self.params['cci_oversold']:
                    open_signal = True
                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max']):
                    open_signal = True

                if open_signal:
                    # 交易量过滤（使用品种专属阈值）
                    if not np.isnan(vol_ratio) and vol_ratio < self.vol_threshold:
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
    print("品种专属量比阈值回测".center(120))
    print("="*120)

    print("""
【测试说明】

对比两种情况：
1. 无过滤（原版）
2. 品种专属阈值（每个品种使用自己的缩量标准）
""")

    # 显示各品种阈值
    print("\n各品种量比阈值:")
    print("-"*60)
    for symbol in sorted(VOLUME_RATIO_THRESHOLDS.keys(), key=lambda x: VOLUME_RATIO_THRESHOLDS[x]):
        print(f"  {symbol:<10} {VOLUME_RATIO_THRESHOLDS[symbol]:.2f}")

    test_symbols = ['铜', '纯碱', '玻璃', '白银', '镍', '黄金', '锌', '铝']

    all_results = {}

    for symbol in test_symbols:
        params = OPTIMAL_PARAMS[symbol]
        code = CODE_MAP[symbol]
        vol_threshold = VOLUME_RATIO_THRESHOLDS.get(symbol, 0.6)

        df = get_full_data(code)
        if df is None:
            continue

        print(f"\n{'='*100}")
        print(f"[{symbol}] 专属阈值: {vol_threshold}".center(100))
        print("="*100)

        all_results[symbol] = {}

        # 无过滤
        bt0 = PersonalizedVolumeBacktest(df, params, vol_threshold=0.0)
        result0 = bt0.run()

        # 品种专属阈值
        bt1 = PersonalizedVolumeBacktest(df, params, vol_threshold=vol_threshold)
        result1 = bt1.run()

        if result0:
            all_results[symbol]['no_filter'] = result0
            print(f"\n无过滤:")
            print(f"  收益率: {result0['total_return']:+.2f}%")
            print(f"  最大回撤: {result0['max_dd']:.2f}%")
            print(f"  胜率: {result0['win_rate']:.1f}%")
            print(f"  盈亏比: {result0['profit_ratio']:.2f}")
            print(f"  交易次数: {result0['total_trades']}")

        if result1:
            all_results[symbol]['personalized'] = result1
            print(f"\n专属阈值 (量比>{vol_threshold}):")
            print(f"  收益率: {result1['total_return']:+.2f}%")
            print(f"  最大回撤: {result1['max_dd']:.2f}%")
            print(f"  胜率: {result1['win_rate']:.1f}%")
            print(f"  盈亏比: {result1['profit_ratio']:.2f}")
            print(f"  交易次数: {result1['total_trades']}")
            print(f"  因低量比跳过: {result1['skipped_low_vol']}次")

    # 汇总对比
    print("\n" + "="*120)
    print("汇总对比".center(120))
    print("="*120)

    print(f"\n{'品种':<10}{'阈值':<8}{'无过滤收益':<15}{'专属阈值收益':<15}{'收益变化':<12}{'无过滤胜率':<12}{'专属胜率':<12}{'胜率变化':<10}")
    print("-"*110)

    for symbol in test_symbols:
        if symbol in all_results:
            r0 = all_results[symbol].get('no_filter', {})
            r1 = all_results[symbol].get('personalized', {})
            threshold = VOLUME_RATIO_THRESHOLDS.get(symbol, 0.6)

            if r0 and r1:
                ret_change = r1['total_return'] - r0['total_return']
                wr_change = r1['win_rate'] - r0['win_rate']

                print(f"{symbol:<10}{threshold:<8.2f}{r0['total_return']:>+12.2f}%{r1['total_return']:>+12.2f}%"
                      f"{ret_change:>+10.2f}%{r0['win_rate']:>10.1f}%{r1['win_rate']:>10.1f}%{wr_change:>+8.1f}%")

    # 效果统计
    print("\n" + "="*120)
    print("效果统计".center(120))
    print("="*120)

    improved = 0
    degraded = 0
    total_ret_change = 0

    print(f"\n{'品种':<10}{'过滤信号':<12}{'收益变化':<15}{'回撤变化':<15}{'胜率变化':<12}{'效果':<10}")
    print("-"*80)

    for symbol in test_symbols:
        if symbol in all_results:
            r0 = all_results[symbol].get('no_filter', {})
            r1 = all_results[symbol].get('personalized', {})

            if r0 and r1:
                skipped = r0['total_trades'] - r1['total_trades']
                ret_change = r1['total_return'] - r0['total_return']
                dd_change = r1['max_dd'] - r0['max_dd']
                wr_change = r1['win_rate'] - r0['win_rate']

                if ret_change > 0:
                    effect = "改善"
                    improved += 1
                else:
                    effect = "下降"
                    degraded += 1

                total_ret_change += ret_change

                print(f"{symbol:<10}{skipped:<12}{ret_change:>+12.2f}%{dd_change:>+12.2f}%{wr_change:>+10.1f}%{effect:<10}")

    # 结论
    print("\n" + "="*120)
    print("结论".center(120))
    print("="*120)

    print(f"""
【品种专属量比阈值效果】

统计结果:
  - 改善品种数: {improved}
  - 下降品种数: {degraded}
  - 总收益变化: {total_ret_change:+.2f}%

【建议】

1. 对于收益改善的品种:
   - 建议启用量比过滤
   - 使用品种专属阈值

2. 对于收益下降的品种:
   - 可以不使用量比过滤
   - 或者调整阈值

3. 综合考虑:
   - 如果胜率提升明显，即使收益略降也可接受
   - 量比过滤主要作用是提高信号质量
""")


if __name__ == "__main__":
    main()
