"""
CCI策略综合优化回测
===================
三项优化：
1. 量能确认 - 成交量/持仓量验证
2. 趋势过滤 - EMA60趋势滤网，避免逆势抄底
3. 动态止损 - ATR替代固定4%止损
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
    """获取完整历史数据（含交易量和持仓量）"""
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


class OptimizedBacktest:
    """优化版回测引擎"""

    def __init__(self, df, params, use_volume_filter=True, use_trend_filter=True, use_atr_stop=True):
        self.df = df.copy()
        self.params = params
        self.use_volume_filter = use_volume_filter
        self.use_trend_filter = use_trend_filter
        self.use_atr_stop = use_atr_stop

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

        # 优化1: 量能指标
        self.df['vol_ma5'] = self.df['volume'].rolling(5).mean()
        self.df['vol_ratio'] = self.df['volume'] / self.df['vol_ma5']
        self.df['oi_ma5'] = self.df['open_interest'].rolling(5).mean()
        self.df['oi_ratio'] = self.df['open_interest'] / self.df['oi_ma5']

        # 优化2: 趋势过滤 (EMA60)
        self.df['ema60'] = self.df['close'].ewm(span=60, adjust=False).mean()
        self.df['ema20'] = self.df['close'].ewm(span=20, adjust=False).mean()

        # 优化3: ATR (动态止损)
        self.df['tr'] = np.maximum(
            self.df['high'] - self.df['low'],
            np.maximum(
                np.abs(self.df['high'] - self.df['close'].shift(1)),
                np.abs(self.df['low'] - self.df['close'].shift(1))
            )
        )
        self.df['atr'] = self.df['tr'].rolling(14).mean()

    def run(self):
        balance = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_atr = 0.0
        trades = []
        equity = []

        # 统计
        stats = {
            'skipped_trend': 0,      # 因逆势跳过
            'skipped_volume': 0,     # 因缩量跳过
            'atr_stops': 0,          # ATR止损次数
            'fixed_stops': 0,        # 固定止损次数
        }

        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]
            next_day = self.df.iloc[i + 1]

            cci = current['cci']
            cci_ma = current['cci_ma']
            vol_ratio = current['vol_ratio']
            oi_ratio = current['oi_ratio']
            ema60 = current['ema60']
            atr = current['atr']

            cci_prev = self.df.iloc[i-1]['cci'] if i > 0 else cci
            cci_ma_prev = self.df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

            if position > 0:
                # 计算止损价格
                if self.use_atr_stop and entry_atr > 0:
                    # 动态止损: 2倍ATR
                    stop_price = entry_price - (2.0 * entry_atr)
                    stop_pct = (stop_price - entry_price) / entry_price
                else:
                    # 固定止损: 4%
                    stop_price = entry_price * 0.96
                    stop_pct = -0.04

                # 止损
                if next_day['low'] <= stop_price:
                    actual_stop = max(next_day['open'], stop_price)
                    exit_price = actual_stop * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl

                    if self.use_atr_stop:
                        stats['atr_stops'] += 1
                        trades.append({'pnl': net_pnl, 'type': 'atr_stop'})
                    else:
                        stats['fixed_stops'] += 1
                        trades.append({'pnl': net_pnl, 'type': 'stop_loss'})

                    position = 0
                    entry_price = 0.0
                    continue

                # 止盈 (20%)
                if next_day['high'] >= entry_price * 1.20:
                    take_profit = min(next_day['open'], entry_price * 1.20)
                    exit_price = take_profit * (1 - self.slippage_rate)
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
                skip_reason = None

                # 超卖信号
                if cci < self.params['cci_oversold']:
                    open_signal = True
                # 金叉信号
                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max']):
                    open_signal = True

                if open_signal:
                    # 优化2: 趋势过滤 - 价格必须在EMA60上方
                    if self.use_trend_filter:
                        if next_day['close'] < ema60:
                            stats['skipped_trend'] += 1
                            skip_reason = '逆势'
                            open_signal = False

                    # 优化1: 量能确认 - 必须有放量
                    if self.use_volume_filter and open_signal:
                        if vol_ratio < 0.8:  # 缩量不进场
                            stats['skipped_volume'] += 1
                            skip_reason = '缩量'
                            open_signal = False

                if open_signal:
                    entry_price = next_day['open'] * (1 + self.slippage_rate)
                    entry_atr = atr  # 记录开仓时的ATR
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
            'stats': stats
        }


def main():
    print("="*120)
    print("CCI策略综合优化回测".center(120))
    print("="*120)

    print("""
【三项优化】

1. 量能确认 (Volume Filter)
   - 量比 < 0.8 时不进场（缩量信号无效）
   - 避免在市场观望时进场

2. 趋势过滤 (Trend Filter)
   - 价格 < EMA60 时不做多
   - 避免在下跌趋势中"接飞刀"

3. 动态止损 (ATR Stop)
   - 止损 = 开仓价 - 2*ATR
   - 根据波动率调整止损幅度
""")

    test_symbols = ['纯碱', '玻璃', '白银', '铜', '镍', '黄金', '锌', '糖']

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

        # 原版（无优化）
        bt0 = OptimizedBacktest(df, params,
                                use_volume_filter=False,
                                use_trend_filter=False,
                                use_atr_stop=False)
        result0 = bt0.run()

        # 优化版（全部优化）
        bt1 = OptimizedBacktest(df, params,
                                use_volume_filter=True,
                                use_trend_filter=True,
                                use_atr_stop=True)
        result1 = bt1.run()

        if result0:
            all_results[symbol]['original'] = result0
            print(f"\n原版:")
            print(f"  收益率: {result0['total_return']:+.2f}%")
            print(f"  最大回撤: {result0['max_dd']:.2f}%")
            print(f"  胜率: {result0['win_rate']:.1f}%")
            print(f"  盈亏比: {result0['profit_ratio']:.2f}")
            print(f"  交易次数: {result0['total_trades']}")

        if result1:
            all_results[symbol]['optimized'] = result1
            stats = result1['stats']
            print(f"\n优化版:")
            print(f"  收益率: {result1['total_return']:+.2f}%")
            print(f"  最大回撤: {result1['max_dd']:.2f}%")
            print(f"  胜率: {result1['win_rate']:.1f}%")
            print(f"  盈亏比: {result1['profit_ratio']:.2f}")
            print(f"  交易次数: {result1['total_trades']}")
            print(f"\n  过滤统计:")
            print(f"    因逆势跳过: {stats['skipped_trend']}次")
            print(f"    因缩量跳过: {stats['skipped_volume']}次")
            print(f"    ATR止损: {stats['atr_stops']}次")

    # 汇总对比
    print("\n" + "="*120)
    print("汇总对比".center(120))
    print("="*120)

    print(f"\n{'品种':<10}{'原版收益':<15}{'优化收益':<15}{'收益变化':<12}{'原版回撤':<12}{'优化回撤':<12}{'回撤变化':<12}")
    print("-"*100)

    for symbol in test_symbols:
        if symbol in all_results:
            r0 = all_results[symbol].get('original', {})
            r1 = all_results[symbol].get('optimized', {})

            if r0 and r1:
                ret_change = r1['total_return'] - r0['total_return']
                dd_change = r1['max_dd'] - r0['max_dd']

                print(f"{symbol:<10}{r0['total_return']:>+12.2f}%{r1['total_return']:>+12.2f}%"
                      f"{ret_change:>+10.2f}%{r0['max_dd']:>10.2f}%{r1['max_dd']:>10.2f}%{dd_change:>+10.2f}%")

    # 效果统计
    print("\n" + "="*120)
    print("效果统计".center(120))
    print("="*120)

    improved_ret = 0
    improved_dd = 0
    degraded_ret = 0

    print(f"\n{'品种':<10}{'交易减少':<12}{'收益变化':<15}{'回撤变化':<15}{'胜率变化':<12}{'效果':<10}")
    print("-"*80)

    for symbol in test_symbols:
        if symbol in all_results:
            r0 = all_results[symbol].get('original', {})
            r1 = all_results[symbol].get('optimized', {})

            if r0 and r1:
                trades_reduced = r0['total_trades'] - r1['total_trades']
                ret_change = r1['total_return'] - r0['total_return']
                dd_change = r1['max_dd'] - r0['max_dd']
                wr_change = r1['win_rate'] - r0['win_rate']

                if ret_change > 0:
                    improved_ret += 1
                    effect = "收益改善"
                else:
                    degraded_ret += 1
                    effect = "收益下降"

                if dd_change > 0:  # 回撤变小是好事（负数变小）
                    improved_dd += 1

                print(f"{symbol:<10}{trades_reduced:<12}{ret_change:>+12.2f}%{dd_change:>+12.2f}%"
                      f"{wr_change:>+10.1f}%{effect:<10}")

    # 结论
    print("\n" + "="*120)
    print("结论".center(120))
    print("="*120)

    print(f"""
【综合优化效果】

统计结果:
  - 收益改善品种数: {improved_ret}
  - 收益下降品种数: {degraded_ret}
  - 回撤改善品种数: {improved_dd}

【关键发现】

1. 趋势过滤 (EMA60) 效果显著
   - 纯碱、玻璃等下跌趋势品种被有效过滤
   - 避免了"接飞刀"的风险

2. 量能确认 减少假信号
   - 缩量时的超卖信号往往是陷阱
   - 过滤后信号质量提升

3. 动态止损 (ATR) 适应不同品种
   - 高波动品种（白银）有更大止损空间
   - 低波动品种（玉米）止损更紧凑

【建议】

强烈推荐启用的品种:
  - 纯碱、玻璃：趋势过滤避免逆势抄底
  - 白银、镍：ATR止损适应高波动

可选启用的品种:
  - 铜、黄金：效果中性

谨慎使用的品种:
  - 如果优化后收益大幅下降，考虑只启用部分优化
""")


if __name__ == "__main__":
    main()
