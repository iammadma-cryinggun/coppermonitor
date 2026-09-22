"""
量能专属参数优化 - 为每个品种找出最优量比阈值
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


class VolumeOptimizedBacktest:
    """带量能过滤的回测引擎"""

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

        # 量比
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
                    # 量能过滤
                    if self.min_vol_ratio > 0 and not np.isnan(vol_ratio) and vol_ratio < self.min_vol_ratio:
                        pass  # 跳过
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

        return {
            'total_return': total_return,
            'max_dd': max_dd,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_ratio': profit_ratio,
        }


def find_optimal_volume_threshold():
    """为每个品种找出最优量比阈值"""

    print("="*120)
    print("量能专属参数优化 - 网格搜索最优阈值".center(120))
    print("="*120)

    print("""
【优化目标】
找出每个品种的最优量比阈值，使收益率最大化

【测试范围】
阈值范围: 0.0 ~ 1.0，步长0.05
0.0 = 无过滤
0.5 = 过滤极端缩量
1.0 = 必须放量
""")

    test_symbols = ['铜', '纯碱', '玻璃', '白银', '镍', '黄金', '锌', '铝', '糖', '棉花', '锡', '铅']
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
        print(f"[{symbol}] 量能阈值优化".center(100))
        print("="*100)

        results = []

        for threshold in thresholds:
            bt = VolumeOptimizedBacktest(df, params, min_vol_ratio=threshold)
            result = bt.run()

            if result:
                results.append({
                    'threshold': threshold,
                    'return': result['total_return'],
                    'max_dd': result['max_dd'],
                    'trades': result['total_trades'],
                    'win_rate': result['win_rate'],
                    'profit_ratio': result['profit_ratio']
                })

        if not results:
            continue

        # 找出最优阈值
        results_df = pd.DataFrame(results)

        # 按收益率排序
        best_by_return = results_df.loc[results_df['return'].idxmax()]
        best_by_dd = results_df.loc[results_df['max_dd'].idxmax()]  # 回撤最小
        baseline = results_df[results_df['threshold'] == 0.0].iloc[0] if len(results_df[results_df['threshold'] == 0.0]) > 0 else None

        # 显示所有结果
        print(f"\n{'阈值':<8}{'收益率':<15}{'最大回撤':<12}{'交易次数':<10}{'胜率':<10}{'盈亏比':<10}{'评价':<15}")
        print("-"*90)

        for _, row in results_df.iterrows():
            ret_change = row['return'] - baseline['return'] if baseline is not None else 0
            if row['threshold'] == best_by_return['threshold']:
                marker = "*** 最优收益"
            elif row['threshold'] == best_by_dd['threshold']:
                marker = "** 最小回撤"
            elif ret_change > 0:
                marker = "+ 改善"
            elif ret_change < -50:
                marker = "- 下降较多"
            else:
                marker = ""

            print(f"{row['threshold']:<8.2f}{row['return']:>+12.2f}%{row['max_dd']:>10.2f}%"
                  f"{row['trades']:>10}{row['win_rate']:>8.1f}%{row['profit_ratio']:>8.2f}  {marker}")

        all_optimal[symbol] = {
            'best_return_threshold': best_by_return['threshold'],
            'best_return': best_by_return['return'],
            'best_dd_threshold': best_by_dd['threshold'],
            'best_dd': best_by_dd['max_dd'],
            'baseline_return': baseline['return'] if baseline is not None else 0,
            'baseline_dd': baseline['max_dd'] if baseline is not None else 0,
        }

        print(f"\n推荐: 阈值 {best_by_return['threshold']:.2f} (收益率 {best_by_return['return']:+.2f}%)")

    # 汇总结果
    print("\n" + "="*120)
    print("汇总 - 各品种最优量能参数".center(120))
    print("="*120)

    print(f"\n{'品种':<10}{'原版收益':<15}{'最优阈值':<12}{'优化收益':<15}{'收益变化':<12}{'回撤变化':<12}{'建议':<15}")
    print("-"*100)

    optimal_thresholds = {}

    for symbol, data in all_optimal.items():
        ret_change = data['best_return'] - data['baseline_return']
        dd_change = data['best_dd'] - data['baseline_dd']

        if ret_change > 10:
            suggestion = "强烈推荐启用"
        elif ret_change > 0:
            suggestion = "推荐启用"
        elif ret_change > -20:
            suggestion = "可选"
        else:
            suggestion = "不推荐"

        print(f"{symbol:<10}{data['baseline_return']:>+12.2f}%{data['best_return_threshold']:>10.2f}"
              f"{data['best_return']:>+12.2f}%{ret_change:>+10.2f}%{dd_change:>+10.2f}%  {suggestion}")

        optimal_thresholds[symbol] = data['best_return_threshold']

    # 生成配置代码
    print("\n" + "="*120)
    print("最优配置代码".center(120))
    print("="*120)

    print("\n# 各品种最优量比阈值（基于收益率最大化）")
    print("OPTIMAL_VOLUME_THRESHOLDS = {")
    for symbol in sorted(optimal_thresholds.keys()):
        print(f"    '{symbol}': {optimal_thresholds[symbol]:.2f},")
    print("}")

    # 分类建议
    print("\n" + "="*120)
    print("分类建议".center(120))
    print("="*120)

    strong_recommend = []
    recommend = []
    optional = []
    not_recommend = []

    for symbol, data in all_optimal.items():
        ret_change = data['best_return'] - data['baseline_return']
        if ret_change > 10:
            strong_recommend.append((symbol, data['best_return_threshold']))
        elif ret_change > 0:
            recommend.append((symbol, data['best_return_threshold']))
        elif ret_change > -20:
            optional.append((symbol, data['best_return_threshold']))
        else:
            not_recommend.append((symbol, data['best_return_threshold']))

    print(f"\n【强烈推荐启用量能过滤】")
    for symbol, threshold in strong_recommend:
        print(f"  {symbol}: 阈值 {threshold:.2f}")

    print(f"\n【推荐启用量能过滤】")
    for symbol, threshold in recommend:
        print(f"  {symbol}: 阈值 {threshold:.2f}")

    print(f"\n【可选启用】")
    for symbol, threshold in optional:
        print(f"  {symbol}: 阈值 {threshold:.2f}")

    print(f"\n【不推荐启用量能过滤】")
    for symbol, threshold in not_recommend:
        print(f"  {symbol}: 阈值 {threshold:.2f} (使用0.0)")

    return optimal_thresholds


if __name__ == "__main__":
    find_optimal_volume_threshold()
