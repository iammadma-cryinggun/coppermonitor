"""
盲回测v3 - 只在极端情况下暂停交易
设计思路：
1. 不使用趋势过滤（保留大部分信号）
2. 只在极端情况下暂停开多：
   - 连续下跌 >= 10天
   - 最近20天跌幅 > 20%
   - 最近5天跌幅 > 15%
3. 其他时间正常交易
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


def get_akshare_data(code):
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
            low = window.min()
            high = window.max()
            if high != low:
                stc_normalized.iloc[i] = (stc.iloc[i] - low) / (high - low) * 100 - 50
            else:
                stc_normalized.iloc[i] = stc_normalized.iloc[i-1] if i > 0 else 0

    return stc_normalized


class ExtremeConditionDetector:
    """极端市场情况检测器"""

    def __init__(self, df):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """准备检测所需数据"""
        # 计算收益率
        self.df['return'] = self.df['close'].pct_change()

        # 计算连续下跌天数
        self.df['is_down'] = self.df['return'] < 0
        self.df['consecutive_down'] = 0

        count = 0
        for i in range(len(self.df)):
            if self.df.iloc[i]['is_down']:
                count += 1
            else:
                count = 0
            self.df.iloc[self.df.index.get_loc(self.df.index[i]), self.df.columns.get_loc('consecutive_down')] = count

    def is_extreme_condition(self, current_idx):
        """检测是否为极端情况"""
        current_date = self.df.index[current_idx]

        if current_idx < 20:
            return False, "数据不足"

        # 1. 检查连续下跌天数
        consecutive_down = self.df.iloc[current_idx]['consecutive_down']
        if consecutive_down >= 10:
            return True, f"连续下跌{int(consecutive_down)}天"

        # 2. 检查最近20天跌幅
        recent_20d = self.df.iloc[current_idx-19:current_idx+1]
        if len(recent_20d) >= 20:
            drop_20d = (recent_20d['close'].iloc[0] - recent_20d['close'].iloc[-1]) / recent_20d['close'].iloc[0]
            if drop_20d > 0.20:
                return True, f"20天跌{drop_20d*100:.1f}%"

        # 3. 检查最近5天跌幅
        recent_5d = self.df.iloc[current_idx-4:current_idx+1]
        if len(recent_5d) >= 5:
            drop_5d = (recent_5d['close'].iloc[0] - recent_5d['close'].iloc[-1]) / recent_5d['close'].iloc[0]
            if drop_5d > 0.15:
                return True, f"5天跌{drop_5d*100:.1f}%"

        return False, "正常"


class BlindBacktestV3:
    """盲回测v3 - 只在极端情况下暂停"""

    def __init__(self, full_df, params, vol_threshold=0.0):
        self.full_df = full_df.copy()
        self.params = params
        self.vol_threshold = vol_threshold
        self.commission_rate = 0.0003
        self.slippage_rate = 0.0002
        self.multiplier = params.get('multiplier', 5)
        self.initial_capital = 100000

        # 统计
        self.total_signals = 0
        self.filtered_signals = 0
        self.filter_reasons = {}

    def run(self):
        """运行盲回测"""
        balance = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_date = None
        trades = []
        equity = [self.initial_capital]
        equity_dates = [self.full_df.index[0]]

        # 滚动计算指标
        min_data_needed = max(self.params['cci_length'], self.params['ma_length'], 60) + 50

        for i in range(min_data_needed, len(self.full_df) - 1):
            current_date = self.full_df.index[i]

            # 使用截至当天的所有数据（盲回测原则）
            historical_df = self.full_df.iloc[:i+1].copy()

            # 计算指标
            cci_length = self.params['cci_length']
            ma_length = self.params['ma_length']

            tp = (historical_df['high'] + historical_df['low'] + historical_df['close']) / 3
            sma = tp.rolling(window=cci_length).mean()
            mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
            historical_df['cci'] = (tp - sma) / (0.015 * mad)
            historical_df['cci_ma'] = historical_df['cci'].rolling(window=ma_length).mean()
            historical_df['stc'] = calculate_stc(historical_df['close'])
            historical_df['vol_ma5'] = historical_df['volume'].rolling(5).mean()
            historical_df['vol_ratio'] = historical_df['volume'] / historical_df['vol_ma5']

            current = historical_df.iloc[-1]
            prev = historical_df.iloc[-2] if len(historical_df) > 1 else current
            next_day = self.full_df.iloc[i + 1]

            cci = current['cci']
            cci_ma = current['cci_ma']
            cci_prev = prev['cci']
            cci_ma_prev = prev['cci_ma']
            stc = current['stc']
            vol_ratio = current['vol_ratio']

            # 持仓处理
            if position > 0:
                # 止损
                if next_day['low'] <= entry_price * 0.96:
                    stop_price = max(next_day['open'], entry_price * 0.96)
                    exit_price = stop_price * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': next_day.name,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'stop_loss'
                    })
                    position = 0
                    entry_price = 0.0
                    equity.append(balance)
                    equity_dates.append(next_day.name)
                    continue

                # 止盈
                if next_day['high'] >= entry_price * 1.20:
                    take_profit_price = min(next_day['open'], entry_price * 1.20)
                    exit_price = take_profit_price * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': next_day.name,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'take_profit'
                    })
                    position = 0
                    entry_price = 0.0
                    equity.append(balance)
                    equity_dates.append(next_day.name)
                    continue

                # 超买平仓
                if cci > self.params['cci_overbought']:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': next_day.name,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'overbought'
                    })
                    position = 0
                    entry_price = 0.0
                    equity.append(balance)
                    equity_dates.append(next_day.name)
                    continue

                # 死叉平仓
                if cci_prev >= cci_ma_prev and cci < cci_ma:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': next_day.name,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'death_cross'
                    })
                    position = 0
                    entry_price = 0.0
                    equity.append(balance)
                    equity_dates.append(next_day.name)
                    continue

            # 开仓信号
            if position == 0:
                open_signal = False

                # 超卖信号
                if (cci < self.params['cci_oversold'] and
                    stc >= self.params['stc_oversold']):
                    open_signal = True

                # 金叉信号
                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max'] and
                      stc >= self.params['stc_cross']):
                    open_signal = True

                if open_signal:
                    self.total_signals += 1

                    # 量能过滤
                    if self.vol_threshold > 0 and not np.isnan(vol_ratio) and vol_ratio < self.vol_threshold:
                        self.filtered_signals += 1
                        reason = "量能不足"
                        self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1
                        continue

                    # 极端情况检测
                    detector = ExtremeConditionDetector(historical_df)
                    is_extreme, reason = detector.is_extreme_condition(len(historical_df) - 1)

                    if is_extreme:
                        self.filtered_signals += 1
                        self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1
                        continue

                    # 开仓
                    entry_price = next_day['open'] * (1 + self.slippage_rate)
                    max_value = balance * 0.9 * 2
                    qty = int(max_value / (entry_price * self.multiplier))
                    qty = max(1, qty)
                    commission = entry_price * qty * self.multiplier * self.commission_rate
                    balance -= commission
                    position = qty
                    entry_date = next_day.name

            # 更新净值
            val = balance
            if position > 0:
                unrealized_pnl = (next_day['close'] - entry_price) * position * self.multiplier
                val += unrealized_pnl
            equity.append(val)
            equity_dates.append(next_day.name)

        return self._calculate_metrics(trades, equity, equity_dates)

    def _calculate_metrics(self, trades, equity, equity_dates):
        if len(trades) < 5:
            return None

        trades_df = pd.DataFrame(trades)
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100

        # 计算最大回撤
        equity_series = pd.Series(equity)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        # 交易统计
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

        # 出场原因统计
        exit_reasons = trades_df['type'].value_counts().to_dict()

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'exit_reasons': exit_reasons,
            'total_signals': self.total_signals,
            'filtered_signals': self.filtered_signals,
            'filter_reasons': self.filter_reasons,
        }


class OriginalBacktest:
    """原始回测（无过滤）"""

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
        entry_date = None
        trades = []
        equity = [self.initial_capital]

        for i in range(100, len(self.df) - 1):
            current = self.df.iloc[i]
            next_day = self.df.iloc[i + 1]
            prev = self.df.iloc[i-1] if i > 0 else current

            cci = current['cci']
            cci_ma = current['cci_ma']
            cci_prev = prev['cci']
            cci_ma_prev = prev['cci_ma']
            stc = current['stc']
            vol_ratio = current['vol_ratio']

            if position > 0:
                if next_day['low'] <= entry_price * 0.96:
                    stop_price = max(next_day['open'], entry_price * 0.96)
                    exit_price = stop_price * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'type': 'stop_loss', 'pnl': net_pnl})
                    position = 0
                    continue

                if next_day['high'] >= entry_price * 1.20:
                    take_profit_price = min(next_day['open'], entry_price * 1.20)
                    exit_price = take_profit_price * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'type': 'take_profit', 'pnl': net_pnl})
                    position = 0
                    continue

                if cci > self.params['cci_overbought']:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'type': 'overbought', 'pnl': net_pnl})
                    position = 0
                    continue

                if cci_prev >= cci_ma_prev and cci < cci_ma:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'type': 'death_cross', 'pnl': net_pnl})
                    position = 0
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
                    if self.vol_threshold > 0 and not np.isnan(vol_ratio) and vol_ratio < self.vol_threshold:
                        continue

                    entry_price = next_day['open'] * (1 + self.slippage_rate)
                    max_value = balance * 0.9 * 2
                    qty = int(max_value / (entry_price * self.multiplier))
                    qty = max(1, qty)
                    commission = entry_price * qty * self.multiplier * self.commission_rate
                    balance -= commission
                    position = qty
                    entry_date = next_day.name

            val = balance
            if position > 0:
                unrealized_pnl = (next_day['close'] - entry_price) * position * self.multiplier
                val += unrealized_pnl
            equity.append(val)

        if len(trades) < 5:
            return None

        trades_df = pd.DataFrame(trades)
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100
        equity_series = pd.Series(equity)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()

        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] < 0]
        win_rate = len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0

        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades_df),
            'win_rate': win_rate,
        }


def main():
    print("=" * 120)
    print("盲回测v3 - 只在极端情况下暂停交易".center(120))
    print("=" * 120)

    print("""
【设计思路】
1. 不使用趋势过滤（MA60）- 保留大部分交易信号
2. 只在极端情况下暂停开多：
   - 连续下跌 >= 10天
   - 最近20天跌幅 > 20%
   - 最近5天跌幅 > 15%
3. 其他时间正常交易
""")

    test_symbols = ['黄金', '镍', '锡', '白银', '玻璃', '纯碱', '铜', '糖', '锌', '棉花', '铝', '铅']

    results = []

    for symbol in test_symbols:
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            continue

        code = CODE_MAP[symbol]
        vol_threshold = VOL_THRESHOLDS.get(symbol, 0.0)

        df = get_akshare_data(code)
        if df is None:
            continue

        print(f"\n{'='*100}")
        print(f"[{symbol}] 分析中...".center(100))
        print("="*100)

        # 原始回测
        original_bt = OriginalBacktest(df, params, vol_threshold)
        original_result = original_bt.run()

        # 盲回测v3
        blind_bt = BlindBacktestV3(df, params, vol_threshold)
        blind_result = blind_bt.run()

        if original_result and blind_result:
            print(f"\n【对比结果】")
            print(f"{'指标':<20}{'原始回测':>20}{'盲回测v3':>20}{'差异':>20}")
            print("-"*80)
            print(f"{'总收益率':.<20}{original_result['total_return']:>19.2f}%{blind_result['total_return']:>19.2f}%{blind_result['total_return']-original_result['total_return']:>+19.2f}%")
            print(f"{'最大回撤':.<20}{original_result['max_drawdown']:>19.2f}%{blind_result['max_drawdown']:>19.2f}%{blind_result['max_drawdown']-original_result['max_drawdown']:>+19.2f}%")
            print(f"{'交易次数':.<20}{original_result['total_trades']:>19}{blind_result['total_trades']:>19}{blind_result['total_trades']-original_result['total_trades']:>+19}")
            print(f"{'胜率':.<20}{original_result['win_rate']:>18.1f}%{blind_result['win_rate']:>18.1f}%{blind_result['win_rate']-original_result['win_rate']:>+18.1f}%")

            print(f"\n【信号过滤统计】")
            print(f"总信号数: {blind_result['total_signals']}")
            print(f"被过滤信号: {blind_result['filtered_signals']} ({blind_result['filtered_signals']/blind_result['total_signals']*100:.1f}%)")

            if blind_result['filter_reasons']:
                print(f"\n过滤原因:")
                for reason, count in sorted(blind_result['filter_reasons'].items(), key=lambda x: -x[1]):
                    print(f"  {reason}: {count}次")

            results.append({
                'symbol': symbol,
                'original_return': original_result['total_return'],
                'blind_return': blind_result['total_return'],
                'return_diff': blind_result['total_return'] - original_result['total_return'],
                'original_dd': original_result['max_drawdown'],
                'blind_dd': blind_result['max_drawdown'],
                'dd_diff': blind_result['max_drawdown'] - original_result['max_drawdown'],
                'original_trades': original_result['total_trades'],
                'blind_trades': blind_result['total_trades'],
                'filter_rate': blind_result['filtered_signals']/blind_result['total_signals']*100 if blind_result['total_signals'] > 0 else 0,
            })

    # 汇总
    print("\n" + "="*120)
    print("汇总对比".center(120))
    print("="*120)

    print(f"\n{'品种':<8}{'原始收益':>12}{'v3收益':>12}{'收益变化':>12}{'原始回撤':>12}{'v3回撤':>12}{'回撤改善':>12}{'过滤率':>10}")
    print("-"*100)

    for r in results:
        dd_improve = r['original_dd'] - r['blind_dd']  # 正数表示改善
        print(f"{r['symbol']:<8}{r['original_return']:>11.1f}%{r['blind_return']:>11.1f}%{r['return_diff']:>+11.1f}%{r['original_dd']:>11.1f}%{r['blind_dd']:>11.1f}%{dd_improve:>+11.1f}%{r['filter_rate']:>9.1f}%")

    # 统计
    avg_return_diff = np.mean([r['return_diff'] for r in results])
    avg_dd_improve = np.mean([r['original_dd'] - r['blind_dd'] for r in results])
    avg_filter_rate = np.mean([r['filter_rate'] for r in results])

    print("-"*100)
    print(f"{'平均':<8}{'':<12}{'':<12}{avg_return_diff:>+11.1f}%{'':<12}{'':<12}{avg_dd_improve:>+11.1f}%{avg_filter_rate:>9.1f}%")

    print(f"\n【结论】")
    print(f"1. 平均收益变化: {avg_return_diff:+.1f}%")
    print(f"2. 平均回撤改善: {avg_dd_improve:+.1f}%")
    print(f"3. 平均信号过滤率: {avg_filter_rate:.1f}%")

    improved = [r for r in results if r['blind_dd'] > r['original_dd']]
    worsened = [r for r in results if r['blind_dd'] <= r['original_dd']]

    print(f"\n回撤改善的品种: {len(improved)}/{len(results)}")
    if improved:
        for r in improved:
            print(f"  {r['symbol']}: {r['original_dd']:.1f}% -> {r['blind_dd']:.1f}% ({r['original_dd']-r['blind_dd']:+.1f}%)")

    return results


if __name__ == "__main__":
    main()
