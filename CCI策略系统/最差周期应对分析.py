"""
各品种最差周期应对分析
分析每个品种在最差周期的表现，提出应对方案
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


class Backtest:
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

    def run_with_details(self):
        balance = self.initial_capital
        position = 0
        entry_price = 0.0
        trades = []
        equity = []
        drawdowns = []

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
                        pass
                    else:
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
            return None, None

        trades_df = pd.DataFrame(trades)
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100

        return total_return, trades_df


def analyze_worst_period(df, params, vol_threshold):
    """分析最差周期"""
    # 分成2年周期
    periods = []
    start_date = df.index[0]
    end_date = df.index[-1]

    period_start = start_date
    while period_start < end_date:
        period_end = period_start + pd.DateOffset(years=2)
        if period_end > end_date:
            period_end = end_date

        mask = (df.index >= period_start) & (df.index < period_end)
        period_df = df[mask]

        if len(period_df) > 250:
            bt = Backtest(period_df, params, vol_threshold)
            ret, trades = bt.run_with_details()
            if ret is not None:
                periods.append({
                    'start': period_start,
                    'end': period_df.index[-1],
                    'return': ret,
                    'trades': trades,
                    'df': period_df
                })

        period_start = period_end

    if not periods:
        return None

    # 找最差周期
    worst = min(periods, key=lambda x: x['return'])
    return worst


def main():
    print("="*120)
    print("各品种最差周期应对分析".center(120))
    print("="*120)

    test_symbols = ['黄金', '镍', '锡', '白银', '玻璃', '纯碱', '铜', '糖', '锌', '棉花', '铝', '铅']

    all_analysis = {}

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

        worst = analyze_worst_period(df, params, vol_threshold)

        if worst:
            print(f"\n最差周期: {worst['start'].strftime('%Y-%m')} ~ {worst['end'].strftime('%Y-%m')}")
            print(f"收益率: {worst['return']:+.2f}%")

            if worst['trades'] is not None and len(worst['trades']) > 0:
                trades = worst['trades']

                # 统计亏损原因
                losing_trades = trades[trades['pnl'] < 0]

                print(f"\n交易统计:")
                print(f"  总交易: {len(trades)}笔")
                print(f"  盈利: {len(trades[trades['pnl'] > 0])}笔")
                print(f"  亏损: {len(losing_trades)}笔")

                if len(losing_trades) > 0:
                    print(f"\n亏损原因分析:")
                    for loss_type in losing_trades['type'].unique():
                        type_trades = losing_trades[losing_trades['type'] == loss_type]
                        total_loss = type_trades['pnl'].sum()
                        avg_loss = type_trades['pnl'].mean()
                        print(f"  {loss_type}: {len(type_trades)}笔, 总亏损${total_loss:,.0f}, 平均${avg_loss:,.0f}")

                    # 最大单笔亏损
                    max_loss_trade = losing_trades.loc[losing_trades['pnl'].idxmin()]
                    print(f"\n最大单笔亏损:")
                    print(f"  入场: {max_loss_trade['entry_date'].strftime('%Y-%m-%d')} @ {max_loss_trade['entry_price']:.2f}")
                    print(f"  出场: {max_loss_trade['exit_date'].strftime('%Y-%m-%d')} @ {max_loss_trade['exit_price']:.2f}")
                    print(f"  亏损: ${max_loss_trade['pnl']:,.0f} ({max_loss_trade['type']})")

                # 连续亏损分析
                trades_sorted = trades.sort_values('entry_date')
                max_consecutive = 0
                current_consecutive = 0
                for _, trade in trades_sorted.iterrows():
                    if trade['pnl'] < 0:
                        current_consecutive += 1
                        max_consecutive = max(max_consecutive, current_consecutive)
                    else:
                        current_consecutive = 0

                print(f"\n最大连续亏损: {max_consecutive}笔")

            # 应对方案
            print(f"\n应对方案:")
            if worst['return'] < -30:
                print(f"  1. 降低仓位至50%")
                print(f"  2. 增加止损幅度或改用ATR止损")
                print(f"  3. 暂停交易等待趋势明朗")
            elif worst['return'] < -15:
                print(f"  1. 适度降低仓位至70%")
                print(f"  2. 严格执行止损纪律")
            else:
                print(f"  1. 保持正常交易")
                print(f"  2. 注意风险控制")

            all_analysis[symbol] = {
                'worst_return': worst['return'],
                'worst_period': f"{worst['start'].strftime('%Y-%m')} ~ {worst['end'].strftime('%Y-%m')}",
                'total_trades': len(trades) if worst['trades'] is not None else 0,
            }

    # 汇总
    print("\n" + "="*120)
    print("汇总".center(120))
    print("="*120)

    print(f"\n{'品种':<8}{'最差周期':<25}{'收益率':<12}{'交易次数':<10}{'风险等级':<10}")
    print("-"*70)

    for symbol in sorted(all_analysis.keys(), key=lambda x: all_analysis[x]['worst_return']):
        data = all_analysis[symbol]
        risk = "高风险" if data['worst_return'] < -40 else "中风险" if data['worst_return'] < -20 else "低风险"
        print(f"{symbol:<8}{data['worst_period']:<25}{data['worst_return']:>+10.2f}%{data['total_trades']:>10}  {risk}")

    return all_analysis


if __name__ == "__main__":
    main()
