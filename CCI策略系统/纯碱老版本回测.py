"""
纯碱(SA0) 老版本回测 - 使用完整回测_量能优化_akshare.py的类和逻辑
完全复制老版本的回测逻辑
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

DATA_CACHE = {}

def get_akshare_data(code):
    """从akshare获取完整历史数据"""
    if code in DATA_CACHE:
        return DATA_CACHE[code].copy()
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()
        df.set_index('date', inplace=True)
        DATA_CACHE[code] = df.copy()
        return df
    except:
        return None

def calculate_stc(close_prices, fast_period=23, slow_period=50, cycle_period=10):
    """计算STC指标（与老版本一致）"""
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
    """完整回测引擎（与老版本一致）"""

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
    print("=" * 120)
    print("纯碱(SA0) 老版本回测".center(120))
    print("=" * 120)

    symbol = '纯碱'
    params = OPTIMAL_PARAMS.get(symbol)
    if not params:
        print("未找到纯碱参数")
        return

    code = CODE_MAP[symbol]
    vol_threshold = VOL_THRESHOLDS.get(symbol, 0.0)

    print(f"\n[1/3] 获取纯碱数据...")
    df = get_akshare_data(code)
    if df is None:
        print("数据获取失败")
        return

    print(f"数据量: {len(df)}条")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")

    print(f"\n[2/3] 运行回测...")
    print(f"参数:")
    print(f"  CCI: length={params['cci_length']}, oversold={params['cci_oversold']}, overbought={params['cci_overbought']}, cross_max={params['cci_cross_max']}")
    print(f"  STC: oversold={params['stc_oversold']}, cross={params['stc_cross']}")
    print(f"  量能过滤: vol_ratio >= {vol_threshold}")
    print(f"  杠杆: {2}x, 乘数: {params['multiplier']}")

    bt = CompleteBacktest(df, params, min_vol_ratio=vol_threshold)
    result = bt.run()

    if result:
        print(f"\n[3/3] 回测结果:")
        print(f"  总收益率:     {result['total_return']:>10.2f}%")
        print(f"  最大回撤:     {result['max_dd']:>10.2f}%")
        print(f"  交易次数:     {result['total_trades']:>10}笔")
        print(f"  胜率:         {result['win_rate']:>10.2f}%")
        print(f"  盈亏比:       {result['profit_ratio']:>10.2f}")
        print(f"  盈亏因子:     {result['profit_factor']:>10.2f}")
        print(f"  期望收益:     ${result['expectancy']:>10.2f}/笔")
        print(f"  跳过低量能: {result['skipped_low_vol']:>10}次")

        # 与文档对比
        print(f"\n与文档对比:")
        print(f"  收益率: {result['total_return']:>10.2f}% vs 预期{params['expected_return']:>10.2f}% (差{result['total_return'] - params['expected_return']:>10.2f}%)")
        print(f"  交易次数: {result['total_trades']:>10}笔 vs 预期{params['total_trades']:>10}笔 (差{result['total_trades'] - params['total_trades']:>10}笔)")
    else:
        print("\n回测失败或无交易")

if __name__ == "__main__":
    main()
