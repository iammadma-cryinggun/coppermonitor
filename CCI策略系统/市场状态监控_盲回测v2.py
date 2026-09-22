"""
市场状态监控 + 盲回测 v2
调整风险评分阈值，减少误杀有效信号
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
    except:
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


class MarketStateDetector:
    """市场状态检测器 v2 - 更宽松的阈值"""

    def __init__(self, df):
        self.df = df.copy()
        self._calculate_indicators()

    def _calculate_indicators(self):
        self.df['ma20'] = self.df['close'].rolling(20).mean()
        self.df['ma60'] = self.df['close'].rolling(60).mean()

        self.df['returns'] = self.df['close'].pct_change()
        self.df['volatility'] = self.df['returns'].rolling(20).std() * np.sqrt(250) * 100

        self.df['vol_pct'] = self.df['volatility'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        self.df['down_day'] = (self.df['returns'] < 0).astype(int)
        self.df['consecutive_down'] = 0
        count = 0
        for i in range(len(self.df)):
            if self.df['down_day'].iloc[i] == 1:
                count += 1
            else:
                count = 0
            self.df.iloc[i, self.df.columns.get_loc('consecutive_down')] = count

        self.df['ma_gap'] = (self.df['ma60'] - self.df['ma20']) / self.df['ma60'] * 100
        self.df['below_ma60'] = (self.df['close'] < self.df['ma60']).astype(int)
        self.df['days_below_ma60'] = self.df['below_ma60'].rolling(20).sum()

        # 新增：近期跌幅（过去20天）
        self.df['recent_drop'] = (self.df['close'].shift(20) - self.df['close']) / self.df['close'].shift(20) * 100

    def get_state(self, idx):
        """获取某一时点的市场状态 - 更宽松的评分"""
        row = self.df.loc[idx]

        close = row['close']
        ma20 = row['ma20']
        ma60 = row['ma60']
        ma_gap = row['ma_gap']
        consecutive_down = row['consecutive_down']
        vol_pct = row['vol_pct']
        days_below_ma60 = row['days_below_ma60']
        recent_drop = row['recent_drop'] if not np.isnan(row['recent_drop']) else 0

        risk_score = 0
        signals = []

        # 1. 趋势方向：价格 vs MA60（保持）
        if close < ma60:
            risk_score += 15  # 降低：20->15
            signals.append("价格<MA60")
        else:
            signals.append("价格>MA60")

        # 2. 趋势强度：MA距离（提高阈值）
        if ma_gap > 5:  # 提高：3%->5%
            risk_score += 20  # 降低：25->20
            signals.append(f"强下跌(MA差距{ma_gap:.1f}%)")
        elif ma_gap > 3:  # 新增中间档
            risk_score += 10
            signals.append(f"中等下跌(MA差距{ma_gap:.1f}%)")

        # 3. 连续下跌（提高阈值）
        if consecutive_down >= 7:  # 提高：5->7
            risk_score += 15  # 降低：20->15
            signals.append(f"连续下跌{consecutive_down}天")
        elif consecutive_down >= 5:
            risk_score += 8
            signals.append(f"连续下跌{consecutive_down}天")

        # 4. 持续弱势（提高阈值）
        if days_below_ma60 >= 17:  # 提高：15->17
            risk_score += 15  # 降低：20->15
            signals.append(f"持续弱势({days_below_ma60}/20天<MA60)")
        elif days_below_ma60 >= 15:
            risk_score += 8
            signals.append(f"偏弱({days_below_ma60}/20天<MA60)")

        # 5. 波动率异常（保持）
        if vol_pct > 0.85:  # 提高：0.8->0.85
            risk_score += 10  # 降低：15->10
            signals.append(f"高波动(历史{vol_pct*100:.0f}%分位)")

        # 6. 新增：近期急跌
        if recent_drop > 15:  # 20天跌超15%
            risk_score += 15
            signals.append(f"近期急跌{recent_drop:.1f}%")
        elif recent_drop > 10:
            risk_score += 8
            signals.append(f"近期下跌{recent_drop:.1f}%")

        # 风险等级（更宽松）
        if risk_score >= 50:  # 提高：60->50
            level = "高风险"
            action = "暂停开仓"
            position_pct = 0
        elif risk_score >= 35:  # 提高：40->35
            level = "中高风险"
            action = "仓位减半"
            position_pct = 50
        elif risk_score >= 20:  # 保持
            level = "中风险"
            action = "启用趋势过滤"
            position_pct = 100
        else:
            level = "低风险"
            action = "正常交易"
            position_pct = 100

        # 只在极高风险时才启用趋势过滤
        use_trend_filter = risk_score >= 35  # 提高：20->35

        return {
            'risk_score': risk_score,
            'risk_level': level,
            'action': action,
            'position_pct': position_pct,
            'signals': signals,
            'use_trend_filter': use_trend_filter,
        }


class AdaptiveBacktest:
    """自适应回测引擎 v2"""

    def __init__(self, df, params, vol_threshold=0.0):
        self.df = df.copy()
        self.params = params
        self.vol_threshold = vol_threshold
        self.commission_rate = 0.0003
        self.slippage_rate = 0.0002
        self.multiplier = params.get('multiplier', 5)
        self.initial_capital = 100000
        self.disable_adaptive = False

        self.detector = MarketStateDetector(self.df)
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
        state_records = []

        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]
            next_day = self.df.iloc[i + 1]
            current_date = self.df.index[i]

            cci = current['cci']
            cci_ma = current['cci_ma']
            stc = current['stc']
            vol_ratio = current['vol_ratio']

            cci_prev = self.df.iloc[i-1]['cci'] if i > 0 else cci
            cci_ma_prev = self.df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

            market_state = self.detector.get_state(current_date)

            if position > 0:
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
                        'pnl': net_pnl,
                        'type': 'stop_loss',
                        'risk_level': risk_at_entry
                    })
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
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': next_day.name,
                        'pnl': net_pnl,
                        'type': 'take_profit',
                        'risk_level': risk_at_entry
                    })
                    position = 0
                    entry_price = 0.0
                    continue

                if cci > self.params['cci_overbought']:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': next_day.name,
                        'pnl': net_pnl,
                        'type': 'overbought',
                        'risk_level': risk_at_entry
                    })
                    position = 0
                    entry_price = 0.0
                    continue

                if cci_prev >= cci_ma_prev and cci < cci_ma:
                    exit_price = next_day['close'] * (1 - self.slippage_rate)
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = exit_price * position * self.multiplier * self.commission_rate * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': next_day.name,
                        'pnl': net_pnl,
                        'type': 'death_cross',
                        'risk_level': risk_at_entry
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
                    if not self.disable_adaptive:
                        if market_state['risk_score'] >= 50:
                            state_records.append({
                                'date': current_date,
                                'state': market_state,
                                'action': 'skipped'
                            })
                            continue

                    if self.vol_threshold > 0 and not np.isnan(vol_ratio) and vol_ratio < self.vol_threshold:
                        continue

                    if not self.disable_adaptive and market_state['use_trend_filter']:
                        ma60 = self.detector.df.loc[current_date, 'ma60']
                        if next_day['close'] < ma60:
                            state_records.append({
                                'date': current_date,
                                'state': market_state,
                                'action': 'filtered_by_trend'
                            })
                            continue

                    if self.disable_adaptive:
                        position_pct = 1.0
                    else:
                        position_pct = market_state['position_pct'] / 100

                    entry_price = next_day['open'] * (1 + self.slippage_rate)
                    max_value = balance * 0.9 * 2 * position_pct
                    qty = int(max_value / (entry_price * self.multiplier))
                    qty = max(1, qty)
                    commission = entry_price * qty * self.multiplier * self.commission_rate
                    balance -= commission
                    position = qty
                    entry_date = next_day.name
                    risk_at_entry = market_state['risk_level']

                    state_records.append({
                        'date': current_date,
                        'state': market_state,
                        'action': 'opened',
                        'qty': qty
                    })

            val = balance
            if position > 0:
                unrealized_pnl = (next_day['close'] - entry_price) * position * self.multiplier
                val += unrealized_pnl
            equity.append(val)

        if len(trades) < 5:
            return None, None, None

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
        }, trades_df, state_records


def main():
    print("="*120)
    print("市场状态监控 + 盲回测 v2（更宽松阈值）".center(120))
    print("="*120)

    print("""
【v2 调整内容】

1. 风险评分降低：
   - 价格<MA60: 20->15分
   - MA差距>3%: 25->20分（且阈值提高到5%）
   - 连续下跌>=5天: 20->15分（且阈值提高到7天）
   - 持续弱势: 20->15分（且阈值提高到17天）
   - 波动率异常: 15->10分

2. 新增指标：
   - 近期急跌>15%: +15分
   - 近期下跌>10%: +8分

3. 风险等级调整：
   - 高风险(>=50分, 原60): 暂停开仓
   - 中高风险(>=35分, 原40): 仓位减半
   - 中风险(>=20分): 启用趋势过滤
   - 低风险(<20分): 正常交易

4. 趋势过滤只在>=35分时启用（原20分）
""")

    test_symbols = ['铜', '黄金', '白银', '棉花', '锌', '玻璃', '糖', '纯碱']

    results = {}

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
        print(f"[{symbol}] 盲回测中...".center(100))
        print("="*100)

        # 原版
        bt_original = AdaptiveBacktest(df, params, vol_threshold)
        bt_original.disable_adaptive = True
        result_original, _, _ = bt_original.run()

        # v2自适应
        bt_adaptive = AdaptiveBacktest(df, params, vol_threshold)
        result_adaptive, trades, states = bt_adaptive.run()

        if result_original and result_adaptive:
            print(f"\n{'版本':<18}{'收益率':<15}{'最大回撤':<12}{'交易次数':<10}{'胜率':<10}{'盈亏比':<10}")
            print("-"*80)
            print(f"{'原版(无调整)':<18}{result_original['total_return']:>+12.2f}%"
                  f"{result_original['max_dd']:>10.2f}%{result_original['total_trades']:>10}"
                  f"{result_original['win_rate']:>8.1f}%{result_original['profit_ratio']:>8.2f}")
            print(f"{'v2自适应(宽松)':<18}{result_adaptive['total_return']:>+12.2f}%"
                  f"{result_adaptive['max_dd']:>10.2f}%{result_adaptive['total_trades']:>10}"
                  f"{result_adaptive['win_rate']:>8.1f}%{result_adaptive['profit_ratio']:>8.2f}")

            ret_change = result_adaptive['total_return'] - result_original['total_return']
            dd_change = result_adaptive['max_dd'] - result_original['max_dd']

            print(f"\n变化: 收益{ret_change:+.2f}%, 回撤{dd_change:+.2f}%")

            if states:
                states_df = pd.DataFrame([{
                    'date': s['date'],
                    'risk_level': s['state']['risk_level'],
                    'action': s['action']
                } for s in states])

                opened = (states_df['action'] == 'opened').sum()
                skipped = (states_df['action'] == 'skipped').sum()
                filtered = (states_df['action'] == 'filtered_by_trend').sum()

                print(f"\n【信号处理统计】")
                print(f"  原版开仓: {result_original['total_trades']}次")
                print(f"  v2开仓: {opened}次 (保留{opened/result_original['total_trades']*100:.0f}%)")
                print(f"  跳过(高风险): {skipped}次")
                print(f"  过滤(趋势): {filtered}次")

            results[symbol] = {
                'original': result_original,
                'adaptive': result_adaptive,
                'ret_change': ret_change,
                'dd_change': dd_change,
            }

    # 汇总
    print("\n" + "="*120)
    print("汇总对比".center(120))
    print("="*120)

    print(f"\n{'品种':<8}{'原版收益':<12}{'v2收益':<12}{'收益变化':<12}{'原版回撤':<10}{'v2回撤':<10}{'回撤变化':<10}{'效果':<10}")
    print("-"*90)

    for symbol, data in results.items():
        if data['ret_change'] > 0 and data['dd_change'] > 0:
            effect = "双赢"
        elif data['ret_change'] > -50 and data['dd_change'] > 5:
            effect = "可接受"
        elif data['dd_change'] > 0:
            effect = "回撤改善"
        else:
            effect = "下降"

        print(f"{symbol:<8}{data['original']['total_return']:>+10.2f}%"
              f"{data['adaptive']['total_return']:>+10.2f}%{data['ret_change']:>+10.2f}%"
              f"{data['original']['max_dd']:>8.2f}%{data['adaptive']['max_dd']:>8.2f}%"
              f"{data['dd_change']:>+8.2f}%  {effect}")

    # 与v1对比
    print("\n" + "="*120)
    print("v1 vs v2 对比（关键品种）".center(120))
    print("="*120)

    print(f"""
【v1问题】（严格阈值）
- 白银: 收益6140% -> 473% (-5667%)
- 铜: 收益1279% -> 276% (-1003%)

【v2改进】（宽松阈值）
趋势过滤只在极高风险(>=35分)时启用
""")

    improved_ret = sum(1 for d in results.values() if d['ret_change'] > -100)
    improved_dd = sum(1 for d in results.values() if d['dd_change'] > 0)

    print(f"\n【v2结果】")
    print(f"  收益下降<100%的品种: {improved_ret}/{len(results)}")
    print(f"  回撤改善的品种: {improved_dd}/{len(results)}")

    return results


if __name__ == "__main__":
    main()
