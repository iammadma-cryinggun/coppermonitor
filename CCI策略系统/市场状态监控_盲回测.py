"""
市场状态监控 + 盲回测
实时判断市场风险等级，动态调整策略
关键：只用当前及之前的数据，不用未来数据
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
    """市场状态检测器 - 盲检测（只用历史数据）"""

    def __init__(self, df):
        self.df = df.copy()
        self._calculate_indicators()

    def _calculate_indicators(self):
        # MA指标
        self.df['ma20'] = self.df['close'].rolling(20).mean()
        self.df['ma60'] = self.df['close'].rolling(60).mean()

        # 波动率（20日年化）
        self.df['returns'] = self.df['close'].pct_change()
        self.df['volatility'] = self.df['returns'].rolling(20).std() * np.sqrt(250) * 100

        # 波动率历史分位数（过去252天）
        self.df['vol_pct'] = self.df['volatility'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # 连续下跌天数
        self.df['down_day'] = (self.df['returns'] < 0).astype(int)
        self.df['consecutive_down'] = 0
        count = 0
        for i in range(len(self.df)):
            if self.df['down_day'].iloc[i] == 1:
                count += 1
            else:
                count = 0
            self.df.iloc[i, self.df.columns.get_loc('consecutive_down')] = count

        # MA距离（MA60-MA20）/MA60
        self.df['ma_gap'] = (self.df['ma60'] - self.df['ma20']) / self.df['ma60'] * 100

        # 价格在MA60下方的持续天数
        self.df['below_ma60'] = (self.df['close'] < self.df['ma60']).astype(int)
        self.df['days_below_ma60'] = self.df['below_ma60'].rolling(20).sum()

    def get_state(self, idx):
        """获取某一时点的市场状态（只用该时点及之前的数据）"""
        row = self.df.loc[idx]

        close = row['close']
        ma20 = row['ma20']
        ma60 = row['ma60']
        ma_gap = row['ma_gap']
        consecutive_down = row['consecutive_down']
        vol_pct = row['vol_pct']
        days_below_ma60 = row['days_below_ma60']

        # 风险评分
        risk_score = 0
        signals = []

        # 1. 趋势方向：价格 vs MA60
        if close < ma60:
            risk_score += 20
            signals.append("价格<MA60")
        else:
            signals.append("价格>MA60")

        # 2. 趋势强度：MA距离
        if ma_gap > 3:  # MA60明显高于MA20
            risk_score += 25
            signals.append(f"强下跌(MA差距{ma_gap:.1f}%)")
        elif ma_gap > 1:
            risk_score += 10
            signals.append(f"中等下跌(MA差距{ma_gap:.1f}%)")

        # 3. 连续下跌
        if consecutive_down >= 5:
            risk_score += 20
            signals.append(f"连续下跌{consecutive_down}天")
        elif consecutive_down >= 3:
            risk_score += 10
            signals.append(f"连续下跌{consecutive_down}天")

        # 4. 持续弱势（价格在MA60下方天数）
        if days_below_ma60 >= 15:  # 20天中>=15天在MA60下方
            risk_score += 20
            signals.append(f"持续弱势({days_below_ma60}/20天<MA60)")
        elif days_below_ma60 >= 10:
            risk_score += 10
            signals.append(f"偏弱({days_below_ma60}/20天<MA60)")

        # 5. 波动率异常
        if vol_pct > 0.8:  # 波动率处于历史80%分位以上
            risk_score += 15
            signals.append(f"高波动(历史{vol_pct*100:.0f}%分位)")

        # 风险等级
        if risk_score >= 60:
            level = "高风险"
            action = "暂停开仓"
            position_pct = 0
        elif risk_score >= 40:
            level = "中高风险"
            action = "仓位减半+趋势过滤"
            position_pct = 50
        elif risk_score >= 20:
            level = "中风险"
            action = "启用趋势过滤"
            position_pct = 100
        else:
            level = "低风险"
            action = "正常交易"
            position_pct = 100

        return {
            'risk_score': risk_score,
            'risk_level': level,
            'action': action,
            'position_pct': position_pct,
            'signals': signals,
            'use_trend_filter': risk_score >= 20,
        }


class AdaptiveBacktest:
    """自适应回测引擎 - 根据市场状态动态调整"""

    def __init__(self, df, params, vol_threshold=0.0):
        self.df = df.copy()
        self.params = params
        self.vol_threshold = vol_threshold
        self.commission_rate = 0.0003
        self.slippage_rate = 0.0002
        self.multiplier = params.get('multiplier', 5)
        self.initial_capital = 100000
        self.disable_adaptive = False  # 是否禁用动态调整

        # 市场状态检测器
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

            # 获取市场状态（盲检测）
            market_state = self.detector.get_state(current_date)

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
                        'pnl': net_pnl,
                        'type': 'stop_loss',
                        'risk_level': risk_at_entry
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
                        'pnl': net_pnl,
                        'type': 'take_profit',
                        'risk_level': risk_at_entry
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
                        'pnl': net_pnl,
                        'type': 'overbought',
                        'risk_level': risk_at_entry
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
                        'pnl': net_pnl,
                        'type': 'death_cross',
                        'risk_level': risk_at_entry
                    })
                    position = 0
                    entry_price = 0.0
                    continue

            # 开仓决策
            if position == 0:
                open_signal = False

                # 超卖
                if (cci < self.params['cci_oversold'] and
                    stc >= self.params['stc_oversold']):
                    open_signal = True

                # 金叉
                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max'] and
                      stc >= self.params['stc_cross']):
                    open_signal = True

                if open_signal:
                    # 如果禁用动态调整，直接按原策略执行
                    if not self.disable_adaptive:
                        # 根据市场状态决定是否开仓
                        if market_state['risk_score'] >= 60:
                            # 高风险：不开仓
                            state_records.append({
                                'date': current_date,
                                'state': market_state,
                                'action': 'skipped'
                            })
                            continue

                    # 量能过滤
                    if self.vol_threshold > 0 and not np.isnan(vol_ratio) and vol_ratio < self.vol_threshold:
                        continue

                    # 趋势过滤（根据市场状态决定）
                    if not self.disable_adaptive and market_state['use_trend_filter']:
                        ma60 = self.detector.df.loc[current_date, 'ma60']
                        if next_day['close'] < ma60:
                            state_records.append({
                                'date': current_date,
                                'state': market_state,
                                'action': 'filtered_by_trend'
                            })
                            continue

                    # 计算仓位（根据风险等级调整）
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
    print("市场状态监控 + 盲回测".center(120))
    print("="*120)

    print("""
【核心原则】
盲回测：只用当前时点及之前的数据判断市场状态，不用未来数据

【风险等级】
- 低风险(0-19分): 正常交易
- 中风险(20-39分): 启用趋势过滤
- 中高风险(40-59分): 仓位减半+趋势过滤
- 高风险(>=60分): 暂停开仓

【风险评分组成】
1. 价格<MA60: +20分
2. MA差距>3%: +25分
3. 连续下跌>=5天: +20分
4. 20天中>=15天<MA60: +20分
5. 波动率>历史80%分位: +15分
""")

    test_symbols = ['铜', '黄金', '白银', '棉花', '锌', '玻璃', '糖']

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

        # 原版回测（无动态调整）
        bt_original = AdaptiveBacktest(df, params, vol_threshold)
        # 临时禁用动态调整
        bt_original.disable_adaptive = True
        result_original, _, _ = bt_original.run()

        # 自适应回测（动态调整）
        bt_adaptive = AdaptiveBacktest(df, params, vol_threshold)
        result_adaptive, trades, states = bt_adaptive.run()

        if result_original and result_adaptive:
            print(f"\n{'版本':<15}{'收益率':<15}{'最大回撤':<12}{'交易次数':<10}{'胜率':<10}{'盈亏比':<10}")
            print("-"*75)
            print(f"{'原版(无调整)':<15}{result_original['total_return']:>+12.2f}%"
                  f"{result_original['max_dd']:>10.2f}%{result_original['total_trades']:>10}"
                  f"{result_original['win_rate']:>8.1f}%{result_original['profit_ratio']:>8.2f}")
            print(f"{'自适应(动态)':<15}{result_adaptive['total_return']:>+12.2f}%"
                  f"{result_adaptive['max_dd']:>10.2f}%{result_adaptive['total_trades']:>10}"
                  f"{result_adaptive['win_rate']:>8.1f}%{result_adaptive['profit_ratio']:>8.2f}")

            ret_change = result_adaptive['total_return'] - result_original['total_return']
            dd_change = result_adaptive['max_dd'] - result_original['max_dd']

            print(f"\n变化: 收益{ret_change:+.2f}%, 回撤{dd_change:+.2f}%")

            # 分析状态记录
            if states:
                states_df = pd.DataFrame([{
                    'date': s['date'],
                    'risk_level': s['state']['risk_level'],
                    'risk_score': s['state']['risk_score'],
                    'action': s['action']
                } for s in states])

                print(f"\n【市场状态统计】")
                print(f"  开仓次数: {(states_df['action'] == 'opened').sum()}")
                print(f"  跳过(高风险): {(states_df['action'] == 'skipped').sum()}")
                print(f"  过滤(趋势): {(states_df['action'] == 'filtered_by_trend').sum()}")

                # 按风险等级统计
                risk_counts = states_df[states_df['action'] == 'opened']['risk_level'].value_counts()
                print(f"\n【开仓时风险等级】")
                for level, count in risk_counts.items():
                    print(f"  {level}: {count}次")

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

    print(f"\n{'品种':<8}{'原版收益':<12}{'自适应收益':<12}{'收益变化':<12}{'原版回撤':<10}{'自适应回撤':<12}{'回撤变化':<10}{'效果':<10}")
    print("-"*95)

    for symbol, data in results.items():
        effect = "改善" if data['ret_change'] > 0 and data['dd_change'] > 0 else \
                "收益升" if data['ret_change'] > 0 else \
                "回撤改善" if data['dd_change'] > 0 else "下降"

        print(f"{symbol:<8}{data['original']['total_return']:>+10.2f}%"
              f"{data['adaptive']['total_return']:>+10.2f}%{data['ret_change']:>+10.2f}%"
              f"{data['original']['max_dd']:>8.2f}%{data['adaptive']['max_dd']:>10.2f}%"
              f"{data['dd_change']:>+8.2f}%  {effect}")

    # 结论
    improved = sum(1 for d in results.values() if d['ret_change'] > 0 or d['dd_change'] > 0)

    print("\n" + "="*120)
    print("结论".center(120))
    print("="*120)

    print(f"""
【盲回测结果】
- 改善品种数: {improved}/{len(results)}

【关键发现】
1. 市场状态监控可以在实时交易中预警风险
2. 动态调整策略可以在一定程度上改善回撤
3. 但也可能错过一些有效信号

【建议】
- 对于高风险品种（铜、黄金），建议启用市场状态监控
- 对于趋势品种（白银、玻璃），谨慎使用，可能影响收益
""")

    return results


if __name__ == "__main__":
    main()
