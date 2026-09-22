"""
分析为什么风控导致收益率大幅下降
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')
from cci_calculations import calculate_cci_tv, calculate_stc, f_normalize


def analyze_why_return_dropped():
    from 最优参数配置 import OPTIMAL_PARAMS

    params = OPTIMAL_PARAMS['黄金']

    # 读取黄金数据
    data_dir = "D:\\期货数据\\铜期货监控\\global_futures_daily"
    df = pd.read_csv(os.path.join(data_dir, "黄金_daily.csv"))
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    # 计算指标
    cci_length = params['cci_length']
    ma_length = params['ma_length']

    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=cci_length).mean()
    mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()

    stc_raw = calculate_stc(df, length=10, fast=23, slow=50, aaa=0.5)
    df['stc'] = f_normalize(stc_raw, 20, 80)

    # 正常回测
    balance = 100000
    position = 0
    entry_price = 0.0
    multiplier = 1000

    equity_curve = []
    equity_dates = []
    signal_dates = []  # 记录所有交易信号

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        cci = current['cci']
        cci_ma = current['cci_ma']
        stc = current['stc']
        cci_prev = df.iloc[i-1]['cci'] if i > 0 else cci
        cci_ma_prev = df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

        # 记录交易信号
        if position == 0:
            open_signal = False
            if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
                open_signal = True
            elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                  cci <= params['cci_cross_max'] and
                  stc >= params['stc_cross']):
                open_signal = True

            if open_signal:
                signal_dates.append({
                    'date': next_day.name,
                    'type': 'oversold' if cci < params['cci_oversold'] else 'golden_cross',
                    'price': next_day['open']
                })

        if position > 0:
            if next_day['low'] <= entry_price * 0.96:
                stop_price = max(next_day['open'], entry_price * 0.96)
                exit_price = stop_price * (1 - 0.0002)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * 0.0003 * 2
                balance += pnl - commission
                position = 0
                entry_price = 0.0
                continue

            if next_day['high'] >= entry_price * 1.20:
                take_profit_price = min(next_day['open'], entry_price * 1.20)
                exit_price = take_profit_price * (1 - 0.0002)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * 0.0003 * 2
                balance += pnl - commission
                position = 0
                entry_price = 0.0
                continue

            if cci > params['cci_overbought']:
                exit_price = next_day['close'] * (1 - 0.0002)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * 0.0003 * 2
                balance += pnl - commission
                position = 0
                entry_price = 0.0
                continue

            if cci_prev >= cci_ma_prev and cci < cci_ma:
                exit_price = next_day['close'] * (1 - 0.0002)
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * 0.0003 * 2
                balance += pnl - commission
                position = 0
                entry_price = 0.0
                continue

        if position == 0:
            open_signal = False
            if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
                open_signal = True
            elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                  cci <= params['cci_cross_max'] and
                  stc >= params['stc_cross']):
                open_signal = True

            if open_signal:
                entry_price = next_day['open'] * (1 + 0.0002)
                max_value = balance * 0.9 * 2
                qty = int(max_value / (entry_price * multiplier))
                qty = max(1, qty)
                commission = entry_price * qty * multiplier * 0.0003
                balance -= commission
                position = qty

        val = balance
        if position > 0:
            unrealized_pnl = (next_day['close'] - entry_price) * position * multiplier
            val += unrealized_pnl
        equity_curve.append(val)
        equity_dates.append(next_day.name)

    # 分析
    equity_series = pd.Series(equity_curve, index=equity_dates)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100

    print("="*120)
    print("为什么风控导致收益率大幅下降？".center(120))
    print("="*120)

    print("\n【核心问题：峰值回撤的陷阱】")
    print("""
风控使用的是：当前回撤 = (当前权益 - 历史最高权益) / 历史最高权益

问题：
1. 策略前期大涨，权益从$100,000涨到$300,000
2. 然后开始回撤，权益跌到$200,000
3. 此时回撤 = ($200,000 - $300,000) / $300,000 = -33.3%
4. 虽然账户还赚了100%，但按20%阈值，系统认为"风险太大"，暂停交易
5. 后续的反弹交易全部错过！
""")

    print("【实际数据分析】")

    # 找到权益超过初始2倍后，回撤超过20%的期间
    high_equity_periods = equity_series[equity_series > 100000 * 2]

    if len(high_equity_periods) > 0:
        first_high = high_equity_periods.index[0]
        print(f"\n权益首次翻倍日期: {first_high.strftime('%Y-%m-%d')}")

    # 找到回撤超过20%的期间
    dd_over_20 = dd[dd <= -20]

    if len(dd_over_20) > 0:
        print(f"回撤超过20%的交易日数: {len(dd_over_20)}")
        print(f"首次超过20%回撤: {dd_over_20.index[0].strftime('%Y-%m-%d')}")
        print(f"最后超过20%回撤: {dd_over_20.index[-1].strftime('%Y-%m-%d')}")

    # 统计在这些期间有多少交易信号
    signals_in_dd = [s for s in signal_dates if s['date'] in dd_over_20.index]
    print(f"\n在回撤>20%期间的交易信号数: {len(signals_in_dd)}")

    # 统计这些期间的权益情况
    print("\n【关键数据】")
    for date in dd_over_20.index[::max(1, len(dd_over_20)//10)][:10]:
        eq = equity_series.loc[date]
        dd_val = dd.loc[date]
        profit_vs_initial = (eq - 100000) / 100000 * 100
        print(f"  {date.strftime('%Y-%m-%d')}: 权益 ${eq:>10,.0f} | "
              f"峰值回撤 {dd_val:>+6.1f}% | 相对初始 {profit_vs_initial:>+6.1f}%")

    print("\n" + "="*120)
    print("问题根源".center(120))
    print("="*120)

    print("""
风控逻辑的问题：
  1. 使用"峰值回撤"作为指标
  2. 策略大赚后即使回撤，账户仍然远高于初始资金
  3. 但风控认为"回撤太大"，暂停交易
  4. 错过了后续的反弹机会

例如：
  - 初始资金: $100,000
  - 峰值权益: $300,000（+200%）
  - 当前权益: $200,000（相对于峰值-33%）
  - 相对于初始资金: +100%

  风控认为"回撤33%太危险，暂停交易"
  但实际上账户还赚了100%！

  结果：错过了从$200,000涨回$300,000的机会
""")

    print("\n" + "="*120)
    print("解决方案".center(120))
    print("="*120)

    print("""
方案1: 使用"相对初始资金"的风控
  - 只有当权益低于初始资金的某个比例（如80%）时才暂停
  - 即权益 < $80,000 时暂停
  - 优点：允许策略充分发挥，只在真正亏损时干预

方案2: 分阶段风控
  - 赚钱后提高风控阈值
  - 赚50%后，允许回撤30%
  - 赚100%后，允许回撤40%
  - 优点：动态调整，更灵活

方案3: 回撤恢复后快速重启
  - 回撤恢复到15%时立即恢复交易（不是等到20%）
  - 减少错过交易的天数

方案4: 不对黄金使用风控
  - 黄金的回撤模式不适合这种风控
  - 选择其他回撤更可控的品种
""")


if __name__ == "__main__":
    analyze_why_return_dropped()
