"""
连续亏损分析 - 分析最大回撤期间的交易序列
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')
from cci_calculations import calculate_cci_tv, calculate_stc, f_normalize


def analyze_losing_streak():
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

    # 回测
    balance = 100000
    position = 0
    entry_price = 0.0
    entry_date = None
    multiplier = 1000

    all_trades = []
    equity_curve = []
    equity_dates = []

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        cci = current['cci']
        cci_ma = current['cci_ma']
        stc = current['stc']
        cci_prev = df.iloc[i-1]['cci'] if i > 0 else cci
        cci_ma_prev = df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

        if position > 0:
            exit_type = None
            exit_price = None

            # 止损
            if next_day['low'] <= entry_price * 0.96:
                stop_price = max(next_day['open'], entry_price * 0.96)
                exit_price = stop_price * (1 - 0.0002)
                exit_type = 'stop_loss'

            # 止盈
            elif next_day['high'] >= entry_price * 1.20:
                take_profit_price = min(next_day['open'], entry_price * 1.20)
                exit_price = take_profit_price * (1 - 0.0002)
                exit_type = 'take_profit'

            # CCI超买平仓
            elif cci > params['cci_overbought']:
                exit_price = next_day['close'] * (1 - 0.0002)
                exit_type = 'overbought'

            # CCI死叉平仓
            elif cci_prev >= cci_ma_prev and cci < cci_ma:
                exit_price = next_day['close'] * (1 - 0.0002)
                exit_type = 'death_cross'

            if exit_price is not None:
                pnl = (exit_price - entry_price) * position * multiplier
                commission = exit_price * position * multiplier * 0.0003 * 2
                net_pnl = pnl - commission
                balance += net_pnl

                all_trades.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'type': exit_type,
                    'balance_after': balance
                })

                position = 0
                entry_price = 0.0

        # 开仓
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
                entry_date = next_day.name

        # 更新权益
        val = balance
        if position > 0:
            unrealized_pnl = (next_day['close'] - entry_price) * position * multiplier
            val += unrealized_pnl
        equity_curve.append(val)
        equity_dates.append(next_day.name)

    # 分析
    trades_df = pd.DataFrame(all_trades)
    equity_series = pd.Series(equity_curve, index=equity_dates)

    print("="*120)
    print("黄金连续亏损分析".center(120))
    print("="*120)

    # 找到最大回撤期间
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    max_dd_idx = dd.idxmin()
    peak_date = peak.loc[:max_dd_idx].idxmax()

    print(f"\n【最大回撤区间】")
    print(f"  前期高点: {peak_date.strftime('%Y-%m-%d')} (${equity_series.loc[peak_date]:,.2f})")
    print(f"  最大回撤: {max_dd_idx.strftime('%Y-%m-%d')} (${equity_series.loc[max_dd_idx]:,.2f})")
    print(f"  回撤幅度: {dd.min():.2f}%")

    # 筛选这个期间的交易
    period_trades = trades_df[(trades_df['exit_date'] >= peak_date) &
                               (trades_df['exit_date'] <= max_dd_idx)]

    print(f"\n【期间交易统计】")
    print(f"  交易次数: {len(period_trades)}")
    print(f"  盈利次数: {len(period_trades[period_trades['pnl'] > 0])}")
    print(f"  亏损次数: {len(period_trades[period_trades['pnl'] <= 0])}")
    print(f"  总盈亏: ${period_trades['pnl'].sum():,.2f}")

    # 按类型统计
    print(f"\n【按平仓类型统计】")
    for exit_type in period_trades['type'].unique():
        type_trades = period_trades[period_trades['type'] == exit_type]
        print(f"  {exit_type}: {len(type_trades)}次, 平均${type_trades['pnl'].mean():,.2f}, 总计${type_trades['pnl'].sum():,.2f}")

    # 显示所有亏损交易
    losing_trades = period_trades[period_trades['pnl'] < 0].sort_values('pnl')

    print(f"\n【期间所有亏损交易（按金额排序）】")
    for idx, row in losing_trades.head(15).iterrows():
        print(f"  {row['exit_date'].strftime('%Y-%m-%d')} | {row['type']:<15} | "
              f"${row['pnl']:>+10,.2f} ({row['pnl_pct']:>+.2f}%) | "
              f"余额: ${row['balance_after']:>10,.2f}")

    # 计算连续亏损
    print(f"\n【连续亏损序列】")
    period_trades_sorted = period_trades.sort_values('exit_date')

    consecutive_losses = []
    current_streak = []

    for idx, row in period_trades_sorted.iterrows():
        if row['pnl'] < 0:
            current_streak.append(row)
        else:
            if len(current_streak) > 0:
                consecutive_losses.append(current_streak.copy())
                current_streak = []

    if len(current_streak) > 0:
        consecutive_losses.append(current_streak)

    # 显示最长的连续亏损
    longest_streak = max(consecutive_losses, key=len) if consecutive_losses else []
    print(f"  最长连续亏损次数: {len(longest_streak)}")

    if longest_streak:
        total_loss = sum(t['pnl'] for t in longest_streak)
        print(f"  累计亏损: ${total_loss:,.2f}")
        print(f"\n  详细:")
        for t in longest_streak[:10]:
            print(f"    {t['exit_date'].strftime('%Y-%m-%d')} | {t['type']:<15} | ${t['pnl']:>+10,.2f}")

    # 资金曲线变化
    print(f"\n【资金变化过程】")
    period_equity = equity_series.loc[peak_date:max_dd_idx]

    # 每10%的时间输出一次
    n_points = len(period_equity)
    for i in range(0, n_points, max(1, n_points // 10)):
        date = period_equity.index[i]
        val = period_equity.iloc[i]
        peak_val = peak.loc[date]
        dd_val = (val - peak_val) / peak_val * 100
        print(f"  {date.strftime('%Y-%m-%d')}: ${val:>12,.2f} | 回撤: {dd_val:>+6.2f}%")

    # 结论
    print("\n" + "="*120)
    print("结论".center(120))
    print("="*120)

    total_period_loss = period_trades['pnl'].sum()
    print(f"""
回撤来源:
1. 期间交易总亏损: ${total_period_loss:,.2f}
2. 这不是单笔大亏损，而是多笔小亏损的累积
3. 死叉平仓和止损都是4%左右的小亏损
4. 但连续多次4%亏损 = 大幅回撤

改进建议:
1. 增加最大回撤止损: 当账户回撤超过20%时，暂停交易
2. 降低仓位: 在连续亏损后降低杠杆
3. 增加过滤条件: 在不利市场环境下减少交易频率
""")


if __name__ == "__main__":
    analyze_losing_streak()
