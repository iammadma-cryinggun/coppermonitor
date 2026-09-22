"""
诊断黄金回撤问题
===============
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')
from cci_calculations import calculate_cci_tv, calculate_stc, f_normalize


def diagnose_gold():
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

    equity_curve = []
    trade_records = []

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        cci = current['cci']
        cci_ma = current['cci_ma']
        stc = current['stc']
        cci_prev = df.iloc[i-1]['cci'] if i > 0 else cci
        cci_ma_prev = df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

        if position > 0:
            # 止损
            if next_day['low'] <= entry_price * 0.96:
                stop_price = max(next_day['open'], entry_price * 0.96)
                exit_price = stop_price * (1 - 0.0002)  # 滑点
                pnl = (exit_price - entry_price) * position * 1000
                commission = exit_price * position * 1000 * 0.0003 * 2
                net_pnl = pnl - commission
                balance += net_pnl

                trade_records.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'type': 'stop_loss'
                })

                position = 0
                entry_price = 0.0
                continue

            # 止盈
            if next_day['high'] >= entry_price * 1.20:
                take_profit_price = min(next_day['open'], entry_price * 1.20)
                exit_price = take_profit_price * (1 - 0.0002)
                pnl = (exit_price - entry_price) * position * 1000
                commission = exit_price * position * 1000 * 0.0003 * 2
                net_pnl = pnl - commission
                balance += net_pnl

                trade_records.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'type': 'take_profit'
                })

                position = 0
                entry_price = 0.0
                continue

            # CCI超买平仓
            if cci > params['cci_overbought']:
                exit_price = next_day['close'] * (1 - 0.0002)
                pnl = (exit_price - entry_price) * position * 1000
                commission = exit_price * position * 1000 * 0.0003 * 2
                net_pnl = pnl - commission
                balance += net_pnl

                trade_records.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'type': 'overbought'
                })

                position = 0
                entry_price = 0.0
                continue

            # CCI死叉平仓
            if cci_prev >= cci_ma_prev and cci < cci_ma:
                exit_price = next_day['close'] * (1 - 0.0002)
                pnl = (exit_price - entry_price) * position * 1000
                commission = exit_price * position * 1000 * 0.0003 * 2
                net_pnl = pnl - commission
                balance += net_pnl

                trade_records.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100,
                    'type': 'death_cross'
                })

                position = 0
                entry_price = 0.0
                continue

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
                entry_price = next_day['open'] * (1 + 0.0002)  # 滑点
                max_value = balance * 0.9 * 2
                qty = int(max_value / (entry_price * 1000))
                qty = max(1, qty)
                commission = entry_price * qty * 1000 * 0.0003
                balance -= commission
                position = qty
                entry_date = next_day.name

        # 更新权益
        val = balance
        if position > 0:
            unrealized_pnl = (next_day['close'] - entry_price) * position * 1000
            val += unrealized_pnl
        equity_curve.append(val)

    # 分析
    print("="*120)
    print("黄金回撤诊断报告".center(120))
    print("="*120)

    # 权益曲线统计
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100

    print(f"\n【权益曲线统计】")
    print(f"  初始资金: $100,000")
    print(f"  最终资金: ${equity_series.iloc[-1]:,.2f}")
    print(f"  总收益率: {(equity_series.iloc[-1] - 100000) / 100000 * 100:+.2f}%")
    print(f"  最大回撤: {dd.min():.2f}%")

    # 找到最大回撤位置
    max_dd_idx = dd.idxmin()
    max_dd_date = df.index[max_dd_idx]
    print(f"  最大回撤发生日期: {max_dd_date.strftime('%Y-%m-%d')}")
    print(f"  最大回撤时权益: ${equity_series.iloc[max_dd_idx]:,.2f}")

    # 交易统计
    trades_df = pd.DataFrame(trade_records)

    print(f"\n【交易统计】")
    print(f"  总交易次数: {len(trades_df)}")
    print(f"  止损次数: {len(trades_df[trades_df['type'] == 'stop_loss'])}")
    print(f"  止盈次数: {len(trades_df[trades_df['type'] == 'take_profit'])}")
    print(f"  超买平仓: {len(trades_df[trades_df['type'] == 'overbought'])}")
    print(f"  死叉平仓: {len(trades_df[trades_df['type'] == 'death_cross'])}")

    # 止损交易分析
    stop_loss_trades = trades_df[trades_df['type'] == 'stop_loss']

    if len(stop_loss_trades) > 0:
        print(f"\n【止损交易详情】")
        print(f"  止损平均亏损: ${stop_loss_trades['pnl'].mean():,.2f}")
        print(f"  止损最大单笔亏损: ${stop_loss_trades['pnl'].min():,.2f}")
        print(f"  止损平均亏损百分比: {stop_loss_trades['pnl_pct'].mean():.2f}%")

    # 找出最大连续亏损
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    trades_df['is_loss'] = trades_df['pnl'] < 0

    # 计算连续亏损
    max_consecutive_losses = 0
    current_losses = 0

    for pnl in trades_df['pnl']:
        if pnl < 0:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0

    print(f"\n【连续亏损】")
    print(f"  最大连续亏损次数: {max_consecutive_losses}")

    # 最严重的5次亏损
    worst_trades = trades_df.nsmallest(5, 'pnl')

    print(f"\n【最严重的5次亏损】")
    for idx, row in worst_trades.iterrows():
        print(f"  {row['entry_date'].strftime('%Y-%m-%d')} -> {row['exit_date'].strftime('%Y-%m-%d')}")
        print(f"    开仓价: {row['entry_price']:.2f}, 平仓价: {row['exit_price']:.2f}")
        print(f"    亏损: ${row['pnl']:,.2f} ({row['pnl_pct']:.2f}%), 类型: {row['type']}")

    # 分析：为什么回撤这么大？
    print(f"\n" + "="*120)
    print("回撤分析".center(120))
    print("="*120)

    print(f"\n【可能原因】")

    # 原因1: 复利效应
    print(f"\n1. 复利效应:")
    print(f"   - 每次亏损后，资金减少，下次开仓手数也会减少")
    print(f"   - 但权益曲线是按实际资金计算的")
    print(f"   - 如果连续亏损，资金会快速缩水")

    # 原因2: 权益计算包含未实现盈亏
    print(f"\n2. 权益计算包含未实现盈亏:")
    print(f"   - 持仓期间，按收盘价计算浮动盈亏")
    print(f"   - 如果持仓期间价格大跌，未实现亏损会很大")
    print(f"   - 即使最终止损只有4%，持仓期间的浮亏可能更大")

    # 检查持仓期间的浮亏
    print(f"\n【持仓期间浮亏分析】")
    print(f"   回撤-87.72%说明：")
    print(f"   - 要么连续多次止损（每次4%）")
    print(f"   - 要么持仓期间浮亏巨大")
    print(f"   - 要么数据或计算有问题")

    # 计算需要连续多少次4%亏损才能达到87.72%
    losses_needed = 0
    temp_balance = 100000
    while temp_balance > 100000 * (1 - 0.8772):
        temp_balance *= 0.96
        losses_needed += 1

    print(f"\n【理论计算】")
    print(f"   连续{losses_needed}次4%止损后，资金剩余: ${temp_balance:,.2f}")
    print(f"   理论回撤: {(temp_balance - 100000) / 100000 * 100:.2f}%")

    # 实际最大连续亏损
    actual_max_losses = trades_df['is_loss'].astype(int).groupby(
        (trades_df['is_loss'] != trades_df['is_loss'].shift()).cumsum()
    ).sum().max()

    print(f"   实际最大连续亏损次数: {actual_max_losses}")


if __name__ == "__main__":
    diagnose_gold()
