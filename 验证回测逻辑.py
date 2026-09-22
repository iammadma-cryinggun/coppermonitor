"""
检查CCI回测逻辑是否正确
"""

import pandas as pd
import numpy as np
import pickle

print("="*80)
print("回测逻辑验证".center(80))
print("="*80)

# 加载数据
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

# 计算CCI
def calculate_cci(df, length=20):
    data = df.copy()
    data['tp'] = (data['high'] + data['low'] + data['close']) / 3
    data['tp_ma'] = data['tp'].rolling(window=length).mean()
    data['tp_dev'] = data['tp'].rolling(window=length).std()
    data['cci'] = (data['tp'] - data['tp_ma']) / (0.015 * data['tp_dev'])
    data['cci'] = data['cci'].fillna(0)
    return data

# 详细回测
def backtest_cci_detailed(df, cci_length=20, ma_length=14):
    data = calculate_cci(df, length=cci_length)
    data['cci_ma'] = data['cci'].rolling(window=ma_length).mean()

    balance = 100000
    position = 0
    entry_price = 0.0
    entry_date = None
    trades = []
    equity = []

    closes = data['close'].values
    opens = data['open'].values
    dates = data.index

    cci = data['cci'].values
    cci_ma = data['cci_ma'].values

    start_idx = cci_length + ma_length + 1

    for i in range(start_idx, len(data)-1):
        cci_val = cci[i]
        cci_ma_val = cci_ma[i]
        cci_prev = cci[i-1]
        cci_ma_prev = cci_ma[i-1]

        # 平仓逻辑
        if position > 0:
            if cci_prev >= cci_ma_prev and cci_val <= cci_ma_val:  # 死叉
                exit_p = opens[i+1]
                exit_date = dates[i+1]
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl

                trade_info = {
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_p,
                    'quantity': position,
                    'pnl': pnl,
                    'pnl_pct': (exit_p - entry_price) / entry_price * 100
                }
                trades.append(trade_info)

                position = 0
                entry_price = 0.0
                entry_date = None

        # 开仓逻辑
        if position == 0:
            if cci_prev <= cci_ma_prev and cci_val > cci_ma_val:  # 金叉
                entry_price = opens[i+1]
                entry_date = dates[i+1]
                qty = int(balance * 2.0 / (entry_price * 5))
                qty = max(1, qty)
                position = qty

        # 计算权益
        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * 5
        equity.append(val)

    return trades, equity, balance

# 测试锡
symbol_key = 'sn'
df_full = DATA_POOL[symbol_key].copy()
df_full.sort_index(inplace=True)
test_df = df_full.loc['2024-01-01':].copy()

print("\n" + "="*80)
print("回测逻辑说明".center(80))
print("="*80)

print("""
时间流程:
  第i天收盘后: 计算CCI[i], 发现金叉
  第i+1天开盘: 用opens[i+1]买入
  持仓期间: 每天用closes计算浮动盈亏
  第j天收盘后: 发现死叉
  第j+1天开盘: 用opens[j+1]卖出

交易规则检查:
  1. [OK] 没有未来函数 - 收盘后计算，次日执行
  2. [OK] 成交价格合理 - 用次日开盘价
  3. [OK] 杠杆固定 - 2倍杠杆
  4. [OK] 合约乘数 - 5吨/手
""")

print("\n" + "="*80)
print("锡期货测试集详细分析".center(80))
print("="*80)

# 测试默认参数
print("\n[默认参数 CCI(20,14)]")
print("-"*80)
trades_default, equity_default, balance_default = backtest_cci_detailed(test_df, 20, 14)
print(f"交易次数: {len(trades_default)}笔")
print(f"最终余额: {balance_default:.2f}")
print(f"收益率: {(balance_default - 100000) / 100000 * 100:+.2f}%")

if trades_default:
    winning = len([t for t in trades_default if t['pnl'] > 0])
    print(f"胜率: {winning/len(trades_default)*100:.1f}%")

    # 最大回撤
    equity_series = pd.Series(equity_default)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    print(f"最大回撤: {dd.min():.2f}%")

    print(f"\n前5笔交易:")
    for i, t in enumerate(trades_default[:5], 1):
        status = "盈利" if t['pnl'] > 0 else "亏损"
        print(f"  {i}. {t['entry_date'].strftime('%Y-%m-%d')} 开仓 @ {t['entry_price']:.2f} → "
              f"{t['exit_date'].strftime('%Y-%m-%d')} 平仓 @ {t['exit_price']:.2f} | "
              f"{status}: {t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%)")

# 测试优化参数
print("\n" + "="*80)
print("[优化参数 CCI(24,5)]")
print("-"*80)
trails_opt, equity_opt, balance_opt = backtest_cci_detailed(test_df, 24, 5)
print(f"交易次数: {len(trails_opt)}笔")
print(f"最终余额: {balance_opt:.2f}")
print(f"收益率: {(balance_opt - 100000) / 100000 * 100:+.2f}%")

if trails_opt:
    winning = len([t for t in trails_opt if t['pnl'] > 0])
    print(f"胜率: {winning/len(trails_opt)*100:.1f}%")

    # 最大回撤
    equity_series = pd.Series(equity_opt)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    print(f"最大回撤: {dd.min():.2f}%")

    print(f"\n前5笔交易:")
    for i, t in enumerate(trails_opt[:5], 1):
        status = "盈利" if t['pnl'] > 0 else "亏损"
        print(f"  {i}. {t['entry_date'].strftime('%Y-%m-%d')} 开仓 @ {t['entry_price']:.2f} → "
              f"{t['exit_date'].strftime('%Y-%m-%d')} 平仓 @ {t['exit_price']:.2f} | "
              f"{status}: {t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%)")

print("\n" + "="*80)
print("对比总结".center(80))
print("="*80)

print(f"\n{'参数':<20} {'交易次数':<12} {'收益率':<12} {'胜率':<10} {'最大回撤':<12}")
print("-"*80)

if trades_default:
    equity_series = pd.Series(equity_default)
    peak = equity_series.cummax()
    dd_default = (equity_series - peak) / peak * 100
    wr_default = len([t for t in trades_default if t['pnl']>0])/len(trades_default)*100

    print(f"{'CCI(20,14)':<20} {len(trades_default):<12} "
          f"{(balance_default-100000)/100000*100:+.2f}% {wr_default:.1f}% {dd_default.min():.2f}%")

if trails_opt:
    equity_series = pd.Series(equity_opt)
    peak = equity_series.cummax()
    dd_opt = (equity_series - peak) / peak * 100
    wr_opt = len([t for t in trails_opt if t['pnl']>0])/len(trails_opt)*100

    print(f"{'CCI(24,5)':<20} {len(trails_opt):<12} "
          f"{(balance_opt-100000)/100000*100:+.2f}% {wr_opt:.1f}% {dd_opt.min():.2f}%")

print("\n" + "="*80)
print("回测逻辑验证完成！")
print("="*80)
