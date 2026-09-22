"""
检查CCI回测逻辑是否正确
"""

print("="*80)
print("回测逻辑检查".center(80))
print("="*80)

print("\n【当前回测逻辑】")
print("-"*80)
print("""
时间轴：
第i天：
  - 盘中: 高开低收数据产生
  - 收盘后: 计算CCI[i], CCI_MA[i]
  - 判断: 如果CCI[i-1] <= CCI_MA[i-1] 且 CCI[i] > CCI_MA[i] → 金叉

第i+1天：
  - 开盘: 用opens[i+1]买入开仓
  - 收盘后: 计算浮动盈亏（用closes[i+1]）

第j天（死叉日）：
  - 收盘后: 发现CCI[j-1] >= CCI_MA[j-1] 且 CCI[j] <= CCI_MA[j] → 死叉

第j+1天：
  - 开盘: 用opens[j+1]卖出平仓
""")

print("\n【潜在问题检查】")
print("-"*80)

print("\n1. 未来函数检查:")
print("   [X] 问题: CCI计算使用了当天的close，但在收盘后才能计算")
print("   [OK] 实际: 这是正确的！收盘后计算，次日开盘执行")
print("   [OK] 没有未来函数")

print("\n2. 持仓盈亏计算:")
print("   当前代码: val += (closes[i+1] - entry_price) * position * 5")
print("   问题: 第i天金叉，第i+1天开仓，用closes[i+1]计算盈亏")
print("   ✓ 正确！开仓当天用收盘价计算")

print("\n3. 交易次数对比:")
print("   原版CCI(20,14): 33笔交易")
print("   优化版CCI(24,5): 让我重新测试看看交易次数")

# 让我重新测试一下，并记录每笔交易
import pandas as pd
import numpy as np
import pickle

with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

def calculate_cci(df, length=20):
    data = df.copy()
    data['tp'] = (data['high'] + data['low'] + data['close']) / 3
    data['tp_ma'] = data['tp'].rolling(window=length).mean()
    data['tp_dev'] = data['tp'].rolling(window=length).std()
    data['cci'] = (data['tp'] - data['tp_ma']) / (0.015 * data['tp_dev'])
    data['cci'] = data['cci'].fillna(0)
    return data

def backtest_cci_detailed(df, cci_length=20, ma_length=14):
    """详细回测，记录每笔交易"""
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
                    'pnl_pct': (exit_price - entry_price) / entry_price * 100
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
print("详细回测分析 - 锡期货测试集".center(80))
print("="*80)

# 测试默认参数
print("\n【默认参数 CCI(20,14)】")
trades_default, equity_default, balance_default = backtest_cci_detailed(test_df, 20, 14)
print(f"交易次数: {len(trades_default)}笔")
print(f"最终余额: {balance_default:.2f}")
print(f"收益率: {(balance_default - 100000) / 100000 * 100:+.2f}%")

if trades_default:
    winning = len([t for t in trades_default if t['pnl'] > 0])
    print(f"胜率: {winning/len(trades_default)*100:.1f}%")

    print(f"\n前5笔交易:")
    for i, t in enumerate(trades_default[:5], 1):
        print(f"  {i}. {t['entry_date'].strftime('%Y-%m-%d')} 买入 @ {t['entry_price']:.2f} → "
              f"{t['exit_date'].strftime('%Y-%m-%d')} 卖出 @ {t['exit_price']:.2f} | "
              f"盈亏: {t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%)")

# 测试优化参数
print("\n" + "="*80)
print("【优化参数 CCI(24,5)】")
trails_opt, equity_opt, balance_opt = backtest_cci_detailed(test_df, 24, 5)
print(f"交易次数: {len(trails_opt)}笔")
print(f"最终余额: {balance_opt:.2f}")
print(f"收益率: {(balance_opt - 100000) / 100000 * 100:+.2f}%")

if trails_opt:
    winning = len([t for t in trails_opt if t['pnl'] > 0])
    print(f"胜率: {winning/len(trails_opt)*100:.1f}%")

    print(f"\n前5笔交易:")
    for i, t in enumerate(trails_opt[:5], 1):
        print(f"  {i}. {t['entry_date'].strftime('%Y-%m-%d')} 买入 @ {t['entry_price']:.2f} → "
              f"{t['exit_date'].strftime('%Y-%m-%d')} 卖出 @ {t['exit_price']:.2f} | "
              f"盈亏: {t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%)")

print("\n" + "="*80)
print("【对比分析】")
print("-"*80)
print(f"{'参数':<20} {'交易次数':<12} {'收益率':<12} {'胜率':<10}")
print("-"*80)
print(f"{'CCI(20,14)':<20} {len(trades_default):<12} {(balance_default-100000)/100000*100:+.2f}% "
      f"{len([t for t in trades_default if t['pnl']>0])/len(trades_default)*100:.1f}%")
print(f"{'CCI(24,5)':<20} {len(trails_opt):<12} {(balance_opt-100000)/100000*100:+.2f}% "
      f"{len([t for t in trails_opt if t['pnl']>0])/len(trails_opt)*100:.1f}%")

print("\n" + "="*80)
