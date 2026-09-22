"""
FTT策略 - 数据质量影响对比测试
===============================
对比：原始数据 vs 修复后数据

目的：验证OHLC逻辑异常等数据问题对回测结果的实际影响
"""

import pandas as pd
import numpy as np
import pickle
import copy

print("="*80)
print("FTT策略 - 数据质量影响对比测试".center(80))
print("="*80)

# ============ 1. 加载原始数据 ============
print("\n正在加载原始数据...")
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL_ORIGINAL = pickle.load(f)

df_original = DATA_POOL_ORIGINAL['cu'].copy()
print(f"原始数据: {len(df_original)}根K线")
print(f"时间范围: {df_original.index[0]} ~ {df_original.index[-1]}")

# 检查原始数据的OHLC异常
ohlc_issues_original = (df_original['high'] < df_original['open']).sum() + \
                       (df_original['high'] < df_original['close']).sum() + \
                       (df_original['low'] > df_original['open']).sum() + \
                       (df_original['low'] > df_original['close']).sum()
print(f"OHLC逻辑异常: {ohlc_issues_original}个")

# ============ 2. 创建修复后的数据副本 ============
print("\n正在创建修复后数据...")
df_fixed = df_original.copy()

# 修复OHLC逻辑
df_fixed['high'] = df_fixed[['high', 'open', 'close']].max(axis=1)
df_fixed['low'] = df_fixed[['low', 'open', 'close']].min(axis=1)

# 重新计算ATR
df_fixed['tr'] = np.maximum(df_fixed['high'] - df_fixed['low'],
                            np.abs(df_fixed['high'] - df_fixed['close'].shift(1)))
df_fixed['tr'] = np.maximum(df_fixed['tr'],
                            np.abs(df_fixed['low'] - df_fixed['close'].shift(1)))
df_fixed['atr'] = df_fixed['tr'].rolling(14).mean()
df_fixed['atr_pct'] = df_fixed['atr'] / df_fixed['close'] * 100
df_fixed = df_fixed.drop(columns=['tr'])

ohlc_issues_fixed = (df_fixed['high'] < df_fixed['open']).sum() + \
                    (df_fixed['high'] < df_fixed['close']).sum() + \
                    (df_fixed['low'] > df_fixed['open']).sum() + \
                    (df_fixed['low'] > df_fixed['close']).sum()

print(f"修复后OHLC异常: {ohlc_issues_fixed}个")
print(f"修复异常数: {ohlc_issues_original - ohlc_issues_fixed}个")

# ============ 3. FTT策略回测函数 ============
def run_ftt_backtest(df, dataset_name="数据集"):
    """运行FTT策略回测"""
    data = df.copy()

    # 计算指标
    data['ema_fast'] = data['close'].ewm(span=5, adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=15, adjust=False).mean()

    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd_dif'] = exp1 - exp2
    data['macd_dea'] = data['macd_dif'].ewm(span=9, adjust=False).mean()
    data['ratio'] = data.apply(lambda x: x['macd_dif'] / x['macd_dea']
                               if x['macd_dea'] != 0 else 0, axis=1)

    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # STC
    stc_macd = data['close'].ewm(span=23, adjust=False).mean() - \
               data['close'].ewm(span=50, adjust=False).mean()
    stoch_period = 10
    min_macd = stc_macd.rolling(window=stoch_period).min()
    max_macd = stc_macd.rolling(window=stoch_period).max()
    denom = max_macd - min_macd
    denom = denom.replace(0, np.nan)
    stoch_k = 100 * (stc_macd - min_macd) / denom
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=3).mean()
    min_stoch_d = stoch_d.rolling(window=stoch_period).min()
    max_stoch_d = stoch_d.rolling(window=stoch_period).max()
    denom_2 = max_stoch_d - min_stoch_d
    denom_2 = denom_2.replace(0, np.nan)
    stc_raw = 100 * (stoch_d - min_stoch_d) / denom_2
    stc_raw = stc_raw.fillna(50)
    data['stc'] = stc_raw.rolling(window=3).mean()

    # 回测
    balance = 100000
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    closes = data['close'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    ema_fast = data['ema_fast'].values
    ema_slow = data['ema_slow'].values
    dif = data['macd_dif'].values
    ratio = data['ratio'].values
    rsi = data['rsi'].values
    stc = data['stc'].values

    for i in range(70, len(data)-1):
        price = closes[i]
        low_price = lows[i]
        next_open = opens[i+1]

        # 获取前一天的值
        ema_fast_prev = ema_fast[i-1]
        ema_slow_prev = ema_slow[i-1]
        prev_dif = dif[i-1]
        prev_ratio = ratio[i-1]
        prev_stc = stc[i-1]

        # 平仓
        if position > 0:
            stop_price = entry_price * 0.98

            if low_price <= stop_price:
                exit_p = stop_price
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append({'pnl': pnl, 'type': 'stop_loss'})
                position = 0

            elif (prev_stc > 90) and (stc[i] < prev_stc):
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append({'pnl': pnl, 'type': 'stc_profit'})
                position = 0

            elif ema_fast[i] < ema_slow[i]:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append({'pnl': pnl, 'type': 'trend_end'})
                position = 0

        # 开仓
        if position == 0:
            trend_up = (ema_fast[i] > ema_slow[i])

            ratio_safe = (0 < ratio[i] < 1.15)
            ratio_shrinking = (ratio[i] < prev_ratio)
            turning_up = (dif[i] > prev_dif)
            is_strong = (rsi[i] > 50)

            sniper_entry = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong

            ema_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast[i] > ema_slow[i])
            chase_entry = ema_cross and is_strong

            if sniper_entry or chase_entry:
                risk_amt = balance * 0.02
                stop_dist = price * 0.02
                if stop_dist > 0:
                    qty = int(risk_amt / (stop_dist * 5))
                    qty = max(1, min(qty, int(balance * 2.0 / (next_open * 5))))
                    position = qty
                    entry_price = next_open

        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * 5
        equity.append(val)

    # 统计
    if not trades:
        return {'return_pct': 0, 'max_dd': 0, 'total_trades': 0,
                'win_rate': 0, 'profit_factor': 0, 'trades': []}

    final_ret = (equity[-1] - 100000) / 100000 * 100
    win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100

    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    max_dd = dd.min()

    winning = [t['pnl'] for t in trades if t['pnl'] > 0]
    losing = [t['pnl'] for t in trades if t['pnl'] <= 0]
    profit_factor = abs(sum(winning) / sum(losing)) if losing else 0

    return {
        'return_pct': final_ret,
        'max_dd': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'trades': trades
    }

# ============ 4. 运行对比测试 ============
print("\n" + "="*80)
print("开始对比测试...".center(80))
print("="*80)

# 原始数据回测
print("\n[测试1] 原始数据回测")
result_original = run_ftt_backtest(df_original)
print(f"  收益率: {result_original['return_pct']:+.2f}%")
print(f"  胜率: {result_original['win_rate']:.1f}%")
print(f"  回撤: {result_original['max_dd']:.2f}%")
print(f"  交易: {result_original['total_trades']}笔")

# 修复后数据回测
print("\n[测试2] 修复后数据回测")
result_fixed = run_ftt_backtest(df_fixed)
print(f"  收益率: {result_fixed['return_pct']:+.2f}%")
print(f"  胜率: {result_fixed['win_rate']:.1f}%")
print(f"  回撤: {result_fixed['max_dd']:.2f}%")
print(f"  交易: {result_fixed['total_trades']}笔")

# ============ 5. 对比分析 ============
print("\n" + "="*80)
print("对比分析结果".center(80))
print("="*80)

print(f"\n{'指标':<15} | {'原始数据':<18} | {'修复后数据':<18} | {'差异':<15}")
print("-"*80)

# 收益率
ret_diff = result_fixed['return_pct'] - result_original['return_pct']
print(f"{'总收益率':<15} | {result_original['return_pct']:>16.2f}% | {result_fixed['return_pct']:>16.2f}% | {ret_diff:>+13.2f}%")

# 胜率
win_diff = result_fixed['win_rate'] - result_original['win_rate']
print(f"{'胜率':<15} | {result_original['win_rate']:>16.1f}% | {result_fixed['win_rate']:>16.1f}% | {win_diff:>+13.1f}%")

# 回撤
dd_diff = result_fixed['max_dd'] - result_original['max_dd']
print(f"{'最大回撤':<15} | {result_original['max_dd']:>16.2f}% | {result_fixed['max_dd']:>16.2f}% | {dd_diff:>+13.2f}%")

# 交易次数
trade_diff = result_fixed['total_trades'] - result_original['total_trades']
print(f"{'交易次数':<15} | {result_original['total_trades']:>16} | {result_fixed['total_trades']:>16} | {trade_diff:>+13d}")

# 盈亏比
pf_diff = result_fixed['profit_factor'] - result_original['profit_factor']
print(f"{'盈亏比':<15} | {result_original['profit_factor']:>16.2f} | {result_fixed['profit_factor']:>16.2f} | {pf_diff:>+13.2f}")

print("-"*80)

# ============ 6. 结论 ============
print("\n" + "="*80)
print("结论".center(80))
print("="*80)

if abs(ret_diff) < 1:
    print("[OK] 数据修复对结果影响很小 (<1%)")
    print("    OHLC异常对FTT策略的实际影响有限")
    print("    原因：FTT策略主要基于close价格，不依赖high/low突破")
elif abs(ret_diff) < 5:
    print("[NOTE] 数据修复对结果有轻微影响")
    print(f"      收益率变化 {ret_diff:.2f}%，在可接受范围内")
else:
    print("[WARN] 数据修复对结果影响较大！")
    print(f"      收益率变化 {ret_diff:.2f}%")
    print("      建议使用修复后的数据进行回测")

# 找出不同的交易
print(f"\n详细对比:")
trades_orig = len(result_original['trades'])
trades_fix = len(result_fixed['trades'])
print(f"  原始数据: {trades_orig}笔交易")
print(f"  修复后:   {trades_fix}笔交易")
print(f"  差异:     {abs(trades_fix - trades_orig)}笔交易")

print("\n" + "="*80)
print("测试完成！")
print("="*80)
