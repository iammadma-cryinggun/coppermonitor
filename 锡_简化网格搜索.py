"""
锡(Sn) - 简化网格搜索优化
=======================
只优化最重要的3-4个参数，其他固定
快速找到top参数组合
"""

import pandas as pd
import numpy as np
import pickle
import itertools

print("="*80)
print("锡(Sn) - 简化网格搜索优化".center(80))
print("="*80)

with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df_full = DATA_POOL['sn'].copy()
df_full.sort_index(inplace=True)

train_df = df_full.loc[:'2023-12-31'].copy()
test_df = df_full.loc['2024-01-01':].copy()

# 固定参数（基于铜的参数）
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50

# 只优化最重要的3个参数
ema_fast_grid = [5, 8, 10, 12, 15]
ema_slow_grid = [15, 20, 25, 30, 35]
stc_sell_grid = [85, 90, 95]
stop_pct_grid = [0.015, 0.02, 0.025, 0.03]

print(f"\n简化参数网格:")
print(f"  EMA快线: {ema_fast_grid} (5个值)")
print(f"  EMA慢线: {ema_slow_grid} (5个值)")
print(f"  STC卖出: {stc_sell_grid} (3个值)")
print(f"  止损: {stop_pct_grid} (4个值)")
print(f"  其他参数: 固定 (MACD 12/26/9, RSI 50, Ratio 1.15)")

# 总组合数：5×5×3×4 = 300个
param_combinations = list(itertools.product(
    ema_fast_grid, ema_slow_grid, stc_sell_grid, stop_pct_grid
))

# 过滤无效组合
valid_combinations = [p for p in param_combinations if p[0] < p[1]]
print(f"\n有效组合数: {len(valid_combinations)}")

# ============ 回测函数 ============
def run_backtest(df, params):
    ema_fast, ema_slow, stc_sell, stop_pct = params

    data = df.copy()
    data['ema_fast'] = data['close'].ewm(span=ema_fast, adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=ema_slow, adjust=False).mean()

    exp1 = data['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = data['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    data['macd_dif'] = exp1 - exp2
    data['macd_dea'] = data['macd_dif'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    data['ratio'] = data.apply(lambda x: x['macd_dif'] / x['macd_dea']
                               if x['macd_dea'] != 0 else 0, axis=1)

    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    stc_macd = data['close'].ewm(span=STC_FAST, adjust=False).mean() - \
               data['close'].ewm(span=STC_SLOW, adjust=False).mean()
    stoch_period = STC_LENGTH
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

    balance = 100000
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    closes = data['close'].values
    opens = data['open'].values
    lows = data['low'].values
    ema_fast_arr = data['ema_fast'].values
    ema_slow_arr = data['ema_slow'].values
    dif = data['macd_dif'].values
    ratio = data['ratio'].values
    rsi = data['rsi'].values
    stc = data['stc'].values

    for i in range(70, len(data)-1):
        price = closes[i]
        low_price = lows[i]
        next_open = opens[i+1]

        ema_fast_prev = ema_fast_arr[i-1]
        ema_slow_prev = ema_slow_arr[i-1]
        prev_dif = dif[i-1]
        prev_ratio = ratio[i-1]
        prev_stc = stc[i-1]

        if position > 0:
            stop_price = entry_price * (1 - stop_pct)
            if low_price <= stop_price:
                exit_p = stop_price
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0
            elif (prev_stc > stc_sell) and (stc[i] < prev_stc):
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0
            elif ema_fast_arr[i] < ema_slow_arr[i]:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0

        if position == 0:
            trend_up = (ema_fast_arr[i] > ema_slow_arr[i])
            ratio_safe = (0 < ratio[i] < 1.15)
            ratio_shrinking = (ratio[i] < prev_ratio)
            turning_up = (dif[i] > prev_dif)
            is_strong = (rsi[i] > 50)

            sniper_entry = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
            ema_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast_arr[i] > ema_slow_arr[i])
            chase_entry = ema_cross and is_strong

            if sniper_entry or chase_entry:
                risk_amt = balance * 0.02
                stop_dist = price * stop_pct
                if stop_dist > 0:
                    qty = int(risk_amt / (stop_dist * 5))
                    qty = max(1, min(qty, int(balance * 2.0 / (next_open * 5))))
                    position = qty
                    entry_price = next_open

        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * 5
        equity.append(val)

    if not trades or len(trades) < 5:
        return None

    ret = (equity[-1] - 100000) / 100000 * 100
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100

    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    max_dd = dd.min()

    return {
        'return_pct': ret,
        'max_dd': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'params': params
    }

# ============ 运行网格搜索 ============
print("\n" + "="*80)
print("开始简化网格搜索...")
print("="*80)

import time
start_time = time.time()

results = []
for i, params in enumerate(valid_combinations):
    result = run_backtest(train_df, params)
    if result:
        results.append(result)
        print(f"[{i+1:3d}/{len(valid_combinations)}] EMA({params[0]:2d},{params[1]:2d}) "
              f"STC{params[2]} Stop{params[3]*100:.1f}% -> {result['return_pct']:+.2f}% "
              f"胜率{result['win_rate']:.1f}% 交易{result['total_trades']:3d}")

elapsed = time.time() - start_time

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('return_pct', ascending=False)

print(f"\n网格搜索完成! 总耗时: {elapsed:.1f}秒")
print(f"有效结果: {len(results_df)}个")

# ============ 样本外测试Top 10 ============
print("\n" + "="*80)
print("Top 10参数样本外验证".center(80))
print("="*80)

out_of_sample_results = []
for idx, row in results_df.head(10).iterrows():
    params = row['params']
    test_result = run_backtest(test_df, params)

    if test_result:
        test_result['train_return'] = row['return_pct']
        test_result['train_win_rate'] = row['win_rate']
        test_result['train_dd'] = row['max_dd']
        test_result['params'] = params
        out_of_sample_results.append(test_result)

        print(f"\n排名 {results_df.index.get_loc(idx)+1}:")
        print(f"  参数: EMA({params[0]},{params[1]}) STC{params[2]} Stop{params[3]*100:.1f}%")
        print(f"  训练: {row['return_pct']:+.2f}% | 测试: {test_result['return_pct']:+.2f}% | "
              f"差异: {test_result['return_pct']-row['return_pct']:+.2f}%")

# ============ 与通用参数对比 ============
print("\n" + "="*80)
print("与通用参数对比".center(80))
print("="*80)

generic_params = (5, 15, 90, 0.02)
generic_train = run_backtest(train_df, generic_params)
generic_test = run_backtest(test_df, generic_params)

print(f"\n{'参数版本':<15} | {'训练收益':<12} | {'测试收益':<12} | {'测试胜率':<10}")
print("-"*80)
print(f"{'通用参数':<15} | {generic_train['return_pct']:>+10.2f}% | {generic_test['return_pct']:>+10.2f}% | "
      f"{generic_test['win_rate']:>9.1f}%")

if out_of_sample_results:
    best_grid = max(out_of_sample_results, key=lambda x: x['return_pct'])
    print(f"{'网格最优':<15} | {best_grid['train_return']:>+10.2f}% | {best_grid['return_pct']:>+10.2f}% | "
          f"{best_grid['win_rate']:>9.1f}%")

    improvement = best_grid['return_pct'] - generic_test['return_pct']
    print("-"*80)
    if improvement > 0:
        print(f"\n[SUCCESS] 网格优化优于通用参数 {improvement:.2f}%")
        print(f"          推荐使用网格优化参数")
    elif improvement > -10:
        print(f"\n[OK] 网格优化与通用参数接近 ({improvement:+.2f}%)")
        print(f"       两者都可以，建议使用通用参数（更稳健）")
    else:
        print(f"\n[INFO] 通用参数优于网格优化 {abs(improvement):.2f}%")
        print(f"       网格优化未能提升表现")

print("\n" + "="*80)
