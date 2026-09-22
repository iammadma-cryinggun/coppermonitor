"""
锡(Sn) - 网格参数优化
===================
通过网格搜索全面测试参数组合
找到top参数组合并做样本外验证
"""

import pandas as pd
import numpy as np
import pickle
import itertools
from datetime import datetime

print("="*80)
print("锡(Sn) - 网格参数优化".center(80))
print("="*80)

# 加载数据
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df_full = DATA_POOL['sn'].copy()
df_full.sort_index(inplace=True)

train_df = df_full.loc[:'2023-12-31'].copy()
test_df = df_full.loc['2024-01-01':].copy()

print(f"\n数据分割:")
print(f"  训练集: {len(train_df)}根K线 (2016-2023)")
print(f"  测试集: {len(test_df)}根K线 (2024-2026)")

# ============ 定义参数网格 ============
# 基于贝叶斯优化的结果和铜的参数，设置合理的网格

ema_fast_grid = [5, 8, 12, 15, 18]
ema_slow_grid = [15, 20, 25, 30, 35, 40]
macd_fast_grid = [8, 12, 15]
macd_slow_grid = [20, 26, 30, 35, 40]
macd_sig_grid = [7, 9, 12]
rsi_filter_grid = [45, 50, 55]
ratio_trigger_grid = [1.10, 1.15, 1.20, 1.30]
stc_sell_grid = [85, 90, 95]
stop_pct_grid = [0.015, 0.02, 0.025, 0.03]

# 固定参数（不优化）
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50
RSI_PERIOD = 14

print(f"\n参数网格:")
print(f"  EMA快线: {ema_fast_grid}")
print(f"  EMA慢线: {ema_slow_grid}")
print(f"  MACD: ({macd_fast_grid}, {macd_slow_grid}, {macd_sig_grid})")
print(f"  RSI阈值: {rsi_filter_grid}")
print(f"  Ratio触发: {ratio_trigger_grid}")
print(f"  STC卖出: {stc_sell_grid}")
print(f"  止损: {stop_pct_grid}")

# 计算总组合数
total_combinations = 1
for grid in [ema_fast_grid, ema_slow_grid, macd_fast_grid, macd_slow_grid,
             macd_sig_grid, rsi_filter_grid, ratio_trigger_grid, stc_sell_grid, stop_pct_grid]:
    total_combinations *= len(grid)

print(f"\n总组合数: {total_combinations:,}")

# ============ 生成参数组合 ============
print(f"\n生成参数组合...")
param_combinations = list(itertools.product(
    ema_fast_grid, ema_slow_grid, macd_fast_grid, macd_slow_grid, macd_sig_grid,
    rsi_filter_grid, ratio_trigger_grid, stc_sell_grid, stop_pct_grid
))

# 过滤掉无效组合（EMA快线必须 < EMA慢线）
valid_combinations = []
for combo in param_combinations:
    if combo[0] < combo[1] and combo[2] < combo[3]:  # EMA和MACD约束
        valid_combinations.append(combo)

print(f"有效组合数: {len(valid_combinations):,}")

# ============ 回测函数 ============
def run_backtest(df, params):
    ema_fast, ema_slow, macd_fast, macd_slow, macd_sig, rsi_filter, \
    ratio_trigger, stc_sell, stop_pct = params

    data = df.copy()

    # 计算指标
    data['ema_fast'] = data['close'].ewm(span=ema_fast, adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=ema_slow, adjust=False).mean()

    exp1 = data['close'].ewm(span=macd_fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=macd_slow, adjust=False).mean()
    data['macd_dif'] = exp1 - exp2
    data['macd_dea'] = data['macd_dif'].ewm(span=macd_sig, adjust=False).mean()
    data['ratio'] = data.apply(lambda x: x['macd_dif'] / x['macd_dea']
                               if x['macd_dea'] != 0 else 0, axis=1)

    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # STC
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

    # 回测
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
            ratio_safe = (0 < ratio[i] < ratio_trigger)
            ratio_shrinking = (ratio[i] < prev_ratio)
            turning_up = (dif[i] > prev_dif)
            is_strong = (rsi[i] > rsi_filter)

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

    if not trades or len(trades) < 10:
        return None

    ret = (equity[-1] - 100000) / 100000 * 100
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100

    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    max_dd = dd.min()

    winning = [t for t in trades if t > 0]
    losing = [t for t in trades if t <= 0]
    profit_factor = abs(sum(winning) / sum(losing)) if losing else 0

    return {
        'return_pct': ret,
        'max_dd': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'params': params
    }

# ============ 运行网格搜索 ============
print("\n" + "="*80, flush=True)
print("开始网格搜索...", flush=True)
print("="*80, flush=True)

start_time = datetime.now()

results = []
for i, params in enumerate(valid_combinations):
    result = run_backtest(train_df, params)
    if result:
        results.append(result)

    if (i + 1) % 100 == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"已测试: {i+1}/{len(valid_combinations)} ({(i+1)/len(valid_combinations)*100:.1f}%) | "
              f"耗时: {elapsed:.0f}秒 | 有效结果: {len(results)}", flush=True)

elapsed_total = (datetime.now() - start_time).total_seconds()
print(f"\n网格搜索完成! 总耗时: {elapsed_total:.1f}秒")
print(f"有效参数组合: {len(results)}")

# ============ 排序并分析 ============
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('return_pct', ascending=False)

print("\n" + "="*80)
print("训练集Top 20参数组合".center(80))
print("="*80)

print(f"\n{'排名':<6} {'收益率':<10} {'胜率':<8} {'回撤':<10} {'交易':<8} {'参数':<60}")
print("-"*80)

for idx, row in results_df.head(20).iterrows():
    rank = results_df.index.get_loc(idx) + 1
    p = row['params']
    param_str = f"EMA({p[0]},{p[1]}) MACD({p[2]},{p[3]},{p[4]}) RSI{p[5]} Ratio{p[6]:.2f} STC{p[7]} Stop{p[8]*100:.1f}%"
    print(f"{rank:<6} {row['return_pct']:>+8.2f}% {row['win_rate']:>7.1f}% {row['max_dd']:>8.2f}% "
          f"{row['total_trades']:>6} {param_str}")

# ============ 测试Top参数的样本外表现 ============
print("\n" + "="*80)
print("Top 10参数样本外验证".center(80))
print("="*80)

top_params = results_df.head(10)

out_of_sample_results = []
for idx, row in top_params.iterrows():
    params = row['params']
    test_result = run_backtest(test_df, params)

    if test_result:
        test_result['train_return'] = row['return_pct']
        test_result['train_win_rate'] = row['win_rate']
        test_result['params'] = params
        out_of_sample_results.append(test_result)

print(f"\n{'排名':<6} {'训练收益':<12} {'测试收益':<12} {'训练胜率':<10} {'测试胜率':<10} {'差异':<10}")
print("-"*80)

for i, res in enumerate(out_of_sample_results):
    diff = res['return_pct'] - res['train_return']
    status = "[OK]" if res['return_pct'] > 30 else "[WARN]" if res['return_pct'] > 0 else "[FAIL]"
    print(f"{i+1:<6} {res['train_return']:>+10.2f}% {res['return_pct']:>+10.2f}% "
          f"{res['train_win_rate']:>9.1f}% {res['win_rate']:>9.1f}% {diff:>+8.2f}% {status}")

# ============ 与通用参数对比 ============
print("\n" + "="*80)
print("与通用参数对比".center(80))
print("="*80)

generic_params = (5, 15, 12, 26, 9, 50, 1.15, 90, 0.02)
generic_train = run_backtest(train_df, generic_params)
generic_test = run_backtest(test_df, generic_params)

print(f"\n{'参数版本':<15} | {'训练收益':<12} | {'测试收益':<12} | {'测试胜率':<10} | {'测试回撤':<10}")
print("-"*80)

print(f"{'通用参数':<15} | {generic_train['return_pct']:>+10.2f}% | {generic_test['return_pct']:>+10.2f}% | "
      f"{generic_test['win_rate']:>9.1f}% | {generic_test['max_dd']:>9.2f}%")

if out_of_sample_results:
    best_grid = out_of_sample_results[0]
    best_params_str = f"EMA({best_grid['params'][0]},{best_grid['params'][1]})..."
    print(f"{'网格最优':<15} | {best_grid['train_return']:>+10.2f}% | {best_grid['return_pct']:>+10.2f}% | "
          f"{best_grid['win_rate']:>9.1f}% | {best_grid['max_dd']:>9.2f}%")

    improvement = best_grid['return_pct'] - generic_test['return_pct']
    if improvement > 0:
        print(f"\n[SUCCESS] 网格优化优于通用参数 {improvement:.2f}%")
    else:
        print(f"\n[INFO] 通用参数优于网格优化 {abs(improvement):.2f}%")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
