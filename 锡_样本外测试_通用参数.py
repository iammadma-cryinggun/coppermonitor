"""
锡(Sn) - 样本外测试（使用铜的通用参数）
=====================================
先用通用参数测试，等优化完成后再对比
"""

import pandas as pd
import numpy as np
import pickle

print("="*80)
print("锡(Sn) - 样本外测试（通用参数）".center(80))
print("="*80)

# 加载数据
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df_full = DATA_POOL['sn'].copy()
df_full.sort_index(inplace=True)

train_df = df_full.loc[:'2023-12-31'].copy()
test_df = df_full.loc['2024-01-01':].copy()

print(f"\n数据分割:")
print(f"  训练集(样本内): 2016-2023 ({len(train_df)}根K线)")
print(f"  测试集(样本外): 2024-2026 ({len(test_df)}根K线)")

# 铜的原始参数
params = [5, 15, 12, 26, 9, 14, 50, 1.15, 10, 23, 50, 90, 0.02]

# 计算指标
def calculate_indicators(df, p):
    data = df.copy()
    data['ema_fast'] = data['close'].ewm(span=int(p[0]), adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=int(p[1]), adjust=False).mean()

    exp1 = data['close'].ewm(span=int(p[2]), adjust=False).mean()
    exp2 = data['close'].ewm(span=int(p[3]), adjust=False).mean()
    data['macd_dif'] = exp1 - exp2
    data['macd_dea'] = data['macd_dif'].ewm(span=int(p[4]), adjust=False).mean()
    data['ratio'] = data.apply(lambda x: x['macd_dif'] / x['macd_dea']
                               if x['macd_dea'] != 0 else 0, axis=1)

    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/int(p[5]), adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/int(p[5]), adjust=False).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    stc_macd = data['close'].ewm(span=int(p[9]), adjust=False).mean() - \
               data['close'].ewm(span=int(p[10]), adjust=False).mean()
    stoch_period = int(p[8])
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

    return data

def run_backtest(df, params):
    data = calculate_indicators(df, params)
    balance = 100000
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    closes = data['close'].values
    opens = data['open'].values
    lows = data['low'].values
    ema_fast = data['ema_fast'].values
    ema_slow = data['ema_slow'].values
    dif = data['macd_dif'].values
    ratio = data['ratio'].values
    rsi = data['rsi'].values
    stc = data['stc'].values

    stop_pct = params[11]

    for i in range(70, len(data)-1):
        price = closes[i]
        low_price = lows[i]
        next_open = opens[i+1]

        ema_fast_prev = ema_fast[i-1]
        ema_slow_prev = ema_slow[i-1]
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
            elif (prev_stc > params[10]) and (stc[i] < prev_stc):
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0
            elif ema_fast[i] < ema_slow[i]:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0

        if position == 0:
            trend_up = (ema_fast[i] > ema_slow[i])
            ratio_safe = (0 < ratio[i] < params[6])
            ratio_shrinking = (ratio[i] < prev_ratio)
            turning_up = (dif[i] > prev_dif)
            is_strong = (rsi[i] > params[7])

            sniper_entry = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
            ema_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast[i] > ema_slow[i])
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

    if not trades:
        return {'return_pct': 0, 'max_dd': 0, 'total_trades': 0, 'win_rate': 0, 'profit_factor': 0}

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
        'profit_factor': profit_factor
    }

# 运行测试
print("\n" + "="*80)
print("开始测试...")
print("="*80)

result_train = run_backtest(train_df, params)
result_test = run_backtest(test_df, params)

print(f"\n{'指标':<12} | {'训练集 (2016-2023)':<20} | {'测试集 (2024-2026)':<20} | {'状态':<12}")
print("-"*80)

ret_train = result_train['return_pct']
ret_test = result_test['return_pct']
status_ret = "[OK]" if ret_test > 0 else "[FAIL]"
if ret_test > ret_train * 0.7:
    status_ret = "[GOOD]"
print(f"{'总收益率':<12} | {ret_train:>18.2f}% | {ret_test:>18.2f}% | {status_ret}")

win_train = result_train['win_rate']
win_test = result_test['win_rate']
status_win = "[OK]" if win_test >= 35 else "[WARN]"
print(f"{'胜率':<12} | {win_train:>18.1f}% | {win_test:>18.1f}% | {status_win}")

dd_train = result_train['max_dd']
dd_test = result_test['max_dd']
status_dd = "[OK]" if abs(dd_test) < 25 else "[WARN]"
print(f"{'最大回撤':<12} | {dd_train:>18.2f}% | {dd_test:>18.2f}% | {status_dd}")

cnt_train = result_train['total_trades']
cnt_test = result_test['total_trades']
print(f"{'交易次数':<12} | {cnt_train:>18} | {cnt_test:>18} | {'-'}")

pf_train = result_train['profit_factor']
pf_test = result_test['profit_factor']
print(f"{'盈亏比':<12} | {pf_train:>18.2f} | {pf_test:>18.2f} | {'-'}")

print("-"*80)

print("\n" + "="*80)
if ret_test > 20 and win_test >= 35:
    print("[SUCCESS] 锡通过了样本外测试！")
    print("           即使使用铜的参数，锡在样本外依然表现优异。")
elif ret_test > 10 and win_test >= 30:
    print("[OK] 锡样本外表现尚可")
    print("    等待优化完成后，应该会有更好表现。")
else:
    print("[WARN] 锡样本外表现一般")
    print("       可能需要专门优化的参数。")
print("="*80)
