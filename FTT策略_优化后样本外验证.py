"""
FTT策略 - 优化后参数样本外验证
=============================
对比：
1. 原始参数（固定值）
2. 优化后参数（贝叶斯200次迭代）
"""

import pandas as pd
import numpy as np
import pickle

# ============ 加载数据 ============
print("正在加载铜数据...")
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df_full = DATA_POOL['cu'].copy()
df_full.sort_index(inplace=True)

train_df = df_full.loc[:'2023-12-31'].copy()
test_df = df_full.loc['2024-01-01':].copy()

print(f"训练集: {len(train_df)}根K线")
print(f"测试集: {len(test_df)}根K线")

# ============ 参数定义 ============
# 原始参数
ORIGINAL_PARAMS = {
    'ema_fast': 5,
    'ema_slow': 15,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_sig': 9,
    'rsi_period': 14,
    'rsi_filter': 50,
    'ratio_trigger': 1.15,
    'stc_len': 10,
    'stc_fast': 23,
    'stc_slow': 50,
    'stc_sell': 90,
    'stop_pct': 0.02
}

# 优化后参数
OPTIMIZED_PARAMS = {
    'ema_fast': 13,
    'ema_slow': 17,
    'macd_fast': 8,
    'macd_slow': 20,
    'macd_sig': 10,
    'rsi_period': 14,
    'rsi_filter': 51,
    'ratio_trigger': 1.05,
    'stc_len': 5,
    'stc_fast': 15,
    'stc_slow': 64,
    'stc_sell': 87,
    'stop_pct': 0.03
}

# ============ 计算指标 ============
def calculate_indicators(df, p):
    data = df.copy()

    data['ema_fast'] = data['close'].ewm(span=int(p['ema_fast']), adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=int(p['ema_slow']), adjust=False).mean()

    exp1 = data['close'].ewm(span=int(p['macd_fast']), adjust=False).mean()
    exp2 = data['close'].ewm(span=int(p['macd_slow']), adjust=False).mean()
    data['macd_dif'] = exp1 - exp2
    data['macd_dea'] = data['macd_dif'].ewm(span=int(p['macd_sig']), adjust=False).mean()
    data['ratio'] = data.apply(lambda x: x['macd_dif'] / x['macd_dea']
                               if x['macd_dea'] != 0 else 0, axis=1)

    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/int(p['rsi_period']), adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/int(p['rsi_period']), adjust=False).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    stc_macd = data['close'].ewm(span=int(p['stc_fast']), adjust=False).mean() - \
               data['close'].ewm(span=int(p['stc_slow']), adjust=False).mean()
    stoch_period = int(p['stc_len'])
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

# ============ 回测函数 ============
def run_backtest(df, params):
    data = calculate_indicators(df, params)

    balance = 100000
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    for i in range(70, len(data)-1):
        date = data.index[i]
        price = data['close'].iloc[i]
        low_price = data['low'].iloc[i]
        next_open = data['open'].iloc[i+1]

        ema_fast = data['ema_fast'].iloc[i]
        ema_slow = data['ema_slow'].iloc[i]
        ema_fast_prev = data['ema_fast'].iloc[i-1]
        ema_slow_prev = data['ema_slow'].iloc[i-1]

        dif = data['macd_dif'].iloc[i]
        prev_dif = data['macd_dif'].iloc[i-1]

        ratio = data['ratio'].iloc[i]
        prev_ratio = data['ratio'].iloc[i-1]

        rsi = data['rsi'].iloc[i]

        stc = data['stc'].iloc[i]
        prev_stc = data['stc'].iloc[i-1]

        stop_pct = params['stop_pct']

        # 平仓
        if position > 0:
            stop_price = entry_price * (1 - stop_pct)

            if low_price <= stop_price:
                exit_p = stop_price
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0

            elif (prev_stc > params['stc_sell']) and (stc < prev_stc):
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0

            elif ema_fast < ema_slow:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0

        # 开仓
        if position == 0:
            trend_up = (ema_fast > ema_slow)

            ratio_safe = (0 < ratio < params['ratio_trigger'])
            ratio_shrinking = (ratio < prev_ratio)
            turning_up = (dif > prev_dif)
            is_strong = (rsi > params['rsi_filter'])

            sniper_entry = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong

            ema_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast > ema_slow)
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
            val += (price - entry_price) * position * 5
        equity.append(val)

    if not trades:
        return {'return_pct': 0, 'max_dd': 0, 'total_trades': 0,
                'win_rate': 0, 'profit_factor': 0}

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

# ============ 主程序 ============
print("\n" + "="*80)
print("FTT策略：原始参数 vs 优化参数 对比".center(80))
print("="*80)

# 原始参数测试
print("\n[测试1] 原始参数")
print("-"*80)
orig_train = run_backtest(train_df, ORIGINAL_PARAMS)
orig_test = run_backtest(test_df, ORIGINAL_PARAMS)

print(f"训练集: {orig_train['return_pct']:+.2f}%  胜率:{orig_train['win_rate']:.1f}%  "
      f"回撤:{orig_train['max_dd']:.1f}%  交易:{orig_train['total_trades']}")
print(f"测试集: {orig_test['return_pct']:+.2f}%  胜率:{orig_test['win_rate']:.1f}%  "
      f"回撤:{orig_test['max_dd']:.1f}%  交易:{orig_test['total_trades']}")

# 优化参数测试
print("\n[测试2] 优化参数")
print("-"*80)
opt_train = run_backtest(train_df, OPTIMIZED_PARAMS)
opt_test = run_backtest(test_df, OPTIMIZED_PARAMS)

print(f"训练集: {opt_train['return_pct']:+.2f}%  胜率:{opt_train['win_rate']:.1f}%  "
      f"回撤:{opt_train['max_dd']:.1f}%  交易:{opt_train['total_trades']}")
print(f"测试集: {opt_test['return_pct']:+.2f}%  胜率:{opt_test['win_rate']:.1f}%  "
      f"回撤:{opt_test['max_dd']:.1f}%  交易:{opt_test['total_trades']}")

# 对比表格
print("\n" + "="*80)
print("样本外测试集 (2024-2026) 对比".center(80))
print("="*80)
print(f"\n{'指标':<15} | {'原始参数':<20} | {'优化参数':<20} | {'改进':<15}")
print("-"*80)

ret_imp = orig_test['return_pct']
ret_imp2 = opt_test['return_pct']
imp = ret_imp2 - ret_imp
status = "[OK]" if imp > 0 else "[BAD]"
print(f"{'收益率':<15} | {ret_imp:>18.2f}% | {ret_imp2:>18.2f}% | {imp:>+13.2f}% {status}")

wr_imp = orig_test['win_rate']
wr_imp2 = opt_test['win_rate']
imp_wr = wr_imp2 - wr_imp
status_wr = "[OK]" if imp_wr > 0 else "[BAD]"
print(f"{'胜率':<15} | {wr_imp:>18.1f}% | {wr_imp2:>18.1f}% | {imp_wr:>+13.1f}% {status_wr}")

dd_orig = orig_test['max_dd']
dd_opt = opt_test['max_dd']
imp_dd = dd_opt - dd_orig
status_dd = "[OK]" if imp_dd > 0 else "[BAD]"
print(f"{'最大回撤':<15} | {dd_orig:>18.2f}% | {dd_opt:>18.2f}% | {imp_dd:>+13.2f}% {status_dd}")

pf_orig = orig_test['profit_factor']
pf_opt = opt_test['profit_factor']
imp_pf = pf_opt - pf_orig
print(f"{'盈亏比':<15} | {pf_orig:>18.2f} | {pf_opt:>18.2f} | {imp_pf:>+13.2f}")

trades_orig = orig_test['total_trades']
trades_opt = opt_test['total_trades']
print(f"{'交易次数':<15} | {trades_orig:>18} | {trades_opt:>18} | {trades_opt-trades_orig:>+13d}")

print("-"*80)

# 最终结论
print("\n" + "="*80)
if opt_test['return_pct'] > orig_test['return_pct']:
    print("[SUCCESS] 优化参数在样本外表现更优！")
    print("           贝叶斯优化找到了更好的参数组合。")
elif opt_test['return_pct'] >= orig_test['return_pct'] * 0.8:
    print("[OK] 优化参数表现接近，可接受。")
    print("     两套参数各有优势，可根据风险偏好选择。")
else:
    print("[FAIL] 优化参数在样本外表现更差。")
    print("        说明优化存在过拟合，建议使用原始参数。")
print("="*80)
