"""
铜FTT策略 - 贝叶斯参数优化（200次迭代）
=======================================
目标：在训练集(2016-2023)上寻找最优参数

优化参数：
- EMA快慢线
- MACD参数
- RSI阈值
- Ratio阈值
- STC参数
- 止损幅度
"""

import pandas as pd
import numpy as np
import pickle
import time
from skopt import gp_minimize
from skopt.space import Integer, Real
import warnings

warnings.filterwarnings('ignore')

# ============ 加载数据 ============
print("正在加载铜数据...")
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df_full = DATA_POOL['cu'].copy()
df_full.sort_index(inplace=True)

# 划分训练集
train_df = df_full.loc[:'2023-12-31'].copy()
print(f"训练集: 2016-2023 ({len(train_df)}根K线)")

# ============ 计算指标函数 ============
def calculate_indicators(df, p):
    """
    p = [ema_fast, ema_slow, macd_fast, macd_slow, macd_sig,
         rsi_period, rsi_filter, ratio_trigger,
         stc_len, stc_fast, stc_slow, stc_sell, stop_pct]
    """
    data = df.copy()

    # 1. EMA
    data['ema_fast'] = data['close'].ewm(span=int(p[0]), adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=int(p[1]), adjust=False).mean()

    # 2. MACD & Ratio
    exp1 = data['close'].ewm(span=int(p[2]), adjust=False).mean()
    exp2 = data['close'].ewm(span=int(p[3]), adjust=False).mean()
    data['macd_dif'] = exp1 - exp2
    data['macd_dea'] = data['macd_dif'].ewm(span=int(p[4]), adjust=False).mean()
    data['ratio'] = data.apply(lambda x: x['macd_dif'] / x['macd_dea']
                               if x['macd_dea'] != 0 else 0, axis=1)

    # 3. RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/int(p[5]), adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/int(p[5]), adjust=False).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # 4. STC
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

# ============ 回测函数 ============
def run_backtest(df, params):
    # 约束检查：EMA快线必须小于慢线
    if params[0] >= params[1]:
        return 10000

    # 约束检查：MACD快线必须小于慢线
    if params[2] >= params[3]:
        return 10000

    data = calculate_indicators(df, params)

    balance = 100000
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    for i in range(70, len(data)-1):
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

        stop_pct = params[11]

        # 平仓
        if position > 0:
            stop_price = entry_price * (1 - stop_pct)

            if low_price <= stop_price:
                exit_p = stop_price
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0

            elif (prev_stc > params[10]) and (stc < prev_stc):
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

            ratio_safe = (0 < ratio < params[6])
            ratio_shrinking = (ratio < prev_ratio)
            turning_up = (dif > prev_dif)
            is_strong = (rsi > params[7])

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

    if not equity:
        return 10000

    ret = (equity[-1] - 100000) / 100000 * 100
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0

    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    max_dd = dd.min()

    # 评分：收益优先，兼顾胜率和回撤
    score = ret * 3
    score += (win_rate - 30) * 8
    score += max_dd

    if win_rate < 25:
        score += 2000
    if len(trades) < 20:
        score += 3000
    if max_dd < -30:
        score += 1000

    print(f"Ret:{ret:+.1f}% WR:{win_rate:.1f}% Trades:{len(trades)} DD:{max_dd:.1f}%")

    return -score

# ============ 参数空间 ============
space = [
    Integer(3, 15, name='ema_fast'),       # EMA快线
    Integer(10, 30, name='ema_slow'),      # EMA慢线
    Integer(8, 20, name='macd_fast'),      # MACD快线
    Integer(20, 40, name='macd_slow'),     # MACD慢线
    Integer(5, 15, name='macd_sig'),       # MACD信号线
    Integer(10, 20, name='rsi_period'),    # RSI周期
    Integer(40, 60, name='rsi_filter'),    # RSI阈值
    Real(1.0, 1.5, name='ratio_trigger'),  # Ratio阈值
    Integer(5, 15, name='stc_len'),        # STC周期
    Integer(15, 35, name='stc_fast'),      # STC快线
    Integer(40, 70, name='stc_slow'),      # STC慢线
    Integer(80, 95, name='stc_sell'),      # STC卖出阈值
    Real(0.015, 0.03, name='stop_pct')     # 止损百分比
]

# ============ 主程序 ============
def main():
    print("\n" + "="*80)
    print("铜FTT策略 - 贝叶斯参数优化（200次迭代）".center(80))
    print("="*80)

    print("\n参数空间:")
    print("  EMA:      (3-15, 10-30)")
    print("  MACD:     (8-20, 20-40, 5-15)")
    print("  RSI:      周期(10-20) 阈值(40-60)")
    print("  Ratio:    阈值(1.0-1.5)")
    print("  STC:      周期(5-15) 快线(15-35) 慢线(40-70) 卖出(80-95)")
    print("  止损:     1.5%-3.0%")

    print("\n开始优化（200次迭代，预计5-8分钟）...")
    print("="*80)

    start_time = time.time()

    res = gp_minimize(
        lambda p: run_backtest(train_df, p),
        space,
        n_calls=200,
        n_random_starts=30,
        random_state=42,
        verbose=False
    )

    elapsed = time.time() - start_time

    print(f"\n优化完成! 耗时: {elapsed:.1f}秒")
    print("="*80)

    print("\n最优参数:")
    print(f"  EMA:       ({int(res.x[0])}, {int(res.x[1])})")
    print(f"  MACD:      ({int(res.x[2])}, {int(res.x[3])}, {int(res.x[4])})")
    print(f"  RSI:       周期{int(res.x[5])} 阈值{int(res.x[6])}")
    print(f"  Ratio:     阈值{res.x[7]:.2f}")
    print(f"  STC:       周期{int(res.x[8])} ({int(res.x[9])}, {int(res.x[10])}) 卖出{int(res.x[11])}")
    print(f"  止损:      {res.x[12]*100:.2f}%")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
