"""
锡(Sn) - FTT策略贝叶斯深度优化（200次迭代）
=========================================
目标：找到适合锡特性的最优参数

锡的特点：
- 价格高（46000+美元/吨）
- 波动大
- 趋势性强
- 可能需要与铜不同的参数
"""

import pandas as pd
import numpy as np
import pickle
import time
from skopt import gp_minimize
from skopt.space import Integer, Real
import warnings

warnings.filterwarnings('ignore')

# ============ 加载锡数据 ============
print("正在加载锡数据...")
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df_full = DATA_POOL['sn'].copy()
df_full.sort_index(inplace=True)

# 划分训练集和测试集
train_df = df_full.loc[:'2023-12-31'].copy()
test_df = df_full.loc['2024-01-01':].copy()

print(f"训练集: {len(train_df)}根K线 (2016-2023)")
print(f"测试集: {len(test_df)}根K线 (2024-2026)")

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

        # 平仓
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

        # 开仓
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
    if len(trades) < 15:
        score += 3000
    if max_dd < -35:
        score += 1000

    print(f"Ret:{ret:+.1f}% WR:{win_rate:.1f}% Trades:{len(trades)} DD:{max_dd:.1f}%")

    return -score

# ============ 参数空间 ============
# 针对锡的特点调整搜索范围
space = [
    Integer(3, 20, name='ema_fast'),       # 锡可能需要更快的EMA
    Integer(10, 40, name='ema_slow'),
    Integer(8, 20, name='macd_fast'),
    Integer(20, 40, name='macd_slow'),
    Integer(5, 15, name='macd_sig'),
    Integer(10, 20, name='rsi_period'),
    Integer(40, 60, name='rsi_filter'),    # 锡可能需要不同的RSI阈值
    Real(1.0, 1.5, name='ratio_trigger'),
    Integer(5, 15, name='stc_len'),
    Integer(15, 35, name='stc_fast'),
    Integer(40, 70, name='stc_slow'),
    Integer(80, 95, name='stc_sell'),
    Real(0.015, 0.03, name='stop_pct')
]

# ============ 主程序 ============
def main():
    print("\n" + "="*80)
    print("锡(Sn) - FTT策略贝叶斯优化（200次迭代）".center(80))
    print("="*80)

    print("\n参数空间（针对锡优化）:")
    print("  EMA:      (3-20, 10-40)")
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

    print("\n[OK] 锡的最优参数:")
    print("-"*80)
    print(f"  EMA:       ({int(res.x[0])}, {int(res.x[1])})")
    print(f"  MACD:      ({int(res.x[2])}, {int(res.x[3])}, {int(res.x[4])})")
    print(f"  RSI:       周期{int(res.x[5])} 阈值{int(res.x[6])}")
    print(f"  Ratio:     阈值{res.x[7]:.2f}")
    print(f"  STC:       周期{int(res.x[8])} ({int(res.x[9])}, {int(res.x[10])}) 卖出{int(res.x[11])}")
    print(f"  止损:      {res.x[12]*100:.2f}%")

    # 与铜的原始参数对比
    print("\n与铜原始参数对比:")
    print("-"*80)
    print(f"{'参数':<15} | {'铜原始':<20} | {'锡优化':<20} | {'差异'}")
    print("-"*80)
    print(f"{'EMA快线':<15} | {5:<20} | {int(res.x[0]):<20} | {int(res.x[0])-5:+d}")
    print(f"{'EMA慢线':<15} | {15:<20} | {int(res.x[1]):<20} | {int(res.x[1])-15:+d}")
    print(f"{'MACD快线':<15} | {12:<20} | {int(res.x[2]):<20} | {int(res.x[2])-12:+d}")
    print(f"{'MACD慢线':<15} | {26:<20} | {int(res.x[3]):<20} | {int(res.x[3])-26:+d}")
    print(f"{'RSI阈值':<15} | {50:<20} | {int(res.x[6]):<20} | {int(res.x[6])-50:+d}")
    print(f"{'STC卖出':<15} | {90:<20} | {int(res.x[11]):<20} | {int(res.x[11])-90:+d}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
