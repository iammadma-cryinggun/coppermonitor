"""
FTT策略 - 全品种扫描测试
========================
在data_pool中的所有品种上测试FTT策略
找出最适合FTT策略的标的
"""

import pandas as pd
import numpy as np
import pickle

print("="*80)
print("FTT策略 - 全品种扫描测试".center(80))
print("="*80)

# 加载数据
print("\n正在加载数据...")
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

symbols = list(DATA_POOL.keys())
print(f"品种数量: {len(symbols)}")
print(f"测试品种: {symbols}")

# ============ FTT策略回测函数 ============
def run_ftt_backtest(df, symbol="品种"):
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
        return {
            'symbol': symbol,
            'return_pct': 0,
            'max_dd': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'annual_return': 0,
            'score': 0
        }

    final_ret = (equity[-1] - 100000) / 100000 * 100
    win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100

    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    max_dd = dd.min()

    winning = [t['pnl'] for t in trades if t['pnl'] > 0]
    losing = [t['pnl'] for t in trades if t['pnl'] <= 0]
    profit_factor = abs(sum(winning) / sum(losing)) if losing else 0

    # 年化收益（假设数据跨度10年）
    years = (data.index[-1] - data.index[0]).days / 365.25
    annual_return = final_ret / years if years > 0 else 0

    # 综合评分（收益*0.4 + 胜率*0.3 + 盈亏比*0.3 - 回撤惩罚）
    score = final_ret * 0.4 + win_rate * 0.5 + profit_factor * 10 - abs(max_dd) * 0.5

    return {
        'symbol': symbol,
        'return_pct': final_ret,
        'max_dd': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'annual_return': annual_return,
        'score': score
    }

# ============ 运行扫描 ============
print("\n" + "="*80)
print("开始扫描所有品种...")
print("="*80)

results = []
for symbol in symbols:
    print(f"\n测试品种: {symbol}...")
    try:
        result = run_ftt_backtest(DATA_POOL[symbol], symbol)
        results.append(result)
        print(f"  收益率: {result['return_pct']:+.2f}%")
        print(f"  胜率: {result['win_rate']:.1f}%")
        print(f"  年化: {result['annual_return']:.1f}%")
        print(f"  回撤: {result['max_dd']:.2f}%")
        print(f"  交易: {result['total_trades']}笔")
        print(f"  评分: {result['score']:.1f}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append({
            'symbol': symbol,
            'return_pct': 0,
            'max_dd': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'annual_return': 0,
            'score': -999
        })

# ============ 排序并展示结果 ============
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('score', ascending=False)

print("\n" + "="*80)
print("扫描结果排名（按综合评分）".center(80))
print("="*80)

print(f"\n{'排名':<6} {'品种':<8} {'总收益率':<12} {'年化收益':<12} {'胜率':<10} {'盈亏比':<10} {'回撤':<10} {'交易':<8} {'评分':<10}")
print("-"*80)

for idx, row in results_df.iterrows():
    rank = results_df.index.get_loc(idx) + 1
    symbol = row['symbol']
    ret = row['return_pct']
    ann = row['annual_return']
    win = row['win_rate']
    pf = row['profit_factor']
    dd = row['max_dd']
    trades = row['total_trades']
    score = row['score']

    # 评级标记
    if rank == 1:
        symbol = f"[最佳] {symbol}"
    elif rank == 2:
        symbol = f"[银牌] {symbol}"
    elif rank == 3:
        symbol = f"[铜牌] {symbol}"

    print(f"{rank:<6} {symbol:<8} {ret:>+10.2f}% {ann:>+10.1f}% {win:>9.1f}% {pf:>9.2f} {dd:>9.2f}% {trades:>6} {score:>9.1f}")

print("-"*80)

# ============ 推荐分析 ============
print("\n" + "="*80)
print("推荐分析".center(80))
print("="*80)

best = results_df.iloc[0]
print(f"\n[冠军] {best['symbol']}")
print(f"  总收益率: {best['return_pct']:+.2f}%")
print(f"  年化收益: {best['annual_return']:.1f}%")
print(f"  胜率: {best['win_rate']:.1f}%")
print(f"  盈亏比: {best['profit_factor']:.2f}")
print(f"  最大回撤: {best['max_dd']:.2f}%")
print(f"  综合评分: {best['score']:.1f}")

# 高胜率品种
high_win_rate = results_df[results_df['win_rate'] >= 40].sort_values('win_rate', ascending=False)
if len(high_win_rate) > 0:
    print(f"\n[高胜率品种] (胜率>=40%)")
    for idx, row in high_win_rate.head(3).iterrows():
        print(f"  {row['symbol']}: {row['win_rate']:.1f}% 胜率, {row['return_pct']:+.2f}% 收益")

# 高收益品种
high_return = results_df[results_df['return_pct'] >= 50].sort_values('return_pct', ascending=False)
if len(high_return) > 0:
    print(f"\n[高收益品种] (总收益>=50%)")
    for idx, row in high_return.head(3).iterrows():
        print(f"  {row['symbol']}: {row['return_pct']:+.2f}% 收益, {row['win_rate']:.1f}% 胜率")

# 稳健品种（回撤小）
stable = results_df[(results_df['max_dd'] > -20) & (results_df['return_pct'] > 20)].sort_values('max_dd', ascending=False)
if len(stable) > 0:
    print(f"\n[稳健品种] (回撤<20% 且 收益>20%)")
    for idx, row in stable.head(3).iterrows():
        print(f"  {row['symbol']}: {row['max_dd']:.2f}% 回撤, {row['return_pct']:+.2f}% 收益")

print("\n" + "="*80)
print("扫描完成！")
print("="*80)
