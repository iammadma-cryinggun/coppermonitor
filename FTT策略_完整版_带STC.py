"""
铜FTT策略 - 完整版（带STC止盈）
===============================
完整策略包括：
1. EMA趋势系统
2. MACD Ratio动能系统
3. RSI强弱过滤
4. STC逃顶止盈
5. 硬止损 + 趋势结束离场
"""

import pandas as pd
import numpy as np
import pickle

# ==========================================
# 策略核心配置
# ==========================================
DATA_PATH = 'data_pool.pkl'

# 核心指标参数
EMA_FAST = 5
EMA_SLOW = 15
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RSI_FILTER = 50

# 进场参数
RATIO_TRIGGER = 1.15      # 狙击阈值

# STC 参数 (用于逃顶)
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50
STC_SELL_ZONE = 90        # STC 超买线

# 止损参数
STOP_LOSS_PCT = 0.02      # 2% 硬止损

def calculate_indicators(df):
    """计算技术指标（包含STC）"""
    data = df.copy()

    # 1. EMA 趋势系统
    data['ema_fast'] = data['close'].ewm(span=EMA_FAST, adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    # 2. MACD & Ratio 动能系统
    exp1 = data['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = data['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    data['macd_dif'] = exp1 - exp2
    data['macd_dea'] = data['macd_dif'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    data['ratio'] = data.apply(lambda x: x['macd_dif'] / x['macd_dea'] if x['macd_dea'] != 0 else 0, axis=1)

    # 3. RSI 强弱系统
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # 4. STC 震荡系统 (用于止盈)
    stc_macd = data['close'].ewm(span=STC_FAST, adjust=False).mean() - data['close'].ewm(span=STC_SLOW, adjust=False).mean()
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

    return data

def run_backtest(df, initial_capital=100000):
    """运行回测"""
    data = calculate_indicators(df.copy())

    balance = initial_capital
    position = 0
    entry_price = 0.0
    trades = []
    equity_curve = []

    for i in range(70, len(data)-1):
        date = data.index[i]
        price = data['close'].iloc[i]
        low_price = data['low'].iloc[i]
        next_open = data['open'].iloc[i+1]

        # 指标获取
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

        # ============ 平仓逻辑 ============
        if position > 0:
            stop_price = entry_price * (1 - STOP_LOSS_PCT)

            # 1. 硬止损 (保命底线)
            if low_price <= stop_price:
                exit_p = stop_price
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append({
                    'pnl': pnl,
                    'type': 'stop_loss'
                })
                position = 0

            # 2. STC 逃顶 (保住利润)
            elif (prev_stc > STC_SELL_ZONE) and (stc < prev_stc):
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append({
                    'pnl': pnl,
                    'type': 'stc_profit'
                })
                position = 0

            # 3. 趋势结束 (最后防线)
            elif ema_fast < ema_slow:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append({
                    'pnl': pnl,
                    'type': 'trend_end'
                })
                position = 0

        # ============ 开仓逻辑 ============
        if position == 0:
            trend_up = (ema_fast > ema_slow)

            # 1. 狙击条件 (回调精准买入)
            ratio_safe = (0 < ratio < RATIO_TRIGGER)
            ratio_shrinking = (ratio < prev_ratio)
            turning_up = (dif > prev_dif)
            is_strong = (rsi > RSI_FILTER)

            sniper_entry = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong

            # 2. 追涨条件 (趋势启动买入)
            ema_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast > ema_slow)
            chase_entry = ema_cross and is_strong

            if sniper_entry or chase_entry:
                # 仓位管理：风险2%
                risk_amt = balance * 0.02
                stop_dist = price * STOP_LOSS_PCT
                if stop_dist > 0:
                    qty = int(risk_amt / (stop_dist * 5))
                    qty = max(1, min(qty, int(balance * 2.0 / (next_open * 5))))
                    position = qty
                    entry_price = next_open

        # 记录净值
        val = balance
        if position > 0:
            val += (price - entry_price) * position * 5
        equity_curve.append(val)

    # 统计结果
    if not trades:
        return {
            'return_pct': 0,
            'max_dd': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'equity': []
        }

    final_ret = (equity_curve[-1] - initial_capital) / initial_capital * 100
    win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100

    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_dd = drawdown.min()

    winning_trades = [t['pnl'] for t in trades if t['pnl'] > 0]
    losing_trades = [t['pnl'] for t in trades if t['pnl'] <= 0]
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0

    # 统计离场方式
    stop_loss_count = len([t for t in trades if t['type'] == 'stop_loss'])
    stc_profit_count = len([t for t in trades if t['type'] == 'stc_profit'])
    trend_end_count = len([t for t in trades if t['type'] == 'trend_end'])

    return {
        'return_pct': final_ret,
        'max_dd': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'equity': equity_curve,
        'trades': trades,
        'exit_types': {
            'stop_loss': stop_loss_count,
            'stc_profit': stc_profit_count,
            'trend_end': trend_end_count
        }
    }

def main():
    print("\n" + "="*80)
    print("铜FTT策略 - 完整版（带STC止盈）".center(80))
    print("="*80)

    # 加载数据
    print("\n正在加载铜数据...")
    with open(DATA_PATH, 'rb') as f:
        DATA_POOL = pickle.load(f)

    df = DATA_POOL['cu'].copy()
    df.sort_index(inplace=True)

    print(f"数据范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"K线数量: {len(df)}")

    # 运行回测
    print("\n开始回测...")
    result = run_backtest(df)

    # 打印结果
    print("\n" + "="*80)
    print("回测结果（带STC止盈版本）")
    print("="*80)
    print(f"总收益率:     {result['return_pct']:+.2f}%")
    print(f"年化收益:     {result['return_pct']/10:.1f}%")
    print(f"最大回撤:     {result['max_dd']:.2f}%")
    print(f"交易次数:     {result['total_trades']}")
    print(f"胜率:         {result['win_rate']:.1f}%")
    print(f"平均盈利:     {result['avg_win']:,.0f}")
    print(f"平均亏损:     {result['avg_loss']:,.0f}")
    print(f"盈亏比:       {result['profit_factor']:.2f}")

    print(f"\n离场方式统计:")
    print(f"  硬止损:     {result['exit_types']['stop_loss']}笔")
    print(f"  STC止盈:    {result['exit_types']['stc_profit']}笔")
    print(f"  趋势结束:   {result['exit_types']['trend_end']}笔")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
