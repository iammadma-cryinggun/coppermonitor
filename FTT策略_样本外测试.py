"""
铜FTT策略 - 严格样本外测试
==========================
流程：
1. 将数据劈开：训练集(2016-2023) vs 测试集(2024-2026)
2. 在训练集上回测，看策略表现
3. 将策略应用到测试集，验证是否过拟合
"""

import pandas as pd
import numpy as np
import pickle

# ==========================================
# 策略核心配置（固定参数）
# ==========================================
EMA_FAST = 5
EMA_SLOW = 15
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RSI_FILTER = 50
RATIO_TRIGGER = 1.15

STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50
STC_SELL_ZONE = 90

STOP_LOSS_PCT = 0.02

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

    # 4. STC 震荡系统
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

            # 1. 硬止损
            if low_price <= stop_price:
                exit_p = stop_price
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append({'pnl': pnl, 'date': date, 'type': 'stop_loss'})
                position = 0

            # 2. STC 止盈
            elif (prev_stc > STC_SELL_ZONE) and (stc < prev_stc):
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append({'pnl': pnl, 'date': date, 'type': 'stc_profit'})
                position = 0

            # 3. 趋势结束
            elif ema_fast < ema_slow:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append({'pnl': pnl, 'date': date, 'type': 'trend_end'})
                position = 0

        # ============ 开仓逻辑 ============
        if position == 0:
            trend_up = (ema_fast > ema_slow)

            ratio_safe = (0 < ratio < RATIO_TRIGGER)
            ratio_shrinking = (ratio < prev_ratio)
            turning_up = (dif > prev_dif)
            is_strong = (rsi > RSI_FILTER)

            sniper_entry = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong

            ema_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast > ema_slow)
            chase_entry = ema_cross and is_strong

            if sniper_entry or chase_entry:
                risk_amt = balance * 0.02
                stop_dist = price * STOP_LOSS_PCT
                if stop_dist > 0:
                    qty = int(risk_amt / (stop_dist * 5))
                    qty = max(1, min(qty, int(balance * 2.0 / (next_open * 5))))
                    position = qty
                    entry_price = next_open

        val = balance
        if position > 0:
            val += (price - entry_price) * position * 5
        equity_curve.append(val)

    # 统计
    if not trades:
        return {
            'return_pct': 0,
            'max_dd': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'trades': []
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

    return {
        'return_pct': final_ret,
        'max_dd': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'trades': trades
    }

def main():
    print("\n" + "="*80)
    print("铜FTT策略 - 严格样本外测试".center(80))
    print("="*80)

    # 加载数据
    print("\n正在加载铜数据...")
    with open('data_pool.pkl', 'rb') as f:
        DATA_POOL = pickle.load(f)

    df_full = DATA_POOL['cu'].copy()
    df_full.sort_index(inplace=True)

    # 划分训练集和测试集
    train_df = df_full.loc[:'2023-12-31'].copy()
    test_df = df_full.loc['2024-01-01':].copy()

    print(f"\n数据分割:")
    print(f"  训练集(样本内): 2016-2023 ({len(train_df)}根K线)")
    print(f"  测试集(样本外): 2024-2026 ({len(test_df)}根K线)")
    print(f"  测试集占比: {len(test_df)/(len(train_df)+len(test_df))*100:.1f}%")

    # 运行回测
    print("\n开始样本内回测...")
    result_train = run_backtest(train_df)

    print("\n开始样本外回测...")
    result_test = run_backtest(test_df)

    # 打印对比结果
    print("\n" + "="*80)
    print("样本内 vs 样本外 对比".center(80))
    print("="*80)
    print(f"\n{'指标':<15} | {'训练集 (2016-2023)':<20} | {'测试集 (2024-2026)':<20} | {'状态':<15}")
    print("-"*80)

    # 收益率
    ret_train = result_train['return_pct']
    ret_test = result_test['return_pct']
    status_ret = "[OK] 通过" if ret_test > 0 else "[FAIL] 失败"
    if ret_test > ret_train * 0.7:
        status_ret = "[GOOD] 稳定"
    print(f"{'总收益率':<15} | {ret_train:>18.2f}% | {ret_test:>18.2f}% | {status_ret}")

    # 胜率
    win_train = result_train['win_rate']
    win_test = result_test['win_rate']
    status_win = "[OK] 稳定" if abs(win_test - win_train) < 15 else "[WARN] 偏差大"
    if win_test >= 35:
        status_win = "[GOOD] 优秀"
    print(f"{'胜率':<15} | {win_train:>18.1f}% | {win_test:>18.1f}% | {status_win}")

    # 回撤
    dd_train = result_train['max_dd']
    dd_test = result_test['max_dd']
    status_dd = "[OK] 控制良好" if abs(dd_test) < 25 else "[WARN] 偏大"
    print(f"{'最大回撤':<15} | {dd_train:>18.2f}% | {dd_test:>18.2f}% | {status_dd}")

    # 交易次数
    cnt_train = result_train['total_trades']
    cnt_test = result_test['total_trades']
    print(f"{'交易次数':<15} | {cnt_train:>18} | {cnt_test:>18} | {'-'}")

    # 盈亏比
    pf_train = result_train['profit_factor']
    pf_test = result_test['profit_factor']
    print(f"{'盈亏比':<15} | {pf_train:>18.2f} | {pf_test:>18.2f} | {'-'}")

    print("-"*80)

    # 性能下降分析
    print("\n" + "="*80)
    print("过拟合检测".center(80))
    print("="*80)

    performance_drop = ret_train - ret_test
    win_rate_drop = win_train - win_test

    if performance_drop > 20:
        print(f"[WARN] 收益率大幅下降 {performance_drop:.1f}%")
        print(f"      策略可能存在过拟合！")
    elif performance_drop > 10:
        print(f"[NOTE] 收益率下降 {performance_drop:.1f}%")
        print(f"      轻微过拟合，可接受")
    else:
        print(f"[OK] 收益率表现稳定")

    if result_test['return_pct'] < 0:
        print(f"[FAIL] 严重警告：样本外亏损 {result_test['return_pct']:.1f}%")
        print(f"       策略在样本外完全失效！")

    if win_rate_drop > 15:
        print(f"[WARN] 胜率下降 {win_rate_drop:.1f}%")

    # 最终判决
    print("\n" + "="*80)
    if ret_test > 15 and win_test >= 30:
        print("[SUCCESS] 结论：通过样本外测试！策略具有实战价值。")
        print("           可以考虑小资金实盘验证。")
    elif ret_test > 5 and win_test >= 25:
        print("[OK] 结论：样本外表现尚可，策略有一定价值。")
        print("       建议继续优化或谨慎使用。")
    else:
        print("[FAIL] 结论：样本外表现不佳，策略存在过拟合。")
        print("        不建议直接使用，需要重新设计或优化参数。")
    print("="*80)

if __name__ == "__main__":
    main()
