# -*- coding: utf-8 -*-
"""
验证新发现的铜参数是否稳定
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# 固定参数
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50
STOP_LOSS_PCT = 0.02
INITIAL_CAPITAL = 100000
MARGIN_RATIO = 0.15
CONTRACT_SIZE = 5

def calculate_indicators(df, params):
    """计算技术指标"""
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

    exp1 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    stc_macd = df['close'].ewm(span=STC_FAST, adjust=False).mean() - \
               df['close'].ewm(span=STC_SLOW, adjust=False).mean()
    stoch_period = STC_LENGTH
    min_macd = stc_macd.rolling(window=stoch_period).min()
    max_macd = stc_macd.rolling(window=stoch_period).max()
    stoch_k = 100 * (stc_macd - min_macd) / (max_macd - min_macd).replace(0, np.nan)
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=3).mean()
    min_stoch_d = stoch_d.rolling(window=stoch_period).min()
    max_stoch_d = stoch_d.rolling(window=stoch_period).max()
    stc_raw = 100 * (stoch_d - min_stoch_d) / (max_stoch_d - min_stoch_d).replace(0, np.nan)
    stc_raw = stc_raw.fillna(50)
    df['stc'] = stc_raw.rolling(window=3).mean()

    df['ratio_prev'] = df['ratio'].shift(1)
    df['stc_prev'] = df['stc'].shift(1)

    return df

def backtest_with_params(df, params):
    """用指定参数回测"""
    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > params['RSI_FILTER']
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong
        buy_signal = sniper_signal or chase_signal

        if df['ratio_prev'].iloc[i] > 0:
            if df['ratio_prev'].iloc[i] > 2.0:
                position_size = 2.0
            elif df['ratio_prev'].iloc[i] > 1.5:
                position_size = 1.5
            elif df['ratio_prev'].iloc[i] > 1.0:
                position_size = 1.2
            else:
                position_size = 1.0
        else:
            position_size = 1.0

        stop_loss = current['close'] * (1 - STOP_LOSS_PCT)

        if buy_signal and position is None:
            entry_price = current['close']
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATIO
            available_margin = capital * position_size
            contracts = int(available_margin / margin_per_contract)

            if contracts <= 0:
                continue

            position = {
                'entry_datetime': current['datetime'],
                'entry_price': entry_price,
                'contracts': contracts,
                'stop_loss': stop_loss,
                'entry_index': i
            }

        elif position is not None:
            for j in range(position['entry_index'] + 1, len(df)):
                bar = df.iloc[j]

                exit_triggered = False
                exit_price = None

                if bar['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_triggered = True
                elif (df['stc_prev'].iloc[j] > params['STC_SELL_ZONE'] and
                      bar['stc'] < df['stc_prev'].iloc[j]):
                    exit_price = bar['close']
                    exit_triggered = True
                elif bar['ema_fast'] < bar['ema_slow']:
                    exit_price = bar['close']
                    exit_triggered = True

                if exit_triggered:
                    pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                    capital += pnl
                    trades.append({'pnl': pnl})
                    position = None
                    break

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    total_pnl = trades_df['pnl'].sum()
    return_pct = total_pnl / INITIAL_CAPITAL * 100
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    return {
        'return_pct': return_pct,
        'trades': total_trades,
        'win_rate': win_rate,
        'final_capital': capital
    }

def main():
    print("=" * 80)
    print("验证新发现的铜参数")
    print("=" * 80)

    # 加载沪铜数据
    csv_file = Path('futures_data_4h/沪铜_4hour.csv')
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 已验证参数
    verified_params = {
        'EMA_FAST': 5,
        'EMA_SLOW': 15,
        'RSI_FILTER': 45,
        'RATIO_TRIGGER': 1.15,
        'STC_SELL_ZONE': 85
    }

    # 新发现参数
    new_params = {
        'EMA_FAST': 5,
        'EMA_SLOW': 20,
        'RSI_FILTER': 40,
        'RATIO_TRIGGER': 1.15,
        'STC_SELL_ZONE': 80
    }

    print("\n已验证参数: EMA(5,15), RSI=45, RATIO=1.15, STC=85")
    print("新发现参数: EMA(5,20), RSI=40, RATIO=1.15, STC=80")

    print("\n" + "=" * 80)
    print("验证已验证参数 (10次)")
    print("=" * 80)

    verified_results = []
    for i in range(10):
        result = backtest_with_params(df.copy(), verified_params)
        if result:
            verified_results.append(result['return_pct'])
        time.sleep(0.01)

    print(f"\n已验证参数 10次运行结果:")
    print(f"  收益率: {verified_results}")
    print(f"  平均: {np.mean(verified_results):+.2f}%")
    print(f"  标准差: {np.std(verified_results):.4f}")
    print(f"  最小: {min(verified_results):+.2f}%")
    print(f"  最大: {max(verified_results):+.2f}%")

    all_same = len(set(verified_results)) == 1
    if all_same:
        print("  [OK] 10次结果完全一致")
    else:
        print(f"  [WARNING] 10次结果不一致，有 {len(set(verified_results))} 种不同值")

    print("\n" + "=" * 80)
    print("验证新发现参数 (10次)")
    print("=" * 80)

    new_results = []
    for i in range(10):
        result = backtest_with_params(df.copy(), new_params)
        if result:
            new_results.append(result['return_pct'])
            if i == 0:
                print(f"  第1次: {result['return_pct']:+.2f}%, {result['trades']}笔, "
                      f"胜率{result['win_rate']:.1f}%")
        time.sleep(0.01)

    print(f"\n新参数 10次运行结果:")
    print(f"  收益率: {new_results}")
    print(f"  平均: {np.mean(new_results):+.2f}%")
    print(f"  标准差: {np.std(new_results):.4f}")
    print(f"  最小: {min(new_results):+.2f}%")
    print(f"  最大: {max(new_results):+.2f}%")

    all_same_new = len(set(new_results)) == 1
    if all_same_new:
        print("  [OK] 10次结果完全一致")
    else:
        print(f"  [WARNING] 10次结果不一致，有 {len(set(new_results))} 种不同值")

    print("\n" + "=" * 80)
    print("对比结论")
    print("=" * 80)

    verified_avg = np.mean(verified_results)
    new_avg = np.mean(new_results)
    improvement = new_avg - verified_avg

    print(f"\n已验证参数平均收益: {verified_avg:+.2f}%")
    print(f"新参数平均收益:     {new_avg:+.2f}%")
    print(f"提升:               {improvement:+.2f}%")

    if all_same and all_same_new:
        if improvement > 1:
            print("\n[WARNING] 新参数确实更好且稳定")
            print("           已验证参数可能需要更新")
        elif improvement < -1:
            print("\n[INFO] 新参数反而更差，可能是偶然")
        else:
            print("\n[OK] 两种参数基本相当")
    else:
        print("\n[WARNING] 结果不稳定，无法下结论")

if __name__ == '__main__':
    main()
