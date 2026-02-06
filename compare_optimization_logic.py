# -*- coding: utf-8 -*-
"""
验证 optimize_rigorous.py 的优化逻辑与测试铜时的一致
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

# 导入 optimize_rigorous.py 的回测函数
sys.path.insert(0, str(Path(__file__).parent))
from optimize_rigorous import backtest_with_params as rigorous_backtest

# 测试用的回测函数（与 test_optimize_copper.py 相同）
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

def test_backtest(df, params):
    """测试用回测函数"""
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
    print("验证优化逻辑一致性")
    print("=" * 80)

    # 加载沪铜数据
    csv_file = Path('futures_data_4h/沪铜_4hour.csv')
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"\n数据: {len(df)}条")
    print(f"时间: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 测试多组参数
    test_params = [
        # 已验证参数
        {'EMA_FAST': 5, 'EMA_SLOW': 15, 'RSI_FILTER': 45, 'RATIO_TRIGGER': 1.15, 'STC_SELL_ZONE': 85},
        # 新发现的更好参数
        {'EMA_FAST': 5, 'EMA_SLOW': 20, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.15, 'STC_SELL_ZONE': 80},
        # 参数空间边界值
        {'EMA_FAST': 10, 'EMA_SLOW': 20, 'RSI_FILTER': 50, 'RATIO_TRIGGER': 1.10, 'STC_SELL_ZONE': 80},
        {'EMA_FAST': 7, 'EMA_SLOW': 15, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.10, 'STC_SELL_ZONE': 85},
    ]

    print(f"\n{'='*80}")
    print(f"{'参数组合':<40} {'rigorous':<15} {'test':<15} {'一致':<8}")
    print("-" * 80)

    all_match = True

    for i, params in enumerate(test_params, 1):
        # 使用 optimize_rigorous.py 的函数
        result_rigorous = rigorous_backtest(df.copy(), params)

        # 使用测试函数
        result_test = test_backtest(df.copy(), params)

        # 对比结果
        match = False
        if result_rigorous is None and result_test is None:
            match = True
            status = "OK"
        elif result_rigorous is None or result_test is None:
            match = False
            status = "ERROR"
        elif (abs(result_rigorous['return_pct'] - result_test['return_pct']) < 0.01 and
              result_rigorous['trades'] == result_test['trades'] and
              abs(result_rigorous['win_rate'] - result_test['win_rate']) < 0.1):
            match = True
            status = "OK"
        else:
            match = False
            status = "ERROR"

        param_str = f"#{i} EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']}, RATIO={params['RATIO_TRIGGER']:.2f}, STC={params['STC_SELL_ZONE']}"

        if result_rigorous and result_test:
            print(f"{param_str:<40} {result_rigorous['return_pct']:>+10.2f}%     {result_test['return_pct']:>+10.2f}%     {status:<8}")
        else:
            print(f"{param_str:<40} {'None':>15} {'None':>15} {status:<8}")

        if not match:
            all_match = False
            print(f"  [详细] rigorous: {result_rigorous}")
            print(f"  [详细] test:      {result_test}")

    print("-" * 80)

    if all_match:
        print("\n[OK] 所有测试参数的结果都一致")
        print("    optimize_rigorous.py 的优化逻辑与测试逻辑完全一致")
    else:
        print("\n[ERROR] 存在不一致的结果")
        print("    optimize_rigorous.py 的优化逻辑与测试逻辑不同")

    # 测试沪镍的优化结果（Top 1）
    print(f"\n{'='*80}")
    print("额外测试：验证沪镍优化结果")
    print(f"{'='*80}")

    nickel_file = Path('futures_data_4h/沪镍_4hour.csv')
    if nickel_file.exists():
        df_nickel = pd.read_csv(nickel_file)
        df_nickel['datetime'] = pd.to_datetime(df_nickel['datetime'])

        # progress.csv 中沪镍的最优参数
        nickel_best_params = {
            'EMA_FAST': 7,
            'EMA_SLOW': 15,
            'RSI_FILTER': 40,
            'RATIO_TRIGGER': 1.10,
            'STC_SELL_ZONE': 80
        }

        print(f"\n沪镍最优参数: EMA(7,15), RSI=40, RATIO=1.10, STC=80")
        print(f"预期收益: +344.77%")

        result_rigorous = rigorous_backtest(df_nickel.copy(), nickel_best_params)
        result_test = test_backtest(df_nickel.copy(), nickel_best_params)

        if result_rigorous and result_test:
            print(f"\nrigorous结果: {result_rigorous['return_pct']:+.2f}%, {result_rigorous['trades']}笔, 胜率{result_rigorous['win_rate']:.1f}%")
            print(f"test结果:      {result_test['return_pct']:+.2f}%, {result_test['trades']}笔, 胜率{result_test['win_rate']:.1f}%")

            if abs(result_rigorous['return_pct'] - 344.77) < 0.1:
                print("[OK] 与 progress.csv 中的结果一致")
            else:
                print(f"[WARNING] 与 progress.csv 不一致 (预期 344.77%)")

            if abs(result_rigorous['return_pct'] - result_test['return_pct']) < 0.01:
                print("[OK] 两个函数结果一致")
            else:
                print("[ERROR] 两个函数结果不一致")
        else:
            print("[ERROR] 无法回测沪镍")

if __name__ == '__main__':
    main()
