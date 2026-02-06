# -*- coding: utf-8 -*-
"""
用真实保证金比例（13%）重新优化沪锡
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import time
from datetime import datetime

# 真实沪锡合约规格
CONTRACT_SIZE = 1  # 1吨/手
MARGIN_RATE = 0.13  # 13%保证金（真实比例）

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

# 风险控制参数
MAX_POSITION_RATIO = 0.8  # 最大仓位使用率
MAX_SINGLE_LOSS_PCT = 0.15  # 单笔最大亏损

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

def backtest_realistic(df, params):
    """真实沪锡回测（13%保证金）"""
    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    position = None  # 一次只持有一个仓位
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 只有在没有持仓时才考虑开仓
        if position is not None:
            # 检查止损止盈
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if df['low'].iloc[i] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_triggered = True
                exit_reason = '止损'
            elif (df['stc_prev'].iloc[i] > params['STC_SELL_ZONE'] and
                  current['stc'] < df['stc_prev'].iloc[i]):
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'STC止盈'
            elif current['ema_fast'] < current['ema_slow']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = '趋势反转'

            if exit_triggered:
                pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'entry_price': position['entry_price'],
                    'exit_datetime': current['datetime'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'capital_after': capital
                })

                position = None
            continue

        # 检查开仓信号
        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > params['RSI_FILTER']
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong
        buy_signal = sniper_signal or chase_signal

        if buy_signal and position is None:
            entry_price = current['close']

            # 计算合约价值和保证金（13%）
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATE

            # 计算止损价格
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)

            # 检查单笔最大亏损
            potential_loss_per_contract = (entry_price - stop_loss) * CONTRACT_SIZE
            max_contracts_by_risk = int((capital * MAX_SINGLE_LOSS_PCT) / potential_loss_per_contract)

            # 计算基于保证金的最大手数
            max_contracts_by_margin = int((capital * MAX_POSITION_RATIO) / margin_per_contract)

            # 取较小值
            contracts = min(max_contracts_by_margin, max_contracts_by_risk)

            if contracts <= 0:
                continue

            position = {
                'entry_datetime': current['datetime'],
                'entry_price': entry_price,
                'contracts': contracts,
                'stop_loss': stop_loss,
                'entry_index': i,
                'capital_at_entry': capital
            }

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
    print("沪锡真实保证金重新优化（13%保证金）")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\n真实合约规格:")
    print(f"  合约单位: {CONTRACT_SIZE} 吨/手")
    print(f"  保证金比例: {MARGIN_RATE*100}%")
    print(f"  初始资金: {INITIAL_CAPITAL:,.0f} 元")

    # 加载数据
    csv_file = Path('futures_data_4h/沪锡_4hour.csv')
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"\n数据信息:")
    print(f"  品种: 沪锡")
    print(f"  数据量: {len(df)}条")
    print(f"  时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 参数空间
    PARAM_GRID = {
        'EMA_FAST': [3, 5, 7, 10, 12],
        'EMA_SLOW': [10, 15, 20, 25, 30],
        'RSI_FILTER': [35, 40, 45, 50, 55],
        'RATIO_TRIGGER': [1.05, 1.10, 1.15, 1.20, 1.25],
        'STC_SELL_ZONE': [75, 80, 85, 90]
    }

    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    total_combinations = len(all_combinations)
    print(f"\n参数空间: {total_combinations}种组合")

    best_result = None
    best_params = None
    all_results = []

    start_time = time.time()

    for i, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))

        try:
            result = backtest_realistic(df.copy(), params)

            if result is None:
                continue

            all_results.append({**params, **result})

            if best_result is None or result['return_pct'] > best_result['return_pct']:
                best_result = result
                best_params = params

                if i % 100 == 0 or i == 1:
                    elapsed = time.time() - start_time
                    remaining = elapsed / i * (total_combinations - i) if i > 0 else 0

                    print(f"  [{i}/{total_combinations}] 更好: {result['return_pct']:+.2f}% | "
                          f"EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']}, "
                          f"RATIO={params['RATIO_TRIGGER']:.2f}, STC={params['STC_SELL_ZONE']} | "
                          f"剩余: {remaining:.0f}秒")

        except Exception as e:
            pass

    elapsed = time.time() - start_time

    if best_result is None:
        print(f"\n[失败] 无有效交易")
        return

    print(f"\n[完成] 耗时: {elapsed:.1f}秒")
    print(f"\n最优参数（13%保证金）:")
    print(f"  EMA({best_params['EMA_FAST']},{best_params['EMA_SLOW']}), RSI={best_params['RSI_FILTER']}, "
          f"RATIO={best_params['RATIO_TRIGGER']:.2f}, STC={best_params['STC_SELL_ZONE']}")
    print(f"  最优结果: {best_result['return_pct']:+.2f}%, {best_result['trades']}笔, 胜率{best_result['win_rate']:.1f}%")

    # 保存所有结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('return_pct', ascending=False)

    results_dir = Path('optimization_results_realistic')
    results_dir.mkdir(exist_ok=True)
    results_df.to_csv(results_dir / '沪锡_13percent_margin.csv', index=False, encoding='utf-8-sig')

    print(f"\nTop 10 参数组合:")
    print(f"\n{'排名':<6} {'收益':>10} {'交易数':>6} {'胜率':>8} {'完整参数':<35}")
    print("-" * 80)

    for i, row in results_df.head(10).iterrows():
        params_str = f"EMA({int(row['EMA_FAST'])},{int(row['EMA_SLOW'])}), RSI={int(row['RSI_FILTER'])}, RATIO={row['RATIO_TRIGGER']:.2f}, STC={int(row['STC_SELL_ZONE'])}"
        print(f"{i:<6} {row['return_pct']:>+10.2f}% {int(row['trades']):>6} {row['win_rate']:>7.1f}%  {params_str:<35}")

    print(f"\n所有结果已保存到: {results_dir / '沪锡_13percent_margin.csv'}")
    print(f"\n对比:")
    print(f"  12%保证金最优: EMA(3,10), RSI=35, RATIO=1.25, STC=75 → +458.51%")
    print(f"  13%保证金最优: EMA({best_params['EMA_FAST']},{best_params['EMA_SLOW']}), RSI={best_params['RSI_FILTER']}, RATIO={best_params['RATIO_TRIGGER']:.2f}, STC={best_params['STC_SELL_ZONE']} → {best_result['return_pct']:+.2f}%")

if __name__ == '__main__':
    main()
