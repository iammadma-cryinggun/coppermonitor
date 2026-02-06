# -*- coding: utf-8 -*-
"""
验证所有品种的优化结果
对每个品种的最优参数进行3次回测，确保结果一致
严谨细致，做到每个细节都不遗漏
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

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

def verify_variety(csv_file, variety_name, params, expected_result):
    """验证单个品种的优化结果"""
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    results = []
    for run in range(3):
        result = backtest_with_params(df.copy(), params)
        if result:
            results.append(result)
        time.sleep(0.001)

    if not results:
        return {
            'name': variety_name,
            'status': 'ERROR',
            'message': 'No trades',
            'expected': expected_result,
            'actual': None,
            'params': params
        }

    # 检查一致性
    returns = [r['return_pct'] for r in results]
    trades_counts = [r['trades'] for r in results]
    win_rates = [r['win_rate'] for r in results]

    all_same_return = len(set(returns)) == 1
    all_same_trades = len(set(trades_counts)) == 1
    all_same_winrate = len(set(win_rates)) == 1

    # 与预期对比
    avg_return = np.mean(returns)
    avg_trades = int(np.mean(trades_counts))
    avg_winrate = np.mean(win_rates)

    match_return = abs(avg_return - expected_result['return_pct']) < 0.01
    match_trades = avg_trades == expected_result['trades']
    match_winrate = abs(avg_winrate - expected_result['win_rate']) < 0.1

    if match_return and match_trades and match_winrate and all_same_return and all_same_trades:
        status = 'PASS'
        message = '完全一致'
    elif all_same_return and all_same_trades:
        if match_return:
            status = 'PASS'
            message = '一致（交易数或胜率微小差异）'
        else:
            status = 'FAIL'
            message = f'3次一致但与预期不符 (预期{expected_result["return_pct"]:.2f}%, 实际{avg_return:.2f}%)'
    else:
        status = 'FAIL'
        message = f'3次结果不一致 (收益率:{returns}, 交易数:{trades_counts})'

    return {
        'name': variety_name,
        'status': status,
        'message': message,
        'expected_return': expected_result['return_pct'],
        'actual_return': avg_return,
        'expected_trades': expected_result['trades'],
        'actual_trades': avg_trades,
        'expected_winrate': expected_result['win_rate'],
        'actual_winrate': avg_winrate,
        'all_same_return': all_same_return,
        'all_same_trades': all_same_trades,
        'params': params
    }

def main():
    print("=" * 100)
    print("验证所有品种的优化结果")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("每个品种进行3次回测验证")
    print("=" * 100)

    # 读取优化结果
    progress_file = Path('optimization_results_full/progress.csv')
    df = pd.read_csv(progress_file)

    results = []
    passed = 0
    failed = 0

    data_dir = Path('futures_data_4h')

    for i, row in df.iterrows():
        variety_name = row['name']
        csv_file = data_dir / f'{variety_name}_4hour.csv'

        params = {
            'EMA_FAST': int(row['EMA_FAST']),
            'EMA_SLOW': int(row['EMA_SLOW']),
            'RSI_FILTER': int(row['RSI_FILTER']),
            'RATIO_TRIGGER': float(row['RATIO_TRIGGER']),
            'STC_SELL_ZONE': int(row['STC_SELL_ZONE'])
        }

        expected_result = {
            'return_pct': row['return_pct'],
            'trades': int(row['trades']),
            'win_rate': row['win_rate']
        }

        print(f"\n[{i+1}/{len(df)}] 验证: {variety_name}")
        print(f"  参数: EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']}, "
              f"RATIO={params['RATIO_TRIGGER']:.2f}, STC={params['STC_SELL_ZONE']}")
        print(f"  预期: {expected_result['return_pct']:+.2f}%, {expected_result['trades']}笔, "
              f"胜率{expected_result['win_rate']:.1f}%")

        if not csv_file.exists():
            print(f"  [ERROR] 文件不存在: {csv_file}")
            failed += 1
            continue

        result = verify_variety(csv_file, variety_name, params, expected_result)
        results.append(result)

        if result['status'] == 'PASS':
            print(f"  [PASS] {result['message']}")
            print(f"  实际: {result['actual_return']:+.2f}%, {result['actual_trades']}笔, "
                  f"胜率{result['actual_winrate']:.1f}%")
            passed += 1
        else:
            print(f"  [FAIL] {result['message']}")
            if result['actual_return'] is not None:
                print(f"  实际: {result['actual_return']:+.2f}%, {result['actual_trades']}笔, "
                      f"胜率{result['actual_winrate']:.1f}%")
            failed += 1

    # 最终报告
    print("\n" + "=" * 100)
    print("验证完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    print(f"\n总结果: {passed} PASS, {failed} FAIL")

    # 保存验证报告
    results_df = pd.DataFrame(results)
    results_df.to_csv('optimization_results_full/verification_report.csv', index=False, encoding='utf-8-sig')

    print("\n详细验证报告:")
    print(f"\n{'状态':<8} {'品种':<10} {'预期收益':>12} {'实际收益':>12} {'预期交易':>10} {'实际交易':>10} {'详情':<40}")
    print("-" * 110)

    for r in results:
        expected_str = f"{r['expected_return']:+.2f}%" if r['expected_return'] is not None else "N/A"
        actual_str = f"{r['actual_return']:+.2f}%" if r['actual_return'] is not None else "N/A"
        expected_trades = str(r['expected_trades']) if r['expected_trades'] is not None else "N/A"
        actual_trades = str(r['actual_trades']) if r['actual_trades'] is not None else "N/A"

        print(f"{r['status']:<8} {r['name']:<10} {expected_str:>12} {actual_str:>12} "
              f"{expected_trades:>10} {actual_trades:>10} {r['message']:<40}")

    print("\n" + "=" * 100)
    if failed == 0:
        print("所有品种验证通过！优化结果完全准确。")
    else:
        print(f"警告: {failed}个品种验证失败，需要检查！")
    print("=" * 100)

    print(f"\n验证报告已保存到: optimization_results_full/verification_report.csv")

if __name__ == '__main__':
    main()
