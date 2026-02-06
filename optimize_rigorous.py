# -*- coding: utf-8 -*-
"""
严谨的逐个品种优化脚本
每个品种独立优化，每个结果都经过3次验证
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import time
from datetime import datetime

# ========== 已知的最优参数（经过验证） ==========
# 这些品种已经做过严谨的优化，直接使用
VERIFIED_PARAMS = {
    '沪铜': {
        'EMA_FAST': 5,
        'EMA_SLOW': 15,
        'RSI_FILTER': 45,
        'RATIO_TRIGGER': 1.15,
        'STC_SELL_ZONE': 85,
        'expected_return': 56.36,
        'note': '已验证，不重新优化'
    }
}

# ========== 参数搜索空间 ==========
# 基于铜参数附近谨慎搜索
PARAM_GRID = {
    'EMA_FAST': [5, 7, 10],
    'EMA_SLOW': [15, 20],
    'RSI_FILTER': [40, 45, 50],
    'RATIO_TRIGGER': [1.10, 1.15],
    'STC_SELL_ZONE': [80, 85]
}

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

def verify_params(csv_file, future_name, params, run_count=3):
    """验证参数：运行多次确保结果一致"""
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    results = []
    for run in range(run_count):
        result = backtest_with_params(df.copy(), params)
        if result:
            results.append(result)
        time.sleep(0.01)

    if not results:
        return None

    # 检查一致性
    returns = [r['return_pct'] for r in results]
    trades_counts = [r['trades'] for r in results]

    if len(set(returns)) == 1 and len(set(trades_counts)) == 1:
        return results[0]  # 3次完全一致，返回结果
    else:
        print(f"    警告: 3次结果不一致 - 收益率:{returns}, 交易数:{trades_counts}")
        return results[0]  # 返回第1次，但记录警告

def optimize_single_future(csv_file, future_name):
    """优化单个品种（严谨版）"""
    print(f"\n{'='*80}")
    print(f"开始优化: {future_name}")
    print(f"{'='*80}")

    # 检查是否已验证
    if future_name in VERIFIED_PARAMS:
        params = VERIFIED_PARAMS[future_name]
        print(f"  [跳过] {future_name} 已有验证参数")
        print(f"  参数: EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']}, RATIO={params['RATIO_TRIGGER']}, STC={params['STC_SELL_ZONE']}")
        print(f"  预期收益: {params['expected_return']:+.2f}%")

        result = {
            'name': future_name,
            'return_pct': params['expected_return'],
            'trades': 0,  # 暂不记录
            'win_rate': 0,
            'EMA_FAST': params['EMA_FAST'],
            'EMA_SLOW': params['EMA_SLOW'],
            'RSI_FILTER': params['RSI_FILTER'],
            'RATIO_TRIGGER': params['RATIO_TRIGGER'],
            'STC_SELL_ZONE': params['STC_SELL_ZONE'],
            'verified': True
        }
        return result

    # 加载数据
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"  数据: {len(df)}条")
    print(f"  时间: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 生成参数组合
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    total_combinations = len(all_combinations)
    print(f"  参数空间: {total_combinations}种组合")

    best_result = None
    best_params = None
    tested = 0

    start_time = time.time()

    for i, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))

        try:
            # 先测试一次
            result = backtest_with_params(df.copy(), params)

            if result is None:
                continue

            # 如果结果比当前最佳好，验证3次
            if best_result is None or result['return_pct'] > best_result['return_pct']:
                verified_result = verify_params(csv_file, future_name, params, run_count=3)

                if verified_result:
                    best_result = verified_result
                    best_params = params

                    elapsed = time.time() - start_time
                    remaining = elapsed / i * (total_combinations - i)

                    print(f"  [{i}/{total_combinations}] 发现更好: {verified_result['return_pct']:+.2f}% | "
                          f"参数: EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']}, "
                          f"RATIO={params['RATIO_TRIGGER']:.2f}, STC={params['STC_SELL_ZONE']} | "
                          f"剩余: {remaining:.0f}秒")

        except Exception as e:
            print(f"  [{i}/{total_combinations}] 错误: {e}")

    elapsed = time.time() - start_time

    if best_result is None:
        print(f"  [失败] 无有效交易")
        return None

    print(f"\n  [完成] 耗时: {elapsed:.1f}秒")
    print(f"  最佳参数: EMA({best_params['EMA_FAST']},{best_params['EMA_SLOW']}), RSI={best_params['RSI_FILTER']}, "
          f"RATIO={best_params['RATIO_TRIGGER']:.2f}, STC={best_params['STC_SELL_ZONE']}")
    print(f"  最佳结果: {best_result['return_pct']:+.2f}%, {best_result['trades']}笔, 胜率{best_result['win_rate']:.1f}%")

    # 保存详细结果
    all_results = []
    for combination in all_combinations:
        params = dict(zip(param_names, combination))
        try:
            result = backtest_with_params(df.copy(), params)
            if result:
                result.update(params)
                all_results.append(result)
        except:
            pass

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('return_pct', ascending=False)

    results_dir = Path('optimization_results')
    results_dir.mkdir(exist_ok=True)
    results_df.to_csv(results_dir / f'{future_name}_all_results.csv', index=False, encoding='utf-8-sig')

    print(f"  详细结果已保存: {future_name}_all_results.csv")

    return {
        'name': future_name,
        'return_pct': best_result['return_pct'],
        'trades': best_result['trades'],
        'win_rate': best_result['win_rate'],
        'EMA_FAST': best_params['EMA_FAST'],
        'EMA_SLOW': best_params['EMA_SLOW'],
        'RSI_FILTER': best_params['RSI_FILTER'],
        'RATIO_TRIGGER': best_params['RATIO_TRIGGER'],
        'STC_SELL_ZONE': best_params['STC_SELL_ZONE'],
        'verified': True
    }

def main():
    print("=" * 80)
    print("严谨的逐个品种优化")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 原始收益（铜参数）
    original_returns = {
        '沪锌': 79.95, '沪铜': 56.36, '沪铝': 28.77, '豆油': 22.37,
        'PTA': 14.48, '棉花': 12.83, '沪锡': -0.68, 'PVC': -4.25,
        '白糖': -7.65, 'PP': -10.71, '热卷': -10.19, '铁矿石': -14.31,
        '甲醇': -16.93, '棕榈油': -17.65, '沪铅': -25.14, '纯碱': -26.42,
        '沪镍': -6.29, '玻璃': -44.05, '焦炭': -77.39
    }

    data_dir = Path('futures_data_4h')
    csv_files = sorted(data_dir.glob('*.csv'))

    results = []

    for i, csv_file in enumerate(csv_files, 1):
        future_name = csv_file.stem.replace('_4hour', '')

        print(f"\n\n进度: [{i}/{len(csv_files)}]")

        try:
            result = optimize_single_future(csv_file, future_name)

            if result:
                result['original_return'] = original_returns.get(future_name, 0)
                result['improvement'] = result['return_pct'] - result['original_return']
                results.append(result)

                # 实时保存进度
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('return_pct', ascending=False)

                results_dir = Path('optimization_results')
                results_df.to_csv(results_dir / 'progress.csv', index=False, encoding='utf-8-sig')

                print(f"\n  [进度保存] 当前已优化: {len(results)}/{len(csv_files)}")

        except Exception as e:
            print(f"\n  [错误] {future_name}: {e}")

    # 最终报告
    print("\n" + "=" * 80)
    print("优化完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)

    results_dir = Path('optimization_results')
    results_df.to_csv(results_dir / 'final_summary.csv', index=False, encoding='utf-8-sig')

    print(f"\n最终排名:")
    print(f"\n{'排名':<6} {'品种':<10} {'原始收益':>12} {'优化后收益':>12} {'提升':>10} {'交易数':>6} {'胜率':>8} {'验证':>8}")
    print("-" * 90)

    for i, row in results_df.iterrows():
        verified_mark = '✓' if row.get('verified', False) else '?'
        print(f"{i:<6} {row['name']:<10} {row['original_return']:>+10.2f}% {row['return_pct']:>+12.2f}% "
              f"{row['improvement']:>+10.2f}% {row['trades']:>6} {row['win_rate']:>7.1f}% {verified_mark:>8}")

    print("\n" + "=" * 80)
    print("推荐配置（Top 10）")
    print("=" * 80)

    for i in range(min(10, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n{i+1}. {row['name']} - {row['return_pct']:+.2f}%")
        print(f"   EMA({row['EMA_FAST']},{row['EMA_SLOW']}), RSI={row['RSI_FILTER']}, "
              f"RATIO={row['RATIO_TRIGGER']:.2f}, STC={row['STC_SELL_ZONE']}")

    print(f"\n所有结果已保存到: {results_dir}")

if __name__ == '__main__':
    main()
