# -*- coding: utf-8 -*-
"""
完整优化所有22个期货品种
每个品种独立网格搜索，找到最优参数
预计耗时：2-4小时
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import time
from datetime import datetime

# ========== 参数搜索空间 ==========
# 3^4 x 2 = 162种组合
PARAM_GRID = {
    'EMA_FAST': [3, 5, 7],
    'EMA_SLOW': [10, 15, 20],
    'RSI_FILTER': [40, 45, 50],
    'RATIO_TRIGGER': [1.10, 1.15, 1.20],
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
    df = calculate_indicators(df.copy(), params)

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

def optimize_future(csv_file, future_name, log_file):
    """优化单个品种"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{timestamp}] 开始优化: {future_name}\n")
        f.write(f"{'='*80}\n")

    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 优化: {future_name}")
    print(f"{'='*80}")

    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"数据量: {len(df)}条\n")
        f.write(f"时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}\n")

    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"参数组合数: {len(all_combinations)}\n")

    best_result = None
    best_params = None
    all_results = []

    start_time = time.time()

    for i, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))

        try:
            result = backtest_with_params(df.copy(), params)

            if result is None:
                continue

            result.update(params)
            all_results.append(result)

            if best_result is None or result['return_pct'] > best_result['return_pct']:
                best_result = result
                best_params = params

            # 每20次更新一次进度
            if i % 20 == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / i * (len(all_combinations) - i)
                progress = i / len(all_combinations) * 100

                log_msg = f"  进度: {i}/{len(all_combinations)} ({progress:.0f}%) | 最佳: {best_result['return_pct']:+.2f}% | 剩余: {remaining:.0f}秒"
                print(log_msg)

                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(log_msg + '\n')

        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"  错误: {e}\n")

    elapsed = time.time() - start_time

    if best_result is None:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"  结果: 无有效交易\n")
        print(f"  结果: 无有效交易")
        return None

    # 保存详细结果
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('return_pct', ascending=False)
    detail_file = f'optimization_results/{future_name}_detailed.csv'
    results_df.to_csv(detail_file, index=False, encoding='utf-8-sig')

    result_msg = f"\n  完成! 耗时: {elapsed:.1f}秒\n"
    result_msg += f"  最佳参数: EMA({best_params['EMA_FAST']},{best_params['EMA_SLOW']}), RSI={best_params['RSI_FILTER']}, RATIO={best_params['RATIO_TRIGGER']:.2f}, STC={best_params['STC_SELL_ZONE']}\n"
    result_msg += f"  最佳结果: {best_result['return_pct']:+.2f}%, {best_result['trades']}笔, 胜率{best_result['win_rate']:.1f}%\n"
    result_msg += f"  详细结果: {detail_file}"

    print(result_msg)

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(result_msg + '\n')

    return {
        'name': future_name,
        'return_pct': best_result['return_pct'],
        'trades': best_result['trades'],
        'win_rate': best_result['win_rate'],
        'final_capital': best_result['final_capital'],
        'EMA_FAST': best_params['EMA_FAST'],
        'EMA_SLOW': best_params['EMA_SLOW'],
        'RSI_FILTER': best_params['RSI_FILTER'],
        'RATIO_TRIGGER': best_params['RATIO_TRIGGER'],
        'STC_SELL_ZONE': best_params['STC_SELL_ZONE'],
        'optimization_time': elapsed
    }

def main():
    start_timestamp = datetime.now()

    print("=" * 80)
    print("完整优化所有22个期货品种")
    print(f"开始时间: {start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"参数空间: 3×3×3×3×2 = 162种组合")
    print(f"预计耗时: 2-4小时")
    print("=" * 80)

    # 创建结果目录
    results_dir = Path('optimization_results')
    results_dir.mkdir(exist_ok=True)

    # 创建日志文件
    log_file = results_dir / f'optimization_log_{start_timestamp.strftime("%Y%m%d_%H%M%S")}.txt'

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"优化开始时间: {start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"参数空间: {len(PARAM_GRID['EMA_FAST'])}×{len(PARAM_GRID['EMA_SLOW'])}×{len(PARAM_GRID['RSI_FILTER'])}×{len(PARAM_GRID['RATIO_TRIGGER'])}×{len(PARAM_GRID['STC_SELL_ZONE'])} = {len(list(product(*PARAM_GRID.values())))}种组合\n")
        f.write("="*80 + "\n")

    data_dir = Path('futures_data_4h')
    csv_files = sorted(data_dir.glob('*.csv'))

    # 原始收益（铜参数）
    original_returns = {
        '白糖': -7.65, '沪锌': 79.95, '沪铜': 56.36, '沪铝': 28.77, '豆油': 22.37,
        'PTA': 14.48, '棉花': 12.83, '沪锡': -0.68, 'PVC': -4.25, '沪镍': -6.29,
        'PP': -10.71, '热卷': -10.19, '铁矿石': -14.31, '甲醇': -16.93,
        '棕榈油': -17.65, '沪铅': -25.14, '纯碱': -26.42, '玻璃': -44.05, '焦炭': -77.39
    }

    results = []
    completed = 0

    for csv_file in csv_files:
        future_name = csv_file.stem.replace('_4hour', '')

        try:
            result = optimize_future(csv_file, future_name, log_file)

            if result:
                result['original_return'] = original_returns.get(future_name, 0)
                result['improvement'] = result['return_pct'] - result['original_return']
                results.append(result)
                completed += 1

                # 保存当前进度
                if len(results) > 0:
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('return_pct', ascending=False)
                    results_df.to_csv(results_dir / 'current_progress.csv', index=False, encoding='utf-8-sig')

        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n[错误] {future_name}: {e}\n")
            print(f"\n[错误] {future_name}: {e}")

    # 最终报告
    end_timestamp = datetime.now()
    total_time = (end_timestamp - start_timestamp).total_seconds() / 60

    print("\n" + "=" * 80)
    print("优化完成！")
    print(f"结束时间: {end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time:.1f}分钟 ({total_time/60:.1f}小时)")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)

    # 保存最终汇总
    results_df.to_csv(results_dir / 'final_summary.csv', index=False, encoding='utf-8-sig')

    # 输出报告
    print(f"\n{'='*80}")
    print("最终排名（优化后）")
    print(f"{'='*80}")
    print(f"\n{'排名':<6} {'品种':<10} {'原始收益':>12} {'优化后收益':>12} {'提升':>10} {'交易数':>6} {'胜率':>8}")
    print("-" * 80)

    for i, row in results_df.iterrows():
        print(f"{i:<6} {row['name']:<10} {row['original_return']:>+10.2f}% {row['return_pct']:>+12.2f}% "
              f"{row['improvement']:>+10.2f}% {row['trades']:>6} {row['win_rate']:>7.1f}%")

    print("\n" + "=" * 80)
    print("推荐配置（优化后Top 10）")
    print("=" * 80)

    for i in range(min(10, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n{i+1}. {row['name']} - {row['return_pct']:+.2f}% (提升{row['improvement']:+.2f}%)")
        print(f"   参数: EMA({row['EMA_FAST']},{row['EMA_SLOW']}), RSI={row['RSI_FILTER']}, "
              f"RATIO={row['RATIO_TRIGGER']:.2f}, STC={row['STC_SELL_ZONE']}")
        print(f"   统计: {row['trades']}笔交易, 胜率{row['win_rate']:.1f}%")

    print(f"\n所有结果已保存到: {results_dir}")
    print(f"  - final_summary.csv: 汇总报告")
    print(f"  - *_detailed.csv: 每个品种的详细优化结果")
    print(f"  - optimization_log_*.txt: 完整日志")

if __name__ == '__main__':
    main()
