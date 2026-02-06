# -*- coding: utf-8 -*-
"""
对所有22个期货品种分别进行参数优化
每个品种独立网格搜索，找到最优参数
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import time

# ========== 参数搜索空间 ==========
# 基于铜参数附近扩展搜索
PARAM_GRID = {
    'EMA_FAST': [3, 5, 7, 10],
    'EMA_SLOW': [10, 15, 20, 25],
    'RSI_FILTER': [35, 40, 45, 50],
    'RATIO_TRIGGER': [1.05, 1.10, 1.15, 1.20],
    'STC_SELL_ZONE': [75, 80, 85, 90]
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

        # 入场条件
        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > params['RSI_FILTER']
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong
        buy_signal = sniper_signal or chase_signal

        # 仓位计算
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

        # 买入
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

        # 卖出
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
                    trades.append({
                        'pnl': pnl,
                        'holding_bars': j - position['entry_index']
                    })
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

def optimize_future(csv_file, future_name):
    """优化单个品种"""
    print(f"\n{'='*80}")
    print(f"[{future_name}] 开始优化")
    print(f"{'='*80}")

    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"数据量: {len(df)}条")
    print(f"时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 生成所有参数组合
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    total_combinations = len(all_combinations)
    print(f"参数组合数: {total_combinations} (4^5 = 1024)")

    best_result = None
    best_params = None
    results = []

    start_time = time.time()

    for i, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))

        try:
            result = backtest_with_params(df.copy(), params)

            if result is None:
                continue

            result.update(params)
            results.append(result)

            if best_result is None or result['return_pct'] > best_result['return_pct']:
                best_result = result
                best_params = params

            # 进度报告
            if i % 100 == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / i * (total_combinations - i)
                print(f"  进度: {i}/{total_combinations} ({i/total_combinations*100:.1f}%) | "
                      f"当前最佳: {best_result['return_pct']:+.2f}% | "
                      f"预计剩余: {remaining:.0f}秒")

        except Exception as e:
            pass

    elapsed = time.time() - start_time

    if best_result is None:
        print(f"  [失败] 无有效交易")
        return None

    print(f"\n  [完成] 耗时: {elapsed:.1f}秒")
    print(f"\n  最佳参数:")
    print(f"    EMA_FAST:        {best_params['EMA_FAST']}")
    print(f"    EMA_SLOW:        {best_params['EMA_SLOW']}")
    print(f"    RSI_FILTER:      {best_params['RSI_FILTER']}")
    print(f"    RATIO_TRIGGER:   {best_params['RATIO_TRIGGER']:.2f}")
    print(f"    STC_SELL_ZONE:   {best_params['STC_SELL_ZONE']}")

    print(f"\n  最佳结果:")
    print(f"    收益率:          {best_result['return_pct']:+.2f}%")
    print(f"    交易次数:        {best_result['trades']}")
    print(f"    胜率:            {best_result['win_rate']:.1f}%")
    print(f"    最终资金:        {best_result['final_capital']:,.0f}")

    # 保存详细结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)
    output_file = f'optimization_results/{future_name}_full_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

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
    print("=" * 80)
    print("对所有22个期货品种分别进行参数优化")
    print(f"参数空间: 4^5 = 1024种组合")
    print("=" * 80)

    # 创建结果目录
    results_dir = Path('optimization_results')
    results_dir.mkdir(exist_ok=True)

    data_dir = Path('futures_data_4h')
    csv_files = sorted(data_dir.glob('*.csv'))

    results = []

    for csv_file in csv_files:
        future_name = csv_file.stem.replace('_4hour', '')

        try:
            result = optimize_future(csv_file, future_name)

            if result:
                results.append(result)
        except Exception as e:
            print(f"  [错误] {future_name}: {e}")

    if not results:
        print("\n无有效结果")
        return

    # 保存汇总
    results_df = pd.DataFrame(results)

    # 添加原始收益（铜参数）
    original_returns = {
        '白糖': -7.65,
        '沪锌': 79.95,
        '沪铜': 56.36,
        '沪铝': 28.77,
        '豆油': 22.37,
        'PTA': 14.48,
        '棉花': 12.83,
        '沪锡': -0.68,
        'PVC': -4.25,
        '沪镍': -6.29,
        'PP': -10.71,
        '热卷': -10.19,
        '铁矿石': -14.31,
        '甲醇': -16.93,
        '焦煤': -17.02,
        '棕榈油': -17.65,
        '沪铅': -25.14,
        '纯碱': -26.42,
        '玻璃': -44.05,
        '焦炭': -77.39
    }

    results_df['original_return'] = results_df['name'].map(original_returns)
    results_df['improvement'] = results_df['return_pct'] - results_df['original_return']
    results_df['improvement_pct'] = (results_df['improvement'] / results_df['original_return'].abs() * 100)

    # 排序
    results_df = results_df.sort_values('return_pct', ascending=False)

    # 保存汇总
    summary_file = 'optimization_results/all_futures_summary.csv'
    results_df.to_csv(summary_file, index=False, encoding='utf-8-sig')

    # 输出报告
    print("\n" + "=" * 80)
    print("优化完成！汇总报告")
    print("=" * 80)

    print(f"\n详细结果已保存到: optimization_results/")
    print(f"汇总文件: {summary_file}")

    print(f"\n{'排名':<6} {'品种':<10} {'原始收益':>12} {'优化后收益':>12} {'提升':>10} {'提升幅度':>10} {'交易数':>6} {'胜率':>8}")
    print("-" * 100)

    for i, row in results_df.iterrows():
        print(f"{i+1:<6} {row['name']:<10} {row['original_return']:>+10.2f}% {row['return_pct']:>+12.2f}% "
              f"{row['improvement']:>+10.2f}% {row['improvement_pct']:>+9.1f}% {row['trades']:>6} {row['win_rate']:>7.1f}%")

    print("\n" + "=" * 80)
    print("推荐配置（优化后前10名）")
    print("=" * 80)

    for i in range(min(10, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n{i+1}. {row['name']} - 收益 {row['return_pct']:+.2f}%")
        print(f"   参数: EMA({row['EMA_FAST']},{row['EMA_SLOW']}), RSI={row['RSI_FILTER']}, "
              f"RATIO={row['RATIO_TRIGGER']:.2f}, STC={row['STC_SELL_ZONE']}")
        print(f"   统计: {row['trades']}笔交易, 胜率{row['win_rate']:.1f}%")

if __name__ == '__main__':
    main()
