# -*- coding: utf-8 -*-
"""
分批优化期货品种
每次只优化部分品种，避免卡死
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import time

# ========== 缩小的参数搜索空间 ==========
# 减少到 3^4 = 81种组合
PARAM_GRID = {
    'EMA_FAST': [3, 5, 7],
    'EMA_SLOW': [10, 15, 20],
    'RSI_FILTER': [40, 45, 50],
    'RATIO_TRIGGER': [1.10, 1.15, 1.20],
    'STC_SELL_ZONE': [80, 85]  # 只测试2个值
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

def optimize_future(csv_file, future_name):
    """优化单个品种"""
    print(f"\n{'='*70}")
    print(f"优化: {future_name}")
    print(f"{'='*70}")

    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    print(f"参数组合数: {len(all_combinations)}")

    best_result = None
    best_params = None

    start_time = time.time()

    for i, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))

        try:
            result = backtest_with_params(df.copy(), params)

            if result is None:
                continue

            if best_result is None or result['return_pct'] > best_result['return_pct']:
                best_result = result
                best_params = params

            if i % 20 == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / i * (len(all_combinations) - i)
                print(f"  {i}/{len(all_combinations)} ({i/len(all_combinations)*100:.0f}%) | "
                      f"最佳: {best_result['return_pct']:+.2f}% | 剩余: {remaining:.0f}秒")

        except Exception as e:
            pass

    elapsed = time.time() - start_time

    if best_result is None:
        print(f"  无有效交易")
        return None

    print(f"\n  完成! 耗时: {elapsed:.1f}秒")
    print(f"  最佳参数: EMA({best_params['EMA_FAST']},{best_params['EMA_SLOW']}), "
          f"RSI={best_params['RSI_FILTER']}, RATIO={best_params['RATIO_TRIGGER']:.2f}, STC={best_params['STC_SELL_ZONE']}")
    print(f"  最佳结果: {best_result['return_pct']:+.2f}%, {best_result['trades']}笔, 胜率{best_result['win_rate']:.1f}%")

    return {
        'name': future_name,
        'return_pct': best_result['return_pct'],
        'trades': best_result['trades'],
        'win_rate': best_result['win_rate'],
        'EMA_FAST': best_params['EMA_FAST'],
        'EMA_SLOW': best_params['EMA_SLOW'],
        'RSI_FILTER': best_params['RSI_FILTER'],
        'RATIO_TRIGGER': best_params['RATIO_TRIGGER'],
        'STC_SELL_ZONE': best_params['STC_SELL_ZONE']
    }

def main():
    print("=" * 70)
    print("分批优化期货品种")
    print(f"参数空间: 3×3×3×3×2 = 162种组合")
    print("=" * 70)

    # 按收益排序的前10名
    top_futures = [
        ('沪锌_4hour.csv', '沪锌', 79.95),
        ('沪铜_4hour.csv', '沪铜', 56.36),
        ('沪铝_4hour.csv', '沪铝', 28.77),
        ('豆油_4hour.csv', '豆油', 22.37),
        ('PTA_4hour.csv', 'PTA', 14.48),
        ('棉花_4hour.csv', '棉花', 12.83),
        ('沪锡_4hour.csv', '沪锡', -0.68),
        ('PVC_4hour.csv', 'PVC', -4.25),
        ('沪镍_4hour.csv', '沪镍', -6.29),
        ('白糖_4hour.csv', '白糖', -7.65),
    ]

    data_dir = Path('futures_data_4h')
    results = []

    for csv_file, future_name, original_return in top_futures:
        csv_path = data_dir / csv_file

        if not csv_path.exists():
            print(f"\n[跳过] {future_name} - 文件不存在")
            continue

        result = optimize_future(csv_path, future_name)

        if result:
            result['original_return'] = original_return
            result['improvement'] = result['return_pct'] - original_return
            results.append(result)

    if not results:
        print("\n无有效结果")
        return

    # 输出报告
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)

    print("\n" + "=" * 70)
    print("优化完成！对比报告")
    print("=" * 70)

    print(f"\n{'品种':<8} {'原始收益':>10} {'优化后收益':>12} {'提升':>10} {'交易数':>6} {'胜率':>8}")
    print("-" * 70)

    for _, row in results_df.iterrows():
        print(f"{row['name']:<8} {row['original_return']:>+9.2f}% {row['return_pct']:>+12.2f}% "
              f"{row['improvement']:>+10.2f}% {row['trades']:>6} {row['win_rate']:>7.1f}%")

    print("\n" + "=" * 70)
    print("推荐配置（优化后参数）")
    print("=" * 70)

    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"\n{i}. {row['name']} - {row['return_pct']:+.2f}% (提升{row['improvement']:+.2f}%)")
        print(f"   EMA({row['EMA_FAST']},{row['EMA_SLOW']}), RSI={row['RSI_FILTER']}, "
              f"RATIO={row['RATIO_TRIGGER']:.2f}, STC={row['STC_SELL_ZONE']}")

if __name__ == '__main__':
    main()
