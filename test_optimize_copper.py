# -*- coding: utf-8 -*-
"""
测试优化脚本准确性 - 强制优化沪铜
验证是否能找到与已验证参数一致的解
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import time

# ========== 参数搜索空间 ==========
# 与 optimize_rigorous.py 相同的参数空间
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

def verify_params(csv_file, params, run_count=3):
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
        return results[0]
    else:
        print(f"    警告: 3次结果不一致 - 收益率:{returns}, 交易数:{trades_counts}")
        return results[0]

def main():
    print("=" * 80)
    print("测试优化脚本准确性 - 强制优化沪铜")
    print("=" * 80)

    # 加载沪铜数据
    csv_file = Path('futures_data_4h/沪铜_4hour.csv')

    if not csv_file.exists():
        print(f"错误: 文件不存在 {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"\n数据: {len(df)}条")
    print(f"时间: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 已验证的铜参数
    verified_params = {
        'EMA_FAST': 5,
        'EMA_SLOW': 15,
        'RSI_FILTER': 45,
        'RATIO_TRIGGER': 1.15,
        'STC_SELL_ZONE': 85
    }

    print(f"\n{'='*80}")
    print("第一步: 验证已验证参数是否准确")
    print(f"{'='*80}")
    print(f"参数: EMA({verified_params['EMA_FAST']},{verified_params['EMA_SLOW']}), "
          f"RSI={verified_params['RSI_FILTER']}, RATIO={verified_params['RATIO_TRIGGER']:.2f}, "
          f"STC={verified_params['STC_SELL_ZONE']}")

    verified_result = verify_params(csv_file, verified_params, run_count=3)

    if verified_result:
        print(f"\n结果: {verified_result['return_pct']:+.2f}%, {verified_result['trades']}笔, "
              f"胜率{verified_result['win_rate']:.1f}%")

        if abs(verified_result['return_pct'] - 56.36) < 0.1:
            print("[OK] 与已验证结果一致 (+56.36%)")
        else:
            print(f"[ERROR] 与已验证结果不一致 (预期 +56.36%, 实际 {verified_result['return_pct']:+.2f}%)")
    else:
        print("[ERROR] 无法验证")
        return

    print(f"\n{'='*80}")
    print("第二步: 网格搜索寻找最优参数")
    print(f"{'='*80}")

    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    print(f"参数空间: {len(all_combinations)}种组合")

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

                print(f"  [{i}/{len(all_combinations)}] 发现更好: {result['return_pct']:+.2f}% | "
                      f"参数: EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']}, "
                      f"RATIO={params['RATIO_TRIGGER']:.2f}, STC={params['STC_SELL_ZONE']}")

        except Exception as e:
            print(f"  [{i}/{len(all_combinations)}] 错误: {e}")

    elapsed = time.time() - start_time

    if best_result is None:
        print(f"\n[失败] 无有效交易")
        return

    print(f"\n[完成] 耗时: {elapsed:.1f}秒")
    print(f"\n最佳参数: EMA({best_params['EMA_FAST']},{best_params['EMA_SLOW']}), "
          f"RSI={best_params['RSI_FILTER']}, RATIO={best_params['RATIO_TRIGGER']:.2f}, "
          f"STC={best_params['STC_SELL_ZONE']}")
    print(f"最佳结果: {best_result['return_pct']:+.2f}%, {best_result['trades']}笔, "
          f"胜率{best_result['win_rate']:.1f}%")

    print(f"\n{'='*80}")
    print("第三步: 对比分析")
    print(f"{'='*80}")

    print(f"\n已验证参数:")
    print(f"  EMA({verified_params['EMA_FAST']},{verified_params['EMA_SLOW']}), "
          f"RSI={verified_params['RSI_FILTER']}, RATIO={verified_params['RATIO_TRIGGER']:.2f}, "
          f"STC={verified_params['STC_SELL_ZONE']}")
    print(f"  收益: {verified_result['return_pct']:+.2f}%")

    print(f"\n优化后参数:")
    print(f"  EMA({best_params['EMA_FAST']},{best_params['EMA_SLOW']}), "
          f"RSI={best_params['RSI_FILTER']}, RATIO={best_params['RATIO_TRIGGER']:.2f}, "
          f"STC={best_params['STC_SELL_ZONE']}")
    print(f"  收益: {best_result['return_pct']:+.2f}%")

    improvement = best_result['return_pct'] - verified_result['return_pct']
    print(f"\n提升: {improvement:+.2f}%")

    if improvement > 1:
        print(f"\n[WARNING] 优化找到了更好的参数！")
        print(f"   这说明之前的优化脚本可能有问题")
        print(f"   或者已验证参数不是真正的最优")
    elif abs(improvement) < 0.01:
        print(f"\n[OK] 一致: 优化结果与已验证参数基本一致")
        print(f"   说明优化脚本逻辑正确")
    else:
        print(f"\n[INFO] 注意: 优化结果略差，可能是参数空间限制")

    # 查找已验证参数在所有结果中的排名
    all_results_df = pd.DataFrame(all_results)
    all_results_df = all_results_df.sort_values('return_pct', ascending=False)

    verified_rank = None
    for idx, row in all_results_df.iterrows():
        if (row['EMA_FAST'] == verified_params['EMA_FAST'] and
            row['EMA_SLOW'] == verified_params['EMA_SLOW'] and
            row['RSI_FILTER'] == verified_params['RSI_FILTER'] and
            row['RATIO_TRIGGER'] == verified_params['RATIO_TRIGGER'] and
            row['STC_SELL_ZONE'] == verified_params['STC_SELL_ZONE']):
            verified_rank = list(all_results_df.index).index(idx) + 1
            break

    if verified_rank:
        print(f"\n已验证参数在 {len(all_combinations)} 种组合中排名: 第 {verified_rank} 名")

    print(f"\nTop 5 参数组合:")
    for i in range(min(5, len(all_results_df))):
        row = all_results_df.iloc[i]
        print(f"  {i+1}. EMA({row['EMA_FAST']},{row['EMA_SLOW']}), RSI={row['RSI_FILTER']}, "
              f"RATIO={row['RATIO_TRIGGER']:.2f}, STC={row['STC_SELL_ZONE']} -> "
              f"{row['return_pct']:+.2f}%")

if __name__ == '__main__':
    main()
