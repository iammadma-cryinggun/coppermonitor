# -*- coding: utf-8 -*-
"""
PTA策略鲁棒性验证
1. Walk-Forward滚动窗口测试
2. OOS（Out-of-Sample）样本外测试
3. 参数扰动鲁棒性测试
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime, timedelta
from itertools import product
import json

def calculate_indicators(df, params):
    """计算技术指标"""
    df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

    exp1 = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    exp2 = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    def calculate_stc(series, fast_period=23, slow_period=50, period=10):
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        stoch_k = 100 * (macd - macd.rolling(window=period).min()) / \
                  (macd.rolling(window=period).max() - macd.rolling(window=period).min())
        stoch_d = stoch_k.ewm(span=period, adjust=False).mean()
        stoch_k_d = 100 * (stoch_k - stoch_d.rolling(window=period).min()) / \
                    (stoch_d.rolling(window=period).max() - stoch_d.rolling(window=period).min())
        stc = stoch_k_d.ewm(span=period, adjust=False).mean()
        return stc

    df['stc'] = calculate_stc(df['close'])
    df['stc_prev'] = df['stc'].shift(1)

    return df

def backtest_long_only(df, params):
    """只做多回测（简化版）"""
    df = calculate_indicators(df.copy(), params)

    STOP_LOSS_PCT = 0.02
    INITIAL_CAPITAL = 50000
    CONTRACT_SIZE = 5
    MARGIN_RATE = 0.08

    capital = INITIAL_CAPITAL
    trades = []

    for i in range(50, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 检查做多信号
        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > params['RSI_FILTER']

        sniper_long = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
        chase_long = ema_cross and is_strong

        if sniper_long or chase_long:
            entry_price = current['close']
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)

            # 简化：固定1手
            contracts = 1

            # 向前找平仓点
            for j in range(i+1, min(i+50, len(df))):
                future = df.iloc[j]
                future_prev = df.iloc[j-1]

                if future['low'] <= stop_loss:
                    exit_price = stop_loss
                    break
                elif (future_prev['stc'] > params['STC_SELL_ZONE']) and (future['stc'] < future_prev['stc']):
                    exit_price = future['close']
                    break
                elif future['ema_fast'] < future['ema_slow']:
                    exit_price = future['close']
                    break
            else:
                exit_price = df.iloc[min(i+50, len(df)-1)]['close']

            pnl = (exit_price - entry_price) * CONTRACT_SIZE
            pnl_pct = (exit_price - entry_price) / entry_price * 100

            trades.append({
                'entry_index': i,
                'pnl_pct': pnl_pct
            })

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_pnl_pct = trades_df['pnl_pct'].sum()
    win_rate = (len(trades_df[trades_df['pnl_pct'] > 0]) / len(trades_df) * 100)

    return {
        'return_pct': total_pnl_pct,
        'trades': len(trades_df),
        'win_rate': win_rate
    }

def optimize_params(df_train, param_grid):
    """在训练集上优化参数"""
    best_result = None
    best_params = None

    for ema_fast, ema_slow, rsi, ratio, stc in product(
        param_grid['EMA_FAST'],
        param_grid['EMA_SLOW'],
        param_grid['RSI_FILTER'],
        param_grid['RATIO_TRIGGER'],
        param_grid['STC_SELL_ZONE']
    ):
        params = {
            'EMA_FAST': ema_fast,
            'EMA_SLOW': ema_slow,
            'RSI_FILTER': rsi,
            'RATIO_TRIGGER': ratio,
            'STC_SELL_ZONE': stc
        }

        try:
            result = backtest_long_only(df_train, params)

            if result is None:
                continue

            if best_result is None or result['return_pct'] > best_result['return_pct']:
                best_result = result
                best_params = params
        except:
            continue

    return best_params, best_result

def walk_forward_validation():
    """Walk-Forward滚动窗口验证"""

    print("=" * 120)
    print("PTA策略鲁棒性验证".center(120))
    print("=" * 120)

    # 获取数据
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('TA', days=365)  # 获取1年数据

    if df is None or df.empty:
        print("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"K线数量: {len(df)}")

    # ========== 测试1：Walk-Forward滚动窗口 ==========
    print(f"\n{'='*120}")
    print("测试1：Walk-Forward滚动窗口验证")
    print(f"{'='*120}")

    # 参数空间
    PARAM_GRID = {
        'EMA_FAST': [10, 12, 14],
        'EMA_SLOW': [8, 10, 12],
        'RSI_FILTER': [35, 40, 45],
        'RATIO_TRIGGER': [1.15, 1.25, 1.35],
        'STC_SELL_ZONE': [70, 75, 80]
    }

    # Walk-Forward窗口设置
    TRAIN_SIZE = 200  # 训练窗口约200根K线
    TEST_SIZE = 50    # 测试窗口50根K线
    STEP_SIZE = 50    # 滚动步长50根K线

    walk_forward_results = []

    total_windows = (len(df) - TRAIN_SIZE - TEST_SIZE) // STEP_SIZE
    print(f"\n窗口设置:")
    print(f"  训练窗口: {TRAIN_SIZE}根K线")
    print(f"  测试窗口: {TEST_SIZE}根K线")
    print(f"  滚动步长: {STEP_SIZE}根K线")
    print(f"  总窗口数: {total_windows}")

    for window_idx in range(total_windows):
        train_start = window_idx * STEP_SIZE
        train_end = train_start + TRAIN_SIZE
        test_start = train_end
        test_end = test_start + TEST_SIZE

        if test_end >= len(df):
            break

        df_train = df.iloc[train_start:train_end].copy()
        df_test = df.iloc[test_start:test_end].copy()

        # 训练集优化
        best_params, train_result = optimize_params(df_train, PARAM_GRID)

        if best_params is None:
            continue

        # 测试集验证
        test_result = backtest_long_only(df_test, best_params)

        if test_result is None:
            continue

        walk_forward_results.append({
            'window': window_idx + 1,
            'train_period': f"{df['datetime'].iloc[train_start]} ~ {df['datetime'].iloc[train_end]}",
            'test_period': f"{df['datetime'].iloc[test_start]} ~ {df['datetime'].iloc[test_end]}",
            'best_params': best_params,
            'train_return': train_result['return_pct'],
            'test_return': test_result['return_pct'],
            'train_win_rate': train_result['win_rate'],
            'test_win_rate': test_result['win_rate']
        })

        print(f"\n窗口{window_idx + 1}:")
        print(f"  训练期: {df['datetime'].iloc[train_start]} ~ {df['datetime'].iloc[train_end]}")
        print(f"  测试期: {df['datetime'].iloc[test_start]} ~ {df['datetime'].iloc[test_end]}")
        print(f"  最优参数: EMA({best_params['EMA_FAST']},{best_params['EMA_SLOW']}), RSI={best_params['RSI_FILTER']}, RATIO={best_params['RATIO_TRIGGER']}")
        print(f"  训练收益: {train_result['return_pct']:+.2f}%, 胜率{train_result['win_rate']:.1f}%")
        print(f"  测试收益: {test_result['return_pct']:+.2f}%, 胜率{test_result['win_rate']:.1f}%")

    # Walk-Forward统计
    if walk_forward_results:
        wf_df = pd.DataFrame(walk_forward_results)

        print(f"\n{'='*120}")
        print("Walk-Forward汇总统计")
        print(f"{'='*120}")

        print(f"\n总窗口数: {len(wf_df)}")
        print(f"训练平均收益: {wf_df['train_return'].mean():.2f}%")
        print(f"测试平均收益: {wf_df['test_return'].mean():.2f}%")
        print(f"测试胜率: {len(wf_df[wf_df['test_return'] > 0]) / len(wf_df) * 100:.1f}%")
        print(f"测试标准差: {wf_df['test_return'].std():.2f}%")

        # 参数稳定性
        print(f"\n参数稳定性:")
        print(f"  EMA_FAST: {wf_df['best_params'].apply(lambda x: x['EMA_FAST']).mean():.1f} ± {wf_df['best_params'].apply(lambda x: x['EMA_FAST']).std():.1f}")
        print(f"  EMA_SLOW: {wf_df['best_params'].apply(lambda x: x['EMA_SLOW']).mean():.1f} ± {wf_df['best_params'].apply(lambda x: x['EMA_SLOW']).std():.1f}")
        print(f"  RSI_FILTER: {wf_df['best_params'].apply(lambda x: x['RSI_FILTER']).mean():.1f} ± {wf_df['best_params'].apply(lambda x: x['RSI_FILTER']).std():.1f}")

    # ========== 测试2：OOS样本外测试 ==========
    print(f"\n{'='*120}")
    print("测试2：OOS（Out-of-Sample）样本外测试")
    print(f"{'='*120}")

    # 分割数据
    SPLIT_RATIO = 0.7
    split_idx = int(len(df) * SPLIT_RATIO)

    df_is = df.iloc[:split_idx].copy()  # In-Sample
    df_oos = df.iloc[split_idx:].copy()  # Out-of-Sample

    print(f"\n数据分割:")
    print(f"  IS（样本内）: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[split_idx-1]} ({len(df_is)}根K线)")
    print(f"  OOS（样本外）: {df['datetime'].iloc[split_idx]} ~ {df['datetime'].iloc[-1]} ({len(df_oos)}根K线)")

    # 在IS上优化
    best_params_is, is_result = optimize_params(df_is, PARAM_GRID)

    print(f"\nIS优化结果:")
    print(f"  最优参数: EMA({best_params_is['EMA_FAST']},{best_params_is['EMA_SLOW']}), RSI={best_params_is['RSI_FILTER']}")
    print(f"  IS收益: {is_result['return_pct']:+.2f}%, 胜率{is_result['win_rate']:.1f}%")

    # 在OOS上验证
    oos_result = backtest_long_only(df_oos, best_params_is)

    if oos_result:
        print(f"\nOOS验证结果:")
        print(f"  OOS收益: {oos_result['return_pct']:+.2f}%, 胜率{oos_result['win_rate']:.1f}%")

        # OOS/IS比率
        oos_is_ratio = oos_result['return_pct'] / is_result['return_pct'] if is_result['return_pct'] != 0 else 0
        print(f"\nOOS/IS比率: {oos_is_ratio:.2f}")
        if oos_is_ratio >= 0.5:
            print(f"  [OK] 策略在样本外表现良好（比率≥0.5）")
        else:
            print(f"  [WARNING] 策略在样本外表现较差（比率<0.5）")

    # ========== 测试3：参数扰动鲁棒性测试 ==========
    print(f"\n{'='*120}")
    print("测试3：参数扰动鲁棒性测试")
    print(f"{'='*120}")

    # 使用"已知"的最优参数
    base_params = {
        'EMA_FAST': 12,
        'EMA_SLOW': 10,
        'RSI_FILTER': 40,
        'RATIO_TRIGGER': 1.25,
        'STC_SELL_ZONE': 75
    }

    print(f"\n基准参数: EMA(12,10), RSI=40, RATIO=1.25, STC=75")

    # 扰动范围
    perturbations = [
        {'EMA_FAST': 10, 'EMA_SLOW': 10, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.25, 'STC_SELL_ZONE': 75},
        {'EMA_FAST': 14, 'EMA_SLOW': 10, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.25, 'STC_SELL_ZONE': 75},
        {'EMA_FAST': 12, 'EMA_SLOW': 8, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.25, 'STC_SELL_ZONE': 75},
        {'EMA_FAST': 12, 'EMA_SLOW': 12, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.25, 'STC_SELL_ZONE': 75},
        {'EMA_FAST': 12, 'EMA_SLOW': 10, 'RSI_FILTER': 35, 'RATIO_TRIGGER': 1.25, 'STC_SELL_ZONE': 75},
        {'EMA_FAST': 12, 'EMA_SLOW': 10, 'RSI_FILTER': 45, 'RATIO_TRIGGER': 1.25, 'STC_SELL_ZONE': 75},
        {'EMA_FAST': 12, 'EMA_SLOW': 10, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.15, 'STC_SELL_ZONE': 75},
        {'EMA_FAST': 12, 'EMA_SLOW': 10, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.35, 'STC_SELL_ZONE': 75},
        {'EMA_FAST': 12, 'EMA_SLOW': 10, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.25, 'STC_SELL_ZONE': 70},
        {'EMA_FAST': 12, 'EMA_SLOW': 10, 'RSI_FILTER': 40, 'RATIO_TRIGGER': 1.25, 'STC_SELL_ZONE': 80},
    ]

    base_result = backtest_long_only(df, base_params)

    print(f"\n基准结果: {base_result['return_pct']:+.2f}%, {base_result['trades']}笔, 胜率{base_result['win_rate']:.1f}%")

    print(f"\n{'扰动':<60} {'收益率':<10} {'变化':<10} {'胜率':<10}")
    print(f"{'-'*100}")

    robust_results = []

    for params in perturbations:
        result = backtest_long_only(df, params)

        if result:
            param_str = f"EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']}, RATIO={params['RATIO_TRIGGER']}, STC={params['STC_SELL_ZONE']}"
            change = result['return_pct'] - base_result['return_pct']

            print(f"{param_str:<60} {result['return_pct']:+.2f}%{' ':<6} {change:+.2f}%{' ':<6} {result['win_rate']:.1f}%")

            robust_results.append({
                'params': param_str,
                'return': result['return_pct'],
                'change': change,
                'win_rate': result['win_rate']
            })

    # 鲁棒性统计
    if robust_results:
        robust_df = pd.DataFrame(robust_results)

        avg_return = robust_df['return'].mean()
        std_return = robust_df['return'].std()
        min_return = robust_df['return'].min()
        max_return = robust_df['return'].max()

        print(f"\n鲁棒性统计:")
        print(f"  平均收益: {avg_return:.2f}%")
        print(f"  标准差: {std_return:.2f}%")
        print(f"  最小收益: {min_return:.2f}%")
        print(f"  最大收益: {max_return:.2f}%")
        print(f"  收益范围: {max_return - min_return:.2f}%")

        if std_return / abs(avg_return) < 0.5:
            print(f"\n  [OK] 策略对参数不敏感（变异系数<0.5）")
        else:
            print(f"\n  [WARNING] 策略对参数较敏感（变异系数≥0.5）")

    # ========== 总体评估 ==========
    print(f"\n{'='*120}")
    print("总体评估")
    print(f"{'='*120}")

    if walk_forward_results and oos_result:
        wf_test_avg = wf_df['test_return'].mean()
        wf_positive_ratio = len(wf_df[wf_df['test_return'] > 0]) / len(wf_df)

        score = 0
        max_score = 3

        print(f"\n评分:")

        # Walk-Forward评分
        if wf_test_avg > 0:
            print(f"  [✓] Walk-Forward平均收益为正: +{wf_test_avg:.2f}%")
            score += 1
        else:
            print(f"  [✗] Walk-Forward平均收益为负: {wf_test_avg:.2f}%")

        # OOS评分
        if oos_is_ratio >= 0.5:
            print(f"  [✓] OOS/IS比率合格: {oos_is_ratio:.2f} ≥ 0.5")
            score += 1
        else:
            print(f"  [✗] OOS/IS比率不合格: {oos_is_ratio:.2f} < 0.5")

        # 鲁棒性评分
        if robust_results:
            cv = std_return / abs(avg_return) if avg_return != 0 else 999
            if cv < 0.5:
                print(f"  [✓] 参数鲁棒性良好: 变异系数{cv:.2f} < 0.5")
                score += 1
            else:
                print(f"  [✗] 参数鲁棒性较差: 变异系数{cv:.2f} ≥ 0.5")

        print(f"\n总得分: {score}/{max_score}")

        if score == max_score:
            print(f"\n[优秀] 策略通过所有鲁棒性测试！")
        elif score >= max_score * 0.67:
            print(f"\n[良好] 策略通过大部分鲁棒性测试")
        elif score >= max_score * 0.33:
            print(f"\n[一般] 策略仅通过部分鲁棒性测试")
        else:
            print(f"\n[差] 策略未通过鲁棒性测试")

    # 保存结果
    output_dir = Path('logs/pta_robustness_test')
    output_dir.mkdir(exist_ok=True)

    if walk_forward_results:
        pd.DataFrame(walk_forward_results).to_csv(output_dir / 'walk_forward_results.csv', index=False, encoding='utf-8-sig')

    if robust_results:
        pd.DataFrame(robust_results).to_csv(output_dir / 'robustness_results.csv', index=False, encoding='utf-8-sig')

    print(f"\n结果已保存到: {output_dir}")

if __name__ == '__main__':
    walk_forward_validation()
