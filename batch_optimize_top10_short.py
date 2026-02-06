# -*- coding: utf-8 -*-
"""
批量优化TOP 10期货品种的做空参数
对每个品种进行参数网格搜索，找到最优做空参数
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime
from itertools import product
import time
import json

# TOP 10品种配置
TOP10_FUTURES_CONFIG = {
    'PTA': {
        'name': 'PTA',
        'code': 'TA',
        'exchange': 'CZCE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 10,
            'RSI_FILTER': 40,
            'RATIO_TRIGGER': 1.25,
            'STC_SELL_ZONE': 75,
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '沪镍': {
        'name': '沪镍',
        'code': 'NI',
        'exchange': 'SHFE',
        'long_params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 20,
            'RSI_FILTER': 55,
            'RATIO_TRIGGER': 1.10,
            'STC_SELL_ZONE': 75,
        },
        'contract_size': 1,
        'margin_rate': 0.12
    },
    '棕榈油': {
        'name': '棕榈油',
        'code': 'P',
        'exchange': 'DCE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 10,
            'RSI_FILTER': 45,
            'RATIO_TRIGGER': 1.20,
            'STC_SELL_ZONE': 75,
        },
        'contract_size': 10,
        'margin_rate': 0.08
    },
    '纯碱': {
        'name': '纯碱',
        'code': 'SA',
        'exchange': 'CZCE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 20,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
        },
        'contract_size': 20,
        'margin_rate': 0.08
    },
    'PVC': {
        'name': 'PVC',
        'code': 'V',
        'exchange': 'DCE',
        'long_params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 25,
            'RSI_FILTER': 55,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '沪铜': {
        'name': '沪铜',
        'code': 'CU',
        'exchange': 'SHFE',
        'long_params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 20,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '豆粕': {
        'name': '豆粕',
        'code': 'M',
        'exchange': 'DCE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 20,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.25,
            'STC_SELL_ZONE': 75,
        },
        'contract_size': 10,
        'margin_rate': 0.08
    },
    '沪锡': {
        'name': '沪锡',
        'code': 'SN',
        'exchange': 'SHFE',
        'long_params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 10,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.25,
            'STC_SELL_ZONE': 75,
        },
        'contract_size': 1,
        'margin_rate': 0.13
    },
    '沪铅': {
        'name': '沪铅',
        'code': 'PB',
        'exchange': 'SHFE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 10,
            'RSI_FILTER': 40,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '玻璃': {
        'name': '玻璃',
        'code': 'FG',
        'exchange': 'CZCE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 10,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.10,
            'STC_SELL_ZONE': 75,
        },
        'contract_size': 20,
        'margin_rate': 0.08
    }
}

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

    # STC指标
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

def backtest_short_only(df, params, contract_size, margin_rate):
    """
    只做空回测（简化版，固定1手，用于快速优化）
    优化完成后用完整版回测最优参数
    """
    df = calculate_indicators(df.copy(), params)

    STOP_LOSS_PCT = 0.02
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 做空信号
        trend_down = current['ema_fast'] < current['ema_slow']
        ratio_safe = (-params['RATIO_TRIGGER'] < current['ratio'] < 0)
        ratio_falling = current['ratio'] < prev['ratio']
        turning_down = current['macd_dif'] < prev['macd_dif']
        is_weak = current['rsi'] < params['RSI_FILTER']

        sniper_short = trend_down and ratio_safe and ratio_falling and turning_down and is_weak
        ema_death_cross = (prev['ema_fast'] >= prev['ema_slow']) and (current['ema_fast'] < current['ema_slow'])
        chase_short = ema_death_cross and is_weak

        if sniper_short or chase_short:
            entry_price = current['close']
            stop_loss = entry_price * (1 + STOP_LOSS_PCT)

            # 简化：固定1手
            contracts = 1

            # 向前找平仓点
            for j in range(i+1, min(i+50, len(df))):
                future = df.iloc[j]
                future_prev = df.iloc[j-1]

                # 止损
                if future['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = '止损'
                    break

                # STC止盈
                stc_exit = (future_prev['stc'] < params['STC_BUY_ZONE']) and (future['stc'] > future_prev['stc'])
                trend_exit = future['ema_fast'] > future['ema_slow']

                if stc_exit:
                    exit_price = future['close']
                    exit_reason = 'STC止盈'
                    break
                elif trend_exit:
                    exit_price = future['close']
                    exit_reason = '趋势反转'
                    break
            else:
                exit_price = df.iloc[min(i+50, len(df)-1)]['close']
                exit_reason = '超时'

            pnl = (entry_price - exit_price) * contract_size * contracts
            pnl_pct = (entry_price - exit_price) / entry_price * 100

            trades.append({
                'entry_index': i,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason
            })

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_pnl_pct = trades_df['pnl_pct'].sum()
    win_rate = (len(trades_df[trades_df['pnl_pct'] > 0]) / len(trades_df)) * 100

    return {
        'return_pct': total_pnl_pct,
        'trades': len(trades_df),
        'win_rate': win_rate
    }

def optimize_short_params(code, name, long_params):
    """优化单个品种的做空参数"""

    print(f"\n{'='*100}")
    print(f"优化 {name} ({code}) 做空参数")
    print(f"{'='*100}")

    # 获取数据
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data(code, days=300)

    if df is None or df.empty:
        print(f"  [失败] 数据获取失败")
        return None

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"  数据: {len(df)}条")
    print(f"  时间: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 做空参数空间（独立搜索，不依赖做多参数）
    PARAM_GRID = {
        'EMA_FAST': [3, 5, 7, 10, 12],      # 快线：3-12
        'EMA_SLOW': [10, 15, 20, 25, 30],   # 慢线：10-30
        'RSI_FILTER': [40, 45, 50, 55, 60, 65],  # 做空专用：RSI弱势阈值
        'RATIO_TRIGGER': [1.05, 1.10, 1.15, 1.20, 1.25],
        'STC_BUY_ZONE': [15, 20, 25, 30, 35]
    }

    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    # 过滤掉EMA_FAST >= EMA_SLOW的组合
    valid_combinations = []
    for combo in all_combinations:
        params = dict(zip(param_names, combo))
        if params['EMA_FAST'] < params['EMA_SLOW']:
            valid_combinations.append(combo)

    total_combinations = len(valid_combinations)
    print(f"  参数空间: {total_combinations}种组合（已过滤EMA_FAST<EMA_SLOW）")

    best_result = None
    best_params = None

    start_time = time.time()

    for i, combination in enumerate(valid_combinations, 1):
        params = dict(zip(param_names, combination))

        try:
            result = backtest_short_only(df, params, 1, 0.08)

            if result is None:
                continue

            if best_result is None or result['return_pct'] > best_result['return_pct']:
                best_result = result
                best_params = params.copy()

                if i % 100 == 0 or i == 1:
                    elapsed = time.time() - start_time
                    remaining = elapsed / i * (total_combinations - i) if i > 0 else 0

                    print(f"  [{i}/{total_combinations}] 更好: {result['return_pct']:+.2f}% | "
                          f"EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']}, "
                          f"RATIO={params['RATIO_TRIGGER']:.2f}, STC={params['STC_BUY_ZONE']} | "
                          f"{result['trades']}笔, 胜率{result['win_rate']:.1f}% | "
                          f"剩余: {remaining:.0f}秒")

        except Exception as e:
            pass

    elapsed = time.time() - start_time

    if best_result is None:
        print(f"  [失败] 无有效交易")
        return None

    print(f"\n  [完成] 耗时: {elapsed:.1f}秒")
    print(f"  最佳参数: EMA({best_params['EMA_FAST']},{best_params['EMA_SLOW']}), "
          f"RSI={best_params['RSI_FILTER']}, RATIO={best_params['RATIO_TRIGGER']:.2f}, STC_BUY={best_params['STC_BUY_ZONE']}")
    print(f"  最佳结果: {best_result['return_pct']:+.2f}%, {best_result['trades']}笔, 胜率{best_result['win_rate']:.1f}%")

    return {
        'code': code,
        'name': name,
        'short_params': best_params,
        'short_result': best_result
    }

def batch_optimize_all_short():
    """批量优化TOP 10品种的做空参数"""

    print("=" * 120)
    print("TOP 10期货品种做空参数批量优化".center(120))
    print("=" * 120)

    results = []

    start_time = time.time()

    for key, config in TOP10_FUTURES_CONFIG.items():
        print(f"\n\n处理 {key} ({config['name']})...")

        result = optimize_short_params(
            config['code'],
            config['name'],
            config['long_params']
        )

        if result:
            results.append(result)

            # 暂停一下，避免请求过快
            time.sleep(1)

    total_elapsed = time.time() - start_time

    # 汇总结果
    print(f"\n\n{'='*120}")
    print("优化完成汇总")
    print(f"{'='*120}")

    print(f"\n总耗时: {total_elapsed/60:.1f}分钟")

    print(f"\n{'品种':<10} {'代码':<6} {'EMA_FAST':<10} {'EMA_SLOW':<10} {'RSI_FILTER':<12} {'RATIO_TRIGGER':<14} {'STC_BUY_ZONE':<14} {'收益率':<12} {'交易':<8} {'胜率':<10}")
    print(f"{'-'*120}")

    for r in results:
        print(f"{r['name']:<10} {r['code']:<6} "
              f"{r['short_params']['EMA_FAST']:<10} "
              f"{r['short_params']['EMA_SLOW']:<10} "
              f"{r['short_params']['RSI_FILTER']:<12} "
              f"{r['short_params']['RATIO_TRIGGER']:<14.2f} "
              f"{r['short_params']['STC_BUY_ZONE']:<14} "
              f"{r['short_result']['return_pct']:+.2f}%{' ':<4} "
              f"{r['short_result']['trades']}笔{' ':<4} "
              f"{r['short_result']['win_rate']:.1f}%")

    # 保存结果
    output_file = Path('logs/top10_short_optimization.json')
    output_file.parent.mkdir(exist_ok=True)

    # 转换为可序列化的格式
    results_serializable = []
    for r in results:
        results_serializable.append({
            'code': r['code'],
            'name': r['name'],
            'short_params': r['short_params'],
            'short_result': r['short_result']
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_file}")

    # 生成Python配置代码
    config_code = f"# -*- coding: utf-8 -*-\n"
    config_code += f"# TOP 10期货品种做空优化参数（生成时间：{datetime.now()}）\n\n"
    config_code += f"TOP10_SHORT_PARAMS = {{\n"

    for r in results:
        params = r['short_params']
        config_code += f"    '{r['code']}': {{\n"
        config_code += f"        'name': '{r['name']}',\n"
        config_code += f"        'EMA_FAST': {params['EMA_FAST']},\n"
        config_code += f"        'EMA_SLOW': {params['EMA_SLOW']},\n"
        config_code += f"        'RSI_FILTER': {params['RSI_FILTER']},\n"
        config_code += f"        'RATIO_TRIGGER': {params['RATIO_TRIGGER']},\n"
        config_code += f"        'STC_BUY_ZONE': {params['STC_BUY_ZONE']},\n"
        config_code += f"        'return_pct': {r['short_result']['return_pct']},\n"
        config_code += f"        'trades': {r['short_result']['trades']},\n"
        config_code += f"        'win_rate': {r['short_result']['win_rate']}\n"
        config_code += f"    }},\n"

    config_code += "}\n"

    config_file = Path('logs/top10_short_params.py')
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_code)

    print(f"配置代码已保存到: {config_file}")

    return results

if __name__ == '__main__':
    batch_optimize_all_short()
