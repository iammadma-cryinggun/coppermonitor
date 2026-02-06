# -*- coding: utf-8 -*-
"""
TOP10期货品种做空参数优化（正确版本）
基于optimize_pta_short_only.py的正确逻辑
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime
from itertools import product
import time
import json

# TOP10品种配置
TOP10_FUTURES_CONFIG = {
    'PTA': {
        'name': 'PTA',
        'code': 'TA',
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '沪镍': {
        'name': '沪镍',
        'code': 'NI',
        'contract_size': 1,
        'margin_rate': 0.12
    },
    '棕榈油': {
        'name': '棕榈油',
        'code': 'P',
        'contract_size': 10,
        'margin_rate': 0.08
    },
    '纯碱': {
        'name': '纯碱',
        'code': 'SA',
        'contract_size': 20,
        'margin_rate': 0.08
    },
    'PVC': {
        'name': 'PVC',
        'code': 'V',
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '沪铜': {
        'name': '沪铜',
        'code': 'CU',
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '豆粕': {
        'name': '豆粕',
        'code': 'M',
        'contract_size': 10,
        'margin_rate': 0.08
    },
    '沪锡': {
        'name': '沪锡',
        'code': 'SN',
        'contract_size': 1,
        'margin_rate': 0.13
    },
    '沪铅': {
        'name': '沪铅',
        'code': 'PB',
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '玻璃': {
        'name': '玻璃',
        'code': 'FG',
        'contract_size': 20,
        'margin_rate': 0.08
    }
}

# 固定参数
INITIAL_CAPITAL = 50000
MAX_POSITION_RATIO = 0.80
MAX_SINGLE_LOSS_PCT = 0.15
STOP_LOSS_PCT = 0.02

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
    做空回测 - 正确版本
    使用准确逻辑（止损按止损价平仓，收益用绝对金额计算）
    """
    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 持仓管理
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 止损检查（做空：高点触及止损价）
            if current['high'] >= position['stop_loss']:
                exit_price = position['stop_loss']  # 按止损价平仓
                exit_triggered = True
                exit_reason = '止损'

            # 平空仓信号检查
            if not exit_triggered:
                stc_exit = (prev['stc'] < params['STC_BUY_ZONE']) and (current['stc'] > prev['stc'])
                trend_exit = current['ema_fast'] > current['ema_slow']

                if stc_exit:
                    exit_price = current['close']
                    exit_triggered = True
                    exit_reason = 'STC止盈'
                elif trend_exit:
                    exit_price = current['close']
                    exit_triggered = True
                    exit_reason = '趋势反转'

            # 执行平仓
            if exit_triggered:
                pnl = (position['entry_price'] - exit_price) * position['contracts'] * contract_size
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'entry_price': position['entry_price'],
                    'exit_datetime': current['datetime'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / position['capital_at_entry']) * 100,
                    'capital_after': capital,
                    'exit_reason': exit_reason
                })

                position = None
                continue

        # 检查做空开仓信号
        if position is None:
            # 做空信号
            trend_down = current['ema_fast'] < current['ema_slow']
            ratio_safe_short = (-params['RATIO_TRIGGER_SHORT'] < current['ratio'] < 0)
            ratio_falling_short = current['ratio'] < prev['ratio']
            turning_down_short = current['macd_dif'] < prev['macd_dif']
            is_weak_short = current['rsi'] < params['RSI_FILTER_SHORT']

            sniper_short = trend_down and ratio_safe_short and ratio_falling_short and turning_down_short and is_weak_short

            ema_death_cross = (prev['ema_fast'] >= prev['ema_slow']) and (current['ema_fast'] < current['ema_slow'])
            chase_short = ema_death_cross and is_weak_short

            # 开空仓
            if sniper_short or chase_short:
                signal_type = 'sniper_short' if sniper_short else 'chase_short'
                entry_price = current['close']
                stop_loss = entry_price * (1 + STOP_LOSS_PCT)

                # 计算手数
                contract_value = entry_price * contract_size
                margin_per_contract = contract_value * margin_rate
                potential_loss_per_contract = (stop_loss - entry_price) * contract_size

                max_contracts_by_risk = int((capital * MAX_SINGLE_LOSS_PCT) / potential_loss_per_contract)
                max_contracts_by_margin = int((capital * MAX_POSITION_RATIO) / margin_per_contract)
                contracts = min(max_contracts_by_margin, max_contracts_by_risk)

                if contracts > 0:
                    position = {
                        'entry_datetime': current['datetime'],
                        'entry_price': entry_price,
                        'contracts': contracts,
                        'stop_loss': stop_loss,
                        'capital_at_entry': capital,
                        'signal_type': signal_type
                    }

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    total_pnl = trades_df['pnl'].sum()
    return_pct = total_pnl / INITIAL_CAPITAL * 100  # ✅ 正确：用绝对金额计算
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    return {
        'return_pct': return_pct,
        'trades': total_trades,
        'win_rate': win_rate,
        'final_capital': capital,
        'trades_detail': trades_df
    }

def optimize_single_short(code, name, contract_size, margin_rate):
    """优化单个品种做空参数"""

    print(f"\n{'='*100}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 优化: {name} ({code}) 做空")
    print(f"真实合约: {contract_size}吨/手, 保证金{margin_rate*100:.0f}%")
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

    # 参数空间（与PTA做空优化一致）
    PARAM_GRID = {
        'EMA_FAST': [3, 5, 7, 10, 12],
        'EMA_SLOW': [10, 15, 20, 25, 30],
        'RSI_FILTER_SHORT': [45, 50, 55, 60, 65, 70],  # 做空专用
        'RATIO_TRIGGER_SHORT': [1.05, 1.10, 1.15, 1.20, 1.25],
        'STC_BUY_ZONE': [15, 20, 25, 30, 35]  # 平空仓STC阈值
    }

    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combinations = list(product(*param_values))

    total_combinations = len(all_combinations)
    print(f"  参数空间: {total_combinations}种组合")

    best_result = None
    best_params = None

    start_time = time.time()

    for i, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))

        try:
            result = backtest_short_only(df.copy(), params, contract_size, margin_rate)

            if result is None:
                continue

            if best_result is None or result['return_pct'] > best_result['return_pct']:
                best_result = result
                best_params = params.copy()

                if i % 100 == 0 or i == 1:
                    elapsed = time.time() - start_time
                    remaining = elapsed / i * (total_combinations - i) if i > 0 else 0

                    print(f"  [{i}/{total_combinations}] 更好: {result['return_pct']:+.2f}% | "
                          f"EMA({params['EMA_FAST']},{params['EMA_SLOW']}), "
                          f"RSI={params['RSI_FILTER_SHORT']}, "
                          f"RATIO={params['RATIO_TRIGGER_SHORT']:.2f}, "
                          f"STC_BUY={params['STC_BUY_ZONE']} | "
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
          f"RSI={best_params['RSI_FILTER_SHORT']}, "
          f"RATIO={best_params['RATIO_TRIGGER_SHORT']:.2f}, "
          f"STC_BUY={best_params['STC_BUY_ZONE']}")
    print(f"  最佳结果: {best_result['return_pct']:+.2f}%, "
          f"{best_result['trades']}笔, "
          f"胜率{best_result['win_rate']:.1f}%, "
          f"最终资金{best_result['final_capital']:,.0f}元")

    return {
        'code': code,
        'name': name,
        'short_params': best_params,
        'short_result': best_result
    }

def batch_optimize_all_short():
    """批量优化TOP10做空参数"""

    print("=" * 140)
    print("TOP10期货品种做空参数批量优化（正确版本 - 基于PTA做空优化逻辑）".center(140))
    print("=" * 140)

    results = []
    start_time = time.time()

    for key, config in TOP10_FUTURES_CONFIG.items():
        result = optimize_single_short(
            config['code'],
            config['name'],
            config['contract_size'],
            config['margin_rate']
        )

        if result:
            results.append(result)
            time.sleep(1)

    total_elapsed = time.time() - start_time

    # 汇总结果
    print(f"\n\n{'='*140}")
    print("优化完成汇总")
    print(f"{'='*140}")

    print(f"\n总耗时: {total_elapsed/60:.1f}分钟")

    print(f"\n{'品种':<10} {'代码':<6} {'EMA_FAST':<10} {'EMA_SLOW':<10} {'RSI':<10} {'RATIO':<10} {'STC':<10} "
          f"{'收益率':<12} {'交易':<8} {'胜率':<10} {'最终资金':<15}")
    print(f"{'-'*140}")

    for r in results:
        print(f"{r['name']:<10} {r['code']:<6} "
              f"{r['short_params']['EMA_FAST']:<10} "
              f"{r['short_params']['EMA_SLOW']:<10} "
              f"{r['short_params']['RSI_FILTER_SHORT']:<10} "
              f"{r['short_params']['RATIO_TRIGGER_SHORT']:<10.2f} "
              f"{r['short_params']['STC_BUY_ZONE']:<10} "
              f"{r['short_result']['return_pct']:+.2f}%{' ':<4} "
              f"{r['short_result']['trades']}笔{' ':<4} "
              f"{r['short_result']['win_rate']:.1f}%{' ':<4} "
              f"{r['short_result']['final_capital']:>13,.0f}元")

    # 排名
    print(f"\n{'='*140}")
    print("收益率排名")
    print(f"{'='*140}")

    results_sorted = sorted(results, key=lambda x: x['short_result']['return_pct'], reverse=True)

    print(f"\n{'排名':<6} {'品种':<10} {'收益率':<15} {'交易':<8} {'胜率':<10}")
    print(f"{'-'*60}")

    for i, r in enumerate(results_sorted, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"{medal:<6} {r['name']:<10} {r['short_result']['return_pct']:+.2f}%{' ':<8} "
              f"{r['short_result']['trades']}笔{' ':<4} {r['short_result']['win_rate']:.1f}%")

    # 保存结果
    output_file = Path('logs/top10_short_optimization_correct.json')
    output_file.parent.mkdir(exist_ok=True)

    results_serializable = []
    for r in results:
        results_serializable.append({
            'code': r['code'],
            'name': r['name'],
            'short_params': r['short_params'],
            'short_result': {
                'return_pct': r['short_result']['return_pct'],
                'trades': r['short_result']['trades'],
                'win_rate': r['short_result']['win_rate'],
                'final_capital': r['short_result']['final_capital']
            }
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_file}")

    return results

if __name__ == '__main__':
    batch_optimize_all_short()
