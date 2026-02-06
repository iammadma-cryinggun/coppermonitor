# -*- coding: utf-8 -*-
"""
完全符合中国期货真实交易规则的回测系统
1. 真实合约单位
2. 真实保证金比例
3. 一次只持有一个仓位
4. 真实的仓位管理和风险控制
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import time
from datetime import datetime

# ========== 真实期货合约规格 ==========
# 根据各大交易所官网信息整理
FUTURES_SPEC = {
    # 上期所 (SHFE) - 保证金约7-8%
    '沪铜': {'contract_size': 5, 'margin_rate': 0.08, 'exchange': 'SHFE'},      # 5吨/手, 8%
    '沪铝': {'contract_size': 5, 'margin_rate': 0.08, 'exchange': 'SHFE'},      # 5吨/手, 8%
    '沪锌': {'contract_size': 5, 'margin_rate': 0.08, 'exchange': 'SHFE'},      # 5吨/手, 8%
    '沪镍': {'contract_size': 1, 'margin_rate': 0.12, 'exchange': 'SHFE'},      # 1吨/手, 12%
    '沪锡': {'contract_size': 1, 'margin_rate': 0.12, 'exchange': 'SHFE'},      # 1吨/手, 12%
    '沪铅': {'contract_size': 5, 'margin_rate': 0.08, 'exchange': 'SHFE'},      # 5吨/手, 8%
    '螺纹钢': {'contract_size': 10, 'margin_rate': 0.08, 'exchange': 'SHFE'},    # 10吨/手, 8%
    '热卷': {'contract_size': 10, 'margin_rate': 0.08, 'exchange': 'SHFE'},      # 10吨/手, 8%

    # 郑商所 (CZCE) - 保证金约7%
    'PTA': {'contract_size': 5, 'margin_rate': 0.08, 'exchange': 'CZCE'},        # 5吨/手, 8%
    '甲醇': {'contract_size': 50, 'margin_rate': 0.08, 'exchange': 'CZCE'},      # 50吨/手, 8%
    '白糖': {'contract_size': 10, 'margin_rate': 0.08, 'exchange': 'CZCE'},      # 10吨/手, 8%
    '棉花': {'contract_size': 5, 'margin_rate': 0.08, 'exchange': 'CZCE'},        # 5吨/手, 8%
    '纯碱': {'contract_size': 20, 'margin_rate': 0.08, 'exchange': 'CZCE'},      # 20吨/手, 8%
    '玻璃': {'contract_size': 20, 'margin_rate': 0.08, 'exchange': 'CZCE'},      # 20吨/手, 8%

    # 大商所 (DCE) - 保证金约8%
    '铁矿石': {'contract_size': 100, 'margin_rate': 0.08, 'exchange': 'DCE'},    # 100吨/手, 8%
    '焦炭': {'contract_size': 100, 'margin_rate': 0.08, 'exchange': 'DCE'},      # 100吨/手, 8%
    '焦煤': {'contract_size': 60, 'margin_rate': 0.08, 'exchange': 'DCE'},       # 60吨/手, 8%
    '豆油': {'contract_size': 10, 'margin_rate': 0.08, 'exchange': 'DCE'},        # 10吨/手, 8%
    '豆粕': {'contract_size': 10, 'margin_rate': 0.08, 'exchange': 'DCE'},        # 10吨/手, 8%
    '棕榈油': {'contract_size': 10, 'margin_rate': 0.08, 'exchange': 'DCE'},     # 10吨/手, 8%
    'PP': {'contract_size': 5, 'margin_rate': 0.08, 'exchange': 'DCE'},            # 5吨/手, 8%
    'PVC': {'contract_size': 5, 'margin_rate': 0.08, 'exchange': 'DCE'},          # 5吨/手, 8%
}

# 固定技术参数
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
MAX_POSITION_RATIO = 0.8  # 最大仓位使用率（可用资金的80%）
MAX_SINGLE_LOSS_PCT = 0.15  # 单笔最大亏损不超过资金的15%

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

def backtest_realistic(df, params, variety_name):
    """
    真实期货回测
    1. 使用真实合约单位和保证金比例
    2. 一次只持有一个仓位
    3. 严格的仓位管理
    """
    spec = FUTURES_SPEC.get(variety_name, {'contract_size': 5, 'margin_rate': 0.08})

    contract_size = spec['contract_size']
    margin_rate = spec['margin_rate']

    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    position = None  # 当前持仓（一次只持有一个）
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
                pnl = (exit_price - position['entry_price']) * position['contracts'] * contract_size
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

        # 检查开仓信号（只有在没有持仓时）
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

            # 计算合约价值和保证金
            contract_value = entry_price * contract_size
            margin_per_contract = contract_value * margin_rate

            # 计算止损价格
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)

            # 检查单笔最大亏损
            potential_loss_per_contract = (entry_price - stop_loss) * contract_size
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

def optimize_single_future(csv_file, variety_name):
    """优化单个品种（真实规则）"""
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 优化: {variety_name}")
    print(f"真实合约: {FUTURES_SPEC[variety_name]['contract_size']}吨/手, 保证金{FUTURES_SPEC[variety_name]['margin_rate']*100:.0f}%")
    print(f"{'='*80}")

    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"  数据: {len(df)}条")
    print(f"  时间: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 参数空间（保持一致）
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
    print(f"  参数空间: {total_combinations}种组合")

    best_result = None
    best_params = None

    start_time = time.time()

    for i, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))

        try:
            result = backtest_realistic(df.copy(), params, variety_name)

            if result is None:
                continue

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
        print(f"  [失败] 无有效交易")
        return None

    print(f"\n  [完成] 耗时: {elapsed:.1f}秒")
    print(f"  最佳参数: EMA({best_params['EMA_FAST']},{best_params['EMA_SLOW']}), RSI={best_params['RSI_FILTER']}, "
          f"RATIO={best_params['RATIO_TRIGGER']:.2f}, STC={best_params['STC_SELL_ZONE']}")
    print(f"  最佳结果: {best_result['return_pct']:+.2f}%, {best_result['trades']}笔, 胜率{best_result['win_rate']:.1f}%")

    return {
        'name': variety_name,
        'return_pct': best_result['return_pct'],
        'trades': best_result['trades'],
        'win_rate': best_result['win_rate'],
        'EMA_FAST': best_params['EMA_FAST'],
        'EMA_SLOW': best_params['EMA_SLOW'],
        'RSI_FILTER': best_params['RSI_FILTER'],
        'RATIO_TRIGGER': best_params['RATIO_TRIGGER'],
        'STC_SELL_ZONE': best_params['STC_SELL_ZONE'],
        'optimization_time': elapsed
    }

def main():
    print("=" * 80)
    print("真实期货规则重新优化所有品种")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\n真实规则说明：")
    print("  1. 使用真实合约单位和保证金比例")
    print("  2. 一次只持有一个仓位（不允许同时多仓）")
    print("  3. 严格仓位管理（最大仓位80%，单笔最大亏损15%）")
    print("  4. 参数空间：3125种组合")
    print("=" * 80)

    data_dir = Path('futures_data_4h')
    csv_files = sorted(data_dir.glob('*.csv'))

    results = []

    for i, csv_file in enumerate(csv_files, 1):
        future_name = csv_file.stem.replace('_4hour', '')

        if future_name not in FUTURES_SPEC:
            print(f"\n[跳过] {future_name} - 无合约规格信息")
            continue

        print(f"\n\n进度: [{i}/{len(csv_files)}]")

        try:
            result = optimize_single_future(csv_file, future_name)

            if result:
                results.append(result)

                # 实时保存进度
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('return_pct', ascending=False)

                results_dir = Path('optimization_results_realistic')
                results_dir.mkdir(exist_ok=True)
                results_df.to_csv(results_dir / 'progress.csv', index=False, encoding='utf-8-sig')

                print(f"\n  [进度保存] 当前已优化: {len(results)}/{len(csv_files)}")

        except Exception as e:
            print(f"\n  [错误] {future_name}: {e}")

    # 最终报告
    print("\n" + "=" * 80)
    print("优化完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    if not results:
        print("\n无有效结果")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)

    results_dir = Path('optimization_results_realistic')
    results_df.to_csv(results_dir / 'final_summary.csv', index=False, encoding='utf-8-sig')

    print(f"\n最终排名:")
    print(f"\n{'排名':<6} {'品种':<10} {'收益':>12} {'交易数':>6} {'胜率':>8} {'完整参数':<35}")
    print("-" * 90)

    for i, row in results_df.iterrows():
        params_str = f"EMA({row['EMA_FAST']},{row['EMA_SLOW']}), RSI={row['RSI_FILTER']}, RATIO={row['RATIO_TRIGGER']:.2f}, STC={row['STC_SELL_ZONE']}"
        print(f"{i:<6} {row['name']:<10} {row['return_pct']:>+12.2f}% {row['trades']:>6} {row['win_rate']:>7.1f}%  {params_str:<35}")

    print(f"\n所有结果已保存到: {results_dir}")

if __name__ == '__main__':
    main()
