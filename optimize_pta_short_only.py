# -*- coding: utf-8 -*-
"""
PTA做空参数优化 - 只优化做空部分
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime
from itertools import product
import time

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

def backtest_short_only(df, params):
    """
    只做空回测
    使用准确逻辑（止损按止损价平仓）
    """
    # 合约规格
    CONTRACT_SIZE = 5
    MARGIN_RATE = 0.08

    # 资金管理
    INITIAL_CAPITAL = 50000
    MAX_POSITION_RATIO = 0.80
    MAX_SINGLE_LOSS_PCT = 0.15
    STOP_LOSS_PCT = 0.02

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
                pnl = (position['entry_price'] - exit_price) * position['contracts'] * CONTRACT_SIZE
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
                contract_value = entry_price * CONTRACT_SIZE
                margin_per_contract = contract_value * MARGIN_RATE
                potential_loss_per_contract = (stop_loss - entry_price) * CONTRACT_SIZE

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
    return_pct = total_pnl / INITIAL_CAPITAL * 100
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    return {
        'return_pct': return_pct,
        'trades': total_trades,
        'win_rate': win_rate,
        'final_capital': capital,
        'trades_detail': trades_df
    }

def optimize_pta_short():
    """优化PTA做空参数"""

    print("=" * 100)
    print("PTA做空参数优化".center(100))
    print("=" * 100)

    # 获取数据
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('TA', days=300)

    if df is None or df.empty:
        print("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"K线数量: {len(df)}")

    # 参数空间（做空专用）
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
    print(f"\n参数空间: {total_combinations}种组合")

    best_result = None
    best_params = None
    all_results = []

    start_time = time.time()

    print(f"\n开始优化...")
    print(f"{'='*100}")

    for i, combination in enumerate(all_combinations, 1):
        params = dict(zip(param_names, combination))

        try:
            result = backtest_short_only(df.copy(), params)

            if result is None:
                continue

            # 记录所有结果
            all_results.append({
                'params': params.copy(),
                'return_pct': result['return_pct'],
                'trades': result['trades'],
                'win_rate': result['win_rate']
            })

            # 更新最佳结果
            if best_result is None or result['return_pct'] > best_result['return_pct']:
                best_result = result
                best_params = params.copy()

                if i % 100 == 0 or i == 1:
                    elapsed = time.time() - start_time
                    remaining = elapsed / i * (total_combinations - i) if i > 0 else 0

                    print(f"[{i}/{total_combinations}] 更好: {result['return_pct']:+.2f}% | "
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
        print("\n[失败] 无有效交易")
        return

    # 最终结果
    print(f"\n{'='*100}")
    print("优化完成")
    print(f"{'='*100}")
    print(f"\n耗时: {elapsed:.1f}秒")
    print(f"\n最佳参数:")
    print(f"  EMA_FAST: {best_params['EMA_FAST']}")
    print(f"  EMA_SLOW: {best_params['EMA_SLOW']}")
    print(f"  RSI_FILTER_SHORT: {best_params['RSI_FILTER_SHORT']}")
    print(f"  RATIO_TRIGGER_SHORT: {best_params['RATIO_TRIGGER_SHORT']:.2f}")
    print(f"  STC_BUY_ZONE: {best_params['STC_BUY_ZONE']}")

    print(f"\n最佳结果:")
    print(f"  收益率: {best_result['return_pct']:+.2f}%")
    print(f"  交易次数: {best_result['trades']}笔")
    print(f"  胜率: {best_result['win_rate']:.1f}%")
    print(f"  最终资金: {best_result['final_capital']:,.2f}元")

    # 详细交易记录
    print(f"\n{'='*100}")
    print("详细交易记录")
    print(f"{'='*100}")

    trades_df = best_result['trades_detail']

    print(f"\n{'-'*100}")
    print(f"{'序号':<5} {'信号':<15} {'入场时间':<19} {'入场价':<8} {'手数':<4} "
          f"{'出场时间':<19} {'出场价':<8} {'盈亏金额':<10} {'盈亏%':<8} {'原因':<10} {'资金后':<12}")
    print(f"{'-'*100}")

    for i, trade in enumerate(trades_df.to_dict('records'), 1):
        pnl_str = f"{'+' if trade['pnl'] >= 0 else ''}{trade['pnl']:.2f}"
        pnl_pct_str = f"{'+' if trade['pnl_pct'] >= 0 else ''}{trade['pnl_pct']:.2f}%"
        capital_str = f"{trade['capital_after']:,.0f}"

        print(f"{i:<5} {trade.get('signal_type', 'N/A'):<15} "
              f"{str(trade['entry_datetime']):<19} {trade['entry_price']:<8.2f} {trade['contracts']:<4} "
              f"{str(trade['exit_datetime']):<19} {trade['exit_price']:<8.2f} "
              f"{pnl_str:<10} {pnl_pct_str:<8} {trade['exit_reason']:<10} {capital_str:<12}")

    # 与镜像参数对比
    print(f"\n{'='*100}")
    print("与镜像参数对比")
    print(f"{'='*100}")

    print(f"\n{'参数':<25} {'镜像参数(原)':<20} {'优化后参数':<20}")
    print(f"{'-'*60}")
    print(f"{'EMA_FAST':<25} {'12':<20} {best_params['EMA_FAST']:<20}")
    print(f"{'EMA_SLOW':<25} {'10':<20} {best_params['EMA_SLOW']:<20}")
    print(f"{'RSI_FILTER_SHORT':<25} {'60':<20} {best_params['RSI_FILTER_SHORT']:<20}")
    print(f"{'RATIO_TRIGGER_SHORT':<25} {'1.25':<20} {best_params['RATIO_TRIGGER_SHORT']:<20}")
    print(f"{'STC_BUY_ZONE':<25} {'25':<20} {best_params['STC_BUY_ZONE']:<20}")

    print(f"\n{'结果':<25} {'镜像参数':<20} {'优化后':<20}")
    print(f"{'-'*60}")
    print(f"{'收益率':<25} {'-33.37%':<20} {best_result['return_pct']:+.2f}%")
    print(f"{'交易次数':<25} {'6笔':<20} {best_result['trades']}笔")
    print(f"{'胜率':<25} {'50.0%':<20} {best_result['win_rate']:.1f}%")

    improvement = best_result['return_pct'] - (-33.37)
    print(f"\n改进幅度: {improvement:+.2f}%")

    # 保存结果
    output_file = Path('logs/pta_short_optimization_result.csv')
    output_file.parent.mkdir(exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('return_pct', ascending=False)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    trades_output = Path('logs/pta_short_optimized_trades.csv')
    trades_df.to_csv(trades_output, index=False, encoding='utf-8-sig')

    print(f"\n优化结果已保存到: {output_file}")
    print(f"交易记录已保存到: {trades_output}")

    return best_params, best_result

if __name__ == '__main__':
    optimize_pta_short()
