# -*- coding: utf-8 -*-
"""
===================================
策略参数网格优化 - 沪铜4小时K线
===================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import itertools
import json
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 加载数据
# ==========================================

def load_data(csv_path='沪铜4小时K线_1年.csv'):
    """加载4小时K线数据"""
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    logger.info(f"加载数据: {len(df)} 条, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    return df

# ==========================================
# 指标计算
# ==========================================

def calculate_indicators(df, params):
    """计算技术指标"""
    # EMA
    df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

    # MACD & Ratio
    exp1 = df['close'].ewm(span=params['MACD_FAST'], adjust=False).mean()
    exp2 = df['close'].ewm(span=params['MACD_SLOW'], adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=params['MACD_SIGNAL'], adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/params['RSI_PERIOD'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/params['RSI_PERIOD'], adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # STC
    stc_macd = df['close'].ewm(span=params['STC_FAST'], adjust=False).mean() - \
               df['close'].ewm(span=params['STC_SLOW'], adjust=False).mean()
    stoch_period = params['STC_LENGTH']
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

    # 前值
    df['ratio_prev'] = df['ratio'].shift(1)
    df['stc_prev'] = df['stc'].shift(1)

    return df

# ==========================================
# 信号检测与回测
# ==========================================

def run_backtest(df, params):
    """运行单次回测"""
    # 合并默认参数
    default_params = {
        'EMA_FAST': 5,
        'EMA_SLOW': 15,
        'MACD_FAST': 12,
        'MACD_SLOW': 26,
        'MACD_SIGNAL': 9,
        'RSI_PERIOD': 14,
        'RSI_FILTER': 45,
        'RATIO_TRIGGER': 1.15,
        'STC_LENGTH': 10,
        'STC_FAST': 23,
        'STC_SLOW': 50,
        'STC_SELL_ZONE': 85,
        'STOP_LOSS_PCT': 0.02
    }

    # 使用测试参数覆盖默认参数
    default_params.update(params)
    params = default_params

    # 计算指标
    df = calculate_indicators(df, params)

    # 初始资金
    INITIAL_CAPITAL = 100000
    MARGIN_RATIO = 0.15
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 趋势条件
        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > params['RSI_FILTER']
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong
        buy_signal = sniper_signal or chase_signal

        # 买入
        if buy_signal and position is None:
            entry_price = current['close']

            # 计算仓位
            ratio_prev_val = df['ratio_prev'].iloc[i]
            if ratio_prev_val > 0:
                if ratio_prev_val > 2.0:
                    position_size = 2.0
                elif ratio_prev_val > 1.5:
                    position_size = 1.5
                elif ratio_prev_val > 1.0:
                    position_size = 1.2
                else:
                    position_size = 1.0
            else:
                position_size = 1.0

            # 计算合约数量
            contract_value = entry_price * 5
            margin_per_contract = contract_value * MARGIN_RATIO
            available_margin = capital * position_size
            contracts = int(available_margin / margin_per_contract)

            if contracts > 0:
                position = {
                    'entry_datetime': current['datetime'],
                    'entry_price': entry_price,
                    'contracts': contracts,
                    'position_size': position_size,
                    'stop_loss': entry_price * (1 - params['STOP_LOSS_PCT']),
                    'entry_index': i
                }

        # 卖出
        elif position is not None:
            stc_prev_val = df['stc_prev'].iloc[i]
            stc_exit = (stc_prev_val > params['STC_SELL_ZONE'] and current['stc'] < stc_prev_val)
            trend_exit = current['ema_fast'] < current['ema_slow']
            stop_loss_hit = current['low'] <= position['stop_loss']

            if stc_exit or trend_exit or stop_loss_hit:
                exit_price = position['stop_loss'] if stop_loss_hit else current['close']
                exit_reason = 'stop_loss' if stop_loss_hit else ('stc' if stc_exit else 'trend')

                pnl = (exit_price - position['entry_price']) * position['contracts'] * 5
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'position_size': position['position_size'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': exit_reason,
                    'holding_bars': i - position['entry_index']
                })

                position = None

    # 统计结果
    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])

    total_pnl = trades_df['pnl'].sum()
    final_capital = INITIAL_CAPITAL + total_pnl
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    avg_holding = trades_df['holding_bars'].mean()

    # 最大回撤
    capital_curve = [INITIAL_CAPITAL]
    for _, trade in trades_df.iterrows():
        capital_curve.append(capital_curve[-1] + trade['pnl'])

    peak = capital_curve[0]
    max_drawdown = 0
    for val in capital_curve:
        if val > peak:
            peak = val
        drawdown = (peak - val) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'total_pnl': total_pnl,
        'final_capital': final_capital,
        'return_pct': total_pnl / INITIAL_CAPITAL * 100,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
        'avg_holding_bars': avg_holding,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe_ratio': (total_pnl / INITIAL_CAPITAL) / (max_drawdown + 0.01)  # 简化夏普比率
    }

# ==========================================
# 网格优化
# ==========================================

def grid_optimization():
    """网格优化参数"""

    # 加载数据
    df = load_data()

    # 定义参数网格
    param_grid = {
        # EMA周期
        'EMA_FAST': [3, 5, 7],
        'EMA_SLOW': [10, 15, 20],

        # MACD参数
        'MACD_FAST': [8, 12, 16],
        'MACD_SLOW': [20, 26, 32],
        'MACD_SIGNAL': [6, 9, 12],

        # RSI参数
        'RSI_PERIOD': [12, 14, 16],
        'RSI_FILTER': [40, 45, 50],

        # Ratio触发阈值
        'RATIO_TRIGGER': [1.10, 1.15, 1.20],

        # STC参数
        'STC_LENGTH': [8, 10, 12],
        'STC_FAST': [20, 23, 26],
        'STC_SLOW': [45, 50, 55],
        'STC_SELL_ZONE': [80, 85, 90],

        # 止损
        'STOP_LOSS_PCT': [0.015, 0.02, 0.025]
    }

    logger.info("="*80)
    logger.info("开始网格优化")
    logger.info("="*80)

    # 计算总组合数
    total_combinations = 1
    for key, values in param_grid.items():
        total_combinations *= len(values)

    logger.info(f"参数组合总数: {total_combinations:,}")

    # 当前最优参数（基准）
    current_params = {
        'EMA_FAST': 5,
        'EMA_SLOW': 15,
        'MACD_FAST': 12,
        'MACD_SLOW': 26,
        'MACD_SIGNAL': 9,
        'RSI_PERIOD': 14,
        'RSI_FILTER': 45,
        'RATIO_TRIGGER': 1.15,
        'STC_LENGTH': 10,
        'STC_FAST': 23,
        'STC_SLOW': 50,
        'STC_SELL_ZONE': 85,
        'STOP_LOSS_PCT': 0.02
    }

    logger.info("\n当前参数（基准）:")
    for k, v in current_params.items():
        logger.info(f"  {k}: {v}")

    # 运行基准回测
    logger.info("\n运行基准回测...")
    baseline_result = run_backtest(df.copy(), current_params)
    logger.info(f"基准结果: 收益 {baseline_result['return_pct']:.2f}%, 胜率 {baseline_result['win_rate']:.1f}%, "
               f"盈亏比 {baseline_result['profit_factor']:.2f}, 最大回撤 {baseline_result['max_drawdown_pct']:.2f}%")

    # 如果参数太多，只优化核心参数
    if total_combinations > 1000:
        logger.warning("\n参数组合过多，仅优化核心参数...")

        # 精简网格（只优化核心参数）
        param_grid = {
            'EMA_FAST': [3, 5, 7],
            'EMA_SLOW': [10, 15, 20],
            'RSI_FILTER': [40, 45, 50],
            'RATIO_TRIGGER': [1.10, 1.15, 1.20],
            'STOP_LOSS_PCT': [0.015, 0.02, 0.025]
        }

        total_combinations = 1
        for key, values in param_grid.items():
            total_combinations *= len(values)

        logger.info(f"精简后组合数: {total_combinations}")

    # 生成所有参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    logger.info(f"\n开始测试 {len(combinations)} 种参数组合...")

    # 测试所有组合
    results = []
    best_result = None
    best_score = -float('inf')

    for i, combination in enumerate(combinations, 1):
        params = dict(zip(keys, combination))

        # 运行回测
        result = run_backtest(df.copy(), params)

        if result:
            # 综合评分（收益/回撤/胜率）
            score = (
                result['return_pct'] * 0.4 +  # 收益权重40%
                (100 - result['max_drawdown_pct']) * 0.3 +  # 回撤控制30%
                result['win_rate'] * 0.2 +  # 胜率20%
                result['profit_factor'] * 10 * 0.1  # 盈亏比10%
            )

            result['score'] = score
            result['params'] = params
            results.append(result)

            # 更新最优结果
            if score > best_score:
                best_score = score
                best_result = result

        # 进度显示
        if i % 10 == 0 or i == len(combinations):
            logger.info(f"进度: {i}/{len(combinations)} ({i/len(combinations)*100:.1f}%)")

    # 排序结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    # 输出最优参数
    logger.info("\n" + "="*80)
    logger.info("优化完成")
    logger.info("="*80)

    logger.info(f"\n【基准参数】")
    logger.info(f"  收益: {baseline_result['return_pct']:.2f}%")
    logger.info(f"  胜率: {baseline_result['win_rate']:.1f}%")
    logger.info(f"  盈亏比: {baseline_result['profit_factor']:.2f}")
    logger.info(f"  最大回撤: {baseline_result['max_drawdown_pct']:.2f}%")
    logger.info(f"  夏普比率: {baseline_result['sharpe_ratio']:.2f}")

    logger.info(f"\n【最优参数】(综合评分: {best_result['score']:.2f})")
    logger.info(f"  收益: {best_result['return_pct']:.2f}%")
    logger.info(f"  胜率: {best_result['win_rate']:.1f}%")
    logger.info(f"  盈亏比: {best_result['profit_factor']:.2f}")
    logger.info(f"  最大回撤: {best_result['max_drawdown_pct']:.2f}%")
    logger.info(f"  夏普比率: {best_result['sharpe_ratio']:.2f}")

    logger.info(f"\n【参数】")
    for k, v in best_result['params'].items():
        logger.info(f"  {k}: {v}")

    # Top 10 参数组合
    logger.info(f"\n【Top 10 参数组合】")
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        logger.info(f"\n#{idx+1} (评分: {row['score']:.2f})")
        logger.info(f"  收益: {row['return_pct']:.2f}%, 胜率: {row['win_rate']:.1f}%, 盈亏比: {row['profit_factor']:.2f}, 回撤: {row['max_drawdown_pct']:.2f}%")
        for k, v in row['params'].items():
            logger.info(f"  {k}: {v}")

    # 保存结果
    output_file = 'optimization_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"\n优化结果已保存: {output_file}")

    # 保存最优参数
    best_params_file = 'best_params.json'
    with open(best_params_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': best_result['params'],
            'metrics': {
                'return_pct': best_result['return_pct'],
                'win_rate': best_result['win_rate'],
                'profit_factor': best_result['profit_factor'],
                'max_drawdown_pct': best_result['max_drawdown_pct'],
                'sharpe_ratio': best_result['sharpe_ratio'],
                'total_trades': best_result['total_trades']
            }
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"最优参数已保存: {best_params_file}")

    return best_result

if __name__ == "__main__":
    grid_optimization()
