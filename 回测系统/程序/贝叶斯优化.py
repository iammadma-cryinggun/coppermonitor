# -*- coding: utf-8 -*-
"""
LME铜贝叶斯参数优化（修正版）

修复了以下严重bug：
1. 内层循环导致重复交易和时间穿越
2. 资金状况判断顺序错误
3. actual_leverage未更新问题
4. 保证金计算逻辑优化
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import time
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# 加载LME铜数据
with open('../原始数据/data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df = DATA_POOL['cu'].copy()

print("="*80)
print("LME铜日线数据 - 贝叶斯参数优化（修复bug版）")
print("="*80)
print(f"数据量: {len(df)}条")
print(f"时间范围: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"最新价格: {df['close'].iloc[-1]:.2f} 美元/吨")
print("="*80)

# 固定参数（真实交易规则）
INITIAL_CAPITAL = 100000
MARGIN_RATIO = 0.11  # 11%保证金
CONTRACT_SIZE = 5     # 5吨/手
MAX_LEVERAGE = 3.0    # 最大3倍杠杆
MAX_DRAWDOWN_PCT = 50.0
MIN_CAPITAL_PCT = 0.3
STOP_LOSS_PCT = 0.03  # 调整为3%以提高胜率

# 定义参数搜索空间（全部12个参数）
dimension = [
    Integer(3, 15, name='EMA_FAST'),
    Integer(10, 40, name='EMA_SLOW'),
    Integer(8, 16, name='MACD_FAST'),
    Integer(20, 35, name='MACD_SLOW'),
    Integer(5, 12, name='MACD_SIGNAL'),
    Integer(10, 20, name='RSI_PERIOD'),
    Integer(25, 50, name='RSI_FILTER'),
    Real(1.01, 1.30, name='RATIO_TRIGGER'),
    Integer(5, 15, name='STC_LENGTH'),
    Integer(15, 30, name='STC_FAST'),
    Integer(35, 65, name='STC_SLOW'),
    Integer(70, 90, name='STC_SELL_ZONE'),
]

print(f"\n真实交易规则:")
print(f"  保证金率: {MARGIN_RATIO*100}%")
print(f"  最大杠杆: {MAX_LEVERAGE}倍")
print(f"  初始资金: {INITIAL_CAPITAL:,}")


def calculate_indicators(df, params):
    """计算技术指标"""
    df = df.copy()

    df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

    exp1 = df['close'].ewm(span=params['MACD_FAST'], adjust=False).mean()
    exp2 = df['close'].ewm(span=params['MACD_SLOW'], adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=params['MACD_SIGNAL'], adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'].abs() > 1e-10, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/params['RSI_PERIOD'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/params['RSI_PERIOD'], adjust=False).mean()
    loss_safe = loss.where(loss.abs() > 1e-10, 1e-10)
    rs = gain / loss_safe
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

    stc_macd = df['close'].ewm(span=params['STC_FAST'], adjust=False).mean() - \
               df['close'].ewm(span=params['STC_SLOW'], adjust=False).mean()
    stoch_period = params['STC_LENGTH']
    min_macd = stc_macd.rolling(window=stoch_period).min()
    max_macd = stc_macd.rolling(window=stoch_period).max()
    macd_range = max_macd - min_macd
    stoch_k = 100 * (stc_macd - min_macd) / macd_range.where(macd_range.abs() > 1e-10, np.nan)
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=3).mean()
    min_stoch_d = stoch_d.rolling(window=stoch_period).min()
    max_stoch_d = stoch_d.rolling(window=stoch_period).max()
    stoch_range = max_stoch_d - min_stoch_d
    stc_raw = 100 * (stoch_d - min_stoch_d) / stoch_range.where(stoch_range.abs() > 1e-10, np.nan)
    stc_raw = stc_raw.fillna(50)
    df['stc'] = stc_raw.rolling(window=3).mean()

    return df


def backtest_with_params(df, params):
    """
    严格按真实规则回测（修复bug版）

    修复内容：
    1. 去除内层循环，改为逐bar判断
    2. 修正资金状况判断顺序
    3. 修正actual_leverage计算
    4. 优化保证金计算逻辑
    """
    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    peak_capital = INITIAL_CAPITAL

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 风险管理：检查是否达到停止交易条件
        current_drawdown = (capital - peak_capital) / peak_capital * 100

        if current_drawdown < -MAX_DRAWDOWN_PCT:
            if position is not None:
                # 强制平仓（用close价格，更合理）
                pnl = (current['close'] - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                capital += pnl
                trades.append({
                    'pnl': pnl,
                    'holding_days': (current.name - position['entry_datetime']).days,
                    'exit_reason': 'force_close_max_dd'
                })
                position = None
            break

        if capital < INITIAL_CAPITAL * MIN_CAPITAL_PCT:
            if position is not None:
                pnl = (current['close'] - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                capital += pnl
                trades.append({
                    'pnl': pnl,
                    'holding_days': (current.name - position['entry_datetime']).days,
                    'exit_reason': 'force_close_low_capital'
                })
                position = None
            break

        # ============ 修复1：持仓中逐bar判断卖出（去掉内层循环）============
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 止损检查
            if current['low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_reason = 'stop_loss'
                exit_triggered = True
            # STC止盈
            elif (prev['stc'] > params['STC_SELL_ZONE'] and
                  current['stc'] < prev['stc']):
                exit_price = current['close']
                exit_reason = 'stc'
                exit_triggered = True
            # 趋势反转
            elif current['ema_fast'] < current['ema_slow']:
                exit_price = current['close']
                exit_reason = 'trend'
                exit_triggered = True

            if exit_triggered:
                pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                capital += pnl

                if capital > peak_capital:
                    peak_capital = capital

                trades.append({
                    'pnl': pnl,
                    'holding_days': (current.name - position['entry_datetime']).days,
                    'exit_reason': exit_reason
                })
                position = None
                continue  # 平仓后当日不开新仓

        # ============ 无持仓：检查买入条件 =============
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
        if current['ratio'] > 0:
            if current['ratio'] > 2.0:
                base_position_size = 2.0
            elif current['ratio'] > 1.5:
                base_position_size = 1.5
            elif current['ratio'] > 1.0:
                base_position_size = 1.2
            else:
                base_position_size = 1.0
        else:
            base_position_size = 1.0

        # ============ 修复2：修正资金状况判断顺序 ============
        capital_ratio = capital / INITIAL_CAPITAL
        if capital_ratio < 0.3:  # 更严格的条件放前面
            base_position_size = 0
        elif capital_ratio < 0.5:
            base_position_size = base_position_size * 0.5

        # 限制最大杠杆
        position_size = min(base_position_size, MAX_LEVERAGE)

        stop_loss = current['close'] * (1 - STOP_LOSS_PCT)

        # 买入
        if buy_signal and position is None and position_size > 0:
            entry_price = current['close']
            contract_value = entry_price * CONTRACT_SIZE

            # 以损定仓法（真实期货交易逻辑）
            # 1. 确定每笔交易的风险比例
            # position_size作为风险放大系数：1.0x-2.0x
            # 基础风险2%，根据ratio调整到2%-4%
            base_risk_pct = 0.02  # 基础风险2%
            risk_pct = base_risk_pct * position_size  # 实际风险2%-4%

            # 2. 计算每笔最大可承受亏损
            max_loss_per_trade = capital * risk_pct

            # 3. 计算每手止损时的亏损
            loss_per_contract = entry_price * STOP_LOSS_PCT * CONTRACT_SIZE

            # 4. 计算可开手数
            contracts = int(max_loss_per_trade / loss_per_contract)

            if contracts <= 0:
                continue

            # 5. 计算实际杠杆
            total_notional = contracts * contract_value
            actual_leverage = total_notional / capital

            # 6. 验证杠杆不超标
            if actual_leverage > MAX_LEVERAGE:
                contracts = int((capital * MAX_LEVERAGE) / contract_value)
                if contracts <= 0:
                    continue
                total_notional = contracts * contract_value
                actual_leverage = total_notional / capital

            # 7. 验证保证金充足
            required_margin = total_notional * MARGIN_RATIO
            if required_margin > capital:
                contracts = int(capital / (contract_value * MARGIN_RATIO))
                if contracts <= 0:
                    continue
                total_notional = contracts * contract_value
                actual_leverage = total_notional / capital

            position = {
                'entry_datetime': current.name,
                'entry_price': entry_price,
                'contracts': contracts,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'entry_index': i,
                'leverage': actual_leverage
            }

            if capital > peak_capital:
                peak_capital = capital

    if not trades:
        return {
            'total_trades': 0,
            'return_pct': -100,
            'max_drawdown': -100,
            'score': -1000
        }

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df['pnl'].sum()
    return_pct = total_pnl / INITIAL_CAPITAL * 100

    # 计算最大回撤
    capital_curve = [INITIAL_CAPITAL]
    for _, trade in trades_df.iterrows():
        capital_curve.append(capital_curve[-1] + trade['pnl'])

    equity_series = pd.Series(capital_curve)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    # 计算胜率
    winning_trades = trades_df[trades_df['pnl'] > 0]
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    # 综合评分（胜率优先）
    # 胜率权重50%，收益率权重30%，回撤权重20%
    score = (win_rate * 0.5) + (return_pct * 0.3) - (abs(max_drawdown) * 0.2)

    return {
        'total_trades': len(trades_df),
        'return_pct': return_pct,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'score': score
    }


# 定义目标函数（优化胜率）
@use_named_args(dimensions=dimension)
def objective(**params):
    result = backtest_with_params(df.copy(), params)

    # 惩罚项
    if result['total_trades'] < 10:
        return 1000 + (10 - result['total_trades']) * 10

    if result['total_trades'] > 200:
        return 1000 + (result['total_trades'] - 200) * 10

    if result['return_pct'] < 0:
        return 1000 + abs(result['return_pct'])

    if result['max_drawdown'] < -60:
        return 1000 + abs(result['max_drawdown'] + 60)

    # 胜率惩罚：低于40%胜率额外惩罚
    if result['win_rate'] < 40:
        return 1000 + (40 - result['win_rate']) * 20

    return -result['score']


def main():
    """主函数"""
    print(f"\n风险管理设置:")
    print(f"  最大回撤限制: {MAX_DRAWDOWN_PCT}%")
    print(f"  最低资金要求: {MIN_CAPITAL_PCT*100}%初始资金")
    print(f"  固定止损: {STOP_LOSS_PCT*100}%")
    print(f"\n开始贝叶斯优化...")
    print("="*80)

    start_time = time.time()

    # 贝叶斯优化
    result = gp_minimize(
        func=objective,
        dimensions=dimension,
        n_calls=200,
        n_random_starts=20,
        random_state=42,
        verbose=True
    )

    elapsed = time.time() - start_time

    print(f"\n优化完成! 耗时: {elapsed:.1f}秒")
    print("="*80)

    # 提取最优参数
    best_params = {
        'EMA_FAST': int(result.x[0]),
        'EMA_SLOW': int(result.x[1]),
        'MACD_FAST': int(result.x[2]),
        'MACD_SLOW': int(result.x[3]),
        'MACD_SIGNAL': int(result.x[4]),
        'RSI_PERIOD': int(result.x[5]),
        'RSI_FILTER': int(result.x[6]),
        'RATIO_TRIGGER': round(result.x[7], 3),
        'STC_LENGTH': int(result.x[8]),
        'STC_FAST': int(result.x[9]),
        'STC_SLOW': int(result.x[10]),
        'STC_SELL_ZONE': int(result.x[11]),
    }

    # 用最优参数回测
    final_result = backtest_with_params(df.copy(), best_params)

    print("\n最优参数:")
    print("-"*80)
    print(f"  EMA_FAST:        {best_params['EMA_FAST']}")
    print(f"  EMA_SLOW:        {best_params['EMA_SLOW']}")
    print(f"  MACD_FAST:       {best_params['MACD_FAST']}")
    print(f"  MACD_SLOW:       {best_params['MACD_SLOW']}")
    print(f"  MACD_SIGNAL:     {best_params['MACD_SIGNAL']}")
    print(f"  RSI_PERIOD:      {best_params['RSI_PERIOD']}")
    print(f"  RSI_FILTER:      {best_params['RSI_FILTER']}")
    print(f"  RATIO_TRIGGER:   {best_params['RATIO_TRIGGER']:.3f}")
    print(f"  STC_LENGTH:      {best_params['STC_LENGTH']}")
    print(f"  STC_FAST:        {best_params['STC_FAST']}")
    print(f"  STC_SLOW:        {best_params['STC_SLOW']}")
    print(f"  STC_SELL_ZONE:   {best_params['STC_SELL_ZONE']}")

    print(f"\n回测表现:")
    print("-"*80)
    print(f"  总收益率:       {final_result['return_pct']:+.2f}%")
    print(f"  最大回撤:       {final_result['max_drawdown']:.2f}%")
    print(f"  交易次数:       {final_result['total_trades']}")
    print(f"  胜率:          {final_result['win_rate']:.1f}%")
    print(f"  综合评分:       {final_result['score']:.2f}")

    # 保存参数
    with open('../优化结果/cu_optimized_params.json', 'w', encoding='utf-8') as f:
        json.dump({
            'params': best_params,
            'result': final_result,
            'optimization_time': elapsed,
            'method': 'Bayesian Optimization - Bug Fixed Version',
            'margin_ratio': MARGIN_RATIO,
            'max_leverage': MAX_LEVERAGE,
            'bugs_fixed': [
                'Removed inner loop for exit (fixed time travel)',
                'Fixed capital ratio condition order',
                'Fixed actual_leverage calculation',
                'Optimized margin calculation logic'
            ]
        }, f, indent=2, ensure_ascii=False)

    print(f"\n最优参数已保存: cu_optimized_params_CORRECTED.json")
    print("="*80)

    return best_params, final_result


if __name__ == "__main__":
    main()
