"""
LME铜突破策略 - 贝叶斯参数优化
=====================================
激进版本：最大化收益，允许合理回撤
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# ============ 加载数据 ============
print("加载数据...")
with open('../原始数据/data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df = DATA_POOL['cu'].copy()
print(f"数据范围：{len(df)}根K线")
print(f"时间范围：{df.index[0]} ~ {df.index[-1]}")

# ============ 固定参数 ============
INITIAL_CAPITAL = 100000
MARGIN_RATIO = 0.11
CONTRACT_SIZE = 5
MAX_LEVERAGE = 3.0
MAX_DRAWDOWN_PCT = 30.0  # 放宽到30%
POSITION_SIZING = 'aggressive'

# ============ 定义参数空间 ============
# 我们优化4个关键参数
dimensions = [
    Integer(10, 30, name='donchian_period'),      # 唐奇安通道周期
    Integer(5, 15, name='stop_period'),            # 止损周期
    Integer(10, 20, name='atr_period'),            # ATR周期
    Real(1.0, 3.0, name='atr_multiplier')           # ATR倍数
]

# ============ 计算指标 ============
def calculate_indicators(df, params):
    """动态计算指标"""

    donchian_period = params['donchian_period']
    stop_period = params['stop_period']
    atr_period = params['atr_period']

    # 唐奇安通道
    df['donchian_high'] = df['high'].rolling(window=donchian_period).max()
    df['donchian_low'] = df['low'].rolling(window=donchian_period).min()
    df['donchian_middle'] = (df['donchian_high'] + df['donchian_low']) / 2

    # ATR
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=atr_period).mean()

    # 移动止损线
    df['stop_low'] = df['low'].rolling(window=stop_period).min()

    return df

# ============ 回测函数 ============
def backtest_with_params(df, params):
    """使用给定参数回测"""

    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    peak_capital = capital
    position = None

    trades = []
    capital_history = [capital]

    donchian_period = params['donchian_period']
    stop_period = params['stop_period']
    atr_multiplier = params['atr_multiplier']

    for i in range(max(donchian_period, stop_period) + 1, len(df) - 1):
        prev = df.iloc[i - 1]
        current = df.iloc[i]
        next_bar = df.iloc[i + 1]

        # 持仓管理
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 止损
            if current['low'] <= current['stop_low']:
                exit_price = min(current['stop_low'], current['open'])
                exit_triggered = True
                exit_reason = 'stop_loss'

            # 回撤风控
            current_drawdown = (peak_capital - capital) / peak_capital * 100
            if current_drawdown > MAX_DRAWDOWN_PCT:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'risk_control'

            # 趋势反转
            elif current['close'] < current['donchian_middle']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'trend_reversal'

            if exit_triggered:
                pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                capital += pnl
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': current.name,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': pnl / (capital - pnl) * 100 if capital > pnl else 0,
                    'exit_reason': exit_reason
                })
                position = None

        # 开仓逻辑
        if position is None:
            # 突破信号：收盘价突破前一根K线的唐奇安通道上轨
            if current['close'] > prev['donchian_high']:
                # 检查ATR是否有效
                if pd.isna(current['atr']) or current['atr'] <= 0:
                    continue

                entry_price = next_bar['open']
                contract_value = entry_price * CONTRACT_SIZE

                # 激进仓位
                risk_per_contract = current['atr'] * atr_multiplier * CONTRACT_SIZE
                risk_per_trade = capital * 0.02
                contracts = int(risk_per_trade / risk_per_contract)

                if contracts < 1:
                    contracts = 1

                # 杠杆验证
                total_notional = contracts * contract_value
                actual_leverage = total_notional / capital
                if actual_leverage > MAX_LEVERAGE:
                    contracts = int((capital * MAX_LEVERAGE) / contract_value)
                    if contracts < 1:
                        continue
                    total_notional = contracts * contract_value
                    actual_leverage = total_notional / capital

                position = {
                    'entry_date': next_bar.name,
                    'entry_price': entry_price,
                    'contracts': contracts,
                    'leverage': actual_leverage
                }

        capital_history.append(capital)
        if capital > peak_capital:
            peak_capital = capital

    # 统计结果
    if not trades:
        return {
            'total_trades': 0,
            'return_pct': 0,
            'max_drawdown': 0,
            'score': -10000
        }

    trades_df = pd.DataFrame(trades)

    capital_series = pd.Series(capital_history)
    peak = capital_series.cummax()
    drawdown = (capital_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    return_pct = (capital_history[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # 激进评分函数
    # 目标：最大化收益，只要回撤不超过30%
    if max_drawdown < -MAX_DRAWDOWN_PCT:
        score = return_pct  # 直接用收益率作为分数
    else:
        score = return_pct + (max_drawdown + MAX_DRAWDOWN_PCT) * 2  # 超过30%回撤则重罚

    return {
        'total_trades': len(trades_df),
        'return_pct': return_pct,
        'max_drawdown': max_drawdown,
        'score': score
    }

# ============ 目标函数 ============
@use_named_args(dimensions=dimensions)
def objective(**params):
    result = backtest_with_params(df.copy(), params)

    # 惩罚项
    if result['total_trades'] < 5:
        return 10000 + (5 - result['total_trades']) * 100

    if result['total_trades'] > 500:
        return 10000 + (result['total_trades'] - 500) * 100

    if result['return_pct'] < 0:
        return 10000 + abs(result['return_pct']) * 10

    if result['max_drawdown'] < -50:
        return 10000 + abs(result['max_drawdown'] + 50) * 100

    return -result['score']

# ============ 主程序 ============
def main():
    print("\n" + "="*80)
    print("LME铜突破策略 - 贝叶斯参数优化（激进版）".center(80))
    print("="*80)
    print(f"\n数据量: {len(df)}根K线")
    print(f"时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}美元")

    print(f"\n优化规则:")
    print(f"  最大回撤限制:  {MAX_DRAWDOWN_PCT}%")
    print(f"  最大杠杆:        {MAX_LEVERAGE}倍")
    print(f"  成交方式:        次日开盘价")

    print(f"\n优化参数空间:")
    print(f"  唐奇安通道周期:  10-30日")
    print(f"  止损周期:        5-15日")
    print(f"  ATR周期:         10-20日")
    print(f"  ATR倍数:         1.0-3.0")

    print("\n开始贝叶斯优化...")
    print("="*80)

    start_time = time.time()

    # 贝叶斯优化
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=100,  # 减少到100次迭代（激进版参数空间小）
        n_random_starts=20,
        random_state=42,
        verbose=True
    )

    elapsed = time.time() - start_time

    print(f"\n优化完成! 耗时: {elapsed:.1f}秒")
    print("="*80)

    # 提取最优参数
    best_params = {
        'donchian_period': int(result.x[0]),
        'stop_period': int(result.x[1]),
        'atr_period': int(result.x[2]),
        'atr_multiplier': round(result.x[3], 2)
    }

    # 用最优参数回测
    final_result = backtest_with_params(df.copy(), best_params)

    print("\n最优参数:")
    print("-"*80)
    print(f"  唐奇安通道周期:  {best_params['donchian_period']}日")
    print(f"  止损周期:        {best_params['stop_period']}日")
    print(f"  ATR周期:         {best_params['atr_period']}日")
    print(f"  ATR倍数:         {best_params['atr_multiplier']}")

    print(f"\n回测表现:")
    print("-"*80)
    print(f"  总收益率:       {final_result['return_pct']:+.2f}%")
    print(f"  最大回撤:       {final_result['max_drawdown']:.2f}%")
    print(f"  交易次数:       {final_result['total_trades']}")
    print(f"  综合评分:       {final_result['score']:.2f}")

    # 保存参数
    with open('../优化结果/突破策略_最优参数.json', 'w', encoding='utf-8') as f:
        json.dump({
            'params': best_params,
            'result': final_result,
            'optimization_time': elapsed,
            'strategy': 'Donchian Channel Breakout - Aggressive'
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 参数已保存到: ../优化结果/突破策略_最优参数.json")
    print("="*80)

if __name__ == '__main__':
    main()
