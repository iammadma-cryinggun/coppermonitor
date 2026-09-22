"""
LME铜EMA+MACD策略 - 贝叶斯参数优化
====================================
灵感来源：加密货币EMA+MACD策略测试报告
- 单币种EMA+MACD夏普比率21.979，胜率66.87%
- MACD有效过滤假信号，提升胜率7.85pp

策略逻辑：
1. EMA金叉/死叉触发信号
2. MACD柱状图确认趋势方向
3. 2%固定止损
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
print("正在加载铜数据...")
with open('../原始数据/data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df = DATA_POOL['cu'].copy()
df.sort_index(inplace=True)
print(f"数据范围：{len(df)}根K线")
print(f"时间范围：{df.index[0]} ~ {df.index[-1]}")

# ============ 固定参数 ============
INITIAL_CAPITAL = 100000
CONTRACT_SIZE = 5
MAX_LEVERAGE = 2.0
COMMISSION_RATE = 0.0001

# ============ 计算指标 ============
def calculate_indicators(df, params):
    """计算EMA和MACD指标"""
    df = df.copy()

    # EMA
    df['ema_fast'] = df['close'].ewm(span=params['ema_fast'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['ema_slow'], adjust=False).mean()

    # MACD
    df['ema_12'] = df['close'].ewm(span=params['macd_fast'], adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=params['macd_slow'], adjust=False).mean()
    df['macd_line'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd_line'].ewm(span=params['macd_signal'], adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']

    # ATR（用于动态止损）
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

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

    ema_fast = params['ema_fast']
    ema_slow = params['ema_slow']
    stop_loss_pct = params['stop_loss'] / 100

    # 需要足够的数据来计算指标
    min_idx = max(ema_fast, ema_slow, params['macd_fast'], params['macd_slow'], 50) + 1

    for i in range(min_idx, len(df) - 1):
        prev = df.iloc[i - 1]
        current = df.iloc[i]
        next_bar = df.iloc[i + 1]

        # 检查指标有效性
        if pd.isna(current['ema_fast']) or pd.isna(current['ema_slow']) or pd.isna(current['macd_histogram']):
            continue

        # ===== 持仓管理 =====
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 1. 止损
            if position['side'] == 'long':
                stop_loss_price = position['entry_price'] * (1 - stop_loss_pct)
                if current['low'] <= stop_loss_price:
                    exit_price = min(stop_loss_price, current['open'])
                    exit_triggered = True
                    exit_reason = 'stop_loss'

                # EMA死叉（平多仓）
                elif current['ema_fast'] < current['ema_slow']:
                    exit_price = current['close']
                    exit_triggered = True
                    exit_reason = 'ema_cross_down'

            else:  # short
                stop_loss_price = position['entry_price'] * (1 + stop_loss_pct)
                if current['high'] >= stop_loss_price:
                    exit_price = max(stop_loss_price, current['open'])
                    exit_triggered = True
                    exit_reason = 'stop_loss'

                # EMA金叉（平空仓）
                elif current['ema_fast'] > current['ema_slow']:
                    exit_price = current['close']
                    exit_triggered = True
                    exit_reason = 'ema_cross_up'

            if exit_triggered:
                if position['side'] == 'long':
                    pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                else:
                    pnl = (position['entry_price'] - exit_price) * position['contracts'] * CONTRACT_SIZE

                capital += pnl
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': current.name,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'side': position['side'],
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': pnl / (capital - pnl) * 100 if capital > pnl else 0,
                    'exit_reason': exit_reason
                })
                position = None

        # ===== 开仓逻辑 =====
        if position is None:
            # 多头信号
            long_signal = (
                prev['ema_fast'] <= prev['ema_slow'] and
                current['ema_fast'] > current['ema_slow'] and  # EMA金叉
                current['macd_histogram'] > 0  # MACD柱状图>0确认
            )

            # 空头信号
            short_signal = (
                prev['ema_fast'] >= prev['ema_slow'] and
                current['ema_fast'] < current['ema_slow'] and  # EMA死叉
                current['macd_histogram'] < 0  # MACD柱状图<0确认
            )

            if long_signal or short_signal:
                side = 'long' if long_signal else 'short'
                entry_price = next_bar['open']
                contract_value = entry_price * CONTRACT_SIZE

                # 固定1.5倍杠杆
                contracts = int((capital * 1.5) / contract_value)
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
                    'side': side,
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
            'win_rate': 0,
            'score': -10000
        }

    trades_df = pd.DataFrame(trades)

    capital_series = pd.Series(capital_history)
    peak = capital_series.cummax()
    drawdown = (capital_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    return_pct = (capital_history[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100

    # 评分函数：优先考虑高胜率 + 高收益
    score = return_pct * 3  # 收益权重
    score += (win_rate - 50) * 15  # 胜率奖励（超过50%才有奖励）
    score += max_drawdown  # 回撤惩罚

    return {
        'total_trades': len(trades_df),
        'return_pct': return_pct,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'score': score
    }

# ============ 参数空间 ============
# 根据加密货币测试报告的参数范围，但需要适应铜期货特性
dimensions = [
    Integer(3, 10, name='ema_fast'),      # 快EMA周期（加密货币用5）
    Integer(10, 25, name='ema_slow'),     # 慢EMA周期（加密货币用15）
    Integer(10, 20, name='macd_fast'),    # MACD快线（标准12）
    Integer(20, 35, name='macd_slow'),    # MACD慢线（标准26）
    Integer(7, 12, name='macd_signal'),   # MACD信号线（标准9）
    Real(1.0, 3.0, name='stop_loss')      # 止损百分比（1%-3%）
]

# ============ 目标函数 ============
@use_named_args(dimensions=dimensions)
def objective(**params):
    result = backtest_with_params(df.copy(), params)

    # 惩罚项
    if result['total_trades'] < 20:
        return 10000 + (20 - result['total_trades']) * 100

    if result['total_trades'] > 200:
        return 10000 + (result['total_trades'] - 200) * 50

    if result['return_pct'] < 0:
        return 10000 + abs(result['return_pct']) * 10

    if result['max_drawdown'] < -30:
        return 10000 + abs(result['max_drawdown'] + 30) * 100

    if result['win_rate'] < 40:
        return 5000 + (40 - result['win_rate']) * 100

    return -result['score']

# ============ 主程序 ============
def main():
    print("\n" + "="*80)
    print("LME铜EMA+MACD策略 - 贝叶斯参数优化".center(80))
    print("="*80)
    print(f"\n数据量: {len(df)}根K线")
    print(f"时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}美元")

    print(f"\n策略逻辑:")
    print(f"  1. EMA金叉/死叉 → 触发信号")
    print(f"  2. MACD柱状图确认 → 过滤假信号")
    print(f"  3. 固定百分比止损")

    print(f"\n优化参数空间:")
    print(f"  EMA快线:         3-10日")
    print(f"  EMA慢线:         10-25日")
    print(f"  MACD快线:        10-20日")
    print(f"  MACD慢线:        20-35日")
    print(f"  MACD信号线:      7-12日")
    print(f"  止损百分比:      1%-3%")

    print("\n开始贝叶斯优化...")
    print("="*80)

    start_time = time.time()

    # 贝叶斯优化
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=80,  # 80次迭代
        n_random_starts=20,
        random_state=42,
        verbose=True
    )

    elapsed = time.time() - start_time

    print(f"\n优化完成! 耗时: {elapsed:.1f}秒")
    print("="*80)

    # 提取最优参数
    best_params = {
        'ema_fast': int(result.x[0]),
        'ema_slow': int(result.x[1]),
        'macd_fast': int(result.x[2]),
        'macd_slow': int(result.x[3]),
        'macd_signal': int(result.x[4]),
        'stop_loss': round(result.x[5], 2)
    }

    # 用最优参数回测
    final_result = backtest_with_params(df.copy(), best_params)

    print("\n最优参数:")
    print("-"*80)
    print(f"  EMA快线:         {best_params['ema_fast']}日")
    print(f"  EMA慢线:         {best_params['ema_slow']}日")
    print(f"  MACD参数:        ({best_params['macd_fast']}, {best_params['macd_slow']}, {best_params['macd_signal']})")
    print(f"  止损百分比:      {best_params['stop_loss']}%")

    print(f"\n回测表现:")
    print("-"*80)
    print(f"  总收益率:       {final_result['return_pct']:+.2f}%")
    print(f"  最大回撤:       {final_result['max_drawdown']:.2f}%")
    print(f"  交易次数:       {final_result['total_trades']}")
    print(f"  胜率:           {final_result['win_rate']:.1f}%")
    print(f"  综合评分:       {final_result['score']:.2f}")

    # 保存参数
    with open('../优化结果/EMA_MACD_最优参数.json', 'w', encoding='utf-8') as f:
        json.dump({
            'params': best_params,
            'result': final_result,
            'optimization_time': elapsed,
            'strategy': 'EMA+MACD Dual Filter Strategy'
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 参数已保存到: ../优化结果/EMA_MACD_最优参数.json")
    print("="*80)

if __name__ == '__main__':
    main()
