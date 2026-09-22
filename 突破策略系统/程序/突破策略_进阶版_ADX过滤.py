"""
LME铜突破策略 - 进阶版（ADX过滤 + EMA确认）
====================================================
目标：在单品种上通过多层过滤器，大幅提升胜率和盈亏比

核心理念：
1. ADX过滤：只在趋势强劲时交易（避开震荡市）
2. EMA确认：只在多头趋势中做多（避开熊市反弹）
3. 大周期突破：捕捉铜的大级别趋势
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
    """计算所有技术指标"""

    df = df.copy()

    # 1. 唐奇安通道
    df['donchian_high'] = df['high'].rolling(window=params['donchian_period']).max()
    df['donchian_low'] = df['low'].rolling(window=params['donchian_period']).min()
    df['donchian_middle'] = (df['donchian_high'] + df['donchian_low']) / 2

    # 2. EMA趋势过滤器
    df['ema_filter'] = df['close'].ewm(span=params['ema_period'], adjust=False).mean()

    # 3. ADX趋势强度指标
    # True Range
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)

    # Directional Movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']

    df['pdm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['ndm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    # 平滑
    adx_window = 14
    df['tr_s'] = df['tr'].rolling(adx_window).mean()
    df['pdm_s'] = df['pdm'].rolling(adx_window).mean()
    df['ndm_s'] = df['ndm'].rolling(adx_window).mean()

    # DI & DX
    df['pdi'] = 100 * (df['pdm_s'] / df['tr_s'])
    df['ndi'] = 100 * (df['ndm_s'] / df['tr_s'])
    df['dx'] = 100 * abs(df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi'])

    # ADX
    df['adx'] = df['dx'].rolling(adx_window).mean()

    # 4. ATR (用于止损)
    df['atr'] = df['tr'].rolling(20).mean()

    # 5. 移动止损线
    df['stop_low'] = df['low'].rolling(window=params['stop_period']).min()

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
    ema_period = params['ema_period']
    adx_threshold = params['adx_threshold']
    atr_multiplier = params['atr_multiplier']

    # 需要足够的数据来计算所有指标
    min_idx = max(donchian_period, stop_period, ema_period, 50) + 1

    for i in range(min_idx, len(df) - 1):
        prev = df.iloc[i - 1]
        current = df.iloc[i]
        next_bar = df.iloc[i + 1]

        # 检查指标有效性
        if pd.isna(current['adx']) or pd.isna(current['ema_filter']) or pd.isna(current['atr']):
            continue

        # ===== 持仓管理 =====
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 1. 止损
            if current['low'] <= current['stop_low']:
                exit_price = min(current['stop_low'], current['open'])
                exit_triggered = True
                exit_reason = 'stop_loss'

            # 2. 趋势反转
            elif current['close'] < current['donchian_middle']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'trend_reversal'

            # 3. EMA破位（额外保护）
            elif current['close'] < current['ema_filter']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'ema_break'

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

        # ===== 开仓逻辑（多层过滤）=====
        if position is None:
            # 基础突破信号
            breakout_signal = current['close'] > prev['donchian_high']

            # ADX趋势强度过滤
            adx_filter = current['adx'] > adx_threshold

            # EMA趋势方向过滤
            ema_filter = current['close'] > current['ema_filter']

            # 三者同时满足才开仓
            if breakout_signal and adx_filter and ema_filter:
                entry_price = next_bar['open']
                contract_value = entry_price * CONTRACT_SIZE

                # 固定杠杆（激进因为有过滤）
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

    # 进阶评分函数
    # 优先考虑：高收益 + 高胜率 + 低回撤
    score = return_pct * 5  # 收益权重
    score += (win_rate - 40) * 20  # 胜率奖励（超过40%才有奖励）
    score += max_drawdown  # 回撤惩罚（负值相加）

    return {
        'total_trades': len(trades_df),
        'return_pct': return_pct,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'score': score
    }

# ============ 参数空间 ============
dimensions = [
    Integer(30, 100, name='donchian_period'),   # 突破周期（给大一点）
    Integer(10, 30, name='stop_period'),         # 止损周期
    Integer(60, 200, name='ema_period'),         # EMA过滤器（60-200日线）
    Integer(15, 35, name='adx_threshold'),       # ADX阈值（15-35）
    Real(1.5, 4.0, name='atr_multiplier')        # ATR止损倍数（宽止损）
]

# ============ 目标函数 ============
@use_named_args(dimensions=dimensions)
def objective(**params):
    result = backtest_with_params(df.copy(), params)

    # 惩罚项
    if result['total_trades'] < 10:
        return 10000 + (10 - result['total_trades']) * 100

    if result['total_trades'] > 200:
        return 10000 + (result['total_trades'] - 200) * 50

    if result['return_pct'] < 0:
        return 10000 + abs(result['return_pct']) * 10

    if result['max_drawdown'] < -40:
        return 10000 + abs(result['max_drawdown'] + 40) * 100

    if result['win_rate'] < 35:
        return 5000 + (35 - result['win_rate']) * 100

    return -result['score']

# ============ 主程序 ============
def main():
    print("\n" + "="*80)
    print("LME铜突破策略 - 进阶版优化（ADX + EMA双过滤）".center(80))
    print("="*80)
    print(f"\n数据量: {len(df)}根K线")
    print(f"时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}美元")

    print(f"\n优化规则:")
    print(f"  最大杠杆:        {MAX_LEVERAGE}倍")
    print(f"  成交方式:        次日开盘价")

    print(f"\n过滤器逻辑:")
    print(f"  1. 唐奇安通道突破  → 触发信号")
    print(f"  2. ADX > 阈值      → 确认趋势强度")
    print(f"  3. 价格 > EMA      → 确认多头方向")
    print(f"  三者同时满足才开仓")

    print(f"\n优化参数空间:")
    print(f"  唐奇安通道周期:  30-100日")
    print(f"  止损周期:        10-30日")
    print(f"  EMA周期:         60-200日")
    print(f"  ADX阈值:         15-35")
    print(f"  ATR止损倍数:     1.5-4.0")

    print("\n开始贝叶斯优化...")
    print("="*80)

    start_time = time.time()

    # 贝叶斯优化
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=50,  # 50次迭代
        n_random_starts=15,
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
        'ema_period': int(result.x[2]),
        'adx_threshold': int(result.x[3]),
        'atr_multiplier': round(result.x[4], 2)
    }

    # 用最优参数回测
    final_result = backtest_with_params(df.copy(), best_params)

    print("\n最优参数:")
    print("-"*80)
    print(f"  唐奇安通道周期:  {best_params['donchian_period']}日")
    print(f"  止损周期:        {best_params['stop_period']}日")
    print(f"  EMA周期:         {best_params['ema_period']}日")
    print(f"  ADX阈值:         {best_params['adx_threshold']}")
    print(f"  ATR止损倍数:     {best_params['atr_multiplier']}")

    print(f"\n回测表现:")
    print("-"*80)
    print(f"  总收益率:       {final_result['return_pct']:+.2f}%")
    print(f"  最大回撤:       {final_result['max_drawdown']:.2f}%")
    print(f"  交易次数:       {final_result['total_trades']}")
    print(f"  胜率:           {final_result['win_rate']:.1f}%")
    print(f"  综合评分:       {final_result['score']:.2f}")

    # 保存参数
    with open('../优化结果/突破策略_进阶版_最优参数.json', 'w', encoding='utf-8') as f:
        json.dump({
            'params': best_params,
            'result': final_result,
            'optimization_time': elapsed,
            'strategy': 'Donchian Breakout with ADX & EMA Filters'
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 参数已保存到: ../优化结果/突破策略_进阶版_最优参数.json")
    print("="*80)

if __name__ == '__main__':
    main()
