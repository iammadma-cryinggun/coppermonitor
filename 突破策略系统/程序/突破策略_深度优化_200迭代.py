"""
LME铜突破策略 - 深度贝叶斯优化（200次迭代）
=============================================
这是进阶突破策略的完整优化版本

改进点：
1. 迭代次数从50次提升到200次
2. 扩大参数空间（更全面的搜索）
3. 优化收敛性监控
4. 自动保存中间结果
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver

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
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)

    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']

    df['pdm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['ndm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    adx_window = 14
    df['tr_s'] = df['tr'].rolling(adx_window).mean()
    df['pdm_s'] = df['pdm'].rolling(adx_window).mean()
    df['ndm_s'] = df['ndm'].rolling(adx_window).mean()

    df['pdi'] = 100 * (df['pdm_s'] / df['tr_s'])
    df['ndi'] = 100 * (df['ndm_s'] / df['tr_s'])
    df['dx'] = 100 * abs(df['pdi'] - df['ndi']) / (df['pdi'] + df['ndi'])
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

    min_idx = max(donchian_period, stop_period, ema_period, 50) + 1

    for i in range(min_idx, len(df) - 1):
        prev = df.iloc[i - 1]
        current = df.iloc[i]
        next_bar = df.iloc[i + 1]

        if pd.isna(current['adx']) or pd.isna(current['ema_filter']) or pd.isna(current['atr']):
            continue

        # 持仓管理
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if current['low'] <= current['stop_low']:
                exit_price = min(current['stop_low'], current['open'])
                exit_triggered = True
                exit_reason = 'stop_loss'

            elif current['close'] < current['donchian_middle']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'trend_reversal'

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

        # 开仓逻辑
        if position is None:
            breakout = current['close'] > prev['donchian_high']
            adx_ok = current['adx'] > adx_threshold
            ema_ok = current['close'] > current['ema_filter']

            if breakout and adx_ok and ema_ok:
                entry_price = next_bar['open']
                contract_value = entry_price * CONTRACT_SIZE

                contracts = int((capital * 1.5) / contract_value)
                if contracts < 1:
                    contracts = 1

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

    # 评分函数
    score = return_pct * 5  # 收益权重
    score += (win_rate - 40) * 20  # 胜率奖励
    score += max_drawdown  # 回撤惩罚

    return {
        'total_trades': len(trades_df),
        'return_pct': return_pct,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'score': score
    }

# ============ 扩大的参数空间 ============
dimensions = [
    Integer(20, 120, name='donchian_period'),   # 扩大到20-120日
    Integer(5, 40, name='stop_period'),          # 扩大到5-40日
    Integer(50, 250, name='ema_period'),         # 扩大到50-250日
    Integer(10, 40, name='adx_threshold'),       # 扩大到10-40
    Real(1.0, 5.0, name='atr_multiplier')        # 扩大到1.0-5.0
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
    print("LME铜突破策略 - 深度贝叶斯优化（200次迭代）".center(80))
    print("="*80)
    print(f"\n数据量: {len(df)}根K线")
    print(f"时间范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}美元")

    print(f"\n改进点:")
    print(f"  - 迭代次数: 50次 → 200次")
    print(f"  - 参数空间: 全面扩大")
    print(f"  - 自动保存: 每10次迭代保存一次")

    print(f"\n扩大后的参数空间:")
    print(f"  唐奇安通道周期:  20-120日（原30-100）")
    print(f"  止损周期:        5-40日（原10-30）")
    print(f"  EMA周期:         50-250日（原60-200）")
    print(f"  ADX阈值:         10-40（原15-35）")
    print(f"  ATR止损倍数:     1.0-5.0（原1.5-4.0）")

    print("\n开始深度优化（预计5-10分钟）...")
    print("="*80)

    start_time = time.time()

    # 设置检查点保存
    checkpoint_saver = CheckpointSaver("../优化结果/优化检查点.pkl")

    # 贝叶斯优化 - 200次迭代
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=200,  # 提升到200次
        n_random_starts=30,  # 随机起始点也增加到30
        random_state=42,
        verbose=True,
        callback=[checkpoint_saver]
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

    print("\n最优参数（深度优化）:")
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

    # 对比之前的50次迭代结果
    print(f"\n与50次迭代结果对比:")
    print("-"*80)
    print(f"  50次迭代: +51.63%, 胜率50.0%, 14笔交易")
    print(f"  200次迭代: {final_result['return_pct']:+.2f}%, 胜率{final_result['win_rate']:.1f}%, {final_result['total_trades']}笔交易")

    # 保存参数
    with open('../优化结果/突破策略_深度优化_200迭代.json', 'w', encoding='utf-8') as f:
        json.dump({
            'params': best_params,
            'result': final_result,
            'optimization_time': elapsed,
            'n_calls': 200,
            'strategy': 'Donchian Breakout with ADX & EMA - Deep Optimization'
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 参数已保存到: ../优化结果/突破策略_深度优化_200迭代.json")
    print("="*80)

if __name__ == '__main__':
    main()
