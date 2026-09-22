"""
LME铜EMA+MACD策略 - 深度优化（200次迭代）
===========================================
目标：找到EMA+MACD在铜期货上的最优参数

策略特点：
- 只做多（顺势而为）
- EMA金叉触发
- MACD的DIFF>0确认
- ATR动态止损
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

# ============ 计算指标 ============
def calculate_indicators(df, p):
    """计算 EMA + MACD"""
    data = df.copy()

    # EMA 均线
    data['ema_fast'] = data['close'].ewm(span=int(p[0]), adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=int(p[1]), adjust=False).mean()

    # MACD 指标
    ema_m_fast = data['close'].ewm(span=int(p[2]), adjust=False).mean()
    ema_m_slow = data['close'].ewm(span=int(p[3]), adjust=False).mean()
    data['diff'] = ema_m_fast - ema_m_slow
    data['dea'] = data['diff'].ewm(span=int(p[4]), adjust=False).mean()
    data['macd_bar'] = (data['diff'] - data['dea']) * 2

    # ATR 用于止损
    data['h-l'] = data['high'] - data['low']
    data['h-pc'] = abs(data['high'] - data['close'].shift(1))
    data['l-pc'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    data['atr'] = data['tr'].rolling(20).mean()

    return data

# ============ 回测函数 ============
def run_backtest(params):
    """
    params: [fast_ema, slow_ema, macd_fast, macd_slow, macd_sig, stop_atr]
    """
    fast_period = int(params[0])
    slow_period = int(params[1])
    macd_fast = int(params[2])
    macd_slow = int(params[3])
    macd_sig = int(params[4])
    stop_atr_mult = params[5]

    # 约束检查
    if fast_period >= slow_period:
        return 10000
    if macd_fast >= macd_slow:
        return 10000

    # 计算指标
    data = calculate_indicators(df, params)

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    equity = []

    # 转换为 numpy
    closes = data['close'].values
    opens = data['open'].values
    emas_f = data['ema_fast'].values
    emas_s = data['ema_slow'].values
    diffs = data['diff'].values
    atrs = data['atr'].values

    for i in range(50, len(data)-1):
        # 信号
        cross_up = (emas_f[i-1] < emas_s[i-1]) and (emas_f[i] > emas_s[i])
        cross_down = (emas_f[i-1] > emas_s[i-1]) and (emas_f[i] < emas_s[i])
        macd_condition = diffs[i] > 0

        next_open = opens[i+1]

        # 平仓
        if position > 0:
            stop_price = entry_price - (atrs[i] * stop_atr_mult)
            if next_open < stop_price or cross_down:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * CONTRACT_SIZE
                balance += pnl
                trades.append(1 if pnl > 0 else -1)
                position = 0

        # 开仓
        if position == 0:
            if cross_up and macd_condition:
                stop_dist = atrs[i] * stop_atr_mult
                if stop_dist > 0:
                    qty = max(1, min(int(balance * 0.02 / (stop_dist * CONTRACT_SIZE)),
                                   int(balance * MAX_LEVERAGE / (next_open * CONTRACT_SIZE))))
                    position = qty
                    entry_price = next_open

        # 净值
        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * CONTRACT_SIZE
        equity.append(val)

    # 统计
    if not equity:
        return 10000

    ret = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = trades.count(1) / len(trades) * 100 if trades else 0

    # 最大回撤
    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_dd = drawdown.min()

    # 评分：收益优先，兼顾胜率
    score = ret * 4  # 收益权重
    score += (win_rate - 30) * 10  # 胜率奖励（30%以上）
    score += max_dd  # 回撤惩罚

    # 惩罚
    if win_rate < 25:
        score += 2000
    if len(trades) < 10:
        score += 5000
    if max_dd < -35:
        score += 1000

    print(f"EMA({fast_period},{slow_period}) MACD | 收益:{ret:+.1f}% 胜率:{win_rate:.1f}% 交易:{len(trades)} 回撤:{max_dd:.2f}%")

    return -score

# ============ 参数空间 ============
space = [
    Integer(5, 40, name='ema_fast'),
    Integer(20, 100, name='ema_slow'),
    Integer(10, 20, name='macd_fast'),
    Integer(20, 40, name='macd_slow'),
    Integer(5, 15, name='macd_sig'),
    Real(1.5, 4.0, name='stop_atr')
]

# ============ 包装函数 ============
@use_named_args(dimensions=space)
def objective(**params):
    return run_backtest([params['ema_fast'], params['ema_slow'],
                        params['macd_fast'], params['macd_slow'],
                        params['macd_sig'], params['stop_atr']])

# ============ 主程序 ============
def main():
    print("\n" + "="*80)
    print("LME铜EMA+MACD策略 - 深度优化（200次迭代）".center(80))
    print("="*80)
    print(f"\n优化目标:")
    print(f"  - 最大化收益率")
    print(f"  - 保持胜率在30%以上")
    print(f"  - 控制回撤在35%以内")

    print(f"\n参数空间:")
    print(f"  EMA快线: 5-40日")
    print(f"  EMA慢线: 20-100日")
    print(f"  MACD: (10-20, 20-40, 5-15)")
    print(f"  ATR止损: 1.5-4.0倍")

    print("\n开始贝叶斯优化（200次迭代，预计3-5分钟）...")
    print("="*80)

    start_time = time.time()

    res = gp_minimize(
        objective,
        space,
        n_calls=200,
        n_random_starts=30,
        random_state=42,
        verbose=True
    )

    elapsed = time.time() - start_time

    print(f"\n优化完成! 耗时: {elapsed:.1f}秒")
    print("="*80)

    # 用最优参数重新回测
    best_params = [res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5]]

    data = calculate_indicators(df, best_params)

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades_list = []
    equity = []

    emas_f = data['ema_fast'].values
    emas_s = data['ema_slow'].values
    diffs = data['diff'].values
    atrs = data['atr'].values
    closes = data['close'].values
    opens = data['open'].values

    for i in range(50, len(data)-1):
        cross_up = (emas_f[i-1] < emas_s[i-1]) and (emas_f[i] > emas_s[i])
        cross_down = (emas_f[i-1] > emas_s[i-1]) and (emas_f[i] < emas_s[i])
        macd_condition = diffs[i] > 0

        next_open = opens[i+1]

        if position > 0:
            stop_price = entry_price - (atrs[i] * best_params[5])
            if next_open < stop_price or cross_down:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * CONTRACT_SIZE
                balance += pnl
                trades_list.append(pnl)
                position = 0

        if position == 0:
            if cross_up and macd_condition:
                stop_dist = atrs[i] * best_params[5]
                if stop_dist > 0:
                    qty = max(1, min(int(balance * 0.02 / (stop_dist * CONTRACT_SIZE)),
                                   int(balance * MAX_LEVERAGE / (next_open * CONTRACT_SIZE))))
                    position = qty
                    entry_price = next_open

        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * CONTRACT_SIZE
        equity.append(val)

    # 统计
    final_ret = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = len([t for t in trades_list if t > 0]) / len(trades_list) * 100 if trades_list else 0

    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_dd = drawdown.min()

    # 盈亏比
    winning_trades = [t for t in trades_list if t > 0]
    losing_trades = [t for t in trades_list if t < 0]
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else 0

    print("\n最优参数（200次迭代）:")
    print("-"*80)
    print(f"  EMA快线: {int(res.x[0])}日")
    print(f"  EMA慢线: {int(res.x[1])}日")
    print(f"  MACD参数: ({int(res.x[2])}, {int(res.x[3])}, {int(res.x[4])})")
    print(f"  ATR止损倍数: {res.x[5]:.2f}")

    print(f"\n回测表现:")
    print("-"*80)
    print(f"  总收益率: {final_ret:+.2f}%")
    print(f"  年化收益: {final_ret/10:.1f}%")
    print(f"  最大回撤: {max_dd:.2f}%")
    print(f"  交易次数: {len(trades_list)}")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  平均盈利: {avg_win:,.0f}")
    print(f"  平均亏损: {avg_loss:,.0f}")
    print(f"  盈亏比: {profit_factor:.2f}")

    # 与50次迭代对比
    print(f"\n与50次迭代对比:")
    print("-"*80)
    print(f"  50次迭代:  +93.46%, 胜率31.8%, 22笔交易")
    print(f"  200次迭代: {final_ret:+.2f}%, 胜率{win_rate:.1f}%, {len(trades_list)}笔交易")

    # 保存
    with open('../优化结果/EMA_MACD_深度优化_200迭代.json', 'w', encoding='utf-8') as f:
        json.dump({
            'params': {
                'ema_fast': int(res.x[0]),
                'ema_slow': int(res.x[1]),
                'macd_fast': int(res.x[2]),
                'macd_slow': int(res.x[3]),
                'macd_signal': int(res.x[4]),
                'atr_multiplier': res.x[5]
            },
            'result': {
                'return_pct': final_ret,
                'max_drawdown': max_dd,
                'total_trades': len(trades_list),
                'win_rate': win_rate,
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'profit_factor': float(profit_factor)
            },
            'optimization': '200 iterations'
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 结果已保存到: ../优化结果/EMA_MACD_深度优化_200迭代.json")
    print("="*80)

if __name__ == '__main__':
    main()
