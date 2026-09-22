"""
LME铜EMA+MACD+ADX策略 - 三重过滤版本
=========================================
结合方案A的ADX过滤 + 方案B的EMA+MACD逻辑

三层过滤器：
1. EMA金叉（趋势启动）
2. MACD的DIFF > 0（确认多头）
3. ADX > 阈值（确认趋势强度）

目标：在保持高收益的同时，提升胜率到50%+
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
    """计算 EMA + MACD + ADX"""
    data = df.copy()

    # 1. EMA 均线
    data['ema_fast'] = data['close'].ewm(span=int(p[0]), adjust=False).mean()
    data['ema_slow'] = data['close'].ewm(span=int(p[1]), adjust=False).mean()

    # 2. MACD 指标
    ema_m_fast = data['close'].ewm(span=int(p[2]), adjust=False).mean()
    ema_m_slow = data['close'].ewm(span=int(p[3]), adjust=False).mean()
    data['diff'] = ema_m_fast - ema_m_slow
    data['dea'] = data['diff'].ewm(span=int(p[4]), adjust=False).mean()
    data['macd_bar'] = (data['diff'] - data['dea']) * 2

    # 3. ADX 趋势强度指标
    data['h-l'] = data['high'] - data['low']
    data['h-pc'] = abs(data['high'] - data['close'].shift(1))
    data['l-pc'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['h-l', 'h-pc', 'l-pc']].max(axis=1)

    data['up_move'] = data['high'] - data['high'].shift(1)
    data['down_move'] = data['low'].shift(1) - data['low']

    data['pdm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
    data['ndm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)

    adx_window = 14
    data['tr_s'] = data['tr'].rolling(adx_window).mean()
    data['pdm_s'] = data['pdm'].rolling(adx_window).mean()
    data['ndm_s'] = data['ndm'].rolling(adx_window).mean()

    data['pdi'] = 100 * (data['pdm_s'] / data['tr_s'])
    data['ndi'] = 100 * (data['ndm_s'] / data['tr_s'])
    data['dx'] = 100 * abs(data['pdi'] - data['ndi']) / (data['pdi'] + data['ndi'])
    data['adx'] = data['dx'].rolling(adx_window).mean()

    # 4. ATR 用于止损
    data['atr'] = data['tr'].rolling(20).mean()

    return data

# ============ 回测函数 ============
def run_backtest(params):
    """
    params: [fast_ema, slow_ema, macd_fast, macd_slow, macd_sig, adx_threshold, stop_atr]
    """
    # 参数解包
    fast_period = int(params[0])
    slow_period = int(params[1])
    macd_fast = int(params[2])
    macd_slow = int(params[3])
    macd_sig = int(params[4])
    adx_threshold = params[5]
    stop_atr_mult = params[6]

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

    # 转换为 numpy 加速
    closes = data['close'].values
    opens = data['open'].values
    emas_f = data['ema_fast'].values
    emas_s = data['ema_slow'].values
    diffs = data['diff'].values
    adxs = data['adx'].values
    atrs = data['atr'].values

    for i in range(50, len(data)-1):
        # 信号逻辑（三层过滤）
        # 1. EMA金叉
        cross_up = (emas_f[i-1] < emas_s[i-1]) and (emas_f[i] > emas_s[i])
        cross_down = (emas_f[i-1] > emas_s[i-1]) and (emas_f[i] < emas_s[i])

        # 2. MACD过滤（DIFF > 0）
        macd_condition = diffs[i] > 0

        # 3. ADX过滤（趋势强度）- 新增！
        adx_condition = adxs[i] > adx_threshold

        next_open = opens[i+1]

        # --- 平仓逻辑 ---
        if position > 0:
            stop_price = entry_price - (atrs[i] * stop_atr_mult)
            is_stop = next_open < stop_price
            is_exit = cross_down

            if is_stop or is_exit:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * CONTRACT_SIZE
                balance += pnl
                trades.append(1 if pnl > 0 else -1)
                position = 0

        # --- 开仓逻辑（三层过滤）---
        if position == 0:
            # 必须同时满足：金叉 + MACD多头 + ADX强劲
            if cross_up and macd_condition and adx_condition:
                stop_dist = atrs[i] * stop_atr_mult
                if stop_dist > 0:
                    qty = max(1, min(int(balance * 0.02 / (stop_dist * CONTRACT_SIZE)),
                                   int(balance * MAX_LEVERAGE / (next_open * CONTRACT_SIZE))))
                    position = qty
                    entry_price = next_open

        # 记录净值
        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * CONTRACT_SIZE
        equity.append(val)

    # --- 评分系统 ---
    if not equity:
        return 10000

    ret = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # 胜率
    win_rate = 0
    if len(trades) > 0:
        win_rate = trades.count(1) / len(trades) * 100

    # 最大回撤
    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_dd = drawdown.min()

    # 评分：收益 + 胜率优先
    score = ret * 5  # 收益权重
    score += (win_rate - 45) * 15  # 胜率奖励（45%以上才有奖励）
    score += max_dd  # 回撤惩罚

    # 惩罚项
    if win_rate < 40:
        score += 1000
    if len(trades) < 10:
        score += 5000
    if max_dd < -35:
        score += 1000

    print(f"EMA({fast_period},{slow_period}) + MACD + ADX({adx_threshold:.0f}) | 收益: {ret:+.1f}% | 胜率: {win_rate:.1f}% | 交易: {len(trades)} | 回撤: {max_dd:.2f}%")

    return -score

# ============ 参数空间（增加ADX） ============
space = [
    Integer(5, 40, name='ema_fast'),       # 快EMA
    Integer(20, 100, name='ema_slow'),     # 慢EMA
    Integer(10, 20, name='macd_fast'),     # MACD快线
    Integer(20, 40, name='macd_slow'),     # MACD慢线
    Integer(5, 15, name='macd_sig'),       # MACD信号线
    Integer(15, 40, name='adx_threshold'), # ADX阈值（新增！）
    Real(1.5, 4.0, name='stop_atr')        # ATR止损
]

# ============ 包装目标函数 ============
@use_named_args(dimensions=space)
def objective(**params):
    return run_backtest([params['ema_fast'], params['ema_slow'],
                        params['macd_fast'], params['macd_slow'],
                        params['macd_sig'], params['adx_threshold'],
                        params['stop_atr']])

# ============ 主程序 ============
def main():
    print("\n" + "="*80)
    print("LME铜EMA+MACD+ADX策略 - 三重过滤版本".center(80))
    print("="*80)
    print(f"\n三层过滤器:")
    print(f"  1. EMA金叉 → 趋势启动")
    print(f"  2. MACD的DIFF > 0 → 确认多头")
    print(f"  3. ADX > 阈值 → 确认趋势强度（新增）")

    print(f"\n参数空间:")
    print(f"  EMA: 5-40 / 20-100日")
    print(f"  MACD: (10-20, 20-40, 5-15)")
    print(f"  ADX阈值: 15-40（新增）")
    print(f"  ATR止损: 1.5-4.0倍")

    print("\n开始贝叶斯优化（100次迭代）...")
    print("="*80)

    start_time = time.time()

    res = gp_minimize(
        objective,
        space,
        n_calls=100,  # 增加到100次
        n_random_starts=25,
        random_state=42,
        verbose=True
    )

    elapsed = time.time() - start_time

    print(f"\n优化完成! 耗时: {elapsed:.1f}秒")
    print("="*80)

    # 用最优参数重新回测
    best_params = [res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5], res.x[6]]

    data = calculate_indicators(df, best_params)

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades_list = []
    equity = []

    emas_f = data['ema_fast'].values
    emas_s = data['ema_slow'].values
    diffs = data['diff'].values
    adxs = data['adx'].values
    atrs = data['atr'].values
    closes = data['close'].values
    opens = data['open'].values

    for i in range(50, len(data)-1):
        cross_up = (emas_f[i-1] < emas_s[i-1]) and (emas_f[i] > emas_s[i])
        cross_down = (emas_f[i-1] > emas_s[i-1]) and (emas_f[i] < emas_s[i])
        macd_condition = diffs[i] > 0
        adx_condition = adxs[i] > best_params[5]

        next_open = opens[i+1]

        if position > 0:
            stop_price = entry_price - (atrs[i] * best_params[6])
            if next_open < stop_price or cross_down:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * CONTRACT_SIZE
                balance += pnl
                trades_list.append(pnl)
                position = 0

        if position == 0:
            if cross_up and macd_condition and adx_condition:
                stop_dist = atrs[i] * best_params[6]
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

    print("\n最优参数:")
    print("-"*80)
    print(f"  EMA快线: {int(res.x[0])}日")
    print(f"  EMA慢线: {int(res.x[1])}日")
    print(f"  MACD参数: ({int(res.x[2])}, {int(res.x[3])}, {int(res.x[4])})")
    print(f"  ADX阈值: {int(res.x[5])}")
    print(f"  ATR止损倍数: {res.x[6]:.2f}")

    print(f"\n回测表现:")
    print("-"*80)
    print(f"  总收益率: {final_ret:+.2f}%")
    print(f"  最大回撤: {max_dd:.2f}%")
    print(f"  交易次数: {len(trades_list)}")
    print(f"  胜率: {win_rate:.1f}%")

    # 对比之前的版本
    print(f"\n与之前版本对比:")
    print("-"*80)
    print(f"  200次迭代进阶突破: +33.09%, 胜率58.8%, 17笔")
    print(f"  EMA+MACD原版:    +93.46%, 胜率31.8%, 22笔")
    print(f"  EMA+MACD+ADX:     {final_ret:+.2f}%, 胜率{win_rate:.1f}%, {len(trades_list)}笔")

    # 保存
    with open('../优化结果/EMA_MACD_ADX三重过滤.json', 'w', encoding='utf-8') as f:
        json.dump({
            'params': {
                'ema_fast': int(res.x[0]),
                'ema_slow': int(res.x[1]),
                'macd_fast': int(res.x[2]),
                'macd_slow': int(res.x[3]),
                'macd_signal': int(res.x[4]),
                'adx_threshold': int(res.x[5]),
                'atr_multiplier': res.x[6]
            },
            'result': {
                'return_pct': final_ret,
                'max_drawdown': max_dd,
                'total_trades': len(trades_list),
                'win_rate': win_rate
            },
            'strategy': 'EMA+MACD+ADX Triple Filter'
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 结果已保存")
    print("="*80)

if __name__ == '__main__':
    main()
