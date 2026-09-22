"""
铜(Cu) 实验：ICT 流动性掠夺策略 (Liquidity Sweep)
核心逻辑：
1. 趋势向上 (EMA Filter)
2. 价格跌破过去 N 天的最低点 (扫止损/获取流动性)
3. 当天收盘强势收回 (大户进场) -> 进场做多
验证：简化的ICT逻辑是否有效？
"""

import pandas as pd
import numpy as np
import pickle
from skopt import gp_minimize
from skopt.space import Integer, Real
import warnings

warnings.filterwarnings('ignore')

# ============ 1. 数据加载 ============
print("正在读取铜数据...")
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

# 获取铜数据
if '铜' in DATA_POOL:
    df_full = DATA_POOL['铜'].copy()
elif 'cu' in DATA_POOL:
    df_full = DATA_POOL['cu'].copy()
elif 'copper' in DATA_POOL:
    df_full = DATA_POOL['copper'].copy()
else:
    df_full = DATA_POOL[list(DATA_POOL.keys())[0]].copy()

df_full.sort_index(inplace=True)

# 划分 训练集(2016-2023) vs 测试集(2024-2026)
train_df = df_full.loc[:'2023-12-31'].copy()
test_df = df_full.loc['2024-01-01':].copy()

print(f"训练集: {len(train_df)}根K线 (2016-2023)")
print(f"测试集: {len(test_df)}根K线 (2024-2026)")

# ============ 2. 策略逻辑 (ICT 简化版) ============
def run_ict_strategy(df, params, return_details=False):
    """
    params:
    [0] ema_period: 趋势判断均线
    [1] sweep_lookback: 寻找过去多少天的低点作为流动性参考
    [2] stop_atr: 止损宽度
    [3] tp_atr: 止盈宽度
    """
    data = df.copy()

    # 1. 市场结构 (Market Structure)
    data['ema_trend'] = data['close'].ewm(span=int(params[0]), adjust=False).mean()

    # 2. 流动性参考点 (Liquidity Pool)
    data['prev_low_n'] = data['low'].rolling(int(params[1])).min().shift(1)

    # ATR 计算
    data['tr'] = np.maximum(data['high'] - data['low'],
                           np.abs(data['high'] - data['close'].shift(1)))
    data['atr'] = data['tr'].rolling(20).mean()

    capital = 500000
    balance = capital
    position = 0
    entry_price = 0
    contract_size = 5
    commission = 0.0001

    equity = []
    trades = []

    # 转换为 numpy 加速
    closes = data['close'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    ema_trend = data['ema_trend'].values
    prev_lows = data['prev_low_n'].values
    atrs = data['atr'].values

    stop_mult = params[2]
    tp_mult = params[3]

    min_idx = max(int(params[0]), int(params[1])) + 2

    for i in range(min_idx, len(data)-1):
        # --- 信号识别 ---

        # 1. 结构为多头
        is_uptrend = closes[i] > ema_trend[i]

        # 2. 发生了流动性掠夺 (Sweep)
        swept_liquidity = lows[i] < prev_lows[i]

        # 3. 强势拒绝/收回 (Rejection)
        strong_close = (closes[i] > opens[i]) and (closes[i] > prev_lows[i])

        signal_buy = is_uptrend and swept_liquidity and strong_close

        next_open = opens[i+1]

        # --- 持仓管理 ---
        if position > 0:
            sl_price = entry_price - (atrs[i] * stop_mult)
            tp_price = entry_price + (atrs[i] * tp_mult)

            if lows[i+1] < sl_price:
                exit_p = sl_price
                pnl = (exit_p - entry_price) * position * contract_size
                balance += pnl - (exit_p * position * contract_size * commission)
                trades.append(-1)
                position = 0
            elif highs[i+1] > tp_price:
                exit_p = tp_price
                pnl = (exit_p - entry_price) * position * contract_size
                balance += pnl - (exit_p * position * contract_size * commission)
                trades.append(1)
                position = 0
            elif closes[i+1] < ema_trend[i+1]:
                exit_p = closes[i+1]
                pnl = (exit_p - entry_price) * position * contract_size
                balance += pnl - (exit_p * position * contract_size * commission)
                trades.append(1 if pnl > 0 else -1)
                position = 0

        # --- 开仓管理 ---
        if position == 0 and signal_buy:
            risk = balance * 0.02
            dist = atrs[i] * stop_mult
            if dist > 0:
                qty = int(risk / (dist * contract_size))
                qty = max(1, min(qty, int(balance * 2 / (next_open * contract_size))))

                position = qty
                entry_price = next_open
                balance -= next_open * qty * contract_size * commission

        # 记录资金曲线
        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * contract_size
        equity.append(val)

    # 统计结果
    if not equity:
        return 0 if not return_details else {}

    final_ret = (equity[-1] - capital) / capital * 100

    if return_details:
        win_rate = trades.count(1) / len(trades) * 100 if trades else 0
        eq = np.array(equity)
        dd = (np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)
        return {'ret': final_ret, 'win': win_rate, 'trades': len(trades), 'dd': dd.max()*100}

    # 优化目标
    score = -final_ret
    if len(trades) < 15:
        score += 1000

    return score

# ============ 3. 贝叶斯优化 ============
print("\n" + "="*80)
print("ICT流动性掠夺策略 - 贝叶斯优化 (100次迭代)".center(80))
print("="*80)
print("\n正在验证 'ICT流动性掠夺' 逻辑 (训练集 2016-2023)...")

space = [
    Integer(60, 200, name='ema_trend'),
    Integer(5, 20, name='sweep_lookback'),
    Real(1.0, 3.0, name='stop_atr'),
    Real(2.0, 6.0, name='tp_atr')
]

res = gp_minimize(lambda p: run_ict_strategy(train_df, p),
                  space,
                  n_calls=100,
                  n_random_starts=20,
                  random_state=42,
                  verbose=False)

best_p = res.x
print("\n[OK] 优化完成！")
print("\n最优参数 (ICT 映射):")
print(f"  趋势定义 (EMA):  {int(best_p[0])} 日线")
print(f"  流动性池 (Lookback): 跌破过去 {int(best_p[1])} 日低点")
print(f"  止损 (SL):       {best_p[2]:.2f} ATR")
print(f"  止盈 (TP):       {best_p[3]:.2f} ATR (盈亏比 {best_p[3]/best_p[2]:.1f}:1)")

# ============ 4. 样本外验证 ============
print("\n" + "="*80)
print("真相时刻：样本外测试 (2024-2026)".center(80))
print("="*80)

test_res = run_ict_strategy(test_df, best_p, return_details=True)
train_res = run_ict_strategy(train_df, best_p, return_details=True)

print(f"\n{'指标':<12} | {'训练集 (拟合)':<18} | {'测试集 (真实)':<18} | {'状态':<15}")
print("-"*80)

# 收益率
ret_train = train_res['ret']
ret_test = test_res['ret']
status_ret = "[OK]" if ret_test > 0 else "[FAIL]"
print(f"{'总收益率':<12} | {ret_train:>16.2f}% | {ret_test:>16.2f}% | {status_ret}")

# 胜率
win_train = train_res['win']
win_test = test_res['win']
status_win = "[OK]" if win_test > 35 else "[WARN]"
print(f"{'胜率':<12} | {win_train:>16.1f}% | {win_test:>16.1f}% | {status_win}")

# 交易次数
trades_train = train_res['trades']
trades_test = test_res['trades']
print(f"{'交易次数':<12} | {trades_train:>16} | {trades_test:>16} | {'-'}")

# 回撤
dd_train = train_res['dd']
dd_test = test_res['dd']
status_dd = "[OK]" if dd_test < 20 else "[WARN]"
print(f"{'最大回撤':<12} | {dd_train:>16.1f}% | {dd_test:>16.1f}% | {status_dd}")

print("-"*80)

# 最终结论
print("\n" + "="*80)
if ret_test > 10 and win_test > 40:
    print("[SUCCESS] 验证成功！")
    print("           即使简化了，ICT 的'扫止损后进场'逻辑在铜上依然有效！")
    print("           这证明了：抓住'第一性原理'比复杂画线更重要。")
elif ret_test > 0 and win_test > 30:
    print("[OK] 验证部分成功")
    print("    简化版ICT在铜上能赚钱，但表现一般。")
    print("    说明核心逻辑有效，但需要进一步优化参数或增加过滤条件。")
else:
    print("[FAIL] 验证失败")
    print("       简化版丢失了太多信息，或者铜的流动性特征与该模型不符。")
    print("       ICT的'扫止损'逻辑在铜期货上可能不适用。")
print("="*80)
