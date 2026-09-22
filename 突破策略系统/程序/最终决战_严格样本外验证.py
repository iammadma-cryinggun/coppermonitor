"""
🏆 最终决战：进阶突破策略 - 严格样本外验证
=====================================
流程：
1. 将数据劈开：训练集(2016-2023) vs 测试集(2024-2026)
2. 在训练集上进行200次贝叶斯优化，寻找"抗震荡"参数
3. 将最优参数应用到测试集，看是否依然有效（还是过拟合）
"""

import pandas as pd
import numpy as np
import pickle
import time
from skopt import gp_minimize
from skopt.space import Integer, Real
import warnings

warnings.filterwarnings('ignore')

# ============ 1. 数据准备 ============
print("正在加载铜数据...")
with open('../原始数据/data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df_full = DATA_POOL['cu'].copy()
df_full.sort_index(inplace=True)

# --- 关键步骤：划分 训练集 vs 测试集 ---
train_df = df_full.loc[:'2023-12-31'].copy()
test_df = df_full.loc['2024-01-01':].copy()

print(f"训练集数据: {len(train_df)}条 (2016-2023)")
print(f"测试集数据: {len(test_df)}条 (2024-2026)")
print(f"测试集占比: {len(test_df)/(len(train_df)+len(test_df))*100:.1f}%")
print("-" * 60)

# ============ 2. 策略逻辑 (进阶突破+ADX+EMA) ============
def calculate_signals(df, p):
    """
    p = [donchian_period, exit_period, ema_period, adx_threshold, stop_atr]
    """
    data = df.copy()

    # 1. 唐奇安通道
    data['upper'] = data['high'].rolling(int(p[0])).max().shift(1)
    data['lower'] = data['low'].rolling(int(p[1])).min().shift(1)

    # 2. EMA 趋势过滤
    data['ema'] = data['close'].ewm(span=int(p[2]), adjust=False).mean()

    # 3. ADX 趋势强度
    data['tr'] = np.maximum(data['high'] - data['low'],
                        np.abs(data['high'] - data['close'].shift(1)))
    data['up'] = data['high'] - data['high'].shift(1)
    data['down'] = data['low'].shift(1) - data['low']
    data['pdm'] = np.where((data['up'] > data['down']) & (data['up'] > 0), data['up'], 0)
    data['ndm'] = np.where((data['down'] > data['up']) & (data['down'] > 0), data['down'], 0)

    window = 14
    data['tr_s'] = data['tr'].rolling(window).mean()
    data['pdm_s'] = data['pdm'].rolling(window).mean()
    data['ndm_s'] = data['ndm'].rolling(window).mean()

    # 避免除零错误
    data['pdm_s'] = np.where(data['pdm_s'] == 0, 0.0001, data['pdm_s'])
    data['ndm_s'] = np.where(data['ndm_s'] == 0, 0.0001, data['ndm_s'])
    data['tr_s'] = np.where(data['tr_s'] == 0, 0.0001, data['tr_s'])

    data['dx'] = 100 * abs(data['pdm_s']/data['tr_s'] - data['ndm_s']/data['tr_s']) / \
              (data['pdm_s']/data['tr_s'] + data['ndm_s']/data['tr_s'])
    data['adx'] = data['dx'].rolling(window).mean()

    # ATR
    data['atr'] = data['tr'].rolling(20).mean()

    return data

def run_backtest(df, params, return_details=False):
    data = calculate_signals(df, params)

    capital = 100000  # 10万美金
    balance = capital
    position = 0
    entry_price = 0
    contract_size = 5
    commission = 0.0001

    equity_curve = []
    trades = []

    # Numpy 加速
    closes = data['close'].values
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    uppers = data['upper'].values
    lowers = data['lower'].values
    emas = data['ema'].values
    adxs = data['adx'].values
    atrs = data['atr'].values

    donchian_p = params[0]
    adx_thres = params[3]
    stop_mult = params[4]

    min_idx = max(donchian_p, 60)

    for i in range(min_idx, len(data)-1):
        # 信号判定
        signal_buy = (highs[i] > uppers[i]) and \
                     (closes[i] > emas[i]) and \
                     (adxs[i] > adx_thres)

        signal_sell = lows[i] < lowers[i]

        next_open = opens[i+1]

        # 持仓处理
        if position > 0:
            stop_price = entry_price - (atrs[i] * stop_mult)
            is_stopped = next_open < stop_price

            if is_stopped or signal_sell:
                exit_p = next_open
                pnl = (exit_p - entry_price) * position * contract_size
                balance += pnl
                trades.append({'pnl': pnl, 'date': data.index[i+1]})
                position = 0

        # 开仓处理
        if position == 0 and signal_buy:
            risk_amt = balance * 0.02
            dist = atrs[i] * stop_mult
            if dist > 0:
                qty = int(risk_amt / (dist * contract_size))
                qty = max(1, min(qty, int(balance * 2.0 / (next_open * contract_size))))
                position = qty
                entry_price = next_open

        # 记录净值
        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * contract_size
        equity_curve.append(val)

    # 结果统计
    if not equity_curve:
        return -10000 if not return_details else {}

    final_ret = (equity_curve[-1] - capital) / capital * 100

    if return_details:
        wins = [t for t in trades if t['pnl'] > 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        # 最大回撤
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        max_dd = dd.max() * 100

        return {
            'return': final_ret,
            'win_rate': win_rate,
            'trades': len(trades),
            'max_dd': max_dd,
            'equity': equity_curve
        }

    # 优化目标函数
    score = -final_ret

    win_rate = 0
    if len(trades) > 0:
        win_rate = len([t for t in trades if t['pnl'] > 0]) / len(trades) * 100

    if win_rate < 40:
        score += 1000
    if len(trades) < 10:
        score += 5000

    return score

# ============ 3. 执行贝叶斯优化 (仅在训练集) ============
print("\n" + "="*80)
print("最终决战：严格样本外验证".center(80))
print("="*80)
print(f"\n[START] 开始在训练集 (2016-2023) 上寻找最强参数...")
print(f"目标：高收益 + 胜率>40% + 抵抗震荡")

space = [
    Integer(20, 100, name='donchian'),
    Integer(10, 50, name='exit'),
    Integer(60, 200, name='ema'),
    Integer(15, 35, name='adx'),
    Real(1.5, 4.0, name='atr_stop')
]

print("\n参数空间:")
print(f"  唐奇安周期:  20-100日")
print(f"  离场周期:    10-50日")
print(f"  EMA周期:     60-200日")
print(f"  ADX阈值:     15-35")
print(f"  ATR止损:     1.5-4.0倍")

print("\n开始贝叶斯优化（200次迭代，预计5-8分钟）...")
print("-"*80)

start_time = time.time()

res = gp_minimize(
    lambda p: run_backtest(train_df, p),
    space,
    n_calls=200,
    n_random_starts=30,
    random_state=42,
    verbose=False
)

elapsed = time.time() - start_time

best_params = res.x
print(f"\n优化完成! 耗时: {elapsed:.1f}秒")
print("-"*80)

print("\n[OK] 训练完成！找到的最优参数：")
print("-"*80)
print(f"  唐奇安周期: {int(best_params[0])} 日")
print(f"  离场周期:   {int(best_params[1])} 日")
print(f"  EMA均线:    {int(best_params[2])} 日")
print(f"  ADX阈值:    {int(best_params[3])}")
print(f"  ATR倍数:    {best_params[4]:.2f}")

# ============ 4. 最终验证 ============
print("\n[TEST] 最终对比测试开始...")
print("="*80)

# 跑训练集
res_train = run_backtest(train_df, best_params, return_details=True)
# 跑测试集
res_test = run_backtest(test_df, best_params, return_details=True)

print(f"\n{'='*80}")
print(f"策略：进阶突破 (唐奇安{int(best_params[0])}日 + ADX>{int(best_params[3])} + EMA{int(best_params[2])}日)")
print(f"{'='*80}")
print(f"{'指标':<12} | {'训练集 (2016-2023)':<20} | {'测试集 (2024-2026)':<20} | {'状态':<12}")
print("-"*80)

# 收益率
ret_train = res_train['return']
ret_test = res_test['return']
status_ret = "[OK] 通过" if ret_test > 0 else "[FAIL] 失败"
if ret_test > ret_train:
    status_ret = "[HOT] 超预期"
print(f"{'总收益率':<12} | {ret_train:>18.2f}% | {ret_test:>18.2f}% | {status_ret}")

# 胜率
win_train = res_train['win_rate']
win_test = res_test['win_rate']
status_win = "[OK] 稳定" if abs(win_test - win_train) < 15 else "[WARN] 偏差大"
if win_test >= 40:
    status_win = "[OK] 优秀"
print(f"{'胜率':<12} | {win_train:>18.1f}% | {win_test:>18.1f}% | {status_win}")

# 回撤
dd_train = res_train['max_dd']
dd_test = res_test['max_dd']
status_dd = "[OK] 控制良好" if dd_test < -20 else "[WARN] 偏大"
print(f"{'最大回撤':<12} | {dd_train:>18.1f}% | {dd_test:>18.1f}% | {status_dd}")

# 交易次数
cnt_train = res_train['trades']
cnt_test = res_test['trades']
print(f"{'交易次数':<12} | {cnt_train:>18} | {cnt_test:>18} | {'-'}")

print("-"*80)

# 最终判决
print(f"\n{'='*80}")
if ret_test > 10 and win_test >= 40:
    print("[SUCCESS] 结论：通过样本外测试！策略具有真实实战价值。")
    print("   这个策略可以在实盘使用，但建议先用小资金验证3-6个月。")
elif ret_test > 0 and win_test >= 30:
    print("[OK] 结论：样本外表现尚可，策略有一定价值。")
    print("   可以实盘使用，但需要降低预期并严格风控。")
else:
    print("[FAIL] 结论：样本外表现不佳，策略存在过拟合。")
    print("   不建议直接用于实盘，需要重新设计或调整参数。")

print(f"{'='*80}")
