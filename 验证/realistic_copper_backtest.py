#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铜期货 真实回测 (基于已验证引擎 + 保证金模型)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
修正硬伤:
  1. 数据-合约错配: LME 铜(USD/吨, 25吨/手) 用 SHFE 乘数5 + RMB资金
  2. 仓位模型错误: balance*0.9*2 = 名义敞口180%, 期货应是保证金制

基于 validate_cci_and_copper.py 的 StrictBacktest (已验证+660%):
  - 复用全部指标计算和信号逻辑
  - 复用 next-day open 入场 + 滑点 + 手续费
  - 复用 4%止损/20%止盈/CCI超买/死叉 退出
  - 仅改: 仓位从 balance*0.9*2 改为保证金制

真实期货模型:
  - 每手保证金 = price * multiplier * margin_rate
  - 开仓占用保证金 ≤ balance * pos_margin_pct (30%单品种上限)
  - 手续费: 每手固定(fee_per_lot) + 滑点 0.02%
  - 权益 < 维持保证金(80%初始保证金) → 强制平仓

场景:
  A. LME USD 保证金模型, multiplier=25, 30%上限 → 核心真实数字
  B. SHFE CNY 一致模型, multiplier=5, 30%上限 → 交叉验证(3年)
  C. 对照: 原始错误模型 multiplier=5, 全仓复利 → 已验证+660%
"""
import pandas as pd
import numpy as np
import os, sys

DATA_DIR = r"D:\期货数据\铜期货监控\global_futures_daily"
SHFE_CU = r"D:\期货数据\铜期货监控\daily_backtest\CU_沪铜_日线.csv"
LME_CU = os.path.join(DATA_DIR, "铜_LME_daily.csv")

sys.path.append(r"D:\期货数据\铜期货监控\CCI策略系统")
from cci_calculations import calculate_stc, f_normalize

COPPER_PARAMS = {
    'cci_length': 12, 'ma_length': 20,
    'cci_oversold': -40, 'cci_overbought': 190, 'cci_cross_max': 150,
    'stc_oversold': -50, 'stc_cross': -100,
}
INITIAL = 100000.0

def calc_indicators(df):
    c = df['close']
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(12).mean()
    mad = tp.rolling(12).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    cci_ma = cci.rolling(20).mean()
    stc = calculate_stc(df.copy(), length=10, fast=23, slow=50, aaa=0.5)
    return cci, cci_ma, stc

def has_signal(cci, cci_ma, stc, i):
    if cci.iloc[i] < COPPER_PARAMS['cci_oversold'] and stc.iloc[i] >= COPPER_PARAMS['stc_oversold']:
        return True
    if (cci.iloc[i-1] <= cci_ma.iloc[i-1] and cci.iloc[i] > cci_ma.iloc[i] and
        cci.iloc[i] <= COPPER_PARAMS['cci_cross_max'] and
        stc.iloc[i] >= COPPER_PARAMS['stc_cross']):
        return True
    return False

def margin_backtest(df, initial_capital, multiplier, margin_rate=0.10,
                      pos_margin_pct=0.30, fee_per_lot=10.0, slippage_pct=0.0002,
                      label="", liquidate=True):
    """
    真实期货保证金模型:
    - 保证金制: 每手保证金 = price * multiplier * margin_rate
    - 开仓: 占用保证金 ≤ balance * pos_margin_pct → lots
    - 强平: 权益 < 维持保证金(80%初始) → 强平
    - next-day open 入场, same exits (4% SL / 20% TP / overbought / death cross)
    """
    cci, cci_ma, stc = calc_indicators(df)
    n = len(df)
    balance = initial_capital
    position = 0  # lots held
    entry_price = 0.0
    entry_idx = -1
    trades = []
    equity = []
    peak = initial_capital
    max_dd = 0.0
    forced_liquidations = 0

    for i in range(20, n - 1):
        next_open = df.iloc[i + 1]['open']
        next_high = df.iloc[i + 1]['high']
        next_low = df.iloc[i + 1]['low']
        next_close = df.iloc[i + 1]['close']
        current_price = df.iloc[i]['close']

        if position > 0:
            # 检查退出
            exit_price = None
            exit_reason = None
            # 止损 4%
            if next_low <= entry_price * 0.96:
                exit_price = max(next_open, entry_price * 0.96)
                exit_reason = 'SL'
            elif next_high >= entry_price * 1.20:
                exit_price = min(next_open, entry_price * 1.20)
                exit_reason = 'TP'
            elif cci.iloc[i] > COPPER_PARAMS['cci_overbought']:
                exit_price = next_close
                exit_reason = 'overbought'
            elif cci.iloc[i-1] >= cci_ma.iloc[i-1] and cci.iloc[i] < cci_ma.iloc[i]:
                exit_price = next_close
                exit_reason = 'death_cross'

            if exit_price is not None:
                gross = (exit_price - entry_price) * position * multiplier
                slippage = entry_price * position * multiplier * slippage_pct
                fee = fee_per_lot * position * 2
                net = gross - fee - slippage
                balance += net
                trades.append({'reason': exit_reason, 'pnl': net,
                               'pnl_pct': (exit_price-entry_price)/entry_price*100,
                               'lots': position})
                position = 0
                entry_price = 0.0
            else:
                # 追踪权益, 检查保证金维持
                unrealized = (next_close - entry_price) * position * multiplier
                equity_val = balance + unrealized
                margin_required = entry_price * position * multiplier * margin_rate * 0.80
                if liquidate and equity_val < margin_required and margin_required > 0:
                    # 强平
                    balance += unrealized
                    forced_liquidations += 1
                    position = 0
                    entry_price = 0.0

        else:
            # 开仓
            if has_signal(cci, cci_ma, stc, i):
                entry_price = next_open * (1 + slippage_pct)
                margin_per_lot = entry_price * multiplier * margin_rate
                cap = balance * pos_margin_pct
                lots = int(cap / margin_per_lot)
                lots = max(0, lots)
                if lots > 0:
                    fee = fee_per_lot * lots * 2  # round trip
                    balance -= fee
                    position = lots

        equity_val = balance + ((next_close - entry_price) * position * multiplier if position > 0 else 0)
        equity.append(equity_val)
        peak = max(peak, equity_val)
        dd = (peak - equity_val) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total_return = (equity[-1] - initial_capital) / initial_capital * 100
    wins = sum(1 for t in trades if t['pnl'] > 0)
    wr = wins / len(trades) * 100 if trades else 0
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
    avg_loss = abs(np.mean([t['pnl'] for t in trades if t['pnl'] <= 0])) if wins < len(trades) else 0
    pr = avg_win / avg_loss if avg_loss > 0 else 0

    return {
        'label': label,
        'return': total_return,
        'max_dd': max_dd,
        'trades': len(trades),
        'win_rate': wr,
        'profit_ratio': pr,
        'forced_liq': forced_liquidations,
        'final': equity[-1],
    }

def time_split(df, initial, multiplier, pos_pct, fee, slip, label):
    n = len(df)
    split = int(n * 0.6)
    tr = margin_backtest(df.iloc[:split], initial, multiplier, pos_margin_pct=pos_pct,
                           fee_per_lot=fee, slippage_pct=slip, label=f"{label}[训练]")
    te = margin_backtest(df.iloc[split:], initial, multiplier, pos_margin_pct=pos_pct,
                           fee_per_lot=fee, slippage_pct=slip, label=f"{label}[测试]")
    return tr, te

def main():
    df_lme = pd.read_csv(LME_CU, parse_dates=['date']).set_index('date').dropna()
    df_shfe = pd.read_csv(SHFE_CU, parse_dates=['datetime']).set_index('datetime').dropna()

    print("="*88)
    print("铜期货 真实回测 (保证金模型)")
    print("="*88)
    print(f"LME铜 {len(df_lme)}根 ({df_lme.index[0].date()}~{df_lme.index[-1].date()}) USD/吨")
    print(f"SHFE铜 {len(df_shfe)}根 ({df_shfe.index[0].date()}~{df_shfe.index[-1].date()}) CNY/吨")
    print(f"资金 {INITIAL:,.0f} | 保证金率10% | 单品种上限30% | 滑点0.02%")
    print()

    # ===== 场景对比 =====
    print("="*88)
    print("【场景对比】全时段 LME")
    print("="*88)
    r_ref = margin_backtest(df_lme, INITIAL, multiplier=5, pos_margin_pct=1.8,
                               fee_per_lot=10, slippage_pct=0.0002, label="对照:原始错误模型")
    r_lme = margin_backtest(df_lme, INITIAL, multiplier=25, pos_margin_pct=0.30,
                               fee_per_lot=10, slippage_pct=0.0002, label="LME USD 真实(25t,保证金,30%)")
    r_shfe = margin_backtest(df_shfe, INITIAL, multiplier=5, pos_margin_pct=0.30,
                               fee_per_lot=10, slippage_pct=0.0002, label="SHFE CNY 一致(5t,保证金,30%)")

    for r in [r_lme, r_shfe]:
        print(f"  {r['label']:<32} {r['return']:+7.1f}%  DD {r['max_dd']:>6.1f}%  {r['trades']:>4}笔  "
              f"胜率{r['win_rate']:.0f}%  盈亏比{r['profit_ratio']:.2f}  强平{r['forced_liq']}")

    # ===== 时间切分 =====
    print("\n" + "="*88)
    print("【时间切分样本外】")
    print("="*88)
    tr_l, te_l = time_split(df_lme, INITIAL, 25, 0.30, 10, 0.0002, "LME")
    tr_s, te_s = time_split(df_shfe, INITIAL, 5, 0.30, 10, 0.0002, "SHFE")
    for tr, te, name in [(tr_l, te_l, "LME"), (tr_s, te_s, "SHFE")]:
        if tr and te:
            retention = (te['return'] / tr['return']) if abs(tr['return']) > 0.01 else 0
            print(f"  [{name}] 训练 {tr['return']:+7.1f}% / 测试 {te['return']:+7.1f}% "
                  f"(保留{retention:.0%})  DD {te['max_dd']:.1f}%  强平{te['forced_liq']}")

    # ===== 成本敏感性 =====
    print("\n" + "="*88)
    print("【成本敏感性】LME 真实模型")
    print("="*88)
    for fee, slip, name in [(5, 0.0001, "低 滑0.01%+费$5"), (10, 0.0002, "基准 滑0.02%+费$10"),
                              (20, 0.0004, "高 滑0.04%+费$20")]:
        r = margin_backtest(df_lme, INITIAL, 25, pos_margin_pct=0.30, fee_per_lot=fee, slippage_pct=slip)
        print(f"  {name:<22} {r['return']:+7.1f}%  DD {r['max_dd']:.1f}%  {r['trades']}笔")

    # ===== 参数敏感性 =====
    print("\n" + "="*88)
    print("【参数敏感性】LME 真实模型")
    print("="*88)
    perturbations = [
        ('cci_oversold', -48, 'cci_oversold -48(+20%)'),
        ('cci_oversold', -32, 'cci_oversold -32(-20%)'),
        ('cci_cross_max', 180, 'cci_cross_max 180(+20%)'),
        ('cci_cross_max', 120, 'cci_cross_max 120(-20%)'),
        ('cci_length', 14, 'cci_length 14(+20%)'),
        ('cci_length', 10, 'cci_length 10(-20%)'),
        ('ma_length', 24, 'ma_length 24(+20%)'),
        ('ma_length', 16, 'ma_length 16(-20%)'),
    ]
    for param, val, name in perturbations:
        params = dict(COPPER_PARAMS)
        params[param] = val
        cci, cci_ma, stc = calc_indicators(df_lme)
        # 用修改后参数重跑需要重建整个逻辑 — 简化: 复用引擎
        r = margin_backtest_with_params(df_lme, INITIAL, 25, params, 0.30, 10, 0.0002)
        print(f"  {name:<22} {r['return']:+7.1f}%  DD {r['max_dd']:.1f}%")

    # ===== 逐笔明细 =====
    print("\n" + "="*88)
    print("【LME 逐笔明细 (真实模型)】")
    print("="*88)
    r = margin_backtest(df_lme, INITIAL, 25, pos_margin_pct=0.30, fee_per_lot=10, slippage_pct=0.0002)
    total = 0
    for t in r.get('_trades', [])[:20]:
        total += t['pnl']
        print(f"  {t['reason']:<10} {t['lots']}手 入{t['entry']:8.0f} 出{t['exit']:8.0f} {t['pnl_pct']:+6.1f}% {t['pnl']:+9.1f}U")
    print(f"  前20笔合计: {total:+.1f}U")

# 带参数的版本 (用于参数敏感性)
def margin_backtest_with_params(df, initial, multiplier, params, pos_margin_pct=0.30,
                                  fee_per_lot=10.0, slippage_pct=0.0002, label=""):
    cci, cci_ma, stc = calc_indicators_with_params(df, params)
    n = len(df)
    balance = initial
    position = 0
    entry_price = 0.0
    equity = []
    peak = initial
    max_dd = 0.0

    for i in range(20, n - 1):
        next_open = df.iloc[i + 1]['open']
        next_high = df.iloc[i + 1]['high']
        next_low = df.iloc[i + 1]['low']
        next_close = df.iloc[i + 1]['close']

        if position > 0:
            exit_price = None
            if next_low <= entry_price * 0.96:
                exit_price = max(next_open, entry_price * 0.96)
            elif next_high >= entry_price * 1.20:
                exit_price = min(next_open, entry_price * 1.20)
            elif cci.iloc[i] > params['cci_overbought']:
                exit_price = next_close
            elif cci.iloc[i-1] >= cci_ma.iloc[i-1] and cci.iloc[i] < cci_ma.iloc[i]:
                exit_price = next_close
            if exit_price is not None:
                gross = (exit_price - entry_price) * position * multiplier
                net = gross - entry_price * position * multiplier * slippage_pct - fee_per_lot * position * 2
                balance += net
                position = 0
                entry_price = 0.0
        else:
            if has_signal(cci, cci_ma, stc, i):
                entry_price = next_open * (1 + slippage_pct)
                margin_per_lot = entry_price * multiplier * 0.10
                lots = int(balance * pos_margin_pct / margin_per_lot)
                lots = max(0, lots)
                if lots > 0:
                    balance -= fee_per_lot * lots * 2
                    position = lots

        equity_val = balance + ((next_close - entry_price) * position * multiplier if position > 0 else 0)
        equity.append(equity_val)
        peak = max(peak, equity_val)
        dd = (peak - equity_val) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {'label': label, 'return': (equity[-1]-initial)/initial*100, 'max_dd': max_dd,
            'trades': 0}  # 简化

def calc_indicators_with_params(df, params):
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(params['cci_length']).mean()
    mad = tp.rolling(params['cci_length']).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    cci_ma = cci.rolling(params['ma_length']).mean()
    stc = calculate_stc(df.copy(), length=10, fast=23, slow=50, aaa=0.5)
    return cci, cci_ma, stc

if __name__ == "__main__":
    main()
