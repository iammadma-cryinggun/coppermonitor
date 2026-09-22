#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCI 铜 +660% 严格样本外验证
━━━━━━━━━━━━━━━━━━━━━━━━
回答核心问题: 声称的 LME 铜 +659.87% / -18.53% DD / 51% 胜率 是否真实 edge?

验证对象: CCI策略系统/全品种严格回测.py 的 StrictBacktest 引擎 (复刻)
  - 入场: 信号在 bar i 收盘计算, next_day 开盘成交 (无 look-ahead)
  - 成本: 滑点 0.02%/边 + 手续费 0.03%/边
  - 止损 4% / 止盈 20% / CCI 超买 / CCI 死叉

6 项验证:
  A. 复现: 铜参数 + LME 10年数据 → 应 ≈ +660%
  B. 时间切分样本外: 前60%训练定死参数 → 后40%测试期
  C. 参数边界敏感性: 每个参数 ±20% 扰动, 看 +660% 是高原还是尖峰
  D. 数据源对比: LME (USD/吨) vs SHFE 沪铜 (CNY/吨), 同参数
  E. 仓位去杠杆: 90%×2x 全仓复利 vs 固定30%仓位上限 (文档声称但代码未执行的规则)
  F. 成本敏感性: round-trip 0.10% → 0.20% → 0.30%
"""
import pandas as pd
import numpy as np
import os
import sys

DATA_DIR = r"D:\期货数据\铜期货监控\global_futures_daily"
SHFE_CU = r"D:\期货数据\铜期货监控\daily_backtest\CU_沪铜_日线.csv"
LME_CU = os.path.join(DATA_DIR, "铜_LME_daily.csv")

sys.path.append(r"D:\期货数据\铜期货监控\CCI策略系统")
from cci_calculations import calculate_stc, f_normalize

# 铜的最优参数 (来自 最优参数配置.py)
COPPER_PARAMS = {
    'cci_length': 12,
    'ma_length': 20,
    'cci_oversold': -40,
    'cci_overbought': 190,
    'cci_cross_max': 150,
    'stc_oversold': -50,
    'stc_cross': -100,
    'multiplier': 5,
}
CLAIMED_RETURN = 659.87
CLAIMED_MAX_DD = -18.53


class StrictBacktest:
    """复刻 全品种严格回测.py 的引擎 (严格, 无 look-ahead, 含成本)"""

    def __init__(self, df, params, commission_rate=0.0003, slippage_rate=0.0002,
                 pos_frac=0.9, leverage=2.0, max_single_pct=None):
        self.df = df.copy()
        self.params = params
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.multiplier = params.get('multiplier', 5)
        self.initial_capital = 100000
        self.leverage = leverage
        self.pos_frac = pos_frac
        # max_single_pct: 若设置, 单一品种仓位不超过总资金比例 (文档声称但原代码未执行)
        self.max_single_pct = max_single_pct
        self._calculate_indicators()

    def _calculate_indicators(self):
        cci_length = self.params['cci_length']
        ma_length = self.params['ma_length']

        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma = tp.rolling(window=cci_length).mean()
        mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
        self.df['cci'] = (tp - sma) / (0.015 * mad)
        self.df['cci_ma'] = self.df['cci'].rolling(window=ma_length).mean()

        stc_raw = calculate_stc(self.df, length=10, fast=23, slow=50, aaa=0.5)
        self.df['stc'] = f_normalize(stc_raw, 20, 80)

    def apply_slippage(self, price, direction='buy'):
        if direction == 'buy':
            return price * (1 + self.slippage_rate)
        else:
            return price * (1 - self.slippage_rate)

    def calculate_commission(self, price, qty):
        return price * qty * self.multiplier * self.commission_rate

    def run(self, start_idx=0, end_idx=None):
        """在 [start_idx, end_idx) 区间回测, 返回完整指标"""
        df = self.df
        if end_idx is None:
            end_idx = len(df)
        balance = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_date = None
        trades = []
        equity = []

        for i in range(max(start_idx, 1), end_idx - 1):
            current = df.iloc[i]
            next_day = df.iloc[i + 1]

            cci = current['cci']
            cci_ma = current['cci_ma']
            stc = current['stc']
            cci_prev = df.iloc[i-1]['cci']
            cci_ma_prev = df.iloc[i-1]['cci_ma']

            if position > 0:
                # 止损 4%
                if next_day['low'] <= entry_price * 0.96:
                    stop_price = max(next_day['open'], entry_price * 0.96)
                    exit_price = self.apply_slippage(stop_price, 'sell')
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = self.calculate_commission(exit_price, position) * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'stop_loss'})
                    position = 0
                    entry_price = 0.0
                    continue

                # 止盈 20%
                if next_day['high'] >= entry_price * 1.20:
                    take_profit_price = min(next_day['open'], entry_price * 1.20)
                    exit_price = self.apply_slippage(take_profit_price, 'sell')
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = self.calculate_commission(exit_price, position) * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'take_profit'})
                    position = 0
                    entry_price = 0.0
                    continue

                # CCI 超买
                if cci > self.params['cci_overbought']:
                    exit_price = self.apply_slippage(next_day['close'], 'sell')
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = self.calculate_commission(exit_price, position) * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'overbought'})
                    position = 0
                    entry_price = 0.0
                    continue

                # CCI 死叉
                if cci_prev >= cci_ma_prev and cci < cci_ma:
                    exit_price = self.apply_slippage(next_day['close'], 'sell')
                    pnl = (exit_price - entry_price) * position * self.multiplier
                    commission = self.calculate_commission(exit_price, position) * 2
                    net_pnl = pnl - commission
                    balance += net_pnl
                    trades.append({'pnl': net_pnl, 'type': 'death_cross'})
                    position = 0
                    entry_price = 0.0
                    continue

            # 开仓
            if position == 0:
                open_signal = False
                if cci < self.params['cci_oversold'] and stc >= self.params['stc_oversold']:
                    open_signal = True
                elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                      cci <= self.params['cci_cross_max'] and
                      stc >= self.params['stc_cross']):
                    open_signal = True

                if open_signal:
                    entry_price = self.apply_slippage(next_day['open'], 'buy')
                    max_value = balance * self.pos_frac * self.leverage
                    if self.max_single_pct is not None:
                        max_value = min(max_value, balance * self.max_single_pct * self.leverage)
                    qty = int(max_value / (entry_price * self.multiplier))
                    qty = max(1, qty)
                    commission = self.calculate_commission(entry_price, qty)
                    balance -= commission
                    position = qty
                    entry_date = next_day.name

            val = balance
            if position > 0:
                unrealized_pnl = (next_day['close'] - entry_price) * position * self.multiplier
                val += unrealized_pnl
            equity.append(val)

        if not equity:
            return None
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame({'pnl': [], 'type': []})
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital * 100
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(winning) / len(trades) * 100 if trades else 0
        avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
        avg_loss = abs(losing['pnl'].mean()) if len(losing) > 0 else 0
        profit_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        equity_s = pd.Series(equity)
        peak = equity_s.cummax()
        dd = (equity_s - peak) / peak * 100
        max_dd = dd.min()
        return {
            'total_return': total_return, 'max_dd': max_dd, 'total_trades': len(trades),
            'win_rate': win_rate, 'profit_ratio': profit_ratio, 'expectancy': (win_rate/100*avg_win - (1-win_rate/100)*avg_loss) if avg_loss > 0 else 0,
        }


def load_lme():
    df = pd.read_csv(LME_CU)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df


def load_shfe():
    df = pd.read_csv(SHFE_CU)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df


def fmt(r):
    if r is None:
        return "交易不足"
    return (f"收益{r['total_return']:>+8.1f}%  回撤{r['max_dd']:>6.1f}%  "
            f"交易{r['total_trades']:>3}  胜率{r['win_rate']:>4.1f}%  盈亏比{r['profit_ratio']:>4.2f}")


def main():
    lme = load_lme()
    n = len(lme)
    split = int(n * 0.6)
    print("=" * 100)
    print("CCI 铜 +660% 严格样本外验证")
    print("=" * 100)
    print(f"LME 数据: {n} 根 ({lme.index[0].date()} ~ {lme.index[-1].date()})")
    print(f"切分: 训练期前60% = {lme.index[0].date()} ~ {lme.index[split].date()}")
    print(f"      测试期后40% = {lme.index[split].date()} ~ {lme.index[-1].date()}")
    print(f"声称值: +{CLAIMED_RETURN}% / DD {CLAIMED_MAX_DD}% / 143笔 / 51.0%胜率 / 盈亏比2.33")
    print()

    # ═══ A. 复现 ═══
    print("=" * 100)
    print("【A】复现: 铜参数 + LME 全时段 → 应 ≈ +660%")
    print("=" * 100)
    r = StrictBacktest(lme, COPPER_PARAMS).run()
    print("  " + fmt(r))
    print(f"  与声称差异: {r['total_return'] - CLAIMED_RETURN:+.1f}pp" if r else "")
    print()

    # ═══ B. 时间切分样本外 ═══
    print("=" * 100)
    print("【B】时间切分: 前60%训练期定死参数 → 后40%测试期 (核心)")
    print("=" * 100)
    r_tr = StrictBacktest(lme, COPPER_PARAMS).run(0, split)
    r_te = StrictBacktest(lme, COPPER_PARAMS).run(split, n)
    print(f"  [训练期] {fmt(r_tr)}")
    print(f"  [测试期] {fmt(r_te)}")
    if r_te and r_tr:
        keep = r_te['total_return'] / abs(r_tr['total_return']) * 100 if r_tr['total_return'] != 0 else 0
        verdict = "样本外保持 → edge 可能真实" if r_te['total_return'] > 0 and keep > 30 else \
                  ("样本外为正但大幅衰减" if r_te['total_return'] > 0 else "样本外为负 → 判定过拟合")
        print(f"  测试期/训练期收益比: {keep:.0f}%  → {verdict}")
    print()

    # ═══ C. 参数边界敏感性 ═══
    print("=" * 100)
    print("【C】参数 ±20% 扰动敏感性 (LME 全时段)")
    print("=" * 100)
    base = StrictBacktest(lme, COPPER_PARAMS).run()
    print(f"  基准(原始参数)        {fmt(base)}")
    for key in ['cci_length', 'ma_length', 'cci_oversold', 'cci_cross_max', 'stc_oversold', 'stc_cross']:
        v = COPPER_PARAMS[key]
        for tag, newv in [('-20%', v * 0.8), ('+20%', v * 1.2)]:
            p = dict(COPPER_PARAMS)
            p[key] = int(round(newv)) if isinstance(v, int) else round(newv, 1)
            rr = StrictBacktest(lme, p).run()
            print(f"  {key:<14} {v:>5} → {p[key]:<6} {fmt(rr)}")
    print()

    # ═══ D. 数据源对比 ═══
    print("=" * 100)
    print("【D】数据源对比: LME (USD/吨) vs SHFE 沪铜 (CNY/吨)")
    print("     同参数重跑 — LME 数据套 SHFE 合约乘数5 + 人民币资金 = 数据-合约错配")
    print("=" * 100)
    r_lme = StrictBacktest(lme, COPPER_PARAMS).run()
    shfe = load_shfe()
    r_shfe = StrictBacktest(shfe, COPPER_PARAMS).run()
    print(f"  LME  (2016-2026, 乘数5) {fmt(r_lme)}")
    print(f"  SHFE (2023-2026, 乘数5) {fmt(r_shfe)}")
    print(f"  SHFE 时段更短且为人民币计价 — 若 LME +660% 而 SHFE 显著不同 → 错配风险坐实")
    print()

    # ═══ E. 仓位去杠杆 ═══
    print("=" * 100)
    print("【E】仓位规则: 90%×2x 全仓复利 vs 文档声称的 '单一品种≤30%'")
    print("=" * 100)
    r_full = StrictBacktest(lme, COPPER_PARAMS).run()
    r_cap = StrictBacktest(lme, COPPER_PARAMS, max_single_pct=0.30).run()
    print(f"  90%×2x 全仓复利 (代码实际执行)  {fmt(r_full)}")
    print(f"  单一品种30%上限 (文档声称规则)  {fmt(r_cap)}")
    if r_full and r_cap:
        print(f"  收益差距: {r_full['total_return'] - r_cap['total_return']:+.1f}pp (全仓高估了 {(r_full['total_return']/abs(r_cap['total_return'])-1)*100 if r_cap['total_return'] else 0:.0f}%)")
    print()

    # ═══ F. 成本敏感性 ═══
    print("=" * 100)
    print("【F】成本敏感性: 滑点0.02%+手续费0.03% (基准) → 翻倍 → 三倍")
    print("=" * 100)
    combos = [
        ("滑0.02%+费0.03% (基准, rt≈0.10%)", 0.0003, 0.0002),
        ("滑0.04%+费0.06% (rt≈0.20%)", 0.0006, 0.0004),
        ("滑0.06%+费0.09% (rt≈0.30%)", 0.0009, 0.0006),
    ]
    for label, comm, slip in combos:
        rr = StrictBacktest(lme, COPPER_PARAMS, commission_rate=comm, slippage_rate=slip).run()
        print(f"  {label:<32} {fmt(rr)}")
    print()

    # ═══ 汇总判定 ═══
    print("=" * 100)
    print("【判定汇总】")
    print("=" * 100)
    print(f"  A 复现:          {'[通过] 复现' if r and abs(r['total_return'] - CLAIMED_RETURN) < 50 else '[警告] 未复现'}")
    print(f"  B 样本外测试期:  {fmt(r_te)}")
    if r_te:
        print(f"                    → {'[通过] 为正, 可能真实' if r_te['total_return'] > 0 else '[失败] 为负, 过拟合'}")
    print(f"  C 参数敏感性:    看上方 12 组扰动, 若多数转负 → 尖峰过拟合")
    print(f"  D 数据-合约错配:  LME 美元数据套 SHFE 乘数5 → {'[警告] 坐实风险' if r_shfe and abs(r_shfe['total_return'] - r_lme['total_return']) > 200 else '[警告] 需要人工判断'}")
    print(f"  E 仓位规则:      {'[警告] 文档声称30%上限, 代码实为90%x2x' if abs(r_full['total_return'] - r_cap['total_return']) > 50 else '[警告] 差异较小'}")
    print(f"  F 成本:          看上方, 若 rt 0.30% 时转负 → edge 薄如纸")


if __name__ == "__main__":
    main()
