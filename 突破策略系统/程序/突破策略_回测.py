"""
LME铜突破策略回测系统
======================
核心理念：让利润奔跑，捕捉大趋势

主要改进：
1. 使用唐奇安通道突破（Donchian Channel Breakout）
2. 次日开盘价成交（Next Open）
3. 激进的资金管理（高收益导向）
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# ============ 固定参数 ============
INITIAL_CAPITAL = 100000
MARGIN_RATIO = 0.11  # 11%保证金
CONTRACT_SIZE = 5     # 5吨/手
MAX_LEVERAGE = 3.0    # 最大3倍杠杆

# 突破策略参数
DONCHIAN_PERIOD = 20      # 唐奇安通道周期（突破20日高点）
STOP_PERIOD = 10          # 止损周期（跌破10日低点）
ATR_PERIOD = 14           # ATR周期
ATR_MULTIPLIER = 1.5      # ATR止损倍数
MAX_DRAWDOWN_PCT = 30.0   # 最大回撤限制（放宽到30%）
POSITION_SIZING = 'aggressive'  # 激进仓位管理

# 尝试加载优化后的参数
import os
if os.path.exists('../优化结果/突破策略_最优参数.json'):
    with open('../优化结果/突破策略_最优参数.json', 'r', encoding='utf-8') as f:
        opt_result = json.load(f)
        DONCHIAN_PERIOD = opt_result['params']['donchian_period']
        STOP_PERIOD = opt_result['params']['stop_period']
        ATR_PERIOD = opt_result['params']['atr_period']
        ATR_MULTIPLIER = opt_result['params']['atr_multiplier']
    print(f"\n已加载优化参数: DONCHIAN_PERIOD={DONCHIAN_PERIOD}, STOP_PERIOD={STOP_PERIOD}, ATR_MULTIPLIER={ATR_MULTIPLIER}")
else:
    print(f"\n使用默认参数（未找到优化结果文件）")

# ============ 数据加载 ============
print("加载数据...")
with open('../原始数据/data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df = DATA_POOL['cu'].copy()
print(f"数据范围：{len(df)}根K线")
print(f"时间范围：{df.index[0]} ~ {df.index[-1]}")
print(f"最新价格：{df['close'].iloc[-1]:.2f}")

# ============ 计算指标 ============
def calculate_indicators(df):
    """计算突破策略所需指标"""

    # 唐奇安通道
    df['donchian_high'] = df['high'].rolling(window=DONCHIAN_PERIOD).max()
    df['donchian_low'] = df['low'].rolling(window=DONCHIAN_PERIOD).min()
    df['donchian_middle'] = (df['donchian_high'] + df['donchian_low']) / 2

    # ATR（真实波动幅度）
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)

    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()

    # 移动止损线
    df['stop_low'] = df['low'].rolling(window=STOP_PERIOD).min()

    return df

df = calculate_indicators(df)

# ============ 回测核心 ============
def backtest_breakout(df):
    """突破策略回测"""

    capital = INITIAL_CAPITAL
    peak_capital = capital
    position = None

    trades = []
    capital_history = [capital]

    for i in range(DONCHIAN_PERIOD + 1, len(df) - 1):  # 需要下一根K线数据
        prev = df.iloc[i - 1]
        current = df.iloc[i]
        next_bar = df.iloc[i + 1]  # 次日开盘价

        # ===== 持仓管理 =====
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 1. 止损：跌破移动止损线
            if current['low'] <= current['stop_low']:
                exit_price = min(current['stop_low'], current['open'])  # 考虑开盘价
                exit_triggered = True
                exit_reason = 'stop_loss'

            # 2. 回撤风控
            current_drawdown = (peak_capital - capital) / peak_capital * 100
            if current_drawdown > MAX_DRAWDOWN_PCT:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'risk_control'

            # 3. 趋势反转：跌破中轨
            elif current['close'] < current['donchian_middle']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'trend_reversal'

            if exit_triggered:
                # 计算盈亏
                pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                capital += pnl

                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': current.name,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'leverage': position['leverage'],
                    'pnl': pnl,
                    'pnl_pct': pnl / (capital - pnl) * 100,
                    'exit_reason': exit_reason
                })

                position = None

        # ===== 开仓逻辑 =====
        if position is None:
            # 突破信号：收盘价突破前一根K线的唐奇安通道上轨
            if current['close'] > prev['donchian_high']:
                # 检查ATR是否有效
                if pd.isna(current['atr']) or current['atr'] <= 0:
                    continue

                # 关键修改：使用次日开盘价成交
                entry_price = next_bar['open']
                contract_value = entry_price * CONTRACT_SIZE

                # ATR动态止损
                stop_loss_atr = entry_price - current['atr'] * ATR_MULTIPLIER

                # 激进仓位管理
                if POSITION_SIZING == 'aggressive':
                    # 基于ATR的Kelly仓位（更激进）
                    risk_per_contract = current['atr'] * ATR_MULTIPLIER * CONTRACT_SIZE
                    risk_per_trade = capital * 0.02  # 单笔风险2%
                    contracts = int(risk_per_trade / risk_per_contract)
                else:
                    # 固定杠杆
                    contracts = int((capital * 2.0) / contract_value)

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
                    'entry_date': next_bar.name,  # 记录次日为入场日
                    'entry_price': entry_price,
                    'contracts': contracts,
                    'leverage': actual_leverage,
                    'stop_loss_atr': stop_loss_atr
                }

        capital_history.append(capital)

        if capital > peak_capital:
            peak_capital = capital

    return trades, capital_history

# ============ 运行回测 ============
print("\n" + "="*80)
print("突破策略回测系统".center(80))
print("="*80)
print(f"\n策略参数:")
print(f"  唐奇安通道周期:  {DONCHIAN_PERIOD}日")
print(f"  止损周期:        {STOP_PERIOD}日")
print(f"  ATR倍数:         {ATR_MULTIPLIER}")
print(f"  最大回撤限制:    {MAX_DRAWDOWN_PCT}%")
print(f"  成交方式:        次日开盘价（Next Open）")

print(f"\n交易规则:")
print(f"  初始资金:  {INITIAL_CAPITAL:,.0f}美元")
print(f"  保证金率:  {MARGIN_RATIO*100}%")
print(f"  合约单位:  {CONTRACT_SIZE}吨/手")
print(f"  最大杠杆:  {MAX_LEVERAGE}倍")

print("\n开始回测...")
print("="*80)

trades, capital_history = backtest_breakout(df)

# ============ 结果统计 ============
if not trades:
    print("❌ 未产生任何交易！")
    exit()

final_capital = capital_history[-1]
total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

capital_series = pd.Series(capital_history)
peak = capital_series.cummax()
drawdown = (capital_series - peak) / peak * 100
max_drawdown = drawdown.min()

trades_df = pd.DataFrame(trades)
winning_trades = trades_df[trades_df['pnl'] > 0]
win_rate = len(winning_trades) / len(trades_df) * 100

print("\n" + "="*80)
print("回测结果".center(80))
print("="*80)
print(f"初始资金:  {INITIAL_CAPITAL:,.0f}")
print(f"最终资金:  {final_capital:,.0f}")
print(f"总盈亏:    {final_capital - INITIAL_CAPITAL:+,.0f}")
print(f"总收益率:  {total_return:+.2f}%")
print(f"最大回撤:  {max_drawdown:.2f}%")

print(f"\n交易统计:")
print(f"  总交易:    {len(trades_df)}笔")
print(f"  盈利:      {len(winning_trades)}笔")
print(f"  亏损:      {len(trades_df) - len(winning_trades)}笔")
print(f"  胜率:      {win_rate:.1f}%")

if len(winning_trades) > 0:
    print(f"  平均盈利:  {winning_trades['pnl'].mean():,.0f}")
else:
    print(f"  平均盈利:  N/A")

if len(trades_df) - len(winning_trades) > 0:
    losing_trades = trades_df[trades_df['pnl'] < 0]
    print(f"  平均亏损:  {losing_trades['pnl'].mean():,.0f}")
else:
    print(f"  平均亏损:  N/A")

if len(winning_trades) > 0 and (len(trades_df) - len(winning_trades)) > 0:
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
    print(f"  盈亏比:    {profit_factor:.2f}")

# 杠杆统计
leverages = [t['leverage'] for t in trades]
print(f"\n杠杆统计:")
print(f"  平均杠杆:  {np.mean(leverages):.2f}倍")
print(f"  最大杠杆:  {np.max(leverages):.2f}倍")

print("\n" + "="*80)
print("交易明细（前10笔）".center(80))
print("="*80)
print(trades_df[['entry_date', 'exit_date', 'entry_price', 'exit_price',
                 'contracts', 'leverage', 'pnl', 'pnl_pct', 'exit_reason']].head(10).to_string())

# ============ 保存结果 ============
import os
os.makedirs('../优化结果', exist_ok=True)

trades_df.to_csv('../优化结果/突破策略_交易明细.csv', index=False, encoding='utf-8-sig')

# 确保capital_history与date长度匹配
start_idx = DONCHIAN_PERIOD + 1
capital_df = pd.DataFrame({
    'date': df.index[start_idx:start_idx+len(capital_history)],
    'capital': capital_history
})
capital_df.to_csv('../优化结果/突破策略_资金曲线.csv', index=False, encoding='utf-8-sig')

result_summary = {
    'strategy': 'Donchian Channel Breakout',
    'parameters': {
        'DONCHIAN_PERIOD': DONCHIAN_PERIOD,
        'STOP_PERIOD': STOP_PERIOD,
        'ATR_PERIOD': ATR_PERIOD,
        'ATR_MULTIPLIER': ATR_MULTIPLIER,
        'MAX_DRAWDOWN_PCT': MAX_DRAWDOWN_PCT
    },
    'results': {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'initial_capital': INITIAL_CAPITAL,
        'final_capital': final_capital
    }
}

with open('../优化结果/突破策略_结果.json', 'w', encoding='utf-8') as f:
    json.dump(result_summary, f, indent=2, ensure_ascii=False)

print("\n[OK] 结果已保存到: ../优化结果/")
print("="*80)
