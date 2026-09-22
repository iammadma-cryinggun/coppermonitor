"""
LME铜突破策略 - 进阶版详细回测
==================================
带ADX和EMA过滤器的详细交易分析
"""

import pandas as pd
import numpy as np
import pickle
import json
import os

# ============ 加载数据 ============
print("加载数据...")
with open('../原始数据/data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df = DATA_POOL['cu'].copy()
print(f"数据范围：{len(df)}根K线")
print(f"时间范围：{df.index[0]} ~ {df.index[-1]}")

# ============ 加载最优参数 ============
if os.path.exists('../优化结果/突破策略_进阶版_最优参数.json'):
    with open('../优化结果/突破策略_进阶版_最优参数.json', 'r', encoding='utf-8') as f:
        opt_result = json.load(f)
        params = opt_result['params']
    print(f"\n已加载进阶版优化参数:")
    print(f"  唐奇安周期: {params['donchian_period']}日")
    print(f"  止损周期: {params['stop_period']}日")
    print(f"  EMA周期: {params['ema_period']}日")
    print(f"  ADX阈值: {params['adx_threshold']}")
    print(f"  ATR倍数: {params['atr_multiplier']}")
else:
    print("未找到优化参数，使用默认值")
    params = {
        'donchian_period': 92,
        'stop_period': 14,
        'ema_period': 200,
        'adx_threshold': 29,
        'atr_multiplier': 2.56
    }

# ============ 固定参数 ============
INITIAL_CAPITAL = 100000
CONTRACT_SIZE = 5
MAX_LEVERAGE = 2.0

# ============ 计算指标 ============
def calculate_indicators(df, params):
    """计算所有技术指标"""
    df = df.copy()

    # 唐奇安通道
    df['donchian_high'] = df['high'].rolling(window=params['donchian_period']).max()
    df['donchian_low'] = df['low'].rolling(window=params['donchian_period']).min()
    df['donchian_middle'] = (df['donchian_high'] + df['donchian_low']) / 2

    # EMA
    df['ema_filter'] = df['close'].ewm(span=params['ema_period'], adjust=False).mean()

    # ADX
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

    # ATR
    df['atr'] = df['tr'].rolling(20).mean()

    # 移动止损线
    df['stop_low'] = df['low'].rolling(window=params['stop_period']).min()

    return df

df = calculate_indicators(df, params)

# ============ 回测核心 ============
def backtest_advanced(df, params):
    """进阶版回测"""

    capital = INITIAL_CAPITAL
    peak_capital = capital
    position = None

    trades = []
    capital_history = [capital]

    min_idx = max(params['donchian_period'], params['stop_period'], params['ema_period'], 50) + 1

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

            # 止损
            if current['low'] <= current['stop_low']:
                exit_price = min(current['stop_low'], current['open'])
                exit_triggered = True
                exit_reason = 'stop_loss'

            # 趋势反转
            elif current['close'] < current['donchian_middle']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'trend_reversal'

            # EMA破位
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
                    'leverage': position['leverage'],
                    'pnl': pnl,
                    'pnl_pct': pnl / (capital - pnl) * 100 if capital > pnl else 0,
                    'exit_reason': exit_reason,
                    'entry_adx': position['entry_adx'],
                    'entry_ema_dist': position['entry_ema_dist']
                })
                position = None

        # 开仓逻辑
        if position is None:
            breakout = current['close'] > prev['donchian_high']
            adx_ok = current['adx'] > params['adx_threshold']
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
                    'leverage': actual_leverage,
                    'entry_adx': current['adx'],
                    'entry_ema_dist': (current['close'] - current['ema_filter']) / current['close'] * 100
                }

        capital_history.append(capital)
        if capital > peak_capital:
            peak_capital = capital

    return trades, capital_history

# ============ 运行回测 ============
print("\n" + "="*80)
print("进阶版突破策略 - 详细回测".center(80))
print("="*80)

trades, capital_history = backtest_advanced(df, params)

# ============ 结果统计 ============
if not trades:
    print("未产生交易")
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

print(f"\n初始资金:  {INITIAL_CAPITAL:,.0f}")
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
    avg_profit_days = (pd.to_datetime(winning_trades['exit_date']) - pd.to_datetime(winning_trades['entry_date'])).mean().days
    print(f"  平均持仓天数: {avg_profit_days:.0f}天")

if len(trades_df) - len(winning_trades) > 0:
    losing_trades = trades_df[trades_df['pnl'] < 0]
    print(f"  平均亏损:  {losing_trades['pnl'].mean():,.0f}")
    avg_loss_days = (pd.to_datetime(losing_trades['exit_date']) - pd.to_datetime(losing_trades['entry_date'])).mean().days
    print(f"  平均亏损持仓: {avg_loss_days:.0f}天")

if len(winning_trades) > 0 and (len(trades_df) - len(winning_trades)) > 0:
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum())
    print(f"  盈亏比:    {profit_factor:.2f}")

# 杠杆统计
leverages = [t['leverage'] for t in trades]
print(f"\n杠杆统计:")
print(f"  平均杠杆:  {np.mean(leverages):.2f}倍")
print(f"  最大杠杆:  {np.max(leverages):.2f}倍")

# 过滤器统计
print(f"\n入场条件统计:")
print(f"  平均ADX:   {np.mean([t['entry_adx'] for t in trades]):.1f}")
print(f"  平均价格距离EMA: {np.mean([t['entry_ema_dist'] for t in trades]):.2f}%")

print("\n" + "="*80)
print("交易明细".center(80))
print("="*80)

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

trades_display = trades_df[[
    'entry_date', 'exit_date', 'entry_price', 'exit_price',
    'contracts', 'leverage', 'pnl', 'pnl_pct', 'exit_reason',
    'entry_adx', 'entry_ema_dist'
]].copy()

trades_display.columns = [
    '入场日期', '出场日期', '入场价', '出场价',
    '手数', '杠杆', '盈亏', '盈亏%', '出场原因',
    '入场ADX', '距离EMA%'
]

print(trades_display.to_string(index=False))

# ============ 保存结果 ============
os.makedirs('../优化结果', exist_ok=True)

trades_df.to_csv('../优化结果/进阶版_交易明细.csv', index=False, encoding='utf-8-sig')

capital_df = pd.DataFrame({
    'date': df.index[min(params['donchian_period'], params['stop_period'], params['ema_period'], 50):],
    'capital': capital_history[:len(df.index[min(params['donchian_period'], params['stop_period'], params['ema_period'], 50):])]
})
capital_df = capital_df[capital_df['date'].isin(df.index[min(params['donchian_period'], params['stop_period'], params['ema_period'], 50)+1:min(params['donchian_period'], params['stop_period'], params['ema_period'], 50)+1+len(capital_history)])]
capital_df.to_csv('../优化结果/进阶版_资金曲线.csv', index=False, encoding='utf-8-sig')

print(f"\n[OK] 结果已保存到: ../优化结果/")
print("="*80)
