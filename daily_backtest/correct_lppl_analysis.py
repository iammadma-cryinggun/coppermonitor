# -*- coding: utf-8 -*-
"""
正确分析：LPPL在不同市场趋势中的预警准确率
核心问题：LPPL能否识别趋势反转（上涨→下跌，下跌→上涨）
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

from futures_monitor import calculate_indicators

# 加载交易数据
lppl_trades = pd.read_csv('backtest_lppl_bidirectional_trades.csv')
lppl_trades['entry_datetime'] = pd.to_datetime(lppl_trades['entry_datetime'])
lppl_trades['exit_datetime'] = pd.to_datetime(lppl_trades['exit_datetime'])

# 加载原始数据
df = pd.read_csv('SN_沪锡_日线_10年.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

df = df.rename(columns={
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low',
    '收盘价': 'close',
    '成交量': 'volume',
    '持仓量': 'hold'
})

df = calculate_indicators(df, {
    'EMA_FAST': 3,
    'EMA_SLOW': 15,
    'RSI_FILTER': 30,
    'RATIO_TRIGGER': 1.05,
    'STC_SELL_ZONE': 65,
    'STOP_LOSS_PCT': 0.02
})

df['MA60'] = df['close'].rolling(window=60).mean()

print("="*80)
print("LPPL在不同市场趋势中的表现分析")
print("="*80)

# 分析每笔交易时的市场趋势
def get_market_regime(row, df):
    """判断交易时的市场状态"""
    dt = row['entry_datetime']

    # 找到对应日期的数据
    idx = df[df['datetime'] == dt].index
    if len(idx) == 0:
        return 'UNKNOWN'

    idx = idx[0]

    # 判断趋势
    if idx < 60:
        return 'INSUFFICIENT_DATA'

    price = df.loc[idx, 'close']
    ma60 = df.loc[idx, 'MA60']

    if price > ma60:
        if df.loc[idx, 'ema_fast'] > df.loc[idx, 'ema_slow']:
            return 'STRONG_UP'
        else:
            return 'WEAK_UP'
    else:
        if df.loc[idx, 'ema_fast'] < df.loc[idx, 'ema_slow']:
            return 'STRONG_DOWN'
        else:
            return 'WEAK_DOWN'

# 分析交易结果
results = []
for _, trade in lppl_trades.iterrows():
    regime = get_market_regime(trade, df)
    results.append({
        'entry_datetime': trade['entry_datetime'],
        'pnl': trade['pnl'],
        'pnl_pct': trade['pnl_pct'],
        'lppl_signal': trade['lppl_signal'],
        'regime': regime
    })

results_df = pd.DataFrame(results)

# 统计分析
print(f"\n总交易数: {len(results_df)}")

# 按LPPL信号和市场状态分组
signal_regime_stats = results_df.groupby(['lppl_signal', 'regime']).agg({
    'pnl': ['sum', 'mean', 'count'],
    'pnl_pct': ['mean']
}).round(2)

print(f"\n按LPPL信号和市场状态的交易表现:")
print(f"\n{'LPPL信号':<10} {'市场状态':<15} {'交易数':<8} {'总盈亏':<12} {'平均盈亏':<12}")
print("-"*80)

for (signal, regime), group in signal_regime_stats.groupby(level=[0, 1]):
    count = int(group[('pnl', 'count')].iloc[0])
    total_pnl = group[('pnl', 'sum')].iloc[0]
    avg_pnl = group[('pnl_pct', 'mean')].iloc[0]

    print(f"{signal:<10} {regime:<15} {count:<8} {total_pnl:>11,.0f}元 {avg_pnl:>10.2f}%")

# 关键分析
print(f"\n" + "="*80)
print("关键分析：LPPL在趋势反转中的表现")
print("="*80)

# 分析1: LONG信号在不同市场状态的表现
long_trades = results_df[results_df['lppl_signal'] == 'LONG']
print(f"\nLPPL做多信号(LPPL=LONG)的表现:")

if len(long_trades) > 0:
    strong_up = long_trades[long_trades['regime'] == 'STRONG_UP']
    weak_up = long_trades[long_trades['regime'] == 'WEAK_UP']
    strong_down = long_trades[long_trades['regime'] == 'STRONG_DOWN']
    weak_down = long_trades[long_trades['regime'] == 'WEAK_DOWN']

    print(f"  强上升趋势: {len(strong_up)}笔, 胜率: {len(strong_up[strong_up['pnl']>0])/len(strong_up)*100:.1f}%" if len(strong_up) > 0 else "N/A")
    print(f"  弱上升趋势: {len(weak_up)}笔, 胜率: {len(weak_up[weak_up['pnl']>0])/len(weak_up)*100:.1f}%" if len(weak_up) > 0 else "N/A")
    print(f"  强下降趋势: {len(strong_down)}笔, 胜率: {len(strong_down[strong_down['pnl']>0])/len(strong_down)*100:.1f}%" if len(strong_down) > 0 else "N/A")
    print(f"  弱下降趋势: {len(weak_down)}笔, 胜率: {len(weak_down[weak_down['pnl']>0])/len(weak_down)*100:.1f}%" if len(weak_down) > 0 else "N/A")

    if len(strong_up) > 0 and len(strong_down) > 0:
        up_avg = strong_up['pnl'].mean()
        down_avg = strong_down['pnl'].mean()
        print(f"\n  对比: 强上升做多平均{up_avg:,.0f}元, 强下降做多平均{down_avg:,.0f}元")

# 分析2: SHORT信号在不同市场状态的表现
short_trades = results_df[results_df['lppl_signal'] == 'SHORT']
print(f"\nLPPL做空信号(LPPL=SHORT)的表现:")

if len(short_trades) > 0:
    strong_down = short_trades[short_trades['regime'] == 'STRONG_DOWN']
    weak_down = short_trades[short_trades['regime'] == 'WEAK_DOWN']
    strong_up = short_trades[short_trades['regime'] == 'STRONG_UP']
    weak_up = short_trades[short_trades['regime'] == 'WEAK_UP']

    print(f"  强下降趋势: {len(strong_down)}笔, 胜率: {len(strong_down[strong_down['pnl']>0])/len(strong_down)*100:.1f}%" if len(strong_down) > 0 else "N/A")
    print(f"  弱下降趋势: {len(weak_down)}笔, 胜率: {len(weak_down[weak_down['pnl']>0])/len(weak_down)*100:.1f}%" if len(weak_down) > 0 else "N/A")
    print(f"  强上升趋势: {len(strong_up)}笔, 胜率: {len(strong_up[strong_up['pnl']>0])/len(strong_up)*100:.1f}%" if len(strong_up) > 0 else "N/A")
    print(f"  弱上升趋势: {len(weak_up)}笔, 胜率: {len(weak_up[weak_up['pnl']>0])/len(weak_up)*100:.1f}%" if len(weak_up) > 0 else "N/A")

# 分析3: LPPL预测正确的真正价值
print(f"\n" + "="*80)
print("LPPL的真正价值：识别'泡沫破裂'的时机")
print("="*80)

# 找出盈利最大的亏损交易
top_losses = results_df[results_df['pnl'] < 0].nsmallest(10, 'pnl')
print(f"\n亏损最大的10笔交易:")

for _, trade in top_losses.iterrows():
    regime = get_market_regime(trade, df)
    print(f"  {trade['entry_datetime'].date()} {trade['lppl_signal']:5} "
          f"{regime:<15} 亏损:{trade['pnl']:.0f}元 ({trade['pnl_pct']:.2f}%)")

# 最终结论
print(f"\n" + "="*80)
print("最终结论：LPPL在趋势反转中的实际作用")
print("="*80)

print(f"""
基于实际交易数据的分析:

1. LPPL信号的设计目的:
   - LONG信号: 低D值期 → 做多
   - SHORT信号: 高D值期 → 做空
   - 核心逻辑: 识别"泡沫破裂"时机

2. LPPL在实际应用中的问题:
   - 不是"牛市vs熊市"的问题
   - 而是"泡沫期vs非泡沫期"的问题
   - 泡沫期可能出现在上升或下跌趋势中

3. 真正的价值:
   - 如果LPPL能准确识别泡沫破裂点
   - 那么在泡沫破裂前后开仓（做空或减仓）
   - 需要验证：LPPL SHORT信号在强下跌趋势中是否有效

4. 建议的正确分析方向:
   - 不看总收益（因为那是多空混合的结果）
   - 看"LPPL SHORT信号在强下跌趋势前的表现"
   - 看"LPPL LONG信号在强上涨趋势末期的表现"
""")

print("\n" + "="*80)
print("分析完成")
print("="*80)
