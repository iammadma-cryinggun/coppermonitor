# -*- coding: utf-8 -*-
"""
从日线失败案例反推趋势判断规则
核心思路：什么样的日线状态下做多是逆势？
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.chdir(r"D:\期货数据\铜期货监控\daily_backtest")

print("="*80)
print("从日线失败案例反推趋势判断规则")
print("="*80)

# ============== 第一步：读取日线交易数据 ==============
print("\n正在读取日线交易数据...")

# 读取交易记录
df_trades = pd.read_csv('trades_with_ghe_lppl.csv')
df_trades['entry_datetime'] = pd.to_datetime(df_trades['entry_datetime'])
df_trades['exit_datetime'] = pd.to_datetime(df_trades['exit_datetime'])

# 读取日线价格数据
df_daily = pd.read_csv('SN_沪锡_日线_10年.csv')
df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
df_daily = df_daily.rename(columns={
    '收盘价': 'close',
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low',
    '成交量': 'volume',
    '持仓量': 'open_interest'
})

# 计算日线EMA趋势指标
def calculate_ema_trend(df, fast=3, slow=15):
    """计算EMA趋势"""
    df[f'ema_{fast}'] = df['close'].ewm(span=fast, adjust=False).mean()
    df[f'ema_{slow}'] = df['close'].ewm(span=slow, adjust=False).mean()

    # 趋势方向：1=上涨，-1=下跌，0=横盘
    df['trend_direction'] = 0
    df.loc[df[f'ema_{fast}'] > df[f'ema_{slow}'], 'trend_direction'] = 1
    df.loc[df[f'ema_{fast}'] < df[f'ema_{slow}'], 'trend_direction'] = -1

    # 趋势强度（快线偏离慢线的程度）
    df['trend_strength'] = (df[f'ema_{fast}'] - df[f'ema_{slow}']) / df[f'ema_{slow}'] * 100

    return df

df_daily = calculate_ema_trend(df_daily, fast=3, slow=15)

print(f"交易记录: {len(df_trades)}笔")
print(f"日线数据: {len(df_daily)}天")

# ============== 第二步：分析失败交易的特征 ==============
print(f"\n{'='*80}")
print("失败交易分析")
print(f"{'='*80}")

# 分离失败交易
losing_trades = df_trades[df_trades['pnl'] <= 0].copy()
winning_trades = df_trades[df_trades['pnl'] > 0].copy()

print(f"\n失败交易: {len(losing_trades)}笔")
print(f"成功交易: {len(winning_trades)}笔")
print(f"胜率: {len(winning_trades)/len(df_trades)*100:.1f}%")

# 为每笔交易添加日线趋势特征
def add_daily_trend_features(trades_df, daily_df):
    """为交易添加日线趋势特征"""
    features_list = []

    for idx, trade in trades_df.iterrows():
        entry_date = trade['entry_datetime']

        # 找到入场日在日线数据中的位置
        daily_idx = daily_df[daily_df['datetime'] == entry_date].index

        if len(daily_idx) == 0:
            continue

        daily_idx = daily_idx[0]

        # 获取入场前后的日线特征
        if daily_idx >= 5:  # 确保有足够的数据
            # 入场当天的日线趋势
            trend_dir = daily_df.loc[daily_idx, 'trend_direction']
            trend_str = daily_df.loc[daily_idx, 'trend_strength']

            # 入场前5天的平均收盘价变化
            past_5_days = daily_df.iloc[daily_idx-5:daily_idx]
            price_change_5d = (past_5_days['close'].iloc[-1] - past_5_days['close'].iloc[0]) / past_5_days['close'].iloc[0] * 100

            # 入场前10天的平均收盘价变化
            if daily_idx >= 10:
                past_10_days = daily_df.iloc[daily_idx-10:daily_idx]
                price_change_10d = (past_10_days['close'].iloc[-1] - past_10_days['close'].iloc[0]) / past_10_days['close'].iloc[0] * 100
            else:
                price_change_10d = price_change_5d

            features_list.append({
                'trade_idx': idx,
                'entry_date': entry_date,
                'trend_direction': trend_dir,
                'trend_strength': trend_str,
                'price_change_5d': price_change_5d,
                'price_change_10d': price_change_10d,
                'pnl': trade['pnl'],
                'pnl_pct': trade['pnl_pct'],
                'exit_reason': trade['exit_reason']
            })

    return pd.DataFrame(features_list)

# 为失败和成功交易添加日线特征
losing_features = add_daily_trend_features(losing_trades, df_daily)
winning_features = add_daily_trend_features(winning_trades, df_daily)

print(f"\n成功提取特征的失败交易: {len(losing_features)}笔")
print(f"成功提取特征的成功交易: {len(winning_features)}笔")

# ============== 第三步：对比失败vs成功交易的日线特征 ==============
print(f"\n{'='*80}")
print("失败 vs 成功交易的日线特征对比")
print(f"{'='*80}")

# 1. 日线趋势方向对比
print(f"\n【日线趋势方向对比】")
print(f"{'趋势方向':<12} {'失败交易占比':<15} {'成功交易占比':<15} {'差异':<10}")
print("-"*60)

for trend_dir in [-1, 0, 1]:
    trend_name = {1: '上涨', -1: '下跌', 0: '横盘'}[trend_dir]

    losing_count = (losing_features['trend_direction'] == trend_dir).sum()
    winning_count = (winning_features['trend_direction'] == trend_dir).sum()

    losing_pct = losing_count / len(losing_features) * 100 if len(losing_features) > 0 else 0
    winning_pct = winning_count / len(winning_features) * 100 if len(winning_features) > 0 else 0

    diff = losing_pct - winning_pct

    print(f"{trend_name:<12} {losing_pct:>8.1f}%       {winning_pct:>8.1f}%       {diff:>+8.1f}%")

# 2. 价格变化对比
print(f"\n【入场前价格变化对比】")
print(f"{'指标':<20} {'失败交易均值':<15} {'成功交易均值':<15} {'差异':<10}")
print("-"*70)

metrics = [
    ('入场前5天涨幅', 'price_change_5d'),
    ('入场前10天涨幅', 'price_change_10d'),
    ('趋势强度', 'trend_strength')
]

for metric_name, metric_col in metrics:
    losing_avg = losing_features[metric_col].mean()
    winning_avg = winning_features[metric_col].mean()
    diff = losing_avg - winning_avg

    print(f"{metric_name:<20} {losing_avg:>10.2f}%       {winning_avg:>10.2f}%       {diff:>+8.2f}%")

# ============== 第四步：找出最危险的日线状态 ==============
print(f"\n{'='*80}")
print("最危险的日线状态（逆势交易的特征）")
print(f"{'='*80}")

# 分析：在日线下跌时做多，是最典型的逆势交易
downtrend_losing = losing_features[losing_features['trend_direction'] == -1]
downtrend_winning = winning_features[winning_features['trend_direction'] == -1]

print(f"\n【日线下跌时做多的结果】")
print(f"  失败交易: {len(downtrend_losing)}笔")
print(f"  成功交易: {len(downtrend_winning)}笔")

if len(downtrend_losing) + len(downtrend_winning) > 0:
    win_rate = len(downtrend_winning) / (len(downtrend_losing) + len(downtrend_winning)) * 100
    avg_loss = downtrend_losing['pnl_pct'].mean()
    avg_win = downtrend_winning['pnl_pct'].mean() if len(downtrend_winning) > 0 else 0

    print(f"  胜率: {win_rate:.1f}%")
    print(f"  平均亏损: {avg_loss:.2f}%")
    print(f"  平均盈利: {avg_win:.2f}%")

    if win_rate < 40:
        print(f"  结论: [!] 典型的逆势交易！胜率过低，应该避免！")

# 分析：在日线横盘时做多
ranging_losing = losing_features[losing_features['trend_direction'] == 0]
ranging_winning = winning_features[winning_features['trend_direction'] == 0]

print(f"\n【日线横盘时做多的结果】")
print(f"  失败交易: {len(ranging_losing)}笔")
print(f"  成功交易: {len(ranging_winning)}笔")

if len(ranging_losing) + len(ranging_winning) > 0:
    win_rate = len(ranging_winning) / (len(ranging_losing) + len(ranging_winning)) * 100
    avg_loss = ranging_losing['pnl_pct'].mean()
    avg_win = ranging_winning['pnl_pct'].mean() if len(ranging_winning) > 0 else 0

    print(f"  胜率: {win_rate:.1f}%")
    print(f"  平均亏损: {avg_loss:.2f}%")
    print(f"  平均盈利: {avg_win:.2f}%")

    if win_rate < 40:
        print(f"  结论: [!] 横盘时做多胜率低，应该等待趋势明确！")

# ============== 第五步：总结趋势判断规则 ==============
print(f"\n{'='*80}")
print("【最终结论】日线趋势判断规则")
print(f"{'='*80}")

# 找出最佳的趋势状态
uptrend_winning = winning_features[winning_features['trend_direction'] == 1]
uptrend_losing = losing_features[losing_features['trend_direction'] == 1]

if len(uptrend_winning) + len(uptrend_losing) > 0:
    uptrend_win_rate = len(uptrend_winning) / (len(uptrend_winning) + len(uptrend_losing)) * 100
    uptrend_avg_return = uptrend_winning['pnl_pct'].mean() if len(uptrend_winning) > 0 else 0
else:
    uptrend_win_rate = 0
    uptrend_avg_return = 0

print(f"""
基于{len(df_trades)}笔日线交易的实证分析：

【推荐】顺势而为：
  日线EMA(3) > EMA(15)（上涨趋势）时做多
  → 胜率: {uptrend_win_rate:.1f}%
  → 平均收益: {uptrend_avg_return:.2f}%
  → 结论: 最佳入场时机

【禁止】逆势交易：
  日线EMA(3) < EMA(15)（下跌趋势）时做多
  → 胜率: {len(downtrend_winning)/(len(downtrend_losing)+len(downtrend_winning))*100 if len(downtrend_losing)+len(downtrend_winning) > 0 else 0:.1f}%
  → 结论: 典型逆势，应该严格禁止

【谨慎】横盘观望：
  日线EMA(3) ≈ EMA(15)（横盘震荡）时
  → 胜率: {len(ranging_winning)/(len(ranging_losing)+len(ranging_winning))*100 if len(ranging_losing)+len(ranging_winning) > 0 else 0:.1f}%
  → 结论: 胜率不确定，建议等待趋势明确后再入场

【实战建议】：
1. 战略层面（日线）：
   - 只在日线EMA(3) > EMA(15)时寻找做多机会
   - 日线EMA(3) < EMA(15)时，严格禁止做多
   - 日线EMA(3) ≈ EMA(15)时，观望等待

2. 战术层面（4小时）：
   - 日线确认上涨后，在4小时寻找入场点
   - 等待4小时EMA交叉或其他确认信号
   - 严格止损，保护本金

3. 风险控制：
   - 即使日线向上，也要设置2%止损
   - 如果日线趋势反转（EMA3下穿EMA15），立即平仓
   - 不要试图抄底，不要猜测转折点
""")

print("\n" + "="*80)
print("分析完成")
print("="*80)

# 保存特征数据供后续使用
losing_features.to_csv('losing_trades_daily_features.csv', index=False)
winning_features.to_csv('winning_trades_daily_features.csv', index=False)
print("\n特征数据已保存：")
print("  - losing_trades_daily_features.csv")
print("  - winning_trades_daily_features.csv")
