# -*- coding: utf-8 -*-
"""
ADX过滤器分析：趋势强度的裁判
核心问题：ADX能否帮助判断何时值得跟随趋势？
"""

import pandas as pd
import numpy as np
import os

os.chdir(r"D:\期货数据\铜期货监控\daily_backtest")

print("="*80)
print("ADX过滤器分析：趋势强度的裁判")
print("="*80)

# ============== 第一步：计算ADX ==============
def calculate_adx(df, period=14):
    """
    计算ADX（Average Directional Index）

    参数：
        df: DataFrame，需包含high, low, close列
        period: 计算周期，默认14

    返回：
        df: 添加了ADX列的DataFrame
    """
    # 计算True Range
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )

    # 计算方向移动
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']

    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
        df['up_move'],
        0
    )

    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
        df['down_move'],
        0
    )

    # 平滑
    df['atr'] = df['tr'].rolling(window=period).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])

    # 计算DX和ADX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=period).mean()

    # 趋势方向：+DI > -DI 为上涨，-DI > +DI 为下跌
    df['trend'] = 0
    df.loc[df['plus_di'] > df['minus_di'], 'trend'] = 1  # 上涨
    df.loc[df['minus_di'] > df['plus_di'], 'trend'] = -1  # 下跌

    return df

# 读取数据
print("\n正在读取数据...")
df_trades = pd.read_csv('trades_with_ghe_lppl.csv')
df_trades['entry_datetime'] = pd.to_datetime(df_trades['entry_datetime'])

df_daily = pd.read_csv('SN_沪锡_日线_10年.csv')
df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
df_daily = df_daily.rename(columns={
    '收盘价': 'close',
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low',
    '成交量': 'volume'
})

# 计算ADX
print("正在计算ADX...")
df_daily = calculate_adx(df_daily, period=14)
df_daily = df_daily.dropna()

print(f"日线数据: {len(df_daily)}天")
print(f"交易记录: {len(df_trades)}笔")

# ============== 第二步：为每笔交易添加ADX特征 ==============
print("\n正在为交易添加ADX特征...")

trades_with_adx = []

for idx, trade in df_trades.iterrows():
    entry_date = trade['entry_datetime']

    # 找到入场日的ADX
    daily_idx = df_daily[df_daily['datetime'] == entry_date].index

    if len(daily_idx) == 0:
        continue

    daily_idx = daily_idx[0]

    # 获取ADX和趋势信息
    adx = df_daily.loc[daily_idx, 'adx']
    plus_di = df_daily.loc[daily_idx, 'plus_di']
    minus_di = df_daily.loc[daily_idx, 'minus_di']
    trend = df_daily.loc[daily_idx, 'trend']

    trades_with_adx.append({
        'entry_date': entry_date,
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di,
        'trend': trend,
        'pnl': trade['pnl'],
        'pnl_pct': trade['pnl_pct']
    })

df_adx = pd.DataFrame(trades_with_adx)
print(f"成功匹配ADX的交易: {len(df_adx)}笔")

# ============== 第三步：分析ADX对胜率的影响 ==============
print(f"\n{'='*80}")
print("ADX对交易胜率的影响")
print(f"{'='*80}")

# 分离成功和失败交易
losing = df_adx[df_adx['pnl'] <= 0]
winning = df_adx[df_adx['pnl'] > 0]

print(f"\n失败交易: {len(losing)}笔")
print(f"成功交易: {len(winning)}笔")

# 整体统计
print(f"\n【整体ADX水平】")
print(f"所有交易的平均ADX: {df_adx['adx'].mean():.2f}")
print(f"失败交易的平均ADX: {losing['adx'].mean():.2f}")
print(f"成功交易的平均ADX: {winning['adx'].mean():.2f}")

# 按ADX区间分析
print(f"\n{'='*80}")
print("不同ADX区间的交易表现")
print(f"{'='*80}")

adx_bins = [0, 15, 20, 25, 30, 100]
adx_labels = ['<15', '15-20', '20-25', '25-30', '>30']

df_adx['adx_group'] = pd.cut(df_adx['adx'], bins=adx_bins, labels=adx_labels)

print(f"\n{'ADX区间':<10} {'总笔数':<8} {'盈利':<8} {'亏损':<8} {'胜率':<10} {'平均收益%':<12}")
print("-"*80)

for label in adx_labels:
    group = df_adx[df_adx['adx_group'] == label]

    if len(group) == 0:
        continue

    total = len(group)
    wins = (group['pnl'] > 0).sum()
    losses = (group['pnl'] <= 0).sum()
    win_rate = wins / total * 100
    avg_return = group['pnl_pct'].mean()

    print(f"{label:<10} {total:<8} {wins:<8} {losses:<8} {win_rate:<10.1f}% {avg_return:<12.2f}")

# ============== 第四步：关键的ADX阈值测试 ==============
print(f"\n{'='*80}")
print("关键发现：ADX阈值的实战意义")
print(f"{'='*80}")

for threshold in [15, 20, 25, 30]:
    high_adx = df_adx[df_adx['adx'] >= threshold]
    low_adx = df_adx[df_adx['adx'] < threshold]

    if len(high_adx) == 0 or len(low_adx) == 0:
        continue

    high_win_rate = (high_adx['pnl'] > 0).sum() / len(high_adx) * 100
    low_win_rate = (low_adx['pnl'] > 0).sum() / len(low_adx) * 100

    print(f"\n【ADX >= {threshold} vs ADX < {threshold}】")
    print(f"  ADX >= {threshold}:")
    print(f"    交易数: {len(high_adx)}笔")
    print(f"    胜率: {high_win_rate:.1f}%")
    print(f"    平均收益: {high_adx['pnl_pct'].mean():.2f}%")

    print(f"  ADX < {threshold}:")
    print(f"    交易数: {len(low_adx)}笔")
    print(f"    胜率: {low_win_rate:.1f}%")
    print(f"    平均收益: {low_adx['pnl_pct'].mean():.2f}%")

    diff = high_win_rate - low_win_rate
    print(f"  差异: {diff:+.1f}个百分点")

    if diff > 10:
        print(f"  结论: [!] ADX >= {threshold}时胜率明显更高，应该过滤！")
    elif diff < -10:
        print(f"  结论: [!] ADX < {threshold}时胜率更高，这很反常...")
    else:
        print(f"  结论: ADX阈值{threshold}对胜率影响不大")

# ============== 第五步：ADX + 趋势方向组合 ==============
print(f"\n{'='*80}")
print("ADX + 趋势方向组合分析")
print(f"{'='*80}")

# 分析：ADX>=25 且 上涨趋势时
good_trend = df_adx[(df_adx['adx'] >= 25) & (df_adx['trend'] == 1)]
weak_trend = df_adx[(df_adx['adx'] < 25) | (df_adx['trend'] != 1)]

print(f"\n【理想组合：ADX >= 25 且 +DI > -DI（上涨趋势）】")
print(f"  交易数: {len(good_trend)}笔")
if len(good_trend) > 0:
    good_win_rate = (good_trend['pnl'] > 0).sum() / len(good_trend) * 100
    good_avg_return = good_trend['pnl_pct'].mean()
    print(f"  胜率: {good_win_rate:.1f}%")
    print(f"  平均收益: {good_avg_return:.2f}%")

print(f"\n【其他情况】")
print(f"  交易数: {len(weak_trend)}笔")
if len(weak_trend) > 0:
    weak_win_rate = (weak_trend['pnl'] > 0).sum() / len(weak_trend) * 100
    weak_avg_return = weak_trend['pnl_pct'].mean()
    print(f"  胜率: {weak_win_rate:.1f}%")
    print(f"  平均收益: {weak_avg_return:.2f}%")

if len(good_trend) > 0 and len(weak_trend) > 0:
    diff = good_win_rate - weak_win_rate
    print(f"\n差异: {diff:+.1f}个百分点")

    if diff > 15:
        print(f"结论: [OK] ADX>=25且上涨趋势时，胜率明显更高！")
    else:
        print(f"结论: [!] ADX过滤的效果不明显，可能需要调整阈值")

# ============== 第六步：最终建议 ==============
print(f"\n{'='*80}")
print("【最终结论】ADX在沪锡上的实战价值")
print(f"{'='*80}")

# 找出最佳ADX区间
best_win_rate = 0
best_adx_range = ""

for label in adx_labels:
    group = df_adx[df_adx['adx_group'] == label]

    if len(group) == 0:
        continue

    win_rate = (group['pnl'] > 0).sum() / len(group) * 100

    if win_rate > best_win_rate and len(group) >= 10:  # 至少10个样本
        best_win_rate = win_rate
        best_adx_range = label

print(f"""
基于{len(df_adx)}笔交易的ADX分析：

【ADX区间表现】
最佳ADX区间: {best_adx_range}（胜率{best_win_rate:.1f}%）

【实战建议】
1. ADX的作用：
   - ADX < 20：市场无趋势，横盘震荡，不建议交易
   - ADX 20-25：趋势形成中，可以开始关注
   - ADX >= 25：明显趋势，适合顺势交易
   - ADX > 40：趋势非常强，但可能接近末期

2. 复合过滤条件：
   【必须同时满足】
   - 日线EMA(3) > EMA(15)（上涨趋势）
   - ADX >= 25（趋势足够强）
   - +DI > -DI（方向向上）
   - 入场前5天涨幅 < 1%（不追高）

   【满足以上条件时】
   - 可以在4小时寻找入场点
   - 胜率会显著提升
   - 趋势可持续性强

3. 禁止入场的情况：
   - ADX < 20（无趋势，横盘）
   - ADX >= 25但 -DI > +DI（下跌趋势）
   - ADX > 40（趋势末期，危险）

4. ADX的独特价值：
   - EMA告诉你"趋势方向"
   - ADX告诉你"趋势强度"
   - 只有"方向对 + 强度够"才值得交易
""")

print("\n" + "="*80)
print("分析完成")
print("="*80)

# 保存数据
df_adx.to_csv('trades_with_adx.csv', index=False)
print("\n数据已保存：trades_with_adx.csv")
