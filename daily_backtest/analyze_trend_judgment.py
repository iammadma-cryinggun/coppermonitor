# -*- coding: utf-8 -*-
"""
趋势判断要素分析：从失败交易中学习
核心目标：找出什么样的市场状态是"顺势"，什么状态是"逆势"
"""

import pandas as pd
import numpy as np
import os

os.chdir(r"D:\期货数据\铜期货监控\daily_backtest")

print("="*80)
print("趋势判断要素分析：从108笔交易中学习")
print("="*80)

# ============== 第一步：读取数据 ==============
print("\n正在读取数据...")

# 读取交易记录
trades = pd.read_csv('trades_with_ghe_lppl.csv')
trades['entry_datetime'] = pd.to_datetime(trades['entry_datetime'])

# 读取日线数据
df_daily = pd.read_csv('SN_沪锡_日线_10年.csv')
df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
df_daily = df_daily.rename(columns={
    '收盘价': 'close',
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low'
})

# 计算技术指标
def calculate_indicators(df):
    """计算所有技术指标"""
    # EMA
    df['ema_3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
    df['ema_diff'] = (df['ema_3'] - df['ema_15']) / df['ema_15'] * 100

    # ADX
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=14).mean()
    df['trend'] = np.where(df['plus_di'] > df['minus_di'], 1, -1)

    # 波动率
    df['volatility_5d'] = df['close'].pct_change().rolling(5).std() * 100
    df['volatility_10d'] = df['close'].pct_change().rolling(10).std() * 100

    return df

df_daily = calculate_indicators(df_daily)
df_daily = df_daily.dropna()

print(f"交易记录: {len(trades)}笔")
print(f"日线数据: {len(df_daily)}天")

# ============== 第二步：为每笔交易添加市场状态特征 ==============
print("\n正在为交易添加市场状态特征...")

trades_features = []

for idx, trade in trades.iterrows():
    entry_date = trade['entry_datetime']
    daily_idx = df_daily[df_daily['datetime'] == entry_date].index

    if len(daily_idx) == 0:
        continue

    daily_idx = daily_idx[0]

    # 入场时的市场状态
    if daily_idx >= 10:  # 确保有足够的历史数据
        # EMA趋势状态
        ema_up = df_daily.loc[daily_idx, 'ema_3'] > df_daily.loc[daily_idx, 'ema_15']
        ema_diff = df_daily.loc[daily_idx, 'ema_diff']

        # ADX状态
        adx = df_daily.loc[daily_idx, 'adx']
        plus_di = df_daily.loc[daily_idx, 'plus_di']
        minus_di = df_daily.loc[daily_idx, 'minus_di']
        trend = df_daily.loc[daily_idx, 'trend']

        # 入场前涨幅
        past_5d = df_daily.iloc[daily_idx-5:daily_idx]
        past_10d = df_daily.iloc[daily_idx-10:daily_idx]
        change_5d = (past_5d['close'].iloc[-1] - past_5d['close'].iloc[0]) / past_5d['close'].iloc[0] * 100
        change_10d = (past_10d['close'].iloc[-1] - past_10d['close'].iloc[0]) / past_10d['close'].iloc[0] * 100

        # 波动率
        volatility = df_daily.loc[daily_idx, 'volatility_5d']

        # 价格位置（相对于最近10天的高低点）
        past_10d_high = past_10d['high'].max()
        past_10d_low = past_10d['low'].min()
        price_position = (df_daily.loc[daily_idx, 'close'] - past_10d_low) / (past_10d_high - past_10d_low) * 100

        trades_features.append({
            'entry_date': entry_date,
            # 趋势特征
            'ema_up': int(ema_up),
            'ema_diff': ema_diff,
            'adx': adx,
            'trend': int(trend),
            'plus_di': plus_di,
            'minus_di': minus_di,
            # 入场前涨幅
            'change_5d': change_5d,
            'change_10d': change_10d,
            # 波动率和位置
            'volatility': volatility,
            'price_position': price_position,
            # 结果
            'pnl': trade['pnl'],
            'pnl_pct': trade['pnl_pct'],
            'exit_reason': trade['exit_reason']
        })

df_features = pd.DataFrame(trades_features)

print(f"成功提取特征的交易: {len(df_features)}笔")
print(f"成功交易: {(df_features['pnl'] > 0).sum()}笔")
print(f"失败交易: {(df_features['pnl'] <= 0).sum()}笔")
print(f"整体胜率: {(df_features['pnl'] > 0).sum() / len(df_features) * 100:.1f}%")

# ============== 第三步：对比成功 vs 失败交易的市场状态 ==============
print(f"\n{'='*80}")
print("成功 vs 失败交易的市场状态对比")
print(f"{'='*80}")

winning = df_features[df_features['pnl'] > 0]
losing = df_features[df_features['pnl'] <= 0]

print(f"\n{'指标':<20} {'成功交易均值':<15} {'失败交易均值':<15} {'差异':<10}")
print("-"*70)

metrics = [
    ('EMA偏离度', 'ema_diff'),
    ('ADX强度', 'adx'),
    ('入场前5天涨幅%', 'change_5d'),
    ('入场前10天涨幅%', 'change_10d'),
    ('波动率', 'volatility'),
    ('价格位置%', 'price_position')
]

for metric_name, metric_col in metrics:
    win_avg = winning[metric_col].mean()
    lose_avg = losing[metric_col].mean()
    diff = win_avg - lose_avg

    print(f"{metric_name:<20} {win_avg:>10.2f}        {lose_avg:>10.2f}        {diff:>+8.2f}")

# ============== 第四步：找出最危险的市场状态（逆势） ==============
print(f"\n{'='*80}")
print("最危险的5种市场状态（失败率最高）")
print(f"{'='*80}")

# 分析1：高ADX + 上涨趋势时，胜率如何？
high_adx_uptrend = df_features[(df_features['adx'] >= 25) & (df_features['trend'] == 1)]
if len(high_adx_uptrend) > 0:
    win_rate = (high_adx_uptrend['pnl'] > 0).sum() / len(high_adx_uptrend) * 100
    print(f"\n1. ADX>=25 且 上涨趋势:")
    print(f"   交易数: {len(high_adx_uptrend)}笔")
    print(f"   胜率: {win_rate:.1f}%")
    if win_rate < 40:
        print(f"   结论: [!] 即使ADX高且趋势向上，胜率依然很低！")

# 分析2：入场前5天涨幅>1%（追高）
chase_high = df_features[df_features['change_5d'] > 1.0]
if len(chase_high) > 0:
    win_rate = (chase_high['pnl'] > 0).sum() / len(chase_high) * 100
    print(f"\n2. 入场前5天涨幅>1%（追高）:")
    print(f"   交易数: {len(chase_high)}笔")
    print(f"   胜率: {win_rate:.1f}%")
    if win_rate < 35:
        print(f"   结论: [!] 典型的逆势交易！追高必死！")

# 分析3：ADX<20（无趋势）
low_adx = df_features[df_features['adx'] < 20]
if len(low_adx) > 0:
    win_rate = (low_adx['pnl'] > 0).sum() / len(low_adx) * 100
    print(f"\n3. ADX<20（无明确趋势）:")
    print(f"   交易数: {len(low_adx)}笔")
    print(f"   胜率: {win_rate:.1f}%")
    if win_rate < 40:
        print(f"   结论: [!] 趋势不明确时，不要交易！")

# 分析4：价格位置>80%（高位）
high_position = df_features[df_features['price_position'] > 80]
if len(high_position) > 0:
    win_rate = (high_position['pnl'] > 0).sum() / len(high_position) * 100
    print(f"\n4. 价格位置>80%（接近10天高点）:")
    print(f"   交易数: {len(high_position)}笔")
    print(f"   胜率: {win_rate:.1f}%")
    if win_rate < 40:
        print(f"   结论: [!] 高位追涨，危险！")

# 分析5：波动率过高
high_vol = df_features[df_features['volatility'] > df_features['volatility'].quantile(0.75)]
if len(high_vol) > 0:
    win_rate = (high_vol['pnl'] > 0).sum() / len(high_vol) * 100
    print(f"\n5. 波动率>75分位（高波动）:")
    print(f"   交易数: {len(high_vol)}笔")
    print(f"   胜率: {win_rate:.1f}%")
    if win_rate < 40:
        print(f"   结论: [!] 高波动时，市场不稳定！")

# ============== 第五步：最安全的市场状态（顺势） ==============
print(f"\n{'='*80}")
print("最安全的5种市场状态（胜率最高）")
print(f"{'='*80}")

# 分析1：ADX>=25 + 上涨 + 不追高
safe_combo = df_features[
    (df_features['adx'] >= 25) &
    (df_features['trend'] == 1) &
    (df_features['change_5d'] <= 1.0)
]
if len(safe_combo) > 0:
    win_rate = (safe_combo['pnl'] > 0).sum() / len(safe_combo) * 100
    print(f"\n1. ADX>=25 + 上涨趋势 + 入场前涨幅<1%:")
    print(f"   交易数: {len(safe_combo)}笔")
    print(f"   胜率: {win_rate:.1f}%")
    if win_rate > 50:
        print(f"   结论: [OK] 理想的顺势交易状态！")

# 分析2：价格位置<50%（低位）
low_position = df_features[df_features['price_position'] < 50]
if len(low_position) > 0:
    win_rate = (low_position['pnl'] > 0).sum() / len(low_position) * 100
    print(f"\n2. 价格位置<50%（相对低位）:")
    print(f"   交易数: {len(low_position)}笔")
    print(f"   胜率: {win_rate:.1f}%")
    if win_rate > 50:
        print(f"   结论: [OK] 低位买入，安全！")

# 分析3：低波动
low_vol = df_features[df_features['volatility'] <= df_features['volatility'].quantile(0.25)]
if len(low_vol) > 0:
    win_rate = (low_vol['pnl'] > 0).sum() / len(low_vol) * 100
    print(f"\n3. 波动率<25分位（低波动）:")
    print(f"   交易数: {len(low_vol)}笔")
    print(f"   胜率: {win_rate:.1f}%")
    if win_rate > 50:
        print(f"   结论: [OK] 市场稳定时交易，安全！")

# ============== 第六步：总结趋势判断的3个核心要素 ==============
print(f"\n{'='*80}")
print("【最终结论】趋势判断的3个核心要素")
print(f"{'='*80}")

print(f"""
基于108笔交易的实证分析，胜率{100*len(winning)/len(df_features):.1f}%

核心发现：

要素1：趋势强度（ADX）
  ADX < 20：胜率{(low_adx['pnl']>0).sum()/len(low_adx)*100:.1f}% → 无趋势，不交易
  ADX >= 25：胜率{(df_features[df_features['adx']>=25]['pnl']>0).sum()/len(df_features[df_features['adx']>=25])*100:.1f}% → 趋势明确，可以交易

要素2：入场时机（追高 vs 逢低）
  入场前涨幅>1%：胜率{(chase_high['pnl']>0).sum()/len(chase_high)*100:.1f}% → 追高是逆势！
  入场前涨幅<1%：胜率{(df_features[df_features['change_5d']<=1.0]['pnl']>0).sum()/len(df_features[df_features['change_5d']<=1.0])*100:.1f}% → 逢低是顺势！

要素3：价格位置（高位 vs 低位）
  价格位置>80%：胜率{(high_position['pnl']>0).sum()/len(high_position)*100:.1f}% → 高位危险
  价格位置<50%：胜率{(low_position['pnl']>0).sum()/len(low_position)*100:.1f}% → 低位安全

【定义：什么是"顺势"？】

顺势交易 = 同时满足3个条件：
  1. ADX >= 25（趋势足够强）
  2. +DI > -DI（趋势方向向上）
  3. 入场前涨幅 < 1%（不追高，等待回调）

逆势交易 = 触发任意1个危险信号：
  1. ADX < 20（无趋势，横盘）
  2. 入场前涨幅 > 1%（追高）
  3. 价格位置 > 80%（高位接盘）

【实战规则】

战略层面（日线）：
  ├─ 检查ADX：ADX < 20 → 禁止交易
  ├─ 检查趋势：+DI < -DI → 禁止做多
  ├─ 检查位置：价格位置 > 80% → 禁止追涨
  └─ 检查时机：入场前涨幅 > 1% → 等待回调

战术层面（4小时）：
  ├─ 日线确认顺势后
  ├─ 4小时寻找回调入场点
  └─ 严格止损，保护本金

核心思想：
"不要在趋势不明确时交易（ADX < 20）"
"不要在已经涨很多时追高（涨幅 > 1%）"
"不要在高位接盘（位置 > 80%）"
""")

print("\n" + "="*80)
print("分析完成")
print("="*80)
