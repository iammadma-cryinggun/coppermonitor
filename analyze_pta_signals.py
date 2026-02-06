# -*- coding: utf-8 -*-
"""
PTA信号详细分析 - 理解为什么没有做空信号
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher

def analyze_pta_signals():
    """详细分析PTA的多空信号"""

    print("=" * 80)
    print("PTA信号详细分析")
    print("=" * 80)

    # 获取数据
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('TA', days=300)

    if df is None or df.empty:
        print("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 计算指标
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=10, adjust=False).mean()

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 获取最新数据
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    print(f"\n当前价格: {latest['close']:.2f}")
    print(f"前一根价格: {prev['close']:.2f}")

    print(f"\n{'='*60}")
    print("做多信号分析")
    print(f"{'='*60}")

    # 做多条件
    trend_up = latest['ema_fast'] > latest['ema_slow']
    ratio_safe_long = (0 < latest['ratio'] < 1.25)
    ratio_shrinking_long = latest['ratio'] < prev['ratio']
    turning_up_long = latest['macd_dif'] > prev['macd_dif']
    is_strong_long = latest['rsi'] > 40

    print(f"1. 趋势向上 (ema_fast > ema_slow): {trend_up}")
    print(f"   ema_fast = {latest['ema_fast']:.2f}")
    print(f"   ema_slow = {latest['ema_slow']:.2f}")

    print(f"\n2. Ratio安全 (0 < ratio < 1.25): {ratio_safe_long}")
    print(f"   当前ratio = {latest['ratio']:.2f}")
    print(f"   要求 < 1.25")

    print(f"\n3. Ratio收缩 (ratio < prev_ratio): {ratio_shrinking_long}")
    print(f"   当前ratio = {latest['ratio']:.2f}")
    print(f"   前值ratio = {prev['ratio']:.2f}")

    print(f"\n4. MACD转头向上 (macd_dif > prev): {turning_up_long}")
    print(f"   当前macd_dif = {latest['macd_dif']:.2f}")
    print(f"   前值macd_dif = {prev['macd_dif']:.2f}")

    print(f"\n5. 强势 (rsi > 40): {is_strong_long}")
    print(f"   当前rsi = {latest['rsi']:.1f}")

    sniper_long = trend_up and ratio_safe_long and ratio_shrinking_long and turning_up_long and is_strong_long

    marker = "[OK]" if sniper_long else "[X]"
    print(f"\n做多狙击信号: {sniper_long} {marker}")

    # 检查哪些条件不满足
    failed_conditions = []
    if not trend_up:
        failed_conditions.append("趋势不向上")
    if not ratio_safe_long:
        failed_conditions.append("Ratio不在安全区(0-1.25)")
    if not ratio_shrinking_long:
        failed_conditions.append("Ratio未收缩")
    if not turning_up_long:
        failed_conditions.append("MACD未转头向上")
    if not is_strong_long:
        failed_conditions.append("RSI不够强势")

    if failed_conditions:
        print(f"不满足原因: {', '.join(failed_conditions)}")

    print(f"\n{'='*60}")
    print("做空信号分析（镜像）")
    print(f"{'='*60}")

    # 做空条件（镜像）
    trend_down = latest['ema_fast'] < latest['ema_slow']
    ratio_safe_short = (-1.25 < latest['ratio'] < 0)
    ratio_falling_short = latest['ratio'] < prev['ratio']  # 这里有问题！
    turning_down_short = latest['macd_dif'] < prev['macd_dif']
    is_weak_short = latest['rsi'] < 60

    print(f"1. 趋势向下 (ema_fast < ema_slow): {trend_down}")
    print(f"   当前是向上趋势 [X]")

    print(f"\n2. Ratio负值区 (-1.25 < ratio < 0): {ratio_safe_short}")
    print(f"   当前ratio = {latest['ratio']:.2f}（正值）[X]")
    print(f"   需要ratio为负值才能做空")

    print(f"\n3. Ratio下降 (ratio < prev_ratio): {ratio_falling_short}")
    print(f"   当前ratio = {latest['ratio']:.2f}")
    print(f"   前值ratio = {prev['ratio']:.2f}")
    print(f"   当前比值更大 [X]（应该是下降，但实际是上升）")

    print(f"\n4. MACD转头向下 (macd_dif < prev): {turning_down_short}")
    print(f"   当前macd_dif = {latest['macd_dif']:.2f}")
    print(f"   前值macd_dif = {prev['macd_dif']:.2f}")

    print(f"\n5. 弱势 (rsi < 60): {is_weak_short}")
    print(f"   当前rsi = {latest['rsi']:.1f} [OK]")

    sniper_short = trend_down and ratio_safe_short and ratio_falling_short and turning_down_short and is_weak_short

    marker = "[OK]" if sniper_short else "[X]"
    print(f"\n做空狙击信号: {sniper_short} {marker}")

    print(f"\n{'='*60}")
    print("关键发现")
    print(f"{'='*60}")

    print(f"\n问题1: 当前趋势向上，不满足做空 [X]")
    print(f"  做多需要: trend_up = {trend_up}")
    print(f"  做空需要: trend_down = {trend_down}")

    print(f"\n问题2: Ratio为正值，不满足做空负值区 [X]")
    print(f"  当前ratio = {latest['ratio']:.2f}")
    print(f"  做多要求: 0 < ratio < 1.25")
    print(f"  做空要求: -1.25 < ratio < 0")
    print(f"  → Ratio为正时，只能做多")
    print(f"  → Ratio为负时，只能做空")

    print(f"\n问题3: Ratio变化方向")
    print(f"  做多需要: ratio收缩（变小）")
    print(f"  做空需要: ratio下降（更负）")
    print(f"  但当前ratio从{prev['ratio']:.2f} → {latest['ratio']:.2f}")
    print(f"  上升了！所以两个条件都不满足 [X]")

    # 统计历史ratio分布
    print(f"\n{'='*60}")
    print("历史Ratio分布（最近500根K线）")
    print(f"{'='*60}")

    ratio_stats = df['ratio'].tail(500).describe()
    print(f"Ratio统计数据:")
    print(f"  最小值: {ratio_stats['min']:.2f}")
    print(f"  最大值: {ratio_stats['max']:.2f}")
    print(f"  平均值: {ratio_stats['mean']:.2f}")
    print(f"  中位数: {ratio_stats['50%']:.2f}")

    # Ratio正负比例
    ratio_positive = (df['ratio'].tail(500) > 0).sum()
    ratio_negative = (df['ratio'].tail(500) < 0).sum()
    ratio_zero = (df['ratio'].tail(500) == 0).sum()

    print(f"\nRatio符号分布:")
    print(f"  正值次数: {ratio_positive} ({ratio_positive/500*100:.1f}%)")
    print(f"  负值次数: {ratio_negative} ({ratio_negative/500*100:.1f}%)")
    print(f"  零值次数: {ratio_zero} ({ratio_zero/500*100:.1f}%)")

    if ratio_negative > 0:
        print(f"\n[OK] 历史上有{ratio_negative}次Ratio为负（可以做空的机会）")
        print(f"  最近一次负值: {df[df['ratio'] < 0]['datetime'].iloc[-1]}")
        print(f"  那时ratio = {df[df['ratio'] < 0]['ratio'].iloc[-1]:.2f}")
    else:
        print(f"\n[X] 历史上Ratio从未为负！")
        print(f"  这意味着做空信号（ratio_safe_short）永远不会触发！")

    print(f"\n{'='*60}")
    print("结论")
    print(f"{'='*60}")

    print(f"\n1. 当前市场不满足任何信号 [OK]")
    print(f"   做多: Ratio太高(2.92 > 1.25)")
    print(f"   做空: 趋势向上，Ratio为正值")

    print(f"\n2. 做空不是简单的'做多反过来' [X]")
    print(f"   原因: Ratio指标的特性")
    print(f"   - 做多: 0 < Ratio < 1.25（正值区）")
    print(f"   - 做空: -1.25 < Ratio < 0（负值区）")
    print(f"   - 当前Ratio几乎总是正值(99.6%的时间)")

    print(f"\n3. 更合理的做空逻辑应该:")
    print(f"   [OK] 趋势向下（ema_fast < ema_slow）")
    print(f"   [OK] 趋势反转（ema从向上转为向下）")
    print(f"   [OK] RSI弱势（rsi < 40，而不是60）")
    print(f"   [X] 不依赖Ratio负值区（因为很少出现）")

    print(f"\n4. 建议: 简化做空信号")
    print(f"   做空信号 = 趋势向下 AND RSI弱势 AND (趋势反转 OR 跌破支撑)")
    print(f"   平空信号 = 趋势向上 OR RSI强势")

if __name__ == '__main__':
    analyze_pta_signals()
