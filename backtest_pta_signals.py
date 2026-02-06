# -*- coding: utf-8 -*-
"""
PTA历史信号回测 - 统计做多和做空信号次数
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher

def backtest_pta_signals():
    """回测PTA历史信号"""

    print("=" * 80)
    print("PTA历史信号回测分析")
    print("=" * 80)

    # 获取数据
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('TA', days=300)

    if df is None or df.empty:
        print("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"K线数量: {len(df)}")

    # 计算指标
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()

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

    # 参数
    RSI_FILTER = 40
    RATIO_TRIGGER = 1.25

    # 统计信号
    long_sniper_signals = []
    long_chase_signals = []
    short_sniper_signals = []
    short_chase_signals = []

    print(f"\n开始检测信号...")
    print(f"参数: RSI_FILTER={RSI_FILTER}, RATIO_TRIGGER={RATIO_TRIGGER}")

    # 从第50根开始（确保指标计算完整）
    for i in range(50, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # ========== 做多信号 ==========
        # 狙击做多
        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < RATIO_TRIGGER)
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > RSI_FILTER

        sniper_long = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong

        if sniper_long:
            long_sniper_signals.append({
                'datetime': current['datetime'],
                'price': current['close'],
                'ratio': current['ratio'],
                'rsi': current['rsi'],
                'ema_fast': current['ema_fast'],
                'ema_slow': current['ema_slow']
            })

        # 追涨做多
        ema_golden_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
        chase_long = ema_golden_cross and is_strong

        if chase_long:
            long_chase_signals.append({
                'datetime': current['datetime'],
                'price': current['close'],
                'ratio': current['ratio'],
                'rsi': current['rsi']
            })

        # ========== 做空信号 ==========
        # 狙击做空
        trend_down = current['ema_fast'] < current['ema_slow']
        ratio_safe_short = (-RATIO_TRIGGER < current['ratio'] < 0)
        ratio_falling = current['ratio'] < prev['ratio']
        turning_down = current['macd_dif'] < prev['macd_dif']
        is_weak = current['rsi'] < (100 - RSI_FILTER)

        sniper_short = trend_down and ratio_safe_short and ratio_falling and turning_down and is_weak

        if sniper_short:
            short_sniper_signals.append({
                'datetime': current['datetime'],
                'price': current['close'],
                'ratio': current['ratio'],
                'rsi': current['rsi'],
                'ema_fast': current['ema_fast'],
                'ema_slow': current['ema_slow']
            })

        # 杀跌做空
        ema_death_cross = (prev['ema_fast'] >= prev['ema_slow']) and (current['ema_fast'] < current['ema_slow'])
        chase_short = ema_death_cross and is_weak

        if chase_short:
            short_chase_signals.append({
                'datetime': current['datetime'],
                'price': current['close'],
                'ratio': current['ratio'],
                'rsi': current['rsi']
            })

    # ========== 统计结果 ==========
    print(f"\n{'='*80}")
    print("信号统计结果")
    print(f"{'='*80}")

    print(f"\n【做多信号】")
    print(f"  狙击做多: {len(long_sniper_signals)} 次")
    print(f"  追涨做多: {len(long_chase_signals)} 次")
    print(f"  做多总计: {len(long_sniper_signals) + len(long_chase_signals)} 次")

    print(f"\n【做空信号】")
    print(f"  狙击做空: {len(short_sniper_signals)} 次")
    print(f"  杀跌做空: {len(short_chase_signals)} 次")
    print(f"  做空总计: {len(short_sniper_signals) + len(short_chase_signals)} 次")

    print(f"\n{'='*80}")
    print("详细信号列表")
    print(f"{'='*80}")

    if long_sniper_signals:
        print(f"\n做多狙击信号 ({len(long_sniper_signals)}次):")
        for sig in long_sniper_signals[:10]:  # 只显示前10个
            print(f"  {sig['datetime']} | 价格:{sig['price']:.2f} | Ratio:{sig['ratio']:.2f} | RSI:{sig['rsi']:.1f}")
        if len(long_sniper_signals) > 10:
            print(f"  ... 还有 {len(long_sniper_signals) - 10} 次")

    if long_chase_signals:
        print(f"\n追涨做多信号 ({len(long_chase_signals)}次):")
        for sig in long_chase_signals[:10]:
            print(f"  {sig['datetime']} | 价格:{sig['price']:.2f} | RSI:{sig['rsi']:.1f}")
        if len(long_chase_signals) > 10:
            print(f"  ... 还有 {len(long_chase_signals) - 10} 次")

    if short_sniper_signals:
        print(f"\n做空狙击信号 ({len(short_sniper_signals)}次):")
        for sig in short_sniper_signals[:10]:
            print(f"  {sig['datetime']} | 价格:{sig['price']:.2f} | Ratio:{sig['ratio']:.2f} | RSI:{sig['rsi']:.1f}")
        if len(short_sniper_signals) > 10:
            print(f"  ... 还有 {len(short_sniper_signals) - 10} 次")
    else:
        print(f"\n做空狙击信号: 0次 [从未触发]")

    if short_chase_signals:
        print(f"\n杀跌做空信号 ({len(short_chase_signals)}次):")
        for sig in short_chase_signals[:10]:
            print(f"  {sig['datetime']} | 价格:{sig['price']:.2f} | Ratio:{sig['ratio']:.2f} | RSI:{sig['rsi']:.1f}")
        if len(short_chase_signals) > 10:
            print(f"  ... 还有 {len(short_chase_signals) - 10} 次")
    else:
        print(f"\n杀跌做空信号: 0次 [从未触发]")

    # ========== 分析结论 ==========
    print(f"\n{'='*80}")
    print("分析结论")
    print(f"{'='*80}")

    total_long = len(long_sniper_signals) + len(long_chase_signals)
    total_short = len(short_sniper_signals) + len(short_chase_signals)
    total_signals = total_long + total_short

    print(f"\n1. 信号总数: {total_signals}")
    print(f"   做多: {total_long} 次 ({total_long/total_signals*100:.1f}%)")
    print(f"   做空: {total_short} 次 ({total_short/total_signals*100:.1f}%)" if total_short > 0 else "   做空: 0 次 (0%)")

    if total_short == 0:
        print(f"\n2. [关键发现] 历史上从未出现过做空信号！")
        print(f"   原因: Ratio几乎总是正值，做空需要Ratio为负值")
        print(f"   建议: 修改做空逻辑，不要依赖Ratio负值区")
    else:
        print(f"\n2. 做空信号出现了 {total_short} 次")
        print(f"   说明当前做空逻辑可以触发，但频率较低")

    # Ratio分布
    print(f"\n3. Ratio分布统计:")
    ratio_positive = (df['ratio'] > 0).sum()
    ratio_negative = (df['ratio'] < 0).sum()
    print(f"   正值: {ratio_positive} 次 ({ratio_positive/len(df)*100:.1f}%)")
    print(f"   负值: {ratio_negative} 次 ({ratio_negative/len(df)*100:.1f}%)")
    print(f"   做空需要的负值区只占 {ratio_negative/len(df)*100:.1f}% 的时间")

    print(f"\n{'='*80}")
    print("建议")
    print(f"{'='*80}")

    if total_short == 0:
        print(f"\n当前做空逻辑存在严重问题：")
        print(f"  1. Ratio负值条件太严格，几乎永不满足")
        print(f"  2. 导致做空信号历史上从未出现")
        print(f"\n推荐修改方案：")
        print(f"  [方案1] 放宽Ratio条件")
        print(f"    做空: trend_down AND (ratio < 0 OR ratio < prev_ratio)")
        print(f"  [方案2] 完全不使用Ratio")
        print(f"    做空: trend_down AND rsi < 40 AND (trend_reversal OR breaking_support)")
        print(f"  [方案3] 简化做空信号")
        print(f"    做空: (trend_down OR ema_death_cross) AND rsi < 40")

    print(f"\n回测完成时间: {pd.Timestamp.now()}")

if __name__ == '__main__':
    backtest_pta_signals()
