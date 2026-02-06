# -*- coding: utf-8 -*-
"""
检查各品种开仓条件是否满足
"""

from china_futures_fetcher import ChinaFuturesFetcher
from futures_monitor import TOP10_FUTURES_CONFIG, calculate_indicators, check_signals
import pandas as pd

print("=" * 80)
print("开仓条件详细检查")
print("=" * 80)

fetcher = ChinaFuturesFetcher()

for future_name, config in TOP10_FUTURES_CONFIG.items():
    print(f"\n{'=' * 60}")
    print(f"{future_name} (质量分: {config['quality_score']})")
    print('=' * 60)

    # 获取数据
    df = fetcher.get_historical_data(config['code'], days=5)
    if df is None or df.empty:
        print("❌ 数据获取失败")
        continue

    # 计算指标
    df = calculate_indicators(df, config['params'])

    # 检查信号
    signal = check_signals(df, config['params'], future_name)

    if 'error' in signal:
        print(f"❌ {signal['error']}")
        continue

    # 显示当前状态
    indicators = signal['indicators']
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    print(f"\n当前价格: {signal['price']:.2f}")
    print(f"趋势: {signal['trend']} ({signal['strength']})")
    print(f"\n技术指标:")
    print(f"  EMA快线: {indicators['ema_fast']:.2f}")
    print(f"  EMA慢线: {indicators['ema_slow']:.2f}")
    print(f"  MACD DIF: {indicators['macd_dif']:.2f}")
    print(f"  Ratio: {indicators['ratio']:.3f} (阈值: {config['params']['RATIO_TRIGGER']})")
    print(f"  RSI: {indicators['rsi']:.1f} (阈值: {config['params']['RSI_FILTER']})")
    print(f"  STC: {indicators['stc']:.1f} (卖出区: {config['params']['STC_SELL_ZONE']})")

    # 检查开仓条件
    print(f"\n开仓条件检查:")

    # 条件1: 趋势向上
    trend_up = indicators['ema_fast'] > indicators['ema_slow']
    status = "OK" if trend_up else "FAIL"
    print(f"  [1] 趋势向上 (EMA快线 > EMA慢线): {trend_up} [{status}]")

    # 条件2: Ratio安全
    ratio_safe = (0 < indicators['ratio'] < config['params']['RATIO_TRIGGER'])
    status = "OK" if ratio_safe else "FAIL"
    print(f"  [2] Ratio安全 (0 < {indicators['ratio']:.3f} < {config['params']['RATIO_TRIGGER']}): {ratio_safe} [{status}]")

    # 条件3: Ratio收缩
    ratio_shrinking = indicators['ratio'] < indicators['ratio_prev']
    status = "OK" if ratio_shrinking else "FAIL"
    print(f"  [3] Ratio收缩 ({indicators['ratio']:.3f} < {indicators['ratio_prev']:.3f}): {ratio_shrinking} [{status}]")

    # 条件4: 转头向上
    turning_up = indicators['macd_dif'] > prev['macd_dif']
    status = "OK" if turning_up else "FAIL"
    print(f"  [4] MACD转头向上 ({indicators['macd_dif']:.2f} > {prev['macd_dif']:.2f}): {turning_up} [{status}]")

    # 条件5: 强势
    is_strong = indicators['rsi'] > config['params']['RSI_FILTER']
    status = "OK" if is_strong else "FAIL"
    print(f"  [5] 强势 (RSI {indicators['rsi']:.1f} > {config['params']['RSI_FILTER']}): {is_strong} [{status}]")

    # EMA交叉
    ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (indicators['ema_fast'] > indicators['ema_slow'])
    status = "OK" if ema_cross else "FAIL"
    print(f"  [6] EMA金叉: {ema_cross} [{status}]")

    # 总结
    sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
    chase_signal = ema_cross and is_strong
    buy_signal = sniper_signal or chase_signal

    print(f"\n信号总结:")
    print(f"  狙击信号 (5个条件全满足): {sniper_signal}")
    print(f"  追踪信号 (EMA金叉 + 强势): {chase_signal}")
    if buy_signal:
        print(f"  [BUY] 买入信号: YES")
    else:
        print(f"  [BUY] 买入信号: NO")

    # 卖出信号
    print(f"\n卖出信号: {signal['sell_signal']}")
    if signal['sell_signal']:
        print(f"   原因: {signal['signal_type']}")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)
