"""
检查所有计算细节是否一致
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from cci_calculations import calculate_cci_tv, calculate_stc as calc_stc_original

# 测试数据
test_data = pd.DataFrame({
    'high': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 110, 111, 112],
    'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 105, 106, 107],
    'close': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 108, 109, 110]
})

print("=" * 80)
print("1. CCI计算对比")
print("=" * 80)

# 方法1：老版本回测中的计算
tp1 = (test_data['high'] + test_data['low'] + test_data['close']) / 3
cci_length = 12
ma_length = 5

sma1 = tp1.rolling(window=cci_length).mean()
mad1 = tp1.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean()
cci1 = (tp1 - sma1) / (0.015 * mad1)

# 方法2：cci_calculations.py
result2 = calculate_cci_tv(test_data, cci_length=cci_length, ma_length=ma_length)
cci2 = result2['cci']

print(f"\n最后5个CCI值对比:")
print(f"老版本: {cci1.iloc[-5:].values}")
print(f"cci_calculations.py: {cci2.iloc[-5:].values}")
print(f"最大差异: {abs(cci1 - cci2).max():.6f}")

print("\n" + "=" * 80)
print("2. STC计算对比")
print("=" * 80)

def calculate_stc_new(close_prices, fast_period=23, slow_period=50, cycle_period=10):
    """老版本回测中的STC计算"""
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    lowest_macd = macd.rolling(window=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period).max()
    range_macd = highest_macd - lowest_macd

    stc = pd.Series(index=close_prices.index, dtype=float)
    stc.iloc[:cycle_period] = 0

    for i in range(cycle_period, len(macd)):
        if range_macd.iloc[i] != 0:
            stc.iloc[i] = ((macd.iloc[i] - lowest_macd.iloc[i]) / range_macd.iloc[i] - 0.5) * 200
        else:
            stc.iloc[i] = stc.iloc[i-1] if i > cycle_period else 0

    # 归一化到STC范围
    stc_normalized = pd.Series(index=stc.index, dtype=float)
    lookback = 20
    for i in range(len(stc)):
        if i < lookback:
            stc_normalized.iloc[i] = stc.iloc[i]
        else:
            window = stc.iloc[i-lookback+1:i+1]
            low = window.min()
            high = window.max()
            if high != low:
                stc_normalized.iloc[i] = (stc.iloc[i] - low) / (high - low) * 100 - 50
            else:
                stc_normalized.iloc[i] = stc_normalized.iloc[i-1] if i > 0 else 0

    return stc_normalized

# 方法1：老版本
stc1 = calculate_stc_new(test_data['close'])

# 方法2：cci_calculations.py
stc2_raw = calc_stc_original(test_data)

print(f"\n最后5个STC值对比:")
print(f"老版本: {stc1.iloc[-5:].values}")
print(f"cci_calculations.py: {stc2_raw.iloc[-5:].values}")
print(f"最大差异: {abs(stc1 - stc2_raw).max():.6f}")

print("\n" + "=" * 80)
print("3. 检查参数一致性")
print("=" * 80)

from 最优参数配置 import OPTIMAL_PARAMS

sa_params = OPTIMAL_PARAMS.get('纯碱', {})
print(f"\n纯碱参数（来自OPTIMAL_PARAMS）:")
print(f"  cci_length: {sa_params.get('cci_length', 'N/A')}")
print(f"  ma_length: {sa_params.get('ma_length', 'N/A')}")
print(f"  cci_oversold: {sa_params.get('cci_oversold', 'N/A')}")
print(f"  cci_overbought: {sa_params.get('cci_overbought', 'N/A')}")
print(f"  cci_cross_max: {sa_params.get('cci_cross_max', 'N/A')}")
print(f"  stc_oversold: {sa_params.get('stc_oversold', 'N/A')}")
print(f"  stc_cross: {sa_params.get('stc_cross', 'N/A')}")

# 老版本回测中使用的参数
print(f"\n老版本回测中的参数:")
print(f"  cci_length: 12")
print(f"  ma_length: 5")
print(f"  cci_oversold: -40")
print(f"  cci_overbought: 190")
print(f"  cci_cross_max: 180")
print(f"  stc_oversold: -50")
print(f"  stc_cross: -130")

print("\n" + "=" * 80)
print("4. 核心问题总结")
print("=" * 80)

print("\n已确认的问题：")
print("  1. 策略在2019-2022年盈利 +62.77%")
print("  2. 策略在2023-2025年亏损 -52.32%")
print("  3. 文档中的706.4%/1075.48%收益很可能基于早期数据")
print("  4. 2023年后市场环境改变导致策略失效")

print("\n可能原因：")
print("  A. 数据源问题（akshare的SA0数据不准确）")
print("  B. 市场结构改变（2023年后纯碱市场特征变化）")
print("  C. 参数过拟合（老参数只在早期数据上有效）")
print("  D. 交易成本/滑点假设不准确")

print("\n建议解决方案：")
print("  1. 检查数据质量（对比多个数据源）")
print("  2. 重新优化参数（只用2023-2025年数据）")
print("  3. 暂停实盘使用（直到找到问题根源）")

print("\n" + "=" * 80)
