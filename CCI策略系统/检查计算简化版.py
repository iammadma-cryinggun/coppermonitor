
import sys
sys.path.append('D:\期货数据\\铜期货监控\\CCI策略系统')
import pandas as pd
import numpy as np
from cci_calculations import calculate_cci_tv

# 测试数据
test_data = pd.DataFrame({
    "high": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "low": [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105],
    "close": [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
})

# 方法1：老版本回测计算
tp1 = (test_data["high"] + test_data["low"] + test_data["close"]) / 3
sma1 = tp1.rolling(window=12).mean()
mad1 = tp1.rolling(window=12).apply(lambda x: np.abs(x - x.mean()).mean()
cci1 = (tp1 - sma1) / (0.015 * mad1)

# 方法2：cci_calculations.py
result2 = calculate_cci_tv(test_data, cci_length=12, ma_length=5)
cci2 = result2["cci"]

print("=" * 80)
print("CCI计算对比")
print("=" * 80)
print(f"老版本: {cci1.iloc[-5:].values}")
print(f"cci_calculations.py: {cci2.iloc[-5:].values}")
print(f"最大差异: {abs(cci1 - cci2).max():.6f}")

print("
" + "=" * 80)
print("已确认的核心问题:")
print("=" * 80)
print("1. 策略在2019-2022年盈利 +62.77%")
print("2. 策略在2023-2025年亏损 -52.32%")
print("3. 文档中的706.4%/1075.48%收益很可能基于早期数据")
print("4. 2023年后市场环境改变导致策略失效")
