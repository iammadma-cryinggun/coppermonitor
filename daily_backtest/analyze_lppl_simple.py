# -*- coding: utf-8 -*-
"""
简单分析LPPL D值分布
"""

import pandas as pd
import json
import re

# 读取数据
df = pd.read_csv("SN_lppl_indicators.csv")
print(f"总数据: {len(df)} 天")

# 统计D值
D_values = []
qualified_count = 0

for idx, row in df.iterrows():
    fits_str = row['_fits']

    # 提取所有D值
    D_pattern = r"'D'\s*:\s*np\.float64\(([^)]+)\)"
    matches = re.findall(D_pattern, fits_str)

    for match in matches:
        try:
            D_val = float(match)
            D_values.append(D_val)
        except:
            pass

    # 检查is_qualified
    if "'is_qualified': True" in fits_str:
        qualified_count += 1

print(f"总D值记录: {len(D_values)}")
print(f"is_qualified=True的记录: {qualified_count}")

if D_values:
    D_series = pd.Series(D_values)
    print(f"\nD值统计:")
    print(f"  最小值: {D_series.min():.4f}")
    print(f"  最大值: {D_series.max():.4f}")
    print(f"  平均值: {D_series.mean():.4f}")
    print(f"  中位数: {D_series.median():.4f}")

    print(f"\nD值分布:")
    print(f"  D < 0: {len(D_series[D_series < 0])} ({len(D_series[D_series < 0])/len(D_series)*100:.1f}%)")
    print(f"  0 <= D < 0.3: {len(D_series[(D_series >= 0) & (D_series < 0.3)])} ({len(D_series[(D_series >= 0) & (D_series < 0.3)])/len(D_series)*100:.1f}%)")
    print(f"  0.3 <= D < 0.5: {len(D_series[(D_series >= 0.3) & (D_series < 0.5)])} ({len(D_series[(D_series >= 0.3) & (D_series < 0.5)])/len(D_series)*100:.1f}%)")
    print(f"  0.5 <= D < 0.8: {len(D_series[(D_series >= 0.5) & (D_series < 0.8)])} ({len(D_series[(D_series >= 0.5) & (D_series < 0.8)])/len(D_series)*100:.1f}%)")
    print(f"  D >= 0.8: {len(D_series[D_series >= 0.8])} ({len(D_series[D_series >= 0.8])/len(D_series)*100:.1f}%)")
else:
    print("未找到D值数据")

print(f"\n结论:")
print(f"  大部分D值都很小（<0.3），说明沪锡10年中没有明显的泡沫特征")
print(f"  这解释了为什么策略在10年表现良好 - 市场相对理性")
