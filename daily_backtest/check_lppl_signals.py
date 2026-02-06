# -*- coding: utf-8 -*-
"""
检查LPPL指标数据，分析泡沫信号
"""

import pandas as pd
import numpy as np
import json

# 读取数据
df = pd.read_csv("SN_lppl_indicators.csv")
df['time'] = df['time'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))

print(f"总数据: {len(df)} 天")
print(f"时间范围: {df['time'].iloc[0]} ~ {df['time'].iloc[-1]}")

# 解析_fits列，统计D值分布
all_D_values = []

for idx, row in df.iterrows():
    try:
        fits = json.loads(row['_fits'].replace('np.float64(', '').replace(')', ''))
        for f in fits:
            D = f.get('D', None)
            if D is not None and not np.isnan(D):
                all_D_values.append({
                    'date': row['time'],
                    'D': D,
                    'is_qualified': f.get('is_qualified', False)
                })
    except:
        continue

print(f"\n总拟合次数: {len(all_D_values)}")

# D值分布
D_df = pd.DataFrame(all_D_values)
print(f"\nD值分布:")
print(f"  D < 0: {len(D_df[D_df['D'] < 0])}")
print(f"  0 <= D < 0.3: {len(D_df[(D_df['D'] >= 0) & (D_df['D'] < 0.3)])}")
print(f"  0.3 <= D < 0.5: {len(D_df[(D_df['D'] >= 0.3) & (D_df['D'] < 0.5)])}")
print(f"  0.5 <= D < 0.8: {len(D_df[(D_df['D'] >= 0.5) & (D_df['D'] < 0.8)])}")
print(f"  D >= 0.8: {len(D_df[D_df['D'] >= 0.8])}")
print(f"  is_qualified=True: {len(D_df[D_df['is_qualified'] == True])}")

# 找不同阈值的信号天数
print(f"\n不同阈值下的泡沫信号天数:")
for threshold in [0.3, 0.5, 0.8]:
    for min_D in [0.3, 0.5]:
        bubble_days = D_df[(D_df['D'] >= min_D) & (D_df['D'] < threshold)]
        print(f"  {min_D} <= D < {threshold}: {len(bubble_days)} 天")

# 显示最近的高D值
print(f"\n最近30天D值最高的信号:")
recent_D = D_df.tail(30).sort_values('D', ascending=False)
for _, row in recent_D.head(10).iterrows():
    print(f"  {row['date'].date()} D={row['D']:.4f} qualified={row['is_qualified']}")
