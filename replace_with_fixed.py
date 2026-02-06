# -*- coding: utf-8 -*-
"""
用修正后的4小时数据替换原文件
"""

import shutil
from pathlib import Path

DATA_DIR = Path('futures_data_4h')

# 要替换的文件
FILES_TO_REPLACE = [
    '沪铜_4hour.csv',
    '沪锡_4hour.csv',
    'PTA_4hour.csv',
]

print("="*80)
print("替换4小时数据文件")
print("="*80)

for filename in FILES_TO_REPLACE:
    original = DATA_DIR / filename
    backup = DATA_DIR / f"{filename}.bak"
    fixed = DATA_DIR / f"{filename.replace('.csv', '')}_fixed.csv"

    if not fixed.exists():
        print(f"\n跳过: {filename} (修正文件不存在)")
        continue

    # 备份原文件
    if original.exists():
        shutil.copy(original, backup)
        print(f"\n{filename}:")
        print(f"  ✓ 已备份原文件 → {filename}.bak")

    # 替换
    shutil.copy(fixed, original)
    print(f"  ✓ 已替换为修正文件")

    # 验证
    import pandas as pd
    df = pd.read_csv(original)
    df['datetime'] = pd.to_datetime(df['datetime'])
    hours = sorted(df['datetime'].dt.hour.unique())
    print(f"  ✓ 验证: 小时分布 = {hours}")

print("\n" + "="*80)
print("替换完成！")
print("="*80)
print("\n说明:")
print("  - 原文件已备份为 .bak")
print("  - 新文件时间戳: 01:00, 09:00, 13:00, 21:00")
print("  - 符合实际期货交易时间")
