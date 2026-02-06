# -*- coding: utf-8 -*-
"""
重新生成4小时K线数据，对齐到实际期货交易时间
交易时间：01:00, 09:00, 13:00, 21:00
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = Path('futures_data_4h')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 要处理的品种
FUTURES_FILES = [
    ('沪铜_4hour.csv', 'CU'),
    ('沪锡_4hour.csv', 'SN'),
    ('PTA_4hour.csv', 'PTA'),
]

def fix_4h_timestamps(input_file, output_file):
    """
    修复4小时K线时间戳

    原始时间戳 → 修正后时间戳
    00:00     → 01:00  (夜盘收盘)
    08:00     → 09:00  (日盘开盘)
    12:00     → 13:00  (日盘中)
    20:00     → 21:00  (夜盘开盘)
    """
    logger.info(f"处理: {input_file}")

    # 读取数据
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 提取小时并修正
    df['hour'] = df['datetime'].dt.hour

    # 时间戳映射
    hour_map = {
        0: 1,   # 00:00 → 01:00
        8: 9,   # 08:00 → 09:00
        12: 13, # 12:00 → 13:00
        20: 21  # 20:00 → 21:00
    }

    # 应用映射
    df['hour_fixed'] = df['hour'].map(hour_map)

    # 过滤掉无法映射的时间点
    df_before = len(df)
    df = df[df['hour_fixed'].notna()]
    df_after = len(df)
    logger.info(f"  过滤前: {df_before} 条, 过滤后: {df_after} 条")

    if df_after == 0:
        logger.warning(f"  没有有效数据，跳过")
        return None

    # 重建datetime
    df['datetime'] = df.apply(
        lambda row: row['datetime'].replace(hour=int(row['hour_fixed']), minute=0, second=0),
        axis=1
    )

    # 删除临时列
    df = df.drop(columns=['hour', 'hour_fixed'])

    # 按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)

    # 保存
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"  ✓ 已保存: {output_file}")
    logger.info(f"  时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    return df

def main():
    logger.info("="*80)
    logger.info("重新生成4小时K线数据（对齐到实际交易时间）")
    logger.info("="*80)

    for filename, code in FUTURES_FILES:
        input_file = DATA_DIR / filename
        output_file = DATA_DIR / f"{filename.replace('.csv', '')}_fixed.csv"

        if not input_file.exists():
            logger.warning(f"文件不存在: {input_file}")
            continue

        # 处理数据
        df = fix_4h_timestamps(input_file, output_file)

        if df is not None:
            # 验证
            logger.info(f"  验证 - 小时分布:")
            hour_counts = df['datetime'].dt.hour.value_counts().sort_index()
            for hour, count in hour_counts.items():
                logger.info(f"    {hour:02d}:00 → {count} 条")
            logger.info("")

    logger.info("="*80)
    logger.info("处理完成！")
    logger.info("="*80)
    logger.info("\n下一步：")
    logger.info("1. 检查生成的 _fixed.csv 文件")
    logger.info("2. 如果确认无误，替换原文件")
    logger.info("3. 更新回测脚本使用新的数据文件")

if __name__ == '__main__':
    main()
