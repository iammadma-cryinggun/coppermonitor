# -*- coding: utf-8 -*-
"""
验证沪锡日线数据的真实性和完整性
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def verify_data(csv_file):
    """验证数据质量"""
    print("="*80)
    print("沪锡日线数据验证")
    print("="*80)

    # 读取数据
    df = pd.read_csv(csv_file)

    # 转换列名
    df = df.rename(columns={
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '持仓量': 'hold'
    })

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n1. 基本信息")
    print(f"  数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"  数据量: {len(df)} 条")
    print(f"  时间跨度: {(df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days} 天")
    print(f"  理论交易日: {(df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days / 365 * 250:.0f} 天（估算）")
    print(f"  实际覆盖率: {len(df) / ((df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days / 365 * 250) * 100:.1f}%")

    print(f"\n2. 数据完整性")
    # 检查是否有缺失的交易日
    all_dates = pd.date_range(start=df['datetime'].iloc[0], end=df['datetime'].iloc[-1], freq='D')
    trading_dates = df['datetime'].dt.date.unique()
    weekend_dates = set([d.date() for d in all_dates if d.weekday() >= 5])  # 周末
    missing_dates = [d for d in all_dates if d.date() not in trading_dates and d.date() not in weekend_dates]

    if missing_dates:
        print(f"  缺失的交易日: {len(missing_dates)} 天")
        if len(missing_dates) <= 20:
            for d in missing_dates[:10]:
                print(f"    - {d}")
    else:
        print(f"  [OK] 无缺失交易日")

    # 检查是否有重复数据
    duplicates = df[df.duplicated(subset=['datetime'], keep=False)]
    if len(duplicates) > 0:
        print(f"  [X] 重复数据: {len(duplicates)} 条")
    else:
        print(f"  [OK] 无重复数据")

    print(f"\n3. 数据合理性")
    # 检查价格关系
    invalid_high_low = df[df['high'] < df['low']]
    if len(invalid_high_low) > 0:
        print(f"  [X] 最高价 < 最低价: {len(invalid_high_low)} 条")
    else:
        print(f"  [OK] 最高价 >= 最低价（全部数据）")

    # 检查收盘价是否在最高最低价之间
    invalid_close = df[(df['close'] > df['high']) | (df['close'] < df['low'])]
    if len(invalid_close) > 0:
        print(f"  [X] 收盘价超出区间: {len(invalid_close)} 条")
    else:
        print(f"  [OK] 收盘价在合理区间（全部数据）")

    # 检查是否有零值或负值
    zero_price = df[(df['open'] <= 0) | (df['high'] <= 0) | (df['low'] <= 0) | (df['close'] <= 0)]
    if len(zero_price) > 0:
        print(f"  [X] 零值或负值价格: {len(zero_price)} 条")
    else:
        print(f"  [OK] 无零值或负值价格")

    print(f"\n4. 数据统计")
    print(f"  价格区间:")
    print(f"    开盘价: {df['open'].min():.0f} ~ {df['open'].max():.0f}")
    print(f"    最高价: {df['high'].min():.0f} ~ {df['high'].max():.0f}")
    print(f"    最低价: {df['low'].min():.0f} ~ {df['low'].max():.0f}")
    print(f"    收盘价: {df['close'].min():.0f} ~ {df['close'].max():.0f}")
    print(f"  成交量: {df['volume'].sum():,.0f}")
    print(f"  平均日成交量: {df['volume'].mean():,.0f}")

    print(f"\n5. 连续性检查")
    # 检查是否有长时间中断（超过1周没有数据）
    df['date_diff'] = df['datetime'].diff().dt.days
    long_gaps = df[df['date_diff'] > 10]
    if len(long_gaps) > 0:
        print(f"  [!] 发现{len(long_gaps)}处超过10天的数据中断:")
        for idx, row in long_gaps.iterrows():
            prev_date = df.loc[idx-1, 'datetime']
            print(f"    {prev_date.date()} -> {row['datetime'].date()} (中断{int(row['date_diff'])}天)")
    else:
        print(f"  [OK] 无长时间数据中断")

    print(f"\n6. 最新数据")
    print(f"  最新日期: {df['datetime'].iloc[-1]}")
    print(f"  最新价格: {df['close'].iloc[-1]:.0f}")
    print(f"  今天: {datetime.now().date()}")
    days_ago = (datetime.now() - df['datetime'].iloc[-1]).days
    print(f"  数据滞后: {days_ago} 天")

    print("\n" + "="*80)
    print("数据验证完成")
    print("="*80)


if __name__ == "__main__":
    import os
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    verify_data("SN_沪锡_日线_10年.csv")
