# -*- coding: utf-8 -*-
"""
下载沪锡更久的日线数据
用于测试最优参数在更长周期下的表现
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
import akshare as ak


def download_extended_daily_data(years=10):
    """
    下载指定年数的日线数据

    Args:
        years: 下载数据的年数，默认10年
    """
    print("="*80)
    print(f"下载沪锡日线数据 - {years}年")
    print("="*80)

    # 计算起始日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    print(f"\n数据范围: {start_date.date()} ~ {end_date.date()}")
    print(f"预计数据量: 约 {years*250} 条（每年约250个交易日）")

    try:
        # 下载沪锡日线数据
        print("\n正在下载...")
        df = ak.futures_main_sina(symbol="SN0")

        if df is None or df.empty:
            print("❌ 下载失败：数据为空")
            return None

        # 转换列名
        df = df.rename(columns={
            'date': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'hold': 'hold'
        })

        # 确保有datetime列
        if 'datetime' not in df.columns:
            # 尝试其他可能的列名
            for col in df.columns:
                if 'date' in col.lower() or '时间' in col or '日期' in col:
                    df = df.rename(columns={col: 'datetime'})
                    break

        # 转换日期格式
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 过滤日期范围
        df = df[df['datetime'] >= start_date]
        df = df[df['datetime'] <= end_date]

        # 排序
        df = df.sort_values('datetime').reset_index(drop=True)

        # 保存
        output_file = f"SN_沪锡_日线_{years}年.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"\n[OK] 下载成功!")
        print(f"  实际数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
        print(f"  数据量: {len(df)} 条")
        print(f"  已保存到: {output_file}")

        return df

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import os
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

    # 下载10年数据
    download_extended_daily_data(years=10)
