"""检查akshare导出的纯碱数据格式和频率"""
import pandas as pd
import numpy as np
from datetime import datetime

def check_soda_ash_data():
    """检查纯碱数据"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol='sa0')
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()

        print("=" * 100)
        print("纯碱(SA0) 数据检查".center(100))
        print("=" * 100)

        print(f"\n数据总条数: {len(df)}条")
        print(f"时间范围: {df['date'].min()} 至 {df['date'].max()}")

        # 显示前10条
        print("\n前10条数据:")
        print(df.head(10).to_string())

        # 显示后10条
        print("\n后10条数据:")
        print(df.tail(10).to_string())

        # 检查数据频率
        df_sorted = df.sort_values('date')
        time_diffs = df_sorted['date'].diff()
        print("\n时间间隔统计:")
        print(f"  平均间隔: {time_diffs.mean()}")
        print(f"  最小间隔: {time_diffs.min()}")
        print(f"  最大间隔: {time_diffs.max()}")

        # 统计间隔分布
        diff_days = time_diffs.dropna().dt.days
        print("\n间隔天数分布:")
        print(diff_days.value_counts().sort_index().head(20).to_string())

        # 统计每年数据量
        df_sorted['year'] = df_sorted['date'].dt.year
        year_counts = df_sorted['year'].value_counts().sort_index()
        print("\n每年数据量:")
        for year, count in year_counts.items():
            print(f"  {year}年: {count}条")

        # 检查是否有缺失交易日
        expected_trading_days = len(df)
        print(f"\n预期交易日: 约{expected_trading_days}天")
        print(f"实际数据量: {len(df)}条")

        print("\n" + "=" * 100)

        return df

    except Exception as e:
        print(f"[ERROR] 数据获取失败: {e}")
        return None


if __name__ == "__main__":
    check_soda_ash_data()
