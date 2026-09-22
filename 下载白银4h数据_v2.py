"""
下载白银4小时数据 - 使用替代方法
============================
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time


def download_silver_data():
    """
    下载白银数据（尝试多个时间周期）
    """

    # 白银期货符号
    ticker_symbol = "SI=F"

    print("="*80)
    print("白银数据下载".center(80))
    print("="*80)

    # 计算日期范围（5年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    print(f"\n起始日期: {start_date.strftime('%Y-%m-%d')}")
    print(f"结束日期: {end_date.strftime('%Y-%m-%d')}")

    # 尝试不同的时间周期
    intervals = ["1d", "1wk", "1mo"]  # 日线、周线、月线
    interval_names = ["日线", "周线", "月线"]

    for i, (interval, name) in enumerate(zip(intervals, interval_names)):
        print(f"\n正在尝试下载{name}数据...")
        time.sleep(2)  # 等待2秒避免频率限制

        try:
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)

            if not data.empty:
                print(f"成功下载{name}数据!")
                print(f"数据条数: {len(data)}")
                print(f"日期范围: {data.index[0].strftime('%Y-%m-%d')} 至 {data.index[-1].strftime('%Y-%m-%d')}")

                # 保存数据
                if interval == "1d":
                    output_file = r"D:\期货数据\铜期货监控\global_futures_4h\白银_daily_yahoo.csv"
                elif interval == "1wk":
                    output_file = r"D:\期货数据\铜期货监控\global_futures_4h\白银_weekly_yahoo.csv"
                else:
                    output_file = r"D:\期货数据\铜期货监控\global_futures_4h\白银_monthly_yahoo.csv"

                data.to_csv(output_file)
                print(f"已保存至: {output_file}")

                # 显示数据预览
                print(f"\n数据预览:")
                print(data.head())

                # 只保存第一个成功的数据
                return data
            else:
                print(f"{name}数据为空")

        except Exception as e:
            print(f"下载{name}失败: {str(e)}")
            continue

    print("\n所有时间周期都尝试失败")
    print("提示: Yahoo Finance可能对期货4小时数据有限制")
    return None


def try_alternative_sources():
    """
    尝试其他数据源
    """
    print("\n" + "="*80)
    print("尝试其他数据源".center(80))
    print("="*80)

    # 尝试白银ETF (SLV) - 通常有更多数据
    print("\n尝试下载白银ETF (SLV) 数据...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    try:
        ticker = yf.Ticker("SLV")  # iShares Silver Trust
        data = ticker.history(start=start_date, end=end_date, interval="1d")

        if not data.empty:
            print(f"成功下载SLV数据!")
            print(f"数据条数: {len(data)}")
            print(f"日期范围: {data.index[0].strftime('%Y-%m-%d')} 至 {data.index[-1].strftime('%Y-%m-%d')}")

            output_file = r"D:\期货数据\铜期货监控\global_futures_4h\SLV_daily.csv"
            data.to_csv(output_file)
            print(f"已保存至: {output_file}")
            print("\n注意: SLV是白银ETF，与白银期货高度相关但不是完全相同")

            return data
    except Exception as e:
        print(f"下载SLV失败: {str(e)}")

    return None


def main():
    # 先尝试期货数据
    result = download_silver_data()

    # 如果期货失败，尝试ETF
    if result is None:
        result = try_alternative_sources()

    if result is not None:
        print("\n" + "="*80)
        print("数据统计".center(80))
        print("="*80)

        print(f"\nOHLC数据:")
        print(result.describe())

    else:
        print("\n所有数据源都尝试失败")
        print("建议:")
        print("1. 稍后重试")
        print("2. 使用付费数据源")
        print("3. 手动从TradingView或其他平台导出数据")


if __name__ == "__main__":
    main()
