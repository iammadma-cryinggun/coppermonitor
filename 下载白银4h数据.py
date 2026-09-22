"""
下载白银4小时数据
===============
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def download_silver_4h_data(years=5):
    """
    下载白银4小时数据

    参数:
        years: 下载数据年数
    """

    # 白银期货符号
    ticker = "SI=F"

    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    print(f"正在下载白银4小时数据...")
    print(f"起始日期: {start_date.strftime('%Y-%m-%d')}")
    print(f"结束日期: {end_date.strftime('%Y-%m-%d')}")
    print(f"数据源: Yahoo Finance ({ticker})")

    try:
        # 下载4小时数据
        data = yf.download(ticker, start=start_date, end=end_date, interval="4h")

        if data.empty:
            print("\n❌ 下载失败：数据为空")
            print("可能原因:")
            print("1. Yahoo Finance不提供4小时数据")
            print("2. 网络连接问题")
            print("3. 数据源限制")
            return None

        print(f"\n✅ 下载成功！")
        print(f"数据条数: {len(data)}")
        print(f"日期范围: {data.index[0].strftime('%Y-%m-%d')} 至 {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"\n数据预览:")
        print(data.head())

        # 保存数据
        output_file = r"D:\期货数据\铜期货监控\global_futures_4h\白银_4h.csv"
        data.to_csv(output_file)
        print(f"\n✅ 数据已保存至: {output_file}")

        return data

    except Exception as e:
        print(f"\n❌ 下载失败: {str(e)}")
        print("\n尝试下载日线数据作为替代...")

        try:
            # 尝试下载日线数据
            data_daily = yf.download(ticker, start=start_date, end=end_date, interval="1d")

            if not data_daily.empty:
                print(f"\n✅ 日线数据下载成功！")
                print(f"数据条数: {len(data_daily)}")
                print(f"日期范围: {data_daily.index[0].strftime('%Y-%m-%d')} 至 {data_daily.index[-1].strftime('%Y-%m-%d')}")

                # 保存日线数据
                output_file_daily = r"D:\期货数据\铜期货监控\global_futures_4h\白银_daily_yahoo.csv"
                data_daily.to_csv(output_file_daily)
                print(f"\n✅ 日线数据已保存至: {output_file_daily}")
                print(f"\n⚠️ 注意: Yahoo Finance可能不提供4小时数据，已下载日线数据作为替代")

                return data_daily
        except Exception as e2:
            print(f"❌ 日线数据也下载失败: {str(e2)}")

        return None


def main():
    print("="*80)
    print("白银4小时数据下载".center(80))
    print("="*80)

    # 下载数据
    result = download_silver_4h_data(years=5)

    if result is not None:
        print("\n" + "="*80)
        print("数据统计".center(80))
        print("="*80)

        print(f"\n开高低收统计:")
        print(result[['Open', 'High', 'Low', 'Close']].describe())

        print(f"\n成交量统计:")
        print(result['Volume'].describe())


if __name__ == "__main__":
    main()
