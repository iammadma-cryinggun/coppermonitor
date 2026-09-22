"""
使用AKShare下载白银期货数据
========================
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


def test_akshare_silver():
    """
    测试AKShare白银期货数据
    """

    print("="*80)
    print("AKShare白银期货数据测试".center(80))
    print("="*80)

    # AKShare可能有的白银期货接口
    # 1. 上海期货交易所白银期货
    # 2. 获取历史数据

    print("\n正在尝试获取上海期货交易所白银期货数据...")

    try:
        # 尝试获取白银期货主力合约数据
        # 白银期货合约代码通常是 AG（上海期货交易所）

        # 获取白银期货历史数据
        print("\n方法1: 获取白银期货主力合约日线数据")
        df_ag = ak.future_spot_sina(symbol="AG0")  # 尝试主力连续合约

        if df_ag is not None and not df_ag.empty:
            print(f"成功获取数据！")
            print(f"数据条数: {len(df_ag)}")
            print(f"数据列: {df_ag.columns.tolist()}")
            print(f"\n数据预览:")
            print(df_ag.head())

            return df_ag
        else:
            print("方法1失败")

    except Exception as e:
        print(f"方法1失败: {str(e)}")

    try:
        # 方法2: 尝试获取上海期货交易所白银期货数据
        print("\n方法2: 获取上海期货交易所白银期货数据")

        # 获取白银期货合约列表
        df_ag2 = ak.future_spot_sina(symbol="AG")

        if df_ag2 is not None and not df_ag2.empty:
            print(f"成功获取数据！")
            print(f"数据条数: {len(df_ag2)}")
            print(f"数据列: {df_ag2.columns.tolist()}")
            print(f"\n数据预览:")
            print(df_ag2.head())

            return df_ag2
        else:
            print("方法2失败")

    except Exception as e:
        print(f"方法2失败: {str(e)}")

    try:
        # 方法3: 尝试获取期货数据
        print("\n方法3: 尝试其他白银期货接口")

        # 上海期货交易所白银
        df_shfe = ak.future_sina_spot_hist(symbol="AG0")

        if df_shfe is not None and not df_shfe.empty:
            print(f"成功获取数据！")
            print(f"数据条数: {len(df_shfe)}")
            print(f"数据列: {df_shfe.columns.tolist()}")
            print(f"\n数据预览:")
            print(df_shfe.head())

            return df_shfe
        else:
            print("方法3失败")

    except Exception as e:
        print(f"方法3失败: {str(e)}")

    try:
        # 方法4: 尝试获取商品期货数据
        print("\n方法4: 获取商品期货日线数据")

        # 尝试获取白银期货历史行情
        df_comm = ak.future_hist_sina(symbol="AG0")

        if df_comm is not None and not df_comm.empty:
            print(f"成功获取数据！")
            print(f"数据条数: {len(df_comm)}")
            print(f"数据列: {df_comm.columns.tolist()}")
            print(f"\n数据预览:")
            print(df_comm.head())

            return df_comm
        else:
            print("方法4失败")

    except Exception as e:
        print(f"方法4失败: {str(e)}")

    return None


def check_4h_data_support():
    """
    检查AKShare是否支持4小时数据
    """
    print("\n" + "="*80)
    print("检查AKShare是否支持4小时数据".center(80))
    print("="*80)

    print("\nAKShare主要支持的数据周期:")
    print("- 日线数据: 支持")
    print("- 周线数据: 支持")
    print("- 月线数据: 支持")
    print("- 分钟数据: 部分支持（通常是1分钟、5分钟、15分钟、30分钟、60分钟）")
    print("- 4小时数据: 可能不支持")

    print("\n正在检查可用的白银期货接口...")

    try:
        # 列出可能的接口
        import inspect

        print("\nAKShare期货相关函数:")
        funcs = [name for name in dir(ak) if 'future' in name.lower()]
        for func in funcs[:20]:  # 只显示前20个
            print(f"  - {func}")

    except Exception as e:
        print(f"检查失败: {str(e)}")


def download_silver_daily():
    """
    下载白银期货日线数据
    """
    print("\n" + "="*80)
    print("下载白银期货日线数据".center(80))
    print("="*80)

    try:
        # 使用期货接口
        df = ak.future_zh_hist_sina(symbol="AG0", period="daily")

        if df is not None and not df.empty:
            print(f"\n成功下载白银期货日线数据!")
            print(f"数据条数: {len(df)}")
            print(f"日期范围: {df.index[0]} 至 {df.index[-1]}")
            print(f"数据列: {df.columns.tolist()}")

            # 保存数据
            output_file = r"D:\期货数据\铜期货监控\global_futures_daily\白银_daily_akshare.csv"
            df.to_csv(output_file)
            print(f"\n数据已保存至: {output_file}")

            # 显示数据预览
            print(f"\n数据预览:")
            print(df.head(10))

            return df
        else:
            print("下载失败: 数据为空")

    except Exception as e:
        print(f"下载失败: {str(e)}")
        print("\n正在尝试其他方法...")

        # 尝试其他方法
        try:
            df2 = ak.future_spot_sina(symbol="AG0")

            if df2 is not None and not df2.empty:
                print(f"\n使用备用方法成功!")
                print(f"数据条数: {len(df2)}")
                print(f"数据列: {df2.columns.tolist()}")

                output_file = r"D:\期货数据\铜期货监控\global_futures_daily\白银_daily_akshare.csv"
                df2.to_csv(output_file)
                print(f"\n数据已保存至: {output_file}")

                return df2
        except Exception as e2:
            print(f"备用方法也失败: {str(e2)}")

    return None


def main():
    # 先测试能否获取白银数据
    result = test_akshare_silver()

    if result is None:
        print("\n所有方法都失败，尝试直接下载...")
        result = download_silver_daily()

    # 检查4小时数据支持
    check_4h_data_support()

    print("\n" + "="*80)
    print("总结".center(80))
    print("="*80)

    print("\nAKShare白银数据获取情况:")
    if result is not None:
        print("✓ 成功获取白银期货数据")
        print("✓ 但可能只支持日线数据")
        print("\n关于4小时数据:")
        print("- AKShare主要支持日线、周线、月线数据")
        print("- 4小时数据可能不支持")
        print("- 建议使用日线数据进行回测")
    else:
        print("✗ 未能获取白银期货数据")
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 更新AKShare版本: pip install akshare --upgrade")
        print("3. 使用现有日线数据")


if __name__ == "__main__":
    main()
