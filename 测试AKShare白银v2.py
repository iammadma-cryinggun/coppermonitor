"""
使用AKShare下载白银期货数据 v2
===========================
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


def test_akshare_apis():
    """
    测试AKShare的白银期货接口
    """

    print("="*80)
    print("AKShare白银期货数据测试 v2".center(80))
    print("="*80)

    # 尝试获取上海期货交易所白银期货数据
    print("\n正在尝试获取上海期货交易所白银期货数据...")

    try:
        # 方法1: 使用futures接口获取白银期货数据
        print("\n方法1: futures接口")
        df_ag = ak.futures_spot_sina(symbol="AG0")

        if df_ag is not None and not df_ag.empty:
            print(f"成功获取数据!")
            print(f"数据条数: {len(df_ag)}")
            print(f"数据列: {df_ag.columns.tolist()}")
            print(f"\n数据预览:")
            print(df_ag.head())

            return df_ag
        else:
            print("数据为空")

    except Exception as e:
        print(f"方法1失败: {str(e)}")

    try:
        # 方法2: 尝试获取上海期货交易所白银数据
        print("\n方法2: 尝试直接获取上海期货交易所白银数据")

        # 上海期货交易所白银
        df_shfe = ak.futures_sina_hist(symbol="AG0")

        if df_shfe is not None and not df_shfe.empty:
            print(f"成功获取数据!")
            print(f"数据条数: {len(df_shfe)}")
            print(f"数据列: {df_shfe.columns.tolist()}")
            print(f"\n数据预览:")
            print(df_shfe.head())

            return df_shfe
        else:
            print("数据为空")

    except Exception as e:
        print(f"方法2失败: {str(e)}")

    try:
        # 方法3: 尝试获取中国期货市场数据
        print("\n方法3: 获取中国期货市场数据")

        # 获取期货持仓数据
        df_pos = ak.futures_dce_position_rank(symbol="a", date="20241010")

        if df_pos is not None and not df_pos.empty:
            print(f"成功获取持仓数据!")
            print(f"数据条数: {len(df_pos)}")
            print(f"数据列: {df_pos.columns.tolist()}")
        else:
            print("持仓数据为空")

    except Exception as e:
        print(f"方法3失败: {str(e)}")

    return None


def download_silver_main_contract():
    """
    下载白银期货主力合约日线数据
    """
    print("\n" + "="*80)
    print("下载白银期货主力合约数据".center(80))
    print("="*80)

    try:
        # 获取上海期货交易所白银期货主力合约
        # 白银期货合约代码是AG，加上月份

        # 尝试获取行情数据
        print("\n正在获取白银期货主力合约行情...")

        # 使用sina财经接口
        df = ak.futures_sina_spot(symbol="AG0")

        if df is not None and not df.empty:
            print(f"\n成功下载白银期货主力合约数据!")
            print(f"数据条数: {len(df)}")
            print(f"数据列: {df.columns.tolist()}")
            print(f"\n数据预览:")
            print(df.head(10))

            # 保存数据
            output_file = r"D:\期货数据\铜期货监控\global_futures_daily\白银_daily_akshare.csv"
            df.to_csv(output_file)
            print(f"\n数据已保存至: {output_file}")

            return df
        else:
            print("下载失败: 数据为空")

    except Exception as e:
        print(f"下载失败: {str(e)}")

    return None


def list_all_silver_symbols():
    """
    列出所有白银期货合约
    """
    print("\n" + "="*80)
    print("列出白银期货合约".center(80))
    print("="*80)

    try:
        # 获取白银期货所有合约
        df_ag = ak.futures_contract_info_shfe(symbol="AG")

        if df_ag is not None and not df_ag.empty:
            print(f"\n白银期货合约信息:")
            print(df_ag)

            return df_ag
        else:
            print("未获取到合约信息")

    except Exception as e:
        print(f"获取合约信息失败: {str(e)}")

    return None


def main():
    # 测试AKShare接口
    result = test_akshare_apis()

    if result is None:
        print("\n尝试直接下载主力合约数据...")
        result = download_silver_main_contract()

    # 列出白银合约
    list_all_silver_symbols()

    print("\n" + "="*80)
    print("关于4小时数据".center(80))
    print("="*80)

    print("\nAKShare关于时间周期的说明:")
    print("- 日线数据: 支持")
    print("- 分钟数据: 部分支持（1min, 5min, 15min, 30min, 60min）")
    print("- 4小时数据: 通常不支持")
    print("\n建议:")
    print("1. 使用日线数据进行回测（已有数据充足）")
    print("2. 如需4小时数据，可能需要:")
    print("   - 购买专业数据源（如Wind、Bloomberg）")
    print("   - 从TradingView手动导出")


if __name__ == "__main__":
    main()
