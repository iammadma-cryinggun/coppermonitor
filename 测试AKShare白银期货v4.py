"""
使用AKShare获取白银期货数据 - 最终版本
=====================================
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


def download_silver_futures():
    """
    下载白银期货数据
    """
    print("="*80)
    print("使用AKShare获取白银期货数据".center(80))
    print("="*80)

    # 尝试多个接口
    methods = [
        ("futures_main_sina", lambda: ak.futures_main_sina()),
        ("futures_spot_price", lambda: ak.futures_spot_price()),
        ("futures_display_main_sina", lambda: ak.futures_display_main_sina()),
    ]

    for method_name, method_func in methods:
        print(f"\n正在尝试: {method_name}")

        try:
            df = method_func()

            if df is not None and not df.empty:
                print(f"成功获取数据!")
                print(f"数据条数: {len(df)}")
                print(f"数据列: {df.columns.tolist()}")
                print(f"\n数据预览:")
                print(df.head(10))

                # 查找白银相关的数据
                if '品种' in df.columns or 'symbol' in df.columns or '代码' in df.columns:
                    print("\n正在筛选白银期货数据...")
                    # 尝试筛选白银
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            silver_data = df[df[col].str.contains('白银|AG|银', case=False, na=False)]
                            if not silver_data.empty:
                                print(f"找到白银数据 ({col}列)!")
                                df = silver_data
                                break

                # 保存数据
                output_file = rf"D:\期货数据\铜期货监控\global_futures_daily\白银期货_{method_name}.csv"
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"\n数据已保存至: {output_file}")

                return df
            else:
                print("数据为空")

        except Exception as e:
            print(f"失败: {str(e)}")

    return None


def try_specific_contract():
    """
    尝试获取特定白银期货合约
    """
    print("\n" + "="*80)
    print("尝试获取白银期货合约数据".center(80))
    print("="*80)

    try:
        # 获取上海期货交易所合约信息
        print("\n获取上海期货交易所合约信息...")
        df_shfe = ak.futures_contract_info_shfe()

        if df_shfe is not None and not df_shfe.empty:
            print(f"成功获取合约信息!")
            print(f"数据列: {df_shfe.columns.tolist()}")
            print(f"\n数据预览:")
            print(df_shfe.head(20))

            # 筛选白银合约
            if 'name' in df_shfe.columns or 'symbol' in df_shfe.columns:
                print("\n正在查找白银(AG)合约...")
                for col in df_shfe.columns:
                    if df_shfe[col].dtype == 'object':
                        ag_data = df_shfe[df_shfe[col].str.contains('AG|白银', case=False, na=False)]
                        if not ag_data.empty:
                            print(f"\n找到白银合约:")
                            print(ag_data)

                            # 保存
                            output_file = r"D:\期货数据\铜期货监控\上海期货交易所白银合约.csv"
                            df_shfe.to_csv(output_file, index=False, encoding='utf-8-sig')
                            print(f"\n已保存至: {output_file}")

                            return ag_data

    except Exception as e:
        print(f"失败: {str(e)}")

    return None


def main():
    # 获取期货数据
    result = download_silver_futures()

    # 获取合约信息
    try_specific_contract()

    print("\n" + "="*80)
    print("总结".center(80))
    print("="*80)

    print("\nAKShare白银数据获取总结:")
    print("1. 白银现货数据: ✓ 已成功获取（上海黄金交易所）")
    print("2. 白银期货数据: 正在尝试...")
    print("3. 4小时数据: ✗ AKShare不支持")

    print("\n关于数据源:")
    print("- 您现有的日线数据格式专业（包含settlement等字段）")
    print("- 可能来自Wind、Bloomberg等专业数据源")
    print("- AKShare是免费数据源，数据格式可能不完全一致")

    print("\n建议:")
    print("1. 使用现有日线数据进行策略测试（已验证有效）")
    print("2. 如需4小时数据，需从相同数据源获取（保持一致性）")
    print("3. 或使用TradingView手动导出4小时数据")


if __name__ == "__main__":
    main()
