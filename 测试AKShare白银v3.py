"""
使用AKShare下载白银数据 - 使用最新接口
=====================================
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


def try_silver_spot_data():
    """
    尝试获取白银现货数据
    """
    print("="*80)
    print("AKShare白银数据测试".center(80))
    print("="*80)

    print("\n正在尝试获取白银现货基准价格...")

    try:
        # 上海黄金交易所白银现货
        df_silver = ak.spot_silver_benchmark_sge()

        if df_silver is not None and not df_silver.empty:
            print(f"\n成功获取白银现货数据!")
            print(f"数据条数: {len(df_silver)}")
            print(f"数据列: {df_silver.columns.tolist()}")
            print(f"\n数据预览:")
            print(df_silver.head(10))

            # 保存数据
            output_file = r"D:\期货数据\铜期货监控\白银现货_SGE.csv"
            df_silver.to_csv(output_file)
            print(f"\n数据已保存至: {output_file}")

            return df_silver
        else:
            print("数据为空")

    except Exception as e:
        print(f"获取失败: {str(e)}")

    return None


def try_silver_consumption_data():
    """
    尝试获取白银消费数据
    """
    print("\n" + "="*80)
    print("尝试获取白银消费相关数据".center(80))
    print("="*80)

    try:
        # 白银消费数据
        df_cons = ak.macro_cons_silver()

        if df_cons is not None and not df_cons.empty:
            print(f"\n成功获取白银消费数据!")
            print(f"数据条数: {len(df_cons)}")
            print(f"数据列: {df_cons.columns.tolist()}")
            print(f"\n数据预览:")
            print(df_cons.head())

            return df_cons
        else:
            print("数据为空")

    except Exception as e:
        print(f"获取失败: {str(e)}")

    return None


def check_futures_functions():
    """
    检查期货相关函数
    """
    print("\n" + "="*80)
    print("检查AKShare期货相关函数".center(80))
    print("="*80)

    try:
        # 列出期货相关的所有函数
        funcs = [name for name in dir(ak) if 'future' in name.lower()]

        print(f"\n找到 {len(funcs)} 个期货相关函数:")
        for func in funcs[:50]:  # 只显示前50个
            print(f"  {func}")

    except Exception as e:
        print(f"检查失败: {str(e)}")


def try_qihuo_data():
    """
    尝试使用qihuo（期货）相关接口
    """
    print("\n" + "="*80)
    print("尝试qihuo期货接口".center(80))
    print("="*80)

    try:
        # 尝试获取期货数据
        # qihuo可能是期货的拼音

        # 先检查有没有qihuo相关的函数
        qihuo_funcs = [name for name in dir(ak) if 'qihuo' in name.lower()]

        print(f"\n找到 {len(qihuo_funcs)} 个qihuo相关函数:")
        for func in qihuo_funcs[:20]:
            print(f"  {func}")

    except Exception as e:
        print(f"检查失败: {str(e)}")


def main():
    # 尝试获取白银数据
    result1 = try_silver_spot_data()
    result2 = try_silver_consumption_data()

    # 检查期货函数
    check_futures_functions()

    # 检查qihuo函数
    try_qihuo_data()

    print("\n" + "="*80)
    print("总结".center(80))
    print("="*80)

    print("\nAKShare数据获取情况:")
    print("1. 白银现货数据: 可以获取（上海黄金交易所）")
    print("2. 白银期货数据: 需要确认正确的接口")
    print("3. 4小时数据: AKShare通常不支持")

    print("\n建议:")
    print("- 现有日线数据已经足够进行策略回测")
    print("- 如需4小时数据，建议:")
    print("  1. 从TradingView手动导出")
    print("  2. 购买专业数据源（Wind、Bloomberg等）")
    print("  3. 使用白银现货数据作为替代（与期货高度相关）")


if __name__ == "__main__":
    main()
