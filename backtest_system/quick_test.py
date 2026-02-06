# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证回测系统是否正常工作

功能：
1. 检查数据文件
2. 测试单个品种回测
3. 验证程序完整性
"""

import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
PROGRAMS_DIR = BASE_DIR / 'programs'


def check_data_files():
    """检查数据文件"""
    print("=" * 60)
    print("检查数据文件...")
    print("=" * 60)

    csv_files = list(DATA_DIR.glob('*.csv'))

    if not csv_files:
        print("❌ 未找到数据文件！")
        return False

    print(f"✅ 找到 {len(csv_files)} 个数据文件:\n")

    # 按交易所分组
    futures_by_exchange = {
        'SHFE': [],  # 上期所
        'DCE': [],   # 大商所
        'CZCE': [],  # 郑商所
        'OTHER': []
    }

    for csv_file in sorted(csv_files):
        name = csv_file.stem.replace('_4hour', '').replace('_fixed', '')

        # 简单分组
        if name.startswith('沪') or name in ['螺纹钢', '热卷']:
            futures_by_exchange['SHFE'].append(name)
        elif name in ['PP', 'PVC', '豆粕', '豆油', '棕榈油', '铁矿石', '焦煤', '焦炭']:
            futures_by_exchange['DCE'].append(name)
        elif name in ['PTA', '甲醇', '纯碱', '玻璃', '棉花', '白糖']:
            futures_by_exchange['CZCE'].append(name)
        else:
            futures_by_exchange['OTHER'].append(name)

    # 打印分组结果
    for exchange, futures in futures_by_exchange.items():
        if futures:
            print(f"\n{exchange} ({len(futures)}个):")
            for future in futures:
                print(f"  - {future}")

    print(f"\n数据文件位置: {DATA_DIR}")
    return True


def check_program_files():
    """检查程序文件"""
    print("\n" + "=" * 60)
    print("检查程序文件...")
    print("=" * 60)

    required_programs = [
        'optimize_all_futures.py',
        'verify_optimized_results.py',
        'backtest_all_futures.py',
        'backtest.py'
    ]

    all_exists = True

    for program in required_programs:
        program_path = PROGRAMS_DIR / program

        if program_path.exists():
            print(f"✅ {program}")
        else:
            print(f"❌ {program} - 文件缺失！")
            all_exists = False

    return all_exists


def test_single_future():
    """测试单个品种回测"""
    print("\n" + "=" * 60)
    print("测试单个品种回测...")
    print("=" * 60)

    # 选择沪铜作为测试品种
    copper_file = DATA_DIR / '沪铜_4hour.csv'

    if not copper_file.exists():
        print(f"❌ 测试数据不存在: {copper_file}")
        return False

    try:
        # 读取数据
        df = pd.read_csv(copper_file)

        print(f"✅ 数据文件: {copper_file.name}")
        print(f"   数据量: {len(df)} 条")
        print(f"   时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
        print(f"   列名: {list(df.columns)}")

        # 显示前几行
        print(f"\n前3行数据:")
        print(df.head(3).to_string(index=False))

        return True

    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return False


def main():
    """主函数"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║     期货策略回测系统 - 快速测试                           ║
╚═══════════════════════════════════════════════════════════╝
    """)

    results = []

    # 1. 检查数据文件
    results.append(("数据文件检查", check_data_files()))

    # 2. 检查程序文件
    results.append(("程序文件检查", check_program_files()))

    # 3. 测试单个品种
    results.append(("单品种回测测试", test_single_future()))

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    all_passed = True

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

        if not passed:
            all_passed = False

    print("\n" + "=" * 60)

    if all_passed:
        print("✅ 所有测试通过！系统可以正常使用。")
        print("\n下一步:")
        print("  运行完整回测: python run_backtest.py")
        print("  或直接优化: python programs/optimize_all_futures.py")
    else:
        print("❌ 部分测试失败，请检查上述错误。")

    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
