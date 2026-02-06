# -*- coding: utf-8 -*-
"""
期货策略完整回测验证系统 - 主程序入口

功能：
1. 参数优化
2. 结果验证
3. 对比分析
4. 生成报告
"""

import sys
import subprocess
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/backtest_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
PROGRAMS_DIR = BASE_DIR / 'programs'
RESULTS_DIR = BASE_DIR / 'results'


def print_banner():
    """打印横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║     期货策略完整回测验证系统 v1.0                         ║
║     Futures Strategy Backtest & Optimization System      ║
╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """打印菜单"""
    menu = """
请选择要执行的操作：

[1] 完整验证流程（优化 → 验证 → 报告）
[2] 仅参数优化
[3] 仅结果验证
[4] 单品种回测
[5] 批量回测（统一参数）
[6] 查看历史结果
[0] 退出

═══════════════════════════════════════════════════════════
    """
    print(menu)


def run_optimization():
    """运行参数优化"""
    logger.info("=" * 60)
    logger.info("开始参数优化...")
    logger.info("=" * 60)

    program = PROGRAMS_DIR / 'optimize_all_futures.py'

    if not program.exists():
        logger.error(f"程序不存在: {program}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(program)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        print(result.stdout)
        if result.stderr:
            logger.error(result.stderr)

        return result.returncode == 0

    except Exception as e:
        logger.error(f"运行失败: {e}")
        return False


def run_verification():
    """运行结果验证"""
    logger.info("=" * 60)
    logger.info("开始结果验证...")
    logger.info("=" * 60)

    # 查找验证程序
    verify_programs = [
        'verify_optimized_results.py',
        'verify_all_optimizations.py'
    ]

    for program_name in verify_programs:
        program = PROGRAMS_DIR / program_name

        if program.exists():
            try:
                result = subprocess.run(
                    [sys.executable, str(program)],
                    cwd=BASE_DIR,
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )

                print(result.stdout)
                if result.stderr:
                    logger.error(result.stderr)

            except Exception as e:
                logger.error(f"运行 {program_name} 失败: {e}")
        else:
            logger.warning(f"程序不存在: {program}")

    return True


def run_backtest_all():
    """运行批量回测"""
    logger.info("=" * 60)
    logger.info("开始批量回测...")
    logger.info("=" * 60)

    program = PROGRAMS_DIR / 'backtest_all_futures.py'

    if not program.exists():
        logger.error(f"程序不存在: {program}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(program)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        print(result.stdout)
        if result.stderr:
            logger.error(result.stderr)

        return result.returncode == 0

    except Exception as e:
        logger.error(f"运行失败: {e}")
        return False


def show_results():
    """显示历史结果"""
    logger.info("=" * 60)
    logger.info("历史结果")
    logger.info("=" * 60)

    results_dir = RESULTS_DIR

    if not results_dir.exists():
        logger.info("暂无结果文件")
        return

    # 列出所有CSV文件
    csv_files = sorted(results_dir.glob('*.csv'))

    if not csv_files:
        logger.info("暂无CSV结果文件")
        return

    print(f"\n找到 {len(csv_files)} 个结果文件：\n")

    for csv_file in csv_files:
        print(f"  📊 {csv_file.name}")

    # 显示汇总文件
    summary_file = results_dir / 'all_futures_summary.csv'
    if summary_file.exists():
        print(f"\n═══════════════════════════════════════════════════════════")
        print("所有品种汇总结果:")
        print("═══════════════════════════════════════════════════════════")

        try:
            import pandas as pd
            df = pd.read_csv(summary_file)
            print(df.to_string(index=False))
        except Exception as e:
            logger.error(f"读取汇总文件失败: {e}")


def main():
    """主函数"""
    print_banner()

    # 确保必要目录存在
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / 'logs').mkdir(parents=True, exist_ok=True)

    while True:
        print_menu()
        choice = input("请输入选项 [0-6]: ").strip()

        if choice == '0':
            print("\n感谢使用！")
            break

        elif choice == '1':
            # 完整验证流程
            logger.info("开始完整验证流程...")
            run_optimization()
            run_verification()
            show_results()

        elif choice == '2':
            # 仅参数优化
            run_optimization()

        elif choice == '3':
            # 仅结果验证
            run_verification()

        elif choice == '4':
            # 单品种回测
            logger.info("单品种回测功能开发中...")
            logger.info("请直接运行: python programs/backtest.py")

        elif choice == '5':
            # 批量回测
            run_backtest_all()

        elif choice == '6':
            # 查看结果
            show_results()

        else:
            print("\n❌ 无效选项，请重新选择")

        input("\n按回车键继续...")

    logger.info("程序结束")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序异常退出: {e}")
        sys.exit(1)
