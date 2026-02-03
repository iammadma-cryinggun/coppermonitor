# -*- coding: utf-8 -*-
"""
持续运行监控（Zeabur专用）
让监控脚本每4小时自动运行一次
"""
import time
import logging
import sys
from pathlib import Path
from datetime import datetime

# 导入监控模块
try:
    import copper_monitor
except ImportError:
    print("错误: 无法导入 copper_monitor 模块")
    print("请确保 copper_monitor.py 在同一目录下")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 运行间隔（秒）
RUN_INTERVAL = 4 * 60 * 60  # 4小时


def run_once():
    """运行一次监控"""
    try:
        logger.info("=" * 80)
        logger.info(f"开始监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

        # 调用主监控脚本
        copper_monitor.run_monitoring()

        logger.info("\n" + "=" * 80)
        logger.info("本次监控完成")
        logger.info("=" * 80)
        return True

    except Exception as e:
        logger.error(f"监控运行出错: {e}", exc_info=True)
        return False


def run_continuous():
    """持续运行监控"""
    logger.info("=" * 80)
    logger.info("沪铜期货持续监控系统启动")
    logger.info("=" * 80)
    logger.info(f"运行间隔: {RUN_INTERVAL / 3600} 小时")
    logger.info(f"首次运行: 立即开始")
    logger.info("=" * 80)

    run_count = 0

    while True:
        run_count += 1

        try:
            # 运行监控
            success = run_once()

            if success:
                logger.info(f"\n[统计] 已完成 {run_count} 次监控")
            else:
                logger.warning(f"\n[警告] 第 {run_count} 次监控运行出错")

            # 计算下次运行时间
            now = datetime.now()
            next_run = now.timestamp() + RUN_INTERVAL
            next_run_dt = datetime.fromtimestamp(next_run)

            logger.info(f"\n{'='*80}")
            logger.info(f"等待下次运行...")
            logger.info(f"下次运行时间: {next_run_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*80}\n")

            # 等待指定时间
            time.sleep(RUN_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("收到中断信号，准备退出...")
            logger.info("=" * 80)
            logger.info(f"\n[统计] 总共运行了 {run_count} 次监控")
            logger.info("监控系统已停止")
            break

        except Exception as e:
            logger.error(f"\n[错误] 主循环异常: {e}", exc_info=True)
            logger.info("等待60秒后重试...")
            time.sleep(60)


if __name__ == "__main__":
    # 首次运行（立即执行一次）
    logger.info("首次运行开始...")
    run_once()

    # 进入持续运行模式
    logger.info("\n进入持续运行模式...")
    run_continuous()
