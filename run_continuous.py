# -*- coding: utf-8 -*-
"""
持续运行监控（Zeabur专用）
严格按照4小时K线时间运行：00:00, 04:00, 08:00, 12:00, 16:00, 20:00
"""
import time
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

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

# 4小时K线的整点时间（小时）
SCHEDULE_HOURS = [0, 4, 8, 12, 16, 20]


def get_wait_seconds():
    """
    计算距离下一个4小时K线整点的等待秒数

    Returns:
        int: 等待秒数
    """
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    current_second = now.second

    # 计算当前时间到下一个整点小时的秒数
    seconds_to_next_hour = (59 - current_minute) * 60 + (59 - current_second) + 1

    # 找到下一个 scheduled hour
    next_hour = None
    for hour in SCHEDULE_HOURS:
        if hour > current_hour:
            next_hour = hour
            break

    # 如果今天没有找到，使用明天的第一个
    if next_hour is None:
        next_hour = SCHEDULE_HOURS[0]  # 0点
        hours_to_wait = (24 - current_hour) + next_hour
    else:
        hours_to_wait = next_hour - current_hour

    # 总等待时间 = 小时差 + 秒数差
    total_wait_seconds = hours_to_wait * 3600 + seconds_to_next_hour

    return total_wait_seconds, datetime.now() + timedelta(seconds=total_wait_seconds)


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
    """持续运行监控（严格按照4小时K线整点时间）"""
    logger.info("=" * 80)
    logger.info("沪铜期货持续监控系统启动")
    logger.info("=" * 80)
    logger.info(f"运行时间点: {', '.join([f'{h:02d}:00' for h in SCHEDULE_HOURS])}")
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

            # 计算到下一个4小时整点的等待时间
            wait_seconds, next_run_dt = get_wait_seconds()

            logger.info(f"\n{'='*80}")
            logger.info(f"等待下次运行...")
            logger.info(f"下次运行时间: {next_run_dt.strftime('%Y-%m-%d %H:%M:%S')} (整点)")
            wait_hours = wait_seconds / 3600
            logger.info(f"等待时长: {wait_hours:.2f} 小时 ({wait_seconds} 秒)")
            logger.info(f"{'='*80}\n")

            # 等待到下一个整点
            time.sleep(wait_seconds)

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
