#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zeabur部署检查脚本
用于验证部署环境是否正确配置
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """检查环境变量配置"""
    logger.info("=" * 60)
    logger.info("Zeabur部署环境检查")
    logger.info("=" * 60)

    # 检查Telegram配置
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    logger.info("\n1. 检查Telegram配置:")
    if token:
        logger.info(f"   ✓ TELEGRAM_BOT_TOKEN: {token[:10]}...{token[-4:]}")
    else:
        logger.warning("   ✗ TELEGRAM_BOT_TOKEN 未配置")

    if chat_id:
        logger.info(f"   ✓ TELEGRAM_CHAT_ID: {chat_id}")
    else:
        logger.warning("   ✗ TELEGRAM_CHAT_ID 未配置")

    # 检查依赖
    logger.info("\n2. 检查Python依赖:")
    try:
        import pandas
        logger.info(f"   ✓ pandas {pandas.__version__}")
    except ImportError:
        logger.warning("   ✗ pandas 未安装")

    try:
        import numpy
        logger.info(f"   ✓ numpy {numpy.__version__}")
    except ImportError:
        logger.warning("   ✗ numpy 未安装")

    try:
        import requests
        logger.info(f"   ✓ requests {requests.__version__}")
    except ImportError:
        logger.warning("   ✗ requests 未安装")

    try:
        import akshare
        logger.info(f"   ✓ akshare {akshare.__version__}")
    except ImportError:
        logger.warning("   ✗ akshare 未安装")

    # 检查模块文件
    logger.info("\n3. 检查核心模块:")
    modules = ['china_futures_fetcher', 'notifier', 'futures_monitor']
    for module in modules:
        try:
            __import__(module)
            logger.info(f"   ✓ {module}.py")
        except ImportError as e:
            logger.error(f"   ✗ {module}.py: {e}")

    # 总结
    logger.info("\n" + "=" * 60)
    if token and chat_id:
        logger.info("✓ Telegram配置正确")
        logger.info("\n部署环境检查完成！可以启动监控系统。")
        return True
    else:
        logger.error("✗ 缺少必要的环境变量配置")
        logger.error("\n请在Zeabur中配置:")
        logger.error("  - TELEGRAM_BOT_TOKEN")
        logger.error("  - TELEGRAM_CHAT_ID")
        return False

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
