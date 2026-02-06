#!/bin/bash
# Zeabur Cron Job模式 - 运行一次后退出

echo "开始执行监控任务..."
python futures_monitor.py
echo "监控任务完成，退出"
