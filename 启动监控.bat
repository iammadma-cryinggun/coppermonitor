@echo off
chcp 65001 >nul
echo ========================================
echo 期货多品种监控系统
echo ========================================
echo.
echo 选择运行模式:
echo 1. 单次运行（立即执行一次监控）
echo 2. 定时运行（每4小时自动运行）
echo 3. 退出
echo.
set /p choice=请选择 (1-3):

if "%choice%"=="1" (
    echo.
    echo [启动] 单次运行模式...
    python futures_monitor.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo [启动] 定时运行模式...
    echo.
    echo 运行时间点: 0:00, 4:00, 8:00, 12:00, 16:00, 20:00
    echo 按Ctrl+C可以停止监控
    echo.
    pause
    python futures_monitor.py --scheduled
) else (
    echo.
    echo 退出
    exit
)
