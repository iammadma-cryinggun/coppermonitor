@echo off
chcp 65001 >nul
title CCI策略回测系统

echo ================================================================================
echo                  CCI策略回测系统 - 启动器
echo ================================================================================
echo.

cd /d "%~dp0"

echo 当前目录: %CD%
echo.
echo 数据目录: %CD%\..
echo.

echo 检查数据文件...
if exist "..\data_pool.pkl" (
    echo     [OK] data_pool.pkl
) else (
    echo     [缺失] data_pool.pkl
)

if exist "..\global_futures_daily\白银_daily.csv" (
    echo     [OK] 白银_daily.csv
) else (
    echo     [缺失] 白银_daily.csv
)

if exist "..\global_futures_daily\黄金_daily.csv" (
    echo     [OK] 黄金_daily.csv
) else (
    echo     [缺失] 黄金_daily.csv
)

echo.
echo ================================================================================
echo                      启动主程序...
echo ================================================================================
echo.

python 主程序.py
if errorlevel 1 (
    echo.
    echo [错误] 程序运行出错
    echo.
    pause
)
