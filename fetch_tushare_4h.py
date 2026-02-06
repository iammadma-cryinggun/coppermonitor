# -*- coding: utf-8 -*-
"""
使用 Tushare 获取沪铜4小时K线数据
需要注册并获取 Token: https://tushare.pro/register
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# ==========================================
# 配置
# ==========================================
TUSHARE_TOKEN = 'd6eb370c3f076358c37cdc7e5b9ac9a48b01bcfe5b7bc0d2adfc9b36'  # Tushare Token
START_DATE = '2024-01-01 09:00:00'  # 开始日期
END_DATE = None  # None 表示当前时间

# ==========================================
# 初始化 Tushare
# ==========================================
def init_tushare():
    """初始化 Tushare API"""
    if TUSHARE_TOKEN == '你的Token':
        print('='*80)
        print('错误: 请先设置 TUSHARE_TOKEN')
        print('='*80)
        print('\n步骤:')
        print('1. 注册: https://tushare.pro/register')
        print('2. 获取 Token: https://tushare.pro/user/token')
        print('3. 修改本文件第15行的 TUSHARE_TOKEN 变量')
        print('\n按回车退出...')
        input()
        exit(1)

    return ts.pro_api(TUSHARE_TOKEN)

# ==========================================
# 获取主力合约
# ==========================================
def get_main_contract(pro):
    """获取沪铜主力合约"""
    try:
        # 获取沪铜合约列表
        df = pro.fut_basic(
            exchange='SHF',
            fut_type='1',  # 1=普通期货
            fields='ts_code,symbol,name,list_date,delist_date'
        )

        # 筛选铜合约
        cu_contracts = df[df['symbol'].str.contains('CU', na=False)]

        if cu_contracts.empty:
            print('错误: 未找到铜合约')
            return None

        # 获取最新合约（假设第一个是主力）
        main_contract = cu_contracts.iloc[0]['ts_code']
        print(f'主力合约: {main_contract}')
        return main_contract

    except Exception as e:
        print(f'获取主力合约失败: {e}')
        return None

# ==========================================
# 获取分钟数据
# ==========================================
def fetch_minutes_data(pro, ts_code, start_date, end_date, freq='60min'):
    """
    分批获取分钟数据

    Args:
        pro: Tushare API
        ts_code: 合约代码
        start_date: 开始日期
        end_date: 结束日期
        freq: 频率 (1min, 5min, 15min, 30min, 60min)

    Returns:
        DataFrame
    """
    print(f'开始获取 {ts_code} 的 {freq} 数据...')
    print(f'时间范围: {start_date} ~ {end_date or "现在"}')

    all_data = []
    current_start = pd.to_datetime(start_date)
    current_end = pd.to_datetime(end_date) if end_date else pd.to_datetime.now()

    # 每次最多获取3个月数据
    batch_days = 90

    while current_start < current_end:
        batch_end = min(current_start + timedelta(days=batch_days), current_end)

        start_str = current_start.strftime('%Y-%m-%d %H:%M:%S')
        end_str = batch_end.strftime('%Y-%m-%d %H:%M:%S')

        try:
            df = pro.ft_mins(
                ts_code=ts_code,
                freq=freq,
                start_date=start_str,
                end_date=end_str
            )

            if df is not None and not df.empty:
                all_data.append(df)
                print(f'  ✓ {start_str} ~ {end_str}: {len(df)} 条')
            else:
                print(f'  × {start_str} ~ {end_str}: 无数据')

            # API 限流：每分钟200次
            time.sleep(0.5)

        except Exception as e:
            print(f'  ✗ {start_str} ~ {end_str}: {e}')

        current_start = batch_end + timedelta(days=1)

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values('trade_time').reset_index(drop=True)
        return result
    else:
        return None

# ==========================================
# 重采样成4小时K线
# ==========================================
def resample_to_4h(df):
    """将60分钟数据重采样成4小时K线"""
    print('\n重采样成4小时K线...')

    # 转换时间列
    df['trade_time'] = pd.to_datetime(df['trade_time'])
    df = df.set_index('trade_time')

    # 重采样
    df_4h = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'amount': 'sum'
    }).dropna()

    df_4h = df_4h.reset_index()
    return df_4h

# ==========================================
# 标准化数据格式
# ==========================================
def standardize_data(df):
    """标准化数据格式（与现有代码兼容）"""
    df = df.rename(columns={
        'trade_time': 'datetime',
        'vol': 'volume'
    })

    # 确保列名统一
    required_cols = ['datetime', 'open', 'high', 'low', 'close']
    df = df[required_cols + [col for col in df.columns if col not in required_cols]]

    return df

# ==========================================
# 主程序
# ==========================================
def main():
    print('='*80)
    print('使用 Tushare 获取沪铜4小时K线数据')
    print('='*80)

    # 初始化 API
    pro = init_tushare()

    # 获取主力合约
    main_contract = get_main_contract(pro)
    if not main_contract:
        return

    # 获取60分钟数据
    df_60m = fetch_minutes_data(
        pro,
        ts_code=main_contract,
        start_date=START_DATE,
        end_date=END_DATE,
        freq='60min'
    )

    if df_60m is None or df_60m.empty:
        print('\n错误: 未获取到数据')
        return

    print(f'\n原始数据: {len(df_60m)} 条60分钟K线')

    # 重采样成4小时K线
    df_4h = resample_to_4h(df_60m)

    # 标准化格式
    df_4h = standardize_data(df_4h)

    print(f'4小时K线: {len(df_4h)} 条')
    print(f'时间范围: {df_4h["datetime"].iloc[0]} ~ {df_4h["datetime"].iloc[-1]}')

    # 保存数据
    output_file = '沪铜4小时K线_Tushare.csv'
    df_4h.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f'\n✓ 数据已保存: {output_file}')
    print(f'  数据量: {len(df_4h)} 条')
    print(f'  时间跨度: {(df_4h["datetime"].iloc[-1] - df_4h["datetime"].iloc[0]).days} 天')

if __name__ == '__main__':
    main()
