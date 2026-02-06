# -*- coding: utf-8 -*-
"""
下载TOP7品种的日线数据（3年历史）
用于策略回测
"""

import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os

# TOP7品种配置
TOP7_FUTURES = {
    'NI': {'name': '沪镍', 'sina_code': 'NI0'},
    'SA': {'name': '纯碱', 'sina_code': 'SA0'},
    'V': {'name': 'PVC', 'sina_code': 'V0'},
    'CU': {'name': '沪铜', 'sina_code': 'CU0'},
    'SN': {'name': '沪锡', 'sina_code': 'SN0'},
    'PB': {'name': '沪铅', 'sina_code': 'PB0'},
    'FG': {'name': '玻璃', 'sina_code': 'FG0'},
}

# 输出目录
OUTPUT_DIR = 'daily_backtest'

def download_daily_data(code, info):
    """下载单个品种的日线数据"""
    name = info['name']
    sina_code = info['sina_code']

    print(f"\n{'='*60}")
    print(f"下载 {name} ({code}) 日线数据...")
    print('='*60)

    try:
        # 使用akshare获取主力合约日线数据
        df = ak.futures_main_sina(symbol=sina_code)

        if df is None or df.empty:
            print(f"  [FAIL] 数据为空")
            return None

        # 标准化列名
        df = df.rename(columns={
            '日期': 'datetime',
            '开盘价': 'open',
            '最高价': 'high',
            '最低价': 'low',
            '收盘价': 'close',
            '成交量': 'volume',
            '持仓量': 'hold',
            '动态结算价': 'settlement',
            '涨跌': 'change',
            '涨跌幅': 'pct_change'
        })

        # 转换日期
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')

        # 过滤最近3年数据
        three_years_ago = datetime.now() - timedelta(days=3*365)
        df = df[df['datetime'] >= three_years_ago]

        # 保存文件
        filename = f"{OUTPUT_DIR}/{code}_{name}_日线.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')

        print(f"  [OK] 数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
        print(f"  [OK] 总记录数: {len(df)} 条")
        print(f"  [OK] 已保存: {filename}")

        return df

    except Exception as e:
        print(f"  [FAIL] 下载失败: {e}")
        return None


def main():
    """主函数"""
    print("="*80)
    print("下载TOP7品种日线数据（3年历史）")
    print("="*80)
    print(f"保存目录: {OUTPUT_DIR}")
    print(f"下载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"\n[OK] 创建目录: {OUTPUT_DIR}")
    else:
        print(f"\n[OK] 目录已存在: {OUTPUT_DIR}")

    # 下载各品种数据
    results = {}
    for code, info in TOP7_FUTURES.items():
        df = download_daily_data(code, info)
        if df is not None:
            results[code] = {
                'name': info['name'],
                'records': len(df),
                'start_date': df['datetime'].iloc[0],
                'end_date': df['datetime'].iloc[-1]
            }

    # 总结
    print("\n" + "="*80)
    print("下载完成总结")
    print("="*80)
    print(f"\n{'品种':<8} {'代码':<6} {'记录数':<10} {'起始日期':<12} {'结束日期':<12}")
    print('-'*80)

    for code, data in results.items():
        print(f"{data['name']:<8} {code:<6} {data['records']:<10} "
              f"{str(data['start_date'].date()):<12} {str(data['end_date'].date()):<12}")

    print(f"\n总计: {len(results)}/{len(TOP7_FUTURES)} 个品种下载成功")
    print("="*80)


if __name__ == "__main__":
    main()
