# -*- coding: utf-8 -*-
"""测试回测速度"""
import pandas as pd
import numpy as np
from china_futures_fetcher import ChinaFuturesFetcher
from batch_optimize_top10_short import calculate_indicators, backtest_short_only
import time

# 获取PTA数据
fetcher = ChinaFuturesFetcher()
df = fetcher.get_historical_data('TA', days=300)

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"数据量: {len(df)}条")

# 测试一组参数
test_params = {
    'EMA_FAST': 7,
    'EMA_SLOW': 20,
    'RSI_FILTER': 45,
    'RATIO_TRIGGER': 1.05,
    'STC_BUY_ZONE': 15
}

print(f"\n测试参数: {test_params}")
print("开始回测...")

start = time.time()
result = backtest_short_only(df, test_params, 5, 0.08)
elapsed = time.time() - start

print(f"\n完成! 耗时: {elapsed:.2f}秒")
print(f"结果: {result}")

if result:
    print(f"\n估算总时间:")
    print(f"  单个参数: {elapsed:.2f}秒")
    print(f"  3450个参数: {elapsed * 3450 / 60:.1f}分钟")
