# -*- coding: utf-8 -*-
"""
只测试白糖 - 用retest_all_copper_params.py的方式
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'D:\\期货数据\\铜期货监控')

# 导入retest_all_copper_params.py的函数
from retest_all_copper_params import calculate_indicators, run_backtest

# 加载白糖数据
df = pd.read_csv('futures_data_4h/白糖_4hour.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

print("测试白糖 - 使用retest_all_copper_params.py的函数")
print("="*80)
print(f"数据量: {len(df)}条")
print(f"时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
print()

# 运行3次
for i in range(1, 4):
    stats = run_backtest(df.copy(), '白糖')

    if stats is None:
        print(f"第{i}次: 无交易记录")
    else:
        print(f"第{i}次: 收益 {stats['return_pct']:+.2f}%, 交易{stats['total_trades']}笔, 胜率{stats['win_rate']:.1f}%, 最终资金 {stats['final_capital']:,.0f}")
