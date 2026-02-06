# -*- coding: utf-8 -*-
import pandas as pd
import json

df = pd.read_csv("SN_lppl_indicators.csv")
print(f"总数据: {len(df)} 天")
print(f"列名: {df.columns.tolist()}")
print(f"\n前5行数据:")
print(df.head())
