import akshare as ak
import pandas as pd

# 获取纯碱2020-2025历史数据
print("正在下载纯碱SA0 2020-2025历史数据...")

df = ak.futures_main_sina(symbol='sa0')
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.dropna()

# 过滤2020-2025
start_date = pd.to_datetime('2020-01-01')
end_date = pd.to_datetime('2025-12-31')

df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

print(f"下载完成！数据范围: {df_filtered['date'].iloc[0]} 至 {df_filtered['date'].iloc[-1]}")
print(f"数据量: {len(df_filtered)}条")

# 保存
df_filtered.to_csv('sa0_2020-2025.csv', index=False)

print("数据已保存至: sa0_2020-2025.csv")
