# -*- coding: utf-8 -*-
"""
下载中国期货前20名品种4小时K线数据
数据源: AkShare
"""

import akshare as ak
import pandas as pd
import time
from pathlib import Path

# 中国期货市场前20名主流品种
FUTURES_LIST = [
    # 有色金属
    {'symbol': 'CU0', 'name': '沪铜', 'exchange': 'SHFE'},
    {'symbol': 'AL0', 'name': '沪铝', 'exchange': 'SHFE'},
    {'symbol': 'ZN0', 'name': '沪锌', 'exchange': 'SHFE'},
    {'symbol': 'NI0', 'name': '沪镍', 'exchange': 'SHFE'},
    {'symbol': 'SN0', 'name': '沪锡', 'exchange': 'SHFE'},
    {'symbol': 'PB0', 'name': '沪铅', 'exchange': 'SHFE'},

    # 黑色系
    {'symbol': 'RB0', 'name': '螺纹钢', 'exchange': 'SHFE'},
    {'symbol': 'HC0', 'name': '热卷', 'exchange': 'SHFE'},
    {'symbol': 'I0', 'name': '铁矿石', 'exchange': 'DCE'},
    {'symbol': 'JM0', 'name': '焦煤', 'exchange': 'DCE'},
    {'symbol': 'J0', 'name': '焦炭', 'exchange': 'DCE'},

    # 化工
    {'symbol': 'TA0', 'name': 'PTA', 'exchange': 'CZCE'},
    {'symbol': 'MA0', 'name': '甲醇', 'exchange': 'CZCE'},
    {'symbol': 'PP0', 'name': 'PP', 'exchange': 'DCE'},
    {'symbol': 'V0', 'name': 'PVC', 'exchange': 'DCE'},
    {'symbol': 'SA0', 'name': '纯碱', 'exchange': 'CZCE'},
    {'symbol': 'FG0', 'name': '玻璃', 'exchange': 'CZCE'},

    # 农产品
    {'symbol': 'M0', 'name': '豆粕', 'exchange': 'DCE'},
    {'symbol': 'Y0', 'name': '豆油', 'exchange': 'DCE'},
    {'symbol': 'P0', 'name': '棕榈油', 'exchange': 'DCE'},
    {'symbol': 'CF0', 'name': '棉花', 'exchange': 'CZCE'},
    {'symbol': 'SR0', 'name': '白糖', 'exchange': 'CZCE'},
]

print("=" * 70)
print(f"正在下载 {len(FUTURES_LIST)} 个期货品种的4小时K线数据")
print("=" * 70)

data_dir = Path('futures_data_4h')
data_dir.mkdir(exist_ok=True)

success_count = 0
failed_list = []

for i, future in enumerate(FUTURES_LIST, 1):
    symbol = future['symbol']
    name = future['name']

    print(f"\n[{i}/{len(FUTURES_LIST)}] 下载 {name} ({symbol})...")

    try:
        # 获取60分钟数据
        df_60m = ak.futures_zh_minute_sina(symbol=symbol, period='60')

        if df_60m is None or df_60m.empty:
            print(f"  [FAIL] 无数据")
            failed_list.append({'symbol': symbol, 'name': name, 'reason': '无数据'})
            continue

        # 转换时间列
        df_60m['datetime'] = pd.to_datetime(df_60m['datetime'])
        df_60m = df_60m.set_index('datetime')

        # 重采样成4小时K线
        df_4h = df_60m.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        df_4h = df_4h.reset_index()

        # 过滤最近1年数据
        one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
        df_4h = df_4h[df_4h['datetime'] >= one_year_ago].copy()

        if df_4h.empty:
            print(f"  [FAIL] 过滤后无数据")
            failed_list.append({'symbol': symbol, 'name': name, 'reason': '过滤后无数据'})
            continue

        # 保存
        filename = f"{name}_4hour.csv"
        filepath = data_dir / filename
        df_4h.to_csv(filepath, index=False, encoding='utf-8-sig')

        print(f"  [OK] 成功: {len(df_4h)} 条数据")
        print(f"    时间范围: {df_4h['datetime'].iloc[0]} ~ {df_4h['datetime'].iloc[-1]}")
        print(f"    最新价格: {df_4h['close'].iloc[-1]:.2f}")

        success_count += 1

        # 避免请求过快
        time.sleep(1)

    except Exception as e:
        print(f"  [FAIL] 失败: {e}")
        failed_list.append({'symbol': symbol, 'name': name, 'reason': str(e)})

# 总结
print("\n" + "=" * 70)
print("下载完成")
print("=" * 70)
print(f"\n成功: {success_count}/{len(FUTURES_LIST)}")

if failed_list:
    print(f"\n失败列表:")
    for item in failed_list:
        print(f"  - {item['name']} ({item['symbol']}): {item['reason']}")

print(f"\n数据已保存到: {data_dir.absolute()}")
print("\n文件列表:")
for filepath in sorted(data_dir.glob('*.csv')):
    print(f"  - {filepath.name}")
