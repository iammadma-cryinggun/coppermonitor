# -*- coding: utf-8 -*-
"""
数据清洗工厂 - 外盘数据标准化

功能：
1. 统一单位（美分转美元）
2. 处理缺失值
3. 计算ATR（替代成交量）
4. 生成标准化的DATA_POOL
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# 数据目录
DATA_DIR = Path('global_futures_daily')

# 映射配置
CONFIG = {
    "ag": {"file": "白银_daily.csv",     "unit_fix": 1,     "source": "COMEX", "name": "沪银"},
    "au": {"file": "黄金_daily.csv",     "unit_fix": 1,     "source": "COMEX", "name": "沪金"},
    "cu": {"file": "铜_LME_daily.csv",   "unit_fix": 1,     "source": "LME",   "name": "沪铜"},
    "al": {"file": "铝_daily.csv",       "unit_fix": 1,     "source": "LME",   "name": "沪铝"},
    "zn": {"file": "锌_daily.csv",       "unit_fix": 1,     "source": "LME",   "name": "沪锌"},
    "sn": {"file": "锡_daily.csv",       "unit_fix": 1,     "source": "LME",   "name": "沪锡"},
    "ni": {"file": "镍_daily.csv",       "unit_fix": 1,     "source": "LME",   "name": "沪镍"},
    "CF": {"file": "棉花_daily.csv",     "unit_fix": 0.01,  "source": "ICE",   "name": "郑棉"},
    "SR": {"file": "糖_daily.csv",       "unit_fix": 1,     "source": "CBOT",  "name": "白糖"},
}


def load_and_clean_data():
    """加载并清洗所有数据"""
    data_pool = {}

    print("=" * 80)
    print("数据清洗工厂 - 外盘数据标准化")
    print("=" * 80)
    print(f"数据源目录: {DATA_DIR.absolute()}")
    print(f"品种数量: {len(CONFIG)}")
    print("=" * 80)

    for code, cfg in CONFIG.items():
        file_path = DATA_DIR / cfg['file']

        if not file_path.exists():
            print(f"[跳过] {code} - 文件不存在: {cfg['file']}")
            continue

        try:
            # 1. 读取原始数据
            df = pd.read_csv(file_path)

            # 2. 统一列名
            df.columns = [c.lower().strip() for c in df.columns]

            # 3. 时间索引标准化
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # 4. 价格单位修正（美分转美元）
            if cfg['unit_fix'] != 1:
                price_cols = ['open', 'high', 'low', 'close']
                df[price_cols] = df[price_cols] * cfg['unit_fix']
                print(f"   [单位修正] {cfg['name']}: 乘以 {cfg['unit_fix']}")

            # 5. 处理缺失的成交量
            if 'volume' not in df.columns or (df['volume'] == 0).all():
                df['volume'] = 0

            # 6. 计算ATR（替代成交量）
            df['prev_close'] = df['close'].shift(1)
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    np.abs(df['high'] - df['prev_close']),
                    np.abs(df['low'] - df['prev_close'])
                )
            )
            df['atr'] = df['tr'].rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR百分比

            # 7. 清理临时列
            df.drop(['prev_close', 'tr'], axis=1, inplace=True)

            # 8. 存入数据池
            data_pool[code] = df

            # 打印状态
            data_count = len(df)
            start_date = df.index[0].strftime('%Y-%m-%d')
            end_date = df.index[-1].strftime('%Y-%m-%d')
            last_price = df['close'].iloc[-1]
            last_atr_pct = df['atr_pct'].iloc[-1]

            print(f"[OK] {code} | {cfg['name']:4s} | {cfg['source']:6s} | "
                  f"{data_count:4d}条 | {start_date} ~ {end_date} | "
                  f"现价:{last_price:8.2f} | ATR:{last_atr_pct:.2f}%")

        except Exception as e:
            print(f"[ERROR] {code}: {e}")

    # 保存数据池到文件
    output_file = Path('data_pool.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(data_pool, f)

    print("=" * 80)
    print(f"数据清洗完成！共 {len(data_pool)} 个品种")
    print(f"数据池已保存: {output_file.absolute()}")
    print("=" * 80)

    # 打印使用说明
    print("\n使用方法:")
    print("  import pickle")
    print("  with open('data_pool.pkl', 'rb') as f:")
    print("      DATA_POOL = pickle.load(f)")
    print("  df_ag = DATA_POOL['ag']  # 访问白银数据")
    print("=" * 80)

    return data_pool


if __name__ == "__main__":
    try:
        DATA_POOL = load_and_clean_data()

        # 显示数据概览
        print("\n数据概览:")
        print(f"{'代码':<6} {'名称':<6} {'数据量':<8} {'起止日期':<24} {'最新价格':<12}")
        print("-" * 80)

        for code, cfg in CONFIG.items():
            if code in DATA_POOL:
                df = DATA_POOL[code]
                start = df.index[0].strftime('%Y-%m-%d')
                end = df.index[-1].strftime('%Y-%m-%d')
                price = df['close'].iloc[-1]

                print(f"{code:<6} {cfg['name']:<6} {len(df):<8d} {start} ~ {end:<11} {price:>10.2f}")

    except Exception as e:
        print(f"\n程序异常: {e}")
        import traceback
        traceback.print_exc()
