# -*- coding: utf-8 -*-
"""
纯碱SA0 - 调试：查看指标实际范围
"""
import pandas as pd
import numpy as np
from datetime import datetime

SYMBOL = 'sa0'

def get_data():
    import akshare as ak
    df = ak.futures_main_sina(symbol=SYMBOL)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.dropna()
    df.set_index('date', inplace=True)
    return df


def calc_coo(df, rsi_period=14, stoch_period=14):
    """计算COO"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    low_14 = df['low'].rolling(window=stoch_period).min()
    high_14 = df['high'].rolling(window=stoch_period).max()
    stoch = 100 * (df['close'] - low_14) / (high_14 - low_14)

    coo = (rsi + stoch) / 2 - 50
    return coo


def calc_stc(df, fast_period=23, slow_period=50, cycle_period=10):
    """计算STC"""
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    lowest_macd = macd.rolling(window=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period).max()
    range_macd = highest_macd - lowest_macd

    stc = pd.Series(index=df.index, dtype=float)
    stc.iloc[:cycle_period] = 0

    for i in range(cycle_period, len(macd)):
        if range_macd.iloc[i] != 0:
            stc.iloc[i] = ((macd.iloc[i] - lowest_macd.iloc[i]) / range_macd.iloc[i] - 0.5) * 200
        else:
            stc.iloc[i] = stc.iloc[i-1] if i > cycle_period else 0

    stc_normalized = pd.Series(index=stc.index, dtype=float)
    lookback = 20
    for i in range(len(stc)):
        if i < lookback:
            stc_normalized.iloc[i] = stc.iloc[i]
        else:
            window = stc.iloc[i-lookback+1:i+1]
            low = window.min()
            high = window.max()
            if high != low:
                stc_normalized.iloc[i] = (stc.iloc[i] - low) / (high - low) * 100 - 50
            else:
                stc_normalized.iloc[i] = stc_normalized.iloc[i-1] if i > 0 else 0

    return stc_normalized


def main():
    print("=" * 100)
    print("纯碱SA0 - 指标范围调试".center(100))
    print("=" * 100)

    df = get_data()
    print(f"\n数据量: {len(df)}条")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")

    print("\n计算指标...")
    df['coo'] = calc_coo(df)
    df['stc'] = calc_stc(df)

    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=12).mean()
    mad = tp.rolling(window=12).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=5).mean()

    # StdDev
    price_change = df['close'].pct_change()
    df['stddev'] = price_change.rolling(window=20).std() * 100

    print("\n" + "=" * 100)
    print("指标统计范围")
    print("=" * 100)

    for col in ['coo', 'stc', 'cci', 'cci_ma', 'stddev']:
        if col in df.columns:
            data = df[col].dropna()
            print(f"\n{col.upper()}:")
            print(f"  最小值: {data.min():>10.2f}")
            print(f"  最大值: {data.max():>10.2f}")
            print(f"  平均值: {data.mean():>10.2f}")
            print(f"  中位数: {data.median():>10.2f}")
            print(f"  标准差: {data.std():>10.2f}")

            # 分位数
            print(f"  1%分位: {data.quantile(0.01):>10.2f}")
            print(f"  5%分位: {data.quantile(0.05):>10.2f}")
            print(f"  95%分位: {data.quantile(0.95):>10.2f}")
            print(f"  99%分位: {data.quantile(0.99):>10.2f}")

    # 检查有多少数据点符合条件
    print("\n" + "=" * 100)
    print("开仓条件覆盖范围")
    print("=" * 100)

    valid_data = df.dropna()
    print(f"\n有效数据点: {len(valid_data)}")

    coo_cond = valid_data['coo'] < -40
    stc_cond = valid_data['stc'] > 10
    cci_cond = (valid_data['cci'] > -120) & (valid_data['cci'] > valid_data['cci_ma'])
    stddev_cond = valid_data['stddev'] > 1.5

    print(f"COO < -40: {coo_cond.sum()} ({coo_cond.sum()/len(valid_data)*100:.2f}%)")
    print(f"STC > 10: {stc_cond.sum()} ({stc_cond.sum()/len(valid_data)*100:.2f}%)")
    print(f"CCI > -120 且 CCI > CCI_MA: {cci_cond.sum()} ({cci_cond.sum()/len(valid_data)*100:.2f}%)")
    print(f"StdDev > 1.5: {stddev_cond.sum()} ({stddev_cond.sum()/len(valid_data)*100:.2f}%)")

    all_cond = coo_cond & stc_cond & cci_cond & stddev_cond
    print(f"\n同时满足所有条件: {all_cond.sum()} ({all_cond.sum()/len(valid_data)*100:.4f}%)")

    # 检查两两交集
    print(f"\nCOO & STC: {(coo_cond & stc_cond).sum()}")
    print(f"COO & CCI: {(coo_cond & cci_cond).sum()}")
    print(f"STC & CCI: {(stc_cond & cci_cond).sum()}")
    print(f"COO & STC & CCI: {(coo_cond & stc_cond & cci_cond).sum()}")

    if all_cond.sum() > 0:
        print("\n满足条件的数据点:")
        print(valid_data[all_cond][['coo', 'stc', 'cci', 'cci_ma']].head(10))

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
