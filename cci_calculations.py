"""
CCI计算函数 - TradingView兼容版本
==================================
使用与TradingView ta.cci()一致的计算方法
"""
import pandas as pd
import numpy as np


def calculate_cci_tv(df, cci_length=22, ma_length=13, source='hlc3'):
    """
    计算CCI和CCI_MA - 与TradingView兼容

    参数：
        df: DataFrame with columns ['high', 'low', 'close']
        cci_length: CCI周期长度
        ma_length: CCI均线周期长度
        source: 数据源 ('hlc3' = (high+low+close)/3, 与TradingView默认一致)

    返回：
        DataFrame with added columns: 'cci', 'cci_ma'
    """
    data = df.copy()

    # TradingView使用hlc3作为数据源
    if source == 'hlc3':
        data['hlc3'] = (data['high'] + data['low'] + data['close']) / 3
        src = data['hlc3']
    else:
        raise ValueError(f"Unsupported source: {source}")

    # 计算CCI - 使用TradingView兼容的Mean Absolute Deviation
    tp_sma = src.rolling(window=cci_length).mean()

    # 关键：使用NumPy的mean absolute deviation（与TradingView更接近）
    mad = src.rolling(window=cci_length).apply(
        lambda x: np.abs(x - x.mean()).mean(),
        raw=False
    )

    data['cci'] = (src - tp_sma) / (0.015 * mad)
    data['cci'] = data['cci'].fillna(0)

    # 计算CCI的均线
    data['cci_ma'] = data['cci'].rolling(window=ma_length).mean()

    return data


def calculate_stc(df, length=10, fast=23, slow=50, aaa=0.5):
    """
    计算STC指标
    """
    data = df.copy()
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow

    lowest_macd = macd.rolling(window=length).min()
    highest_macd = macd.rolling(window=length).max()
    _denom = (highest_macd - lowest_macd).replace(0, float('nan'))
    k1 = (100 * (macd - lowest_macd) / _denom).fillna(0)

    d1 = k1.ewm(span=3, adjust=False).mean()

    lowest_d1 = d1.rolling(window=length).min()
    highest_d1 = d1.rolling(window=length).max()
    _denom2 = (highest_d1 - lowest_d1).replace(0, float('nan'))
    k2 = (100 * (d1 - lowest_d1) / _denom2).fillna(0)

    smooth_len = max(1, int(aaa * 10))
    stc = k2.ewm(span=smooth_len, adjust=False).mean()

    return stc


def f_normalize(_src, _lower_band, _upper_band):
    """
    STC归一化
    """
    return 200 * (_src - _lower_band) / (_upper_band - _lower_band) - 100


if __name__ == "__main__":
    # 测试代码
    data_file = "D:\\期货数据\\铜期货监控\\global_futures_daily\\白银_daily.csv"
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # 使用CCI(22, MA13) - 与用户TradingView设置一致
    result = calculate_cci_tv(df, cci_length=22, ma_length=13)

    # 显示1月6日的结果
    target_date = pd.Timestamp('2026-01-06')
    if target_date in result.index:
        row = result.loc[target_date]
        print(f"Testing CCI(22, MA13) for {target_date.strftime('%Y-%m-%d')}:")
        print(f"  CCI:      {row['cci']:.2f}")
        print(f"  CCI_MA:   {row['cci_ma']:.2f}")
        print(f"  Diff:     {row['cci'] - row['cci_ma']:.2f}")
        print(f"\nUser's TradingView: CCI=141.928, CCI_MA=133.919")
        print(f"Differences: CCI={abs(row['cci']-141.928):.2f}, MA={abs(row['cci_ma']-133.919):.2f}")
