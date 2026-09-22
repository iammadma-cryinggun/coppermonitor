"""
纯碱实时分析 - 判断是否适合开多
结合CCI+STC指标和量能数据
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from 最优参数配置 import OPTIMAL_PARAMS


def get_akshare_data(code):
    """从akshare获取数据"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def calculate_stc(close_prices, fast_period=23, slow_period=50, cycle_period=10):
    """计算STC指标"""
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    lowest_macd = macd.rolling(window=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period).max()
    range_macd = highest_macd - lowest_macd

    stc = pd.Series(index=close_prices.index, dtype=float)
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


def analyze_signal(df, params, vol_threshold=0.5):
    """分析当前信号"""
    cci_length = params['cci_length']
    ma_length = params['ma_length']

    # 计算CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=cci_length).mean()
    mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()

    # 计算STC
    df['stc'] = calculate_stc(df['close'])

    # 计算量比
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    # 最新数据
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # 信号判断
    cci = latest['cci']
    cci_ma = latest['cci_ma']
    cci_prev = prev['cci']
    cci_ma_prev = prev['cci_ma']
    stc = latest['stc']
    vol_ratio = latest['vol_ratio']
    close = latest['close']

    signals = []

    # 超卖信号
    oversold_signal = cci < params['cci_oversold'] and stc >= params['stc_oversold']
    if oversold_signal:
        signals.append({
            'type': '超卖反弹',
            'condition': f'CCI={cci:.1f} < {params["cci_oversold"]} 且 STC={stc:.1f} >= {params["stc_oversold"]}',
            'strength': '强' if cci < params['cci_oversold'] - 20 else '中'
        })

    # 金叉信号
    cross_signal = cci_prev <= cci_ma_prev and cci > cci_ma and cci <= params['cci_cross_max'] and stc >= params['stc_cross']
    if cross_signal:
        signals.append({
            'type': 'CCI金叉',
            'condition': f'CCI上穿CCI_MA ({cci_prev:.1f}->{cci:.1f}) 且 <= {params["cci_cross_max"]} 且 STC={stc:.1f} >= {params["stc_cross"]}',
            'strength': '中'
        })

    # 量能判断
    volume_ok = vol_ratio >= vol_threshold

    return {
        'date': df.index[-1].strftime('%Y-%m-%d'),
        'close': close,
        'cci': cci,
        'cci_ma': cci_ma,
        'stc': stc,
        'vol_ratio': vol_ratio,
        'signals': signals,
        'volume_ok': volume_ok,
        'vol_threshold': vol_threshold,
    }


def main():
    print("="*80)
    print("纯碱实时分析 - 是否适合开多".center(80))
    print("="*80)

    symbol = '纯碱'
    code = 'sa0'
    params = OPTIMAL_PARAMS[symbol]
    vol_threshold = 0.5  # 最优量能阈值

    print(f"\n【参数配置】")
    print(f"CCI周期: {params['cci_length']}, MA周期: {params['ma_length']}")
    print(f"CCI超卖: {params['cci_oversold']}, CCI超买: {params['cci_overbought']}")
    print(f"CCI金叉上限: {params['cci_cross_max']}")
    print(f"STC超卖: {params['stc_oversold']}, STC金叉: {params['stc_cross']}")
    print(f"量比阈值: {vol_threshold}")

    # 获取数据
    print(f"\n正在获取最新数据...")
    df = get_akshare_data(code)

    if df is None:
        print("获取数据失败！")
        return

    print(f"数据范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

    # 分析信号
    result = analyze_signal(df, params, vol_threshold)

    print(f"\n{'='*80}")
    print("当前状态".center(80))
    print("="*80)

    print(f"\n日期: {result['date']}")
    print(f"收盘价: {result['close']:.2f}")
    print(f"\n指标值:")
    print(f"  CCI: {result['cci']:.2f}")
    print(f"  CCI_MA: {result['cci_ma']:.2f}")
    print(f"  STC: {result['stc']:.2f}")
    print(f"  量比: {result['vol_ratio']:.2f}")

    print(f"\n{'='*80}")
    print("开多信号分析".center(80))
    print("="*80)

    if result['signals']:
        print(f"\n发现 {len(result['signals'])} 个信号:")
        for i, sig in enumerate(result['signals'], 1):
            print(f"\n{i}. {sig['type']} (强度: {sig['strength']})")
            print(f"   条件: {sig['condition']}")
    else:
        print(f"\n当前无开多信号")

        # 分析原因
        print(f"\n信号条件检查:")
        print(f"  超卖信号: CCI({result['cci']:.1f}) < {params['cci_oversold']} ? {result['cci'] < params['cci_oversold']}")
        if result['cci'] < params['cci_oversold']:
            print(f"            STC({result['stc']:.1f}) >= {params['stc_oversold']} ? {result['stc'] >= params['stc_oversold']}")

        print(f"\n  金叉信号: CCI({result['cci']:.1f}) > CCI_MA({result['cci_ma']:.1f}) ? {result['cci'] > result['cci_ma']}")
        if result['cci'] > result['cci_ma']:
            print(f"            CCI <= {params['cci_cross_max']} ? {result['cci'] <= params['cci_cross_max']}")
            print(f"            STC >= {params['stc_cross']} ? {result['stc'] >= params['stc_cross']}")

    print(f"\n{'='*80}")
    print("量能分析".center(80))
    print("="*80)

    print(f"\n量比: {result['vol_ratio']:.2f} (阈值: {result['vol_threshold']})")
    if result['volume_ok']:
        print(f"量能状态: 合格 (量比 >= {result['vol_threshold']})")
    else:
        print(f"量能状态: 不合格 (量比 < {result['vol_threshold']})")
        print(f"建议: 等待放量后再进场")

    print(f"\n{'='*80}")
    print("综合判断".center(80))
    print("="*80)

    if result['signals'] and result['volume_ok']:
        print(f"\n结论: 适合开多")
        print(f"理由: 有{len(result['signals'])}个有效信号 + 量能合格")
    elif result['signals']:
        print(f"\n结论: 谨慎开多")
        print(f"理由: 有信号但量能不足，建议等待放量")
    else:
        print(f"\n结论: 不适合开多")
        print(f"理由: 当前无有效开多信号")

    # 历史参考
    print(f"\n{'='*80}")
    print("近期历史参考".center(80))
    print("="*80)

    print(f"\n最近10个交易日:")
    print(f"{'日期':<12}{'收盘':<10}{'CCI':<10}{'STC':<10}{'量比':<8}{'信号':<10}")
    print("-"*60)

    for i in range(-10, 0):
        row = df.iloc[i]
        prev_row = df.iloc[i-1] if i > -len(df) else row

        cci = row['cci']
        cci_ma = row['cci_ma']
        cci_prev = prev_row['cci']
        cci_ma_prev = prev_row['cci_ma']
        stc = row['stc']
        vol_ratio = row['vol_ratio']

        signal = ""
        if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
            signal = "超卖"
        elif cci_prev <= cci_ma_prev and cci > cci_ma and cci <= params['cci_cross_max'] and stc >= params['stc_cross']:
            signal = "金叉"

        print(f"{df.index[i].strftime('%Y-%m-%d'):<12}{row['close']:<10.2f}{cci:<10.1f}{stc:<10.1f}{vol_ratio:<8.2f}{signal:<10}")

    return result


if __name__ == "__main__":
    main()
