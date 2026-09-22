"""
各品种量比分位数分析 - 找出适合每个品种的缩量阈值
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from 最优参数配置 import OPTIMAL_PARAMS

# 品种代码映射
CODE_MAP = {
    '铜': 'cu0', '铝': 'al0', '锌': 'zn0', '铅': 'pb0', '镍': 'ni0', '锡': 'sn0',
    '黄金': 'au0', '白银': 'ag0', '玻璃': 'fg0', '纯碱': 'sa0', '糖': 'sr0', '棉花': 'cf0'
}


def get_full_data(code):
    """获取完整历史数据（含交易量）"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def analyze_volume_distribution():
    """分析每个品种的量比分布"""

    print("="*120)
    print("各品种量比分布分析".center(120))
    print("="*120)

    print("""
【说明】

量比 = 当日成交量 / 5日平均成交量

不同品种的量比分布不同：
- 高流动性品种（铜、黄金）：量比通常较稳定
- 低流动性品种（玻璃、纯碱）：量比波动较大

我们需要找出每个品种的"极端缩量"阈值
""")

    all_stats = {}

    for symbol, params in OPTIMAL_PARAMS.items():
        code = CODE_MAP[symbol]
        df = get_full_data(code)

        if df is None:
            continue

        # 计算量比
        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma5']

        # 过滤无效值
        vol_ratios = df['vol_ratio'].dropna()

        # 计算分位数
        percentiles = [1, 5, 10, 15, 20, 25, 50, 75, 90, 95, 99]
        pct_values = vol_ratios.quantile([p/100 for p in percentiles])

        all_stats[symbol] = {
            'count': len(vol_ratios),
            'mean': vol_ratios.mean(),
            'std': vol_ratios.std(),
            'min': vol_ratios.min(),
            'max': vol_ratios.max(),
            'percentiles': pct_values
        }

    # 输出结果
    print(f"\n{'品种':<10}{'数据量':<10}{'均值':<10}{'标准差':<10}{'最小值':<10}{'最大值':<10}")
    print("-"*70)
    for symbol, stats in all_stats.items():
        print(f"{symbol:<10}{stats['count']:<10}{stats['mean']:<10.2f}{stats['std']:<10.2f}"
              f"{stats['min']:<10.2f}{stats['max']:<10.2f}")

    # 输出分位数表
    print("\n" + "="*120)
    print("量比分位数分布".center(120))
    print("="*120)

    print(f"\n{'品种':<10}", end="")
    for p in [1, 5, 10, 15, 20, 25, 50]:
        print(f"{p}%分位:<10", end="")
    print()
    print("-"*90)

    for symbol, stats in all_stats.items():
        print(f"{symbol:<10}", end="")
        for p in [1, 5, 10, 15, 20, 25, 50]:
            val = stats['percentiles'].get(p/100, 0)
            print(f"{val:<10.2f}", end="")
        print()

    # 分析超卖信号时的量比分布
    print("\n" + "="*120)
    print("超卖信号时的量比分布".center(120))
    print("="*120)

    signal_stats = {}

    for symbol, params in OPTIMAL_PARAMS.items():
        code = CODE_MAP[symbol]
        df = get_full_data(code)

        if df is None:
            continue

        # 计算指标
        cci_length = params['cci_length']
        ma_length = params['ma_length']

        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=cci_length).mean()
        mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - sma) / (0.015 * mad)

        df['vol_ma5'] = df['volume'].rolling(5).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma5']

        # 超卖信号时的量比
        oversold_mask = df['cci'] < params['cci_oversold']
        oversold_vol = df.loc[oversold_mask, 'vol_ratio'].dropna()

        if len(oversold_vol) > 0:
            signal_stats[symbol] = {
                'count': len(oversold_vol),
                'mean': oversold_vol.mean(),
                'p10': oversold_vol.quantile(0.10),
                'p15': oversold_vol.quantile(0.15),
                'p20': oversold_vol.quantile(0.20),
                'p25': oversold_vol.quantile(0.25),
            }

    print(f"\n{'品种':<10}{'超卖次数':<12}{'量比均值':<12}{'10%分位':<12}{'15%分位':<12}{'20%分位':<12}{'25%分位':<12}")
    print("-"*95)

    for symbol, stats in signal_stats.items():
        print(f"{symbol:<10}{stats['count']:<12}{stats['mean']:<12.2f}"
              f"{stats['p10']:<12.2f}{stats['p15']:<12.2f}{stats['p20']:<12.2f}{stats['p25']:<12.2f}")

    # 推荐阈值
    print("\n" + "="*120)
    print("推荐阈值（基于超卖信号的15%分位数）".center(120))
    print("="*120)

    print(f"\n{'品种':<10}{'推荐量比阈值':<15}{'说明':<40}")
    print("-"*70)

    recommendations = {}
    for symbol, stats in signal_stats.items():
        threshold = stats['p15']  # 使用15%分位数作为阈值
        recommendations[symbol] = round(threshold, 2)

        if threshold < 0.4:
            note = "低流动性，缩量阈值低"
        elif threshold < 0.6:
            note = "中等流动性"
        elif threshold < 0.8:
            note = "较高流动性"
        else:
            note = "高流动性，缩量阈值高"

        print(f"{symbol:<10}{threshold:<15.2f}{note:<40}")

    # 生成配置代码
    print("\n" + "="*120)
    print("配置代码".center(120))
    print("="*120)

    print("\n# 各品种量比阈值配置（基于超卖信号15%分位数）")
    print("VOLUME_RATIO_THRESHOLDS = {")
    for symbol in sorted(recommendations.keys()):
        print(f"    '{symbol}': {recommendations[symbol]:.2f},")
    print("}")

    return recommendations


if __name__ == "__main__":
    analyze_volume_distribution()
