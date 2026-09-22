"""
交易量深度分析 - 使用完整历史数据
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from datetime import datetime
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


def calculate_indicators(df, params):
    """计算所有指标"""
    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=params['cci_length']).mean()
    mad = tp.rolling(window=params['cci_length']).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=params['ma_length']).mean()

    # 成交量指标
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma10'] = df['volume'].rolling(10).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    # 持仓量指标
    df['oi_ma5'] = df['open_interest'].rolling(5).mean()
    df['oi_change'] = df['open_interest'] / df['oi_ma5']

    # 价格变化
    df['price_change'] = df['close'].pct_change()

    # 量价关系分类
    df['vol_type'] = 'normal'
    df.loc[(df['price_change'] > 0.01) & (df['vol_ratio'] > 1.5), 'vol_type'] = '放量上涨'
    df.loc[(df['price_change'] < -0.01) & (df['vol_ratio'] > 1.5), 'vol_type'] = '放量下跌'
    df.loc[(df['price_change'] > 0.01) & (df['vol_ratio'] < 0.7), 'vol_type'] = '缩量上涨'
    df.loc[(df['price_change'] < -0.01) & (df['vol_ratio'] < 0.7), 'vol_type'] = '缩量下跌'

    return df


def analyze_volume_impact_detailed():
    """详细分析交易量影响"""

    print("="*120)
    print("交易量深度分析 - 10年历史数据".center(120))
    print("="*120)

    results_all = {}

    for symbol in ['铜', '纯碱', '玻璃', '白银', '镍']:
        params = OPTIMAL_PARAMS[symbol]
        code = CODE_MAP[symbol]

        print(f"\n{'='*100}")
        print(f"[{symbol}]".center(100))
        print("="*100)

        df = get_full_data(code)
        if df is None:
            continue

        # 只取最近10年
        df = df.tail(2500)
        df = calculate_indicators(df, params)

        print(f"数据范围: {df['date'].iloc[0].strftime('%Y-%m-%d')} 到 {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"总交易日: {len(df)}")

        # 统计量价关系分布
        print(f"\n【量价关系分布】")
        vol_dist = df['vol_type'].value_counts()
        for vt, count in vol_dist.items():
            pct = count / len(df) * 100
            print(f"  {vt}: {count}次 ({pct:.1f}%)")

        # 分析超卖信号
        print(f"\n【超卖信号分析】CCI < {params['cci_oversold']}")

        oversold_mask = df['cci'] < params['cci_oversold']
        oversold_count = oversold_mask.sum()
        print(f"  超卖信号总数: {oversold_count}")

        if oversold_count > 0:
            # 分析超卖后的表现
            oversold_results = []

            for i in range(len(df) - 10):
                if df.iloc[i]['cci'] < params['cci_oversold']:
                    row = df.iloc[i]

                    # 未来5天和10天的收益
                    entry_price = row['close']
                    ret_5d = (df.iloc[i+5]['close'] - entry_price) / entry_price * 100 if i+5 < len(df) else None
                    ret_10d = (df.iloc[i+10]['close'] - entry_price) / entry_price * 100 if i+10 < len(df) else None

                    oversold_results.append({
                        'vol_type': row['vol_type'],
                        'vol_ratio': row['vol_ratio'],
                        'return_5d': ret_5d,
                        'return_10d': ret_10d,
                        'oi_change': row['oi_change']
                    })

            oversold_df = pd.DataFrame(oversold_results)

            # 按量价关系统计
            print(f"\n  {'量价关系':<12}{'次数':<8}{'5日收益':<12}{'5日胜率':<10}{'10日收益':<12}{'10日胜率':<10}")
            print("  " + "-"*70)

            for vt in ['放量下跌', '缩量下跌', 'normal', '放量上涨', '缩量上涨']:
                vt_data = oversold_df[oversold_df['vol_type'] == vt]
                if len(vt_data) > 5:
                    ret_5 = vt_data['return_5d'].mean()
                    ret_10 = vt_data['return_10d'].mean()
                    win_5 = (vt_data['return_5d'] > 0).sum() / len(vt_data) * 100
                    win_10 = (vt_data['return_10d'] > 0).sum() / len(vt_data) * 100
                    print(f"  {vt:<12}{len(vt_data):<8}{ret_5:>+8.2f}%{win_5:>8.1f}%{ret_10:>+10.2f}%{win_10:>8.1f}%")

            # 按量比分组
            print(f"\n  【按量比分组】")
            print(f"  {'量比范围':<15}{'次数':<8}{'5日收益':<12}{'5日胜率':<10}")
            print("  " + "-"*50)

            bins = [(0, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 999)]
            labels = ['极缩(<0.5)', '缩量(0.5-0.8)', '正常(0.8-1.2)', '放量(1.2-1.5)', '大量(1.5-2)', '巨量(>2)']

            for (low, high), label in zip(bins, labels):
                mask = (oversold_df['vol_ratio'] >= low) & (oversold_df['vol_ratio'] < high)
                bin_data = oversold_df[mask]
                if len(bin_data) > 5:
                    ret = bin_data['return_5d'].mean()
                    win = (bin_data['return_5d'] > 0).sum() / len(bin_data) * 100
                    print(f"  {label:<15}{len(bin_data):<8}{ret:>+8.2f}%{win:>8.1f}%")

        # 分析金叉信号
        print(f"\n【金叉信号分析】")

        df['cci_cross'] = (df['cci'].shift(1) <= df['cci_ma'].shift(1)) & (df['cci'] > df['cci_ma'])
        df['cci_cross'] = df['cci_cross'] & (df['cci'] <= params['cci_cross_max'])

        cross_count = df['cci_cross'].sum()
        print(f"  金叉信号总数: {cross_count}")

        if cross_count > 0:
            cross_results = []

            for i in range(len(df) - 10):
                if df.iloc[i]['cci_cross']:
                    row = df.iloc[i]
                    entry_price = row['close']
                    ret_5d = (df.iloc[i+5]['close'] - entry_price) / entry_price * 100 if i+5 < len(df) else None
                    ret_10d = (df.iloc[i+10]['close'] - entry_price) / entry_price * 100 if i+10 < len(df) else None

                    cross_results.append({
                        'vol_type': row['vol_type'],
                        'vol_ratio': row['vol_ratio'],
                        'return_5d': ret_5d,
                        'return_10d': ret_10d
                    })

            cross_df = pd.DataFrame(cross_results)

            print(f"\n  {'量价关系':<12}{'次数':<8}{'5日收益':<12}{'5日胜率':<10}")
            print("  " + "-"*50)

            for vt in ['放量上涨', '缩量上涨', 'normal']:
                vt_data = cross_df[cross_df['vol_type'] == vt]
                if len(vt_data) > 5:
                    ret = vt_data['return_5d'].mean()
                    win = (vt_data['return_5d'] > 0).sum() / len(vt_data) * 100
                    print(f"  {vt:<12}{len(vt_data):<8}{ret:>+8.2f}%{win:>8.1f}%")

        results_all[symbol] = {
            'oversold_count': oversold_count,
            'cross_count': cross_count
        }

    # 总结
    print("\n" + "="*120)
    print("总结与建议".center(120))
    print("="*120)

    print("""
【交易量分析结论】

1. 量比指标有一定参考价值
   - 极端缩量(量比<0.5)时开仓效果可能较差
   - 适度放量(量比1.2-1.5)时信号可能更可靠

2. 持仓量(Open Interest)也可参考
   - 价格下跌+持仓量增加 = 新空单入场，可能继续跌
   - 价格下跌+持仓量减少 = 空头平仓，可能反弹

【建议添加的交易量过滤】

方式1: 避免极端缩量
   - 开仓条件增加: 量比 > 0.6
   - 简单有效，不会过滤太多信号

方式2: 放量确认（可选）
   - 金叉信号: 量比 > 1.0（有成交量配合）
   - 超卖信号: 不限（恐慌时缩量也正常）

方式3: 仅作为显示信息
   - 不改变交易逻辑
   - 在分析结果中显示量比，供人工判断
""")


if __name__ == "__main__":
    analyze_volume_impact_detailed()
