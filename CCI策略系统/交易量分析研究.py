"""
交易量分析研究 - 评估交易量对CCI策略的帮助
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from datetime import datetime
from 最优参数配置 import OPTIMAL_PARAMS
from cci_calculations import calculate_cci_tv, calculate_stc, f_normalize

# 品种代码映射
CODE_MAP = {
    '铜': 'cu0', '铝': 'al0', '锌': 'zn0', '铅': 'pb0', '镍': 'ni0', '锡': 'sn0',
    '黄金': 'au0', '白银': 'ag0', '玻璃': 'fg0', '纯碱': 'sa0', '糖': 'sr0', '棉花': 'cf0'
}


def get_data_with_volume(code, days=500):
    """获取含交易量的数据"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        # 包含交易量
        df = df.iloc[:, :6]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.tail(days)
        df = df.dropna()
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def calculate_volume_indicators(df):
    """计算交易量相关指标"""
    # 成交量移动平均
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma10'] = df['volume'].rolling(10).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()

    # 量比（当日成交量/5日平均）
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    # 放量/缩量标识
    df['is_high_vol'] = df['vol_ratio'] > 1.5   # 放量（量比>1.5）
    df['is_low_vol'] = df['vol_ratio'] < 0.7    # 缩量（量比<0.7）

    # 量价关系
    df['price_change'] = df['close'].pct_change()
    df['vol_price_trend'] = 'neutral'

    # 放量上涨
    mask_up_high = (df['price_change'] > 0) & (df['is_high_vol'])
    df.loc[mask_up_high, 'vol_price_trend'] = '放量上涨'

    # 放量下跌
    mask_down_high = (df['price_change'] < 0) & (df['is_high_vol'])
    df.loc[mask_down_high, 'vol_price_trend'] = '放量下跌'

    # 缩量上涨
    mask_up_low = (df['price_change'] > 0) & (df['is_low_vol'])
    df.loc[mask_up_low, 'vol_price_trend'] = '缩量上涨'

    # 缩量下跌
    mask_down_low = (df['price_change'] < 0) & (df['is_low_vol'])
    df.loc[mask_down_low, 'vol_price_trend'] = '缩量下跌'

    return df


def analyze_volume_impact():
    """分析交易量对策略的影响"""

    print("="*120)
    print("交易量对CCI策略的影响分析".center(120))
    print("="*120)

    print("""
【理论分析】

交易量可以提供以下信息：

1. 趋势确认
   - 放量上涨 = 强势，趋势可能延续
   - 放量下跌 = 恐慌，可能加速下跌
   - 缩量上涨 = 乏力，可能回调
   - 缩量下跌 = 卖盘枯竭，可能反弹

2. 信号过滤
   - CCI超卖 + 放量下跌 = 真正的恐慌，反弹概率高
   - CCI超卖 + 缩量下跌 = 可能继续阴跌
   - CCI金叉 + 放量 = 真突破
   - CCI金叉 + 缩量 = 假突破风险

3. 风险预警
   - 持续缩量 = 市场观望，可能变盘
   - 异常放量 = 重大事件，注意风险
""")

    # 测试一个品种
    symbol = '铜'
    params = OPTIMAL_PARAMS[symbol]
    code = CODE_MAP[symbol]

    print(f"\n【实际测试: {symbol}】")
    print("="*120)

    df = get_data_with_volume(code)
    if df is None:
        print("获取数据失败")
        return

    df = calculate_volume_indicators(df)

    # 计算CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=params['cci_length']).mean()
    mad = tp.rolling(window=params['cci_length']).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=params['ma_length']).mean()

    # 找出所有超卖信号
    oversold_signals = df[df['cci'] < params['cci_oversold']].copy()

    print(f"\n超卖信号统计: {len(oversold_signals)}次")

    # 分析超卖信号后的表现（按量价关系分组）
    print(f"\n【超卖信号后的表现分析】")
    print("-"*120)

    results = []
    for i, (idx, row) in enumerate(oversold_signals.iterrows()):
        if i >= len(oversold_signals) - 1:
            continue

        # 找到信号后5天的表现
        future_data = df.loc[idx:].head(6)
        if len(future_data) < 6:
            continue

        entry_price = future_data.iloc[0]['close']
        future_return_5d = (future_data.iloc[-1]['close'] - entry_price) / entry_price * 100

        vol_trend = row['vol_price_trend']
        vol_ratio = row['vol_ratio']

        results.append({
            'date': idx,
            'vol_trend': vol_trend,
            'vol_ratio': vol_ratio,
            'return_5d': future_return_5d
        })

    results_df = pd.DataFrame(results)

    # 按量价关系统计
    print(f"\n{'量价关系':<15}{'次数':<10}{'平均5日收益':<15}{'胜率':<10}")
    print("-"*60)

    for trend in ['放量下跌', '缩量下跌', 'neutral']:
        trend_data = results_df[results_df['vol_trend'] == trend]
        if len(trend_data) > 0:
            avg_return = trend_data['return_5d'].mean()
            win_rate = len(trend_data[trend_data['return_5d'] > 0]) / len(trend_data) * 100
            print(f"{trend:<15}{len(trend_data):<10}{avg_return:>+.2f}%{win_rate:>8.1f}%")

    # 分析金叉信号
    print(f"\n【金叉信号后的表现分析】")
    print("-"*120)

    df['cci_cross'] = (df['cci'].shift(1) <= df['cci_ma'].shift(1)) & (df['cci'] > df['cci_ma'])
    df['cci_cross'] = df['cci_cross'] & (df['cci'] <= params['cci_cross_max'])

    cross_signals = df[df['cci_cross']].copy()
    print(f"\n金叉信号统计: {len(cross_signals)}次")

    cross_results = []
    for i, (idx, row) in enumerate(cross_signals.iterrows()):
        if i >= len(cross_signals) - 1:
            continue

        future_data = df.loc[idx:].head(6)
        if len(future_data) < 6:
            continue

        entry_price = future_data.iloc[0]['close']
        future_return_5d = (future_data.iloc[-1]['close'] - entry_price) / entry_price * 100

        vol_trend = row['vol_price_trend']
        vol_ratio = row['vol_ratio']

        cross_results.append({
            'date': idx,
            'vol_trend': vol_trend,
            'vol_ratio': vol_ratio,
            'return_5d': future_return_5d
        })

    if cross_results:
        cross_df = pd.DataFrame(cross_results)

        print(f"\n{'量价关系':<15}{'次数':<10}{'平均5日收益':<15}{'胜率':<10}")
        print("-"*60)

        for trend in ['放量上涨', '缩量上涨', 'neutral']:
            trend_data = cross_df[cross_df['vol_trend'] == trend]
            if len(trend_data) > 0:
                avg_return = trend_data['return_5d'].mean()
                win_rate = len(trend_data[trend_data['return_5d'] > 0]) / len(trend_data) * 100
                print(f"{trend:<15}{len(trend_data):<10}{avg_return:>+.2f}%{win_rate:>8.1f}%")

    # 结论
    print("\n" + "="*120)
    print("结论与建议".center(120))
    print("="*120)

    print("""
【交易量数据的潜在用途】

1. 过滤假信号
   - 开仓信号 + 放量 = 提高信号可靠性
   - 开仓信号 + 缩量 = 降低信号权重

2. 确认趋势强度
   - 持仓期间持续放量 = 趋势强，可继续持有
   - 持仓期间持续缩量 = 趋势弱，考虑提前平仓

3. 风险预警
   - 异常放量（量比>3）= 注意风险
   - 持续缩量 = 市场观望，可能变盘

【建议实现方式】

方式1: 简单过滤
   - 开仓条件增加: 量比 > 0.8（避免极端缩量）

方式2: 权重调整
   - 根据量比调整仓位大小
   - 放量 = 正常仓位，缩量 = 降低仓位

方式3: 独立预警
   - 不改变开仓逻辑
   - 仅作为辅助参考指标

【注意事项】

1. 期货成交量数据可能有延迟或不准确
2. 不同品种的"放量"标准不同
3. 需要更多历史数据验证有效性
4. 增加复杂度可能过拟合
""")


if __name__ == "__main__":
    analyze_volume_impact()
