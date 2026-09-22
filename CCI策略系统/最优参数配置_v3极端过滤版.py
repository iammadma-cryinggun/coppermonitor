"""
最优参数配置 - v3极端过滤版
适用于：铜、纯碱、铅

【v3极端过滤方案】
只在极端情况下暂停开多，不使用MA60趋势过滤：
- 连续下跌 >= 10天
- 20天跌幅 > 20%
- 5天跌幅 > 15%

【回测验证效果】
铜:   收益 723.3% -> 1046.2% (+322.8%), 回撤改善5.9%
纯碱: 收益 706.4% -> 871.8% (+165.5%), 回撤改善8.0%
铅:   收益 29.0% -> 38.4% (+9.3%)
"""

# 品种代码映射
CODE_MAP = {
    '铜': 'cu0', '铝': 'al0', '锌': 'zn0', '铅': 'pb0', '镍': 'ni0', '锡': 'sn0',
    '黄金': 'au0', '白银': 'ag0', '玻璃': 'fg0', '纯碱': 'sa0', '糖': 'sr0', '棉花': 'cf0'
}

# 最优参数配置
OPTIMAL_PARAMS = {
    # ============ 铜配置 ============
    # 回测收益: 1046.2% (v3) vs 723.3% (原始)
    # 最大回撤: -50.3% (v3) vs -56.2% (原始)
    # 交易次数: 481笔
    # 胜率: 50.3%
    '铜': {
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -20,
        'stc_cross': -30,
        'multiplier': 5,
        'vol_threshold': 0.50,
        'use_extreme_filter': True,  # 启用v3极端过滤
    },

    # ============ 纯碱配置 ============
    # 回测收益: 871.8% (v3) vs 706.4% (原始)
    # 最大回撤: -46.6% (v3) vs -54.6% (原始)
    # 交易次数: 191笔
    # 胜率: 46.1%
    '纯碱': {
        'cci_length': 30,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -40,
        'stc_cross': -30,
        'multiplier': 20,
        'vol_threshold': 0.50,
        'use_extreme_filter': True,  # 启用v3极端过滤
    },

    # ============ 铅配置 ============
    # 回测收益: 38.4% (v3) vs 29.0% (原始)
    # 最大回撤: -36.1%
    # 交易次数: 433笔
    # 胜率: 43.6%
    '铅': {
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -20,
        'stc_cross': -30,
        'multiplier': 5,
        'vol_threshold': 0.70,
        'use_extreme_filter': True,  # 启用v3极端过滤
    },
}

# 极端情况检测参数
EXTREME_FILTER_PARAMS = {
    'consecutive_down_days': 10,    # 连续下跌天数阈值
    'drop_20d_threshold': 0.20,     # 20天跌幅阈值 (20%)
    'drop_5d_threshold': 0.15,      # 5天跌幅阈值 (15%)
}


def is_extreme_condition(df, idx, params=None):
    """
    检测极端市场情况

    参数:
        df: DataFrame，包含close列
        idx: 当前索引位置
        params: 极端过滤参数，默认使用EXTREME_FILTER_PARAMS

    返回:
        (is_extreme, reason): (是否极端, 原因描述)
    """
    if params is None:
        params = EXTREME_FILTER_PARAMS

    if idx < 20:
        return False, ""

    # 1. 连续下跌天数检测
    returns = df['close'].pct_change()
    consecutive_down = 0
    for i in range(idx, max(0, idx - 15), -1):
        if returns.iloc[i] < 0:
            consecutive_down += 1
        else:
            break

    if consecutive_down >= params['consecutive_down_days']:
        return True, f"连续下跌{consecutive_down}天"

    # 2. 20天跌幅检测
    if idx >= 19:
        drop_20d = (df['close'].iloc[idx - 19] - df['close'].iloc[idx]) / df['close'].iloc[idx - 19]
        if drop_20d > params['drop_20d_threshold']:
            return True, f"20天跌{drop_20d * 100:.1f}%"

    # 3. 5天跌幅检测
    if idx >= 4:
        drop_5d = (df['close'].iloc[idx - 4] - df['close'].iloc[idx]) / df['close'].iloc[idx - 4]
        if drop_5d > params['drop_5d_threshold']:
            return True, f"5天跌{drop_5d * 100:.1f}%"

    return False, ""


# 不推荐使用v3极端过滤的品种（保持原始策略）
NOT_RECOMMENDED_V3 = [
    '白银',   # -1987.6%  高波动，极端过滤太严
    '锡',     # -68.4%
    '铝',     # -58.7%
    '棉花',   # -6.0%
    '锌',     # -3.7%
    '镍',     # -3.3%
    '糖',     # 0%  本身策略收益低
    '黄金',   # 0%
    '玻璃',   # 0%
]


if __name__ == "__main__":
    print("=" * 80)
    print("v3极端过滤版 - 最优参数配置".center(80))
    print("=" * 80)

    print("\n【适用品种】铜、纯碱、铅")
    print("\n【极端过滤条件】")
    print(f"  - 连续下跌 >= {EXTREME_FILTER_PARAMS['consecutive_down_days']}天")
    print(f"  - 20天跌幅 > {EXTREME_FILTER_PARAMS['drop_20d_threshold']*100}%")
    print(f"  - 5天跌幅 > {EXTREME_FILTER_PARAMS['drop_5d_threshold']*100}%")

    print("\n【回测验证结果】")
    print(f"{'品种':<8}{'原始收益':>12}{'v3收益':>12}{'提升':>12}{'回撤改善':>12}")
    print("-" * 56)
    print(f"{'铜':<8}{'723.3%':>12}{'1046.2%':>12}{'+322.8%':>12}{'+5.9%':>12}")
    print(f"{'纯碱':<8}{'706.4%':>12}{'871.8%':>12}{'+165.5%':>12}{'+8.0%':>12}")
    print(f"{'铅':<8}{'29.0%':>12}{'38.4%':>12}{'+9.3%':>12}{'0%':>12}")

    print("\n【不推荐v3的品种】")
    for symbol in NOT_RECOMMENDED_V3:
        print(f"  - {symbol}")
