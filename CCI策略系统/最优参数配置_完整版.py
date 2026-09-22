"""
最优参数配置 - 完整版
包含所有12个品种的定制化策略配置

【策略分类】
1. v3极端过滤：铜、纯碱、铅
2. 隔夜缺口因子：白银、黄金
3. 组合策略：玻璃、铝
4. ATR自适应：棉花
5. 原始策略：镍、锡、锌、糖
"""

# 品种代码映射
CODE_MAP = {
    '铜': 'cu0', '铝': 'al0', '锌': 'zn0', '铅': 'pb0', '镍': 'ni0', '锡': 'sn0',
    '黄金': 'au0', '白银': 'ag0', '玻璃': 'fg0', '纯碱': 'sa0', '糖': 'sr0', '棉花': 'cf0'
}

# 最优参数配置
OPTIMAL_PARAMS = {
    # ============ v3极端过滤品种 ============

    # 铜: 收益 1046.2% (原始723.3%)
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
        'strategy_type': 'v3_extreme_filter',
    },

    # 纯碱: 收益 871.8% (原始706.4%)
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
        'strategy_type': 'v3_extreme_filter',
    },

    # 铅: 收益 38.4% (原始29.0%)
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
        'strategy_type': 'v3_extreme_filter',
    },

    # ============ 隔夜缺口因子品种 ============

    # 白银: 收益 6675.8% (原始5011.1%)
    '白银': {
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -40,
        'stc_cross': -30,
        'multiplier': 15,
        'vol_threshold': 0.65,
        'strategy_type': 'gap_factor',
        'gap_threshold': -0.3,  # 负缺口超过30%则不开多
    },

    # 黄金: 收益 520.3% (原始499.7%)
    '黄金': {
        'cci_length': 40,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -40,
        'stc_cross': -30,
        'multiplier': 100,
        'vol_threshold': 0.80,
        'strategy_type': 'gap_factor',
        'gap_threshold': -0.3,
    },

    # ============ 组合策略品种 ============

    # 玻璃: 收益 264.0% (原始183.8%)
    '玻璃': {
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -40,
        'stc_cross': -30,
        'multiplier': 20,
        'vol_threshold': 0.90,
        'strategy_type': 'combined',
        'trend_quality_threshold': 0.25,
        'gap_threshold': -0.5,
        'use_atr_adaptive': True,
        'use_extreme_filter': True,
    },

    # 铝: 收益 89.4% (原始87.7%, v3只有29%)
    '铝': {
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -20,
        'stc_cross': -30,
        'multiplier': 5,
        'vol_threshold': 0.75,
        'strategy_type': 'combined',
        'trend_quality_threshold': 0.25,
        'gap_threshold': -0.5,
        'use_atr_adaptive': True,
        'use_extreme_filter': True,
    },

    # ============ ATR自适应品种 ============

    # 棉花: 收益 58.2% (原始21.6%)
    '棉花': {
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -40,
        'stc_cross': -30,
        'multiplier': 5,
        'vol_threshold': 0.90,
        'strategy_type': 'atr_adaptive',
        'atr_ratio_threshold': 1.5,  # ATR超过均值1.5倍时调整
        'adaptive_oversold_high_vol': -150,  # 高波动时超卖阈值
        'adaptive_oversold_normal': -100,  # 正常时超卖阈值
        'adaptive_stop_loss_high_vol': 0.06,  # 高波动止损6%
        'adaptive_stop_loss_normal': 0.04,  # 正常止损4%
    },

    # ============ 原始策略品种 ============

    # 镍: 收益 929.4%
    '镍': {
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -40,
        'stc_cross': -30,
        'multiplier': 1,
        'vol_threshold': 0.00,
        'strategy_type': 'original',
    },

    # 锡: 收益 1581.7%
    '锡': {
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -40,
        'stc_cross': -30,
        'multiplier': 1,
        'vol_threshold': 0.00,
        'strategy_type': 'original',
    },

    # 锌: 收益 182.0%
    '锌': {
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -20,
        'stc_cross': -30,
        'multiplier': 5,
        'vol_threshold': 0.60,
        'strategy_type': 'original',
    },

    # 糖: 收益 0.1% (策略效果差，建议不交易或重新设计)
    '糖': {
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -40,
        'stc_cross': -30,
        'multiplier': 10,
        'vol_threshold': 0.70,
        'strategy_type': 'original',
        'note': '策略效果差，建议不交易或重新设计参数',
    },
}

# 极端过滤参数（用于v3_extreme_filter策略）
EXTREME_FILTER_PARAMS = {
    'consecutive_down_days': 10,
    'drop_20d_threshold': 0.20,
    'drop_5d_threshold': 0.15,
}

# 回测验证结果汇总
BACKTEST_RESULTS = {
    'v3极端过滤': {
        '铜': {'original': 723.3, 'optimized': 1046.2, 'improve': 322.9},
        '纯碱': {'original': 706.4, 'optimized': 871.8, 'improve': 165.4},
        '铅': {'original': 29.0, 'optimized': 38.4, 'improve': 9.4},
    },
    '隔夜缺口因子': {
        '白银': {'original': 5011.1, 'optimized': 6675.8, 'improve': 1664.7},
        '黄金': {'original': 499.7, 'optimized': 520.3, 'improve': 20.6},
    },
    '组合策略': {
        '玻璃': {'original': 183.8, 'optimized': 264.0, 'improve': 80.2},
        '铝': {'original': 87.7, 'optimized': 89.4, 'improve': 1.7},
    },
    'ATR自适应': {
        '棉花': {'original': 21.6, 'optimized': 58.2, 'improve': 36.6},
    },
    '原始策略': {
        '镍': {'return': 929.4},
        '锡': {'return': 1581.7},
        '锌': {'return': 182.0},
        '糖': {'return': 0.1},
    },
}


if __name__ == "__main__":
    print("=" * 80)
    print("完整版最优参数配置".center(80))
    print("=" * 80)

    print("\n【策略分类汇总】")
    print("\n1. v3极端过滤（3个品种）")
    print("   铜: 723% -> 1046% (+323%)")
    print("   纯碱: 706% -> 872% (+165%)")
    print("   铅: 29% -> 38% (+9%)")

    print("\n2. 隔夜缺口因子（2个品种）")
    print("   白银: 5011% -> 6676% (+1665%)")
    print("   黄金: 500% -> 520% (+21%)")

    print("\n3. 组合策略（2个品种）")
    print("   玻璃: 184% -> 264% (+80%)")
    print("   铝: 88% -> 89% (v3亏损恢复盈利)")

    print("\n4. ATR自适应（1个品种）")
    print("   棉花: 22% -> 58% (+37%)")

    print("\n5. 原始策略（4个品种）")
    print("   镍: 929%")
    print("   锡: 1582%")
    print("   锌: 182%")
    print("   糖: 0.1% (建议不交易)")

    print("\n" + "=" * 80)
    print("按收益提升排序".center(80))
    print("=" * 80)

    improvements = [
        ('白银', '隔夜缺口', 1664.7),
        ('铜', 'v3极端', 322.9),
        ('纯碱', 'v3极端', 165.4),
        ('玻璃', '组合', 80.2),
        ('棉花', 'ATR自适应', 36.6),
        ('黄金', '隔夜缺口', 20.6),
        ('铅', 'v3极端', 9.4),
        ('铝', '组合', 1.7),
    ]

    print(f"\n{'排名':<4}{'品种':<8}{'策略':<12}{'收益提升':>10}")
    print("-" * 40)
    for i, (symbol, strategy, improve) in enumerate(improvements, 1):
        print(f"{i:<4}{symbol:<8}{strategy:<12}{improve:>+9.1f}%")
