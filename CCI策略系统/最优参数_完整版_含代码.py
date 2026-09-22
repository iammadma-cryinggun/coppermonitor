"""
最优参数配置 - 论文因子完整版（2026-02-12）
=================================================
包含所有必需字段：code、multiplier、论文因子配置
"""

# 品种代码映射
CODE_MAP = {
    '铜': 'cu0', '铝': 'al0', '锌': 'zn0', '铅': 'pb0', '镍': 'ni0', '锡': 'sn0',
    '黄金': 'au0', '白银': 'ag0', '玻璃': 'fg0', '纯碱': 'sa0', '糖': 'sr0', '棉花': 'cf0'
}

# 合约乘数映射
MULTIPLIER_MAP = {
    '铜': 5, '铝': 5, '锌': 5, '铅': 5, '镍': 1, '锡': 1,
    '黄金': 1000, '白银': 15, '玻璃': 20, '纯碱': 20, '糖': 10, '棉花': 5
}

# 最优参数配置
OPTIMAL_PARAMS = {
    '白银': {
        'code': 'ag0',
        'multiplier': 15,
        'cci_length': 15,
        'ma_length': 15,
        'cci_oversold': -60,
        'cci_overbought': 190,
        'cci_cross_max': 180,
        'stc_oversold': -50,
        'stc_cross': -100,
        'expected_return': 6711.5,
        'max_dd': -40.4,
        'win_rate': 51.1,
        # 论文因子
        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.35,
        'factor_effect': '极其显著',
        'improvement': '+1700.4% (+34%)',
    },

    '铜': {
        'code': 'cu0',
        'multiplier': 5,
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -20,
        'stc_cross': -30,
        'expected_return': 1296.0,
        'max_dd': -50.5,
        'win_rate': 48.3,
        # 论文因子
        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.15,
        'factor_effect': '显著',
        'improvement': '+572.6% (+79%)',
    },

    '纯碱': {
        'code': 'sa0',
        'multiplier': 20,
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 190,
        'cci_cross_max': 180,
        'stc_oversold': -50,
        'stc_cross': -130,
        'expected_return': 738.9,
        'max_dd': -54.4,
        'win_rate': 45.2,
        # 论文因子
        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.15,
        'factor_effect': '小幅',
        'improvement': '+32.5% (+5%)',
    },

    '镍': {
        'code': 'ni0',
        'multiplier': 1,
        'cci_length': 12,
        'ma_length': 8,
        'cci_oversold': -40,
        'cci_overbought': 190,
        'cci_cross_max': 180,
        'stc_oversold': -60,
        'stc_cross': -120,
        'expected_return': 224.4,
        'max_dd': -26.3,
        'win_rate': 46.7,
        # 论文因子
        'use_paper_factor': True,
        'paper_factor_type': 'trend_quality',
        'trend_quality_threshold': 0.25,
        'factor_effect': '中等',
        'improvement': '+40.6% (+22%)',
    },

    '玻璃': {
        'code': 'fg0',
        'multiplier': 20,
        'cci_length': 14,
        'ma_length': 8,
        'cci_oversold': -60,
        'cci_overbought': 100,
        'cci_cross_max': 150,
        'stc_oversold': -40,
        'stc_cross': -70,
        'expected_return': 979.6,
        'max_dd': -41.0,
        'win_rate': 45.4,
        # 论文因子
        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.2,
        'factor_effect': '显著',
        'improvement': '+479.9% (+96%)',
    },

    '锡': {
        'code': 'sn0',
        'multiplier': 1,
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 180,
        'cci_cross_max': 150,
        'stc_oversold': -40,
        'stc_cross': -130,
        'expected_return': 1581.7,
        'max_dd': -38.9,
        'win_rate': 52.9,
        # 不使用论文因子
        'use_paper_factor': False,
        'paper_factor_type': None,
        'factor_effect': '无效（所有因子都降低收益）',
        'improvement': '+0%',
    },

    '铝': {
        'code': 'al0',
        'multiplier': 5,
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 180,
        'cci_cross_max': 150,
        'stc_oversold': -60,
        'stc_cross': -120,
        'expected_return': 121.4,
        'max_dd': -33.9,
        'win_rate': 54.1,
        # 论文因子
        'use_paper_factor': True,
        'paper_factor_type': 'trend_quality',
        'trend_quality_threshold': 0.25,
        'factor_effect': '中等',
        'improvement': '+33.7% (+38%)',
    },

    '铅': {
        'code': 'pb0',
        'multiplier': 5,
        'cci_length': 20,
        'ma_length': 5,
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': -20,
        'stc_oversold': -20,
        'stc_cross': -30,
        'expected_return': 37.8,
        'max_dd': -31.7,
        'win_rate': 43.6,
        # 论文因子
        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.3,
        'factor_effect': '小幅',
        'improvement': '+8.8% (+30%)',
    },

    '棉花': {
        'code': 'cf0',
        'multiplier': 5,
        'cci_length': 15,
        'ma_length': 10,
        'cci_oversold': -100,
        'cci_overbought': 150,
        'cci_cross_max': 120,
        'stc_oversold': -50,
        'stc_cross': -100,
        'expected_return': 549.8,
        'max_dd': -28.5,
        'win_rate': 51.7,
        # 论文因子
        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.1,
        'factor_effect': '显著',
        'improvement': '+50.1% (+10%)',
    },

    '锌': {
        'code': 'zn0',
        'multiplier': 5,
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -80,
        'cci_overbought': 150,
        'cci_cross_max': 120,
        'stc_oversold': -40,
        'stc_cross': -80,
        'expected_return': 229.3,
        'max_dd': -46.4,
        'win_rate': 45.9,
        # 论文因子
        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.15,
        'factor_effect': '中等',
        'improvement': '+47.3% (+26%)',
    },

    '黄金': {
        'code': 'au0',
        'multiplier': 1000,
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 160,
        'cci_cross_max': 150,
        'stc_oversold': -60,
        'stc_cross': -130,
        'expected_return': 2237.97,
        'max_dd': -87.72,
        'win_rate': 55.1,
        # 不使用论文因子
        'use_paper_factor': False,
        'paper_factor_type': None,
        'factor_effect': '无效（所有因子都降低收益）',
        'improvement': '+0%',
        'warning': '最大回撤过大（-87.72%），谨慎实盘',
    },

    '糖': {
        'code': 'sr0',
        'multiplier': 10,
        'cci_length': 15,
        'ma_length': 5,
        'cci_oversold': -60,
        'cci_overbought': 100,
        'cci_cross_max': 120,
        'stc_oversold': -40,
        'stc_cross': -60,
        'expected_return': 47.4,
        'max_dd': -41.6,
        'win_rate': 45.9,
        # 论文因子
        'use_paper_factor': True,
        'paper_factor_type': 'gap_acceptance',
        'gap_acceptance_threshold': 0.001,
        'factor_effect': '小幅',
        'improvement': '+47.3% (+47300%，但基数太低）',
        'warning': '绝对收益太低（47.4%），不推荐实盘',
    },
}


if __name__ == "__main__":
    print("="*100)
    print("最优参数配置验证".center(100))
    print("="*100)

    print(f"\n已配置 {len(OPTIMAL_PARAMS)} 个品种\n")

    for symbol, params in OPTIMAL_PARAMS.items():
        has_code = 'code' in params
        has_multiplier = 'multiplier' in params
        has_factor = 'use_paper_factor' in params

        status = "✓" if all([has_code, has_multiplier, has_factor]) else "✗"

        print(f"{symbol:<10} [必需字段] {status}")

        if has_code:
            print(f"  code: {params['code']}")
        if has_multiplier:
            print(f"  multiplier: {params['multiplier']}")
        if has_factor:
            factor_type = params.get('paper_factor_type', 'None')
            print(f"  论文因子: {factor_type}")
            if factor_type == 'gap':
                print(f"  gap_threshold: {params.get('gap_threshold', 'N/A')}")
            elif factor_type == 'trend_quality':
                print(f"  trend_quality_threshold: {params.get('trend_quality_threshold', 'N/A')}")

    print("\n" + "="*100)
    print("配置验证完成！")
    print("="*100)
