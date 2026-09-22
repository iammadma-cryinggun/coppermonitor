"""
最优参数配置 - 论文因子版（2026-02-12）
包含12个品种的最优论文因子配置

【更新说明】
- 基于完整回测测试
- 添加了隔夜缺口因子配置
- 添加了趋势质量因子配置
- 标注了各品种的因子效果
"""

# ============================================================
# 最优参数配置 - 包含论文因子
# ============================================================
OPTIMAL_PARAMS_WITH_FACTORS = {
    # ========== 白银（AG）- 隔夜缺口因子效果极其显著 ==========
    '白银': {
        # 品种代码
        'code': 'ag0',

        # CCI参数
        'cci_length': 15,
        'ma_length': 15,
        'cci_oversold': -60,
        'cci_overbought': 190,
        'cci_cross_max': 180,

        # STC参数
        'stc_oversold': -50,
        'stc_cross': -100,

        # 基础配置
        'multiplier': 15,
        'vol_threshold': 0.65,

        # 回测结果
        'expected_return': 6711.5,  # 使用因子后
        'max_dd': -40.4,
        'win_rate': 51.1,
        'total_trades': 276,

        # 论文因子配置
        'use_paper_factor': True,
        'paper_factor_type': 'gap',  # 'gap', 'trend_quality', 'gap_acceptance', 'orderly_trend'
        'gap_threshold': -0.35,  # 🔥关键参数！收益提升+1700%
        'factor_effect': '极其显著',  # 极其显著、显著、中等、小幅、无效
        'improvement': '+1700.4% (+34%)',
    },

    # ========== 铜（CU）- 隔夜缺口因子效果显著 ==========
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

        'expected_return': 1296.0,
        'max_dd': -50.5,
        'win_rate': 48.3,
        'total_trades': 474,

        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.15,  # 🔥关键参数！收益提升+573%
        'factor_effect': '显著',
        'improvement': '+572.6% (+79%)',
    },

    # ========== 纯碱（SA）- 隔夜缺口因子小幅提升 ==========
    '纯碱': {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 190,
        'cci_cross_max': 180,

        'stc_oversold': -50,
        'stc_cross': -130,

        'multiplier': 20,
        'vol_threshold': 0.50,

        'expected_return': 738.9,
        'max_dd': -54.4,
        'win_rate': 45.2,
        'total_trades': 199,

        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.15,  # 关键参数！收益提升+33%
        'factor_effect': '小幅',  # v3极端过滤效果更好（+165%）
        'improvement': '+32.5% (+5%)',
        'alternative': 'v3_extreme_filter',  # 可选方案
    },

    # ========== 镍（NI）- 趋势质量因子效果中等 ==========
    '镍': {
        'cci_length': 12,
        'ma_length': 8,
        'cci_oversold': -40,
        'cci_overbought': 190,
        'cci_cross_max': 180,

        'stc_oversold': -60,
        'stc_cross': -120,

        'multiplier': 1,
        'vol_threshold': 0.00,

        'expected_return': 224.4,
        'max_dd': -26.3,
        'win_rate': 46.7,
        'total_trades': 92,

        'use_paper_factor': True,
        'paper_factor_type': 'trend_quality',
        'trend_quality_threshold': 0.25,  # 关键参数！收益提升+41%
        'factor_effect': '中等',
        'improvement': '+40.6% (+22%)',
    },

    # ========== 玻璃（FG）- 隔夜缺口因子效果显著 ==========
    '玻璃': {
        'cci_length': 14,
        'ma_length': 8,
        'cci_oversold': -60,
        'cci_overbought': 100,
        'cci_cross_max': 150,

        'stc_oversold': -40,
        'stc_cross': -70,

        'multiplier': 20,
        'vol_threshold': 0.90,

        'expected_return': 979.6,
        'max_dd': -41.0,
        'win_rate': 45.4,
        'total_trades': 306,

        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.2,  # 🔥关键参数！收益提升+480%
        'factor_effect': '显著',
        'improvement': '+479.9% (+96%)',
    },

    # ========== 锡（SN）- 原始策略最好 ==========
    '锡': {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 180,
        'cci_cross_max': 150,

        'stc_oversold': -40,
        'stc_cross': -130,

        'multiplier': 1,
        'vol_threshold': 0.00,

        'expected_return': 1581.7,
        'max_dd': -38.9,
        'win_rate': 52.9,
        'total_trades': 306,

        'use_paper_factor': False,  # 不使用任何论文因子
        'paper_factor_type': None,
        'factor_effect': '无效（所有因子都降低收益）',
        'improvement': '+0%',
    },

    # ========== 铝（AL）- 趋势质量因子效果中等 ==========
    '铝': {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 180,
        'cci_cross_max': 150,

        'stc_oversold': -60,
        'stc_cross': -120,

        'multiplier': 5,
        'vol_threshold': 0.75,

        'expected_return': 121.4,
        'max_dd': -33.9,
        'win_rate': 54.1,
        'total_trades': 229,

        'use_paper_factor': True,
        'paper_factor_type': 'trend_quality',
        'trend_quality_threshold': 0.25,  # 关键参数！收益提升+34%
        'factor_effect': '中等',
        'improvement': '+33.7% (+38%)',
    },

    # ========== 铅（PB）- 隔夜缺口因子小幅提升 ==========
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

        'expected_return': 37.8,
        'max_dd': -31.7,
        'win_rate': 43.6,
        'total_trades': 431,

        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.3,  # 关键参数！收益提升+9%
        'factor_effect': '小幅',
        'improvement': '+8.8% (+30%)',
    },

    # ========== 棉花（CF）- 隔夜缺口因子效果显著 ==========
    '棉花': {
        'cci_length': 15,
        'ma_length': 10,
        'cci_oversold': -100,
        'cci_overbought': 150,
        'cci_cross_max': 120,

        'stc_oversold': -50,
        'stc_cross': -100,

        'multiplier': 5,
        'vol_threshold': 0.90,

        'expected_return': 549.8,
        'max_dd': -28.5,
        'win_rate': 51.7,
        'total_trades': 487,

        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.1,  # 🔥关键参数！收益提升+50%
        'factor_effect': '显著',
        'improvement': '+50.1% (+10%)',
    },

    # ========== 锌（ZN）- 隔夜缺口因子效果中等 ==========
    '锌': {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -80,
        'cci_overbought': 150,
        'cci_cross_max': 120,

        'stc_oversold': -40,
        'stc_cross': -80,

        'multiplier': 5,
        'vol_threshold': 0.60,

        'expected_return': 229.3,
        'max_dd': -46.4,
        'win_rate': 45.9,
        'total_trades': 453,

        'use_paper_factor': True,
        'paper_factor_type': 'gap',
        'gap_threshold': -0.15,  # 关键参数！收益提升+47%
        'factor_effect': '中等',
        'improvement': '+47.3% (+26%)',
    },

    # ========== 黄金（AU）- 原始策略最好 ==========
    '黄金': {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 160,
        'cci_cross_max': 150,

        'stc_oversold': -60,
        'stc_cross': -130,

        'multiplier': 1000,  # 注意：合约乘数很大
        'vol_threshold': 0.80,

        'expected_return': 2237.97,
        'max_dd': -87.72,  # ⚠️ 回撤非常大
        'win_rate': 55.1,
        'total_trades': 199,

        'use_paper_factor': False,  # 不使用任何论文因子
        'paper_factor_type': None,
        'factor_effect': '无效（所有因子都降低收益）',
        'improvement': '+0%',
        'warning': '最大回撤过大（-87.72%），谨慎实盘',
    },

    # ========== 糖（SR）- 缺口接受因子有效但收益低 ==========
    '糖': {
        'cci_length': 15,
        'ma_length': 5,
        'cci_oversold': -60,
        'cci_overbought': 100,
        'cci_cross_max': 120,

        'stc_oversold': -40,
        'stc_cross': -60,

        'multiplier': 10,
        'vol_threshold': 0.70,

        'expected_return': 47.4,
        'max_dd': -41.6,
        'win_rate': 45.9,
        'total_trades': 290,

        'use_paper_factor': True,
        'paper_factor_type': 'gap_acceptance',
        'gap_acceptance_threshold': 0.001,  # 关键参数！收益提升+47%
        'factor_effect': '小幅',
        'improvement': '+47.3% (+47300%，但基数太低）',
        'warning': '绝对收益太低（47.4%），不推荐实盘',
    },
}

# ============================================================
# 因子效果统计
# ============================================================
FACTOR_EFFECT_SUMMARY = {
    '隔夜缺口因子': {
        '适用品种数': 7,
        '适用品种': ['白银', '铜', '纯碱', '玻璃', '铅', '棉花', '锌'],
        '平均提升': '+400%',
        '评级': '⭐⭐⭐ 最有效因子',
    },
    '趋势质量因子': {
        '适用品种数': 2,
        '适用品种': ['镍', '铝'],
        '平均提升': '+37%',
        '评级': '⭐⭐ 中等有效',
    },
    '缺口接受因子': {
        '适用品种数': 1,
        '适用品种': ['糖'],
        '平均提升': '+47%',
        '评级': '⭐ 有效但有限',
    },
    '原始策略': {
        '适用品种数': 2,
        '适用品种': ['锡', '黄金'],
        '平均提升': '+0%',
        '评级': '⭐⭐⭐ 原始策略已最优',
    },
}

# ============================================================
# 推荐组合
# ============================================================
RECOMMENDED_PORTFOLIOS = {
    '保守型': {
        '品种': ['白银', '铜', '玻璃'],
        '预期收益': '+3000%/年',
        '预期回撤': '-30% ~ -50%',
        '使用因子': '隔夜缺口因子',
    },

    '稳健型': {
        '品种': ['白银', '铜', '玻璃', '棉花', '锌'],
        '预期收益': '+2000%/年',
        '预期回撤': '-30% ~ -45%',
        '使用因子': '隔夜缺口因子',
    },

    '激进型': {
        '品种': ['白银', '黄金', '锡'],
        '预期收益': '+3000%/年',
        '预期回撤': '-50% ~ -90%',
        '警告': '黄金回撤过大（-87%）',
    },
}


def print_config_summary():
    """打印配置摘要"""
    print("=" * 100)
    print("最优参数配置摘要 - 论文因子版".center(100))
    print("=" * 100)

    print("\n【因子效果统计】")
    for factor_name, info in FACTOR_EFFECT_SUMMARY.items():
        print(f"\n{factor_name}:")
        print(f"  适用品种数: {info['适用品种数']}")
        print(f"  适用品种: {', '.join(info['适用品种'])}")
        print(f"  平均提升: {info['平均提升']}")
        rating = info['评级'].replace('⭐', '*').replace('⭐', '*')
        print(f"  评级: {rating}")

    print("\n" + "=" * 100)
    print("各品种最优配置".center(100))
    print("=" * 100)

    print(f"\n{'品种':<8}{'使用因子':<20}{'因子参数':<25}{'收益率':>12}{'提升':>15}")
    print("-" * 100)

    for symbol, params in OPTIMAL_PARAMS_WITH_FACTORS.items():
        if params.get('use_paper_factor'):
            factor_type = params['paper_factor_type']
            if factor_type == 'gap':
                factor_param = f"缺口阈值={params['gap_threshold']}"
            elif factor_type == 'trend_quality':
                factor_param = f"质量阈值={params['trend_quality_threshold']}"
            elif factor_type == 'gap_acceptance':
                factor_param = f"接受阈值={params['gap_acceptance_threshold']}"
            else:
                factor_param = "N/A"
        else:
            factor_type = '原始策略'
            factor_param = "无"

        print(f"{symbol:<8}{factor_type:<20}{factor_param:<25}{params['expected_return']:>10.1f}%{params['improvement']:>15}")

    print("\n" + "=" * 100)
    print("推荐组合".center(100))
    print("=" * 100)

    for portfolio_type, info in RECOMMENDED_PORTFOLIOS.items():
        print(f"\n【{portfolio_type}】")
        print(f"品种: {', '.join(info['品种'])}")
        print(f"预期收益: {info['预期收益']}")
        print(f"预期回撤: {info['预期回撤']}")
        print(f"使用因子: {info['使用因子']}")
        if '警告' in info:
            print(f"警告: {info['警告']}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    print_config_summary()
