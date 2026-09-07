"""
CCI策略最优参数配置 - 最终版（基于全品种严格回测）
=================================================
所有12个品种的真实回测数据（含滑点0.02% + 手续费0.03%）
"""

# 最优参数配置 - 全品种严格回测结果
OPTIMAL_PARAMS = {
    '黄金': {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 160,
        'cci_cross_max': 150,
        'stc_oversold': -60,
        'stc_cross': -130,
        # 真实回测数据
        'expected_return': 2237.97,
        'max_dd': -87.72,
        'return_dd_ratio': 25.52,
        'win_rate': 55.1,
        'profit_ratio': 1.59,
        'profit_factor': 1.94,
        'expectancy': 10010,
        'total_trades': 199,
        'code': 'AU0',
        'exchange': 'SHFE',
        'multiplier': 1000,
        'rating': '优秀'
    },
    '镍': {
        'cci_length': 12,
        'ma_length': 8,
        'cci_oversold': -40,
        'cci_overbought': 190,
        'cci_cross_max': 180,
        'stc_oversold': -60,
        'stc_cross': -120,
        # 真实回测数据
        'expected_return': 1953.53,
        'max_dd': -47.79,
        'return_dd_ratio': 40.90,
        'win_rate': 55.7,
        'profit_ratio': 1.15,
        'profit_factor': 1.45,
        'expectancy': 11252,
        'total_trades': 185,
        'code': 'NI0',
        'exchange': 'SHFE',
        'multiplier': 1,
        'rating': '优秀'
    },
    '锡': {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 180,
        'cci_cross_max': 150,
        'stc_oversold': -40,
        'stc_cross': -130,
        # 真实回测数据
        'expected_return': 1738.56,
        'max_dd': -48.40,
        'return_dd_ratio': 35.92,
        'win_rate': 56.5,
        'profit_ratio': 2.16,
        'profit_factor': 2.82,
        'expectancy': 8248,
        'total_trades': 215,
        'code': 'SN0',
        'exchange': 'SHFE',
        'multiplier': 1,
        'rating': '优秀'
    },
    '白银': {
        'cci_length': 15,
        'ma_length': 15,
        'cci_oversold': -60,
        'cci_overbought': 190,
        'cci_cross_max': 180,
        'stc_oversold': -50,
        'stc_cross': -100,
        # 真实回测数据
        'expected_return': 1441.83,
        'max_dd': -33.40,
        'return_dd_ratio': 43.17,
        'win_rate': 47.1,
        'profit_ratio': 3.54,
        'profit_factor': 3.15,
        'expectancy': 9571,
        'total_trades': 153,
        'code': 'AG0',
        'exchange': 'SHFE',
        'multiplier': 15,
        'rating': '优秀'
    },
    '玻璃': {
        'cci_length': 14,
        'ma_length': 8,
        'cci_oversold': -60,
        'cci_overbought': 190,
        'cci_cross_max': 120,
        'stc_oversold': -30,
        'stc_cross': -120,
        # 真实回测数据
        'expected_return': 1370.91,
        'max_dd': -27.21,
        'return_dd_ratio': 50.38,
        'win_rate': 51.7,
        'profit_ratio': 2.68,
        'profit_factor': 2.86,
        'expectancy': 9250,
        'total_trades': 151,
        'code': 'FG0',
        'exchange': 'CZCE',
        'multiplier': 20,
        'rating': '优秀'
    },
    '纯碱': {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 190,
        'cci_cross_max': 180,
        'stc_oversold': -50,
        'stc_cross': -130,
        # 真实回测数据
        'expected_return': 1075.48,
        'max_dd': -25.52,
        'return_dd_ratio': 42.15,
        'win_rate': 56.8,
        'profit_ratio': 1.27,
        'profit_factor': 1.67,
        'expectancy': 9614,
        'total_trades': 118,
        'code': 'SA0',
        'exchange': 'CZCE',
        'multiplier': 20,
        'rating': '优秀'
    },
    '铜': {
        'cci_length': 12,
        'ma_length': 20,
        'cci_oversold': -40,
        'cci_overbought': 190,
        'cci_cross_max': 150,
        'stc_oversold': -50,
        'stc_cross': -100,
        # 真实回测数据
        'expected_return': 659.87,
        'max_dd': -18.53,
        'return_dd_ratio': 35.61,
        'win_rate': 51.0,
        'profit_ratio': 2.33,
        'profit_factor': 2.43,
        'expectancy': 4759,
        'total_trades': 143,
        'code': 'CU0',
        'exchange': 'SHFE',
        'multiplier': 5,
        'rating': '优秀'
    },
    '糖': {
        'cci_length': 14,
        'ma_length': 13,
        'cci_oversold': -60,
        'cci_overbought': 150,
        'cci_cross_max': 150,
        'stc_oversold': -60,
        'stc_cross': -120,
        # 真实回测数据
        'expected_return': 603.67,
        'max_dd': -37.08,
        'return_dd_ratio': 16.28,
        'win_rate': 51.6,
        'profit_ratio': 1.42,
        'profit_factor': 1.52,
        'expectancy': 4195,
        'total_trades': 151,
        'code': 'SR0',
        'exchange': 'CZCE',
        'multiplier': 10,
        'rating': '良好'
    },
    '锌': {
        'cci_length': 12,
        'ma_length': 16,
        'cci_oversold': -40,
        'cci_overbought': 180,
        'cci_cross_max': 150,
        'stc_oversold': -50,
        'stc_cross': -110,
        # 真实回测数据
        'expected_return': 546.43,
        'max_dd': -27.47,
        'return_dd_ratio': 19.89,
        'win_rate': 49.1,
        'profit_ratio': 1.72,
        'profit_factor': 1.65,
        'expectancy': 3516,
        'total_trades': 154,
        'code': 'ZN0',
        'exchange': 'SHFE',
        'multiplier': 5,
        'rating': '良好'
    },
    '棉花': {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 170,
        'cci_cross_max': 180,
        'stc_oversold': -60,
        'stc_cross': -130,
        # 真实回测数据
        'expected_return': 459.88,
        'max_dd': -31.55,
        'return_dd_ratio': 14.57,
        'win_rate': 55.6,
        'profit_ratio': 1.37,
        'profit_factor': 1.71,
        'expectancy': 3374,
        'total_trades': 130,
        'code': 'CF0',
        'exchange': 'CZCE',
        'multiplier': 5,
        'rating': '良好'
    },
    '铝': {
        'cci_length': 12,
        'ma_length': 20,
        'cci_oversold': -60,
        'cci_overbought': 170,
        'cci_cross_max': 120,
        'stc_oversold': -50,
        'stc_cross': -110,
        # 真实回测数据
        'expected_return': 243.33,
        'max_dd': -27.48,
        'return_dd_ratio': 8.86,
        'win_rate': 50.4,
        'profit_ratio': 1.97,
        'profit_factor': 2.00,
        'expectancy': 1963,
        'total_trades': 138,
        'code': 'AL0',
        'exchange': 'SHFE',
        'multiplier': 5,
        'rating': '一般'
    },
    '铅': {
        'cci_length': 14,
        'ma_length': 5,
        'cci_oversold': -50,
        'cci_overbought': 190,
        'cci_cross_max': 150,
        'stc_oversold': -50,
        'stc_cross': -120,
        # 真实回测数据
        'expected_return': 104.79,
        'max_dd': -21.91,
        'return_dd_ratio': 4.78,
        'win_rate': 45.0,
        'profit_ratio': 1.52,
        'profit_factor': 1.24,
        'expectancy': 444,
        'total_trades': 289,
        'code': 'PB0',
        'exchange': 'SHFE',
        'multiplier': 5,
        'rating': '较差'
    }
}

# 推荐品种（更新）
RECOMMENDED_SYMBOLS = {
    '保守': ['铜', '纯碱', '玻璃'],  # 低回撤，高盈亏比
    '稳健': ['玻璃', '纯碱', '锡'],  # 平衡性好
    '激进': ['黄金', '镍', '白银']   # 高收益，高回撤
}

# 交易规则
TRADING_RULES = {
    'entry': {
        'oversold': 'CCI < cci_oversold AND STC >= stc_oversold -> 次日开盘买入',
        'golden_cross': 'CCI上穿CCI_MA AND CCI <= cci_cross_max AND STC >= stc_cross -> 次日开盘买入'
    },
    'exit': {
        'stop_loss': '最低价 <= 开仓价 * 0.96 -> 次日开盘止损（max(open, 止损价)）',
        'take_profit': '最高价 >= 开仓价 * 1.20 -> 次日开盘止盈（min(open, 止盈价)）',
        'cci_overbought': 'CCI > cci_overbought -> 次日收盘平仓',
        'death_cross': 'CCI下穿CCI_MA -> 次日收盘平仓'
    },
    'risk_management': {
        'position_size': '可用资金 * 90% * 2倍杠杆',
        'max_position': '单一品种不超过总资金30%',
        'transaction_cost': '滑点0.02% + 手续费0.03%（双边）'
    }
}
