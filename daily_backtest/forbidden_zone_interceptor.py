# -*- coding: utf-8 -*-
"""
交易禁区拦截器
既然无法预测趋势，那就避免在已知的高风险时期交易
"""

import pandas as pd
import numpy as np
from trend_prediction_tool import calculate_ghe, calculate_lppl_d
import os

os.chdir(r"D:\期货数据\铜期货监控\daily_backtest")


class ForbiddenZoneManager:
    """
    交易禁区拦截器
    基于历史镜像验证数据，定义低胜率组合为"禁区"
    """

    # 基于历史镜像验证结果定义的黑名单
    # 格式：(GHE范围, LPPL-D范围, 历史胜率, 平均收益, 风险描述)
    BLACK_LIST = [
        {
            'ghe_range': (0.36, 0.40),
            'lppl_range': (0.71, 0.75),
            'win_rate': 0.273,
            'avg_return': -0.96,
            'sample_size': 33,
            'severity': 'HIGH',
            'desc': '均值回归陷阱（沪锡平均状态）'
        },
        {
            'ghe_range': (0.38, 0.42),
            'lppl_range': (0.73, 0.77),
            'win_rate': 0.273,
            'avg_return': -0.96,
            'sample_size': 33,
            'severity': 'HIGH',
            'desc': '典型负收益区间'
        },
        {
            'ghe_range': (0.35, 0.45),
            'lppl_range': (0.65, 0.85),
            'win_rate': 0.375,
            'avg_return': -0.76,
            'sample_size': 40,
            'severity': 'MEDIUM',
            'desc': '中等GHE + 高LPPL'
        },
    ]

    @classmethod
    def check_status(cls, current_ghe, current_lppl_d):
        """
        检查当前状态是否处于禁区

        返回：
            dict: {
                'is_forbidden': bool,
                'reason': str,
                'historical_win_rate': float,
                'historical_avg_return': float,
                'severity': 'HIGH' or 'MEDIUM',
                'suggestion': str
            }
        """
        for zone in cls.BLACK_LIST:
            g_min, g_max = zone['ghe_range']
            l_min, l_max = zone['lppl_range']

            if g_min <= current_ghe <= g_max and l_min <= current_lppl_d <= l_max:
                return {
                    'is_forbidden': True,
                    'reason': zone['desc'],
                    'historical_win_rate': zone['win_rate'],
                    'historical_avg_return': zone['avg_return'],
                    'sample_size': zone['sample_size'],
                    'severity': zone['severity'],
                    'suggestion': cls._get_suggestion(zone['severity'])
                }

        # 如果不在黑名单，但胜率也一般（40-50%）
        return {
            'is_forbidden': False,
            'reason': '不在明确禁区',
            'suggestion': '可正常交易，但建议结合其他指标'
        }

    @classmethod
    def _get_suggestion(cls, severity):
        if severity == 'HIGH':
            return "强制空仓，屏蔽一切买入信号！如持有仓位，建议减仓或止损。"
        else:
            return "谨慎交易，降低仓位，严格止损。"


def analyze_current_with_filter(df):
    """
    分析当前市场状态，并应用禁区拦截器

    参数：
        df: DataFrame，需包含'收盘价'列

    返回：
        dict: 分析结果
    """
    # 计算当前GHE和LPPL-D
    if len(df) < 200:
        return {
            'status': 'error',
            'message': f'数据不足，需要至少200天，当前只有{len(df)}天'
        }

    recent_prices = df['收盘价'].tail(200)

    try:
        current_ghe = calculate_ghe(recent_prices)
    except:
        return {
            'status': 'error',
            'message': 'GHE计算失败'
        }

    try:
        current_lppl_d = calculate_lppl_d(recent_prices)
    except:
        return {
            'status': 'error',
            'message': 'LPPL-D计算失败'
        }

    # 应用禁区拦截器
    filter_status = ForbiddenZoneManager.check_status(current_ghe, current_lppl_d)

    return {
        'status': 'success',
        'current_date': df['datetime'].iloc[-1] if 'datetime' in df.columns else 'Unknown',
        'current_price': df['收盘价'].iloc[-1],
        'current_ghe': current_ghe,
        'current_lppl_d': current_lppl_d,
        'filter_status': filter_status
    }


def print_interceptor_report(analysis_result):
    """
    打印格式化的拦截器报告
    """
    print("="*80)
    print("交易禁区拦截器报告")
    print("="*80)

    if analysis_result['status'] == 'error':
        print(f"\n错误: {analysis_result['message']}")
        return

    # 基本信息
    print(f"\n【当前市场状况】")
    print(f"  日期: {analysis_result['current_date']}")
    print(f"  价格: {analysis_result['current_price']:.2f}")
    print(f"  GHE: {analysis_result['current_ghe']:.3f}")
    print(f"  LPPL-D: {analysis_result['current_lppl_d']:.3f}")

    # 拦截器结果
    filter_status = analysis_result['filter_status']

    print(f"\n{'='*80}")
    print("拦截器决策")
    print(f"{'='*80}")

    if filter_status['is_forbidden']:
        print(f"\n[!] 警告：处于交易禁区！")
        print(f"  禁区类型: {filter_status['reason']}")
        print(f"  风险级别: {filter_status['severity']}")
        print(f"  历史胜率: {filter_status['historical_win_rate']*100:.1f}%")
        print(f"  历史平均收益: {filter_status['historical_avg_return']:.2f}%")
        print(f"  样本数: {filter_status['sample_size']}")
        print(f"\n【操作建议】")
        print(f"  {filter_status['suggestion']}")
    else:
        print(f"\n[OK] 当前不在明确禁区")
        print(f"  状态: {filter_status['reason']}")
        print(f"\n【操作建议】")
        print(f"  {filter_status['suggestion']}")

    print(f"\n{'='*80}")
    print("补充说明")
    print(f"{'='*80}")

    print(f"""
这个拦截器的作用：
1. 不是预测什么时候会涨/跌
2. 而是避开历史上证明高概率亏钱的组合
3. 基于历史镜像验证的实证数据

局限性：
1. 只能避开少数明确禁区
2. 大部分时间还是"模糊区"（40-60%胜率）
3. 需要结合其他指标（EMA、ADX、成交量）

核心思想：
"知道什么时候不交易，比知道什么时候交易更值钱"
""")

    print("\n" + "="*80)
    print("报告完成")
    print("="*80)


# ============== 使用示例 ==============
if __name__ == "__main__":
    print("""
交易禁区拦截器 - 使用示例
""")

    # 读取数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 分析当前状态
    result = analyze_current_with_filter(df)

    # 打印报告
    print_interceptor_report(result)

    # ============== 测试几个典型状态 ==============
    print(f"\n{'='*80}")
    print("测试几个典型状态的拦截结果")
    print(f"{'='*80}")

    test_cases = [
        {'ghe': 0.38, 'lppl': 0.73, 'desc': '沪锡平均状态（禁区）'},
        {'ghe': 0.40, 'lppl': 0.75, 'desc': '中等GHE + 高LPPL（禁区）'},
        {'ghe': 0.30, 'lppl': 0.25, 'desc': '低GHE + 低LPPL（非禁区）'},
        {'ghe': 0.50, 'lppl': 0.85, 'desc': '高GHE + 高LPPL（非禁区）'},
    ]

    for case in test_cases:
        print(f"\n测试: {case['desc']}")
        print(f"  GHE={case['ghe']}, LPPL-D={case['lppl']}")

        status = ForbiddenZoneManager.check_status(case['ghe'], case['lppl'])

        if status['is_forbidden']:
            print(f"  -> [拦截] {status['reason']} (胜率{status['historical_win_rate']*100:.1f}%)")
        else:
            print(f"  -> [通过] {status['reason']}")
