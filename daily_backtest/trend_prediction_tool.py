# -*- coding: utf-8 -*-
"""
GHE+LPPL趋势预判工具
基于10年沪锡实证数据，用于指导未来交易
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


# ============== 核心计算函数 ==============
def calculate_ghe(series, q=2, max_tau=20):
    """
    计算GHE（广义赫斯特指数）

    参数:
        series: 价格序列（pd.Series）
        q: q阶矩，默认2
        max_tau: 最大时间间隔，默认20

    返回:
        float: GHE值
    """
    log_price = np.log(series.values)
    taus = np.arange(2, max_tau + 1)
    K_q = []

    for tau in taus:
        diff = np.abs(log_price[tau:] - log_price[:-tau])
        K_q.append(np.mean(diff ** q))

    log_taus = np.log(taus)
    log_K_q = np.log(K_q)
    slope, _ = np.polyfit(log_taus, log_K_q, 1)
    H_q = slope / q

    return H_q


def lppl_func(t, A, B, tc, m, C, omega, phi):
    """LPPL核心函数"""
    dt = np.clip(tc - t, 1e-8, None)
    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))


def calculate_lppl_d(prices):
    """
    计算LPPL-D值

    参数:
        prices: 价格序列（pd.Series或np.array），至少200个数据点

    返回:
        float: LPPL-D值，如果拟合失败返回None
    """
    if len(prices) < 200:
        return None

    log_prices = np.log(prices.values if hasattr(prices, 'values') else prices)
    t = np.arange(len(log_prices))
    last_t = t[-1]

    # 初始参数
    A0 = log_prices.max()
    B0 = -0.1
    tc0 = last_t + 30
    m0 = 0.5
    C0 = 0.1
    omega0 = 10.0
    phi0 = 0.0
    p0 = [A0, B0, tc0, m0, C0, omega0, phi0]

    # 参数边界
    bounds = (
        [-np.inf, -np.inf, last_t, 0.1, -1, 6, -2*np.pi],
        [np.inf, 0, last_t + 90, 0.9, 1, 13, 2*np.pi]
    )

    try:
        popt, pcov = curve_fit(lppl_func, t, log_prices, p0=p0,
                               bounds=bounds, maxfev=5000, method='trf')

        A, B, tc, m, C, omega, phi = popt

        # 参数有效性检查
        if not (0.1 < m < 0.9):
            return None
        if not (6 < omega < 13):
            return None
        if B >= 0:
            return None

        # 计算D值
        D = m * omega / (2 * np.pi)

        return D

    except:
        return None


# ============== 核心判断函数 ==============
def predict_future_trend(ghe, lppl_d):
    """
    基于GHE和LPPL-D预测未来20天的趋势

    基于10年沪锡日线数据的实证分析结果

    参数:
        ghe: 当前GHE值
        lppl_d: 当前LPPL-D值

    返回:
        dict: {
            'forecast': 'STRONG_UP'/'STRONG_DOWN'/'RANGING'/'UNCERTAIN',
            'confidence': 0-100,
            'probability': {'STRONG_UP': %, 'STRONG_DOWN': %, 'RANGING': %},
            'action': '交易建议',
            'sample_size': 样本数,
            'warning': '警告信息（如果有）'
        }
    """

    # 检查输入有效性
    if ghe is None or lppl_d is None:
        return {
            'forecast': 'UNCERTAIN',
            'confidence': 0,
            'probability': {'STRONG_UP': 0, 'STRONG_DOWN': 0, 'RANGING': 0},
            'action': '数据不足，无法判断',
            'warning': 'GHE或LPPL-D计算失败'
        }

    # 情况1：强下降信号（基于10年数据：21个样本，52.4%概率）
    if 0.35 <= ghe <= 0.45 and 0.4 <= lppl_d <= 0.6:
        return {
            'forecast': 'STRONG_DOWN',
            'confidence': 52,
            'probability': {
                'STRONG_UP': 28.6,
                'STRONG_DOWN': 52.4,
                'RANGING': 19.0
            },
            'action': '预计未来20天强下降概率52% → 建议：观望或做空',
            'sample_size': 21,
            'details': 'GHE 0.35-0.45 + LPPL-D 0.4-0.6 组合'
        }

    # 情况2：横盘震荡信号（基于10年数据：16个样本，62.5%概率）
    if 0.35 <= ghe <= 0.45 and 0.7 <= lppl_d <= 0.9:
        return {
            'forecast': 'RANGING',
            'confidence': 63,
            'probability': {
                'STRONG_UP': 37.5,
                'STRONG_DOWN': 0.0,
                'RANGING': 62.5
            },
            'action': '预计未来20天横盘震荡概率63% → 建议：区间交易或观望',
            'sample_size': 16,
            'details': 'GHE 0.35-0.45 + LPPL-D 0.7-0.9 组合'
        }

    # 情况3：强上升信号（基于10年数据：18个样本，100%概率）
    if 0.50 <= ghe <= 0.60 and 0.7 <= lppl_d <= 0.9:
        return {
            'forecast': 'STRONG_UP',
            'confidence': 100,
            'probability': {
                'STRONG_UP': 100.0,
                'STRONG_DOWN': 0.0,
                'RANGING': 0.0
            },
            'action': '预计未来20天强上升概率100% → 建议：积极做多',
            'sample_size': 18,
            'warning': '⚠️ 样本数较少（18个），建议结合ADX≥25确认趋势',
            'details': 'GHE 0.50-0.60 + LPPL-D 0.7-0.9 组合'
        }

    # 默认：无明确信号，使用基准概率
    return {
        'forecast': 'UNCERTAIN',
        'confidence': 50,
        'probability': {
            'STRONG_UP': 46.3,
            'STRONG_DOWN': 28.7,
            'RANGING': 25.0
        },
        'action': '当前GHE和LPPL组合无明确预测信号 → 建议：使用战术指标（ADX≥25、EMA交叉）决策',
        'warning': f'GHE={ghe:.3f}, LPPL-D={lppl_d:.3f} 不在有效预测区间',
        'details': f'当前值：GHE {ghe:.3f}, LPPL-D {lppl_d:.3f}'
    }


# ============== 实时计算函数 ==============
def analyze_current_market(df, min_data_points=200):
    """
    分析当前市场状况

    参数:
        df: DataFrame，需包含'收盘价'列，按日期升序排列
        min_data_points: 最少数据点数，默认200

    返回:
        dict: 当前市场分析结果
    """
    if len(df) < min_data_points:
        return {
            'status': 'error',
            'message': f'数据不足，需要至少{min_data_points}个数据点，当前只有{len(df)}个'
        }

    # 取最近的数据计算指标
    recent_prices = df['收盘价'].tail(min_data_points)

    # 计算GHE
    try:
        ghe = calculate_ghe(recent_prices, q=2, max_tau=20)
    except:
        return {
            'status': 'error',
            'message': 'GHE计算失败'
        }

    # 计算LPPL-D
    try:
        lppl_d = calculate_lppl_d(recent_prices)
    except:
        return {
            'status': 'error',
            'message': 'LPPL-D计算失败'
        }

    # 预测趋势
    prediction = predict_future_trend(ghe, lppl_d)

    return {
        'status': 'success',
        'current_date': df['datetime'].iloc[-1] if 'datetime' in df.columns else 'Unknown',
        'current_price': df['收盘价'].iloc[-1],
        'ghe': ghe,
        'lppl_d': lppl_d,
        'prediction': prediction
    }


# ============== 打印报告 ==============
def print_analysis_report(analysis_result):
    """
    打印格式化的分析报告

    参数:
        analysis_result: analyze_current_market()的返回结果
    """
    print("=" * 80)
    print("GHE+LPPL趋势预判报告")
    print("=" * 80)

    if analysis_result['status'] == 'error':
        print(f"\n错误: {analysis_result['message']}")
        return

    # 基本信息
    print(f"\n【当前市场状况】")
    print(f"  日期: {analysis_result['current_date']}")
    print(f"  价格: {analysis_result['current_price']:.2f}")
    print(f"  GHE: {analysis_result['ghe']:.3f}")
    print(f"  LPPL-D: {analysis_result['lppl_d']:.3f}")

    # 预测结果
    pred = analysis_result['prediction']
    print(f"\n【趋势预测】")
    print(f"  预测趋势: {pred['forecast']}")
    print(f"  置信度: {pred['confidence']}%")
    print(f"  样本数: {pred.get('sample_size', 'N/A')}")

    print(f"\n【概率分布】")
    print(f"  强上升: {pred['probability']['STRONG_UP']:.1f}%")
    print(f"  强下降: {pred['probability']['STRONG_DOWN']:.1f}%")
    print(f"  横盘震荡: {pred['probability']['RANGING']:.1f}%")

    print(f"\n【交易建议】")
    print(f"  {pred['action']}")

    if 'warning' in pred:
        print(f"\n【⚠️ 注意】")
        print(f"  {pred['warning']}")

    if 'details' in pred:
        print(f"\n【组合详情】")
        print(f"  {pred['details']}")

    print("\n" + "=" * 80)


# ============== 示例使用 ==============
if __name__ == "__main__":
    print("""
GHE+LPPL趋势预判工具

使用示例:
---------
import pandas as pd
from trend_prediction_tool import analyze_current_market, print_analysis_report

# 加载数据
df = pd.read_csv('SN_沪锡_日线_10年.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# 分析当前市场
result = analyze_current_market(df)

# 打印报告
print_analysis_report(result)

# 或者直接获取预测值
ghe = result['ghe']
lppl_d = result['lppl_d']
prediction = result['prediction']

# 根据预测做决策
if prediction['forecast'] == 'STRONG_UP':
    print("建议：积极做多")
elif prediction['forecast'] == 'STRONG_DOWN':
    print("建议：观望或做空")
elif prediction['forecast'] == 'RANGING':
    print("建议：区间交易或观望")
else:
    print("建议：使用战术指标（ADX、EMA交叉）决策")
    """)
