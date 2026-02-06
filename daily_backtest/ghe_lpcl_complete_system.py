# -*- coding: utf-8 -*-
"""
基于论文《Log-periodic power law and generalized hurst exponent analysis》的完整泡沫预测系统
核心组件：
1. 广义赫斯特指数(GHE) - 判断市场效率和羊群效应
2. LPPL多窗口聚类分析 - 预测崩盘时间区间
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta, datetime
import warnings
import os
import sys

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 第一部分：广义赫斯特指数 (GHE) 计算
# ==============================================================================

def calculate_ghe_single_window(series, q=2, max_tau=20):
    """
    计算单个窗口的广义赫斯特指数

    参数:
        series: 价格序列
        q: 阶矩，通常使用q=2（方差标度）
        max_tau: 最大时间间隔

    返回:
        H_q: q阶的赫斯特指数
    """
    # 对数价格
    log_price = np.log(series.values)

    # 计算不同时间间隔的q阶矩
    taus = np.arange(2, max_tau + 1)
    K_q = []

    for tau in taus:
        # 计算tau时间间隔的对数增量
        diff = np.abs(log_price[tau:] - log_price[:-tau])
        # q阶矩
        K_q.append(np.mean(diff ** q))

    # 对数坐标系下的线性拟合
    log_taus = np.log(taus)
    log_K_q = np.log(K_q)

    # 斜率 = q * H(q)，所以 H(q) = 斜率 / q
    slope, _ = np.polyfit(log_taus, log_K_q, 1)
    H_q = slope / q

    return H_q


def calculate_rolling_ghe(df, window_size=100, step=5, q=2):
    """
    计算滚动GHE

    参数:
        df: 包含收盘价的数据框
        window_size: 窗口大小（天数）
        step: 步长
        q: 阶矩

    返回:
        ghe_series: GHE时间序列
    """
    ghe_values = []
    dates = []

    prices = df['收盘价']

    for i in range(window_size, len(prices), step):
        window = prices.iloc[i - window_size:i]
        try:
            h_q = calculate_ghe_single_window(window, q=q)
            ghe_values.append(h_q)
            dates.append(df.iloc[i]['datetime'])
        except:
            continue

    return pd.DataFrame({'datetime': dates, 'GHE': ghe_values})


# ==============================================================================
# 第二部分：LPPL核心引擎
# ==============================================================================

def lppl_func(t, A, B, tc, m, C, omega, phi):
    """
    LPPL核心公式

    参数:
        t: 时间序列
        A: 长期价格基准
        B: 价格偏移幅度
        tc: 临界时间（崩盘时间）
        m: 价格指数增长指数
        C: 振荡幅度
        omega: 振荡频率
        phi: 相位
    """
    dt = np.clip(tc - t, 1e-8, None)
    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))


def fit_lppl_single_window(prices, window_size, start_idx):
    """
    单个窗口的LPPL拟合
    严格遵循Sornette参数约束
    """
    # 对数价格
    log_prices = np.log(prices.values)
    t = np.arange(len(log_prices))

    # 初始猜测
    A0 = log_prices.max()
    B0 = -0.1
    tc0 = t[-1] + 30  # 预测未来30天
    m0 = 0.5
    C0 = 0.1
    omega0 = 10.0
    phi0 = 0.0

    p0 = [A0, B0, tc0, m0, C0, omega0, phi0]

    # 参数边界（论文标准）
    bounds = (
        [-np.inf, -np.inf, t[-1], 0.1, -1, 6, -2*np.pi],  # 下界
        [np.inf, 0, t[-1] + 100, 0.9, 1, 13, 2*np.pi]     # 上界
    )

    try:
        popt, pcov = curve_fit(lppl_func, t, log_prices, p0=p0,
                               bounds=bounds, maxfev=5000, method='trf')

        # 提取参数
        A, B, tc, m, C, omega, phi = popt

        # 参数有效性检查（Sornette标准）
        if not (0.1 < m < 0.9):
            return None
        if not (6 < omega < 13):
            return None
        if B >= 0:
            return None

        # 计算tc在全局时间轴上的位置
        tc_global = start_idx + tc

        return {
            'tc_global': tc_global,
            'tc_relative': tc,
            'A': A,
            'B': B,
            'm': m,
            'C': C,
            'omega': omega,
            'phi': phi,
            'window_size': window_size
        }
    except:
        return None


def multi_window_lpcl_clustering(df, min_window=100, max_window=300, window_step=20):
    """
    多窗口LPPL聚类分析
    变换窗口大小和起始点，生成tc预测的分布

    参数:
        df: 数据框
        min_window: 最小窗口大小
        max_window: 最大窗口大小
        window_step: 窗口步长

    返回:
        tc_predictions: 所有有效拟合的tc预测
    """
    prices = df['收盘价']
    tc_predictions = []

    print("\n开始多窗口LPPL聚类分析...")
    print(f"窗口范围: {min_window} - {max_window}天，步长: {window_step}天")

    # 固定窗口终点，变换窗口大小
    end_idx = len(prices)

    for window_size in range(min_window, max_window + 1, window_step):
        start_idx = end_idx - window_size

        if start_idx < min_window:
            continue

        window_prices = prices.iloc[start_idx:end_idx]

        result = fit_lppl_single_window(window_prices, window_size, start_idx)

        if result is not None:
            tc_predictions.append(result)

            if window_size % 50 == 0:
                print(f"  窗口{window_size}天: tc预测在第{int(result['tc_global'])}天")

    print(f"\n有效拟合数量: {len(tc_predictions)}")

    return tc_predictions


# ==============================================================================
# 第三部分：综合分析和可视化
# ==============================================================================

def analyze_ghe_signal(ghe_df):
    """
    分析GHE信号

    返回:
        signal: 'SAFE', 'WARNING', 'DANGER'
    """
    latest_ghe = ghe_df['GHE'].iloc[-1]
    ghe_trend = ghe_df['GHE'].iloc[-20:].mean() - ghe_df['GHE'].iloc[-40:-20].mean()

    if latest_ghe > 0.55:
        return 'SAFE', latest_ghe, "强持久性，健康上涨"
    elif latest_ghe > 0.45:
        if ghe_trend < 0:
            return 'WARNING', latest_ghe, "GHE下降，需警惕"
        else:
            return 'SAFE', latest_ghe, "稳定区间"
    else:
        return 'DANGER', latest_ghe, "跌破0.5，反持久性，崩盘风险高"


def analyze_tc_clustering(tc_predictions, total_days):
    """
    分析tc预测的聚类情况

    返回:
        clusters: 检测到的时间聚类
    """
    if len(tc_predictions) == 0:
        return []

    # 提取所有tc预测
    tc_values = [p['tc_global'] for p in tc_predictions]

    # 计算tc分布
    hist, bins = np.histogram(tc_values, bins=30)

    # 找出波峰
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(hist, height=len(tc_predictions) * 0.1, distance=3)

    clusters = []
    for peak_idx in peaks:
        cluster_center = (bins[peak_idx] + bins[peak_idx + 1]) / 2
        cluster_count = hist[peak_idx]
        cluster_width = bins[peak_idx + 1] - bins[peak_idx]

        # 转换为日期
        cluster_date = cluster_center

        clusters.append({
            'center_day': int(cluster_date),
            'count': int(cluster_count),
            'width_days': int(cluster_width),
            'percentage': cluster_count / len(tc_predictions) * 100
        })

    # 按数量排序
    clusters.sort(key=lambda x: x['count'], reverse=True)

    return clusters


def plot_complete_system(df, ghe_df, tc_predictions, clusters):
    """
    绘制完整系统的分析图表
    """
    fig = plt.figure(figsize=(18, 12))

    # 创建3x2的网格
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. 价格走势
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['datetime'], df['收盘价'], color='black', linewidth=1.5, label='沪锡价格')
    ax1.set_title('沪锡10年价格走势', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 2. GHE走势
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(ghe_df['datetime'], ghe_df['GHE'], color='purple', linewidth=2, label='GHE')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='效率阈值(0.5)')
    ax2.axhline(y=0.55, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(y=0.45, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax2.fill_between(ghe_df['datetime'], 0.45, 0.55, color='yellow', alpha=0.1, label='观察区')
    ax2.set_title('广义赫斯特指数(GHE) - 市场效率监测', fontsize=12, fontweight='bold')
    ax2.set_ylabel('GHE值', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 3. tc预测直方图
    ax3 = fig.add_subplot(gs[1, 1])
    if len(tc_predictions) > 0:
        tc_values = [p['tc_global'] for p in tc_predictions]

        # 只显示未来的tc预测
        future_tc = [tc for tc in tc_values if tc < len(df) + 200]

        if len(future_tc) > 0:
            n, bins, patches = ax3.hist(future_tc, bins=30, color='orange', alpha=0.7, edgecolor='black')
            ax3.set_title('LPPL tc预测分布（未来200天）', fontsize=12, fontweight='bold')
            ax3.set_xlabel('全局天数', fontsize=11)
            ax3.set_ylabel('频次', fontsize=11)
            ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

            # 标记聚类
            if clusters:
                for i, cluster in enumerate(clusters[:3]):
                    if cluster['count'] > len(tc_predictions) * 0.15:
                        ax3.axvline(x=cluster['center_day'], color='red',
                                   linestyle='--', linewidth=2,
                                   label=f"聚类{i+1}: {cluster['count']}次({cluster['percentage']:.1f}%)")
                ax3.legend(loc='upper right', fontsize=8)

    # 4. GHE与价格的关系
    ax4 = fig.add_subplot(gs[2, 0])

    # 对齐数据
    price_aligned = []
    ghe_aligned = []

    for dt in ghe_df['datetime']:
        if dt in df['datetime'].values:
            idx = df[df['datetime'] == dt].index[0]
            price_aligned.append(df.loc[idx, '收盘价'])
            ghe_aligned.append(ghe_df[ghe_df['datetime'] == dt]['GHE'].values[0])

    if len(price_aligned) > 0:
        scatter = ax4.scatter(ghe_aligned, price_aligned, c=range(len(ghe_aligned)),
                              cmap='viridis', alpha=0.6, s=30)
        ax4.set_xlabel('GHE值', fontsize=11)
        ax4.set_ylabel('价格', fontsize=11)
        ax4.set_title('GHE vs 价格散点图', fontsize=12, fontweight='bold')
        ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='效率阈值')
        ax4.grid(True, linestyle='--', alpha=0.3)
        ax4.legend()

    # 5. 综合结论文本
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # GHE信号分析
    ghe_signal, latest_ghe, ghe_msg = analyze_ghe_signal(ghe_df)

    conclusion_text = f"""
═════════════════════════════════════════════
        论文理论综合分析结论
═════════════════════════════════════════════

【Step 1: GHE市场效率检测】
最新GHE值: {latest_ghe:.3f}
信号等级: {ghe_signal}
解读: {ghe_msg}

【Step 2: LPPL tc聚类分析】
有效拟合数: {len(tc_predictions)}次

"""

    if clusters:
        conclusion_text += "检测到的时间聚类:\n"
        for i, cluster in enumerate(clusters[:3]):
            days_from_now = cluster['center_day'] - len(df)
            conclusion_text += f"  聚类{i+1}: 未来{days_from_now}天左右 ({cluster['percentage']:.1f}%的预测集中)\n"
    else:
        conclusion_text += "未检测到明显的tc聚类\n"

    conclusion_text += """
【综合建议】
"""

    if ghe_signal == 'SAFE' and not clusters:
        conclusion_text += "✓ 市场处于健康上涨期，GHE>0.5，无泡沫信号"
        conclusion_text += "\n✓ 建议: 继续持仓，顺势而为"
    elif ghe_signal == 'WARNING':
        conclusion_text += "⚠ GHE开始下降，需密切监控"
        conclusion_text += "\n⚠ 建议: 收紧止损，停止加仓"
    elif ghe_signal == 'DANGER':
        conclusion_text += "⚠⚠⚠ GHE跌破0.5，反持久性信号!"
        if clusters:
            conclusion_text += "\n⚠⚠⚠ 同时检测到tc聚类，双重确认!"
            conclusion_text += "\n✗ 建议: 果断减仓或离场"
        else:
            conclusion_text += "\n⚠ 建议: 警惕，但暂无LPPL确认"
    elif clusters:
        conclusion_text += "⚠ 虽GHE安全，但LPPL显示tc聚类"
        conclusion_text += "\n⚠ 建议: 关注聚类时间窗口，准备应对"

    conclusion_text += "\n═════════════════════════════════════════════"

    ax5.text(0.05, 0.95, conclusion_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('基于GHE+LPCL的泡沫预测系统（论文复刻版）',
                 fontsize=16, fontweight='bold', y=0.995)

    output_file = 'D:/期货数据/铜期货监控/daily_backtest/ghe_lppl_complete_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    print("="*80)
    print("基于GHE+LPCL的泡沫预测完整系统")
    print("论文：Log-periodic power law and generalized hurst exponent analysis")
    print("="*80)

    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 天")

    # 第一步：计算滚动GHE
    print("\n" + "="*80)
    print("【Step 1】计算广义赫斯特指数(GHE)")
    print("="*80)

    ghe_df = calculate_rolling_ghe(df, window_size=100, step=5, q=2)

    print(f"\nGHE统计:")
    print(f"  平均值: {ghe_df['GHE'].mean():.3f}")
    print(f"  最新值: {ghe_df['GHE'].iloc[-1]:.3f}")
    print(f"  最小值: {ghe_df['GHE'].min():.3f}")
    print(f"  最大值: {ghe_df['GHE'].max():.3f}")

    # 第二步：多窗口LPPL聚类分析
    print("\n" + "="*80)
    print("【Step 2】多窗口LPPL聚类分析")
    print("="*80)

    tc_predictions = multi_window_lpcl_clustering(df, min_window=100, max_window=300, window_step=20)

    # 分析聚类
    clusters = analyze_tc_clustering(tc_predictions, len(df))

    if clusters:
        print(f"\n检测到 {len(clusters)} 个时间聚类:")
        for i, cluster in enumerate(clusters[:5]):
            days_from_now = cluster['center_day'] - len(df)
            print(f"  聚类{i+1}: 未来{days_from_now}天左右 "
                  f"({cluster['count']}次预测, {cluster['percentage']:.1f}%)")
    else:
        print("\n未检测到明显的tc聚类")

    # 第三步：综合分析
    print("\n" + "="*80)
    print("【Step 3】综合分析")
    print("="*80)

    ghe_signal, latest_ghe, ghe_msg = analyze_ghe_signal(ghe_df)

    print(f"\nGHE信号: {ghe_signal}")
    print(f"  最新GHE: {latest_ghe:.3f}")
    print(f"  解读: {ghe_msg}")

    # 第四步：可视化
    print("\n生成综合分析图表...")
    plot_complete_system(df, ghe_df, tc_predictions, clusters)

    # 保存结果
    ghe_df.to_csv('D:/期货数据/铜期货监控/daily_backtest/ghe_results.csv',
                  index=False, encoding='utf-8-sig')

    if tc_predictions:
        tc_df = pd.DataFrame(tc_predictions)
        tc_df.to_csv('D:/期货数据/铜期货监控/daily_backtest/lppl_tc_predictions.csv',
                     index=False, encoding='utf-8-sig')

    print(f"\n数据已保存:")
    print(f"  - ghe_results.csv: GHE时间序列")
    print(f"  - lppl_tc_predictions.csv: LPPL tc预测数据")

    # 最终结论
    print("\n" + "="*80)
    print("最终结论")
    print("="*80)

    print(f"\n基于论文理论的完整分析:")
    print(f"\n1. 市场效率(GHE):")
    print(f"   当前GHE = {latest_ghe:.3f}")
    if latest_ghe > 0.55:
        print(f"   → 强持久性，市场处于健康上涨趋势")
    elif latest_ghe > 0.45:
        print(f"   → 稳定区间，需观察趋势")
    else:
        print(f"   → 反持久性，市场效率降低，需警惕")

    print(f"\n2. 泡沫信号(LPPL聚类):")
    if clusters:
        print(f"   检测到 {len(clusters)} 个tc聚类")
        main_cluster = clusters[0]
        if main_cluster['percentage'] > 30:
            print(f"   → 主聚类包含{main_cluster['percentage']:.1f}%的预测")
            print(f"   → 高置信度的变盘信号!")
        else:
            print(f"   → 聚类不明显，信号较弱")
    else:
        print(f"   无明显聚类，暂无崩盘预测")

    print(f"\n3. 综合建议:")
    if ghe_signal == 'SAFE' and not clusters:
        print(f"   ✓ 市场健康，无泡沫信号")
        print(f"   ✓ 建议: 继续持仓，顺势交易")
    elif ghe_signal == 'WARNING':
        print(f"   ⚠ GHE下降，需要关注")
        print(f"   ⚠ 建议: 收紧止损")
    elif ghe_signal == 'DANGER' and clusters:
        print(f"   ⚠⚠⚠ GHE跌破0.5 + LPPL聚类")
        print(f"   ⚠⚠⚠ 双重确认的泡沫信号!")
        print(f"   ✗ 建议: 果然减仓/离场")
    else:
        print(f"   信号混合，需要密切监控")

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
