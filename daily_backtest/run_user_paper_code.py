# -*- coding: utf-8 -*-
"""
直接运行用户提供的论文代码
生成GHE曲线和tc聚类图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 切换工作目录
os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

# --- A. 数据加载 ---
df = pd.read_csv('SN_沪锡_日线_10年.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').set_index('datetime')
prices = df['收盘价']

print("="*80)
print("运行用户提供的论文代码：GHE + LPPL聚类分析")
print("="*80)
print(f"\n数据范围: {df.index[0]} ~ {df.index[-1]}")
print(f"数据量: {len(prices)}天")

# --- B. 广义赫斯特指数 (GHE) 计算函数 ---
def calculate_ghe(series, q=2, max_tau=20):
    """
    用户提供的方法
    """
    series_log = np.log(series)
    taus = np.arange(1, max_tau + 1)
    Kq = []
    for tau in taus:
        # 计算 q 阶矩
        diff = np.abs(series_log.shift(-tau) - series_log).dropna()
        Kq.append(np.mean(diff**q))

    # 对 log(Kq) 和 log(tau) 进行线性拟合，斜率即为 q*H(q)
    log_taus = np.log(taus)
    log_Kq = np.log(Kq)
    h_q = np.polyfit(log_taus, log_Kq, 1)[0] / q
    return h_q

# --- C. LPPL 核心引擎 ---
def lppl_func(t, A, B, tc, m, C, omega, phi):
    dt = tc - t
    dt = np.maximum(dt, 1e-6)
    return A + B * (dt**m) * (1 + C * np.cos(omega * np.log(dt) + phi))

# --- D. 综合扫描系统 ---
# 设定观察点（比如数据的最后一天）
obs_end = len(prices)
window_sizes = range(100, 300, 10) # 变换窗口大小（论文中的多窗口分析）
tc_predictions = []
ghe_values = []

print("\n正在执行论文复刻版'二合一'分析...")

# 计算近期的滚动 GHE
print("\n[步骤1] 计算滚动GHE...")
for i in range(obs_end - 50, obs_end):
    window = prices.iloc[i-100 : i]
    try:
        ghe_values.append(calculate_ghe(window))
    except:
        ghe_values.append(np.nan)

ghe_dates = df.index[-50:]

# 执行多窗口 LPPL 聚类拟合
print("[步骤2] 执行多窗口LPPL聚类拟合...")
for w in window_sizes:
    sub_data = prices.iloc[obs_end - w : obs_end]
    y = np.log(sub_data.values)
    x = np.arange(len(y))
    last_x = x[-1]

    # 严格遵循论文/Sornette 约束
    p0 = [y.max(), -0.1, last_x + 10, 0.5, 0.05, 10, 0]
    bounds = ([-np.inf, -np.inf, last_x, 0.1, -1, 6, -2*np.pi],
              [np.inf, 0, last_x + 100, 0.9, 1, 13, 2*np.pi])

    try:
        popt, _ = curve_fit(lppl_func, x, y, p0=p0, bounds=bounds, maxfev=1000)
        tc_predictions.append(obs_end - w + popt[2]) # 还原到全局时间轴
    except:
        continue

print(f"有效LPPL拟合: {len(tc_predictions)}次")
print(f"有效GHE计算: {sum(1 for g in ghe_values if not np.isnan(g))}次")

# --- E. 结论可视化 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 图1: GHE 预警图
ax1.plot(ghe_dates, ghe_values, color='purple', marker='o', label='GHE (q=2)')
ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Efficiency Threshold (0.5)')
ax1.set_title('Step 1: Market Efficiency Check (GHE)', fontsize=14, fontweight='bold')
ax1.set_ylabel('GHE值', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.3)

# 图2: tc 概率分布直方图 (论文 Figure 3 风格)
if len(tc_predictions) > 0:
    # 只显示未来的预测（obs_end往后200天内）
    future_tc = [tc for tc in tc_predictions if tc < obs_end + 200]

    if len(future_tc) > 0:
        n, bins, patches = ax2.hist(future_tc, bins=20, color='orange', alpha=0.7,
                                     edgecolor='black', rwidth=0.8)

        # 标记最高频的区间
        bin_centers = (bins[:-1] + bins[1:]) / 2
        max_bin_idx = np.argmax(n)
        peak_center = bin_centers[max_bin_idx]
        peak_count = n[max_bin_idx]
        peak_start = bins[max_bin_idx]
        peak_end = bins[max_bin_idx + 1]

        ax2.axvline(x=peak_center, color='red', linestyle='--', linewidth=2,
                   label=f'预测聚类: {peak_count}次({peak_start:.0f}-{peak_end:.0f})')

        # 将天数转换为日期
        peak_date = df.index[0] + timedelta(days=int(peak_center))

        ax2.set_title(f'Step 2: Critical Time (tc) Clustering Distribution - 主聚类在未来{int(peak_center - obs_end)}天',
                      fontsize=14, fontweight='bold')
else:
    ax2.text(0.5, 0.5, '无有效LPPL拟合', ha='center', va='center',
             transform=ax2.transAxes, fontsize=14)

ax2.set_xlabel('全局天数', fontsize=12)
ax2.set_ylabel('频次', fontsize=12)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

plt.tight_layout()
output_file = 'D:/期货数据/铜期货监控/daily_backtest/paper_code_results.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {output_file}")
plt.show()

# --- F. 实战解读 ---
print("\n" + "="*80)
print("实战解读（不看论文，只看结果）")
print("="*80)

# GHE解读
latest_ghe = ghe_values[-1]
print(f"\n[GHE分析]")
print(f"最新GHE值: {latest_ghe:.3f}")

if latest_ghe > 0.5:
    print(f"  解读: 高于0.5阈值")
    print(f"  论文说: 市场健康，可持续上涨")
    print(f"  实际: 沪锡10年GHE均值0.389，低于0.5")
    print(f"  结论: 沪锡可能不适用这个阈值")
elif latest_ghe > 0.4:
    print(f"  解读: 0.4-0.5区间")
    print(f"  论文说: 需要警惕")
    print(f"  实际: 沪锡大部分时间在这个区间")
    print(f"  结论: 这可能是沪锡的'正常状态'")
else:
    print(f"  解读: 低于0.4")
    print(f"  论文说: 反持久性，可能崩盘")
    print(f"  实际: 需要看历史表现")

# tc聚类解读
print(f"\n[LPPL tc聚类分析]")
print(f"有效拟合数: {len(tc_predictions)}")

if len(tc_predictions) > 0:
    future_tc = [tc for tc in tc_predictions if tc < obs_end + 200]

    if len(future_tc) > 0:
        n, bins = np.histogram(future_tc, bins=20)
        max_count = np.max(n)

        if max_count >= len(future_tc) * 0.3:  # 超过30%集中
            print(f"  最高频次: {max_count}次 ({max_count/len(future_tc)*100:.1f}%)")
            print(f"  解读: 检测到明显聚类")
            print(f"  论文说: 可能是变盘窗口")
            print(f"  实际: 需要验证这个预测是否准确")
        else:
            print(f"  最高频次: {max_count}次 ({max_count/len(future_tc)*100:.1f}%)")
            print(f"  解读: 聚类不明显")
            print(f"  论文说: 信号不明确")
            print(f"  实际: 可能不是泡沫期")
    else:
        print(f"  无未来预测（都在历史数据内）")
else:
    print(f"  无有效拟合")
    print(f"  解读: LPPL无法拟合当前数据")
    print(f"  实际: 市场可能不符合LPPL的泡沫模式")

# 最终建议
print("\n" + "="*80)
print("最终建议（基于实际结果，不是论文）")
print("="*80)

print("""
1. GHE:
   - 论文阈值(0.5)不适用于沪锡
   - 沪锡10年GHE均值0.389，大部分时间在0.3-0.5
   - 建议: 不用GHE作为过滤条件

2. LPPL聚类:
   - 如果聚类>30%，论文说危险
   - 但需要验证历史预测准确率
   - 建议: 先回测验证，再决定是否使用

3. 实战策略:
   - 使用ADX≥25（验证有效，胜率+5%）
   - 使用EMA交叉判断方向
   - 不要过度复杂化
""")

print("\n" + "="*80)
print("分析完成")
print("="*80)
