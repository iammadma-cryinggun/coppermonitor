# -*- coding: utf-8 -*-
"""
简化的LPPL拟合 - 使用scipy.optimize.curve_fit
基于用户提供的教程代码
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('SN_沪锡_日线_10年.csv')
df = df.rename(columns={
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low',
    '收盘价': 'close',
    '成交量': 'volume',
    '持仓量': 'hold'
})
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print("="*80)
print("简化版LPPL拟合 - 使用scipy")
print("="*80)

# 选择拟合区间
# 根据诊断结果，使用最近的泡沫区间: 2025-10-21 ~ 2026-01-29
start_date = pd.Timestamp('2025-10-21')
end_date = pd.Timestamp('2026-01-29')

fit_data = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].copy()

print(f"\n拟合区间: {start_date.date()} ~ {end_date.date()}")
print(f"数据点: {len(fit_data)} 天")
print(f"起止价格: {fit_data['close'].iloc[0]:,.0f} ~ {fit_data['close'].iloc[-1]:,.0f}")
print(f"涨幅: {(fit_data['close'].iloc[-1] / fit_data['close'].iloc[0] - 1) * 100:.1f}%")

# 准备LPPL拟合数据
# 将时间转换为距起点的天数
t = np.arange(len(fit_data))
# 对数价格
log_price = np.log(fit_data['close'].values)

# 定义LPPL函数
def lppl_func(t, tc, a, b, c, m, w, phi):
    """
    LPPL模型: ln(p(t)) = A + B*(tc-t)^m + C*(tc-t)^m * cos(w*ln(tc-t) - phi)

    参数:
    - tc: 临界时间（crash time）
    - a: A参数，log(tc)处的渐近价格
    - b: B参数，控制价格增长幅度
    - c: C参数，控制振荡幅度
    - m: 超指数增长程度 (m > 1 表示泡沫)
    - w: 振荡频率
    - phi: 相位
    """
    # 避免除零和负数
    dt = tc - t
    dt = np.maximum(dt, 0.1)  # 防止tc-t <= 0

    term1 = a
    term2 = b * (dt ** m)
    term3 = c * (dt ** m) * np.cos(w * np.log(dt) - phi)

    return term1 + term2 + term3

# 参数初始猜测
# tc: 估计在数据终点之后不久
# a: 当前log价格附近
# b: 负值，表示上升趋势
# c: 控制振荡
# m: 0.5-1.2之间
# w: 6-12之间
# phi: 0-2π之间
initial_guess = [
    len(t) + 20,      # tc: 数据终点后20天
    13.0,              # a: log价格水平
    -0.5,              # b: 负值表示上升
    0.1,               # c: 振荡幅度
    0.8,               # m: 超指数指数
    10.0,              # w: 频率
    0.0                # phi: 相位
]

# 参数边界
# tc: 必须大于数据长度（在未来）
# a: log价格的合理范围
# b: 负值（上升趋势）
# c: 任意
# m: 0.1-1.5
# w: 1-20
# phi: 0-2π
bounds = (
    [len(t), 10, -5, -1, 0.1, 1, -2*np.pi],      # 下界
    [len(t)+200, 15, 0, 1, 1.5, 20, 2*np.pi]    # 上界
)

print(f"\n开始LPPL拟合...")
print(f"初始猜测: tc={initial_guess[0]:.1f}, a={initial_guess[1]:.2f}, b={initial_guess[2]:.2f}, m={initial_guess[4]:.2f}, w={initial_guess[5]:.2f}")

try:
    # 执行拟合
    popt, pcov = curve_fit(lppl_func, t, log_price, p0=initial_guess, bounds=bounds, maxfev=10000)

    # 提取参数
    tc_fit, a_fit, b_fit, c_fit, m_fit, w_fit, phi_fit = popt

    # 计算D值
    # D = m * w，表示泡沫强度
    D_fit = m_fit * w_fit

    print(f"\n拟合成功!")
    print(f"\n拟合参数:")
    print(f"  tc (临界时间) = {tc_fit:.1f} (距起点天数)")
    print(f"  a (渐近价格) = {a_fit:.4f}")
    print(f"  b (增长幅度) = {b_fit:.4f}")
    print(f"  c (振荡幅度) = {c_fit:.4f}")
    print(f"  m (超指数指数) = {m_fit:.4f}")
    print(f"  w (振荡频率) = {w_fit:.4f}")
    print(f"  phi (相位) = {phi_fit:.4f}")
    print(f"\n推导指标:")
    print(f"  D (泡沫指标) = m * w = {D_fit:.4f}")

    # 判断泡沫状态
    print(f"\n泡沫状态判断:")
    if D_fit < 0.3:
        status = "正常市场"
        level = "SAFE"
    elif D_fit < 0.5:
        status = "泡沫警告"
        level = "CAUTION"
    elif D_fit < 0.8:
        status = "明显泡沫"
        level = "WARNING"
    else:
        status = "强泡沫"
        level = "DANGER"

    print(f"  D值 {D_fit:.4f} -> {status} [{level}]")

    if m_fit > 1:
        print(f"  m值 {m_fit:.4f} > 1 -> 超指数增长确认")
    else:
        print(f"  m值 {m_fit:.4f} < 1 -> 非超指数增长")

    # 计算临界时间对应的日期
    start_datetime = fit_data['datetime'].iloc[0]
    tc_datetime = start_datetime + pd.Timedelta(days=int(tc_fit))
    print(f"  预测临界点: {tc_datetime.date()}")

    # 计算拟合优度
    fitted_values = lppl_func(t, *popt)
    residuals = log_price - fitted_values
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_price - np.mean(log_price))**2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\n拟合优度:")
    print(f"  R2 = {r_squared:.4f}")
    print(f"  RMSE = {np.sqrt(np.mean(residuals**2)):.4f}")

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 图1: 对数价格 + LPPL拟合
    ax1 = axes[0]
    ax1.plot(fit_data['datetime'], log_price, 'o-', label='实际对数价格', linewidth=2, markersize=4)
    ax1.plot(fit_data['datetime'], fitted_values, '--',
             label=f'LPPL拟合 (R2={r_squared:.4f})', linewidth=2, color='red')

    # 标注临界点
    ax1.axvline(tc_datetime, color='orange', linestyle=':', linewidth=2,
                label=f'预测临界点 {tc_datetime.date()}')

    ax1.set_title(f'沪锡LPPL拟合分析 ({start_date.date()} ~ {end_date.date()})\nD={D_fit:.4f} [{level}]  m={m_fit:.4f}  tc={tc_datetime.date()}',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('ln(价格)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 图2: 残差
    ax2 = axes[1]
    ax2.plot(fit_data['datetime'], residuals * 100, 'o-', label='拟合残差', linewidth=1, markersize=4)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('拟合残差 (%)', fontsize=12)
    ax2.set_ylabel('残差 (%)')
    ax2.set_xlabel('日期')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('SN_lppl_scipy_fit.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: SN_lppl_scipy_fit.png")

    # 导出参数到CSV
    result_df = pd.DataFrame([{
        'start_date': start_date,
        'end_date': end_date,
        'data_points': len(fit_data),
        'tc': tc_fit,
        'tc_datetime': tc_datetime,
        'a': a_fit,
        'b': b_fit,
        'c': c_fit,
        'm': m_fit,
        'w': w_fit,
        'phi': phi_fit,
        'D': D_fit,
        'status': status,
        'level': level,
        'r_squared': r_squared,
        'rmse': np.sqrt(np.mean(residuals**2))
    }])

    result_df.to_csv('SN_lppl_fit_result.csv', index=False, encoding='utf-8-sig')
    print(f"拟合结果已保存: SN_lppl_fit_result.csv")

except Exception as e:
    print(f"\n拟合失败: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("LPPL拟合完成")
print("="*80)
