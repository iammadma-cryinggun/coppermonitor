# -*- coding: utf-8 -*-
"""
使用用户建议的参数重新拟合LPPL
关键改进：
1. 窗口大小：250天
2. m范围：0.1-0.9 (更保守)
3. omega范围：6-13 (更严格)
4. 简化的LPPL公式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta, datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 数据加载 ---
df = pd.read_csv('SN_沪锡_日线_10年.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print("="*80)
print("LPPL拟合 - 用户建议参数版本")
print("="*80)
print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
print(f"总数据: {len(df)} 天")

# --- 2. 窗口选择 (用户建议250天) ---
window_size = 250
df_fit = df.tail(window_size).copy()
start_date = df_fit['datetime'].iloc[0]
end_date = df_fit['datetime'].iloc[-1]

print(f"\n拟合窗口: {start_date.date()} ~ {end_date.date()} ({window_size}天)")
print(f"窗口内价格: {df_fit['收盘价'].iloc[0]:,.0f} ~ {df_fit['收盘价'].iloc[-1]:,.0f}")
print(f"窗口涨幅: {(df_fit['收盘价'].iloc[-1] / df_fit['收盘价'].iloc[0] - 1) * 100:.1f}%")

# 转换时间
t_data = (df_fit['datetime'] - df_fit['datetime'].iloc[0]).dt.days.values
p_data = df_fit['收盘价'].values
log_p_data = np.log(p_data)

# --- 3. 定义 LPPL 模型公式 (用户版本) ---
def lppl_func(t, A, B, tc, m, C, omega, phi):
    """
    用户版本: A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))

    参数说明:
    - A: 渐近价格水平
    - B: 控制增长幅度 (通常为负)
    - tc: 临界时间
    - m: 幂律指数 (0.1-0.9, 越小越陡峭)
    - C: 震荡幅度
    - omega: 震荡频率 (理想6-13)
    - phi: 相位
    """
    dt = tc - t
    # 避免数学错误
    dt = np.maximum(dt, 1e-8)

    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))

# --- 4. 拟合配置 ---
last_day = t_data[-1]

# 初始猜测 (用户建议)
p0 = [
    np.max(log_p_data),  # A: 当前最高log价格
    -0.1,                # B: 负值表示上升趋势
    last_day + 20,       # tc: 猜测20天后破裂
    0.5,                 # m: 中等幂律指数
    0.05,                # C: 小震荡
    10.0,                # omega: 中等频率
    0.0                  # phi: 零相位
]

# 参数约束 (用户建议的严格范围)
bounds = (
    [-np.inf, -np.inf, last_day, 0.1, -1, 6.0, -2*np.pi],  # 下界
    [np.inf, 0, last_day + 100, 0.9, 1, 13.0, 2*np.pi]      # 上界
)

print(f"\n初始猜测:")
print(f"  tc: 距今{int(p0[2] - last_day)}天")
print(f"  m: {p0[3]}")
print(f"  omega: {p0[5]}")

print(f"\n参数约束:")
print(f"  m: {bounds[0][3]} ~ {bounds[1][3]}")
print(f"  omega: {bounds[0][5]} ~ {bounds[1][5]}")
print(f"  tc: {last_day} ~ {last_day + 100} (未来100天内)")

# --- 5. 执行计算 ---
print(f"\n开始拟合 LPPL 模型...")
try:
    popt, pcov = curve_fit(lppl_func, t_data, log_p_data, p0=p0, bounds=bounds, maxfev=20000)

    # 提取参数
    A_est, B_est, tc_est, m_est, C_est, omega_est, phi_est = popt

    # 转换tc为日期
    tc_date = start_date + timedelta(days=float(tc_est))
    days_to_tc = int(tc_est - last_day)

    # 计算拟合优度
    fitted_values = lppl_func(t_data, *popt)
    residuals = log_p_data - fitted_values
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_p_data - np.mean(log_p_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean(residuals**2))

    print(f"\n" + "="*80)
    print(f"拟合成功！")
    print(f"="*80)

    print(f"\n核心参数:")
    print(f"  预测临界点 (tc): {tc_date.strftime('%Y-%m-%d')}")
    print(f"  距离数据末端: {days_to_tc} 天")
    print(f"  幂律指数 (m): {m_est:.4f}")
    print(f"  震荡频率 (omega): {omega_est:.4f}")
    print(f"  相位 (phi): {phi_est:.4f}")

    print(f"\n拟合质量:")
    print(f"  R2: {r_squared:.4f}")
    print(f"  RMSE: {rmse:.4f}")

    # 风险评估
    print(f"\n" + "="*80)
    print(f"风险评估")
    print(f"="*80)

    # m值判断
    if m_est < 0.3:
        m_level = "极度陡峭 [DANGER]"
    elif m_est < 0.5:
        m_level = "非常陡峭 [WARNING]"
    elif m_est < 0.7:
        m_level = "正常陡峭"
    else:
        m_level = "相对平缓"

    print(f"\nm值分析: {m_est:.4f} -> {m_level}")

    # omega判断
    if 6 <= omega_est <= 13:
        omega_level = "理想区间 [OK]"
    elif omega_est < 6:
        omega_level = "过低 [异常]"
    else:
        omega_level = "过高 [异常]"

    print(f"omega分析: {omega_est:.4f} -> {omega_level}")

    # tc判断
    if days_to_tc < 0:
        tc_status = f"[!!] tc已过 ({abs(days_to_tc)}天前)，如果没破裂则预测失效"
    elif days_to_tc <= 7:
        tc_status = "[!!!] 极高风险！tc在1周内"
    elif days_to_tc <= 30:
        tc_status = "[!!] 高风险！tc在1个月内"
    elif days_to_tc <= 60:
        tc_status = "[!] 中等风险，tc在2个月内"
    else:
        tc_status = "[OK] tc较远，风险相对较低"

    print(f"tc分析: {tc_status}")

    # 综合判断
    print(f"\n综合判断:")
    if days_to_tc > 0 and m_est < 0.5 and 6 <= omega_est <= 13:
        print(f"  [DANGER] 满足LPPL泡沫条件！")
        print(f"  - m={m_est:.4f} < 0.5 (陡峭增长)")
        print(f"  - omega={omega_est:.4f} 在理想区间")
        print(f"  - tc还有{days_to_tc}天")
        print(f"\n  建议: 密切关注，准备减仓或退出")
    elif days_to_tc < 0:
        print(f"  [INFO] tc已过，检查实际价格走势")
        print(f"  如果价格继续上涨，说明此次LPPL预测失效")
    else:
        print(f"  [CAUTION] 参数不完全满足典型LPPL泡沫特征")
        print(f"  建议: 继续观察，但保持警惕")

    # --- 6. 绘图展示 ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 图1: 拟合曲线
    ax1 = axes[0]
    ax1.plot(df_fit['datetime'], log_p_data, 'k.', label='实际价格(Log)', alpha=0.6, markersize=6)

    # 生成拟合曲线 (延伸到tc)
    t_plot = np.linspace(t_data[0], min(tc_est, last_day + 60), 500)
    date_plot = [start_date + timedelta(days=x) for x in t_plot]
    fitted_plot = lppl_func(t_plot, *popt)

    ax1.plot(date_plot, fitted_plot, 'r-',
             linewidth=2.5, label=f'LPPL拟合 (R2={r_squared:.4f})')

    # 标注tc
    ax1.axvline(x=tc_date, color='g', linestyle='--', linewidth=2,
                label=f'预测临界点 {tc_date.strftime("%Y-%m-%d")}')

    # 标注当前数据末端
    ax1.axvline(x=end_date, color='blue', linestyle=':', linewidth=1.5,
                label=f'数据末端 {end_date.strftime("%Y-%m-%d")}')

    ax1.set_title(f'沪锡LPPL拟合分析 (窗口{window_size}天)\n'
                  f'tc={tc_date.strftime("%Y-%m-%d")} | m={m_est:.4f} | ω={omega_est:.4f}',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('对数价格')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 图2: 残差分析
    ax2 = axes[1]
    ax2.plot(df_fit['datetime'], residuals * 100, 'o-', color='purple',
             linewidth=1, markersize=4, label='拟合残差')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('拟合残差分析 (%)', fontsize=12)
    ax2.set_ylabel('残差 (%)')
    ax2.set_xlabel('日期')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'SN_lppl_user_version_m{m_est:.3f}_w{omega_est:.1f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {filename}")

    # 保存结果
    result = {
        'window_start': start_date,
        'window_end': end_date,
        'window_size': window_size,
        'tc': tc_date,
        'days_to_tc': days_to_tc,
        'A': A_est,
        'B': B_est,
        'm': m_est,
        'C': C_est,
        'omega': omega_est,
        'phi': phi_est,
        'r_squared': r_squared,
        'rmse': rmse
    }

    import json
    with open('SN_lppl_user_result.json', 'w', encoding='utf-8') as f:
        json.dump({k: str(v) if isinstance(v, (pd.Timestamp, datetime)) else v
                   for k, v in result.items()}, f, indent=2, ensure_ascii=False)

    print(f"结果已保存: SN_lppl_user_result.json")

except Exception as e:
    print(f"\n拟合失败: {e}")
    import traceback
    traceback.print_exc()
    print(f"\n建议: 尝试调整 window_size 为 200 或 300")

print(f"\n" + "="*80)
print(f"分析完成")
print(f"="*80)
