# -*- coding: utf-8 -*-
"""
滚动LPPL拟合 - 生成动态D值指标
用于回测验证LPPL过滤效果
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
print("滚动LPPL拟合 - 生成动态D值")
print("="*80)

# 参数设置
WINDOW_SIZE = 100  # 拟合窗口（天）
ROLL_STEP = 20     # 滚动步长（天）
MIN_DATA = 60      # 最小数据点

print(f"\n参数设置:")
print(f"  窗口大小: {WINDOW_SIZE} 天")
print(f"  滚动步长: {ROLL_STEP} 天")
print(f"  最小数据: {MIN_DATA} 天")


def lppl_func(t, tc, a, b, c, m, w, phi):
    """LPPL模型函数"""
    dt = tc - t
    dt = np.maximum(dt, 0.1)
    return a + b * (dt ** m) + c * (dt ** m) * np.cos(w * np.log(dt) - phi)


def fit_lppl_window(start_idx, end_idx):
    """拟合单个窗口"""
    try:
        window_data = df.iloc[start_idx:end_idx+1].copy()

        # 准备数据
        t = np.arange(len(window_data))
        log_price = np.log(window_data['close'].values)

        # 初始猜测
        initial_guess = [
            len(t) + 20,      # tc
            13.0,              # a
            -0.5,              # b
            0.1,               # c
            0.8,               # m
            10.0,              # w
            0.0                # phi
        ]

        # 边界
        bounds = (
            [len(t), 10, -5, -1, 0.1, 1, -2*np.pi],
            [len(t)+200, 15, 0, 1, 1.5, 20, 2*np.pi]
        )

        # 拟合
        popt, pcov = curve_fit(lppl_func, t, log_price,
                               p0=initial_guess, bounds=bounds,
                               maxfev=5000)

        tc_fit, a_fit, b_fit, c_fit, m_fit, w_fit, phi_fit = popt
        D_fit = m_fit * w_fit

        # 计算拟合优度
        fitted_values = lppl_func(t, *popt)
        residuals = log_price - fitted_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_price - np.mean(log_price))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # 转换tc为日期
        start_datetime = window_data['datetime'].iloc[0]
        tc_datetime = start_datetime + pd.Timedelta(days=int(tc_fit))

        return {
            'date': window_data['datetime'].iloc[-1],
            'start_date': window_data['datetime'].iloc[0],
            'end_date': window_data['datetime'].iloc[-1],
            'tc': tc_fit,
            'tc_datetime': tc_datetime,
            'a': a_fit,
            'b': b_fit,
            'c': c_fit,
            'm': m_fit,
            'w': w_fit,
            'phi': phi_fit,
            'D': D_fit,
            'r_squared': r_squared,
            'window_size': len(window_data)
        }

    except Exception as e:
        return None


# 执行滚动拟合
results = []
total_windows = (len(df) - WINDOW_SIZE) // ROLL_STEP + 1

print(f"\n开始滚动拟合...")
print(f"预计拟合 {total_windows} 个窗口")

for i in range(0, len(df) - WINDOW_SIZE + 1, ROLL_STEP):
    start_idx = i
    end_idx = i + WINDOW_SIZE - 1

    if end_idx >= len(df):
        break

    result = fit_lppl_window(start_idx, end_idx)

    if result is not None:
        results.append(result)
        progress = len(results) / total_windows * 100
        print(f"[{progress:5.1f}%] {result['date'].date()} | "
              f"D={result['D']:.4f} | m={result['m']:.4f} | "
              f"tc={result['tc_datetime'].date()} | R2={result['r_squared']:.4f}")

# 保存结果
if results:
    lppl_df = pd.DataFrame(results)

    # 保存到CSV
    lppl_df.to_csv('SN_lppl_rolling.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存: SN_lppl_rolling.csv")

    # 统计分析
    print(f"\nD值分布统计:")
    print(f"  总拟合次数: {len(lppl_df)}")

    D_series = lppl_df['D']
    print(f"\n  D < 0.3: {len(D_series[D_series < 0.3])} ({len(D_series[D_series < 0.3])/len(D_series)*100:.1f}%)")
    print(f"  0.3 ≤ D < 0.5: {len(D_series[(D_series >= 0.3) & (D_series < 0.5)])} ({len(D_series[(D_series >= 0.3) & (D_series < 0.5)])/len(D_series)*100:.1f}%)")
    print(f"  0.5 ≤ D < 0.8: {len(D_series[(D_series >= 0.5) & (D_series < 0.8)])} ({len(D_series[(D_series >= 0.5) & (D_series < 0.8)])/len(D_series)*100:.1f}%)")
    print(f"  D ≥ 0.8: {len(D_series[D_series >= 0.8])} ({len(D_series[D_series >= 0.8])/len(D_series)*100:.1f}%)")

    print(f"\nm值分布统计:")
    m_series = lppl_df['m']
    print(f"  m < 1: {len(m_series[m_series < 1])} ({len(m_series[m_series < 1])/len(m_series)*100:.1f}%)")
    print(f"  m ≥ 1: {len(m_series[m_series >= 1])} ({len(m_series[m_series >= 1])/len(m_series)*100:.1f}%)")

    # 绘图
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 图1: 价格 + D值
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ax1.plot(df['datetime'], df['close'], 'o-', color='blue',
             linewidth=1, markersize=2, label='价格', alpha=0.6)
    ax1_twin.plot(lppl_df['date'], lppl_df['D'], 'r-', linewidth=2,
                  label='D值', marker='o', markersize=4)
    ax1_twin.axhline(0.3, color='orange', linestyle='--', alpha=0.5)
    ax1_twin.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax1_twin.axhline(0.8, color='purple', linestyle='--', alpha=0.5)

    ax1.set_ylabel('价格 (元/吨)', color='blue')
    ax1_twin.set_ylabel('D值', color='red')
    ax1.set_title('滚动LPPL - D值走势', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 图2: m值
    ax2 = axes[1]
    ax2.plot(lppl_df['date'], lppl_df['m'], 'g-', linewidth=2,
             label='m值', marker='o', markersize=4)
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='m=1阈值')

    ax2.set_ylabel('m值')
    ax2.set_title('滚动LPPL - m值走势 (超指数增长指数)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: R²
    ax3 = axes[2]
    ax3.plot(lppl_df['date'], lppl_df['r_squared'], 'purple',
             linewidth=2, label='R²', marker='o', markersize=4)
    ax3.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='R²=0.9')

    ax3.set_ylabel('R²')
    ax3.set_xlabel('日期')
    ax3.set_title('滚动LPPL - 拟合优度', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('SN_lppl_rolling_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n分析图表已保存: SN_lppl_rolling_analysis.png")

    # 生成可用于回测的LPPL信号
    print(f"\n生成LPPL交易信号...")

    # 为每个交易日分配最近的D值
    df_lppl = df.copy()
    df_lppl['D'] = np.nan
    df_lppl['m'] = np.nan
    df_lppl['r_squared'] = np.nan

    for _, row in lppl_df.iterrows():
        mask = df_lppl['datetime'] == row['date']
        df_lppl.loc[mask, 'D'] = row['D']
        df_lppl.loc[mask, 'm'] = row['m']
        df_lppl.loc[mask, 'r_squared'] = row['r_squared']

    # 前向填充（用最近的D值填充空值）
    df_lppl['D'] = df_lppl['D'].fillna(method='ffill')
    df_lppl['m'] = df_lppl['m'].fillna(method='ffill')
    df_lppl['r_squared'] = df_lppl['r_squared'].fillna(method='ffill')

    # 生成交易信号
    df_lppl['lppl_signal'] = '正常'
    df_lppl.loc[df_lppl['D'] >= 0.3, 'lppl_signal'] = '警告'
    df_lppl.loc[df_lppl['D'] >= 0.5, 'lppl_signal'] = '危险'
    df_lppl.loc[df_lppl['D'] >= 0.8, 'lppl_signal'] = '极度危险'

    # 保存
    df_lppl[['datetime', 'close', 'D', 'm', 'r_squared', 'lppl_signal']].to_csv(
        'SN_lppl_daily_signals.csv', index=False, encoding='utf-8-sig')
    print(f"每日LPPL信号已保存: SN_lppl_daily_signals.csv")

    print(f"\n信号分布:")
    print(df_lppl['lppl_signal'].value_counts())

else:
    print("\n未能完成任何拟合!")

print("\n" + "="*80)
print("滚动LPPL分析完成")
print("="*80)
