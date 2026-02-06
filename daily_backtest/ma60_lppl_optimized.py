# -*- coding: utf-8 -*-
"""
MA60+LPPL双模系统 - 优化版
核心改进:
1. 放宽LPPL参数限制，增加信号密度
2. 并行计算加速滚动窗口拟合
3. 多次重启避免局部最优
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from datetime import timedelta
import warnings
import sys
import io

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- LPPL 拟合函数 ---
def lppl_func(t, A, B, tc, m, C, omega, phi):
    dt = np.clip(tc - t, 1e-8, None)  # 防止除以零
    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))


def fit_lppl_once(price_series):
    """单次LPPL拟合"""
    t = np.arange(len(price_series), dtype=float)
    y = np.log(price_series.values.astype(float))
    last = t[-1]

    # 初始猜测
    A0 = y.max()
    B0 = -0.1
    tc0 = last + 30
    m0 = 0.5
    C0 = 0.1
    omega0 = 8.0
    phi0 = 0.0

    p0 = np.array([A0, B0, tc0, m0, C0, omega0, phi0], dtype=float)

    # 放宽的参数边界
    bounds = (
        [-np.inf, -5.0, last + 1, 0.1, -1.0, 3.0, -2*np.pi],
        [ np.inf, -1e-6, last + 120, 1.0, 2.0, 20.0,  2*np.pi]
    )

    try:
        popt, _ = curve_fit(lppl_func, t, y, p0=p0, bounds=bounds, maxfev=8000)
        yhat = lppl_func(t, *popt)
        sse = float(np.sum((y - yhat) ** 2))
        return popt, sse
    except Exception:
        return None, None


def check_lppl_risk(price_series, n_restarts=5, min_days=5, max_days=90):
    """
    检测LPPL风险信号

    优化:
    - 多次重启避免局部最优
    - 放宽参数限制
    - 扩大crash预测窗口
    """
    best = None

    # 多次重启，降低局部最优
    for k in range(n_restarts):
        try:
            popt, sse = fit_lppl_once(price_series)
            if popt is None:
                continue
            if (best is None) or (sse < best[1]):
                best = (popt, sse)
        except Exception:
            continue

    if best is None:
        return False, 0, None

    popt = best[0]
    A, B, tc, m, C, omega, phi = popt
    last_day = len(price_series) - 1
    days_to_crash = tc - last_day

    # 放宽条件
    cond = (
        (min_days < days_to_crash < max_days) and
        (0.1 < m < 1.0) and      # 放宽 m 的限制 (原来是0.95)
        (3.0 < omega < 20.0) and  # 放宽 omega 的限制 (原来是4-15)
        (-1.5 < B < 0) and       # 放宽 B 的范围
        (abs(C) < 2.0)           # 放宽 C 的限制
    )

    return bool(cond), float(days_to_crash), popt


def process_window(i, subset_df, window_size, last_signal_info):
    """处理单个滚动窗口"""
    window = subset_df.iloc[i:i + window_size]
    current_date = pd.to_datetime(window.index[-1])

    is_risk, days_left, popt = check_lppl_risk(window['收盘价'])
    if is_risk:
        last_signal_date = last_signal_info[0]
        # 去重: 同一信号不重复记录（20天内）
        if (last_signal_date is None) or ((current_date - last_signal_date).days >= 20):
            days_left_float = float(days_left)
            last_signal_info[0] = current_date
            return {
                "date": current_date,
                "days_to_tc": days_left,
                "tc_est": current_date + pd.Timedelta(days=days_left_float),
                "m": float(popt[3]),
                "omega": float(popt[5]),
                "B": float(popt[1]),
                "C": float(popt[4]),
            }
    return None


def main():
    print("=" * 80)
    print("MA60+LPPL双模系统 - 优化版")
    print("=" * 80)

    # 加载数据
    df = pd.read_csv('D:/期货数据/铜期货监控/daily_backtest/SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    df = df.set_index('datetime')

    # 计算MA60趋势
    df['MA60'] = df['收盘价'].rolling(window=60).mean()

    # 扫描最近3年数据
    scan_start_date = df.index[-750]
    subset_df = df[df.index >= scan_start_date].copy()

    print(f"\n数据范围: {subset_df.index[0].date()} ~ {subset_df.index[-1].date()}")
    print(f"数据量: {len(subset_df)} 天")

    # 优化参数设置
    window_size = 250  # 增加窗口大小
    step = 5           # 步长设为5天

    print(f"\n滚动窗口设置:")
    print(f"  窗口大小: {window_size}天")
    print(f"  步长: {step}天")
    print(f"  预计计算次数: {(len(subset_df) - window_size) // step}")

    # 并行化处理滚动窗口
    print(f"\n开始并行化滚动扫描LPPL风险信号...")
    print(f"使用所有CPU核心加速计算...")

    last_signal_info = [None]  # 使用列表以便在并行函数中修改

    # 使用并行化加速滚动窗口拟合过程
    results = Parallel(n_jobs=-1)(
        delayed(process_window)(i, subset_df, window_size, last_signal_info)
        for i in range(0, len(subset_df) - window_size, step)
    )

    # 过滤None结果
    signals = [r for r in results if r is not None]

    print(f"\n扫描完成!")
    print(f"共检测到 {len(signals)} 个LPPL风险信号")

    if not signals:
        print("\n未检测到任何LPPL风险信号")
        return

    # 转换为DataFrame
    signals_df = pd.DataFrame(signals)

    # 统计分析
    print("\n" + "=" * 80)
    print("LPPL风险信号统计")
    print("=" * 80)

    # 分析信号与趋势的关系
    for sig in signals:
        sig_date = sig['date']
        if sig_date in subset_df.index:
            sig['price'] = float(subset_df.loc[sig_date, '收盘价'])
            sig['ma60'] = float(subset_df.loc[sig_date, 'MA60'])
            sig['trend_up'] = sig['price'] > sig['ma60']

    trend_up_signals = [s for s in signals if s.get('trend_up', False)]
    trend_down_signals = [s for s in signals if not s.get('trend_up', False)]

    print(f"\n在上升趋势中(价格>MA60)的风险信号: {len(trend_up_signals)} 个")
    print(f"在下降趋势中(价格<=MA60)的风险信号: {len(trend_down_signals)} 个")

    # 显示参数统计
    print(f"\nLPPL参数统计:")
    print(f"  m值范围: {signals_df['m'].min():.3f} ~ {signals_df['m'].max():.3f}")
    print(f"  m平均值: {signals_df['m'].mean():.3f}")
    print(f"  omega范围: {signals_df['omega'].min():.3f} ~ {signals_df['omega'].max():.3f}")
    print(f"  omega平均值: {signals_df['omega'].mean():.3f}")
    print(f"  预测崩盘天数: {signals_df['days_to_tc'].min():.0f} ~ {signals_df['days_to_tc'].max():.0f}天")

    # 显示最近20个信号
    print("\n" + "=" * 80)
    print("最近20个LPPL风险信号")
    print("=" * 80)

    display_df = signals_df[['date', 'days_to_tc', 'm', 'omega']].tail(20).copy()
    display_df['date'] = display_df['date'].dt.date
    display_df.columns = ['日期', '距崩盘天数', 'm值', 'omega值']
    print(display_df.to_string(index=False))

    # 绘制图表
    print("\n生成分析图表...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])

    # 主图: 价格 + MA60 + LPPL信号
    ax1.plot(subset_df.index, subset_df['收盘价'], label='沪锡价格', color='black', alpha=0.6, linewidth=1)
    ax1.plot(subset_df.index, subset_df['MA60'], label='MA60趋势线', color='blue', linewidth=1.5)

    # 绿色背景 = 上升趋势
    trend_up = subset_df['收盘价'] > subset_df['MA60']
    ax1.fill_between(subset_df.index, subset_df['收盘价'].min(), subset_df['收盘价'].max(),
                     where=trend_up, color='green', alpha=0.1, label='多头安全区')

    # 红色竖线 = LPPL风险信号
    for sig in signals:
        ax1.axvline(x=sig['date'], color='red', alpha=0.3, linewidth=2)

    if signals:
        ax1.axvline(x=signals[0]['date'], color='red', alpha=0.3, linewidth=2, label='LPPL泡沫预警')

    ax1.set_title('沪锡：MA60趋势跟踪 + LPPL泡沫风险预警系统（优化版）', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 副图: m值分布
    if signals:
        m_values = [s['m'] for s in signals]
        ax2.hist(m_values, bins=20, color='orange', alpha=0.6, edgecolor='black')
        ax2.axvline(x=np.mean(m_values), color='red', linestyle='--', linewidth=2, label=f'm平均值: {np.mean(m_values):.3f}')
        ax2.set_xlabel('m值', fontsize=12)
        ax2.set_ylabel('频次', fontsize=12)
        ax2.set_title('LPPL m值分布', fontsize=12)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/ma60_lppl_optimized.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()

    # 保存信号数据
    signals_output = 'D:/期货数据/铜期货监控/daily_backtest/lppl_optimized_signals.csv'
    signals_df.to_csv(signals_output, index=False, encoding='utf-8-sig')
    print(f"信号数据已保存: {signals_output}")

    # 当前状态判断
    print("\n" + "=" * 80)
    print("当前市场状态")
    print("=" * 80)

    latest_price = df['收盘价'].iloc[-1]
    latest_ma60 = df['MA60'].iloc[-1]
    latest_date = df.index[-1]

    print(f"\n日期: {latest_date.date()}")
    print(f"价格: {latest_price:.2f}")
    print(f"MA60: {latest_ma60:.2f}")

    if latest_price > latest_ma60:
        print(f"趋势: [UP] 上升趋势（价格>MA60）")
    else:
        print(f"趋势: [DOWN] 下降趋势（价格<=MA60）")

    # 检查最近30天是否有LPPL预警
    recent_risks = [s for s in signals if (latest_date - s['date']).days <= 30]
    if recent_risks:
        print(f"LPPL风险: [WARNING] 最近30天检测到 {len(recent_risks)} 次泡沫预警")
        print(f"\n最近预警详情:")
        for s in recent_risks[-5:]:
            print(f"  {s['date'].date()} - 预计崩盘: {s['days_to_tc']:.0f}天后, m={s['m']:.3f}")
    else:
        print(f"LPPL风险: [SAFE] 最近30天无泡沫预警")

    print("\n" + "=" * 80)
    print("优化完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
