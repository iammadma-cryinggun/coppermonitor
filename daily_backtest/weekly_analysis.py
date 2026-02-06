# -*- coding: utf-8 -*-
"""
周线分析：GHE+LPPL在大周期上的判断效果
从日线数据转换成周线，分析10年周线趋势
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")


# ============== 1. 日线转周线 ==============
def convert_daily_to_weekly(df_daily):
    """
    将日线数据转换为周线数据

    参数:
        df_daily: 日线DataFrame，需包含'datetime'和'收盘价'列

    返回:
        DataFrame: 周线数据
    """
    df = df_daily.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # 按周重采样，取每周的OHLC
    weekly = df.resample('W').agg({
        '收盘价': 'ohlc'  # 生成open, high, low, close四列
    })

    # 扁平化列名
    weekly.columns = ['open', 'high', 'low', 'close']

    # 去除有空值的周（数据缺失的周）
    weekly = weekly.dropna()

    # 重置索引
    weekly = weekly.reset_index()
    weekly.rename(columns={'datetime': 'date'}, inplace=True)

    return weekly


# ============== 2. GHE计算 ==============
def calculate_ghe(series, q=2, max_tau=20):
    """计算GHE"""
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


def calculate_rolling_ghe(df, window_size=50, step=1):
    """
    计算滚动GHE（周线版，默认50周≈1年）

    参数:
        df: 周线DataFrame
        window_size: 窗口大小（周数），默认50周≈1年
        step: 步长，默认1周
    """
    ghe_series = pd.Series(index=df.index, dtype=float)

    for i in range(window_size, len(df), step):
        window = df['close'].iloc[i - window_size:i]
        try:
            h_q = calculate_ghe(window, q=2, max_tau=20)
            ghe_series.iloc[i] = h_q
        except:
            continue

    return ghe_series


# ============== 3. LPPL计算 ==============
def lppl_func(t, A, B, tc, m, C, omega, phi):
    """LPPL核心函数"""
    dt = np.clip(tc - t, 1e-8, None)
    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))


def calculate_lppl_d(prices):
    """计算LPPL-D值"""
    if len(prices) < 100:  # 周线至少需要100周（约2年）
        return None

    log_prices = np.log(prices.values if hasattr(prices, 'values') else prices)
    t = np.arange(len(log_prices))
    last_t = t[-1]

    A0 = log_prices.max()
    B0 = -0.1
    tc0 = last_t + 20  # 预测20周后
    m0 = 0.5
    C0 = 0.1
    omega0 = 10.0
    phi0 = 0.0
    p0 = [A0, B0, tc0, m0, C0, omega0, phi0]

    bounds = (
        [-np.inf, -np.inf, last_t, 0.1, -1, 6, -2*np.pi],
        [np.inf, 0, last_t + 50, 0.9, 1, 13, 2*np.pi]
    )

    try:
        popt, pcov = curve_fit(lppl_func, t, log_prices, p0=p0,
                               bounds=bounds, maxfev=5000, method='trf')

        A, B, tc, m, C, omega, phi = popt

        if not (0.1 < m < 0.9):
            return None
        if not (6 < omega < 13):
            return None
        if B >= 0:
            return None

        D = m * omega / (2 * np.pi)
        return D

    except:
        return None


def calculate_rolling_lppl(df, window_size=100, step=1):
    """
    计算滚动LPPL-D（周线版，默认100周≈2年）

    参数:
        df: 周线DataFrame
        window_size: 窗口大小（周数），默认100周≈2年
        step: 步长，默认1周
    """
    lppl_series = pd.Series(index=df.index, dtype=float)

    for i in range(window_size, len(df), step):
        try:
            d = calculate_lppl_d(df['close'].iloc[i - window_size:i])
            if d is not None:
                lppl_series.iloc[i] = d
        except:
            continue

    return lppl_series


# ============== 4. 趋势判断 ==============
def get_trend_regime(df, idx, forward_weeks=10):
    """
    判断未来forward_weeks周后的市场状态

    返回: 'STRONG_UP', 'STRONG_DOWN', 'RANGING'
    """
    future_idx = min(idx + forward_weeks, len(df) - 1)

    if future_idx < 30:  # 至少需要30周数据
        return 'UNKNOWN'

    price = df.loc[future_idx, 'close']
    ma30 = df['close'].iloc[max(0, future_idx-30):future_idx+1].mean()

    # 计算EMA
    ema_fast = df['close'].iloc[max(0, future_idx-2):future_idx+1].ewm(span=3, adjust=False).mean().iloc[-1]
    ema_slow = df['close'].iloc[max(0, future_idx-15):future_idx+1].ewm(span=15, adjust=False).mean().iloc[-1]

    if price > ma30:
        if ema_fast > ema_slow:
            return 'STRONG_UP'
        else:
            return 'RANGING'
    else:
        if ema_fast < ema_slow:
            return 'STRONG_DOWN'
        else:
            return 'RANGING'


# ============== 5. 主分析 ==============
def main():
    print("="*80)
    print("周线分析：GHE+LPPL在大周期上的判断效果")
    print("="*80)

    # 加载日线数据
    print("\n加载日线数据...")
    df_daily = pd.read_csv('SN_沪锡_日线_10年.csv')
    df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
    df_daily = df_daily.sort_values('datetime').reset_index(drop=True)

    print(f"日线数据范围: {df_daily['datetime'].iloc[0]} ~ {df_daily['datetime'].iloc[-1]}")
    print(f"日线数据量: {len(df_daily)} 天")

    # 转换为周线
    print("\n转换为周线数据...")
    df_weekly = convert_daily_to_weekly(df_daily)

    print(f"周线数据范围: {df_weekly['date'].iloc[0]} ~ {df_weekly['date'].iloc[-1]}")
    print(f"周线数据量: {len(df_weekly)} 周（约{len(df_weekly)/52:.1f}年）")

    # 计算GHE和LPPL-D
    print("\n计算GHE（50周窗口≈1年）...")
    df_weekly['GHE'] = calculate_rolling_ghe(df_weekly, window_size=50, step=1)

    print("计算LPPL-D（100周窗口≈2年）...")
    df_weekly['LPPL_D'] = calculate_rolling_lppl(df_weekly, window_size=100, step=1)

    # 前向填充
    df_weekly['GHE'] = df_weekly['GHE'].ffill()
    df_weekly['LPPL_D'] = df_weekly['LPPL_D'].ffill()

    # 计算未来趋势
    print("计算未来趋势...")
    df_weekly['future_5w'] = df_weekly.apply(lambda row: get_trend_regime(df_weekly, row.name, 5), axis=1)
    df_weekly['future_10w'] = df_weekly.apply(lambda row: get_trend_regime(df_weekly, row.name, 10), axis=1)
    df_weekly['future_20w'] = df_weekly.apply(lambda row: get_trend_regime(df_weekly, row.name, 20), axis=1)

    # 只保留有效数据
    analysis_df = df_weekly.dropna(subset=['GHE', 'LPPL_D']).copy()
    analysis_df = analysis_df[analysis_df.index >= 100]

    print(f"\n有效分析数据: {len(analysis_df)} 周（约{len(analysis_df)/52:.1f}年）")

    # ============== 核心分析 ==============
    print("\n" + "="*80)
    print("【基准分析】周线趋势分布")
    print("="*80)

    for weeks in [5, 10, 20]:
        col = f'future_{weeks}w'
        print(f"\n未来{weeks}周（约{weeks/4:.1f}月）趋势分布:")

        counts = analysis_df[col].value_counts()
        total = len(analysis_df)

        for trend in ['STRONG_UP', 'STRONG_DOWN', 'RANGING']:
            if trend in counts.index:
                print(f"  {trend:<15}: {counts[trend]:>3}周 ({counts[trend]/total*100:>5.1f}%)")

    # 基准概率
    baseline_up = (analysis_df['future_10w'] == 'STRONG_UP').sum() / len(analysis_df) * 100
    baseline_down = (analysis_df['future_10w'] == 'STRONG_DOWN').sum() / len(analysis_df) * 100
    baseline_ranging = (analysis_df['future_10w'] == 'RANGING').sum() / len(analysis_df) * 100

    print(f"\n基准概率（随机猜测）:")
    print(f"  强上升: {baseline_up:.1f}%")
    print(f"  强下降: {baseline_down:.1f}%")
    print(f"  横盘: {baseline_ranging:.1f}%")

    # ============== GHE+LPPL组合分析 ==============
    print("\n" + "="*80)
    print("【GHE+LPPL组合分析】周线预测效果")
    print("="*80)

    # 使用与日线相同的区间划分
    ghe_ranges = [
        (0.30, 0.40),
        (0.35, 0.45),
        (0.40, 0.50),
        (0.45, 0.55),
        (0.50, 0.60)
    ]

    lppl_ranges = [
        (0.4, 0.6),
        (0.5, 0.7),
        (0.6, 0.8),
        (0.7, 0.9),
        (0.8, 1.0)
    ]

    results = []

    for ghe_low, ghe_high in ghe_ranges:
        for lppl_low, lppl_high in lppl_ranges:
            mask = (
                (analysis_df['GHE'] >= ghe_low) &
                (analysis_df['GHE'] < ghe_high) &
                (analysis_df['LPPL_D'] >= lppl_low) &
                (analysis_df['LPPL_D'] < lppl_high)
            )

            group = analysis_df[mask]

            if len(group) >= 5:  # 周线样本较少，阈值降到5
                up_10w = (group['future_10w'] == 'STRONG_UP').sum() / len(group) * 100
                down_10w = (group['future_10w'] == 'STRONG_DOWN').sum() / len(group) * 100
                ranging_10w = (group['future_10w'] == 'RANGING').sum() / len(group) * 100

                results.append({
                    'ghe_range': f'{ghe_low:.2f}-{ghe_high:.2f}',
                    'lppl_range': f'{lppl_low:.1f}-{lppl_high:.1f}',
                    'count': len(group),
                    'up_prob': up_10w,
                    'down_prob': down_10w,
                    'ranging_prob': ranging_10w,
                    'dominant': max(up_10w, down_10w, ranging_10w)
                })

    # 排序
    results_up = sorted(results, key=lambda x: x['up_prob'], reverse=True)
    results_down = sorted(results, key=lambda x: x['down_prob'], reverse=True)
    results_ranging = sorted(results, key=lambda x: x['ranging_prob'], reverse=True)

    print(f"\n--- 最佳强上升预测组合（未来10周≈2.5月）---")
    print(f"{'GHE':<12} {'LPPL-D':<12} {'样本数':<8} {'上升概率':<10} {'下降概率':<10} {'横盘概率':<10} {'提升':<10}")
    print("-*" * 55)

    for i, r in enumerate(results_up[:5]):
        uplift = r['up_prob'] - baseline_up
        print(f"{r['ghe_range']:<12} {r['lppl_range']:<12} {r['count']:<8} "
              f"{r['up_prob']:>9.1f}% {r['down_prob']:>9.1f}% {r['ranging_prob']:>9.1f}% "
              f"{uplift:>+9.1f}%")

    print(f"\n--- 最佳强下降预测组合（未来10周≈2.5月）---")
    print(f"{'GHE':<12} {'LPPL-D':<12} {'样本数':<8} {'上升概率':<10} {'下降概率':<10} {'横盘概率':<10} {'提升':<10}")
    print("-*" * 55)

    for i, r in enumerate(results_down[:5]):
        uplift = r['down_prob'] - baseline_down
        print(f"{r['ghe_range']:<12} {r['lppl_range']:<12} {r['count']:<8} "
              f"{r['up_prob']:>9.1f}% {r['down_prob']:>9.1f}% {r['ranging_prob']:>9.1f}% "
              f"{uplift:>+9.1f}%")

    # ============== 当前周线状态 ==============
    print("\n" + "="*80)
    print("【当前周线状态】")
    print("="*80)

    latest = analysis_df.iloc[-1]
    print(f"\n最新周数据（{latest['date'].date()}）:")
    print(f"  收盘价: {latest['close']:.2f}")
    print(f"  GHE: {latest['GHE']:.3f}")
    print(f"  LPPL-D: {latest['LPPL_D']:.3f}")

    # 判断当前属于哪种组合
    current_ghe = latest['GHE']
    current_lppl = latest['LPPL_D']

    print(f"\n当前组合分析:")

    if 0.35 <= current_ghe <= 0.45 and 0.4 <= current_lppl <= 0.6:
        print(f"  命中: 强下降组合（日线标准）")
        print(f"  建议: 观望或做空")
    elif 0.35 <= current_ghe <= 0.45 and 0.7 <= current_lppl <= 0.9:
        print(f"  命中: 横盘组合（日线标准）")
        print(f"  建议: 区间交易或观望")
    elif 0.50 <= current_ghe <= 0.60 and 0.7 <= current_lppl <= 0.9:
        print(f"  命中: 强上升组合（日线标准）")
        print(f"  建议: 积极做多")
    else:
        print(f"  未命中日线预测区间")
        print(f"  建议: 使用战术指标决策")

    # ============== 可视化 ==============
    print("\n生成可视化图表...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # 图1：周线价格 + GHE
    ax1 = axes[0]
    ax1.plot(analysis_df['date'], analysis_df['close'], color='black', linewidth=2, label='周线收盘价')
    ax1.set_ylabel('价格', fontsize=12)
    ax1.set_title('10年沪锡周线走势', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 双Y轴显示GHE
    ax1b = ax1.twinx()
    ax1b.plot(analysis_df['date'], analysis_df['GHE'], color='purple', linewidth=1, alpha=0.6, label='GHE')
    ax1b.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1b.set_ylabel('GHE值', fontsize=12)
    ax1b.legend(loc='upper right')

    # 图2：周线价格 + LPPL-D
    ax2 = axes[1]
    ax2.plot(analysis_df['date'], analysis_df['close'], color='black', linewidth=2, label='周线收盘价')
    ax2.set_ylabel('价格', fontsize=12)
    ax2.set_title('LPPL-D在周线上的表现', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 双Y轴显示LPPL-D
    ax2b = ax2.twinx()
    ax2b.plot(analysis_df['date'], analysis_df['LPPL_D'], color='blue', linewidth=1, alpha=0.6, label='LPPL-D')
    ax2b.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='阈值0.5')
    ax2b.set_ylabel('LPPL-D值', fontsize=12)
    ax2b.legend(loc='upper right')

    # 图3：GHE vs LPPL-D散点图（颜色表示未来趋势）
    ax3 = axes[2]

    trends_map = {'STRONG_UP': 'red', 'STRONG_DOWN': 'green', 'RANGING': 'gray', 'UNKNOWN': 'lightgray'}

    for trend in ['STRONG_UP', 'STRONG_DOWN', 'RANGING']:
        trend_data = analysis_df[analysis_df['future_10w'] == trend]
        if len(trend_data) > 0:
            ax3.scatter(trend_data['GHE'], trend_data['LPPL_D'],
                       c=trends_map[trend], label=f'{trend} ({len(trend_data)}周)',
                       alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

    # 标记当前点
    ax3.scatter([current_ghe], [current_lppl], c='yellow', s=200,
               marker='*', edgecolors='black', linewidth=2, label='当前', zorder=10)

    ax3.set_xlabel('GHE', fontsize=12)
    ax3.set_ylabel('LPPL-D', fontsize=12)
    ax3.set_title('GHE+LPPL组合与未来10周趋势关系', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/weekly_ghe_lppl_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()

    # ============== 最终总结 ==============
    print("\n" + "="*80)
    print("【周线vs日线对比】")
    print("="*80)

    print(f"""
数据周期对比:
  日线: {len(df_daily)}天 ≈ {len(df_daily)/252:.1f}年
  周线: {len(df_weekly)}周 ≈ {len(df_weekly)/52:.1f}年
  有效分析: {len(analysis_df)}周 ≈ {len(analysis_df)/52:.1f}年

基准概率对比（未来10周 vs 未来20日）:
  日线: 强上升{baseline_up:.1f}%, 强下降{baseline_down:.1f}%, 横盘{baseline_ranging:.1f}%
  周线: 强上升{baseline_up:.1f}%, 强下降{baseline_down:.1f}%, 横盘{baseline_ranging:.1f}%
       （10周≈2.5月，20日≈1月）

周线优势:
  1. 过滤日线噪音，更能反映大周期趋势
  2. GHE和LPPL计算更稳定（波动更小）
  3. 预测周期更长（10周≈2.5月），适合战略判断

周线劣势:
  1. 样本数较少（{len(analysis_df)}周 vs 日线2229天）
  2. 反应较慢，不适合短线交易
  3. 部分日线预测组合在周线上可能不适用

建议:
  - 大周期判断: 使用周线GHE+LPPL（战略层面）
  - 入场时机: 使用日线ADX≥25+EMA交叉（战术层面）
  - 两者结合: 周线看方向，日线找时机
    """)

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
