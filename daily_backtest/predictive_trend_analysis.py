# -*- coding: utf-8 -*-
"""
预测性分析：用GHE和LPPL预判未来趋势
核心问题：当前的GHE/LPPL处于什么状态时，未来容易进入STRONG_UP？
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

from scipy.optimize import curve_fit

# ============== 1. GHE计算 ==============
def calculate_ghe_single_window(series, q=2, max_tau=20):
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

def calculate_rolling_ghe(df, window_size=100, step=1):
    """计算滚动GHE，每个时点都有值"""
    ghe_series = pd.Series(index=df.index, dtype=float)
    for i in range(window_size, len(df)):
        window = df['收盘价'].iloc[i - window_size:i]
        try:
            h_q = calculate_ghe_single_window(window)
            ghe_series.iloc[i] = h_q
        except:
            continue
    return ghe_series

# ============== 2. LPPL计算 ==============
def lppl_func(t, A, B, tc, m, C, omega, phi):
    dt = np.clip(tc - t, 1e-8, None)
    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))

def fit_lppl_window(prices, start_idx, end_idx):
    log_prices = np.log(prices[start_idx:end_idx].values)
    t = np.arange(len(log_prices))
    last_t = t[-1]
    A0 = log_prices.max()
    B0 = -0.1
    tc0 = last_t + 30
    m0 = 0.5
    C0 = 0.1
    omega0 = 10.0
    phi0 = 0.0
    p0 = [A0, B0, tc0, m0, C0, omega0, phi0]
    bounds = (
        [-np.inf, -np.inf, last_t, 0.1, -1, 6, -2*np.pi],
        [np.inf, 0, last_t + 90, 0.9, 1, 13, 2*np.pi]
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
        return {'D': D, 'm': m, 'omega': omega, 'tc_relative': tc}
    except:
        return None

def calculate_rolling_lppl(df, window_size=200, step=1):
    """计算滚动LPPL-D"""
    lppl_series = pd.Series(index=df.index, dtype=float)
    for i in range(window_size, len(df)):
        try:
            result = fit_lppl_window(df['收盘价'], i - window_size, i)
            if result is not None:
                lppl_series.iloc[i] = result['D']
        except:
            continue
    return lppl_series

# ============== 3. 判断市场状态（未来） ==============
def get_future_regime(df, idx, forward_days=20):
    """判断未来forward_days天后的市场状态"""
    future_idx = min(idx + forward_days, len(df) - 1)

    if future_idx < 60:
        return 'UNKNOWN'

    price = df.loc[future_idx, '收盘价']
    ma60 = df['收盘价'].iloc[future_idx-60:future_idx+1].mean()

    # 计算EMA
    ema_fast = df['收盘价'].iloc[future_idx-2:future_idx+1].ewm(span=3, adjust=False).mean().iloc[-1]
    ema_slow = df['收盘价'].iloc[future_idx-15:future_idx+1].ewm(span=15, adjust=False).mean().iloc[-1]

    if price > ma60:
        if ema_fast > ema_slow:
            return 'STRONG_UP'
        else:
            return 'WEAK_UP'
    else:
        if ema_fast < ema_slow:
            return 'STRONG_DOWN'
        else:
            return 'WEAK_DOWN'

# ============== 4. 主分析 ==============
def main():
    print("="*80)
    print("预测性分析：GHE和LPPL能否预判STRONG_UP？")
    print("="*80)

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 天")

    # 计算GHE和LPPL
    print("\n计算GHE和LPPL...")
    df['GHE'] = calculate_rolling_ghe(df, window_size=100, step=1)
    df['LPPL_D'] = calculate_rolling_lppl(df, window_size=200, step=1)

    # 前向填充
    df['GHE'] = df['GHE'].ffill()
    df['LPPL_D'] = df['LPPL_D'].ffill()

    # 判断不同未来时点的市场状态
    print("计算未来市场状态...")
    df['future_5d'] = df.apply(lambda row: get_future_regime(df, row.name, 5), axis=1)
    df['future_10d'] = df.apply(lambda row: get_future_regime(df, row.name, 10), axis=1)
    df['future_20d'] = df.apply(lambda row: get_future_regime(df, row.name, 20), axis=1)

    # 只保留有效数据
    analysis_df = df.dropna(subset=['GHE', 'LPPL_D']).copy()
    analysis_df = analysis_df[analysis_df.index >= 200]  # 确保有足够历史

    print(f"\n有效分析数据量: {len(analysis_df)} 天")

    # ============== 核心分析：GHE/LPPL能否预测未来STRONG_UP？ ==============
    print("\n" + "="*80)
    print("核心问题：当前的GHE/LPPL在什么范围时，未来会进入STRONG_UP？")
    print("="*80)

    # 分析1：按GHE分组，看未来进入STRONG_UP的概率
    print("\n--- 分析1: GHE与未来STRONG_UP的关系 ---")

    for days in [5, 10, 20]:
        col = f'future_{days}d'
        print(f"\n未来{days}天进入STRONG_UP的概率:")

        # 按GHE分组
        analysis_df[f'GHE_group'] = pd.cut(analysis_df['GHE'],
                                            bins=[0, 0.3, 0.4, 0.5, 1.0],
                                            labels=['<0.3', '0.3-0.4', '0.4-0.5', '≥0.5'])

        ghe_pred = analysis_df.groupby('GHE_group').apply(
            lambda x: (x[col] == 'STRONG_UP').sum() / len(x) * 100
        )

        print(f"{'GHE范围':<15} {'样本数':<10} {'未来STRONG_UP概率':<20}")
        print("-"*50)

        for ghe_range in ['<0.3', '0.3-0.4', '0.4-0.5', '≥0.5']:
            if ghe_range in ghe_pred.index:
                group = analysis_df[analysis_df['GHE_group'] == ghe_range]
                prob = ghe_pred[ghe_range]
                print(f"{ghe_range:<15} {len(group):<10} {prob:<20.1f}%")

    # 分析2：按LPPL_D分组，看未来进入STRONG_UP的概率
    print("\n--- 分析2: LPPL-D与未来STRONG_UP的关系 ---")

    for days in [5, 10, 20]:
        col = f'future_{days}d'
        print(f"\n未来{days}天进入STRONG_UP的概率:")

        # 按LPPL_D分组
        analysis_df['LPPL_group'] = pd.cut(analysis_df['LPPL_D'],
                                            bins=[0, 0.3, 0.5, 0.8, 2.0],
                                            labels=['<0.3', '0.3-0.5', '0.5-0.8', '≥0.8'])

        lppl_pred = analysis_df.groupby('LPPL_group').apply(
            lambda x: (x[col] == 'STRONG_UP').sum() / len(x) * 100
        )

        print(f"{'LPPL-D范围':<15} {'样本数':<10} {'未来STRONG_UP概率':<20}")
        print("-"*55)

        for lppl_range in ['<0.3', '0.3-0.5', '0.5-0.8', '≥0.8']:
            if lppl_range in lppl_pred.index:
                group = analysis_df[analysis_df['LPPL_group'] == lppl_range]
                prob = lppl_pred[lppl_range]
                print(f"{lppl_range:<15} {len(group):<10} {prob:<20.1f}%")

    # 分析3：GHE + LPPL-D 组合预测
    print("\n--- 分析3: GHE + LPPL-D 组合预测 ---")

    for days in [10, 20]:
        col = f'future_{days}d'
        print(f"\n未来{days}天进入STRONG_UP的概率（组合信号）:")

        print(f"{'GHE':<10} {'LPPL-D':<12} {'样本数':<10} {'STRONG_UP概率':<15}")
        print("-"*60)

        # 遍历关键组合
        combinations = [
            ('<0.4', '<0.5'),
            ('<0.4', '0.5-0.8'),
            ('0.4-0.5', '<0.5'),
            ('0.4-0.5', '0.5-0.8'),
            ('≥0.5', '<0.5'),
            ('≥0.5', '0.5-0.8'),
        ]

        for ghe_range, lppl_range in combinations:
            mask = (
                ((analysis_df['GHE'] < 0.4) if ghe_range == '<0.4' else
                 (analysis_df['GHE'] >= 0.4) & (analysis_df['GHE'] < 0.5) if ghe_range == '0.4-0.5' else
                 (analysis_df['GHE'] >= 0.5)) &
                ((analysis_df['LPPL_D'] < 0.5) if lppl_range == '<0.5' else
                 (analysis_df['LPPL_D'] >= 0.5) & (analysis_df['LPPL_D'] < 0.8))
            )

            group = analysis_df[mask]
            if len(group) > 0:
                prob = (group[col] == 'STRONG_UP').sum() / len(group) * 100
                print(f"{ghe_range:<10} {lppl_range:<12} {len(group):<10} {prob:<15.1f}%")

    # 分析4：找到最佳预测区间
    print("\n" + "="*80)
    print("核心发现：什么GHE/LPPL组合最容易预示未来的STRONG_UP？")
    print("="*80)

    # 细粒度分析
    ghe_bins = np.arange(0.2, 0.7, 0.05)
    lppl_bins = np.arange(0.3, 1.2, 0.1)

    best_combinations = []

    for ghe_low, ghe_high in zip(ghe_bins[:-1], ghe_bins[1:]):
        for lppl_low, lppl_high in zip(lppl_bins[:-1], lppl_bins[1:]):
            mask = (
                (analysis_df['GHE'] >= ghe_low) &
                (analysis_df['GHE'] < ghe_high) &
                (analysis_df['LPPL_D'] >= lppl_low) &
                (analysis_df['LPPL_D'] < lppl_high)
            )

            group = analysis_df[mask]

            if len(group) >= 10:  # 至少10个样本
                prob_10d = (group['future_10d'] == 'STRONG_UP').sum() / len(group) * 100
                prob_20d = (group['future_20d'] == 'STRONG_UP').sum() / len(group) * 100

                best_combinations.append({
                    'ghe_range': f'{ghe_low:.2f}-{ghe_high:.2f}',
                    'lppl_range': f'{lppl_low:.1f}-{lppl_high:.1f}',
                    'count': len(group),
                    'prob_10d': prob_10d,
                    'prob_20d': prob_20d
                })

    # 排序找出最佳组合
    best_combinations.sort(key=lambda x: x['prob_20d'], reverse=True)

    print(f"\n{'GHE范围':<15} {'LPPL-D范围':<15} {'样本数':<10} {'10日概率':<12} {'20日概率':<12}")
    print("-"*80)

    for i, combo in enumerate(best_combinations[:10]):
        print(f"{combo['ghe_range']:<15} {combo['lppl_range']:<15} {combo['count']:<10} "
              f"{combo['prob_10d']:<11.1f}% {combo['prob_20d']:<11.1f}%")

    # 分析5：与基准对比
    print("\n" + "="*80)
    print("与基准对比：预测信号的效果")
    print("="*80)

    baseline_prob_10d = (analysis_df['future_10d'] == 'STRONG_UP').sum() / len(analysis_df) * 100
    baseline_prob_20d = (analysis_df['future_20d'] == 'STRONG_UP').sum() / len(analysis_df) * 100

    print(f"\n基准（随机选择）:")
    print(f"  未来10日进入STRONG_UP概率: {baseline_prob_10d:.1f}%")
    print(f"  未来20日进入STRONG_UP概率: {baseline_prob_20d:.1f}%")

    if len(best_combinations) > 0:
        best = best_combinations[0]
        print(f"\n最佳组合（{best['ghe_range']} GHE, {best['lppl_range']} LPPL-D）:")
        print(f"  未来10日进入STRONG_UP概率: {best['prob_10d']:.1f}% (提升{best['prob_10d']-baseline_prob_10d:+.1f}个百分点)")
        print(f"  未来20日进入STRONG_UP概率: {best['prob_20d']:.1f}% (提升{best['prob_20d']-baseline_prob_20d:+.1f}个百分点)")

    # 可视化
    print("\n生成可视化图表...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1：GHE vs 未来10日STRONG_UP概率
    ax1 = axes[0, 0]
    ghe_groups = ['<0.3', '0.3-0.4', '0.4-0.5', '≥0.5']
    ghe_probs_10d = []
    ghe_probs_20d = []

    for ghe_range in ghe_groups:
        if ghe_range == '<0.3':
            group = analysis_df[analysis_df['GHE'] < 0.3]
        elif ghe_range == '0.3-0.4':
            group = analysis_df[(analysis_df['GHE'] >= 0.3) & (analysis_df['GHE'] < 0.4)]
        elif ghe_range == '0.4-0.5':
            group = analysis_df[(analysis_df['GHE'] >= 0.4) & (analysis_df['GHE'] < 0.5)]
        else:
            group = analysis_df[analysis_df['GHE'] >= 0.5]

        if len(group) > 0:
            ghe_probs_10d.append((group['future_10d'] == 'STRONG_UP').sum() / len(group) * 100)
            ghe_probs_20d.append((group['future_20d'] == 'STRONG_UP').sum() / len(group) * 100)
        else:
            ghe_probs_10d.append(0)
            ghe_probs_20d.append(0)

    x = np.arange(len(ghe_groups))
    width = 0.35

    ax1.bar(x - width/2, ghe_probs_10d, width, label='10日后', alpha=0.8)
    ax1.bar(x + width/2, ghe_probs_20d, width, label='20日后', alpha=0.8)
    ax1.axhline(y=baseline_prob_10d, color='red', linestyle='--', label='基准')
    ax1.set_xlabel('GHE范围', fontsize=12)
    ax1.set_ylabel('未来进入STRONG_UP概率 (%)', fontsize=12)
    ax1.set_title('GHE与未来趋势的关系', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ghe_groups)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 图2：LPPL-D vs 未来10日STRONG_UP概率
    ax2 = axes[0, 1]
    lppl_groups = ['<0.3', '0.3-0.5', '0.5-0.8', '≥0.8']
    lppl_probs_10d = []
    lppl_probs_20d = []

    for lppl_range in lppl_groups:
        if lppl_range == '<0.3':
            group = analysis_df[analysis_df['LPPL_D'] < 0.3]
        elif lppl_range == '0.3-0.5':
            group = analysis_df[(analysis_df['LPPL_D'] >= 0.3) & (analysis_df['LPPL_D'] < 0.5)]
        elif lppl_range == '0.5-0.8':
            group = analysis_df[(analysis_df['LPPL_D'] >= 0.5) & (analysis_df['LPPL_D'] < 0.8)]
        else:
            group = analysis_df[analysis_df['LPPL_D'] >= 0.8]

        if len(group) > 0:
            lppl_probs_10d.append((group['future_10d'] == 'STRONG_UP').sum() / len(group) * 100)
            lppl_probs_20d.append((group['future_20d'] == 'STRONG_UP').sum() / len(group) * 100)
        else:
            lppl_probs_10d.append(0)
            lppl_probs_20d.append(0)

    x = np.arange(len(lppl_groups))
    ax2.bar(x - width/2, lppl_probs_10d, width, label='10日后', alpha=0.8)
    ax2.bar(x + width/2, lppl_probs_20d, width, label='20日后', alpha=0.8)
    ax2.axhline(y=baseline_prob_10d, color='red', linestyle='--', label='基准')
    ax2.set_xlabel('LPPL-D范围', fontsize=12)
    ax2.set_ylabel('未来进入STRONG_UP概率 (%)', fontsize=12)
    ax2.set_title('LPPL-D与未来趋势的关系', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(lppl_groups)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 图3：热力图 - GHE vs LPPL-D
    ax3 = axes[1, 0]

    ghe_range = np.linspace(0.25, 0.65, 9)
    lppl_range = np.linspace(0.35, 1.15, 9)

    heatmap_data = np.zeros((len(ghe_range)-1, len(lppl_range)-1))

    for i in range(len(ghe_range)-1):
        for j in range(len(lppl_range)-1):
            mask = (
                (analysis_df['GHE'] >= ghe_range[i]) &
                (analysis_df['GHE'] < ghe_range[i+1]) &
                (analysis_df['LPPL_D'] >= lppl_range[j]) &
                (analysis_df['LPPL_D'] < lppl_range[j+1])
            )

            group = analysis_df[mask]
            if len(group) >= 5:
                heatmap_data[i, j] = (group['future_20d'] == 'STRONG_UP').sum() / len(group) * 100
            else:
                heatmap_data[i, j] = np.nan

    im = ax3.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=100)
    ax3.set_xticks(np.arange(len(lppl_range)-1))
    ax3.set_yticks(np.arange(len(ghe_range)-1))
    ax3.set_xticklabels([f'{lppl_range[j]:.1f}' for j in range(len(lppl_range)-1)])
    ax3.set_yticklabels([f'{ghe_range[i]:.2f}' for i in range(len(ghe_range)-1)])
    ax3.set_xlabel('LPPL-D', fontsize=12)
    ax3.set_ylabel('GHE', fontsize=12)
    ax3.set_title('GHE+LPPL-D组合预测20日后STRONG_UP概率热力图', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='STRONG_UP概率(%)')

    # 图4：时间序列示例
    ax4 = axes[1, 1]

    # 取最近500天的数据
    recent = analysis_df.tail(500)

    ax4.plot(range(len(recent)), recent['收盘价'].values, color='black', linewidth=1, alpha=0.5, label='价格')

    # 标记未来进入STRONG_UP的点
    strong_up_signals = recent[recent['future_20d'] == 'STRONG_UP']
    ax4.scatter(strong_up_signals.index - recent.index[0],
                strong_up_signals['收盘价'].values,
                c='red', s=20, alpha=0.5, label='未来20日STRONG_UP')

    ax4.set_xlabel('天数（最近500天）', fontsize=12)
    ax4.set_ylabel('价格', fontsize=12)
    ax4.set_title('时间序列示例：哪些时点预示未来STRONG_UP', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/predictive_trend_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()

    # 最终量化结论
    print("\n" + "="*80)
    print("量化结论：GHE和LPPL能否预判STRONG_UP？")
    print("="*80)

    if len(best_combinations) > 0:
        best = best_combinations[0]

        print(f"""
基于10年历史数据的预测性分析：

1. 预测能力评估:
   基准概率（随机）: 未来20日进入STRONG_UP = {baseline_prob_20d:.1f}%
   最佳组合预测: 未来20日进入STRONG_UP = {best['prob_20d']:.1f}%
   提升: {best['prob_20d']-baseline_prob_20d:+.1f}个百分点

2. 最佳预测组合:
   GHE范围: {best['ghe_range']}
   LPPL-D范围: {best['lppl_range']}
   样本数: {best['count']}个时点
   预测准确率: {best['prob_20d']:.1f}%

3. 预测策略:
   当同时满足以下条件时，未来20日有{best['prob_20d']:.0f}%概率进入STRONG_UP:
   - GHE在 {best['ghe_range']} 范围
   - LPPL-D在 {best['lppl_range']} 范围

4. 实用建议:
   - 如果{best['prob_20d']:.0f}% > {baseline_prob_20d:.0f}% + 10%:
     GHE+LPPL组合有预测价值，可以作为入场参考

   - 如果{best['prob_20d']:.0f}% < {baseline_prob_20d:.0f}% + 5%:
     GHE+LPPL组合预测能力有限，不建议使用

   - 推荐使用更简单的信号: ADX≥25 + EMA交叉（已验证有效）
        """)

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
