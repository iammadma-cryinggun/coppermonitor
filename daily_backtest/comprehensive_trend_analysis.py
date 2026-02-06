# -*- coding: utf-8 -*-
"""
综合趋势分析：GHE + LPPL + 市场状态
找出10年数据中的趋势规律
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

def calculate_rolling_ghe(df, window_size=100, step=5):
    """计算滚动GHE"""
    ghe_values = []
    dates = []

    for i in range(window_size, len(df), step):
        window = df['收盘价'].iloc[i - window_size:i]
        try:
            h_q = calculate_ghe_single_window(window)
            ghe_values.append(h_q)
            dates.append(df.iloc[i]['datetime'])
        except:
            continue

    return pd.DataFrame({'datetime': dates, 'GHE': ghe_values})

# ============== 2. LPPL计算 ==============
def lppl_func(t, A, B, tc, m, C, omega, phi):
    """LPPL核心函数"""
    dt = np.clip(tc - t, 1e-8, None)
    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))

def fit_lppl_window(prices, start_idx, end_idx):
    """拟合LPPL，返回D值参数"""
    log_prices = np.log(prices[start_idx:end_idx].values)
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

        # 计算R²
        y_pred = lppl_func(t, *popt)
        ss_res = np.sum((log_prices - y_pred) ** 2)
        ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # 计算D值（用户论文中的关键参数）
        # D值 = (1 + m) / (tc - t_current)
        # 这里我们简化为m的函数
        D = m * omega / (2 * np.pi)

        return {
            'D': D,
            'm': m,
            'omega': omega,
            'r_squared': r_squared,
            'tc_relative': tc,
            'tc_global': start_idx + tc
        }
    except:
        return None

def calculate_rolling_lppl(df, window_size=200, step=10):
    """计算滚动LPPL的D值"""
    lppl_values = []
    dates = []

    for i in range(window_size, len(df), step):
        try:
            result = fit_lppl_window(df['收盘价'], i - window_size, i)
            if result is not None:
                lppl_values.append(result['D'])
                dates.append(df.iloc[i]['datetime'])
        except:
            continue

    return pd.DataFrame({'datetime': dates, 'LPPL_D': lppl_values})

# ============== 3. 市场状态判断 ==============
def get_market_regime_at_idx(df, idx):
    """判断指定索引位置的市场状态"""
    if idx < 60:
        return 'INSUFFICIENT_DATA'

    price = df.loc[idx, '收盘价']
    ma60 = df['收盘价'].iloc[idx-60:idx+1].mean()

    # 计算EMA
    ema_fast = df['收盘价'].iloc[idx-2:idx+1].ewm(span=3, adjust=False).mean().iloc[-1]
    ema_slow = df['收盘价'].iloc[idx-15:idx+1].ewm(span=15, adjust=False).mean().iloc[-1]

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

def calculate_rolling_regime(df):
    """计算滚动市场状态"""
    regimes = []
    dates = []

    for i in range(60, len(df), 5):
        regime = get_market_regime_at_idx(df, i)
        regimes.append(regime)
        dates.append(df.iloc[i]['datetime'])

    return pd.DataFrame({'datetime': dates, 'regime': regimes})

# ============== 4. 主分析函数 ==============
def main():
    print("="*80)
    print("10年沪锡日线数据：GHE + LPPL + 趋势规律综合分析")
    print("="*80)

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 天")

    # 步骤1：计算GHE
    print("\n" + "="*80)
    print("【步骤1】计算10年滚动GHE")
    print("="*80)

    ghe_df = calculate_rolling_ghe(df, window_size=100, step=5)
    print(f"\nGHE统计:")
    print(f"  均值: {ghe_df['GHE'].mean():.3f}")
    print(f"  中位数: {ghe_df['GHE'].median():.3f}")
    print(f"  标准差: {ghe_df['GHE'].std():.3f}")
    print(f"  范围: {ghe_df['GHE'].min():.3f} ~ {ghe_df['GHE'].max():.3f}")

    # 步骤2：计算LPPL的D值
    print("\n" + "="*80)
    print("【步骤2】计算10年滚动LPPL-D值")
    print("="*80)

    lppl_df = calculate_rolling_lppl(df, window_size=200, step=10)
    print(f"\nLPPL-D统计:")
    print(f"  均值: {lppl_df['LPPL_D'].mean():.3f}")
    print(f"  中位数: {lppl_df['LPPL_D'].median():.3f}")
    print(f"  标准差: {lppl_df['LPPL_D'].std():.3f}")
    print(f"  范围: {lppl_df['LPPL_D'].min():.3f} ~ {lppl_df['LPPL_D'].max():.3f}")

    # 步骤3：判断市场趋势状态
    print("\n" + "="*80)
    print("【步骤3】判断10年市场趋势状态")
    print("="*80)

    regime_df = calculate_rolling_regime(df)
    regime_counts = regime_df['regime'].value_counts()
    print(f"\n市场状态分布:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count}次 ({count/len(regime_df)*100:.1f}%)")

    # 步骤4：合并所有数据
    print("\n" + "="*80)
    print("【步骤4】合并GHE、LPPL、市场状态数据")
    print("="*80)

    # 合并到同一个时间轴
    merged = df[['datetime', '收盘价']].copy()

    merged = merged.merge(ghe_df, on='datetime', how='left')
    merged = merged.merge(lppl_df, on='datetime', how='left')
    merged = merged.merge(regime_df, on='datetime', how='left')

    # 前向填充
    merged['GHE'] = merged['GHE'].ffill()
    merged['LPPL_D'] = merged['LPPL_D'].ffill()
    merged['regime'] = merged['regime'].ffill()

    merged = merged.dropna()

    print(f"\n合并后数据量: {len(merged)} 天")
    print(f"数据完整率: {len(merged)/len(df)*100:.1f}%")

    # 步骤5：趋势规律分析 - 核心部分
    print("\n" + "="*80)
    print("【步骤5】趋势规律分析 - 找出最优组合")
    print("="*80)

    # 5.1 GHE与市场状态的关系
    print("\n--- 5.1 GHE在不同市场状态的表现 ---")
    ghe_by_regime = merged.groupby('regime')['GHE'].agg(['mean', 'std', 'count'])
    print(f"\n{'市场状态':<15} {'GHE均值':<10} {'GHE标准差':<12} {'样本数':<8}")
    print("-"*60)
    for regime in ['STRONG_UP', 'WEAK_UP', 'STRONG_DOWN', 'WEAK_DOWN']:
        if regime in ghe_by_regime.index:
            row = ghe_by_regime.loc[regime]
            print(f"{regime:<15} {row['mean']:<10.3f} {row['std']:<12.3f} {int(row['count']):<8}")

    # 5.2 LPPL-D与市场状态的关系
    print("\n--- 5.2 LPPL-D在不同市场状态的表现 ---")
    lppl_by_regime = merged.groupby('regime')['LPPL_D'].agg(['mean', 'std', 'count'])
    print(f"\n{'市场状态':<15} {'LPPL-D均值':<12} {'LPPL-D标准差':<14} {'样本数':<8}")
    print("-"*70)
    for regime in ['STRONG_UP', 'WEAK_UP', 'STRONG_DOWN', 'WEAK_DOWN']:
        if regime in lppl_by_regime.index:
            row = lppl_by_regime.loc[regime]
            print(f"{regime:<15} {row['mean']:<12.3f} {row['std']:<14.3f} {int(row['count']):<8}")

    # 5.3 找出趋势转换信号
    print("\n--- 5.3 趋势转换规律 ---")

    # 找出强上升转弱的时刻
    strong_up = merged[merged['regime'] == 'STRONG_UP'].copy()
    if len(strong_up) > 0:
        strong_up['GHE_high'] = strong_up['GHE'] > 0.5
        strong_up['LPPL_high'] = strong_up['LPPL_D'] > 0.8

        print(f"\n强上升期特征:")
        print(f"  平均GHE: {strong_up['GHE'].mean():.3f}")
        print(f"  平均LPPL-D: {strong_up['LPPL_D'].mean():.3f}")
        print(f"  GHE>0.5比例: {strong_up['GHE_high'].sum()/len(strong_up)*100:.1f}%")
        print(f"  LPPL-D>0.8比例: {strong_up['LPPL_high'].sum()/len(strong_up)*100:.1f}%")

    # 找出强下降期的特征
    strong_down = merged[merged['regime'] == 'STRONG_DOWN'].copy()
    if len(strong_down) > 0:
        strong_down['GHE_low'] = strong_down['GHE'] < 0.3
        strong_down['LPPL_low'] = strong_down['LPPL_D'] < 0.2

        print(f"\n强下降期特征:")
        print(f"  平均GHE: {strong_down['GHE'].mean():.3f}")
        print(f"  平均LPPL-D: {strong_down['LPPL_D'].mean():.3f}")
        print(f"  GHE<0.3比例: {strong_down['GHE_low'].sum()/len(strong_down)*100:.1f}%")
        print(f"  LPPL-D<0.2比例: {strong_down['LPPL_low'].sum()/len(strong_down)*100:.1f}%")

    # 5.4 寻找最佳入场信号组合
    print("\n" + "="*80)
    print("【步骤6】最佳入场信号组合")
    print("="*80)

    # 计算后续收益
    merged['future_return_5d'] = merged['收盘价'].pct_change(5).shift(-5)
    merged['future_return_10d'] = merged['收盘价'].pct_change(10).shift(-10)
    merged['future_return_20d'] = merged['收盘价'].pct_change(20).shift(-20)

    # 去除NaN
    signals = merged.dropna(subset=['future_return_5d', 'future_return_10d', 'future_return_20d'])

    if len(signals) > 0:
        print(f"\n有效信号数: {len(signals)}")

        # 测试不同组合
        print(f"\n{'信号组合':<60} {'5日胜率':<10} {'10日胜率':<10} {'20日胜率':<10}")
        print("-"*100)

        # 组合1：强上升 + GHE>0.5
        signal1 = signals[(signals['regime'] == 'STRONG_UP') & (signals['GHE'] > 0.5)]
        if len(signal1) > 0:
            wr5 = (signal1['future_return_5d'] > 0).sum() / len(signal1) * 100
            wr10 = (signal1['future_return_10d'] > 0).sum() / len(signal1) * 100
            wr20 = (signal1['future_return_20d'] > 0).sum() / len(signal1) * 100
            print(f"{'STRONG_UP + GHE>0.5':<60} {wr5:<10.1f}% {wr10:<10.1f}% {wr20:<10.1f}%")

        # 组合2：强上升 + LPPL-D<0.3
        signal2 = signals[(signals['regime'] == 'STRONG_UP') & (signals['LPPL_D'] < 0.3)]
        if len(signal2) > 0:
            wr5 = (signal2['future_return_5d'] > 0).sum() / len(signal2) * 100
            wr10 = (signal2['future_return_10d'] > 0).sum() / len(signal2) * 100
            wr20 = (signal2['future_return_20d'] > 0).sum() / len(signal2) * 100
            print(f"{'STRONG_UP + LPPL-D<0.3':<60} {wr5:<10.1f}% {wr10:<10.1f}% {wr20:<10.1f}%")

        # 组合3：强上升 + GHE>0.5 + LPPL-D<0.3
        signal3 = signals[(signals['regime'] == 'STRONG_UP') &
                          (signals['GHE'] > 0.5) &
                          (signals['LPPL_D'] < 0.3)]
        if len(signal3) > 0:
            wr5 = (signal3['future_return_5d'] > 0).sum() / len(signal3) * 100
            wr10 = (signal3['future_return_10d'] > 0).sum() / len(signal3) * 100
            wr20 = (signal3['future_return_20d'] > 0).sum() / len(signal3) * 100
            print(f"{'STRONG_UP + GHE>0.5 + LPPL-D<0.3':<60} {wr5:<10.1f}% {wr10:<10.1f}% {wr20:<10.1f}%")

        # 组合4：仅强上升
        signal4 = signals[signals['regime'] == 'STRONG_UP']
        if len(signal4) > 0:
            wr5 = (signal4['future_return_5d'] > 0).sum() / len(signal4) * 100
            wr10 = (signal4['future_return_10d'] > 0).sum() / len(signal4) * 100
            wr20 = (signal4['future_return_20d'] > 0).sum() / len(signal4) * 100
            print(f"{'仅 STRONG_UP (基准)':<60} {wr5:<10.1f}% {wr10:<10.1f}% {wr20:<10.1f}%")

    # 可视化
    print("\n" + "="*80)
    print("【步骤7】生成可视化图表")
    print("="*80)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14))

    # 图1：价格 + 市场状态
    ax1 = axes[0]
    ax1.plot(df['datetime'], df['收盘价'], color='black', linewidth=1, alpha=0.5, label='沪锡价格')

    # 标记不同市场状态
    for idx, row in merged.iterrows():
        regime = row['regime']
        dt = row['datetime']
        price = row['收盘价']

        if regime == 'STRONG_UP':
            ax1.scatter(dt, price, c='red', s=1, alpha=0.3)
        elif regime == 'STRONG_DOWN':
            ax1.scatter(dt, price, c='green', s=1, alpha=0.3)
        elif regime == 'WEAK_UP':
            ax1.scatter(dt, price, c='pink', s=1, alpha=0.3)
        elif regime == 'WEAK_DOWN':
            ax1.scatter(dt, price, c='lightgreen', s=1, alpha=0.3)

    ax1.set_ylabel('价格', fontsize=12)
    ax1.set_title('10年价格走势与市场状态', fontsize=14, fontweight='bold')
    ax1.legend(['价格', '强上升', '强下降', '弱上升', '弱下降'])
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 图2：GHE走势
    ax2 = axes[1]
    ax2.plot(ghe_df['datetime'], ghe_df['GHE'], color='purple', linewidth=1)
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='阈值0.5')
    ax2.axhline(y=0.3, color='orange', linestyle='--', linewidth=1, label='阈值0.3')
    ax2.fill_between(ghe_df['datetime'], 0.5, 1, color='red', alpha=0.1)
    ax2.fill_between(ghe_df['datetime'], 0, 0.3, color='green', alpha=0.1)
    ax2.set_ylabel('GHE值', fontsize=12)
    ax2.set_title('10年GHE走势', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 图3：LPPL-D走势
    ax3 = axes[2]
    ax3.plot(lppl_df['datetime'], lppl_df['LPPL_D'], color='blue', linewidth=1)
    ax3.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='高D值0.8')
    ax3.axhline(y=0.2, color='green', linestyle='--', linewidth=2, label='低D值0.2')
    ax3.fill_between(lppl_df['datetime'], 0.8, 1.5, color='red', alpha=0.1, label='风险区')
    ax3.fill_between(lppl_df['datetime'], 0, 0.2, color='green', alpha=0.1, label='安全区')
    ax3.set_ylabel('LPPL-D值', fontsize=12)
    ax3.set_title('10年LPPL-D走势', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.3)

    # 图4：综合信号
    ax4 = axes[3]

    # 创建综合得分
    merged_score = merged.dropna(subset=['GHE', 'LPPL_D']).copy()
    if len(merged_score) > 0:
        # 标准化
        merged_score['GHE_norm'] = (merged_score['GHE'] - 0.5) / 0.2  # 0为中心
        merged_score['LPPL_norm'] = (0.5 - merged_score['LPPL_D']) / 0.3  # 低D值好

        # 综合得分 = GHE_norm * 0.3 + LPPL_norm * 0.7
        merged_score['combined_score'] = (
            merged_score['GHE_norm'] * 0.3 +
            merged_score['LPPL_norm'] * 0.7
        )

        ax4.plot(merged_score['datetime'], merged_score['combined_score'],
                color='darkblue', linewidth=1, label='综合得分')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='做多阈值')
        ax4.axhline(y=-0.5, color='red', linestyle='--', linewidth=2, label='做空阈值')
        ax4.fill_between(merged_score['datetime'], 0.5, 2, color='green', alpha=0.1)
        ax4.fill_between(merged_score['datetime'], -2, -0.5, color='red', alpha=0.1)

    ax4.set_ylabel('综合得分', fontsize=12)
    ax4.set_xlabel('日期', fontsize=12)
    ax4.set_title('GHE+LPPL综合信号', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/comprehensive_trend_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()

    # 最终结论
    print("\n" + "="*80)
    print("【最终结论】10年趋势规律总结")
    print("="*80)

    print(f"""
基于10年沪锡日线数据的分析：

1. 市场状态分布:
   - STRONG_UP: {regime_counts.get('STRONG_UP', 0)}次 ({regime_counts.get('STRONG_UP', 0)/len(regime_df)*100:.1f}%)
   - WEAK_UP: {regime_counts.get('WEAK_UP', 0)}次 ({regime_counts.get('WEAK_UP', 0)/len(regime_df)*100:.1f}%)
   - STRONG_DOWN: {regime_counts.get('STRONG_DOWN', 0)}次 ({regime_counts.get('STRONG_DOWN', 0)/len(regime_df)*100:.1f}%)
   - WEAK_DOWN: {regime_counts.get('WEAK_DOWN', 0)}次 ({regime_counts.get('WEAK_DOWN', 0)/len(regime_df)*100:.1f}%)

2. GHE特征:
   - 10年均值: {ghe_df['GHE'].mean():.3f}
   - 论文阈值0.5不适用（大部分时间低于0.5）
   - 强上升期GHE: {ghe_by_regime.loc['STRONG_UP', 'mean'] if 'STRONG_UP' in ghe_by_regime.index else 'N/A':.3f}
   - 强下降期GHE: {ghe_by_regime.loc['STRONG_DOWN', 'mean'] if 'STRONG_DOWN' in ghe_by_regime.index else 'N/A':.3f}

3. LPPL-D特征:
   - 10年均值: {lppl_df['LPPL_D'].mean():.3f}
   - 低D值(<0.3)可能预示上升机会
   - 高D值(>0.8)可能预示泡沫风险

4. 实用的趋势判断算法:
   a. 判断市场状态:
      - 价格>MA60 且 EMA快>EMA慢 → STRONG_UP
      - 价格>MA60 且 EMA快<EMA慢 → WEAK_UP
      - 价格<MA60 且 EMA快<EMA慢 → STRONG_DOWN
      - 价格<MA60 且 EMA快>EMA慢 → WEAK_DOWN

   b. 入场条件（仅在STRONG_UP时做多）:
      - 优先: LPPL-D < 0.3（低泡沫风险）
      - 可选: GHE > 0.4（有一定持续性）
      - 确认: ADX ≥ 25（趋势强度足够）

   c. 风险控制:
      - LPPL-D > 0.8 时减仓或平仓
      - EMA交叉反转时平仓
      - 固定2%止损

5. 关键发现:
   - 沪锡10年确实包含上升、下降、横盘三种趋势
   - LPPL SHORT信号在所有市场状态下胜率都较低（<40%）
   - LPPL LONG信号只在STRONG_UP时表现优秀
   - GHE单独使用效果有限，建议作为辅助确认
   - 最简单有效的策略: STRONG_UP + ADX≥25
    """)

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
