# -*- coding: utf-8 -*-
"""
反向分析：盈利交易 vs 亏损交易的GHE+LPPL特征
核心思路：从历史结果中找规律，而不是预测未来
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

from futures_monitor import calculate_indicators


# ============== GHE和LPPL计算 ==============
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


def lppl_func(t, A, B, tc, m, C, omega, phi):
    """LPPL核心函数"""
    dt = np.clip(tc - t, 1e-8, None)
    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))


def calculate_lppl_d(prices):
    """计算LPPL-D值"""
    if len(prices) < 200:
        return None

    log_prices = np.log(prices.values if hasattr(prices, 'values') else prices)
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
        return D

    except:
        return None


# ============== 回测并记录每笔交易的GHE+LPPL ==============
def backtest_with_indicators(df, params):
    """
    回测策略，记录每笔交易的GHE和LPPL值
    """
    # 计算EMA+STC指标
    df = df.rename(columns={
        '收盘价': 'close',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '成交量': 'volume',
        '持仓量': 'hold'
    })

    df = calculate_indicators(df, params)

    # 计算GHE和LPPL-D
    print("计算GHE和LPPL-D...")
    df['GHE'] = np.nan
    df['LPPL_D'] = np.nan

    for i in range(200, len(df)):
        # GHE
        try:
            ghe = calculate_ghe_single_window(df['close'].iloc[i-100:i])
            df.loc[df.index[i], 'GHE'] = ghe
        except:
            pass

        # LPPL-D
        try:
            lppl_d = calculate_lppl_d(df['close'].iloc[i-200:i])
            df.loc[df.index[i], 'LPPL_D'] = lppl_d
        except:
            pass

    # 前向填充
    df['GHE'] = df['GHE'].ffill()
    df['LPPL_D'] = df['LPPL_D'].ffill()

    # 模拟交易
    capital = 100000
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1] if i > 0 else current

        # 平仓逻辑
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 止损
            if current['low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_triggered = True
                exit_reason = '止损'

            # STC止盈
            elif (prev['stc'] > params['STC_SELL_ZONE'] and
                  current['stc'] < prev['stc']):
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'STC止盈'

            # 趋势反转
            elif current['ema_fast'] < current['ema_slow']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = '趋势反转'

            if exit_triggered:
                pnl = (exit_price - position['entry_price']) * position['contracts']
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': exit_reason,
                    'entry_ghe': position['entry_ghe'],
                    'entry_lppl': position['entry_lppl'],
                    'exit_ghe': current['GHE'],
                    'exit_lppl': current['LPPL_D']
                })

                position = None
                continue

        # 开仓逻辑
        if position is None:
            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe = 0 < current['ratio'] < params['RATIO_TRIGGER']
            ratio_shrinking = current['ratio'] < current['ratio_prev']
            turning_up = current['macd_dif'] > prev['macd_dif']
            is_strong = current['rsi'] > params['RSI_FILTER']

            sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
            ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            buy_signal = sniper_signal or (ema_cross and is_strong)

            if buy_signal:
                entry_price = current['close']
                margin_per_contract = entry_price * 1 * 0.13
                available_capital = capital * 0.8
                max_contracts = int(available_capital / margin_per_contract)

                if max_contracts > 0:
                    stop_loss_price = entry_price * 0.98
                    position = {
                        'entry_datetime': current['datetime'],
                        'entry_price': entry_price,
                        'contracts': max_contracts,
                        'stop_loss': stop_loss_price,
                        'entry_ghe': current['GHE'],
                        'entry_lppl': current['LPPL_D']
                    }

    return {
        'capital': capital,
        'trades': trades,
        'total_return_pct': (capital - 100000) / 100000 * 100,
        'trade_count': len(trades)
    }


# ============== 主分析 ==============
def main():
    print("="*80)
    print("反向分析：盈利交易 vs 亏损交易的GHE+LPPL特征")
    print("="*80)

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 天")

    # 最佳参数
    BEST_PARAMS = {
        'EMA_FAST': 3,
        'EMA_SLOW': 15,
        'RSI_FILTER': 30,
        'RATIO_TRIGGER': 1.05,
        'STC_SELL_ZONE': 65,
        'STOP_LOSS_PCT': 0.02
    }

    print(f"\n使用参数: EMA({BEST_PARAMS['EMA_FAST']},{BEST_PARAMS['EMA_SLOW']}), "
          f"RSI={BEST_PARAMS['RSI_FILTER']}, RATIO={BEST_PARAMS['RATIO_TRIGGER']}, "
          f"STC={BEST_PARAMS['STC_SELL_ZONE']}")

    # 回测
    print("\n执行回测...")
    result = backtest_with_indicators(df, BEST_PARAMS)

    trades_df = pd.DataFrame(result['trades'])

    print(f"\n总交易数: {len(trades_df)}笔")
    print(f"总收益: {result['total_return_pct']:.2f}%")
    print(f"最终资金: {result['capital']:,.0f}元")

    if len(trades_df) == 0:
        print("无交易数据")
        return

    # 分组
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]

    print(f"\n盈利交易: {len(winning_trades)}笔")
    print(f"亏损交易: {len(losing_trades)}笔")
    print(f"胜率: {len(winning_trades)/len(trades_df)*100:.1f}%")

    # ============== 核心分析：对比GHE和LPPL特征 ==============
    print("\n" + "="*80)
    print("【核心分析】盈利交易 vs 亏损交易的GHE+LPPL特征")
    print("="*80)

    # 1. GHE对比
    print("\n--- GHE特征对比 ---")

    win_ghe = winning_trades['entry_ghe'].dropna()
    lose_ghe = losing_trades['entry_ghe'].dropna()

    print(f"\n{'指标':<20} {'盈利交易':<15} {'亏损交易':<15} {'差异':<15}")
    print("-"*70)

    if len(win_ghe) > 0:
        print(f"{'GHE均值':<20} {win_ghe.mean():<15.3f} {lose_ghe.mean():<15.3f} {win_ghe.mean()-lose_ghe.mean():<+15.3f}")
        print(f"{'GHE中位数':<20} {win_ghe.median():<15.3f} {lose_ghe.median():<15.3f} {win_ghe.median()-lose_ghe.median():<+15.3f}")
        print(f"{'GHE标准差':<20} {win_ghe.std():<15.3f} {lose_ghe.std():<15.3f}")
        print(f"{'GHE最小值':<20} {win_ghe.min():<15.3f} {lose_ghe.min():<15.3f}")
        print(f"{'GHE最大值':<20} {win_ghe.max():<15.3f} {lose_ghe.max():<15.3f}")

    # 2. LPPL-D对比
    print("\n--- LPPL-D特征对比 ---")

    win_lppl = winning_trades['entry_lppl'].dropna()
    lose_lppl = losing_trades['entry_lppl'].dropna()

    print(f"\n{'指标':<20} {'盈利交易':<15} {'亏损交易':<15} {'差异':<15}")
    print("-"*70)

    if len(win_lppl) > 0:
        print(f"{'LPPL-D均值':<20} {win_lppl.mean():<15.3f} {lose_lppl.mean():<15.3f} {win_lppl.mean()-lose_lppl.mean():<+15.3f}")
        print(f"{'LPPL-D中位数':<20} {win_lppl.median():<15.3f} {lose_lppl.median():<15.3f} {win_lppl.median()-lose_lppl.median():<+15.3f}")
        print(f"{'LPPL-D标准差':<20} {win_lppl.std():<15.3f} {lose_lppl.std():<15.3f}")
        print(f"{'LPPL-D最小值':<20} {win_lppl.min():<15.3f} {lose_lppl.max():<15.3f}")

    # 3. 区间分布对比
    print("\n--- GHE区间分布对比 ---")

    ghe_bins = [0, 0.3, 0.4, 0.5, 1.0]
    ghe_labels = ['<0.3', '0.3-0.4', '0.4-0.5', '≥0.5']

    win_ghe_groups = pd.cut(win_ghe, bins=ghe_bins, labels=ghe_labels)
    lose_ghe_groups = pd.cut(lose_ghe, bins=ghe_bins, labels=ghe_labels)

    print(f"\n{'GHE区间':<12} {'盈利交易数':<12} {'亏损交易数':<12} {'盈利胜率':<12}")
    print("-"*60)

    for label in ghe_labels:
        win_count = len(win_ghe_groups[win_ghe_groups == label])
        lose_count = len(lose_ghe_groups[lose_ghe_groups == label])
        total = win_count + lose_count

        if total > 0:
            win_rate = win_count / total * 100
            print(f"{label:<12} {win_count:<12} {lose_count:<12} {win_rate:<12.1f}%")

    print("\n--- LPPL-D区间分布对比 ---")

    lppl_bins = [0, 0.3, 0.5, 0.8, 2.0]
    lppl_labels = ['<0.3', '0.3-0.5', '0.5-0.8', '≥0.8']

    win_lppl_groups = pd.cut(win_lppl, bins=lppl_bins, labels=lppl_labels)
    lose_lppl_groups = pd.cut(lose_lppl, bins=lppl_bins, labels=lppl_labels)

    print(f"\n{'LPPL-D区间':<12} {'盈利交易数':<12} {'亏损交易数':<12} {'盈利胜率':<12}")
    print("-"*60)

    for label in lppl_labels:
        win_count = len(win_lppl_groups[win_lppl_groups == label])
        lose_count = len(lose_lppl_groups[lose_lppl_groups == label])
        total = win_count + lose_count

        if total > 0:
            win_rate = win_count / total * 100
            print(f"{label:<12} {win_count:<12} {lose_count:<12} {win_rate:<12.1f}%")

    # 4. GHE+LPPL组合分析
    print("\n" + "="*80)
    print("【组合分析】GHE+LPPL不同组合的胜率")
    print("="*80)

    all_trades_with_indicators = trades_df.dropna(subset=['entry_ghe', 'entry_lppl']).copy()

    if len(all_trades_with_indicators) > 0:
        # 按GHE和LPPL-D分组
        all_trades_with_indicators['ghe_group'] = pd.cut(all_trades_with_indicators['entry_ghe'],
                                                          bins=ghe_bins, labels=ghe_labels)
        all_trades_with_indicators['lppl_group'] = pd.cut(all_trades_with_indicators['entry_lppl'],
                                                           bins=lppl_bins, labels=lppl_labels)

        print(f"\n{'GHE':<8} {'LPPL-D':<10} {'盈利':<6} {'亏损':<6} {'总交易':<8} {'胜率':<10}")
        print("-"*60)

        # 遍历所有组合
        combination_stats = []

        for ghe_label in ghe_labels:
            for lppl_label in lppl_labels:
                group = all_trades_with_indicators[
                    (all_trades_with_indicators['ghe_group'] == ghe_label) &
                    (all_trades_with_indicators['lppl_group'] == lppl_label)
                ]

                if len(group) > 0:
                    wins = len(group[group['pnl'] > 0])
                    total = len(group)
                    win_rate = wins / total * 100

                    print(f"{ghe_label:<8} {lppl_label:<10} {wins:<6} {total-wins:<6} {total:<8} {win_rate:<10.1f}%")

                    combination_stats.append({
                        'ghe': ghe_label,
                        'lppl': lppl_label,
                        'wins': wins,
                        'losses': total - wins,
                        'total': total,
                        'win_rate': win_rate
                    })

        # 找出最佳和最差组合
        if combination_stats:
            best = max(combination_stats, key=lambda x: x['win_rate'])
            worst = min(combination_stats, key=lambda x: x['win_rate'])

            print(f"\n最佳组合: GHE {best['ghe']}, LPPL-D {best['lppl']}")
            print(f"  胜率: {best['win_rate']:.1f}% ({best['wins']}/{best['total']})")

            print(f"\n最差组合: GHE {worst['ghe']}, LPPL-D {worst['lppl']}")
            print(f"  胜率: {worst['win_rate']:.1f}% ({worst['wins']}/{worst['total']})")

    # ============== 可视化 ==============
    print("\n生成可视化图表...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1：GHE分布对比
    ax1 = axes[0, 0]
    ax1.hist(win_ghe, bins=20, color='green', alpha=0.5, label='盈利交易', edgecolor='black')
    ax1.hist(lose_ghe, bins=20, color='red', alpha=0.5, label='亏损交易', edgecolor='black')
    ax1.axvline(win_ghe.mean(), color='green', linestyle='--', linewidth=2, label=f'盈利均值{win_ghe.mean():.3f}')
    ax1.axvline(lose_ghe.mean(), color='red', linestyle='--', linewidth=2, label=f'亏损均值{lose_ghe.mean():.3f}')
    ax1.set_xlabel('GHE值', fontsize=12)
    ax1.set_ylabel('交易数量', fontsize=12)
    ax1.set_title('盈利vs亏损交易的GHE分布', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 图2：LPPL-D分布对比
    ax2 = axes[0, 1]
    ax2.hist(win_lppl, bins=20, color='green', alpha=0.5, label='盈利交易', edgecolor='black')
    ax2.hist(lose_lppl, bins=20, color='red', alpha=0.5, label='亏损交易', edgecolor='black')
    ax2.axvline(win_lppl.mean(), color='green', linestyle='--', linewidth=2, label=f'盈利均值{win_lppl.mean():.3f}')
    ax2.axvline(lose_lppl.mean(), color='red', linestyle='--', linewidth=2, label=f'亏损均值{lose_lppl.mean():.3f}')
    ax2.set_xlabel('LPPL-D值', fontsize=12)
    ax2.set_ylabel('交易数量', fontsize=12)
    ax2.set_title('盈利vs亏损交易的LPPL-D分布', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 图3：GHE vs LPPL-D散点图
    ax3 = axes[1, 0]

    win_scatter = ax3.scatter(win_ghe, win_lppl, c='green', alpha=0.6, s=50,
                              label=f'盈利 ({len(win_ghe)}笔)', edgecolors='black')
    lose_scatter = ax3.scatter(lose_ghe, lose_lppl, c='red', alpha=0.6, s=50,
                               label=f'亏损 ({len(lose_ghe)}笔)', edgecolors='black')

    ax3.set_xlabel('GHE', fontsize=12)
    ax3.set_ylabel('LPPL-D', fontsize=12)
    ax3.set_title('盈利vs亏损交易的GHE-LPPL散点图', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.3)

    # 图4：胜率热力图
    ax4 = axes[1, 1]

    if combination_stats:
        # 创建热力图数据
        pivot_data = {}
        for stat in combination_stats:
            key = (stat['ghe'], stat['lppl'])
            pivot_data[key] = stat['win_rate']

        # 转换为矩阵
        matrix_data = np.zeros((len(ghe_labels), len(lppl_labels)))
        for i, ghe_label in enumerate(ghe_labels):
            for j, lppl_label in enumerate(lppl_labels):
                key = (ghe_label, lppl_label)
                if key in pivot_data:
                    matrix_data[i, j] = pivot_data[key]
                else:
                    matrix_data[i, j] = np.nan

        im = ax4.imshow(matrix_data, cmap='RdYlGn', vmin=0, vmax=100)
        ax4.set_xticks(np.arange(len(lppl_labels)))
        ax4.set_yticks(np.arange(len(ghe_labels)))
        ax4.set_xticklabels(lppl_labels)
        ax4.set_yticklabels(ghe_labels)
        ax4.set_xlabel('LPPL-D', fontsize=12)
        ax4.set_ylabel('GHE', fontsize=12)
        ax4.set_title('GHE+LPPL组合胜率热力图（%）', fontsize=14, fontweight='bold')

        # 添加数值标签
        for i in range(len(ghe_labels)):
            for j in range(len(lppl_labels)):
                if not np.isnan(matrix_data[i, j]):
                    text = ax4.text(j, i, f'{matrix_data[i, j]:.0f}%',
                                   ha="center", va="center", color="black", fontsize=9)

        plt.colorbar(im, ax=ax4, label='胜率(%)')

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/reverse_analysis_trades.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()

    # ============== 最终结论 ==============
    print("\n" + "="*80)
    print("【最终结论】从历史交易中找规律")
    print("="*80)

    print(f"""
基于{len(trades_df)}笔交易的分析：

1. GHE特征:
   - 盈利交易平均GHE: {win_ghe.mean():.3f}
   - 亏损交易平均GHE: {lose_ghe.mean():.3f}
   - 差异: {win_ghe.mean()-lose_ghe.mean():+.3f}

2. LPPL-D特征:
   - 盈利交易平均LPPL-D: {win_lppl.mean():.3f}
   - 亏损交易平均LPPL-D: {lose_lppl.mean():.3f}
   - 差异: {win_lppl.mean()-lose_lppl.mean():+.3f}

3. 实用建议（基于历史数据）:
   - 如果GHE和LPPL处于"{best['ghe']}"和"{best['lppl']}"区间
     → 历史胜率{best['win_rate']:.1f}%，建议积极做多

   - 如果GHE和LPPL处于"{worst['ghe']}"和"{worst['lppl']}"区间
     → 历史胜率{worst['win_rate']:.1f}%，建议谨慎或观望

4. 核心价值:
   - 不是预测未来，而是从历史结果中学习
   - 避免在历史上容易亏钱的组合时开仓
   - 优先在历史上容易赚钱的组合时开仓
    """)

    # 保存结果
    trades_df.to_csv('trades_with_ghe_lppl.csv', index=False, encoding='utf-8-sig')
    print(f"\n交易数据已保存: trades_with_ghe_lppl.csv")

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
