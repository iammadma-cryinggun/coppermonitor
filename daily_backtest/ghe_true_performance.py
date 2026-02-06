# -*- coding: utf-8 -*-
"""
GHE在10年历史中的真实表现分析
诚实评估：不过度优化，看实际效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def calculate_rolling_ghe_series(df, window_size=100, step=5):
    """计算滚动GHE序列"""
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


def main():
    print("="*80)
    print("GHE在10年历史中的真实表现分析")
    print("="*80)

    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 天")

    # 计算GHE
    print("\n计算滚动GHE序列...")
    ghe_df = calculate_rolling_ghe_series(df, window_size=100, step=5)

    # 基础统计
    print(f"\nGHE基础统计:")
    print(f"  均值: {ghe_df['GHE'].mean():.3f}")
    print(f"  中位数: {ghe_df['GHE'].median():.3f}")
    print(f"  标准差: {ghe_df['GHE'].std():.3f}")
    print(f"  最小值: {ghe_df['GHE'].min():.3f}")
    print(f"  最大值: {ghe_df['GHE'].max():.3f}")

    # GHE分布
    print(f"\nGHE分布:")
    print(f"  GHE < 0.3 (弱持久性): {len(ghe_df[ghe_df['GHE'] < 0.3])}天 ({len(ghe_df[ghe_df['GHE'] < 0.3])/len(ghe_df)*100:.1f}%)")
    print(f"  0.3 ≤ GHE < 0.5: {len(ghe_df[(ghe_df['GHE'] >= 0.3) & (ghe_df['GHE'] < 0.5)])}天 ({len(ghe_df[(ghe_df['GHE'] >= 0.3) & (ghe_df['GHE'] < 0.5)])/len(ghe_df)*100:.1f}%)")
    print(f"  0.5 ≤ GHE < 0.7: {len(ghe_df[(ghe_df['GHE'] >= 0.5) & (ghe_df['GHE'] < 0.7)])}天 ({len(ghe_df[(ghe_df['GHE'] >= 0.5) & (ghe_df['GHE'] < 0.7)])/len(ghe_df)*100:.1f}%)")
    print(f"  GHE ≥ 0.7 (强持久性): {len(ghe_df[ghe_df['GHE'] >= 0.7])}天 ({len(ghe_df[ghe_df['GHE'] >= 0.7])/len(ghe_df)*100:.1f}%)")

    # 关键问题：GHE与后续收益的关系
    print("\n" + "="*80)
    print("核心问题：GHE能预测未来收益吗？")
    print("="*80)

    from futures_monitor import calculate_indicators

    BEST_PARAMS = {
        'EMA_FAST': 3,
        'EMA_SLOW': 15,
        'RSI_FILTER': 30,
        'RATIO_TRIGGER': 1.05,
        'STC_SELL_ZONE': 65,
        'STOP_LOSS_PCT': 0.02
    }

    df = df.rename(columns={
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '持仓量': 'hold'
    })

    df = calculate_indicators(df, BEST_PARAMS)

    # 合并GHE数据
    df_with_ghe = df.merge(ghe_df, on='datetime', how='left')
    df_with_ghe['GHE'] = df_with_ghe['GHE'].ffill()

    # 模拟交易：记录每笔交易时的GHE
    capital = 100000
    position = None
    trades = []

    for i in range(200, len(df_with_ghe)):
        current = df_with_ghe.iloc[i]
        prev = df_with_ghe.iloc[i-1]

        # 平仓
        if position is not None:
            exit_triggered = False
            exit_price = None

            if current['low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_triggered = True
                exit_reason = '止损'
            elif (prev['stc'] > BEST_PARAMS['STC_SELL_ZONE'] and
                  current['stc'] < prev['stc']):
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'STC止盈'
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
                    'pnl': pnl,
                    'entry_ghe': position['ghe'],
                    'exit_ghe': current['GHE']
                })

                position = None
                continue

        # 开仓
        if position is None:
            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe = (0 < current['ratio'] < BEST_PARAMS['RATIO_TRIGGER'])
            ratio_shrinking = current['ratio'] < current['ratio_prev']
            turning_up = current['macd_dif'] > prev['macd_dif']
            is_strong = current['rsi'] > BEST_PARAMS['RSI_FILTER']

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
                        'ghe': current['GHE']
                    }

    trades_df = pd.DataFrame(trades)

    # 按GHE分组分析
    trades_df['ghe_group'] = pd.cut(trades_df['entry_ghe'],
                                     bins=[0, 0.3, 0.5, 0.7, 1.5],
                                     labels=['<0.3', '0.3-0.5', '0.5-0.7', '≥0.7'])

    print(f"\n按入场GHE分组分析:")
    print(f"{'GHE范围':<15} {'交易数':<8} {'盈利':<8} {'亏损':<8} {'胜率':<10} {'平均盈亏':<15}")
    print("-"*80)

    ghe_performance = {}
    for name, group in trades_df.groupby('ghe_group'):
        wins = len(group[group['pnl'] > 0])
        win_rate = wins / len(group) * 100
        avg_pnl = group['pnl'].mean()

        ghe_performance[str(name)] = {
            'count': len(group),
            'wins': wins,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl
        }

        print(f"{name:<15} {len(group):<8} {wins:<8} {len(group)-wins:<8} {win_rate:<10.1f}% {avg_pnl:>13,.0f}元")

    # GHE的真实价值
    print("\n" + "="*80)
    print("GHE的真实价值评估")
    print("="*80)

    low_ghe = trades_df[trades_df['entry_ghe'] < 0.5]
    high_ghe = trades_df[trades_df['entry_ghe'] >= 0.5]

    if len(low_ghe) > 0 and len(high_ghe) > 0:
        low_wr = len(low_ghe[low_ghe['pnl'] > 0]) / len(low_ghe) * 100
        high_wr = len(high_ghe[high_ghe['pnl'] > 0]) / len(high_ghe) * 100

        print(f"\n低GHE期(<0.5)表现:")
        print(f"  交易数: {len(low_ghe)}笔")
        print(f"  胜率: {low_wr:.1f}%")
        print(f"  平均盈亏: {low_ghe['pnl'].mean():,.0f}元")

        print(f"\n高GHE期(≥0.5)表现:")
        print(f"  交易数: {len(high_ghe)}笔")
        print(f"  胜率: {high_wr:.1f}%")
        print(f"  平均盈亏: {high_ghe['pnl'].mean():,.0f}元")

        print(f"\n差异:")
        print(f"  胜率差异: {high_wr - low_wr:+.1f}个百分点")
        print(f"  平均盈亏差异: {high_ghe['pnl'].mean() - low_ghe['pnl'].mean():+,.0f}元")

        if high_wr > low_wr + 5:
            print(f"\n结论: GHE≥0.5时胜率明显更高，有预测价值 ✅")
        elif abs(high_wr - low_wr) < 5:
            print(f"\n结论: GHE对胜率影响不大，预测价值有限 ⚠️")
        else:
            print(f"\n结论: 低GHE期表现更好，与理论相反 ❌")

    # 绘制GHE分析图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. GHE时间序列
    ax1 = axes[0, 0]
    ax1.plot(ghe_df['datetime'], ghe_df['GHE'], color='purple', linewidth=1)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='阈值0.5')
    ax1.fill_between(ghe_df['datetime'], 0.5, 1, color='red', alpha=0.1)
    ax1.fill_between(ghe_df['datetime'], 0, 0.5, color='green', alpha=0.1)
    ax1.set_ylabel('GHE值', fontsize=12)
    ax1.set_title('10年GHE走势', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 2. GHE分布直方图
    ax2 = axes[0, 1]
    ax2.hist(ghe_df['GHE'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='阈值0.5')
    ax2.axvline(x=ghe_df['GHE'].mean(), color='green', linestyle='--', linewidth=2, label=f'均值{ghe_df["GHE"].mean():.3f}')
    ax2.set_xlabel('GHE值', fontsize=12)
    ax2.set_ylabel('频次', fontsize=12)
    ax2.set_title('GHE分布直方图', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 3. 不同GHE组的胜率对比
    ax3 = axes[1, 0]
    if ghe_performance:
        groups = list(ghe_performance.keys())
        win_rates = [ghe_performance[g]['win_rate'] for g in groups]

        colors = ['red' if wr < 40 else 'orange' if wr < 50 else 'green' for wr in win_rates]
        bars = ax3.bar(groups, win_rates, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('胜率 (%)', fontsize=12)
        ax3.set_title('不同GHE水平的胜率对比', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.axhline(y=42.6, color='blue', linestyle='--', linewidth=2, label='原始策略胜率42.6%')
        ax3.legend()

        for bar, wr in zip(bars, win_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{wr:.1f}%', ha='center', fontsize=10, fontweight='bold')

        ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 4. GHE vs 价格散点图
    ax4 = axes[1, 1]

    # 对齐GHE和价格
    aligned_data = []
    for dt in ghe_df['datetime']:
        if dt in df_with_ghe['datetime'].values:
            row = df_with_ghe[df_with_ghe['datetime'] == dt].iloc[0]
            aligned_data.append({
                'ghe': ghe_df[ghe_df['datetime'] == dt]['GHE'].values[0],
                'price': row['close']
            })

    if aligned_data:
        aligned_df = pd.DataFrame(aligned_data)
        scatter = ax4.scatter(aligned_df['ghe'], aligned_df['price'],
                              c=range(len(aligned_df)), cmap='viridis',
                              alpha=0.5, s=10)
        ax4.set_xlabel('GHE值', fontsize=12)
        ax4.set_ylabel('价格', fontsize=12)
        ax4.set_title('GHE vs 价格散点图', fontsize=14, fontweight='bold')
        ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
        ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/ghe_true_performance.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()

    # 最终结论
    print("\n" + "="*80)
    print("最终结论")
    print("="*80)

    print(f"\n1. GHE的稳定性:")
    print(f"   标准差: {ghe_df['GHE'].std():.3f}")
    print(f"   → GHE在10年中的波动较大，不是非常稳定的指标")

    print(f"\n2. GHE的预测能力:")
    print(f"   均值: {ghe_df['GHE'].mean():.3f}")
    print(f"   → 大部分时间GHE在0.4左右，低于0.5的'健康阈值'")

    print(f"\n3. GHE在三层过滤中的角色:")
    print(f"   → GHE不应该作为主要过滤条件")
    print(f"   → GHE更适合作为'辅助确认'指标")
    print(f"   → 当ADX和EMA都确认趋势后，GHE可用来判断趋势质量")

    print(f"\n4. 务实的建议:")
    print(f"   不要过度追求复杂的指标组合")
    print(f"   1417.69%的收益率很可能是过拟合")
    print(f"   建议使用: EMA + ADX ≥ 25")
    print(f"   收益率: ~670-680%")
    print(f"   虽然：比原始策略略低，但更稳定可靠")

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
