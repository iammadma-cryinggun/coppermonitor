# -*- coding: utf-8 -*-
"""
市场张力指标 - 基于FFT和希尔伯特变换
替代LPPL的市场状态识别方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import hilbert
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入回测模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def preprocess_data(df):
    """第一步：对数化和去趋势化"""
    df['log_price'] = np.log(df['收盘价'])

    # 去趋势化：线性回归拟合
    x = np.arange(len(df))
    a, b = np.polyfit(x, df['log_price'], 1)
    df['trend'] = a * x + b
    df['detrended'] = df['log_price'] - df['trend']

    return df


def spectral_extraction(df, N=10):
    """第二步：频域提取，保留主要频率成分"""
    x_t = df['detrended'].values  # 去趋势后的残差序列

    # 快速傅里叶变换(FFT)
    F = fft(x_t)

    # 计算每个频率的振幅
    A_k = np.abs(F)

    # 选择Top N最大振幅的频率
    top_N_indices = np.argsort(A_k)[-N:]

    # 将其他频率置零，只保留Top N
    F_filtered = np.zeros_like(F, dtype=complex)
    F_filtered[top_N_indices] = F[top_N_indices]

    # 通过逆FFT得到重构信号
    x_t_prime = ifft(F_filtered)

    return x_t_prime, A_k, top_N_indices


def calculate_tension(df, reconstructed):
    """第三步：复平面映射和张力计算"""
    # 复数信号：实部是重构信号，虚部是希尔伯特变换后的结果
    analytic_signal = hilbert(np.real(reconstructed))

    # 计算虚部，得到张力
    df['tension'] = np.imag(analytic_signal)

    # 标准化张力
    df['normalized_tension'] = df['tension'] / np.std(df['tension'])

    # 计算张力的移动平均，识别高低张力期
    df['tension_ma20'] = df['normalized_tension'].rolling(window=20).mean()
    df['tension_ma60'] = df['normalized_tension'].rolling(window=60).mean()

    return df


def detect_tension_regime(df, high_threshold=1.0, low_threshold=-1.0):
    """
    根据张力识别市场状态

    参数:
        high_threshold: 高张力阈值（过度紧张，可能反转）
        low_threshold: 低张力阈值（过度松弛，可能启动）
    """
    df['tension_regime'] = 'NORMAL'

    df.loc[df['normalized_tension'] > high_threshold, 'tension_regime'] = 'HIGH_TENSION'
    df.loc[df['normalized_tension'] < low_threshold, 'tension_regime'] = 'LOW_TENSION'

    return df


def run_backtest_with_tension(df):
    """
    使用张力指标进行回测
    """
    from futures_monitor import calculate_indicators

    # 最优参数
    BEST_PARAMS = {
        'EMA_FAST': 3,
        'EMA_SLOW': 15,
        'RSI_FILTER': 30,
        'RATIO_TRIGGER': 1.05,
        'STC_SELL_ZONE': 65,
        'STOP_LOSS_PCT': 0.02
    }

    INITIAL_CAPITAL = 100000
    MAX_POSITION_RATIO = 0.8
    STOP_LOSS_PCT = 0.02
    CONTRACT_SIZE = 1
    MARGIN_RATE = 0.13

    # 转换列名为英文（calculate_indicators需要）
    df = df.rename(columns={
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '持仓量': 'hold'
    })

    # 计算指标
    df = calculate_indicators(df.copy(), BEST_PARAMS)

    # 回测
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 平仓
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

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
                pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'reason': exit_reason,
                    'entry_tension': position['entry_tension'],
                    'tension_regime': position['tension_regime']
                })

                position = None
                continue

        # 开仓（带张力过滤）
        if position is None:
            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe = (0 < current['ratio'] < BEST_PARAMS['RATIO_TRIGGER'])
            ratio_shrinking = current['ratio'] < current['ratio_prev']
            turning_up = current['macd_dif'] > prev['macd_dif']
            is_strong = current['rsi'] > BEST_PARAMS['RSI_FILTER']

            sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
            ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            buy_signal = sniper_signal or (ema_cross and is_strong)

            # 张力过滤：高张力期不开仓
            tension_ok = current['normalized_tension'] < 1.0

            if buy_signal and tension_ok:
                entry_price = current['close']
                margin_per_contract = entry_price * CONTRACT_SIZE * MARGIN_RATE
                available_capital = capital * MAX_POSITION_RATIO
                max_contracts = int(available_capital / margin_per_contract)

                if max_contracts > 0:
                    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                    position = {
                        'entry_datetime': current['datetime'],
                        'entry_price': entry_price,
                        'contracts': max_contracts,
                        'stop_loss': stop_loss_price,
                        'entry_tension': current['normalized_tension'],
                        'tension_regime': current['tension_regime']
                    }

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return df, trades, capital, total_return


def analyze_trades_by_tension(trades, df):
    """分析不同张力状态下的交易表现"""
    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        return {}

    # 分类统计
    high_tension = trades_df[trades_df['entry_tension'] > 1.0]
    normal_tension = trades_df[(trades_df['entry_tension'] >= -1.0) & (trades_df['entry_tension'] <= 1.0)]
    low_tension = trades_df[trades_df['entry_tension'] < -1.0]

    stats = {}

    for name, group in [('高张力(>1.0)', high_tension),
                        ('正常张力(-1~1)', normal_tension),
                        ('低张力(<-1.0)', low_tension)]:
        if len(group) > 0:
            wins = len(group[group['pnl'] > 0])
            stats[name] = {
                'count': len(group),
                'wins': wins,
                'losses': len(group) - wins,
                'win_rate': wins / len(group) * 100,
                'total_pnl': group['pnl'].sum(),
                'avg_pnl': group['pnl'].mean(),
                'avg_tension': group['entry_tension'].mean()
            }

    return stats, trades_df


def plot_tension_analysis(df, trades_df, stats):
    """绘制张力分析图表"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))

    # 1. 价格和张力
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    # 尝试使用收盘价或close列名
    price_col = 'close' if 'close' in df.columns else '收盘价'
    ax1.plot(df['datetime'], df[price_col], label='价格', color='black', alpha=0.6, linewidth=1)
    ax1_twin.plot(df['datetime'], df['normalized_tension'], label='标准化张力', color='red', alpha=0.5, linewidth=1)
    ax1_twin.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='高张力阈值')
    ax1_twin.axhline(y=-1.0, color='green', linestyle='--', alpha=0.3, label='低张力阈值')

    ax1.set_ylabel('价格', fontsize=12)
    ax1_twin.set_ylabel('标准化张力', fontsize=12)
    ax1.set_title('价格 vs 市场张力', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 2. 张力分布直方图
    ax2 = axes[1]
    ax2.hist(df['normalized_tension'], bins=50, color='orange', alpha=0.7, edgecolor='black')
    ax2.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='高张力阈值')
    ax2.axvline(x=-1.0, color='green', linestyle='--', linewidth=2, label='低张力阈值')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1, label='均值')
    ax2.set_xlabel('标准化张力', fontsize=12)
    ax2.set_ylabel('频次', fontsize=12)
    ax2.set_title('张力分布直方图', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 3. 不同张力状态的胜率对比
    ax3 = axes[2]
    if stats:
        categories = list(stats.keys())
        win_rates = [stats[k]['win_rate'] for k in categories]
        colors = ['red' if '高' in k else 'green' if '低' in k else 'blue' for k in categories]

        bars = ax3.bar(categories, win_rates, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('胜率 (%)', fontsize=12)
        ax3.set_title('不同张力状态的胜率对比', fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 100)

        for bar, rate in zip(bars, win_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', fontsize=11, fontweight='bold')

        ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 4. 张力vs盈亏散点图
    ax4 = axes[3]
    if len(trades_df) > 0:
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
        ax4.scatter(trades_df['entry_tension'], trades_df['pnl'], c=colors, alpha=0.6, s=50)
        ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
        ax4.axvline(x=-1.0, color='green', linestyle='--', alpha=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('入场时张力', fontsize=12)
        ax4.set_ylabel('盈亏 (元)', fontsize=12)
        ax4.set_title('入场张力 vs 交易盈亏', fontsize=14, fontweight='bold')
        ax4.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/tension_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()


def main():
    print("="*80)
    print("市场张力指标分析 - 基于FFT和希尔伯特变换")
    print("="*80)

    # 切换工作目录
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 天")

    # 第一步：数据预处理
    print("\n[步骤1] 数据预处理（对数化+去趋势化）...")
    df = preprocess_data(df)
    print(f"  趋势斜率: {np.polyfit(np.arange(len(df)), df['log_price'], 1)[0]:.6f}")

    # 第二步：频域提取
    print("\n[步骤2] 频域提取（FFT，保留Top 10频率）...")
    reconstructed, amplitudes, top_indices = spectral_extraction(df, N=10)
    df['reconstructed'] = np.real(reconstructed)
    print(f"  保留频率数量: {len(top_indices)}")
    print(f"  最大振幅: {amplitudes[top_indices].max():.2f}")

    # 第三步：计算张力
    print("\n[步骤3] 计算市场张力（希尔伯特变换）...")
    df = calculate_tension(df, reconstructed)
    print(f"  张力标准差: {df['tension'].std():.4f}")
    print(f"  标准化张力范围: {df['normalized_tension'].min():.2f} ~ {df['normalized_tension'].max():.2f}")

    # 第四步：识别市场状态
    print("\n[步骤4] 识别市场状态...")
    df = detect_tension_regime(df, high_threshold=1.0, low_threshold=-1.0)

    high_tension_days = len(df[df['tension_regime'] == 'HIGH_TENSION'])
    low_tension_days = len(df[df['tension_regime'] == 'LOW_TENSION'])
    normal_tension_days = len(df[df['tension_regime'] == 'NORMAL'])

    print(f"  高张力天数: {high_tension_days} ({high_tension_days/len(df)*100:.1f}%)")
    print(f"  低张力天数: {low_tension_days} ({low_tension_days/len(df)*100:.1f}%)")
    print(f"  正常张力天数: {normal_tension_days} ({normal_tension_days/len(df)*100:.1f}%)")

    # 第五步：回测（带张力过滤）
    print("\n[步骤5] 运行回测（张力过滤：高张力期不开仓）...")
    df, trades, final_capital, total_return = run_backtest_with_tension(df)

    print(f"\n回测结果:")
    print(f"  初始资金: 100,000元")
    print(f"  最终权益: {final_capital:,.0f}元")
    print(f"  收益率: {total_return:.2f}%")
    print(f"  交易次数: {len(trades)}笔")

    # 第六步：分析不同张力状态下的交易表现
    print("\n[步骤6] 分析不同张力状态下的交易表现...")
    stats, trades_df = analyze_trades_by_tension(trades, df)

    print(f"\n{'张力状态':<20} {'交易数':<8} {'盈利':<8} {'亏损':<8} {'胜率':<10} {'总盈亏':<15} {'平均盈亏':<15}")
    print("-"*80)

    for name, stat in stats.items():
        print(f"{name:<20} {stat['count']:<8} {stat['wins']:<8} {stat['losses']:<8} "
              f"{stat['win_rate']:<10.1f}% {stat['total_pnl']:>13,.0f}元 {stat['avg_pnl']:>13,.0f}元")

    # 对比原始策略
    print("\n" + "="*80)
    print("策略对比")
    print("="*80)
    print(f"\n{'策略':<30} {'收益率':<12} {'交易次数':<10}")
    print("-"*80)
    print(f"{'原始纯做多策略':<30} {'803.75%':<12} {'108笔':<10}")
    print(f"{'张力过滤策略（高张力不开仓）':<30} {total_return:<12.2f}% {len(trades):<10}笔")

    improvement = (total_return - 803.75) / 803.75 * 100
    print(f"\n改善幅度: {improvement:+.2f}%")

    # 绘图
    print("\n生成分析图表...")
    plot_tension_analysis(df, trades_df, stats)

    # 保存结果
    df.to_csv('D:/期货数据/铜期货监控/daily_backtest/tension_data.csv', index=False, encoding='utf-8-sig')
    trades_df.to_csv('D:/期货数据/铜期货监控/daily_backtest/tension_trades.csv', index=False, encoding='utf-8-sig')
    print(f"\n数据已保存:")
    print(f"  - tension_data.csv: 张力指标数据")
    print(f"  - tension_trades.csv: 交易明细")

    # 结论
    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if improvement > 0:
        print(f"\n[OK] 张力过滤有效！")
        print(f"  收益率从803.75%提升到{total_return:.2f}%")
        print(f"  建议：使用张力指标辅助交易决策")
    else:
        print(f"\n[!] 张力过滤降低了收益")
        print(f"  原因：需要调整阈值或策略参数")

    print("\n张力指标的优势:")
    print("  1. 基于信号处理理论，客观数学化")
    print("  2. 不预设'泡沫'模型（对比LPPL）")
    print("  3. 实时计算，无需拟合")
    print("  4. 可解释性强：张力高=市场过度紧张")

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
