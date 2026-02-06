# -*- coding: utf-8 -*-
"""
LPPL历史胜率验证脚本
核心思路：找出过去10年所有跌幅>20%的波段，检查LPPL是否提前预警
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")


def lppl_func(t, A, B, tc, m, C, omega, phi):
    """LPPL核心函数"""
    dt = np.clip(tc - t, 1e-8, None)
    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))


def fit_lppl_window(prices, start_idx, end_idx):
    """
    对指定窗口进行LPPL拟合

    返回: 拟合结果dict，如果失败返回None
    """
    log_prices = np.log(prices[start_idx:end_idx].values)
    t = np.arange(len(log_prices))
    last_t = t[-1]

    # 初始参数
    A0 = log_prices.max()
    B0 = -0.1
    tc0 = last_t + 30  # 预测30天后崩盘
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

        # 提取参数
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

        return {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'tc_relative': tc,
            'tc_global': start_idx + tc,
            'm': m,
            'omega': omega,
            'r_squared': r_squared,
            'window_size': end_idx - start_idx
        }
    except:
        return None


def detect_crash_events(df, crash_threshold=0.20, min_duration=10):
    """
    检测历史下跌事件（跌幅>crash_threshold，持续至少min_duration天）

    返回: list of {'peak_idx': int, 'peak_date': datetime, 'peak_price': float,
                    'crash_idx': int, 'crash_date': datetime, 'crash_price': float,
                    'drawdown': float, 'duration_days': int}
    """
    prices = df['收盘价'].values
    dates = df['datetime'].values

    crashes = []
    peak_idx = 0
    peak_price = prices[0]

    for i in range(1, len(prices)):
        current_price = prices[i]

        # 检测是否开始下跌
        if current_price > peak_price:
            peak_idx = i
            peak_price = current_price

        # 检测是否跌破阈值
        drawdown = (peak_price - current_price) / peak_price

        if drawdown >= crash_threshold:
            crash_idx = i

            # 计算持续时间
            duration_days = crash_idx - peak_idx

            if duration_days >= min_duration:
                crashes.append({
                    'peak_idx': peak_idx,
                    'peak_date': dates[peak_idx],
                    'peak_price': peak_price,
                    'crash_idx': crash_idx,
                    'crash_date': dates[crash_idx],
                    'crash_price': current_price,
                    'drawdown': drawdown,
                    'duration_days': duration_days
                })

            # 重置峰值
            peak_idx = i
            peak_price = current_price

    return crashes


def check_lppl_warning_before_crash(df, crash, warning_days=60, window_size=200):
    """
    检查在崩盘前warning_days天内，LPPL是否给出预警

    返回: dict with warning analysis results
    """
    crash_idx = crash['crash_idx']

    # 检查崩盘前60天的LPPL拟合
    start_idx = max(0, crash_idx - window_size - warning_days)
    end_idx = crash_idx - warning_days

    if end_idx <= start_idx:
        return None

    # 进行多窗口LPPL拟合
    warnings = []
    window_sizes = [150, 175, 200, 225, 250]

    for ws in window_sizes:
        fit_start = max(0, end_idx - ws)

        if fit_start < 100:
            continue

        result = fit_lppl_window(df['收盘价'], fit_start, end_idx)

        if result is not None:
            # 预测的tc是否在崩盘前？
            days_to_tc = result['tc_global'] - crash['crash_idx']

            if -30 < days_to_tc < 30:  # 预测时间在崩盘前后30天内
                warnings.append({
                    'window_size': ws,
                    'tc_relative': result['tc_relative'],
                    'tc_global': result['tc_global'],
                    'days_to_crash': days_to_tc,
                    'days_before_crash': crash['crash_idx'] - result['tc_global'],
                    'm': result['m'],
                    'omega': result['omega'],
                    'r_squared': result['r_squared']
                })

    if len(warnings) == 0:
        return None

    # 分析预警情况
    has_warning_before = any(w['days_before_crash'] > 0 for w in warnings)

    # 计算预测tc的聚类程度
    tc_predictions = [w['tc_global'] for w in warnings]
    tc_std = np.std(tc_predictions) if len(tc_predictions) > 1 else 0
    tc_mean = np.mean(tc_predictions)

    return {
        'crash': crash,
        'has_warning': has_warning_before,
        'warning_count': len(warnings),
        'warnings': warnings,
        'tc_mean': tc_mean,
        'tc_std': tc_std,
        'tc_cv': tc_std / tc_mean if tc_mean > 0 else 0
    }


def main():
    print("="*80)
    print("LPPL历史胜率验证：找出所有大跌>20%的波段，检查LPPL预警准确率")
    print("="*80)

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 天")

    # 步骤1：找出所有大跌事件
    print("\n" + "="*80)
    print("【步骤1】检测历史大跌事件（跌幅>20%）")
    print("="*80)

    crashes = detect_crash_events(df, crash_threshold=0.20, min_duration=10)

    print(f"\n找到 {len(crashes)} 个大跌事件：\n")
    print(f"{'序号':<6} {'峰值日期':<12} {'峰值价格':<10} {'崩盘日期':<12} {'崩盘价格':<10} {'跌幅':<8} {'持续天数':<8}")
    print("-"*80)

    for i, crash in enumerate(crashes[:10], 1):  # 只显示前10个
        print(f"{i:<6} {str(crash['peak_date'].date()):<12} {crash['peak_price']:<10.0f} "
              f"{str(crash['crash_date'].date()):<12} {crash['crash_price']:<10.0f} "
              f"{crash['drawdown']*100:>6.1f}%   {crash['duration_days']:<8}")

    if len(crashes) > 10:
        print(f"... 还有 {len(crashes) - 10} 个事件")

    # 步骤2：对每个大跌事件，检查LPPL是否提前预警
    print("\n" + "="*80)
    print("[步骤2] 检查LPPL在大跌前的预警表现")
    print("="*80)

    results = []

    for i, crash in enumerate(crashes):
        analysis = check_lppl_warning_before_crash(df, crash,
                                                    warning_days=60,
                                                    window_size=200)

        if analysis is None:
            results.append({
                'crash': crash,
                'warning_status': 'NO_FIT',
                'has_warning': False
            })
            print(f"\n事件{i+1}: {crash['peak_date'].date()} ~ {crash['crash_date'].date()}")
            print(f"  LPPL拟合失败，无法判断")
        else:
            results.append({
                'crash': crash,
                'warning_status': 'HAS_WARNING' if analysis['has_warning'] else 'NO_WARNING',
                'has_warning': analysis['has_warning'],
                'warning_count': analysis['warning_count'],
                'tc_std': analysis['tc_std']
            })

            print(f"\n事件{i+1}: {crash['peak_date'].date()} ~ {crash['crash_date'].date()} (跌幅{crash['drawdown']*100:.1f}%)")
            print(f"  LPPL拟合成功: {analysis['warning_count']}次")
            print(f"  有提前预警: {'是' if analysis['has_warning'] else '否'}")

            if analysis['warning_count'] > 0:
                tc_before_crash = [w['days_before_crash'] for w in analysis['warnings']]
                print(f"  预测时间距崩盘: {min(tc_before_crash):.0f} ~ {max(tc_before_crash):.0f}天")

    # 步骤3：统计LPPL预警的准确率
    print("\n" + "="*80)
    print("【步骤3】LPPL预警准确率统计")
    print("="*80)

    with_warnings = [r for r in results if r['warning_status'] == 'HAS_WARNING']
    without_warnings = [r for r in results if r['warning_status'] == 'NO_WARNING']
    fit_failures = [r for r in results if r['warning_status'] == 'NO_FIT']

    print(f"\n总大跌事件: {len(crashes)}次")
    print(f"LPPL成功拟合: {len(crashes) - len(fit_failures)}次")
    print(f"  有提前预警: {len(with_warnings)}次")
    print(f"  无提前预警: {len(without_warnings)}次")
    print(f"  拟合失败: {len(fit_failures)}次")

    if len(with_warnings) > 0:
        print(f"\nLPPL预警成功率: {len(with_warnings) / (len(crashes) - len(fit_failures)) * 100:.1f}%")
    else:
        print(f"\nLPPL从未成功预测任何大跌事件！")

    # 步骤4：详细分析有预警的事件
    if len(with_warnings) > 0:
        print("\n" + "="*80)
        print("有LPPL预警的大跌事件详细分析")
        print("="*80)

        for r in with_warnings[:5]:  # 只显示前5个
            crash = r['crash']
            print(f"\n{crash['peak_date'].date()} ~ {crash['crash_date'].date()} (跌幅{crash['drawdown']*100:.1f}%)")
            print(f"  预警次数: {r['warning_count']}次")
            print(f"  tc标准差: {r.get('tc_std', 0):.1f}")

    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 图1：价格曲线，标记所有大跌事件
    ax1 = axes[0]
    ax1.plot(df['datetime'], df['收盘价'], color='black', linewidth=1, alpha=0.6, label='沪锡价格')

    for crash in crashes:
        ax1.axvline(x=crash['peak_date'], color='orange', linestyle='--', alpha=0.3)
        ax1.axvline(x=crash['crash_date'], color='red', linestyle='-', alpha=0.5)

    ax1.set_title('历史大跌事件标注', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend()

    # 图2：预警统计
    ax2 = axes[1]

    categories = ['拟合失败', '无预警', '有预警']
    counts = [
        len(fit_failures),
        len([r for r in without_warnings if r['warning_status'] != 'NO_FIT']),
        len(with_warnings)
    ]

    colors = ['gray', 'orange', 'red']
    bars = ax2.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')

    ax2.set_ylabel('事件数量', fontsize=12)
    ax2.set_title('LPPL预警表现统计', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/lppl_historical_accuracy.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()

    # 最终结论
    print("\n" + "="*80)
    print("最终结论")
    print("="*80)

    success_rate = len(with_warnings) / (len(crashes) - len(fit_failures)) * 100 if (len(crashes) - len(fit_failures)) > 0 else 0

    print(f"\nLPPL历史预警准确率: {success_rate:.1f}%")

    if success_rate >= 50:
        print("\n[OK] LPPL预警准确率>=50%，有参考价值")
        print("  建议: 可以作为辅助预警工具")
    elif success_rate >= 30:
        print("\n[⚠️] LPPL预警准确率一般，信号仅供参考")
        print("  建议: 不要完全依赖，需结合其他指标")
    elif success_rate > 0:
        print("\n[!] LPPL预警准确率较低，需要谨慎使用")
        print("  建议: 仅作为极端情况下的风险提示")
    else:
        print("\n[❌] LPPL从未成功预测过大跌！")
        print("  建议: 不要使用LPPL作为交易依据")

    print("\n与实际交易结果的对比:")
    print(f"  理论上: LPPL预测崩盘 → 应该避开")
    print(f"  实际上: 避开LPPL风险期 → 错过87%利润")
    print(f"  原因: 沪锡10年是大牛市，'泡沫'大部分时间继续涨")

    print("\n" + "="*80)
    print("验证完成")
    print("="*80)


if __name__ == "__main__":
    main()
