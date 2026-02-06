# -*- coding: utf-8 -*-
"""
对比分析：沪锡10年纯做多策略 vs LPPL风险预警
目标：验证LPPL能否识别趋势，避免逆势交易
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_backtest_10years():
    """
    重新运行10年回测，获取交易记录
    """
    print("="*80)
    print("步骤1: 运行沪锡10年纯做多策略回测...")
    print("="*80)

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
                    'reason': exit_reason
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
                margin_per_contract = entry_price * CONTRACT_SIZE * MARGIN_RATE
                available_capital = capital * MAX_POSITION_RATIO
                max_contracts = int(available_capital / margin_per_contract)

                if max_contracts > 0:
                    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                    position = {
                        'entry_datetime': current['datetime'],
                        'entry_price': entry_price,
                        'contracts': max_contracts,
                        'stop_loss': stop_loss_price
                    }

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print(f"\n原始策略结果:")
    print(f"  初始资金: {INITIAL_CAPITAL:,.0f}元")
    print(f"  最终权益: {capital:,.0f}元")
    print(f"  收益率: {total_return:.2f}%")
    print(f"  交易次数: {len(trades)}笔")

    return df, trades, total_return


def generate_lppl_signals_for_10years(df):
    """
    为10年数据生成LPPL信号（简化版，不使用并行以避免复杂性）
    """
    print("\n" + "="*80)
    print("步骤2: 为10年数据生成LPPL风险信号...")
    print("="*80)

    from scipy.optimize import curve_fit

    def lppl_func(t, A, B, tc, m, C, omega, phi):
        dt = np.clip(tc - t, 1e-8, None)
        return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))

    def fit_lppl_window(price_series):
        t = np.arange(len(price_series), dtype=float)
        y = np.log(price_series.values.astype(float))
        last = t[-1]

        A0 = y.max()
        B0 = -0.1
        tc0 = last + 30
        m0 = 0.5
        C0 = 0.1
        omega0 = 8.0
        phi0 = 0.0

        p0 = np.array([A0, B0, tc0, m0, C0, omega0, phi0], dtype=float)

        bounds = (
            [-np.inf, -5.0, last + 1, 0.1, -1.0, 3.0, -2*np.pi],
            [ np.inf, -1e-6, last + 120, 1.0, 2.0, 20.0,  2*np.pi]
        )

        try:
            popt, _ = curve_fit(lppl_func, t, y, p0=p0, bounds=bounds, maxfev=5000)
            return popt
        except:
            return None

    def check_risk(price_series):
        for _ in range(3):  # 3次重启
            popt = fit_lppl_window(price_series)
            if popt is not None:
                A, B, tc, m, C, omega, phi = popt
                last_day = len(price_series) - 1
                days_to_crash = tc - last_day

                if (5 < days_to_crash < 90) and (0.1 < m < 1.0) and (3.0 < omega < 20.0):
                    return True, days_to_crash, m
        return False, 0, 0

    # 计算MA60
    df['MA60'] = df['close'].rolling(window=60).mean()

    # 滚动检测
    window_size = 250
    step = 10  # 10天步长，平衡速度和精度

    signals = []
    print(f"\n开始滚动扫描（窗口{window_size}天，步长{step}天）...")

    for i in range(0, len(df) - window_size, step):
        window = df.iloc[i:i + window_size]
        current_date = df.iloc[i + window_size - 1]['datetime']

        is_risk, days_left, m_val = check_risk(window['close'])

        if is_risk:
            signals.append({
                'date': current_date,
                'days_to_tc': days_left,
                'm': m_val,
                'price': df.iloc[i + window_size - 1]['close'],
                'ma60': df.iloc[i + window_size - 1]['MA60']
            })

        if (i // step) % 10 == 0:
            current_date_str = df.iloc[i + window_size - 1]['datetime'].strftime('%Y-%m-%d')
            print(f"  扫描进度: {current_date_str}...")

    print(f"\n扫描完成！共检测到 {len(signals)} 个LPPL风险信号")

    return pd.DataFrame(signals)


def analyze_trades_with_lppl(trades, lppl_signals):
    """
    分析每笔交易与LPPL信号的关系
    """
    print("\n" + "="*80)
    print("步骤3: 分析交易与LPPL信号的关系...")
    print("="*80)

    # 将LPPL信号转换为日期集合
    lppl_dates = set()
    warning_window_days = 30  # LPPL预警后30天内视为风险期

    for _, signal in lppl_signals.iterrows():
        signal_date = signal['date']
        # 预警前后各30天都视为风险期
        start_date = signal_date - timedelta(days=warning_window_days)
        end_date = signal_date + timedelta(days=int(signal['days_to_tc']))

        current = start_date
        while current <= end_date:
            lppl_dates.add(current)
            current += timedelta(days=1)

    print(f"\nLPPL风险期总天数: {len(lppl_dates)}天")

    # 分析每笔交易
    trades_analysis = []

    for trade in trades:
        entry_date = trade['entry_datetime']
        exit_date = trade['exit_datetime']

        # 检查交易期间是否与LPPL风险期重叠
        trade_dates = set()
        current = entry_date
        while current <= exit_date:
            trade_dates.add(current)
            current += timedelta(days=1)

        overlap = trade_dates & lppl_dates
        in_risk_period = len(overlap) > 0

        trades_analysis.append({
            'entry_datetime': entry_date,
            'exit_datetime': exit_date,
            'pnl': trade['pnl'],
            'pnl_pct': trade['pnl_pct'],
            'in_risk_period': in_risk_period,
            'risk_overlap_days': len(overlap)
        })

    trades_df = pd.DataFrame(trades_analysis)

    # 统计
    risk_trades = trades_df[trades_df['in_risk_period']]
    safe_trades = trades_df[~trades_df['in_risk_period']]

    print(f"\n交易分类统计:")
    print(f"  LPPL风险期交易: {len(risk_trades)}笔")
    print(f"  安全期交易: {len(safe_trades)}笔")

    # 盈亏分析
    risk_trades_profit = risk_trades[risk_trades['pnl'] > 0]
    risk_trades_loss = risk_trades[risk_trades['pnl'] <= 0]
    safe_trades_profit = safe_trades[safe_trades['pnl'] > 0]
    safe_trades_loss = safe_trades[safe_trades['pnl'] <= 0]

    print(f"\nLPPL风险期交易表现:")
    print(f"  盈利交易: {len(risk_trades_profit)}笔")
    print(f"  亏损交易: {len(risk_trades_loss)}笔")
    print(f"  胜率: {len(risk_trades_profit)/len(risk_trades)*100:.1f}%")
    print(f"  总盈亏: {risk_trades['pnl'].sum():,.0f}元")
    print(f"  平均盈亏: {risk_trades['pnl'].mean():,.0f}元")

    print(f"\n安全期交易表现:")
    print(f"  盈利交易: {len(safe_trades_profit)}笔")
    print(f"  亏损交易: {len(safe_trades_loss)}笔")
    print(f"  胜率: {len(safe_trades_profit)/len(safe_trades)*100:.1f}%")
    print(f"  总盈亏: {safe_trades['pnl'].sum():,.0f}元")
    print(f"  平均盈亏: {safe_trades['pnl'].mean():,.0f}元")

    # 模拟：过滤掉LPPL风险期交易
    filtered_pnl = safe_trades['pnl'].sum()
    original_pnl = trades_df['pnl'].sum()
    improvement = (filtered_pnl - original_pnl) / original_pnl * 100

    print(f"\n模拟结果（过滤LPPL风险期交易）:")
    print(f"  原始总盈亏: {original_pnl:,.0f}元")
    print(f"  过滤后盈亏: {filtered_pnl:,.0f}元")
    print(f"  改善幅度: {improvement:+.2f}%")

    return trades_df, {
        'risk_trades': len(risk_trades),
        'safe_trades': len(safe_trades),
        'risk_win_rate': len(risk_trades_profit)/len(risk_trades)*100 if len(risk_trades) > 0 else 0,
        'safe_win_rate': len(safe_trades_profit)/len(safe_trades)*100 if len(safe_trades) > 0 else 0,
        'risk_pnl': risk_trades['pnl'].sum(),
        'safe_pnl': safe_trades['pnl'].sum(),
        'original_pnl': original_pnl,
        'filtered_pnl': filtered_pnl,
        'improvement': improvement
    }


def plot_comparison(trades_df, lppl_signals, stats):
    """
    绘制对比图表
    """
    print("\n生成对比分析图表...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. 交易盈亏分布（按风险分类）
    risk_trades = trades_df[trades_df['in_risk_period']]
    safe_trades = trades_df[~trades_df['in_risk_period']]

    ax1 = axes[0]
    ax1.hist(safe_trades['pnl'], bins=30, alpha=0.6, color='green', label=f'安全期交易 (n={len(safe_trades)})')
    ax1.hist(risk_trades['pnl'], bins=30, alpha=0.6, color='red', label=f'LPPL风险期交易 (n={len(risk_trades)})')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('盈亏 (元)', fontsize=12)
    ax1.set_ylabel('频次', fontsize=12)
    ax1.set_title('交易盈亏分布对比', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 2. 胜率对比
    ax2 = axes[1]
    categories = ['LPPL风险期', '安全期']
    win_rates = [stats['risk_win_rate'], stats['safe_win_rate']]
    colors = ['red', 'green']

    bars = ax2.bar(categories, win_rates, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('胜率 (%)', fontsize=12)
    ax2.set_title('胜率对比', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)

    # 添加数值标签
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold')

    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 3. 累计收益对比
    ax3 = axes[2]

    # 原始策略累计收益
    original_cumsum = trades_df['pnl'].cumsum()

    # 过滤策略累计收益（假设不交易风险期）
    filtered_trades = trades_df[~trades_df['in_risk_period']]
    filtered_cumsum = filtered_trades['pnl'].cumsum()
    filtered_cumsum_full = trades_df['pnl'].copy()
    filtered_cumsum_full[trades_df['in_risk_period']] = 0
    filtered_cumsum_full = filtered_cumsum_full.cumsum()

    ax3.plot(range(len(original_cumsum)), original_cumsum,
             label='原始策略', color='blue', linewidth=2)
    ax3.plot(range(len(filtered_cumsum_full)), filtered_cumsum_full,
             label='过滤LPPL风险期', color='green', linewidth=2, linestyle='--')

    ax3.set_xlabel('交易序号', fontsize=12)
    ax3.set_ylabel('累计盈亏 (元)', fontsize=12)
    ax3.set_title('累计收益曲线对比', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/lppl_vs_trades_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"图表已保存: {output_file}")
    plt.show()


def main():
    print("\n" + "="*80)
    print("沪锡10年: 纯做多策略 vs LPPL风险预警对比分析")
    print("="*80)

    # 切换工作目录
    import os
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

    # 1. 运行10年回测
    df, trades, original_return = load_and_backtest_10years()

    # 2. 生成LPPL信号
    lppl_signals = generate_lppl_signals_for_10years(df)

    # 3. 分析交易与LPPL信号的关系
    trades_df, stats = analyze_trades_with_lppl(trades, lppl_signals)

    # 4. 绘制对比图表
    plot_comparison(trades_df, lppl_signals, stats)

    # 5. 总结报告
    print("\n" + "="*80)
    print("总结报告")
    print("="*80)

    print(f"\n{'指标':<30} {'原始策略':<15} {'过滤LPPL风险期':<15} {'改善':<10}")
    print("-"*80)
    print(f"{'交易次数':<30} {len(trades):<15} {stats['safe_trades']:<15} {-stats['risk_trades']}笔")
    risk_label = 'LPPL风险期交易'
    safe_label = '安全期交易'
    print(f"{risk_label:<30} {stats['risk_trades']:<15} {'0':<15} {'-':<10}")
    print(f"{safe_label:<30} {stats['safe_trades']:<15} {stats['safe_trades']:<15} {'-':<10}")
    label3 = 'LPPL风险期胜率'
    label4 = '安全期胜率'
    label5 = '总盈亏'
    na_label = 'N/A'
    dash_label = '-'
    yuan_label = '元'
    print(f"{label3:<30} {stats['risk_win_rate']:<15.1f}% {na_label:<15} {dash_label:<10}")
    print(f"{label4:<30} {stats['safe_win_rate']:<15.1f}% {stats['safe_win_rate']:<15.1f}% {dash_label:<10}")
    print(f"{label5:<30} {stats['original_pnl']:>13,.0f}{yuan_label} {stats['filtered_pnl']:>13,.0f}{yuan_label} {stats['improvement']:>+6.2f}%")

    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if stats['improvement'] > 0:
        print(f"\n[OK] LPPL过滤有效！")
        print(f"  过滤LPPL风险期交易后，收益提升 {stats['improvement']:.2f}%")
        print(f"  建议在实际交易中使用LPPL作为风险预警工具")
    elif stats['risk_win_rate'] < stats['safe_win_rate']:
        print(f"\n[OK] LPPL能识别风险期！")
        print(f"  LPPL风险期胜率({stats['risk_win_rate']:.1f}%) < 安全期胜率({stats['safe_win_rate']:.1f}%)")
        print(f"  但总体收益下降，说明部分风险期交易是大盈利")
        print(f"  建议：LPPL作为辅助指标，不完全过滤，但降低风险期仓位")
    else:
        print(f"\n[!] LPPL预警效果不显著")
        print(f"  需要调整LPPL参数或策略")

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
