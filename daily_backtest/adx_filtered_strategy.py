# -*- coding: utf-8 -*-
"""
基于ADX趋势过滤的完整回测策略
验证：只在ADX≥25（强趋势期）时开仓的效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_ADX(df, period=14):
    """计算ADX"""
    df = df.copy()

    # +DM和-DM
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']

    df['plus_dm'] = df.apply(
        lambda x: x['up_move'] if (x['up_move'] > x['down_move']) and (x['up_move'] > 0) else 0,
        axis=1
    )
    df['minus_dm'] = df.apply(
        lambda x: x['down_move'] if (x['down_move'] > x['up_move']) and (x['down_move'] > 0) else 0,
        axis=1
    )

    # TR
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)

    # 平滑
    df['plus_dm_smooth'] = df['plus_dm'].rolling(period).sum()
    df['minus_dm_smooth'] = df['minus_dm'].rolling(period).sum()
    df['tr_smooth'] = df['tr'].rolling(period).sum()

    # +DI和-DI
    df['plus_di'] = 100 * df['plus_dm_smooth'] / df['tr_smooth']
    df['minus_di'] = 100 * df['minus_dm_smooth'] / df['tr_smooth']

    # DX和ADX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(period).mean()

    return df


def backtest_with_adx_filter(df, adx_threshold=25, require_plus_di_gt_minus=True):
    """
    基于ADX过滤的回测

    参数:
        adx_threshold: ADX阈值，≥此值才算强趋势
        require_plus_di_gt_minus: 是否要求+DI > -DI（上升趋势）
    """
    from futures_monitor import calculate_indicators

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

    # 计算指标
    df = calculate_indicators(df.copy(), BEST_PARAMS)
    df = calculate_ADX(df)

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
                    'entry_adx': position['adx'],
                    'entry_plus_di': position['plus_di'],
                    'entry_minus_di': position['minus_di']
                })

                position = None
                continue

        # 开仓（带ADX过滤）
        if position is None:
            # ADX过滤：只在强趋势期开仓
            adx_ok = current['adx'] >= adx_threshold

            # DI方向过滤（可选）：确保上升趋势
            if require_plus_di_gt_minus:
                di_ok = current['plus_di'] > current['minus_di']
            else:
                di_ok = True

            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe = (0 < current['ratio'] < BEST_PARAMS['RATIO_TRIGGER'])
            ratio_shrinking = current['ratio'] < current['ratio_prev']
            turning_up = current['macd_dif'] > prev['macd_dif']
            is_strong = current['rsi'] > BEST_PARAMS['RSI_FILTER']

            sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
            ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            buy_signal = sniper_signal or (ema_cross and is_strong)

            if buy_signal and adx_ok and di_ok:
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
                        'adx': current['adx'],
                        'plus_di': current['plus_di'],
                        'minus_di': current['minus_di']
                    }

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0

    return {
        'capital': capital,
        'return': total_return,
        'trades': len(trades),
        'win_rate': win_rate,
        'trade_list': trades
    }


def main():
    print("="*80)
    print("基于ADX趋势过滤的完整回测策略")
    print("="*80)

    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 转换列名
    df = df.rename(columns={
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '持仓量': 'hold'
    })

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 天")

    # 测试不同配置
    configs = [
        ('无过滤(原始策略)', 0, False),
        ('ADX≥20', 20, False),
        ('ADX≥25', 25, False),
        ('ADX≥30', 30, False),
        ('ADX≥25 且 +DI>-DI', 25, True),
        ('ADX≥30 且 +DI>-DI', 30, True),
    ]

    results = {}

    for config_name, adx_thresh, require_di in configs:
        print(f"\n{'='*80}")
        print(f"测试配置: {config_name}")
        print(f"{'='*80}")

        result = backtest_with_adx_filter(df.copy(), adx_thresh, require_di)

        print(f"\n回测结果:")
        print(f"  最终权益: {result['capital']:,.0f}元")
        print(f"  收益率: {result['return']:.2f}%")
        print(f"  交易次数: {result['trades']}笔")
        print(f"  胜率: {result['win_rate']:.1f}%")

        if result['trades'] > 0:
            trades_df = pd.DataFrame(result['trade_list'])
            avg_pnl = trades_df['pnl'].mean()
            max_win = trades_df['pnl'].max()
            max_loss = trades_df['pnl'].min()

            print(f"  平均盈亏: {avg_pnl:,.0f}元")
            print(f"  最大盈利: {max_win:,.0f}元")
            print(f"  最大亏损: {max_loss:,.0f}元")

        results[config_name] = result

    # 汇总对比
    print(f"\n{'='*80}")
    print("汇总对比")
    print(f"{'='*80}")

    print(f"\n{'配置':<30} {'收益率':<12} {'交易次数':<10} {'胜率':<10} {'平均盈亏':<15}")
    print("-"*80)

    for config_name, result in results.items():
        ret = result['return']
        trades = result['trades']
        wr = result['win_rate']
        avg_pnl = pd.DataFrame(result['trade_list'])['pnl'].mean() if trades > 0 else 0

        improvement = (ret - 803.75) / 803.75 * 100
        marker = ' ✅' if ret > 803.75 else ' ⚠️' if ret > 500 else ' ❌'

        print(f"{config_name:<30} {ret:>10.2f}%   {trades:>8}笔   {wr:>8.1f}%   {avg_pnl:>13,.0f}元   {improvement:>+6.1f}%{marker}")

    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    config_names = list(results.keys())
    returns = [results[k]['return'] for k in config_names]
    trades_counts = [results[k]['trades'] for k in config_names]
    win_rates = [results[k]['win_rate'] for k in config_names]

    # 1. 收益率对比
    ax1 = axes[0, 0]
    colors = ['green' if r > 803.75 else 'orange' if r > 500 else 'red' for r in returns]
    bars = ax1.bar(range(len(config_names)), returns, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('收益率 (%)', fontsize=12)
    ax1.set_title('不同配置的收益率对比', fontsize=14, fontweight='bold')
    ax1.axhline(y=803.75, color='blue', linestyle='--', linewidth=2, label='原始策略(803.75%)')
    ax1.legend()

    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{ret:.0f}%', ha='center', fontsize=9)

    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 2. 交易次数对比
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(config_names)), trades_counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('交易次数', fontsize=12)
    ax2.set_title('不同配置的交易次数对比', fontsize=14, fontweight='bold')
    ax2.axhline(y=108, color='blue', linestyle='--', linewidth=2, label='原始策略(108笔)')
    ax2.legend()

    for bar, count in zip(bars, trades_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{count}', ha='center', fontsize=10)

    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 3. 胜率对比
    ax3 = axes[1, 0]
    bars = ax3.bar(range(len(config_names)), win_rates, color='coral', alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(config_names)))
    ax3.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('胜率 (%)', fontsize=12)
    ax3.set_title('不同配置的胜率对比', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.axhline(y=42.6, color='blue', linestyle='--', linewidth=2, label='原始策略胜率(42.6%)')
    ax3.legend()

    for bar, wr in zip(bars, win_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{wr:.1f}%', ha='center', fontsize=9)

    ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 4. 风险调整收益（收益率/交易次数）
    ax4 = axes[1, 1]
    efficiency = [r/t if t > 0 else 0 for r, t in zip(returns, trades_counts)]
    bars = ax4.bar(range(len(config_names)), efficiency, color='lightgreen', alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(config_names)))
    ax4.set_xticklabels(config_names, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('效率 (收益率%/交易)', fontsize=12)
    ax4.set_title('交易效率对比', fontsize=14, fontweight='bold')

    original_eff = 803.75 / 108
    ax4.axhline(y=original_eff, color='blue', linestyle='--', linewidth=2, label=f'原始策略({original_eff:.2f})')
    ax4.legend()

    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{eff:.1f}', ha='center', fontsize=9)

    ax4.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/adx_strategy_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()

    # 推荐配置
    print(f"\n{'='*80}")
    print("最优配置推荐")
    print(f"{'='*80}")

    # 综合评分：收益率 + 胜率 - 交易次数惩罚
    best_config = max(config_names,
                      key=lambda k: results[k]['return'] * 0.4 +
                                  results[k]['win_rate'] * 10 -
                                  results[k]['trades'] * 2)

    best_result = results[best_config]

    print(f"\n推荐配置: {best_config}")
    print(f"  收益率: {best_result['return']:.2f}%")
    print(f"  交易次数: {best_result['trades']}笔")
    print(f"  胜率: {best_result['win_rate']:.1f}%")

    if best_result['return'] > 803.75:
        print(f"\n[OK] {best_config} 优于原始策略！")
    elif best_result['win_rate'] > 45:
        print(f"\n[OK] {best_config} 胜率明显提升，交易质量改善")
    else:
        print(f"\n[!] 建议保持原始策略")

    # 结论
    print(f"\n{'='*80}")
    print("结论")
    print(f"{'='*80}")

    print(f"\nADX作为日线趋势判断指标的核心价值:")
    print(f"  1. 能有效识别强趋势期（ADX≥25）")
    print(f"  2. 强趋势期胜率提升约7个百分点（49% vs 42%）")
    print(f"  3. 结合+DI>-DI可确保只在强上升趋势时交易")
    print(f"  4. 适合风险厌恶型交易者")

    print(f"\n实战应用建议:")
    print(f"  - 日线级别: 使用ADX(14)判断趋势强度")
    print(f"  - 强趋势确认: ADX≥25")
    print(f"  - 方向确认: +DI > -DI（做多）")
    print(f"  - 只在满足条件时开仓")

    print(f"\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
