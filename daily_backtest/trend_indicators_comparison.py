# -*- coding: utf-8 -*-
"""
日线级别趋势判断指标对比测试
目标：找到能真正识别趋势、避免逆势交易的指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_ADX(df, period=14):
    """
    计算ADX（平均趋向指数）- 趋势强度指标
    ADX > 25: 强趋势
    ADX < 20: 弱趋势/震荡
    """
    df = df.copy()

    # 计算+DM和-DM
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = -df['low'].diff()

    df['plus_dm'] = df.apply(
        lambda x: x['high_diff'] if (x['high_diff'] > x['low_diff']) and (x['high_diff'] > 0) else 0,
        axis=1
    )
    df['minus_dm'] = df.apply(
        lambda x: x['low_diff'] if (x['low_diff'] > x['high_diff']) and (x['low_diff'] > 0) else 0,
        axis=1
    )

    # 计算TR（真实波幅）
    df['tr'] = df.apply(
        lambda x: max(x['high'] - x['low'],
                     abs(x['high'] - x['close_prev']),
                     abs(x['low'] - x['close_prev'])) if 'close_prev' in x else x['high'] - x['low'],
        axis=1
    )

    # 平滑+DM、-DM和TR
    df['plus_dm_smooth'] = df['plus_dm'].rolling(window=period).sum()
    df['minus_dm_smooth'] = df['minus_dm'].rolling(window=period).sum()
    df['tr_smooth'] = df['tr'].rolling(window=period).sum()

    # 计算+DI和-DI
    df['plus_di'] = 100 * df['plus_dm_smooth'] / df['tr_smooth']
    df['minus_di'] = 100 * df['minus_dm_smooth'] / df['tr_smooth']

    # 计算DX和ADX
    df['di_diff'] = abs(df['plus_di'] - df['minus_di'])
    df['di_sum'] = df['plus_di'] + df['minus_di']
    df['dx'] = 100 * df['di_diff'] / df['di_sum']
    df['adx'] = df['dx'].rolling(window=period).mean()

    return df


def calculate_trend_strength_score(df):
    """
    综合趋势强度评分（0-100）
    结合多个维度判断趋势强度
    """
    df = df.copy()

    # 1. 价格与均线距离
    df['price_vs_ma60'] = (df['close'] - df['MA60']) / df['MA60'] * 100

    # 2. EMA斜率
    df['ema_slope'] = df['EMA_FAST'].diff(5)

    # 3. 波动率（ATR）
    df['atr'] = df.apply(lambda x: x['high'] - x['low'], axis=1)
    df['atr_ma'] = df['atr'].rolling(window=20).mean()
    df['volatility_ratio'] = df['atr'] / df['atr_ma']

    # 4. 趋势一致性（多个周期EMA同向）
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema60'] = df['close'].ewm(span=60).mean()
    df['ema120'] = df['close'].ewm(span=120).mean()

    df['trend_consistency'] = df.apply(
        lambda x: 1 if (x['close'] > x['ema20'] > x['ema60'] > x['ema120']) else
                 (-1 if (x['close'] < x['ema20'] < x['ema60'] < x['ema120']) else 0),
        axis=1
    )

    # 综合评分（0-100）
    df['trend_score'] = (
        df['price_vs_ma60'].clip(-5, 5) * 10 +  # 价格位置 ±50分
        df['adx'].clip(0, 50) * 0.5 +            # ADX强度 0-25分
        df['trend_consistency'].abs() * 25       # 趋势一致性 0-25分
    ).clip(0, 100)

    return df


def detect_market_regime_adx(df, adx_threshold=25):
    """
    基于ADX的市场状态识别
    """
    df['market_regime'] = 'WEAK'

    strong_trend = df['adx'] >= adx_threshold

    df.loc[strong_trend & (df['plus_di'] > df['minus_di']), 'market_regime'] = 'STRONG_UP'
    df.loc[strong_trend & (df['minus_di'] > df['plus_di']), 'market_regime'] = 'STRONG_DOWN'

    return df


def detect_market_regime_composite(df, trend_score_threshold=60):
    """
    基于综合评分的市场状态识别
    """
    df['regime_composite'] = 'WEAK'

    df.loc[df['trend_score'] >= trend_score_threshold, 'regime_composite'] = 'STRONG_UP'
    df.loc[df['trend_score'] <= (100 - trend_score_threshold), 'regime_composite'] = 'STRONG_DOWN'

    return df


def run_backtest_with_regime(df, regime_method='adx', threshold=25):
    """
    使用指定方法识别市场状态，并只在强趋势期交易
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

    # 转换列名
    df = df.rename(columns={
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '持仓量': 'hold'
    })

    # 计算基础指标
    df = calculate_indicators(df.copy(), BEST_PARAMS)
    df['close_prev'] = df['close'].shift(1)

    # 计算MA60
    df['MA60'] = df['close'].rolling(window=60).mean()
    df['EMA_FAST'] = df['close'].ewm(span=BEST_PARAMS['EMA_FAST']).mean()
    df['EMA_SLOW'] = df['close'].ewm(span=BEST_PARAMS['EMA_SLOW']).mean()

    # 计算趋势指标
    df = calculate_ADX(df)
    df = calculate_trend_strength_score(df)

    # 识别市场状态
    if regime_method == 'adx':
        df = detect_market_regime_adx(df, threshold)
    elif regime_method == 'composite':
        df = detect_market_regime_composite(df, threshold)
    else:
        raise ValueError(f"Unknown method: {regime_method}")

    # 回测（只做强势上涨）
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        regime_col = 'market_regime' if regime_method == 'adx' else 'regime_composite'

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
                    'entry_regime': position['regime'],
                    'entry_adx': position['adx'],
                    'entry_trend_score': position.get('trend_score', 0)
                })

                position = None
                continue

        # 开仓（只在强上升趋势）
        if position is None:
            regime = current[regime_col]

            # 只在强上升趋势时开仓
            regime_ok = regime == 'STRONG_UP'

            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe = (0 < current['ratio'] < BEST_PARAMS['RATIO_TRIGGER'])
            ratio_shrinking = current['ratio'] < current['ratio_prev']
            turning_up = current['macd_dif'] > prev['macd_dif']
            is_strong = current['rsi'] > BEST_PARAMS['RSI_FILTER']

            sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
            ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            buy_signal = sniper_signal or (ema_cross and is_strong)

            if buy_signal and regime_ok:
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
                        'regime': regime,
                        'adx': current['adx'],
                        'trend_score': current.get('trend_score', 0)
                    }

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return df, trades, capital, total_return


def analyze_trades_by_regime(df, trades):
    """分析不同市场状态下的交易表现"""
    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        return {}, trades_df

    # 按ADX分组
    trades_df['adx_group'] = pd.cut(trades_df['entry_adx'],
                                     bins=[0, 20, 25, 30, 100],
                                     labels=['弱趋势(<20)', '震荡(20-25)', '强趋势(25-30)', '超强趋势(>30)'])

    # 按趋势评分分组
    trades_df['score_group'] = pd.cut(trades_df['entry_trend_score'],
                                       bins=[0, 40, 60, 80, 100],
                                       labels=['弱(0-40)', '中(40-60)', '强(60-80)', '超强(80-100)'])

    stats_by_adx = {}
    for name, group in trades_df.groupby('adx_group'):
        if len(group) == 0:
            continue
        wins = len(group[group['pnl'] > 0])
        stats_by_adx[str(name)] = {
            'count': len(group),
            'wins': wins,
            'losses': len(group) - wins,
            'win_rate': wins / len(group) * 100,
            'total_pnl': group['pnl'].sum(),
            'avg_pnl': group['pnl'].mean(),
            'avg_adx': group['entry_adx'].mean()
        }

    stats_by_score = {}
    for name, group in trades_df.groupby('score_group'):
        if len(group) == 0:
            continue
        wins = len(group[group['pnl'] > 0])
        stats_by_score[str(name)] = {
            'count': len(group),
            'wins': wins,
            'losses': len(group) - wins,
            'win_rate': wins / len(group) * 100,
            'total_pnl': group['pnl'].sum(),
            'avg_pnl': group['pnl'].mean(),
            'avg_score': group['entry_trend_score'].mean()
        }

    return stats_by_adx, stats_by_score, trades_df


def plot_comparison(results_dict, trades_stats_dict):
    """绘制不同方法的对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 收益率对比
    ax1 = axes[0, 0]
    methods = list(results_dict.keys())
    returns = [results_dict[m]['return'] for m in methods]
    colors = ['green' if r > 0 else 'red' for r in returns]

    bars = ax1.bar(methods, returns, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('收益率 (%)', fontsize=12)
    ax1.set_title('不同方法的收益率对比', fontsize=14, fontweight='bold')
    ax1.axhline(y=803.75, color='blue', linestyle='--', linewidth=2, label='原始策略(803.75%)')
    ax1.legend()

    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{ret:.1f}%', ha='center', fontsize=11, fontweight='bold')

    ax1.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 2. 交易次数对比
    ax2 = axes[0, 1]
    trades_counts = [results_dict[m]['trades'] for m in methods]

    bars = ax2.bar(methods, trades_counts, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('交易次数', fontsize=12)
    ax2.set_title('不同方法的交易次数对比', fontsize=14, fontweight='bold')
    ax2.axhline(y=108, color='blue', linestyle='--', linewidth=2, label='原始策略(108笔)')
    ax2.legend()

    for bar, count in zip(bars, trades_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{count}', ha='center', fontsize=11, fontweight='bold')

    ax2.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 3. ADX分组胜率
    ax3 = axes[1, 0]
    if 'ADX(25)' in trades_stats_dict:
        adx_stats, _, _ = trades_stats_dict['ADX(25)']
        if adx_stats:
            categories = list(adx_stats.keys())
            win_rates = [adx_stats[k]['win_rate'] for k in categories]

            bars = ax3.bar(range(len(categories)), win_rates, color='coral', alpha=0.7, edgecolor='black')
            ax3.set_xticks(range(len(categories)))
            ax3.set_xticklabels(categories, rotation=15, ha='right')
            ax3.set_ylabel('胜率 (%)', fontsize=12)
            ax3.set_title('不同ADX水平的胜率', fontsize=14, fontweight='bold')
            ax3.set_ylim(0, 100)
            ax3.axhline(y=42.6, color='blue', linestyle='--', linewidth=2, label='原始策略胜率(42.6%)')
            ax3.legend()

            for bar, wr in zip(bars, win_rates):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{wr:.1f}%', ha='center', fontsize=10)

            ax3.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 4. 综合评分分组胜率
    ax4 = axes[1, 1]
    if '综合评分(60)' in trades_stats_dict:
        _, score_stats, _ = trades_stats_dict['综合评分(60)']
        if score_stats:
            categories = list(score_stats.keys())
            win_rates = [score_stats[k]['win_rate'] for k in categories]

            bars = ax4.bar(range(len(categories)), win_rates, color='lightgreen', alpha=0.7, edgecolor='black')
            ax4.set_xticks(range(len(categories)))
            ax4.set_xticklabels(categories, rotation=15, ha='right')
            ax4.set_ylabel('胜率 (%)', fontsize=12)
            ax4.set_title('不同趋势评分的胜率', fontsize=14, fontweight='bold')
            ax4.set_ylim(0, 100)
            ax4.axhline(y=42.6, color='blue', linestyle='--', linewidth=2, label='原始策略胜率(42.6%)')
            ax4.legend()

            for bar, wr in zip(bars, win_rates):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{wr:.1f}%', ha='center', fontsize=10)

            ax4.grid(True, linestyle='--', alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = 'D:/期货数据/铜期货监控/daily_backtest/trend_indicators_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {output_file}")
    plt.show()


def main():
    print("="*80)
    print("日线级别趋势判断指标对比测试")
    print("="*80)

    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 天")

    # 测试不同方法和阈值
    test_configs = [
        ('ADX(20)', 'adx', 20),
        ('ADX(25)', 'adx', 25),
        ('ADX(30)', 'adx', 30),
        ('综合评分(50)', 'composite', 50),
        ('综合评分(60)', 'composite', 60),
        ('综合评分(70)', 'composite', 70),
    ]

    results = {}
    trades_stats = {}

    for method_name, regime_method, threshold in test_configs:
        print(f"\n{'='*80}")
        print(f"测试: {method_name}")
        print(f"{'='*80}")

        df_result, trades, final_capital, total_return = run_backtest_with_regime(
            df.copy(), regime_method, threshold
        )

        print(f"\n回测结果:")
        print(f"  收益率: {total_return:.2f}%")
        print(f"  交易次数: {len(trades)}笔")

        if len(trades) > 0:
            adx_stats, score_stats, trades_df = analyze_trades_by_regime(df_result, trades)

            print(f"\n按ADX分组统计:")
            print(f"{'ADX组':<20} {'交易数':<8} {'胜率':<10}")
            for k, v in adx_stats.items():
                print(f"{k:<20} {v['count']:<8} {v['win_rate']:<10.1f}%")

            results[method_name] = {
                'return': total_return,
                'trades': len(trades)
            }

            if regime_method == 'adx':
                trades_stats[method_name] = (adx_stats, score_stats, trades_df)
            else:
                trades_stats[method_name] = (adx_stats, score_stats, trades_df)
        else:
            results[method_name] = {
                'return': total_return,
                'trades': 0
            }

    # 汇总对比
    print(f"\n{'='*80}")
    print("汇总对比")
    print(f"{'='*80}")

    print(f"\n{'方法':<20} {'收益率':<12} {'交易次数':<10} {'vs原始':<15}")
    print("-"*80)

    for method_name in results.keys():
        ret = results[method_name]['return']
        trades = results[method_name]['trades']
        improvement = (ret - 803.75) / 803.75 * 100

        print(f"{method_name:<20} {ret:>10.2f}%   {trades:>8}笔   {improvement:>+10.2f}%")

    # 绘图
    print("\n生成对比图表...")
    plot_comparison(results, trades_stats)

    # 找出最优方法
    print(f"\n{'='*80}")
    print("最优方法推荐")
    print(f"{'='*80}")

    best_method = max(results.keys(), key=lambda k: results[k]['return'])
    best_result = results[best_method]

    print(f"\n最优方法: {best_method}")
    print(f"  收益率: {best_result['return']:.2f}%")
    print(f"  交易次数: {best_result['trades']}笔")

    if best_result['return'] > 803.75:
        improvement = (best_result['return'] - 803.75) / 803.75 * 100
        print(f"\n[OK] {best_method} 有效！")
        print(f"  比原始策略提升 {improvement:.2f}%")
        print(f"  建议：使用此指标作为趋势判断工具")
    else:
        print(f"\n[!] 所有方法都未能超越原始策略")
        print(f"  原因：沪锡10年是大牛市，任何过滤都会错过部分利润")

    # 结论
    print(f"\n{'='*80}")
    print("结论")
    print(f"{'='*80}")

    print(f"\n趋势判断的核心原则:")
    print(f"  1. 日线级别的趋势应该用MA/EMA判断")
    print(f"  2. ADX可以判断趋势强度（>25为强趋势）")
    print(f"  3. 在牛市中，过度过滤会降低收益")
    print(f"  4. 最佳策略：趋势判断用MA，风险控制用止损")

    print(f"\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()
