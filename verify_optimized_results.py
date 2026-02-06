# -*- coding: utf-8 -*-
"""
严格验证优化后的回测结果
每个品种运行3次，确保结果一致
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

# 固定参数
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50
STOP_LOSS_PCT = 0.02
INITIAL_CAPITAL = 100000
MARGIN_RATIO = 0.15
CONTRACT_SIZE = 5

# 最优参数配置
OPTIMIZED_PARAMS = {
    '白糖': {
        'EMA_FAST': 5,
        'EMA_SLOW': 15,
        'RSI_FILTER': 45,
        'RATIO_TRIGGER': 1.15,
        'STC_SELL_ZONE': 85,
        'note': '保持铜参数（优化后反而变差）'
    },
    '沪锌': {
        'EMA_FAST': 3,
        'EMA_SLOW': 10,
        'RSI_FILTER': 40,
        'RATIO_TRIGGER': 1.10,
        'STC_SELL_ZONE': 80,
        'note': '优化参数'
    },
    '沪铜': {
        'EMA_FAST': 5,
        'EMA_SLOW': 10,
        'RSI_FILTER': 40,
        'RATIO_TRIGGER': 1.20,
        'STC_SELL_ZONE': 80,
        'note': '优化参数'
    },
    '沪铝': {
        'EMA_FAST': 3,
        'EMA_SLOW': 10,
        'RSI_FILTER': 50,
        'RATIO_TRIGGER': 1.10,
        'STC_SELL_ZONE': 80,
        'note': '优化参数'
    },
    '豆油': {
        'EMA_FAST': 7,
        'EMA_SLOW': 15,
        'RSI_FILTER': 40,
        'RATIO_TRIGGER': 1.20,
        'STC_SELL_ZONE': 80,
        'note': '优化参数'
    }
}

# 原始收益（用铜参数）
ORIGINAL_RETURNS = {
    '白糖': 188.15,
    '沪锌': 79.95,
    '沪铜': 56.36,
    '沪铝': 28.77,
    '豆油': 22.37
}

def calculate_indicators(df, params):
    """计算技术指标"""
    # EMA
    df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

    # MACD & Ratio
    exp1 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # STC
    stc_macd = df['close'].ewm(span=STC_FAST, adjust=False).mean() - \
               df['close'].ewm(span=STC_SLOW, adjust=False).mean()
    stoch_period = STC_LENGTH
    min_macd = stc_macd.rolling(window=stoch_period).min()
    max_macd = stc_macd.rolling(window=stoch_period).max()
    stoch_k = 100 * (stc_macd - min_macd) / (max_macd - min_macd).replace(0, np.nan)
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=3).mean()
    min_stoch_d = stoch_d.rolling(window=stoch_period).min()
    max_stoch_d = stoch_d.rolling(window=stoch_period).max()
    stc_raw = 100 * (stoch_d - min_stoch_d) / (max_stoch_d - min_stoch_d).replace(0, np.nan)
    stc_raw = stc_raw.fillna(50)
    df['stc'] = stc_raw.rolling(window=3).mean()

    df['ratio_prev'] = df['ratio'].shift(1)
    df['stc_prev'] = df['stc'].shift(1)

    return df

def run_backtest_detailed(df, params, future_name):
    """运行详细回测"""
    df = calculate_indicators(df.copy(), params)

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 入场条件
        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > params['RSI_FILTER']
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong
        buy_signal = sniper_signal or chase_signal

        # 仓位计算
        if df['ratio_prev'].iloc[i] > 0:
            if df['ratio_prev'].iloc[i] > 2.0:
                position_size = 2.0
            elif df['ratio_prev'].iloc[i] > 1.5:
                position_size = 1.5
            elif df['ratio_prev'].iloc[i] > 1.0:
                position_size = 1.2
            else:
                position_size = 1.0
        else:
            position_size = 1.0

        stop_loss = current['close'] * (1 - STOP_LOSS_PCT)

        # 买入
        if buy_signal and position is None:
            entry_price = current['close']
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATIO
            available_margin = capital * position_size
            contracts = int(available_margin / margin_per_contract)

            if contracts <= 0:
                continue

            position = {
                'entry_datetime': current['datetime'],
                'entry_price': entry_price,
                'contracts': contracts,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'entry_index': i,
                'signal_type': 'sniper' if sniper_signal else 'chase'
            }

        # 卖出
        elif position is not None:
            for j in range(position['entry_index'] + 1, len(df)):
                bar = df.iloc[j]

                exit_triggered = False
                exit_price = None
                exit_reason = None

                if bar['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                    exit_triggered = True
                elif (df['stc_prev'].iloc[j] > params['STC_SELL_ZONE'] and
                      bar['stc'] < df['stc_prev'].iloc[j]):
                    exit_price = bar['close']
                    exit_reason = 'stc'
                    exit_triggered = True
                elif bar['ema_fast'] < bar['ema_slow']:
                    exit_price = bar['close']
                    exit_reason = 'trend'
                    exit_triggered = True

                if exit_triggered:
                    pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                    capital += pnl

                    trades.append({
                        'entry_datetime': position['entry_datetime'],
                        'exit_datetime': bar['datetime'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'contracts': position['contracts'],
                        'position_size': position['position_size'],
                        'pnl': pnl,
                        'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                        'holding_bars': j - position['entry_index'],
                        'signal_type': position['signal_type'],
                        'exit_reason': exit_reason
                    })

                    position = None
                    break

    if not trades:
        return None, INITIAL_CAPITAL

    trades_df = pd.DataFrame(trades)

    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])

    total_pnl = trades_df['pnl'].sum()
    return_pct = total_pnl / INITIAL_CAPITAL * 100
    win_rate = winning_trades / total_trades * 100

    sniper_trades = trades_df[trades_df['signal_type'] == 'sniper']
    chase_trades = trades_df[trades_df['signal_type'] == 'chase']

    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    stats = {
        'future_name': future_name,
        'total_trades': total_trades,
        'sniper_trades': len(sniper_trades),
        'chase_trades': len(chase_trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'return_pct': return_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_capital': capital,
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
        'trades_df': trades_df
    }

    return stats, capital

def verify_future(csv_file, future_name, params):
    """验证单个品种 - 运行3次"""
    print(f"\n{'='*80}")
    print(f"验证品种: {future_name}")
    print(f"参数: EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']}, "
          f"RATIO={params['RATIO_TRIGGER']:.2f}, STC={params['STC_SELL_ZONE']}")
    print(f"备注: {params['note']}")
    print(f"{'='*80}")

    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"\n数据量: {len(df)} 条")
    print(f"时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 运行3次
    results = []
    for run in range(1, 4):
        print(f"\n--- 第{run}次运行 ---")
        stats, _ = run_backtest_detailed(df.copy(), params, future_name)

        if stats is None:
            print(f"无交易记录")
            return None

        print(f"收益率: {stats['return_pct']:+.2f}%")
        print(f"交易次数: {stats['total_trades']}")
        print(f"胜率: {stats['win_rate']:.1f}%")
        print(f"最终资金: {stats['final_capital']:,.0f}")

        results.append(stats)
        time.sleep(0.1)  # 避免时间戳重复

    # 验证一致性
    print(f"\n{'='*80}")
    print(f"一致性验证")
    print(f"{'='*80}")

    returns = [r['return_pct'] for r in results]
    trades_counts = [r['total_trades'] for r in results]

    all_same = len(set(returns)) == 1 and len(set(trades_counts)) == 1

    if all_same:
        print(f"[通过] 3次运行结果完全一致")
        print(f"  收益率: {returns[0]:+.2f}%")
        print(f"  交易次数: {trades_counts[0]}")
    else:
        print(f"[失败] 3次运行结果不一致！")
        print(f"  收益率: {returns}")
        print(f"  交易次数: {trades_counts}")
        return None

    # 使用第1次结果
    stats = results[0]
    original_return = ORIGINAL_RETURNS.get(future_name, 0)
    improvement = stats['return_pct'] - original_return
    improvement_pct = (improvement / original_return * 100) if original_return != 0 else 0

    print(f"\n{'='*80}")
    print(f"最终结果")
    print(f"{'='*80}")

    print(f"\n【对比原始参数】")
    print(f"  原始收益: {original_return:+.2f}%")
    print(f"  优化后收益: {stats['return_pct']:+.2f}%")
    print(f"  提升: {improvement:+.2f}% ({improvement_pct:+.1f}%)")

    print(f"\n【交易统计】")
    print(f"  总交易次数: {stats['total_trades']}")
    print(f"  狙击信号: {stats['sniper_trades']}")
    print(f"  追击信号: {stats['chase_trades']}")
    print(f"  盈利次数: {stats['winning_trades']}")
    print(f"  亏损次数: {stats['losing_trades']}")
    print(f"  胜率: {stats['win_rate']:.1f}%")

    print(f"\n【盈亏分析】")
    print(f"  平均盈利: {stats['avg_win']:>+,.2f}")
    print(f"  平均亏损: {stats['avg_loss']:>+,.2f}")
    print(f"  盈亏比: {stats['profit_factor']:.2f}")

    print(f"\n【退出原因统计】")
    exit_reasons = stats['trades_df']['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/stats['total_trades']*100:.1f}%)")

    # 保存详细交易记录
    output_file = f'trades_{future_name}_optimized.csv'
    stats['trades_df'].to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细交易记录已保存: {output_file}")

    return stats

def main():
    print("=" * 80)
    print("严格验证优化后回测结果")
    print("每个品种运行3次，确保结果一致")
    print("=" * 80)

    futures_config = [
        ('白糖_4hour.csv', '白糖'),
        ('沪锌_4hour.csv', '沪锌'),
        ('沪铜_4hour.csv', '沪铜'),
        ('沪铝_4hour.csv', '沪铝'),
        ('豆油_4hour.csv', '豆油')
    ]

    data_dir = Path('futures_data_4h')
    summary = []

    for csv_file, future_name in futures_config:
        csv_path = data_dir / csv_file
        if not csv_path.exists():
            print(f"\n[跳过] {future_name} - 文件不存在")
            continue

        params = OPTIMIZED_PARAMS[future_name]
        stats = verify_future(csv_path, future_name, params)

        if stats:
            summary.append({
                'name': future_name,
                'original_return': ORIGINAL_RETURNS[future_name],
                'optimized_return': stats['return_pct'],
                'improvement': stats['return_pct'] - ORIGINAL_RETURNS[future_name],
                'trades': stats['total_trades'],
                'win_rate': stats['win_rate'],
                'sniper': stats['sniper_trades'],
                'chase': stats['chase_trades']
            })

    # 总结
    print("\n" + "=" * 80)
    print("最终总结")
    print("=" * 80)

    if not summary:
        print("无有效结果")
        return

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('optimized_return', ascending=False)

    print(f"\n{'品种':<8} {'原始收益':>10} {'优化后收益':>12} {'提升':>10} {'交易数':>6} {'胜率':>8}")
    print("-" * 80)

    for _, row in summary_df.iterrows():
        print(f"{row['name']:<8} {row['original_return']:>+9.2f}% {row['optimized_return']:>+12.2f}% "
              f"{row['improvement']:>+9.2f}% {row['trades']:>6} {row['win_rate']:>7.1f}%")

    print(f"\n推荐监控品种（按优化后收益排序）：")
    for i, (_, row) in enumerate(summary_df.iterrows(), 1):
        params = OPTIMIZED_PARAMS[row['name']]
        print(f"{i}. {row['name']}: {row['optimized_return']:+.2f}% "
              f"(EMA({params['EMA_FAST']},{params['EMA_SLOW']}), RSI={params['RSI_FILTER']})")

if __name__ == '__main__':
    main()
