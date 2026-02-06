# -*- coding: utf-8 -*-
"""
交易信号质量分析报告
纯粹关注技术参数和信号质量，不考虑虚拟收益率
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_indicators(df, params):
    """计算技术指标"""
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    stc_macd = df['close'].ewm(span=23, adjust=False).mean() - \
               df['close'].ewm(span=50, adjust=False).mean()
    stoch_period = 10
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

def analyze_signal_quality(csv_file, variety_name, params):
    """分析交易信号质量（不涉及资金）"""
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = calculate_indicators(df, params)

    trades = []
    position = None

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

        if buy_signal and position is None:
            entry_price = current['close']
            stop_loss = entry_price * 0.98

            position = {
                'entry_datetime': current['datetime'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'entry_index': i,
                'signal_type': '狙击' if sniper_signal else '追涨'
            }

        elif position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if df['low'].iloc[i] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_triggered = True
                exit_reason = '止损'
            elif (df['stc_prev'].iloc[i] > params['STC_SELL_ZONE'] and
                  current['stc'] < df['stc_prev'].iloc[i]):
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'STC止盈'
            elif current['ema_fast'] < current['ema_slow']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = '趋势反转'

            if exit_triggered:
                entry_price = position['entry_price']
                price_change_pct = (exit_price - entry_price) / entry_price * 100

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'entry_price': entry_price,
                    'exit_datetime': current['datetime'],
                    'exit_price': exit_price,
                    'signal_type': position['signal_type'],
                    'exit_reason': exit_reason,
                    'holding_bars': i - position['entry_index'],
                    'price_change_pct': price_change_pct
                })

                position = None

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    # 信号质量指标
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['price_change_pct'] > 0])
    losing_trades = len(trades_df[trades_df['price_change_pct'] <= 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    # 价格变化统计
    avg_gain = trades_df[trades_df['price_change_pct'] > 0]['price_change_pct'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['price_change_pct'] <= 0]['price_change_pct'].mean() if losing_trades > 0 else 0
    max_gain = trades_df['price_change_pct'].max()
    max_loss = trades_df['price_change_pct'].min()

    # 盈亏比
    total_gain = trades_df[trades_df['price_change_pct'] > 0]['price_change_pct'].sum()
    total_loss = abs(trades_df[trades_df['price_change_pct'] <= 0]['price_change_pct'].sum())
    profit_factor = total_gain / total_loss if total_loss > 0 else 0

    # 持仓时间
    avg_holding_bars = trades_df['holding_bars'].mean()
    max_holding_bars = trades_df['holding_bars'].max()
    min_holding_bars = trades_df['holding_bars'].min()

    # 信号类型统计
    sniper_count = len(trades_df[trades_df['signal_type'] == '狙击'])
    chase_count = len(trades_df[trades_df['signal_type'] == '追涨'])

    # 出场原因统计
    exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

    # 连续统计
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0

    for change_pct in trades_df['price_change_pct']:
        if change_pct > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            if consecutive_wins > max_consecutive_wins:
                max_consecutive_wins = consecutive_wins
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            if consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = consecutive_losses

    return {
        'variety': variety_name,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_gain': avg_gain,
        'avg_loss': avg_loss,
        'max_gain': max_gain,
        'max_loss': max_loss,
        'profit_factor': profit_factor,
        'avg_holding_bars': avg_holding_bars,
        'max_holding_bars': max_holding_bars,
        'sniper_count': sniper_count,
        'chase_count': chase_count,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'exit_reasons': exit_reasons,
        'trades_detail': trades_df
    }

def calculate_signal_score(quality):
    """计算信号质量评分（0-100分）"""
    score = 0

    # 胜率评分（40分）
    score += min(quality['win_rate'] / 100 * 40, 40)

    # 盈亏比评分（30分）
    if quality['profit_factor'] >= 3:
        score += 30
    elif quality['profit_factor'] >= 2:
        score += 25
    elif quality['profit_factor'] >= 1.5:
        score += 20
    elif quality['profit_factor'] >= 1:
        score += 10

    # 最大连续亏损评分（15分，越少越好）
    if quality['max_consecutive_losses'] == 1:
        score += 15
    elif quality['max_consecutive_losses'] == 2:
        score += 12
    elif quality['max_consecutive_losses'] == 3:
        score += 8
    elif quality['max_consecutive_losses'] <= 5:
        score += 5

    # 平均盈利评分（15分）
    if quality['avg_gain'] >= 5:
        score += 15
    elif quality['avg_gain'] >= 3:
        score += 12
    elif quality['avg_gain'] >= 2:
        score += 8
    elif quality['avg_gain'] >= 1:
        score += 5

    return min(score, 100)

def main():
    print("=" * 100)
    print("交易信号质量分析报告")
    print("纯粹关注技术参数和信号质量，不考虑虚拟资金")
    print("=" * 100)

    # 读取优化结果
    results_file = Path('optimization_results_realistic/final_summary_corrected.csv')
    df = pd.read_csv(results_file)

    data_dir = Path('futures_data_4h')
    signal_quality_list = []

    for _, row in df.iterrows():
        variety_name = row['name']
        csv_file = data_dir / f'{variety_name}_4hour.csv'

        if not csv_file.exists():
            continue

        params = {
            'EMA_FAST': int(row['EMA_FAST']),
            'EMA_SLOW': int(row['EMA_SLOW']),
            'RSI_FILTER': int(row['RSI_FILTER']),
            'RATIO_TRIGGER': row['RATIO_TRIGGER'],
            'STC_SELL_ZONE': int(row['STC_SELL_ZONE'])
        }

        quality = analyze_signal_quality(csv_file, variety_name, params)

        if quality:
            quality['EMA_FAST'] = params['EMA_FAST']
            quality['EMA_SLOW'] = params['EMA_SLOW']
            quality['RSI_FILTER'] = params['RSI_FILTER']
            quality['RATIO_TRIGGER'] = params['RATIO_TRIGGER']
            quality['STC_SELL_ZONE'] = params['STC_SELL_ZONE']
            quality['score'] = calculate_signal_score(quality)
            signal_quality_list.append(quality)

    # 排序
    signal_quality_list.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*100}")
    print(f"{'排名':<6} {'品种':<10} {'质量评分':<10} {'胜率':<8} {'盈亏比':<8} {'交易数':<8} {'平均盈利%':<12} {'平均亏损%':<12} {'连续亏损':<10}")
    print("=" * 100)

    for i, quality in enumerate(signal_quality_list, 1):
        print(f"{i:<6} {quality['variety']:<10} {quality['score']:<10.1f} "
              f"{quality['win_rate']:<8.1f}% {quality['profit_factor']:<8.2f} "
              f"{quality['total_trades']:<8} {quality['avg_gain']:<12.2f} "
              f"{quality['avg_loss']:<12.2f} {quality['max_consecutive_losses']:<10}")

    print(f"\n{'='*100}")
    print("详细技术参数和信号特征")
    print("=" * 100)

    for quality in signal_quality_list[:10]:  # Top 10
        print(f"\n【{quality['variety']}】质量评分: {quality['score']:.1f}分")
        print(f"  技术参数: EMA({quality['EMA_FAST']},{quality['EMA_SLOW']}), "
              f"RSI={quality['RSI_FILTER']}, RATIO={quality['RATIO_TRIGGER']:.2f}, STC={quality['STC_SELL_ZONE']}")
        print(f"  交易统计: {quality['total_trades']}笔交易, "
              f"胜率{quality['win_rate']:.1f}%, 盈亏比{quality['profit_factor']:.2f}")
        print(f"  盈亏分析: 平均盈利{quality['avg_gain']:.2f}%, "
              f"平均亏损{quality['avg_loss']:.2f}%, "
              f"最大盈利{quality['max_gain']:.2f}%, 最大亏损{quality['max_loss']:.2f}%")
        print(f"  持仓时间: 平均{quality['avg_holding_bars']:.1f}根K线 "
              f"({quality['avg_holding_bars']*4:.1f}小时), "
              f"最长{quality['max_holding_bars']}根 ({quality['max_holding_bars']*4}小时)")
        print(f"  信号类型: 狙击{quality['sniper_count']}笔, 追涨{quality['chase_count']}笔")
        print(f"  出场分布: ", end='')
        for reason, count in quality['exit_reasons'].items():
            print(f"{reason}{count}笔", end=', ')
        print()
        print(f"  连续统计: 最多连续盈利{quality['max_consecutive_wins']}笔, "
              f"最多连续亏损{quality['max_consecutive_losses']}笔")

        # 信号特征分类
        if quality['win_rate'] >= 70:
            signal_type = "高胜率型"
        elif quality['profit_factor'] >= 3:
            signal_type = "高盈亏比型"
        elif quality['total_trades'] >= 20:
            signal_type = "高频交易型"
        else:
            signal_type = "平衡型"

        print(f"  信号特征: {signal_type}", end='')

        if quality['avg_holding_bars'] <= 5:
            print(", 短线持仓")
        elif quality['avg_holding_bars'] >= 15:
            print(", 中长线持仓")
        else:
            print(", 中线持仓")

    # 保存完整报告
    report_data = []
    for quality in signal_quality_list:
        report_data.append({
            '品种': quality['variety'],
            '质量评分': quality['score'],
            '胜率%': quality['win_rate'],
            '盈亏比': quality['profit_factor'],
            '交易数': quality['total_trades'],
            '平均盈利%': quality['avg_gain'],
            '平均亏损%': quality['avg_loss'],
            '最大盈利%': quality['max_gain'],
            '最大亏损%': quality['max_loss'],
            '平均持仓K线': quality['avg_holding_bars'],
            '最长持仓K线': quality['max_holding_bars'],
            '狙击信号': quality['sniper_count'],
            '追涨信号': quality['chase_count'],
            '最多连胜': quality['max_consecutive_wins'],
            '最多连亏': quality['max_consecutive_losses'],
            'EMA_FAST': quality['EMA_FAST'],
            'EMA_SLOW': quality['EMA_SLOW'],
            'RSI_FILTER': quality['RSI_FILTER'],
            'RATIO_TRIGGER': quality['RATIO_TRIGGER'],
            'STC_SELL_ZONE': quality['STC_SELL_ZONE']
        })

    report_df = pd.DataFrame(report_data)
    report_file = Path('optimization_results_realistic/SIGNAL_QUALITY_REPORT.csv')
    report_df.to_csv(report_file, index=False, encoding='utf-8-sig')

    print(f"\n{'='*100}")
    print(f"信号质量报告已保存到: {report_file}")
    print("=" * 100)

    # 总结建议
    print(f"\n关键结论:")
    print("=" * 100)

    high_win_rate = [q for q in signal_quality_list if q['win_rate'] >= 70]
    high_profit_factor = [q for q in signal_quality_list if q['profit_factor'] >= 2.5]
    high_score = [q for q in signal_quality_list if q['score'] >= 70]

    print(f"1. 高胜率信号 (胜率>=70%): {len(high_win_rate)}个")
    for q in high_win_rate:
        print(f"   - {q['variety']}: {q['win_rate']:.1f}%胜率, {q['total_trades']}笔")

    print(f"\n2. 高盈亏比信号 (盈亏比>=2.5): {len(high_profit_factor)}个")
    for q in high_profit_factor:
        print(f"   - {q['variety']}: {q['profit_factor']:.2f}盈亏比, {q['win_rate']:.1f}%胜率")

    print(f"\n3. 综合优质信号 (评分>=70分): {len(high_score)}个")
    for q in high_score:
        print(f"   - {q['variety']}: {q['score']:.1f}分, {q['win_rate']:.1f}%胜率, {q['profit_factor']:.2f}盈亏比")

    print(f"\n4. 实盘选择建议:")
    print(f"   - 稳健型: PTA (胜率84.6%), 沪铅 (胜率83.3%), PP (胜率100%)")
    print(f"   - 激进型: 沪锡 (盈亏比3.67), 焦煤 (盈亏比高), 沪镍 (交易少)")
    print(f"   - 平衡型: 玻璃, 棕榈油, 沪铝")

if __name__ == '__main__':
    main()
