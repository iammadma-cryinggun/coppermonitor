# -*- coding: utf-8 -*-
"""
PTA做多做空选择分析
在每个时间点对比做多和做空的表现，找出最优选择
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime

def analyze_long_short_choice():
    """分析做多和做空的选择优劣"""

    print("=" * 120)
    print("PTA做多做空选择分析".center(120))
    print("=" * 120)

    # ========== 合约规格 ==========
    CONTRACT_SIZE = 5
    MARGIN_RATE = 0.08
    STOP_LOSS_PCT = 0.02

    # ========== 做多最优参数 ==========
    LONG_PARAMS = {
        'EMA_FAST': 12,
        'EMA_SLOW': 10,
        'RSI_FILTER': 40,
        'RATIO_TRIGGER': 1.25,
        'STC_EXIT': 75
    }

    # ========== 做空最优参数 ==========
    SHORT_PARAMS = {
        'EMA_FAST': 7,
        'EMA_SLOW': 20,
        'RSI_FILTER': 45,
        'RATIO_TRIGGER': 1.05,
        'STC_EXIT': 15
    }

    print(f"\n【做多参数】EMA({LONG_PARAMS['EMA_FAST']}, {LONG_PARAMS['EMA_SLOW']}), RSI={LONG_PARAMS['RSI_FILTER']}, RATIO={LONG_PARAMS['RATIO_TRIGGER']}")
    print(f"【做空参数】EMA({SHORT_PARAMS['EMA_FAST']}, {SHORT_PARAMS['EMA_SLOW']}), RSI={SHORT_PARAMS['RSI_FILTER']}, RATIO={SHORT_PARAMS['RATIO_TRIGGER']}")

    # ========== 获取数据 ==========
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data('TA', days=300)

    if df is None or df.empty:
        print("数据获取失败")
        return

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # ========== 计算指标（做多和做空各一套） ==========
    # 做多指标
    df['ema_fast_long'] = df['close'].ewm(span=LONG_PARAMS['EMA_FAST'], adjust=False).mean()
    df['ema_slow_long'] = df['close'].ewm(span=LONG_PARAMS['EMA_SLOW'], adjust=False).mean()

    exp1_long = df['close'].ewm(span=LONG_PARAMS['EMA_FAST'], adjust=False).mean()
    exp2_long = df['close'].ewm(span=LONG_PARAMS['EMA_SLOW'], adjust=False).mean()
    df['macd_dif_long'] = exp1_long - exp2_long
    df['macd_dea_long'] = df['macd_dif_long'].ewm(span=9, adjust=False).mean()
    df['ratio_long'] = np.where(df['macd_dea_long'] != 0, df['macd_dif_long'] / df['macd_dea_long'], 0)

    # 做空指标
    df['ema_fast_short'] = df['close'].ewm(span=SHORT_PARAMS['EMA_FAST'], adjust=False).mean()
    df['ema_slow_short'] = df['close'].ewm(span=SHORT_PARAMS['EMA_SLOW'], adjust=False).mean()

    exp1_short = df['close'].ewm(span=SHORT_PARAMS['EMA_FAST'], adjust=False).mean()
    exp2_short = df['close'].ewm(span=SHORT_PARAMS['EMA_SLOW'], adjust=False).mean()
    df['macd_dif_short'] = exp1_short - exp2_short
    df['macd_dea_short'] = df['macd_dif_short'].ewm(span=9, adjust=False).mean()
    df['ratio_short'] = np.where(df['macd_dea_short'] != 0, df['macd_dif_short'] / df['macd_dea_short'], 0)

    # RSI（共用）
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # STC指标（共用）
    def calculate_stc(series, fast_period=23, slow_period=50, period=10):
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        stoch_k = 100 * (macd - macd.rolling(window=period).min()) / \
                  (macd.rolling(window=period).max() - macd.rolling(window=period).min())
        stoch_d = stoch_k.ewm(span=period, adjust=False).mean()
        stoch_k_d = 100 * (stoch_k - stoch_d.rolling(window=period).min()) / \
                    (stoch_d.rolling(window=period).max() - stoch_d.rolling(window=period).min())
        stc = stoch_k_d.ewm(span=period, adjust=False).mean()
        return stc

    df['stc'] = calculate_stc(df['close'])
    df['stc_prev'] = df['stc'].shift(1)

    # ========== 并行回测：每个时间点同时评估做多和做空 ==========
    long_opportunities = []
    short_opportunities = []

    for i in range(200, len(df) - 20):  # 留出20根K线用于平仓
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # ========== 评估做多机会 ==========
        trend_up_long = current['ema_fast_long'] > current['ema_slow_long']
        ratio_safe_long = (0 < current['ratio_long'] < LONG_PARAMS['RATIO_TRIGGER'])
        ratio_shrinking_long = current['ratio_long'] < prev['ratio_long']
        turning_up_long = current['macd_dif_long'] > prev['macd_dif_long']
        is_strong_long = current['rsi'] > LONG_PARAMS['RSI_FILTER']

        sniper_long = trend_up_long and ratio_safe_long and ratio_shrinking_long and turning_up_long and is_strong_long
        ema_golden_cross = (prev['ema_fast_long'] <= prev['ema_slow_long']) and (current['ema_fast_long'] > current['ema_slow_long'])
        chase_long = ema_golden_cross and is_strong_long

        if sniper_long or chase_long:
            # 模拟做多交易，计算盈亏
            entry_price = current['close']
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)

            # 向前找平仓点
            for j in range(i+1, min(i+100, len(df))):
                future = df.iloc[j]
                future_prev = df.iloc[j-1]

                # 止损
                if future['low'] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = '止损'
                    break

                # STC止盈
                stc_exit = (future_prev['stc'] > LONG_PARAMS['STC_EXIT']) and (future['stc'] < future_prev['stc'])
                trend_exit = future['ema_fast_long'] < future['ema_slow_long']

                if stc_exit:
                    exit_price = future['close']
                    exit_reason = 'STC止盈'
                    break
                elif trend_exit:
                    exit_price = future['close']
                    exit_reason = '趋势反转'
                    break
            else:
                # 没有触发平仓，用最后一根K线
                exit_price = df.iloc[min(i+100, len(df)-1)]['close']
                exit_reason = '超时'

            pnl_pct = (exit_price - entry_price) / entry_price * 100

            long_opportunities.append({
                'datetime': current['datetime'],
                'signal_type': 'sniper_long' if sniper_long else 'chase_long',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'hold_bars': j - i
            })

        # ========== 评估做空机会 ==========
        trend_down_short = current['ema_fast_short'] < current['ema_slow_short']
        ratio_safe_short = (-SHORT_PARAMS['RATIO_TRIGGER'] < current['ratio_short'] < 0)
        ratio_falling_short = current['ratio_short'] < prev['ratio_short']
        turning_down_short = current['macd_dif_short'] < prev['macd_dif_short']
        is_weak_short = current['rsi'] < SHORT_PARAMS['RSI_FILTER']

        sniper_short = trend_down_short and ratio_safe_short and ratio_falling_short and turning_down_short and is_weak_short
        ema_death_cross = (prev['ema_fast_short'] >= prev['ema_slow_short']) and (current['ema_fast_short'] < current['ema_slow_short'])
        chase_short = ema_death_cross and is_weak_short

        if sniper_short or chase_short:
            # 模拟做空交易，计算盈亏
            entry_price = current['close']
            stop_loss = entry_price * (1 + STOP_LOSS_PCT)

            # 向前找平仓点
            for j in range(i+1, min(i+100, len(df))):
                future = df.iloc[j]
                future_prev = df.iloc[j-1]

                # 止损
                if future['high'] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = '止损'
                    break

                # STC止盈
                stc_exit = (future_prev['stc'] < SHORT_PARAMS['STC_EXIT']) and (future['stc'] > future_prev['stc'])
                trend_exit = future['ema_fast_short'] > future['ema_slow_short']

                if stc_exit:
                    exit_price = future['close']
                    exit_reason = 'STC止盈'
                    break
                elif trend_exit:
                    exit_price = future['close']
                    exit_reason = '趋势反转'
                    break
            else:
                # 没有触发平仓，用最后一根K线
                exit_price = df.iloc[min(i+100, len(df)-1)]['close']
                exit_reason = '超时'

            pnl_pct = (entry_price - exit_price) / entry_price * 100

            short_opportunities.append({
                'datetime': current['datetime'],
                'signal_type': 'sniper_short' if sniper_short else 'chase_short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'hold_bars': j - i
            })

    # ========== 统计分析 ==========
    long_df = pd.DataFrame(long_opportunities)
    short_df = pd.DataFrame(short_opportunities)

    print(f"\n{'='*120}")
    print("信号统计")
    print(f"{'='*120}")

    print(f"\n【做多信号】")
    print(f"  总信号数: {len(long_df)} 个")
    print(f"  盈利: {len(long_df[long_df['pnl_pct'] > 0])} 个")
    print(f"  亏损: {len(long_df[long_df['pnl_pct'] <= 0])} 个")
    print(f"  胜率: {len(long_df[long_df['pnl_pct'] > 0]) / len(long_df) * 100:.1f}%")
    print(f"  平均盈亏: {long_df['pnl_pct'].mean():.2f}%")

    print(f"\n【做空信号】")
    print(f"  总信号数: {len(short_df)} 个")
    print(f"  盈利: {len(short_df[short_df['pnl_pct'] > 0])} 个")
    print(f"  亏损: {len(short_df[short_df['pnl_pct'] <= 0])} 个")
    print(f"  胜率: {len(short_df[short_df['pnl_pct'] > 0]) / len(short_df) * 100:.1f}%")
    print(f"  平均盈亏: {short_df['pnl_pct'].mean():.2f}%")

    # ========== 时间对比分析 ==========
    print(f"\n{'='*120}")
    print("时间对比分析（同一时间点的做多vs做空）")
    print(f"{'='*120}")

    # 找出同一时间点既有做多信号又有做空信号的情况
    merged = pd.merge(long_df, short_df, on='datetime', how='inner', suffixes=('_long', '_short'))

    if len(merged) > 0:
        print(f"\n同一时间点既有做多又有做空信号的情况: {len(merged)} 次")
        print(f"\n{'-'*120}")
        print(f"{'时间':<19} {'做多盈亏%':<12} {'做空盈亏%':<12} {'更优选择':<10} {'差异':<10}")
        print(f"{'-'*120}")

        better_count_long = 0
        better_count_short = 0

        for _, row in merged.iterrows():
            long_pnl = row['pnl_pct_long']
            short_pnl = row['pnl_pct_short']

            if long_pnl > short_pnl:
                better = '做多'
                better_count_long += 1
                diff = long_pnl - short_pnl
            else:
                better = '做空'
                better_count_short += 1
                diff = short_pnl - long_pnl

            print(f"{str(row['datetime']):<19} {long_pnl:+.2f}%{' ':<8} {short_pnl:+.2f}%{' ':<8} {better:<10} {diff:+.2f}%")

        print(f"\n汇总:")
        print(f"  做多更好: {better_count_long} 次 ({better_count_long/len(merged)*100:.1f}%)")
        print(f"  做空更好: {better_count_short} 次 ({better_count_short/len(merged)*100:.1f}%)")

    else:
        print("\n没有同一时间点既有做多又有做空信号的情况")

    # ========== 做多失败时，做空是否更好？ ==========
    print(f"\n{'='*120}")
    print("关键分析：做多失败时，做空是否更好？")
    print(f"{'='*120}")

    # 找出做多失败的案例
    long_losses = long_df[long_df['pnl_pct'] <= 0].copy()

    if len(long_losses) > 0:
        # 检查这些时间点附近是否有做空信号
        comparison_data = []

        for _, long_trade in long_losses.iterrows():
            long_time = long_trade['datetime']
            long_pnl = long_trade['pnl_pct']

            # 查找前后5根K线内是否有做空信号
            nearby_shorts = short_df[
                (short_df['datetime'] >= long_time - pd.Timedelta(hours=20)) &
                (short_df['datetime'] <= long_time + pd.Timedelta(hours=20))
            ]

            if len(nearby_shorts) > 0:
                # 取最近的做空信号
                nearby_shorts['time_diff'] = abs(nearby_shorts['datetime'] - long_time)
                nearest_short = nearby_shorts.sort_values('time_diff').iloc[0]

                comparison_data.append({
                    'datetime': long_time,
                    'long_pnl': long_pnl,
                    'short_time': nearest_short['datetime'],
                    'short_pnl': nearest_short['pnl_pct'],
                    'time_diff_hours': nearest_short['time_diff'].total_seconds() / 3600,
                    'better': '做空' if nearest_short['pnl_pct'] > long_pnl else '做多',
                    'improvement': nearest_short['pnl_pct'] - long_pnl
                })

        if len(comparison_data) > 0:
            comparison_df = pd.DataFrame(comparison_data)

            print(f"\n做多失败的{len(long_losses)}次中，{len(comparison_data)}次附近有做空信号")
            print(f"\n{'-'*120}")
            print(f"{'做多时间':<19} {'做多盈亏%':<12} {'做空时间':<19} {'做空盈亏%':<12} {'时间差':<10} {'更优':<10} {'改善':<10}")
            print(f"{'-'*120}")

            improvement_better = 0
            improvement_worse = 0

            for _, row in comparison_df.iterrows():
                print(f"{str(row['datetime']):<19} {row['long_pnl']:+.2f}%{' ':<8} "
                      f"{str(row['short_time']):<19} {row['short_pnl']:+.2f}%{' ':<8} "
                      f"{row['time_diff_hours']:.1f}h{' ':<6} {row['better']:<10} {row['improvement']:+.2f}%")

                if row['improvement'] > 0:
                    improvement_better += 1
                else:
                    improvement_worse += 1

            print(f"\n统计:")
            print(f"  做空更好: {improvement_better} 次 ({improvement_better/len(comparison_df)*100:.1f}%)")
            print(f"  做空更差: {improvement_worse} 次 ({improvement_worse/len(comparison_df)*100:.1f}%)")
            print(f"  平均改善: {comparison_df['improvement'].mean():+.2f}%")

            if improvement_better > improvement_worse:
                print(f"\n结论: 做多失败时，做空信号确实更好！")
            else:
                print(f"\n结论: 做多失败时，做空信号并不一定更好。")
        else:
            print("\n做多失败的时间点附近，没有找到做空信号")
    else:
        print("\n没有做多失败的案例")

    # ========== 保存结果 ==========
    output_long = Path('logs/pta_long_opportunities.csv')
    output_short = Path('logs/pta_short_opportunities.csv')

    long_df.to_csv(output_long, index=False, encoding='utf-8-sig')
    short_df.to_csv(output_short, index=False, encoding='utf-8-sig')

    print(f"\n做多信号已保存到: {output_long}")
    print(f"做空信号已保存到: {output_short}")

if __name__ == '__main__':
    analyze_long_short_choice()
