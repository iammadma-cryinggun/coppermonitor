"""
最差周期市场特征分析 - 找出策略失效原因
分析最差周期期间的：趋势方向、波动率、连续涨跌、市场环境
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from 最优参数配置 import OPTIMAL_PARAMS

CODE_MAP = {
    '铜': 'cu0', '铝': 'al0', '锌': 'zn0', '铅': 'pb0', '镍': 'ni0', '锡': 'sn0',
    '黄金': 'au0', '白银': 'ag0', '玻璃': 'fg0', '纯碱': 'sa0', '糖': 'sr0', '棉花': 'cf0'
}

# 各品种最差周期
WORST_PERIODS = {
    '铜': ('2013-01', '2014-12'),
    '黄金': ('2012-01', '2014-01'),
    '棉花': ('2011-01', '2012-12'),
    '铝': ('2007-01', '2008-12'),
    '白银': ('2012-05', '2014-05'),
    '锌': ('2007-03', '2009-03'),
    '糖': ('2006-01', '2008-01'),
    '铅': ('2011-03', '2013-03'),
    '镍': ('2023-03', '2025-03'),
    '锡': ('2017-03', '2019-03'),
    '玻璃': ('2012-12', '2014-12'),
}


def get_akshare_data(code):
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()
        df.set_index('date', inplace=True)
        return df
    except:
        return None


def analyze_market_environment(df, start, end):
    """分析市场环境"""
    # 截取时间段
    mask = (df.index >= start) & (df.index <= end)
    period_df = df[mask].copy()

    if len(period_df) < 100:
        return None

    # 计算各种指标
    period_df['return'] = period_df['close'].pct_change()
    period_df['ma20'] = period_df['close'].rolling(20).mean()
    period_df['ma60'] = period_df['close'].rolling(60).mean()
    period_df['ma120'] = period_df['close'].rolling(120).mean()

    # 趋势分析
    start_price = period_df['close'].iloc[0]
    end_price = period_df['close'].iloc[-1]
    total_change = (end_price - start_price) / start_price * 100

    # 最高最低点
    highest = period_df['close'].max()
    lowest = period_df['close'].min()
    highest_date = period_df['close'].idxmax()
    lowest_date = period_df['close'].idxmin()

    # 波动率
    volatility = period_df['return'].std() * np.sqrt(250) * 100  # 年化波动率

    # 趋势判断（MA20与MA60的关系）
    ma20_above_ma60 = (period_df['ma20'] > period_df['ma60']).sum() / len(period_df) * 100

    # 连续下跌分析
    period_df['down'] = period_df['return'] < 0
    consecutive_downs = []
    current_down = 0
    for down in period_df['down']:
        if down:
            current_down += 1
        else:
            if current_down > 0:
                consecutive_downs.append(current_down)
            current_down = 0
    if current_down > 0:
        consecutive_downs.append(current_down)

    max_consecutive_down = max(consecutive_downs) if consecutive_downs else 0
    avg_consecutive_down = np.mean(consecutive_downs) if consecutive_downs else 0

    # 涨跌天数统计
    up_days = (period_df['return'] > 0).sum()
    down_days = (period_df['return'] < 0).sum()

    # 大涨大跌统计
    big_up = (period_df['return'] > 0.03).sum()  # 单日涨超3%
    big_down = (period_df['return'] < -0.03).sum()  # 单日跌超3%

    # 反弹分析（从低点反弹幅度）
    if lowest_date < pd.Timestamp(end):
        rebound = (end_price - lowest) / lowest * 100
    else:
        rebound = 0

    # 回撤分析（从高点回撤幅度）
    drawdown = (highest - end_price) / highest * 100

    # 下跌趋势中反弹（熊市反弹）
    # 判断是否整体下跌
    is_downtrend = total_change < -10

    return {
        'start_price': start_price,
        'end_price': end_price,
        'total_change': total_change,
        'highest': highest,
        'lowest': lowest,
        'highest_date': highest_date,
        'lowest_date': lowest_date,
        'volatility': volatility,
        'ma20_above_ma60_pct': ma20_above_ma60,
        'max_consecutive_down': max_consecutive_down,
        'avg_consecutive_down': avg_consecutive_down,
        'up_days': up_days,
        'down_days': down_days,
        'big_up_days': big_up,
        'big_down_days': big_down,
        'rebound_from_low': rebound,
        'drawdown_from_high': drawdown,
        'is_downtrend': is_downtrend,
        'trading_days': len(period_df),
    }


def analyze_strategy_weakness(df, params, start, end):
    """分析策略弱点"""
    from collections import defaultdict

    # 截取时间段
    mask = (df.index >= start) & (df.index <= end)
    period_df = df[mask].copy()

    if len(period_df) < 100:
        return None

    # 计算CCI
    cci_length = params['cci_length']
    tp = (period_df['high'] + period_df['low'] + period_df['close']) / 3
    sma = tp.rolling(window=cci_length).mean()
    mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
    period_df['cci'] = (tp - sma) / (0.015 * mad)

    # 计算MA
    period_df['ma20'] = period_df['close'].rolling(20).mean()
    period_df['ma60'] = period_df['close'].rolling(60).mean()

    # 分析超卖信号
    oversold_signals = period_df[period_df['cci'] < params['cci_oversold']]

    # 超卖后价格走势
    oversold_performance = []
    for idx in oversold_signals.index:
        future_5d = period_df.loc[idx:idx + pd.Timedelta(days=7), 'close']
        if len(future_5d) >= 5:
            entry = period_df.loc[idx, 'close']
            future_ret = (future_5d.iloc[-1] - entry) / entry * 100
            oversold_performance.append({
                'date': idx,
                'cci': period_df.loc[idx, 'cci'],
                'ma20_above_price': period_df.loc[idx, 'ma20'] > period_df.loc[idx, 'close'],
                'ma60_above_price': period_df.loc[idx, 'ma60'] > period_df.loc[idx, 'close'],
                'future_5d_return': future_ret
            })

    if not oversold_performance:
        return None

    perf_df = pd.DataFrame(oversold_performance)

    # 分析失效原因
    total_signals = len(perf_df)
    profitable_signals = (perf_df['future_5d_return'] > 0).sum()
    losing_signals = (perf_df['future_5d_return'] <= 0).sum()

    # 趋势逆势信号
    counter_trend_signals = perf_df[perf_df['ma60_above_price'] == True]  # 价格在MA60下方（下跌趋势）
    counter_trend_loss_rate = (counter_trend_signals['future_5d_return'] <= 0).sum() / len(counter_trend_signals) * 100 if len(counter_trend_signals) > 0 else 0

    # 顺势信号
    trend_signals = perf_df[perf_df['ma60_above_price'] == False]
    trend_win_rate = (trend_signals['future_5d_return'] > 0).sum() / len(trend_signals) * 100 if len(trend_signals) > 0 else 0

    return {
        'total_signals': total_signals,
        'profitable_signals': profitable_signals,
        'losing_signals': losing_signals,
        'win_rate': profitable_signals / total_signals * 100,
        'counter_trend_count': len(counter_trend_signals),
        'counter_trend_loss_rate': counter_trend_loss_rate,
        'trend_count': len(trend_signals),
        'trend_win_rate': trend_win_rate,
        'avg_future_return': perf_df['future_5d_return'].mean(),
    }


def main():
    print("="*120)
    print("最差周期市场特征分析 - 找出策略失效原因".center(120))
    print("="*120)

    print("""
【分析目标】
1. 识别最差周期的市场特征（趋势、波动率）
2. 分析策略在这些市场环境下的弱点
3. 提出针对性改进建议
""")

    results = {}

    for symbol, (start_str, end_str) in WORST_PERIODS.items():
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            continue

        code = CODE_MAP[symbol]
        df = get_akshare_data(code)
        if df is None:
            continue

        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)

        print(f"\n{'='*100}")
        print(f"[{symbol}] 最差周期: {start_str} ~ {end_str}".center(100))
        print("="*100)

        # 市场环境分析
        env = analyze_market_environment(df, start, end)
        if env:
            print(f"\n【市场环境】")
            print(f"  价格变化: {env['start_price']:.2f} -> {env['end_price']:.2f} ({env['total_change']:+.2f}%)")
            print(f"  最高价: {env['highest']:.2f} ({env['highest_date'].strftime('%Y-%m-%d')})")
            print(f"  最低价: {env['lowest']:.2f} ({env['lowest_date'].strftime('%Y-%m-%d')})")
            print(f"  年化波动率: {env['volatility']:.2f}%")
            print(f"  MA20>MA60比例: {env['ma20_above_ma60_pct']:.1f}%")
            print(f"  最大连续下跌: {env['max_consecutive_down']}天")
            print(f"  上涨天数/下跌天数: {env['up_days']}/{env['down_days']}")
            print(f"  大涨天数(>3%): {env['big_up_days']}, 大跌天数(<-3%): {env['big_down_days']}")
            print(f"  从高点回撤: {env['drawdown_from_high']:.2f}%")
            print(f"  从低点反弹: {env['rebound_from_low']:.2f}%")

            # 市场特征判断
            print(f"\n【市场特征】")
            if env['total_change'] < -20:
                print(f"  *** 明显下跌趋势 (跌幅>{20}%)")
            elif env['total_change'] < -10:
                print(f"  *** 下跌趋势 (跌幅>{10}%)")
            elif env['total_change'] < 0:
                print(f"  * 小幅下跌")
            else:
                print(f"  上涨趋势")

            if env['volatility'] > 30:
                print(f"  *** 高波动市场 (波动率>{30}%)")
            elif env['volatility'] > 20:
                print(f"  ** 中等波动 (波动率>{20}%)")

            if env['ma20_above_ma60_pct'] < 30:
                print(f"  *** 持续下跌形态 (MA20<MA60超过70%时间)")
            elif env['ma20_above_ma60_pct'] < 50:
                print(f"  ** 震荡偏弱")

            if env['max_consecutive_down'] > 10:
                print(f"  *** 出现长连续下跌 ({env['max_consecutive_down']}天)")

        # 策略弱点分析
        weakness = analyze_strategy_weakness(df, params, start, end)
        if weakness:
            print(f"\n【策略表现】")
            print(f"  超卖信号总数: {weakness['total_signals']}")
            print(f"  5日后盈利比例: {weakness['win_rate']:.1f}%")
            print(f"  平均5日收益: {weakness['avg_future_return']:+.2f}%")

            print(f"\n【逆势 vs 顺势】")
            print(f"  逆势信号(价格<MA60): {weakness['counter_trend_count']}笔, 亏损率{weakness['counter_trend_loss_rate']:.1f}%")
            print(f"  顺势信号(价格>MA60): {weakness['trend_count']}笔, 胜率{weakness['trend_win_rate']:.1f}%")

        # 问题诊断
        print(f"\n【问题诊断】")
        problems = []

        if env and env['total_change'] < -20:
            problems.append("在明显下跌趋势中做多")

        if env and env['ma20_above_ma60_pct'] < 40:
            problems.append("价格长期在均线下方(弱势市场)")

        if weakness and weakness['counter_trend_loss_rate'] > 60:
            problems.append(f"逆势信号亏损率过高({weakness['counter_trend_loss_rate']:.0f}%)")

        if env and env['volatility'] > 25:
            problems.append("高波动环境下固定4%止损容易被触发")

        if env and env['max_consecutive_down'] > 8:
            problems.append(f"市场出现{env['max_consecutive_down']}天连续下跌")

        if problems:
            for i, p in enumerate(problems, 1):
                print(f"  {i}. {p}")
        else:
            print(f"  暂未发现明显问题")

        # 改进建议
        print(f"\n【改进建议】")
        suggestions = []

        if env and env['total_change'] < -15:
            suggestions.append("添加趋势过滤: 价格<MA60时不开多")

        if weakness and weakness['counter_trend_loss_rate'] > 50:
            suggestions.append("禁止逆势交易: 只在MA60上方做多")

        if env and env['volatility'] > 25:
            suggestions.append("使用ATR动态止损: 替代固定4%止损")

        if env and env['max_consecutive_down'] > 6:
            suggestions.append("等待企稳信号: 连续下跌后观察2-3天")

        if suggestions:
            for i, s in enumerate(suggestions, 1):
                print(f"  {i}. {s}")
        else:
            print(f"  保持现有策略")

        results[symbol] = {
            'env': env,
            'weakness': weakness,
            'problems': problems,
            'suggestions': suggestions,
        }

    # 汇总分析
    print("\n" + "="*120)
    print("汇总分析 - 策略共性问题".center(120))
    print("="*120)

    # 统计问题频率
    all_problems = []
    for symbol, data in results.items():
        all_problems.extend(data.get('problems', []))

    from collections import Counter
    problem_counts = Counter(all_problems)

    print(f"\n【高频问题】")
    for problem, count in problem_counts.most_common(10):
        print(f"  {count}次: {problem}")

    # 统计建议频率
    all_suggestions = []
    for symbol, data in results.items():
        all_suggestions.extend(data.get('suggestions', []))

    suggestion_counts = Counter(all_suggestions)

    print(f"\n【核心改进方向】")
    for suggestion, count in suggestion_counts.most_common(5):
        print(f"  {count}次: {suggestion}")

    # 结论
    print("\n" + "="*120)
    print("结论与建议".center(120))
    print("="*120)

    print(f"""
【策略主要弱点】

1. 趋势敏感性不足
   - 在明显下跌趋势中仍做多
   - 缺少有效的趋势过滤机制

2. 逆势交易
   - 价格在MA60下方时做多胜率低
   - 需要添加趋势滤网

3. 止损方式单一
   - 固定4%止损在高波动市场容易被洗出
   - 建议改用ATR动态止损

【综合改进建议】

1. 添加趋势过滤（EMA60/MA60）
   - 价格 < MA60 时不做多
   - 避免在下跌趋势中"接飞刀"

2. 使用ATR动态止损
   - 止损 = 开仓价 - 2*ATR
   - 适应不同波动率环境

3. 等待企稳信号
   - 连续下跌后不急于进场
   - 观察是否出现止跌迹象
""")

    return results


if __name__ == "__main__":
    main()
