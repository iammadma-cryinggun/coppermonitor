# -*- coding: utf-8 -*-
"""
沪锡10年回测 - 结合LPPL滚动拟合过滤
对比有无LPPL过滤的效果
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_monitor import calculate_indicators
import numpy as np

# 最优参数（基于3年数据优化）
BEST_PARAMS = {
    'EMA_FAST': 3,
    'EMA_SLOW': 15,
    'RSI_FILTER': 30,
    'RATIO_TRIGGER': 1.05,
    'STC_SELL_ZONE': 65,
    'STOP_LOSS_PCT': 0.02
}

# 资金和风险参数
INITIAL_CAPITAL = 100000
MAX_POSITION_RATIO = 0.8
STOP_LOSS_PCT = 0.02
CONTRACT_SIZE = 1
MARGIN_RATE = 0.13


def load_lppl_rolling_data():
    """加载LPPL滚动拟合数据，并计算风险分位"""
    rolling = pd.read_csv('SN_lppl_rolling.csv')
    rolling['date'] = pd.to_datetime(rolling['date'])
    rolling['tc_datetime'] = pd.to_datetime(rolling['tc_datetime'])

    # 计算D值的百分位（用于风险分级）
    rolling['D_percentile'] = rolling['D'].rank(pct=True)

    # 创建风险等级映射
    risk_map = {}
    for idx, row in rolling.iterrows():
        risk_map[row['date']] = {
            'D': row['D'],
            'D_percentile': row['D_percentile'],
            'tc_datetime': row['tc_datetime']
        }

    return risk_map


def get_lppl_risk_level(date, lppl_map, threshold=0.5):
    """
    获取指定日期的LPPL风险等级

    Args:
        date: 交易日期
        lppl_map: LPPL风险映射字典
        threshold: 过滤阈值（百分位）

    Returns:
        dict: {'should_trade': bool, 'risk_level': str, 'D_percentile': float}
    """
    # 找到该日期之前的最近一次LPPL拟合
    prior_dates = [d for d in lppl_map.keys() if d <= date]

    if not prior_dates:
        # 没有LPPL数据，允许交易
        return {
            'should_trade': True,
            'risk_level': '无LPPL数据',
            'D_percentile': None,
            'D': None
        }

    # 获取最近的LPPL拟合
    latest_date = max(prior_dates)
    lppl_data = lppl_map[latest_date]
    d_pct = lppl_data['D_percentile']

    # 风险分级
    if d_pct >= 0.75:
        risk_level = '高危[前25%]'
        should_trade = False  # 高危期不开仓
    elif d_pct >= 0.5:
        risk_level = '中危[25-50%]'
        should_trade = d_pct < threshold  # 根据阈值决定
    elif d_pct >= 0.25:
        risk_level = '低危[0-25%]'
        should_trade = True
    else:
        risk_level = '安全[最低25%]'
        should_trade = True

    return {
        'should_trade': should_trade,
        'risk_level': risk_level,
        'D_percentile': d_pct,
        'D': lppl_data['D'],
        'lppl_date': latest_date
    }


def backtest_with_lppl_filter(df, lppl_map, lppl_threshold=0.5, enable_filter=True):
    """
    带LPPL过滤的回测

    Args:
        df: 价格数据
        lppl_map: LPPL风险映射
        lppl_threshold: LPPL过滤阈值（D百分位）
        enable_filter: 是否启用LPPL过滤
    """
    # 计算指标
    df = calculate_indicators(df.copy(), BEST_PARAMS)

    capital = INITIAL_CAPITAL
    position = None
    trades = []
    filtered_signals = 0

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 检查卖出/止损
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
                    'exit_datetime': df.iloc[i]['datetime'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'reason': exit_reason,
                    'lppl_risk': position['lppl_risk'],
                    'lppl_enabled': position['lppl_enabled']
                })

                position = None
                continue

        # 开仓逻辑
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
                # LPPL过滤检查
                current_date = current['datetime']
                lppl_risk = get_lppl_risk_level(current_date, lppl_map, threshold=lppl_threshold)

                should_trade = lppl_risk['should_trade'] if enable_filter else True

                if not should_trade:
                    filtered_signals += 1
                    continue  # 过滤掉该信号

                # 计算可开仓手数
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
                        'lppl_risk': lppl_risk['risk_level'],
                        'lppl_enabled': enable_filter,
                        'lppl_pct': lppl_risk['D_percentile']
                    }

    # 计算统计数据
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0

    return {
        'capital': capital,
        'return': total_return,
        'trades': len(trades),
        'filtered': filtered_signals,
        'win_rate': win_rate,
        'win_trades': len(win_trades),
        'trade_list': trades
    }


def main():
    print("="*100)
    print("沪锡10年回测 - LPPL滚动拟合过滤效果验证")
    print("="*100)

    # 1. 加载数据
    print("\n[1] 加载数据...")
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

    print(f"  数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"  数据量: {len(df)}条")

    # 2. 加载LPPL数据
    print("\n[2] 加载LPPL滚动拟合数据...")
    lppl_map = load_lppl_rolling_data()
    print(f"  LPPL拟合次数: {len(lppl_map)}次")
    print(f"  时间范围: {min(lppl_map.keys())} ~ {max(lppl_map.keys())}")

    # 3. 回测 - 无LPPL过滤（基准）
    print("\n[3] 回测 - 无LPPL过滤（基准）...")
    result_baseline = backtest_with_lppl_filter(df, lppl_map, enable_filter=False)

    print(f"  最终权益: {result_baseline['capital']:,.0f}元")
    print(f"  收益率: {result_baseline['return']:.2f}%")
    print(f"  交易次数: {result_baseline['trades']}笔")
    print(f"  胜率: {result_baseline['win_rate']:.1f}%")

    # 4. 测试不同LPPL阈值
    print("\n[4] 测试不同LPPL阈值...")

    thresholds = [0.5, 0.6, 0.7, 0.75]
    results = []

    for threshold in thresholds:
        print(f"\n  测试阈值: D百分位 >= {threshold*100:.0f}%")
        result = backtest_with_lppl_filter(df, lppl_map, lppl_threshold=threshold, enable_filter=True)

        print(f"    最终权益: {result['capital']:,.0f}元")
        print(f"    收益率: {result['return']:.2f}%")
        print(f"    交易次数: {result['trades']}笔")
        print(f"    过滤信号: {result['filtered']}笔")
        print(f"    胜率: {result['win_rate']:.1f}%")

        results.append({
            'threshold': threshold,
            'result': result
        })

    # 5. 找出最优阈值
    print("\n" + "="*100)
    print("对比分析")
    print("="*100)

    print(f"\n{'策略':<30} {'收益率':<12} {'交易次数':<10} {'胜率':<10} {'过滤数':<10}")
    print("-"*100)

    # 基准
    print(f"{'无LPPL过滤（基准）':<30} {result_baseline['return']:>10.2f}%   {result_baseline['trades']:>8}笔   {result_baseline['win_rate']:>8.1f}%   {0:>8}笔")

    # 各阈值
    for r in results:
        threshold_name = f"LPPL过滤(D>={r['threshold']*100:.0f}%分位)"
        return_change = r['result']['return'] - result_baseline['return']
        sign = '+' if return_change > 0 else ''

        print(f"{threshold_name:<30} {r['result']['return']:>10.2f}%   {r['result']['trades']:>8}笔   {r['result']['win_rate']:>8.1f}%   {r['result']['filtered']:>8}笔 ({sign}{return_change:.2f}%)")

    # 6. 选择最优方案
    print("\n" + "="*100)
    print("最优方案推荐")
    print("="*100)

    best_result = max(results, key=lambda x: x['result']['return'])
    best_threshold = best_result['threshold']
    best_data = best_result['result']

    improvement = best_data['return'] - result_baseline['return']
    filtered_ratio = best_data['filtered'] / (best_data['trades'] + best_data['filtered']) * 100

    print(f"\n最优阈值: D百分位 >= {best_threshold*100:.0f}%")
    print(f"  收益率提升: {improvement:+.2f}% ({result_baseline['return']:.2f}% -> {best_data['return']:.2f}%)")
    print(f"  过滤信号: {best_data['filtered']}笔 ({filtered_ratio:.1f}%的交易信号被过滤)")
    print(f"  交易次数: {result_baseline['trades']}笔 -> {best_data['trades']}笔")
    print(f"  胜率变化: {result_baseline['win_rate']:.1f}% -> {best_data['win_rate']:.1f}%")

    # 7. 详细交易分析（最优方案）
    print("\n" + "="*100)
    print("最优方案 - 亏损交易分析")
    print("="*100)

    losing_trades = [t for t in best_data['trade_list'] if t['pnl'] <= 0]
    print(f"\n亏损交易: {len(losing_trades)}笔")

    if losing_trades:
        # 按LPPL风险等级统计
        risk_stats = {}
        for t in losing_trades:
            risk = t['lppl_risk']
            risk_stats[risk] = risk_stats.get(risk, 0) + 1

        print("\n亏损交易的LPPL风险分布:")
        for risk, count in sorted(risk_stats.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(losing_trades) * 100
            print(f"  {risk}: {count}笔 ({pct:.1f}%)")

    # 8. 保存最优方案的交易明细
    output_file = 'backtest_lppl_filtered_trades.csv'
    trades_df = pd.DataFrame(best_data['trade_list'])
    trades_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n交易明细已保存: {output_file}")

    # 9. 最终结论
    print("\n" + "="*100)
    print("最终结论")
    print("="*100)

    if improvement > 0:
        print(f"\n✓ LPPL过滤有效！")
        print(f"  使用阈值D百分位>={best_threshold*100:.0f}%可以提升收益率{improvement:.2f}%")
        print(f"  建议实战中采用此参数配置")
    else:
        print(f"\n✗ LPPL过滤在当前参数下未能提升收益")
        print(f"  可能原因：阈值设置不当或LPPL与策略信号不匹配")
        print(f"  建议：调整阈值或优化LPPL拟合参数")

    print("\n" + "="*100)
    print("回测完成")
    print("="*100)


if __name__ == "__main__":
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    main()
