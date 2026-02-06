# -*- coding: utf-8 -*-
"""
沪锡10年回测 - LPPL双向策略（LPPL反向逻辑）
LPPL高D值期做空，低D值期做多
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_monitor import calculate_indicators
import numpy as np

# 最优参数
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
    """加载LPPL滚动拟合数据"""
    rolling = pd.read_csv('SN_lppl_rolling.csv')
    rolling['date'] = pd.to_datetime(rolling['date'])
    rolling['D_percentile'] = rolling['D'].rank(pct=True)

    # 创建风险等级映射
    lppl_map = {}
    for idx, row in rolling.iterrows():
        lppl_map[row['date']] = {
            'D': row['D'],
            'D_percentile': row['D_percentile']
        }

    return lppl_map


def get_lppl_signal(date, lppl_map, long_threshold=0.25, short_threshold=0.75):
    """
    获取LPPL交易信号

    返回:
        'LONG': 低D值期，做多
        'SHORT': 高D值期，做空
        'NEUTRAL': 中D值期，观望
    """
    prior_dates = [d for d in lppl_map.keys() if d <= date]

    if not prior_dates:
        return 'NEUTRAL'

    latest_date = max(prior_dates)
    d_pct = lppl_map[latest_date]['D_percentile']

    if d_pct >= short_threshold:
        return 'SHORT'  # 高D值期：泡沫预期 → 做空
    elif d_pct <= long_threshold:
        return 'LONG'   # 低D值期：安全 → 做多
    else:
        return 'NEUTRAL'  # 中D值期：观望


def backtest_bidirectional(df, lppl_map, long_threshold=0.25, short_threshold=0.75):
    """
    双向交易回测：LPPL低值做多，高值做空
    """
    df = calculate_indicators(df.copy(), BEST_PARAMS)

    capital = INITIAL_CAPITAL
    position = None  # {'type': 'LONG'/'SHORT', ...}
    trades = []
    long_count = 0
    short_count = 0
    neutral_count = 0

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 获取LPPL信号
        lppl_signal = get_lppl_signal(current['datetime'], lppl_map, long_threshold, short_threshold)

        if lppl_signal == 'LONG':
            long_count += 1
        elif lppl_signal == 'SHORT':
            short_count += 1
        else:
            neutral_count += 1

        # 平仓逻辑
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 止损
            if position['type'] == 'LONG':
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
            elif position['type'] == 'SHORT':
                if current['high'] >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_triggered = True
                    exit_reason = '止损'
                elif (prev['stc'] < (100 - BEST_PARAMS['STC_SELL_ZONE']) and
                      current['stc'] > prev['stc']):
                    exit_price = current['close']
                    exit_triggered = True
                    exit_reason = 'STC止盈'
                elif current['ema_fast'] > current['ema_slow']:
                    exit_price = current['close']
                    exit_triggered = True
                    exit_reason = '趋势反转'

            if exit_triggered:
                if position['type'] == 'LONG':
                    pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                else:  # SHORT
                    pnl = (position['entry_price'] - exit_price) * position['contracts'] * CONTRACT_SIZE

                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': df.iloc[i]['datetime'],
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': ((exit_price - position['entry_price']) / position['entry_price'] * 100)
                              if position['type'] == 'LONG'
                              else ((position['entry_price'] - exit_price) / position['entry_price'] * 100),
                    'reason': exit_reason,
                    'lppl_signal': position['lppl_signal']
                })

                position = None
                continue

        # 开仓逻辑
        if position is None:
            # 做多信号：LPPL低值 + 原策略信号
            if lppl_signal == 'LONG':
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
                            'type': 'LONG',
                            'entry_datetime': current['datetime'],
                            'entry_price': entry_price,
                            'contracts': max_contracts,
                            'stop_loss': stop_loss_price,
                            'lppl_signal': lppl_signal
                        }

            # 做空信号：LPPL高值 + 趋势反转信号
            elif lppl_signal == 'SHORT':
                trend_down = current['ema_fast'] < current['ema_slow']
                ratio_expanding = current['ratio'] > current['ratio_prev']
                turning_down = current['macd_dif'] < prev['macd_dif']
                is_weak = current['rsi'] < (100 - BEST_PARAMS['RSI_FILTER'])

                # 简化的做空信号
                sell_signal = (trend_down and is_weak) or (
                    (prev['ema_fast'] >= prev['ema_slow']) and
                    (current['ema_fast'] < current['ema_slow']) and
                    is_weak
                )

                if sell_signal:
                    entry_price = current['close']
                    margin_per_contract = entry_price * CONTRACT_SIZE * MARGIN_RATE
                    available_capital = capital * MAX_POSITION_RATIO
                    max_contracts = int(available_capital / margin_per_contract)

                    if max_contracts > 0:
                        stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)
                        position = {
                            'type': 'SHORT',
                            'entry_datetime': current['datetime'],
                            'entry_price': entry_price,
                            'contracts': max_contracts,
                            'stop_loss': stop_loss_price,
                            'lppl_signal': lppl_signal
                        }

    # 计算统计数据
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0

    long_trades = [t for t in trades if t['type'] == 'LONG']
    short_trades = [t for t in trades if t['type'] == 'SHORT']

    long_pnl = sum(t['pnl'] for t in long_trades)
    short_pnl = sum(t['pnl'] for t in short_trades)

    return {
        'capital': capital,
        'return': total_return,
        'trades': len(trades),
        'win_rate': win_rate,
        'win_trades': len(win_trades),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_pnl': long_pnl,
        'short_pnl': short_pnl,
        'long_count': long_count,
        'short_count': short_count,
        'neutral_count': neutral_count,
        'trade_list': trades
    }


def main():
    print("="*100)
    print("沪锡10年回测 - LPPL双向策略（反向逻辑）")
    print("="*100)

    # 加载数据
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

    # 加载LPPL数据
    print("\n[2] 加载LPPL数据...")
    lppl_map = load_lppl_rolling_data()
    print(f"  LPPL拟合次数: {len(lppl_map)}次")

    # 测试不同阈值组合
    print("\n[3] 测试不同LPPL阈值组合...")

    configs = [
        (0.25, 0.75),  # 标准：低25%做多，高25%做空
        (0.30, 0.70),  # 收紧：减少交易频率
        (0.20, 0.80),  # 放宽：增加交易频率
        (0.35, 0.65),  # 激进：更窄的中性区
    ]

    results = []

    for long_thresh, short_thresh in configs:
        print(f"\n  测试配置: LPPL<={long_thresh*100:.0f}%做多, LPPL>={short_thresh*100:.0f}%做空")
        result = backtest_bidirectional(df, lppl_map, long_thresh, short_thresh)

        print(f"    收益率: {result['return']:.2f}%")
        print(f"    交易次数: {result['trades']}笔 (做多{result['long_trades']}笔, 做空{result['short_trades']}笔)")
        print(f"    胜率: {result['win_rate']:.1f}%")
        print(f"    做多盈亏: {result['long_pnl']:,.0f}元")
        print(f"    做空盈亏: {result['short_pnl']:,.0f}元")

        results.append({
            'config': (long_thresh, short_thresh),
            'result': result
        })

    # 对比分析
    print("\n" + "="*100)
    print("对比分析")
    print("="*100)

    print(f"\n{'策略配置':<40} {'收益率':<12} {'交易':<8} {'做多':<8} {'做空':<8} {'胜率':<8}")
    print("-"*100)

    for r in results:
        long_thresh, short_thresh = r['config']
        config_name = f"LPPL<={long_thresh*100:.0f}%做多, >={short_thresh*100:.0f}%做空"

        print(f"{config_name:<40} {r['result']['return']:>10.2f}%   {r['result']['trades']:>6}笔   {r['result']['long_trades']:>6}笔   {r['result']['short_trades']:>6}笔   {r['result']['win_rate']:>6.1f}%")

    # 找出最优配置
    print("\n" + "="*100)
    print("最优配置推荐")
    print("="*100)

    best_result = max(results, key=lambda x: x['result']['return'])
    best_config = best_result['config']
    best_data = best_result['result']

    print(f"\n最优配置: LPPL<={best_config[0]*100:.0f}%做多, LPPL>={best_config[1]*100:.0f}%做空")
    print(f"  收益率: {best_data['return']:.2f}%")
    print(f"  交易次数: {best_data['trades']}笔")
    print(f"    做多: {best_data['long_trades']}笔, 盈亏: {best_data['long_pnl']:,.0f}元")
    print(f"    做空: {best_data['short_trades']}笔, 盈亏: {best_data['short_pnl']:,.0f}元")
    print(f"  胜率: {best_data['win_rate']:.1f}%")
    print(f"  LPPL信号分布: 做多信号{best_data['long_count']}天, 做空信号{best_data['short_count']}天, 观望{best_data['neutral_count']}天")

    # 详细分析
    print("\n" + "="*100)
    print("做多 vs 做空效果对比")
    print("="*100)

    long_trades_list = [t for t in best_data['trade_list'] if t['type'] == 'LONG']
    short_trades_list = [t for t in best_data['trade_list'] if t['type'] == 'SHORT']

    if long_trades_list:
        long_wins = len([t for t in long_trades_list if t['pnl'] > 0])
        print(f"\n做多交易: {len(long_trades_list)}笔")
        print(f"  盈利: {long_wins}笔")
        print(f"  亏损: {len(long_trades_list) - long_wins}笔")
        print(f"  胜率: {long_wins/len(long_trades_list)*100:.1f}%")
        print(f"  总盈亏: {best_data['long_pnl']:,.0f}元")
        if long_wins > 0:
            long_win_pnl = sum(t['pnl'] for t in long_trades_list if t['pnl'] > 0)
            long_loss_pnl = sum(t['pnl'] for t in long_trades_list if t['pnl'] <= 0)
            print(f"  盈利交易平均: {long_win_pnl/long_wins:,.0f}元")
            print(f"  亏损交易平均: {long_loss_pnl/(len(long_trades_list)-long_wins):,.0f}元")

    if short_trades_list:
        short_wins = len([t for t in short_trades_list if t['pnl'] > 0])
        print(f"\n做空交易: {len(short_trades_list)}笔")
        print(f"  盈利: {short_wins}笔")
        print(f"  亏损: {len(short_trades_list) - short_wins}笔")
        print(f"  胜率: {short_wins/len(short_trades_list)*100:.1f}%")
        print(f"  总盈亏: {best_data['short_pnl']:,.0f}元")
        if short_wins > 0:
            short_win_pnl = sum(t['pnl'] for t in short_trades_list if t['pnl'] > 0)
            short_loss_pnl = sum(t['pnl'] for t in short_trades_list if t['pnl'] <= 0)
            print(f"  盈利交易平均: {short_win_pnl/short_wins:,.0f}元")
            print(f"  亏损交易平均: {short_loss_pnl/(len(short_trades_list)-short_wins):,.0f}元")

    # 保存交易明细
    output_file = 'backtest_lppl_bidirectional_trades.csv'
    trades_df = pd.DataFrame(best_data['trade_list'])
    trades_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n交易明细已保存: {output_file}")

    print("\n" + "="*100)
    print("回测完成")
    print("="*100)


if __name__ == "__main__":
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    main()
