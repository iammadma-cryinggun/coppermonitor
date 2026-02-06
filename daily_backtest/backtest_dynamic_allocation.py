# -*- coding: utf-8 -*-
"""
细化策略：市场趋势 + LPPL强度的多空配置矩阵
两层决策系统
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

INITIAL_CAPITAL = 100000
MAX_POSITION_RATIO = 0.8
STOP_LOSS_PCT = 0.02
CONTRACT_SIZE = 1
MARGIN_RATE = 0.13


def detect_market_regime(df, lookback=60):
    """检测市场趋势状态"""
    if len(df) < lookback:
        return 'RANGING', 50

    recent = df.tail(lookback)

    # 信号判断
    ema_fast = recent['ema_fast'].iloc[-1]
    ema_slow = recent['ema_slow'].iloc[-1]
    ema_trend = (ema_fast - ema_slow) / ema_slow * 100

    price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
    annualized_change = price_change / lookback * 250 * 100

    macd_dif = recent['macd_dif'].iloc[-1]
    current_rsi = recent['rsi'].iloc[-1]

    # 计算信号强度
    bullish_signals = 0
    bearish_signals = 0

    if ema_trend > 0.5:
        bullish_signals += 2
    elif ema_trend < -0.5:
        bearish_signals += 2

    if annualized_change > 20:
        bullish_signals += 2
    elif annualized_change < -20:
        bearish_signals += 2
    elif annualized_change > 10:
        bullish_signals += 1
    elif annualized_change < -10:
        bearish_signals += 1

    if macd_dif > 0:
        bullish_signals += 1
    else:
        bearish_signals += 1

    if current_rsi > 60:
        bullish_signals += 1
    elif current_rsi < 40:
        bearish_signals += 1

    # 判断趋势
    if bullish_signals >= 4:
        return 'BULL', min(100, 70 + (bullish_signals - bearish_signals) * 10)
    elif bearish_signals >= 4:
        return 'BEAR', min(100, 70 + (bearish_signals - bullish_signals) * 10)
    else:
        return 'RANGING', 50 + abs(bullish_signals - bearish_signals) * 5


def get_lppl_signal(date, lppl_map):
    """获取LPPL信号"""
    prior_dates = [d for d in lppl_map.keys() if d <= date]
    if not prior_dates:
        return None, 0.5

    latest_date = max(prior_dates)
    d_pct = lppl_map[latest_date]['D_percentile']

    # LPPL分级
    if d_pct >= 0.8:
        return 'HIGH', d_pct
    elif d_pct >= 0.5:
        return 'MEDIUM_HIGH', d_pct
    elif d_pct >= 0.2:
        return 'MEDIUM_LOW', d_pct
    else:
        return 'LOW', d_pct


def get_position_allocation(regime, lppl_level):
    """
    根据市场趋势和LPPL强度决定多空配置

    Returns:
        tuple: (long_ratio, short_ratio, wait_ratio)
    """

    # 策略矩阵
    matrix = {
        'BULL': {
            'HIGH': (0.2, 0.6, 0.2),
            'MEDIUM_HIGH': (0.4, 0.4, 0.2),
            'MEDIUM_LOW': (0.6, 0.2, 0.2),
            'LOW': (0.8, 0.0, 0.2)
        },
        'BEAR': {
            'HIGH': (0.0, 0.8, 0.2),
            'MEDIUM_HIGH': (0.0, 0.8, 0.2),
            'MEDIUM_LOW': (0.2, 0.6, 0.2),
            'LOW': (0.2, 0.6, 0.2)
        },
        'RANGING': {
            'HIGH': (0.2, 0.6, 0.2),
            'MEDIUM_HIGH': (0.3, 0.5, 0.2),
            'MEDIUM_LOW': (0.5, 0.3, 0.2),
            'LOW': (0.6, 0.2, 0.2)
        }
    }

    return matrix[regime][lppl_level]


def backtest_dynamic_allocation(df, lppl_map):
    """动态多空配置回测"""
    df = calculate_indicators(df.copy(), BEST_PARAMS)

    capital = INITIAL_CAPITAL
    long_position = None
    short_position = None
    trades = []

    # 统计配置使用情况
    allocation_stats = {}

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 检查平仓
        # 平多头
        if long_position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if current['low'] <= long_position['stop_loss']:
                exit_price = long_position['stop_loss']
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
                pnl = (exit_price - long_position['entry_price']) * long_position['contracts'] * CONTRACT_SIZE
                capital += pnl

                trades.append({
                    'entry_datetime': long_position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'type': 'LONG',
                    'entry_price': long_position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': long_position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - long_position['entry_price']) / long_position['entry_price'] * 100,
                    'reason': exit_reason,
                    'regime': long_position['regime'],
                    'lppl_level': long_position['lppl_level'],
                    'allocation': long_position['allocation']
                })

                long_position = None

        # 平空头
        if short_position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            if current['high'] >= short_position['stop_loss']:
                exit_price = short_position['stop_loss']
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
                pnl = (short_position['entry_price'] - exit_price) * short_position['contracts'] * CONTRACT_SIZE
                capital += pnl

                trades.append({
                    'entry_datetime': short_position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'type': 'SHORT',
                    'entry_price': short_position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': short_position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (short_position['entry_price'] - exit_price) / short_position['entry_price'] * 100,
                    'reason': exit_reason,
                    'regime': short_position['regime'],
                    'lppl_level': short_position['lppl_level'],
                    'allocation': short_position['allocation']
                })

                short_position = None

        # 检查开仓
        if long_position is None and short_position is None:
            # 获取市场状态和LPPL信号
            regime, _ = detect_market_regime(df.iloc[:i+1], lookback=60)
            lppl_level, d_pct = get_lppl_signal(current['datetime'], lppl_map)

            if lppl_level is None:
                continue

            # 获取配置
            long_ratio, short_ratio, _ = get_position_allocation(regime, lppl_level)

            # 记录配置使用
            alloc_key = f"{regime}_{lppl_level}"
            allocation_stats[alloc_key] = allocation_stats.get(alloc_key, 0) + 1

            # 开多头
            if long_ratio > 0:
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
                    available_capital = capital * MAX_POSITION_RATIO * long_ratio
                    max_contracts = int(available_capital / margin_per_contract)

                    if max_contracts > 0:
                        stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                        long_position = {
                            'entry_datetime': current['datetime'],
                            'entry_price': entry_price,
                            'contracts': max_contracts,
                            'stop_loss': stop_loss_price,
                            'regime': regime,
                            'lppl_level': lppl_level,
                            'allocation': f'{long_ratio:.0f}:{short_ratio:.0f}'
                        }

            # 开空头
            if short_ratio > 0:
                trend_down = current['ema_fast'] < current['ema_slow']
                ratio_expanding = current['ratio'] > current['ratio_prev']
                turning_down = current['macd_dif'] < prev['macd_dif']
                is_weak = current['rsi'] < (100 - BEST_PARAMS['RSI_FILTER'])

                sell_signal = (trend_down and is_weak) or (
                    (prev['ema_fast'] >= prev['ema_slow']) and
                    (current['ema_fast'] < current['ema_slow']) and
                    is_weak
                )

                if sell_signal:
                    entry_price = current['close']
                    margin_per_contract = entry_price * CONTRACT_SIZE * MARGIN_RATE
                    available_capital = capital * MAX_POSITION_RATIO * short_ratio
                    max_contracts = int(available_capital / margin_per_contract)

                    if max_contracts > 0:
                        stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)
                        short_position = {
                            'entry_datetime': current['datetime'],
                            'entry_price': entry_price,
                            'contracts': max_contracts,
                            'stop_loss': stop_loss_price,
                            'regime': regime,
                            'lppl_level': lppl_level,
                            'allocation': f'{long_ratio:.0f}:{short_ratio:.0f}'
                        }

    # 计算统计
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
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_pnl': long_pnl,
        'short_pnl': short_pnl,
        'trade_list': trades,
        'allocation_stats': allocation_stats
    }


def main():
    print("="*100)
    print("细化策略回测 - 市场趋势 + LPPL强度的多空配置矩阵")
    print("="*100)

    # 加载数据
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

    # 加载LPPL数据
    lppl_map = {}
    rolling = pd.read_csv('SN_lppl_rolling.csv')
    rolling['date'] = pd.to_datetime(rolling['date'])
    rolling['D_percentile'] = rolling['D'].rank(pct=True)
    for idx, row in rolling.iterrows():
        lppl_map[row['date']] = {
            'D': row['D'],
            'D_percentile': row['D_percentile']
        }

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"LPPL拟合: {len(lppl_map)}次")

    # 策略矩阵展示
    print("\n" + "="*100)
    print("策略配置矩阵")
    print("="*100)

    matrix_display = {
        'BULL': {
            'HIGH': (0.2, 0.6, 0.2, '牛市+高D值'),
            'MEDIUM_HIGH': (0.4, 0.4, 0.2, '牛市+中高D值'),
            'MEDIUM_LOW': (0.6, 0.2, 0.2, '牛市+中低D值'),
            'LOW': (0.8, 0.0, 0.2, '牛市+低D值')
        },
        'BEAR': {
            'HIGH': (0.0, 0.8, 0.2, '熊市+高D值'),
            'MEDIUM_HIGH': (0.0, 0.8, 0.2, '熊市+中高D值'),
            'MEDIUM_LOW': (0.2, 0.6, 0.2, '熊市+中低D值'),
            'LOW': (0.2, 0.6, 0.2, '熊市+低D值')
        },
        'RANGING': {
            'HIGH': (0.2, 0.6, 0.2, '震荡+高D值'),
            'MEDIUM_HIGH': (0.3, 0.5, 0.2, '震荡+中高D值'),
            'MEDIUM_LOW': (0.5, 0.3, 0.2, '震荡+中低D值'),
            'LOW': (0.6, 0.2, 0.2, '震荡+低D值')
        }
    }

    print(f"\n{'市场状态':<12} {'LPPL强度':<12} {'多':<6} {'空':<6} {'观望':<6} {'说明':<20}")
    print("-"*100)

    for regime in ['BULL', 'BEAR', 'RANGING']:
        regime_name = {'BULL': '牛市', 'BEAR': '熊市', 'RANGING': '震荡市'}[regime]
        for lppl_level in ['LOW', 'MEDIUM_LOW', 'MEDIUM_HIGH', 'HIGH']:
            long_r, short_r, wait_r, desc = matrix_display[regime][lppl_level]
            print(f"{regime_name:<12} {lppl_level:<12} {long_r*100:>4.0f}% {short_r*100:>4.0f}% {wait_r*100:>4.0f}%   {desc}")

    # 运行回测
    print("\n" + "="*100)
    print("回测执行...")
    print("="*100)

    result = backtest_dynamic_allocation(df, lppl_map)

    print(f"\n回测结果:")
    print(f"  最终权益: {result['capital']:,.0f}元")
    print(f"  收益率: {result['return']:.2f}%")
    print(f"  交易次数: {result['trades']}笔")
    print(f"  胜率: {result['win_rate']:.1f}%")

    print(f"\n  做多: {result['long_trades']}笔, 盈亏: {result['long_pnl']:,.0f}元")
    print(f"  做空: {result['short_trades']}笔, 盈亏: {result['short_pnl']:,.0f}元")

    # 配置使用统计
    print("\n" + "="*100)
    print("配置使用统计")
    print("="*100)

    for alloc_key, count in sorted(result['allocation_stats'].items(), key=lambda x: x[1], reverse=True):
        pct = count / sum(result['allocation_stats'].values()) * 100
        print(f"  {alloc_key}: {count}次 ({pct:.1f}%)")

    # 详细分析
    print("\n" + "="*100)
    print("分配置盈亏分析")
    print("="*100)

    regime_analysis = {}

    for trade in result['trade_list']:
        key = f"{trade['regime']}_{trade['lppl_level']}"
        if key not in regime_analysis:
            regime_analysis[key] = {'count': 0, 'pnl': 0}
        regime_analysis[key]['count'] += 1
        regime_analysis[key]['pnl'] += trade['pnl']

    print(f"\n{'配置':<20} {'交易数':<8} {'总盈亏':<12} {'平均盈亏':<12}")
    print("-"*100)

    for key, data in sorted(regime_analysis.items(), key=lambda x: x[1]['pnl'], reverse=True):
        regime_name = {'BULL': '牛市', 'BEAR': '熊市', 'RANGING': '震荡'}[key.split('_')[0]]
        lppl_name = key.split('_')[1]
        avg_pnl = data['pnl'] / data['count'] if data['count'] > 0 else 0
        print(f"{regime_name}+{lppl_name:<10} {data['count']:>6}笔   {data['pnl']:>10,.0f}元   {avg_pnl:>10,.0f}元")

    # 对比基准
    print("\n" + "="*100)
    print("与基准策略对比")
    print("="*100)

    print(f"\n{'策略':<30} {'收益率':<12} {'交易次数':<10} {'说明':<30}")
    print("-"*100)

    baseline_return = 803.75
    lppl_bidirectional_return = 212.47

    print(f"{'纯做多（基准）':<30} {baseline_return:>10.2f}%   {108:>8}笔   追求高收益")
    print(f"{'LPPL双向（固定配置）':<30} {lppl_bidirectional_return:>10.2f}%   {63:>8}笔   稳健收益")
    print(f"{'动态配置（当前）':<30} {result['return']:>10.2f}%   {result['trades']:>8}笔   趋势+LPPL双层决策")

    # 保存交易明细
    output_file = 'backtest_dynamic_allocation_trades.csv'
    trades_df = pd.DataFrame(result['trade_list'])
    trades_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n交易明细已保存: {output_file}")

    print("\n" + "="*100)
    print("回测完成")
    print("="*100)


if __name__ == "__main__":
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    main()
