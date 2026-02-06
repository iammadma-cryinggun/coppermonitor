# -*- coding: utf-8 -*-
"""
趋势识别系统 - 使用LPPL识别市场状态，然后选择策略
LPPL不作为交易信号，而是作为市场环境判断工具
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


def analyze_lppl_for_regime(lppl_map, lookback_days=60):
    """
    分析LPPL D百分位来识别市场状态

    使用D百分位（相对排名）而不是D绝对值
    """
    if len(lppl_map) == 0:
        return 'RANGING', 50

    # 获取最近的LPPL数据（使用D百分位）
    recent_dates = sorted(lppl_map.keys())[-lookback_days:]
    recent_d_pct = [lppl_map[d]['D_percentile'] for d in recent_dates if d in lppl_map]

    if len(recent_d_pct) < 5:
        return 'RANGING', 50

    # 分析D百分位分布
    avg_d_pct = np.mean(recent_d_pct)
    std_d_pct = np.std(recent_d_pct)
    max_d_pct = np.max(recent_d_pct)
    min_d_pct = np.min(recent_d_pct)

    # 计算各区间的占比
    high_d_ratio = sum(1 for d in recent_d_pct if d >= 0.8) / len(recent_d_pct)
    mid_high_ratio = sum(1 for d in recent_d_pct if 0.6 <= d < 0.8) / len(recent_d_pct)
    mid_low_ratio = sum(1 for d in recent_d_pct if 0.4 <= d < 0.6) / len(recent_d_pct)
    low_d_ratio = sum(1 for d in recent_d_pct if d < 0.4) / len(recent_d_pct)

    # 市场状态判断逻辑（基于D百分位）
    # 高D百分位(>0.7) → 泡沫多，可能是牛市后期或熊市初期
    # 低D百分位(<0.3) → 无泡沫，可能是熊市或牛市初期
    # 中D百分位 → 震荡市

    if avg_d_pct >= 0.7:
        # D百分位普遍偏高 → 市场频繁出现泡沫特征
        regime = 'BULL'
        confidence = 50 + avg_d_pct * 40
    elif avg_d_pct <= 0.3:
        # D百分位普遍偏低 → 市场缺乏泡沫，可能是熊市
        regime = 'BEAR'
        confidence = 50 + (0.3 - avg_d_pct) * 50
    else:
        # D百分位在中等范围 → 震荡市
        regime = 'RANGING'
        confidence = 50

    confidence = max(50, min(100, confidence))

    return regime, confidence


def backtest_long_only(df):
    """纯做多策略（牛市时使用）"""
    df = calculate_indicators(df.copy(), BEST_PARAMS)

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
                    'type': 'LONG',
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'reason': exit_reason
                })

                position = None
                continue

        # 开仓（只做多）
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
    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0

    return {
        'capital': capital,
        'return': total_return,
        'trades': len(trades),
        'win_rate': win_rate,
        'trade_list': trades
    }


def backtest_lppl_bidirectional(df, lppl_map):
    """LPPL双向策略（震荡市/熊市时使用）"""
    df = calculate_indicators(df.copy(), BEST_PARAMS)

    capital = INITIAL_CAPITAL
    long_position = None
    short_position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 获取LPPL信号
        prior_dates = [d for d in lppl_map.keys() if d <= current['datetime']]
        if prior_dates:
            latest_lppl = lppl_map[max(prior_dates)]
            d_pct = latest_lppl['D_percentile']
        else:
            d_pct = 0.5  # 默认值

        # 根据LPPL决定做多还是做空
        go_long = d_pct <= 0.2
        go_short = d_pct >= 0.8

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
                    'reason': exit_reason
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
                    'reason': exit_reason
                })

                short_position = None

        # 开仓
        if long_position is None and short_position is None:
            # 做多信号
            if go_long:
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
                        long_position = {
                            'entry_datetime': current['datetime'],
                            'entry_price': entry_price,
                            'contracts': max_contracts,
                            'stop_loss': stop_loss_price
                        }

            # 做空信号
            if go_short:
                trend_down = current['ema_fast'] < current['ema_slow']
                is_weak = current['rsi'] < (100 - BEST_PARAMS['RSI_FILTER'])

                sell_signal = trend_down and is_weak

                if sell_signal:
                    entry_price = current['close']
                    margin_per_contract = entry_price * CONTRACT_SIZE * MARGIN_RATE
                    available_capital = capital * MAX_POSITION_RATIO
                    max_contracts = int(available_capital / margin_per_contract)

                    if max_contracts > 0:
                        stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)
                        short_position = {
                            'entry_datetime': current['datetime'],
                            'entry_price': entry_price,
                            'contracts': max_contracts,
                            'stop_loss': stop_loss_price
                        }

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
        'trade_list': trades
    }


def backtest_with_regime_switching(df, lppl_map, rebalance_days=60):
    """
    基于LPPL识别市场状态并切换策略
    每60天重新评估市场状态，然后选择对应策略
    """
    total_capital = INITIAL_CAPITAL
    all_trades = []
    regime_history = []

    # 将数据分成多个时期，每个时期根据LPPL状态选择策略
    start_idx = 200
    period_length = rebalance_days

    current_capital = INITIAL_CAPITAL

    for period_start in range(start_idx, len(df), period_length):
        period_end = min(period_start + period_length, len(df))

        # 评估当前市场状态
        period_data = df.iloc[:period_end]
        regime, confidence = analyze_lppl_for_regime(lppl_map, lookback_days=60)

        regime_history.append({
            'start_date': df.iloc[period_start]['datetime'],
            'end_date': df.iloc[period_end-1]['datetime'] if period_end < len(df) else df.iloc[-1]['datetime'],
            'regime': regime,
            'confidence': confidence
        })

        # 根据市场状态选择策略
        period_df = df.iloc[period_start:period_end].copy()

        if regime == 'BULL':
            # 牛市：使用纯做多策略
            period_result = backtest_long_only(period_df)
            strategy_used = '纯做多'
        else:
            # 震荡市或熊市：使用LPPL双向策略
            period_result = backtest_lppl_bidirectional(period_df, lppl_map)
            strategy_used = 'LPPL双向'

        # 记录策略使用情况
        if period_result['trade_list']:
            for trade in period_result['trade_list']:
                trade['regime'] = regime
                trade['strategy'] = strategy_used
                trade['period_start'] = df.iloc[period_start]['datetime']
                trade['period_end'] = df.iloc[period_end-1]['datetime'] if period_end < len(df) else df.iloc[-1]['datetime']
                all_trades.append(trade)

        # 更新资金（简化处理，假设每期独立运作）
        # 实际应该累计，但为了简化先用独立期处理

    # 计算整体统计
    total_pnl = sum(t['pnl'] for t in all_trades)
    total_return = (total_pnl / INITIAL_CAPITAL) * 100
    win_trades = [t for t in all_trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(all_trades) * 100 if all_trades else 0

    return {
        'capital': INITIAL_CAPITAL + total_pnl,
        'return': total_return,
        'trades': len(all_trades),
        'win_rate': win_rate,
        'trade_list': all_trades,
        'regime_history': regime_history
    }


def main():
    print("="*100)
    print("趋势识别系统 - LPPL识别市场状态，动态切换策略")
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
    print(f"交易日总数: {len(df)}天")
    print(f"LPPL拟合: {len(lppl_map)}次")

    # 运行策略切换回测
    print("\n" + "="*100)
    print("执行策略切换回测（每60天评估一次市场状态）...")
    print("="*100)

    result = backtest_with_regime_switching(df, lppl_map, rebalance_days=60)

    print(f"\n回测结果:")
    print(f"  最终权益: {result['capital']:,.0f}元")
    print(f"  收益率: {result['return']:.2f}%")
    print(f"  交易次数: {result['trades']}笔")
    print(f"  胜率: {result['win_rate']:.1f}%")

    # 显示市场状态切换历史
    print(f"\n" + "="*100)
    print("市场状态切换历史")
    print("="*100)

    for i, period in enumerate(result['regime_history'][:20], 1):
        regime_name = {'BULL': '牛市', 'BEAR': '熊市', 'RANGING': '震荡市'}[period['regime']]
        strategy_name = '纯做多' if period['regime'] == 'BULL' else 'LPPL双向'
        print(f"  时期{i}: {period['start_date'].date()} ~ {period['end_date'].date()}")
        print(f"    市场状态: {regime_name} (置信度: {period['confidence']:.1f}%)")
        print(f"    使用策略: {strategy_name}")

    # 统计各状态下的表现
    print(f"\n" + "="*100)
    print("分市场状态表现")
    print("="*100)

    regime_stats = {}
    for trade in result['trade_list']:
        regime = trade['regime']
        if regime not in regime_stats:
            regime_stats[regime] = {'count': 0, 'pnl': 0}
        regime_stats[regime]['count'] += 1
        regime_stats[regime]['pnl'] += trade['pnl']

    for regime, stats in regime_stats.items():
        regime_name = {'BULL': '牛市', 'BEAR': '熊市', 'RANGING': '震荡市'}[regime]
        avg_pnl = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
        print(f"  {regime_name}:")
        print(f"    交易次数: {stats['count']}笔")
        print(f"    总盈亏: {stats['pnl']:,.0f}元")
        print(f"    平均每笔: {avg_pnl:,.0f}元")

    # 对比基准
    print(f"\n" + "="*100)
    print("与基准策略对比")
    print("="*100)

    print(f"\n{'策略':<30} {'收益率':<12} {'说明':<30}")
    print("-"*100)
    print(f"{'纯做多（固定）':<30} {803.75:>10.2f}%   牛市最优，震荡/熊市表现未知")
    print(f"{'LPPL双向（固定）':<30} {212.47:>10.2f}%   震荡/熊市稳健")
    print(f"{'趋势切换（LPPL识别）':<30} {result['return']:>10.2f}%   LPPL识别状态动态切换")

    print("\n" + "="*100)
    print("结论")
    print("="*100)

    if result['return'] > 212.47:
        print(f"\n[OK] 趋势切换策略有效！")
        print(f"  收益率({result['return']:.2f}%)超过固定LPPL双向(212.47%)")
    elif result['return'] > 100:
        print(f"\n[OK] 趋势切换策略有效")
        print(f"  收益率为正，策略可行")
    else:
        print(f"\n[!] 趋势切换策略效果不理想")
        print(f"  可能原因：LPPL识别市场状态的准确性有待提高")

    # 保存交易明细
    output_file = 'backtest_regime_switching_trades.csv'
    trades_df = pd.DataFrame(result['trade_list'])
    trades_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n交易明细已保存: {output_file}")

    print("\n" + "="*100)
    print("回测完成")
    print("="*100)


if __name__ == "__main__":
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    main()
