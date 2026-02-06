# -*- coding: utf-8 -*-
"""
分析失败交易与LPPL预警的关系
"""

import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_monitor import calculate_indicators
import ast
import re

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


def parse_lppl_d(fits_str):
    """解析LPPL D值"""
    try:
        fits_str_clean = re.sub(r"np\.float64\(([^)]+)\)", r"float(\1)", fits_str)
        fits_str_clean = fits_str_clean.replace('nan', 'np.nan')
        fits = eval(fits_str_clean)

        d_values = []
        for f in fits:
            d_val = f.get('D')
            if d_val is not None and not pd.isna(d_val):
                d_values.append(d_val)

        if d_values:
            return max(d_values)
    except:
        pass
    return None


def backtest_with_lppl_tracking(df):
    """回测并记录LPPL状态"""
    df = calculate_indicators(df.copy(), BEST_PARAMS)

    capital = INITIAL_CAPITAL
    position = None
    trades = []

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
                    'reason': exit_reason
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
                entry_price = current['close']
                margin_per_contract = entry_price * CONTRACT_SIZE * MARGIN_RATE
                available_capital = capital * MAX_POSITION_RATIO
                max_contracts = int(available_capital / margin_per_contract)

                if max_contracts > 0:
                    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
                    position = {
                        'entry_datetime': df.iloc[i]['datetime'],
                        'entry_price': entry_price,
                        'contracts': max_contracts,
                        'stop_loss': stop_loss_price
                    }

    return trades


def main():
    print('='*100)
    print('失败交易 vs LPPL预警分析')
    print('='*100)

    # 读取数据
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

    # 读取LPPL指标
    lppl_indicators = pd.read_csv('SN_lppl_indicators.csv')
    lppl_indicators['time'] = pd.to_datetime(lppl_indicators['time'])
    lppl_indicators['D_max'] = lppl_indicators['_fits'].apply(parse_lppl_d)

    # 回测
    trades = backtest_with_lppl_tracking(df)

    # 分析失败交易
    losing_trades = [t for t in trades if t['pnl'] <= 0]

    print(f'\n总交易: {len(trades)}笔')
    print(f'盈利交易: {len(trades) - len(losing_trades)}笔')
    print(f'亏损交易: {len(losing_trades)}笔')

    # 分析每笔亏损交易
    print(f'\n{"序号":<4} {"入场日期":<12} {"出场日期":<12} {"亏损%":<8} {"亏损额":<12} {"LPPL_D值":<10} {"风险等级":<12}')
    print('-'*100)

    high_d_count = 0
    medium_d_count = 0
    low_d_count = 0
    no_d_count = 0

    high_d_loss = 0
    medium_d_loss = 0
    low_d_loss = 0

    for i, trade in enumerate(losing_trades, 1):
        entry_date = trade['entry_datetime']

        # 找到当天的LPPL D值
        lppl_row = lppl_indicators[lppl_indicators['time'] == entry_date]

        if len(lppl_row) > 0 and pd.notna(lppl_row.iloc[0]['D_max']):
            d_val = lppl_row.iloc[0]['D_max']

            if d_val >= 0.5:
                risk_level = '[HIGH]高风险[泡沫]'
                high_d_count += 1
                high_d_loss += trade['pnl']
            elif d_val >= 0.3:
                risk_level = '[MED]中风险[预警]'
                medium_d_count += 1
                medium_d_loss += trade['pnl']
            else:
                risk_level = '[LOW]低风险'
                low_d_count += 1
                low_d_loss += trade['pnl']

            d_str = f'{d_val:.3f}'
        else:
            d_str = 'N/A'
            risk_level = '[N/A]无数据'
            no_d_count += 1

        pnl_str = f"{trade['pnl_pct']:.2f}%"
        loss_str = f"{trade['pnl']:,.0f}"
        print(f"{i:<4} {trade['entry_datetime'].date()} {trade['exit_datetime'].date()} {pnl_str:<8} {loss_str:<12} {d_str:<10} {risk_level}")

    print(f'\n' + '='*100)
    print('统计结果')
    print('='*100)

    total_loss = sum(t['pnl'] for t in losing_trades)

    print(f'\n亏损交易LPPL状态分布:')
    print(f'  [HIGH] 高风险(D>=0.5): {high_d_count}笔 ({high_d_count/len(losing_trades)*100:.1f}%), 亏损: {high_d_loss:,.0f}元')
    print(f'  [MED] 中风险(0.3<=D<0.5): {medium_d_count}笔 ({medium_d_count/len(losing_trades)*100:.1f}%), 亏损: {medium_d_loss:,.0f}元')
    print(f'  [LOW] 低风险(D<0.3): {low_d_count}笔 ({low_d_count/len(losing_trades)*100:.1f}%), 亏损: {low_d_loss:,.0f}元')
    print(f'  [N/A] 无数据: {no_d_count}笔 ({no_d_count/len(losing_trades)*100:.1f}%)')

    avoidable = high_d_count + medium_d_count
    avoidable_loss = high_d_loss + medium_d_loss

    print(f'\n' + '='*100)
    print('LPPL过滤效果评估')
    print('='*100)
    print(f'\n如果使用LPPL过滤(D>=0.3时不开仓):')
    print(f'  [OK] 可避免亏损: {avoidable}笔 ({avoidable/len(losing_trades)*100:.1f}%的亏损交易)')
    print(f'  [OK] 减少亏损: {abs(avoidable_loss):,.0f}元 (占总亏损{abs(avoidable_loss)/abs(total_loss)*100:.1f}%)')
    print(f'  [!] 错过机会: 可能错过部分高风险区的盈利交易')

    print(f'\n建议策略:')
    if avoidable >= len(losing_trades) * 0.5:
        print(f'  [*] 强烈推荐: LPPL过滤可以避免一半以上的亏损交易')
        print(f'  [*] 建议规则: D>=0.3时不开新仓，D>=0.5时只平仓不开仓')
    else:
        print(f'  ⚠ LPPL过滤效果有限，需要结合其他指标')

    print('\n' + '='*100)


if __name__ == "__main__":
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    main()
