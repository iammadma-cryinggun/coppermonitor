# -*- coding: utf-8 -*-
"""
对比LPPL置信度指标和回测交易结果
分析LPPL泡沫信号对交易策略的过滤效果
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys
import os
import ast
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from futures_monitor import calculate_indicators


def parse_np_float64(value):
    """解析np.float64()格式的值"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # 提取np.float64(123.456)中的数字
        match = re.search(r'np\.float64\(([^)]+)\)', value)
        if match:
            try:
                return float(match.group(1))
            except:
                pass
        # 处理'nan'字符串
        if value.lower() == 'nan':
            return np.nan
    return None

# 最优参数
BEST_PARAMS = {
    'EMA_FAST': 3,
    'EMA_SLOW': 15,
    'RSI_FILTER': 30,
    'RATIO_TRIGGER': 1.05,
    'STC_SELL_ZONE': 65,
    'STOP_LOSS_PCT': 0.02
}

# 初始资金
INITIAL_CAPITAL = 100000
MAX_POSITION_RATIO = 0.8
STOP_LOSS_PCT = 0.02
CONTRACT_SIZE = 1
MARGIN_RATE = 0.13


def extract_lppl_indicators(csv_file):
    """从LPPL指标CSV中提取数据"""
    print("读取LPPL指标数据...")

    df = pd.read_csv(csv_file)
    # LPPL的time是ordinal格式，使用fromordinal方法转换
    df['time'] = df['time'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))

    # 解析_fits列，提取泡沫指标
    bubble_indicators = []

    # 使用正则表达式直接从_fits字符串中提取数据
    for idx, row in df.iterrows():
        try:
            fits_str = row['_fits']

            # 使用eval（在受控环境下）来解析
            # 先替换np.float64为float，使字符串可被eval解析
            fits_str_clean = re.sub(r"np\.float64\(([^)]+)\)", r"float(\1)", fits_str)
            fits_str_clean = fits_str_clean.replace('nan', 'np.nan')
            fits = eval(fits_str_clean)

            # 使用更宽松的条件：只要D值在0.2-1.0之间就认为是泡沫
            # 放宽条件是因为is_qualified可能都是False
            valid_fits = []
            for f in fits:
                d_val = parse_np_float64(f.get('D'))
                if d_val is not None and 0.2 < d_val < 1.0 and not pd.isna(d_val):
                    f['D_parsed'] = d_val
                    valid_fits.append(f)

            if valid_fits:
                # 取D值最大的拟合（泡沫程度最高）
                best_fit = max(valid_fits, key=lambda x: x['D_parsed'])
                bubble_indicators.append({
                    'date': row['time'],
                    'price': row['price'],
                    'D': best_fit['D_parsed'],
                    'tc': pd.Timestamp.fromordinal(int(best_fit['tc'])),
                    'm': parse_np_float64(best_fit.get('m', 0)),
                    'w': parse_np_float64(best_fit.get('w', 0)),
                    'is_qualified': best_fit.get('is_qualified', False)
                })
        except Exception as e:
            continue

    lppl_df = pd.DataFrame(bubble_indicators)
    print(f"  找到 {len(lppl_df)} 天有泡沫信号")

    return lppl_df


def backtest_with_lppl_filter(df, lppl_df, enable_filter=False):
    """回测 - 加入LPPL泡沫过滤"""
    # 计算指标
    df = calculate_indicators(df.copy(), BEST_PARAMS)

    # 创建LPPL泡沫日期集合
    bubble_dates = set(lppl_df['date'].dt.date)

    # 回测
    capital = INITIAL_CAPITAL
    position = None
    trades = []
    filtered_count = 0

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
                    'lppl_filter': position['lppl_filter']
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
                # 【新增】LPPL泡沫过滤
                current_date = current['datetime'].date()
                is_bubble = current_date in bubble_dates

                lppl_filter = False
                if enable_filter and is_bubble:
                    filtered_count += 1
                    lppl_filter = True
                    # 可选择：泡沫时不做多，或者减少仓位
                    continue  # 过滤掉泡沫信号

                # 计算可开仓手数
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
                        'stop_loss': stop_loss_price,
                        'lppl_filter': lppl_filter
                    }

    # 计算统计数据
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0

    return {
        'capital': capital,
        'return': total_return,
        'trades': len(trades),
        'filtered': filtered_count,
        'win_rate': win_rate,
        'win_trades': len(win_trades),
        'trade_list': trades
    }


def main():
    print("="*80)
    print("LPPL泡沫信号与回测策略对比分析")
    print("="*80)

    # 1. 读取数据
    print("\n1. 读取数据")
    df = pd.read_csv("SN_沪锡_日线_10年.csv")
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
    print(f"  数据量: {len(df)} 条")

    # 2. 提取LPPL指标
    lppl_df = extract_lppl_indicators('SN_lppl_indicators.csv')

    if len(lppl_df) == 0:
        print("\n[ERROR] 未找到LPPL泡沫信号数据")
        return

    print(f"\n  LPPL泡沫信号分布:")
    print(f"    D值 0.3-0.5: {len(lppl_df[(lppl_df['D'] > 0.3) & (lppl_df['D'] < 0.5)])} 天")
    print(f"    D值 0.5-0.8: {len(lppl_df[(lppl_df['D'] > 0.5) & (lppl_df['D'] < 0.8)])} 天")
    print(f"    D值 >0.8: {len(lppl_df[lppl_df['D'] > 0.8])} 天")

    # 3. 回测 - 无LPPL过滤（基准）
    print("\n2. 回测 - 无LPPL过滤（基准）")
    result_baseline = backtest_with_lppl_filter(df, lppl_df, enable_filter=False)

    print(f"  最终权益: {result_baseline['capital']:,.0f} 元")
    print(f"  收益率: {result_baseline['return']:.2f}%")
    print(f"  交易次数: {result_baseline['trades']}笔")
    print(f"  胜率: {result_baseline['win_rate']:.1f}%")

    # 4. 回测 - 加入LPPL泡沫过滤
    print("\n3. 回测 - 加入LPPL泡沫过滤")
    result_filtered = backtest_with_lppl_filter(df, lppl_df, enable_filter=True)

    print(f"  最终权益: {result_filtered['capital']:,.0f} 元")
    print(f"  收益率: {result_filtered['return']:.2f}%")
    print(f"  交易次数: {result_filtered['trades']}笔")
    print(f"  过滤信号: {result_filtered['filtered']}笔")
    print(f"  胜率: {result_filtered['win_rate']:.1f}%")

    # 5. 对比分析
    print("\n" + "="*80)
    print("对比分析")
    print("="*80)
    print(f"\n{'指标':<20} {'无LPPL过滤':<15} {'LPPL过滤':<15} {'变化'}")
    print("-"*80)
    print(f"{'收益率':<20} {result_baseline['return']:>10.2f}%       {result_filtered['return']:>10.2f}%       "
          f"{result_filtered['return']-result_baseline['return']:>8.2f}%")
    print(f"{'交易次数':<20} {result_baseline['trades']:>8}笔        {result_filtered['trades']:>8}笔        "
          f"{result_filtered['trades']-result_baseline['trades']:>8}")
    print(f"{'过滤信号':<20} {0:>8}笔        {result_filtered['filtered']:>8}笔        -")
    print(f"{'胜率':<20} {result_baseline['win_rate']:>8.1f}%       {result_filtered['win_rate']:>8.1f}%       "
          f"{result_filtered['win_rate']-result_baseline['win_rate']:>8.1f}%")

    # 6. 结论
    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if result_filtered['return'] > result_baseline['return']:
        improvement = result_filtered['return'] - result_baseline['return']
        print(f"[OK] LPPL泡沫过滤提升了收益 {improvement:.2f}%")
        print(f"  过滤了{result_filtered['filtered']}个泡沫信号")
    else:
        decline = result_baseline['return'] - result_filtered['return']
        print(f"[X] LPPL泡沫过滤降低了收益 {decline:.2f}%")
        print(f"  可能原因：LPPL信号与策略信号不匹配，或过度过滤")

    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    main()
