# -*- coding: utf-8 -*-
"""
完整回测：GHE+LPPL过滤器 + EMA+STC策略
验证大周期预判的实际效果
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

from futures_monitor import calculate_indicators
from scipy.optimize import curve_fit

# ============== GHE和LPPL计算 ==============
def calculate_ghe_single_window(series, q=2, max_tau=20):
    """计算GHE"""
    log_price = np.log(series.values)
    taus = np.arange(2, max_tau + 1)
    K_q = []
    for tau in taus:
        diff = np.abs(log_price[tau:] - log_price[:-tau])
        K_q.append(np.mean(diff ** q))
    log_taus = np.log(taus)
    log_K_q = np.log(K_q)
    slope, _ = np.polyfit(log_taus, log_K_q, 1)
    H_q = slope / q
    return H_q


def lppl_func(t, A, B, tc, m, C, omega, phi):
    """LPPL核心函数"""
    dt = np.clip(tc - t, 1e-8, None)
    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))


def calculate_lppl_d(prices):
    """计算LPPL-D值"""
    if len(prices) < 200:
        return None

    log_prices = np.log(prices.values if hasattr(prices, 'values') else prices)
    t = np.arange(len(log_prices))
    last_t = t[-1]

    A0 = log_prices.max()
    B0 = -0.1
    tc0 = last_t + 30
    m0 = 0.5
    C0 = 0.1
    omega0 = 10.0
    phi0 = 0.0
    p0 = [A0, B0, tc0, m0, C0, omega0, phi0]

    bounds = (
        [-np.inf, -np.inf, last_t, 0.1, -1, 6, -2*np.pi],
        [np.inf, 0, last_t + 90, 0.9, 1, 13, 2*np.pi]
    )

    try:
        popt, pcov = curve_fit(lppl_func, t, log_prices, p0=p0,
                               bounds=bounds, maxfev=5000, method='trf')
        A, B, tc, m, C, omega, phi = popt

        if not (0.1 < m < 0.9):
            return None
        if not (6 < omega < 13):
            return None
        if B >= 0:
            return None

        D = m * omega / (2 * np.pi)
        return D

    except:
        return None


def get_ghe_lppl_signal(ghe, lppl_d):
    """
    获取GHE+LPPL信号

    返回:
        'ALLOW_LONG': 允许做多
        'BLOCK_STRONG_DOWN': 强下降，禁止做多
        'BLOCK_RANGING': 横盘，禁止做多
        'NO_SIGNAL': 无明确信号
    """
    if ghe is None or lppl_d is None:
        return 'NO_SIGNAL'

    # 强下降信号（52.4%概率）
    if 0.35 <= ghe <= 0.45 and 0.4 <= lppl_d <= 0.6:
        return 'BLOCK_STRONG_DOWN'

    # 横盘信号（62.5%概率）
    if 0.35 <= ghe <= 0.45 and 0.7 <= lppl_d <= 0.9:
        return 'BLOCK_RANGING'

    # 强上升信号（100%概率）
    if 0.50 <= ghe <= 0.60 and 0.7 <= lppl_d <= 0.9:
        return 'ALLOW_LONG'

    return 'NO_SIGNAL'


# ============== 回测引擎 ==============
def backtest_strategy(df, params, use_ghe_lppl_filter=False, filter_mode='basic'):
    """
    回测策略

    参数:
        df: 原始数据
        params: EMA+STC参数
        use_ghe_lppl_filter: 是否使用GHE+LPPL过滤
        filter_mode: 过滤模式
            - 'basic': 只过滤强下降
            - 'strict': 只在强上升时做
            - 'smart': 智能模式（强上升全仓，其他半仓）
    """
    # 计算EMA+STC指标
    df_work = df.copy()
    df_work = calculate_indicators(df_work, params)

    # 将结果合并回原DataFrame（保持列名一致）
    for col in df_work.columns:
        if col not in df.columns:
            df[col] = df_work[col]

    # 计算GHE和LPPL-D
    df['GHE'] = np.nan
    df['LPPL_D'] = np.nan

    for i in range(200, len(df)):
        # GHE
        try:
            ghe = calculate_ghe_single_window(df['close'].iloc[i-100:i])
            df.loc[df.index[i], 'GHE'] = ghe
        except:
            pass

        # LPPL-D
        try:
            lppl_d = calculate_lppl_d(df['close'].iloc[i-200:i])
            df.loc[df.index[i], 'LPPL_D'] = lppl_d
        except:
            pass

    # 前向填充
    df['GHE'] = df['GHE'].ffill()
    df['LPPL_D'] = df['LPPL_D'].ffill()

    # 模拟交易
    capital = 100000
    position = None
    trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1] if i > 0 else current

        # 平仓逻辑
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 止损
            if current['low'] <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_triggered = True
                exit_reason = '止损'

            # STC止盈
            elif (prev['stc'] > params['STC_SELL_ZONE'] and
                  current['stc'] < prev['stc']):
                exit_price = current['close']
                exit_triggered = True
                exit_reason = 'STC止盈'

            # 趋势反转
            elif current['ema_fast'] < current['ema_slow']:
                exit_price = current['close']
                exit_triggered = True
                exit_reason = '趋势反转'

            if exit_triggered:
                pnl = (exit_price - position['entry_price']) * position['contracts']
                capital += pnl

                trades.append({
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': exit_reason,
                    'ghe_at_entry': position.get('ghe', None),
                    'lppl_at_entry': position.get('lppl', None),
                    'ghe_lppl_signal': position.get('ghe_lppl_signal', None)
                })

                position = None
                continue

        # 开仓逻辑
        if position is None:
            # 基础信号
            trend_up = current['ema_fast'] > current['ema_slow']
            ratio_safe = 0 < current['ratio'] < params['RATIO_TRIGGER']
            ratio_shrinking = current['ratio'] < current['ratio_prev']
            turning_up = current['macd_dif'] > prev['macd_dif']
            is_strong = current['rsi'] > params['RSI_FILTER']

            sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
            ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])
            buy_signal = sniper_signal or (ema_cross and is_strong)

            if buy_signal:
                # GHE+LPPL过滤
                if use_ghe_lppl_filter:
                    ghe = current['GHE']
                    lppl = current['LPPL_D']
                    ghe_lppl_signal = get_ghe_lppl_signal(ghe, lppl)

                    # 根据过滤模式决定是否开仓
                    allow_trade = False
                    position_size = 1.0

                    if filter_mode == 'basic':
                        # 只过滤强下降和横盘
                        if ghe_lppl_signal in ['ALLOW_LONG', 'NO_SIGNAL']:
                            allow_trade = True
                        elif ghe_lppl_signal == 'BLOCK_STRONG_DOWN':
                            allow_trade = False  # 强下降，禁止做多
                        elif ghe_lppl_signal == 'BLOCK_RANGING':
                            allow_trade = False  # 横盘，禁止做多

                    elif filter_mode == 'strict':
                        # 只在强上升时做
                        if ghe_lppl_signal == 'ALLOW_LONG':
                            allow_trade = True
                            position_size = 1.0
                        else:
                            allow_trade = False

                    elif filter_mode == 'smart':
                        # 智能模式
                        if ghe_lppl_signal == 'ALLOW_LONG':
                            allow_trade = True
                            position_size = 1.0  # 全仓
                        elif ghe_lppl_signal == 'NO_SIGNAL':
                            allow_trade = True
                            position_size = 0.5  # 半仓
                        else:
                            allow_trade = False

                    if not allow_trade:
                        continue  # 跳过这次信号

                else:
                    ghe_lppl_signal = None
                    position_size = 1.0

                # 计算仓位
                entry_price = current['close']
                margin_per_contract = entry_price * 1 * 0.13
                available_capital = capital * 0.8 * position_size
                max_contracts = int(available_capital / margin_per_contract)

                if max_contracts > 0:
                    stop_loss_price = entry_price * 0.98
                    position = {
                        'entry_datetime': current['datetime'],
                        'entry_price': entry_price,
                        'contracts': max_contracts,
                        'stop_loss': stop_loss_price,
                        'ghe': current.get('GHE', None),
                        'lppl': current.get('LPPL_D', None),
                        'ghe_lppl_signal': ghe_lppl_signal,
                        'position_size': position_size
                    }

    return {
        'capital': capital,
        'trades': trades,
        'total_return_pct': (capital - 100000) / 100000 * 100,
        'trade_count': len(trades)
    }


# ============== 主测试 ==============
def main():
    print("="*80)
    print("GHE+LPPL过滤器回测验证")
    print("="*80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 重命名列（适配calculate_indicators）
    df = df.rename(columns={
        '收盘价': 'close',
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '成交量': 'volume',
        '持仓量': 'hold'
    })

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 天")

    # 最佳参数（从之前优化得到）
    BEST_PARAMS = {
        'EMA_FAST': 3,
        'EMA_SLOW': 15,
        'RSI_FILTER': 30,
        'RATIO_TRIGGER': 1.05,
        'STC_SELL_ZONE': 65,
        'STOP_LOSS_PCT': 0.02
    }

    print(f"\n使用参数: EMA({BEST_PARAMS['EMA_FAST']},{BEST_PARAMS['EMA_SLOW']}), "
          f"RSI={BEST_PARAMS['RSI_FILTER']}, RATIO={BEST_PARAMS['RATIO_TRIGGER']}, "
          f"STC={BEST_PARAMS['STC_SELL_ZONE']}")

    # 测试三种模式
    results = {}

    print("\n" + "="*80)
    print("【回测1】原始策略（无GHE+LPPL过滤）")
    print("="*80)

    result1 = backtest_strategy(df, BEST_PARAMS, use_ghe_lppl_filter=False)
    results['original'] = result1

    if result1['trade_count'] > 0:
        wins = [t for t in result1['trades'] if t['pnl'] > 0]
        win_rate = len(wins) / result1['trade_count'] * 100

        print(f"\n交易数: {result1['trade_count']}笔")
        print(f"胜率: {win_rate:.1f}%")
        print(f"总收益: {result1['total_return_pct']:.2f}%")
        print(f"最终资金: {result1['capital']:,.0f}元")
    else:
        print("\n无交易")

    print("\n" + "="*80)
    print("【回测2】GHE+LPPL基础过滤（排除强下降和横盘）")
    print("="*80)

    result2 = backtest_strategy(df, BEST_PARAMS, use_ghe_lppl_filter=True, filter_mode='basic')
    results['basic_filter'] = result2

    if result2['trade_count'] > 0:
        wins = [t for t in result2['trades'] if t['pnl'] > 0]
        win_rate = len(wins) / result2['trade_count'] * 100

        print(f"\n交易数: {result2['trade_count']}笔")
        print(f"胜率: {win_rate:.1f}%")
        print(f"总收益: {result2['total_return_pct']:.2f}%")
        print(f"最终资金: {result2['capital']:,.0f}元")

        # 统计被过滤的信号类型
        blocked = [t for t in result2['trades'] if t.get('ghe_lppl_signal') in ['BLOCK_STRONG_DOWN', 'BLOCK_RANGING']]
        if len(blocked) > 0:
            print(f"\n被过滤的交易: {len(blocked)}笔")
            print(f"  强下降信号: {len([t for t in blocked if t['ghe_lppl_signal']=='BLOCK_STRONG_DOWN'])}笔")
            print(f"  横盘信号: {len([t for t in blocked if t['ghe_lppl_signal']=='BLOCK_RANGING'])}笔")
    else:
        print("\n无交易")

    print("\n" + "="*80)
    print("【回测3】GHE+LPPL严格模式（只在强上升时做）")
    print("="*80)

    result3 = backtest_strategy(df, BEST_PARAMS, use_ghe_lppl_filter=True, filter_mode='strict')
    results['strict_filter'] = result3

    if result3['trade_count'] > 0:
        wins = [t for t in result3['trades'] if t['pnl'] > 0]
        win_rate = len(wins) / result3['trade_count'] * 100

        print(f"\n交易数: {result3['trade_count']}笔")
        print(f"胜率: {win_rate:.1f}%")
        print(f"总收益: {result3['total_return_pct']:.2f}%")
        print(f"最终资金: {result3['capital']:,.0f}元")

        # 统计信号类型
        allowed = [t for t in result3['trades'] if t.get('ghe_lppl_signal') == 'ALLOW_LONG']
        print(f"\n强上升信号交易: {len(allowed)}笔")
    else:
        print("\n无交易")

    print("\n" + "="*80)
    print("【回测4】GHE+LPPL智能模式（强上升全仓，其他半仓）")
    print("="*80)

    result4 = backtest_strategy(df, BEST_PARAMS, use_ghe_lppl_filter=True, filter_mode='smart')
    results['smart_filter'] = result4

    if result4['trade_count'] > 0:
        wins = [t for t in result4['trades'] if t['pnl'] > 0]
        win_rate = len(wins) / result4['trade_count'] * 100

        print(f"\n交易数: {result4['trade_count']}笔")
        print(f"胜率: {win_rate:.1f}%")
        print(f"总收益: {result4['total_return_pct']:.2f}%")
        print(f"最终资金: {result4['capital']:,.0f}元")

        # 统计仓位使用
        full_positions = [t for t in result4['trades'] if t.get('position_size', 0) == 1.0]
        half_positions = [t for t in result4['trades'] if t.get('position_size', 0) == 0.5]
        print(f"\n全仓交易: {len(full_positions)}笔")
        print(f"半仓交易: {len(half_positions)}笔")
    else:
        print("\n无交易")

    # ============== 对比总结 ==============
    print("\n" + "="*80)
    print("【结果对比】")
    print("="*80)

    print(f"\n{'策略':<25} {'交易数':<8} {'胜率':<10} {'收益率':<12} {'最终资金':<15}")
    print("-"*80)

    modes = [
        ('原始策略（无过滤）', 'original'),
        ('基础过滤（排除下降/横盘）', 'basic_filter'),
        ('严格模式（只做强上升）', 'strict_filter'),
        ('智能模式（动态仓位）', 'smart_filter')
    ]

    for name, key in modes:
        r = results[key]
        if r['trade_count'] > 0:
            wins = [t for t in r['trades'] if t['pnl'] > 0]
            win_rate = len(wins) / r['trade_count'] * 100
            print(f"{name:<25} {r['trade_count']:<8} {win_rate:<10.1f}% "
                  f"{r['total_return_pct']:<12.2f}% {r['capital']:>14,.0f}元")
        else:
            print(f"{name:<25} {'无交易':<8} {'N/A':<10} {'N/A':<12} {'N/A':<15}")

    # 保存详细结果
    output = {
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_range': f"{df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}",
        'params': BEST_PARAMS,
        'results': {
            'original': {
                'trade_count': results['original']['trade_count'],
                'total_return_pct': results['original']['total_return_pct'],
                'final_capital': results['original']['capital']
            },
            'basic_filter': {
                'trade_count': results['basic_filter']['trade_count'],
                'total_return_pct': results['basic_filter']['total_return_pct'],
                'final_capital': results['basic_filter']['capital']
            },
            'strict_filter': {
                'trade_count': results['strict_filter']['trade_count'],
                'total_return_pct': results['strict_filter']['total_return_pct'],
                'final_capital': results['strict_filter']['capital']
            },
            'smart_filter': {
                'trade_count': results['smart_filter']['trade_count'],
                'total_return_pct': results['smart_filter']['total_return_pct'],
                'final_capital': results['smart_filter']['capital']
            }
        }
    }

    with open('ghe_lppl_filter_backtest_result.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存: ghe_lppl_filter_backtest_result.json")

    # 最终结论
    print("\n" + "="*80)
    print("【最终结论】")
    print("="*80)

    if results['original']['trade_count'] > 0 and results['basic_filter']['trade_count'] > 0:
        original_return = results['original']['total_return_pct']
        basic_return = results['basic_filter']['total_return_pct']
        strict_return = results['strict_filter']['total_return_pct']

        print(f"\n1. 原始策略收益率: {original_return:.2f}%")
        print(f"2. 基础过滤收益率: {basic_return:.2f}% (提升{basic_return-original_return:+.2f}%)")
        print(f"3. 严格模式收益率: {strict_return:.2f}% (提升{strict_return-original_return:+.2f}%)")

        if basic_return > original_return:
            print(f"\n[结论] GHE+LPPL基础过滤有效！收益率提升{basic_return-original_return:+.2f}个百分点")
        elif strict_return > original_return:
            print(f"\n[结论] GHE+LPPL严格模式有效！收益率提升{strict_return-original_return:+.2f}个百分点")
        else:
            print(f"\n[结论] GHE+LPPL过滤在本次回测中未改善收益，需要调整策略")

    print("\n" + "="*80)
    print("回测完成")
    print("="*80)


if __name__ == "__main__":
    main()
