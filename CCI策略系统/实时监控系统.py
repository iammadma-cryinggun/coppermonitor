"""
CCI策略实时监控系统
==================
实时监控所有品种的CCI指标和交易信号
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import time

sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')
from cci_calculations import calculate_cci_tv, calculate_stc, f_normalize
from 最优参数配置 import OPTIMAL_PARAMS


def get_realtime_data(code, days=200):
    """获取实时行情数据"""
    try:
        import akshare as ak

        # 获取期货历史数据
        df = ak.futures_main_sina(symbol=code)

        # 处理数据
        df = df.iloc[:, :5]
        df.columns = ['date', 'open', 'high', 'low', 'close']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.tail(days)  # 只取最近200天
        df = df.dropna()

        return df

    except Exception as e:
        print(f"[ERROR] 获取数据失败 {code}: {e}")
        return None


def calculate_indicators(df, cci_length, ma_length):
    """计算CCI和STC指标"""
    try:
        # 计算CCI
        data = calculate_cci_tv(df, cci_length=cci_length, ma_length=ma_length)

        # 计算STC
        stc_raw = calculate_stc(data, length=10, fast=23, slow=50, aaa=0.5)
        data['stc'] = f_normalize(stc_raw, 20, 80)

        return data

    except Exception as e:
        print(f"[ERROR] 计算指标失败: {e}")
        return None


def check_signals(data, params):
    """检查交易信号"""
    if data is None or len(data) < 2:
        return None

    # 最新数据
    current = data.iloc[-1]
    previous = data.iloc[-2]

    cci = current['cci']
    cci_ma = current['cci_ma']
    stc = current['stc']

    prev_cci = previous['cci']
    prev_cci_ma = previous['cci_ma']

    # 获取日期
    try:
        date_str = current.name.strftime('%Y-%m-%d') if hasattr(current.name, 'strftime') else str(current.name)
    except:
        date_str = data.index[-1].strftime('%Y-%m-%d') if hasattr(data.index[-1], 'strftime') else str(data.index[-1])

    signals = {
        'cci': round(cci, 2),
        'cci_ma': round(cci_ma, 2),
        'stc': round(stc, 2),
        'close': current['close'],
        'date': date_str,
        'oversold_signal': False,
        'golden_cross_signal': False,
        'overbought_warning': False,
        'death_cross_warning': False
    }

    # 检查超卖开仓信号
    if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
        signals['oversold_signal'] = True

    # 检查金叉开仓信号
    if (prev_cci <= prev_cci_ma and cci > cci_ma and
        cci <= params['cci_cross_max'] and stc >= params['stc_cross']):
        signals['golden_cross_signal'] = True

    # 检查超买警告
    if cci > params['cci_overbought']:
        signals['overbought_warning'] = True

    # 检查死叉警告
    if prev_cci >= prev_cci_ma and cci < cci_ma:
        signals['death_cross_warning'] = True

    return signals


def format_signal_output(symbol, params, signals):
    """格式化信号输出"""
    if signals is None:
        return f"{symbol:<10} [无数据]"

    output = []
    output.append(f"\n{'='*100}")
    output.append(f"{symbol} - {signals['date']}".center(100))
    output.append(f"{'='*100}")

    # 基础信息
    output.append(f"\n当前价格: {signals['close']:.2f}")
    output.append(f"CCI: {signals['cci']:>8.2f}  |  CCI_MA: {signals['cci_ma']:>8.2f}  |  STC: {signals['stc']:>8.2f}")

    # 阈值参考
    output.append(f"\n阈值参数:")
    output.append(f"  超卖: {params['cci_oversold']}  |  超买: {params['cci_overbought']}  |  金叉上限: {params['cci_cross_max']}")
    output.append(f"  STC超卖: {params['stc_oversold']}  |  STC金叉: {params['stc_cross']}")

    # 交易信号
    output.append(f"\n交易信号:")
    has_signal = False

    if signals['oversold_signal']:
        output.append(f"  [买入] CCI超卖开仓信号! CCI={signals['cci']:.2f} < {params['cci_oversold']}, STC={signals['stc']:.2f}")
        has_signal = True

    if signals['golden_cross_signal']:
        output.append(f"  [买入] CCI金叉开仓信号! CCI上穿CCI_MA (金叉)")
        has_signal = True

    if signals['overbought_warning']:
        output.append(f"  [注意] CCI超买警告! CCI={signals['cci']:.2f} > {params['cci_overbought']}")
        has_signal = True

    if signals['death_cross_warning']:
        output.append(f"  [注意] CCI死叉警告! CCI下穿CCI_MA (死叉)")
        has_signal = True

    if not has_signal:
        output.append(f"  [观望] 无明确交易信号")

    # 历史表现
    output.append(f"\n历史回测表现:")
    output.append(f"  预期收益: {params['expected_return']:+.2f}%  |  最大回撤: {params['max_dd']:.2f}%  |  收益/回撤比: {params['return_dd_ratio']:.2f}")
    output.append(f"  胜率: {params['win_rate']:.1f}%")

    return '\n'.join(output)


def monitor_all_symbols(selected_symbols=None):
    """监控所有品种"""
    print("\n" + "="*120)
    print("CCI策略实时监控系统".center(120))
    print("="*120)
    print(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120))
    print("="*120)

    symbols_to_monitor = selected_symbols if selected_symbols else list(OPTIMAL_PARAMS.keys())

    results = []

    for i, symbol in enumerate(symbols_to_monitor, 1):
        print(f"\n[{i}/{len(symbols_to_monitor)}] 正在获取 {symbol} 数据...", end=" ")

        params = OPTIMAL_PARAMS[symbol]

        # 获取实时数据
        df = get_realtime_data(params['code'], days=200)

        if df is None:
            print("[失败]")
            continue

        print(f"[成功] {len(df)} 条数据")

        # 计算指标
        data = calculate_indicators(df, params['cci_length'], params['ma_length'])

        # 检查信号
        signals = check_signals(data, params)

        if signals:
            results.append((symbol, params, signals))

            # 输出详细信息
            print(format_signal_output(symbol, params, signals))

        # 避免请求过快
        time.sleep(1)

    # 汇总信号
    print("\n" + "="*120)
    print("信号汇总".center(120))
    print("="*120)

    buy_signals = []
    sell_signals = []
    watch_list = []

    for symbol, params, signals in results:
        if signals['oversold_signal'] or signals['golden_cross_signal']:
            buy_signals.append((symbol, signals))
        elif signals['overbought_warning'] or signals['death_cross_warning']:
            sell_signals.append((symbol, signals))
        else:
            watch_list.append((symbol, signals))

    if buy_signals:
        print(f"\n[买入信号] {len(buy_signals)} 个品种:")
        for symbol, sig in buy_signals:
            signal_type = []
            if sig['oversold_signal']:
                signal_type.append(f"超卖(CCI={sig['cci']:.2f})")
            if sig['golden_cross_signal']:
                signal_type.append("金叉")
            print(f"  {symbol:<10} {', '.join(signal_type)}")

    if sell_signals:
        print(f"\n[卖出/观望信号] {len(sell_signals)} 个品种:")
        for symbol, sig in sell_signals:
            signal_type = []
            if sig['overbought_warning']:
                signal_type.append(f"超买(CCI={sig['cci']:.2f})")
            if sig['death_cross_warning']:
                signal_type.append("死叉")
            print(f"  {symbol:<10} {', '.join(signal_type)}")

    if watch_list:
        print(f"\n[观望] {len(watch_list)} 个品种")
        print(f"  {', '.join([s[0] for s in watch_list])}")

    print("\n" + "="*120)
    print(f"监控完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120))
    print("="*120)


def quick_monitor(top_n=5):
    """快速监控表现最好的前N个品种"""
    print("\n" + "="*120)
    print(f"快速监控 - TOP {top_n} 品种".center(120))
    print("="*120)

    # 按收益率排序
    sorted_symbols = sorted(OPTIMAL_PARAMS.items(),
                           key=lambda x: x[1]['expected_return'],
                           reverse=True)

    top_symbols = [s[0] for s in sorted_symbols[:top_n]]

    monitor_all_symbols(top_symbols)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CCI策略实时监控系统')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'all', 'custom'],
                       help='监控模式: quick(前5), all(全部), custom(自定义)')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='自定义监控品种，例如: --symbols 镍 白银 铜')

    args = parser.parse_args()

    if args.mode == 'quick':
        quick_monitor(top_n=5)
    elif args.mode == 'all':
        monitor_all_symbols()
    elif args.mode == 'custom' and args.symbols:
        # 验证品种名称
        valid_symbols = [s for s in args.symbols if s in OPTIMAL_PARAMS]
        invalid_symbols = [s for s in args.symbols if s not in OPTIMAL_PARAMS]

        if invalid_symbols:
            print(f"\n[WARNING] 无效品种: {', '.join(invalid_symbols)}")
            print(f"有效品种: {', '.join(OPTIMAL_PARAMS.keys())}")

        if valid_symbols:
            monitor_all_symbols(valid_symbols)
        else:
            print("\n[ERROR] 没有有效品种")
    else:
        # 默认快速监控
        quick_monitor(top_n=5)


if __name__ == "__main__":
    main()
