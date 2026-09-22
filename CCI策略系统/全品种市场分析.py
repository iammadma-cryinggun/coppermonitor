"""
全品种市场分析 - 分析所有12个品种
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from datetime import datetime
from 最优参数配置 import OPTIMAL_PARAMS
from cci_calculations import calculate_cci_tv, calculate_stc, f_normalize

def get_realtime_data(code, days=200):
    """获取实时行情数据"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        df = df.iloc[:, :5]
        df.columns = ['date', 'open', 'high', 'low', 'close']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.tail(days)
        df = df.dropna()
        return df
    except Exception as e:
        return None

def calculate_indicators(df, cci_length, ma_length):
    """计算CCI和STC指标"""
    data = calculate_cci_tv(df, cci_length=cci_length, ma_length=ma_length)
    stc_raw = calculate_stc(data, length=10, fast=23, slow=50, aaa=0.5)
    data['stc'] = f_normalize(stc_raw, 20, 80)
    return data

def analyze_all_symbols():
    """分析所有品种"""

    # 品种代码映射（主力合约）
    CODE_MAP = {
        '铜': 'cu0', '铝': 'al0', '锌': 'zn0', '铅': 'pb0', '镍': 'ni0', '锡': 'sn0',
        '黄金': 'au0', '白银': 'ag0', '玻璃': 'fg0', '纯碱': 'sa0', '糖': 'sr0', '棉花': 'cf0'
    }

    print("="*130)
    print(f"全品种市场分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(130))
    print("="*130)

    results = []

    for symbol, params in OPTIMAL_PARAMS.items():
        code = CODE_MAP.get(symbol, symbol)
        print(f"\n[{symbol}]", end=" ")

        try:
            df = get_realtime_data(code)
            if df is None or len(df) < 50:
                print("数据不足")
                continue

            data = calculate_indicators(df, params['cci_length'], params['ma_length'])

            current = data.iloc[-1]
            previous = data.iloc[-2]

            cci = current['cci']
            cci_ma = current['cci_ma']
            stc = current['stc']
            prev_cci = previous['cci']
            prev_cci_ma = previous['cci_ma']

            # 判断信号
            signal = '观望'
            signal_detail = ''

            if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
                signal = '买入'
                signal_detail = f"超卖(CCI={cci:.1f}<{params['cci_oversold']})"
            elif prev_cci <= prev_cci_ma and cci > cci_ma and cci <= params['cci_cross_max'] and stc >= params['stc_cross']:
                signal = '买入'
                signal_detail = f"金叉(CCI={cci:.1f})"
            elif cci > params['cci_overbought']:
                signal = '卖出'
                signal_detail = f"超买(CCI={cci:.1f}>{params['cci_overbought']})"
            elif prev_cci >= prev_cci_ma and cci < cci_ma:
                signal = '卖出'
                signal_detail = f"死叉(CCI={cci:.1f})"

            # 计算止盈止损价格（基于当前收盘价预估次日开盘价）
            entry_price_estimate = current['close']  # 预估次日开盘价≈今日收盘价
            stop_loss_price = entry_price_estimate * 0.96
            take_profit_price = entry_price_estimate * 1.20
            stop_loss_pct = -4.0
            take_profit_pct = 20.0

            results.append({
                'symbol': symbol,
                'price': current['close'],
                'cci': cci,
                'cci_ma': cci_ma,
                'stc': stc,
                'signal': signal,
                'signal_detail': signal_detail,
                'win_rate': params['win_rate'],
                'max_dd': params['max_dd'],
                'expected_return': params['expected_return'],
                'rating': params['rating'],
                'multiplier': params['multiplier'],
                # 止盈止损信息
                'entry_estimate': entry_price_estimate,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct
            })

            print(f"价格: {current['close']:.2f}, CCI: {cci:.2f}, 信号: {signal}")

        except Exception as e:
            print(f"错误: {e}")

    # 输出汇总表
    print("\n" + "="*130)
    print("市场汇总".center(130))
    print("="*130)

    print(f"\n{'品种':<8}{'价格':<12}{'CCI':<10}{'CCI_MA':<10}{'STC':<10}{'信号':<8}{'详情':<25}{'胜率':<8}{'回撤':<10}{'评级':<8}")
    print("-"*130)

    # 按信号排序
    signal_order = {'买入': 0, '卖出': 1, '观望': 2}
    results_sorted = sorted(results, key=lambda x: signal_order.get(x['signal'], 2))

    for r in results_sorted:
        print(f"{r['symbol']:<8}{r['price']:<12.2f}{r['cci']:<10.2f}{r['cci_ma']:<10.2f}{r['stc']:<10.2f}"
              f"{r['signal']:<8}{r['signal_detail']:<25}{r['win_rate']:<.1f}%{r['max_dd']:<.2f}%{r['rating']:<8}")

    # 信号分类
    print("\n" + "="*130)
    print("信号分类".center(130))
    print("="*130)

    buy_signals = [r for r in results if r['signal'] == '买入']
    sell_signals = [r for r in results if r['signal'] == '卖出']
    hold_signals = [r for r in results if r['signal'] == '观望']

    print(f"\n[买入信号] {len(buy_signals)} 个品种:")
    for r in buy_signals:
        print(f"\n  {'='*100}")
        print(f"  {r['symbol']} - {r['signal_detail']}")
        print(f"  {'='*100}")
        print(f"  当前价格: {r['price']:.2f}")
        print(f"  CCI: {r['cci']:.2f} | CCI_MA: {r['cci_ma']:.2f} | STC: {r['stc']:.2f}")
        print(f"  历史表现: 胜率{r['win_rate']:.1f}% | 回撤{r['max_dd']:.2f}% | 预期收益{r['expected_return']:+.2f}%")
        print(f"  合约乘数: {r['multiplier']}")
        print(f"\n  【交易计划】")
        print(f"  预估开仓价: {r['entry_estimate']:.2f} (次日开盘价)")
        print(f"  止损价格:   {r['stop_loss_price']:.2f} (亏损 {r['stop_loss_pct']:.1f}%)")
        print(f"  止盈价格:   {r['take_profit_price']:.2f} (盈利 +{r['take_profit_pct']:.1f}%)")
        print(f"\n  【风险计算】(假设开仓1手)")
        position = 1
        stop_loss_amount = (r['stop_loss_price'] - r['entry_estimate']) * position * r['multiplier']
        take_profit_amount = (r['take_profit_price'] - r['entry_estimate']) * position * r['multiplier']
        print(f"  最大亏损: {stop_loss_amount:,.0f}元 (触发止损)")
        print(f"  最大盈利: {take_profit_amount:,.0f}元 (触发止盈)")
        print(f"  盈亏比:   {abs(take_profit_amount/stop_loss_amount):.1f}:1")

    print(f"\n[卖出信号] {len(sell_signals)} 个品种:")
    for r in sell_signals:
        print(f"  {r['symbol']:<8} 价格: {r['price']:.2f} | {r['signal_detail']}")

    print(f"\n[观望] {len(hold_signals)} 个品种:")
    hold_info = ', '.join([f"{r['symbol']}(CCI:{r['cci']:.0f})" for r in hold_signals])
    print(f"  {hold_info}")

    # 推荐操作
    print("\n" + "="*130)
    print("操作建议".center(130))
    print("="*130)

    if buy_signals:
        print("\n推荐买入:")
        for r in buy_signals:
            if r['max_dd'] > -30:  # 只推荐回撤<30%的品种
                print(f"  ★ {r['symbol']}: 回撤可控({r['max_dd']:.2f}%), 胜率{r['win_rate']:.1f}%")
            else:
                print(f"  ☆ {r['symbol']}: 回撤较大({r['max_dd']:.2f}%), 谨慎操作")

    print("\n" + "="*130)


if __name__ == "__main__":
    analyze_all_symbols()
