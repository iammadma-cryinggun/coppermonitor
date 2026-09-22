"""
CCI策略每日数据记录系统
========================
每日记录各品种价格、指标和信号，用于复盘分析
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from datetime import datetime
import os
from 最优参数配置 import OPTIMAL_PARAMS
from cci_calculations import calculate_cci_tv, calculate_stc, f_normalize

# 品种代码映射
CODE_MAP = {
    '铜': 'cu0', '铝': 'al0', '锌': 'zn0', '铅': 'pb0', '镍': 'ni0', '锡': 'sn0',
    '黄金': 'au0', '白银': 'ag0', '玻璃': 'fg0', '纯碱': 'sa0', '糖': 'sr0', '棉花': 'cf0'
}

# 数据存储路径
DATA_DIR = "D:\\期货数据\\铜期货监控\\CCI策略系统\\daily_records"


def ensure_data_dir():
    """确保数据目录存在"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"创建数据目录: {DATA_DIR}")


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
        print(f"获取数据失败 {code}: {e}")
        return None


def calculate_indicators(df, cci_length, ma_length):
    """计算CCI和STC指标"""
    data = calculate_cci_tv(df, cci_length=cci_length, ma_length=ma_length)
    stc_raw = calculate_stc(data, length=10, fast=23, slow=50, aaa=0.5)
    data['stc'] = f_normalize(stc_raw, 20, 80)
    return data


def check_signal(current, previous, params):
    """检查交易信号"""
    cci = current['cci']
    cci_ma = current['cci_ma']
    stc = current['stc']
    prev_cci = previous['cci']
    prev_cci_ma = previous['cci_ma']

    signal = '观望'
    signal_type = None

    # 买入信号
    if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
        signal = '买入'
        signal_type = '超卖'
    elif prev_cci <= prev_cci_ma and cci > cci_ma and cci <= params['cci_cross_max'] and stc >= params['stc_cross']:
        signal = '买入'
        signal_type = '金叉'

    # 卖出信号
    elif cci > params['cci_overbought']:
        signal = '卖出'
        signal_type = '超买'
    elif prev_cci >= prev_cci_ma and cci < cci_ma:
        signal = '卖出'
        signal_type = '死叉'

    return signal, signal_type


def record_daily_data():
    """记录每日数据"""
    ensure_data_dir()

    today = datetime.now().strftime('%Y-%m-%d')
    print("="*100)
    print(f"CCI策略每日数据记录 - {today}".center(100))
    print("="*100)

    all_records = []

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

            signal, signal_type = check_signal(current, previous, params)

            # 计算止盈止损
            entry_price = current['close']
            stop_loss = entry_price * 0.96
            take_profit = entry_price * 1.20

            record = {
                'date': today,
                'symbol': symbol,
                'code': code,
                'open': current['open'],
                'high': current['high'],
                'low': current['low'],
                'close': current['close'],
                'cci': round(current['cci'], 2),
                'cci_ma': round(current['cci_ma'], 2),
                'stc': round(current['stc'], 2),
                'signal': signal,
                'signal_type': signal_type or '',
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'win_rate': params['win_rate'],
                'max_dd': params['max_dd'],
                'expected_return': params['expected_return']
            }

            all_records.append(record)
            print(f"价格: {current['close']:.2f}, CCI: {current['cci']:.2f}, 信号: {signal}")

        except Exception as e:
            print(f"错误: {e}")

    if not all_records:
        print("\n没有获取到任何数据!")
        return

    # 保存到CSV
    df_records = pd.DataFrame(all_records)

    # 每日记录文件
    daily_file = os.path.join(DATA_DIR, f"daily_{today.replace('-', '')}.csv")
    df_records.to_csv(daily_file, index=False, encoding='utf-8-sig')
    print(f"\n每日记录已保存: {daily_file}")

    # 追加到历史记录
    history_file = os.path.join(DATA_DIR, "history_all.csv")
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        # 删除今天的旧记录（如果存在）
        history_df = history_df[history_df['date'] != today]
        df_records = pd.concat([history_df, df_records], ignore_index=True)

    df_records.to_csv(history_file, index=False, encoding='utf-8-sig')
    print(f"历史记录已更新: {history_file}")

    # 输出信号汇总
    print("\n" + "="*100)
    print("信号汇总".center(100))
    print("="*100)

    buy_signals = df_records[df_records['signal'] == '买入']
    sell_signals = df_records[df_records['signal'] == '卖出']

    if len(buy_signals) > 0:
        print(f"\n[买入信号] {len(buy_signals)} 个品种:")
        for _, r in buy_signals.iterrows():
            print(f"  {r['symbol']:<8} 价格: {r['close']:.2f} | {r['signal_type']} | "
                  f"止损: {r['stop_loss']:.2f} | 止盈: {r['take_profit']:.2f}")

    if len(sell_signals) > 0:
        print(f"\n[卖出信号] {len(sell_signals)} 个品种:")
        for _, r in sell_signals.iterrows():
            print(f"  {r['symbol']:<8} 价格: {r['close']:.2f} | {r['signal_type']}")

    print(f"\n[观望] {len(df_records) - len(buy_signals) - len(sell_signals)} 个品种")

    # 交易计划
    if len(buy_signals) > 0:
        print("\n" + "="*100)
        print("明日交易计划".center(100))
        print("="*100)

        for _, r in buy_signals.iterrows():
            print(f"\n【{r['symbol']}】")
            print(f"  建议操作: 明日开盘买入")
            print(f"  参考价格: {r['close']:.2f}")
            print(f"  止损价格: {r['stop_loss']:.2f} (-4%)")
            print(f"  止盈价格: {r['take_profit']:.2f} (+20%)")
            print(f"  历史胜率: {r['win_rate']:.1f}%")
            print(f"  历史回撤: {r['max_dd']:.2f}%")

    return df_records


def view_history(days=7):
    """查看历史记录"""
    history_file = os.path.join(DATA_DIR, "history_all.csv")

    if not os.path.exists(history_file):
        print("历史记录文件不存在，请先运行记录功能")
        return

    df = pd.read_csv(history_file)
    df['date'] = pd.to_datetime(df['date'])

    # 最近N天
    recent_dates = sorted(df['date'].unique())[-days:]

    print("="*100)
    print(f"最近{days}天信号记录".center(100))
    print("="*100)

    for date in recent_dates:
        day_data = df[df['date'] == date]
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')

        buy = day_data[day_data['signal'] == '买入']
        sell = day_data[day_data['signal'] == '卖出']

        if len(buy) > 0 or len(sell) > 0:
            print(f"\n[{date_str}]")
            if len(buy) > 0:
                print(f"  买入: {', '.join(buy['symbol'].tolist())}")
            if len(sell) > 0:
                print(f"  卖出: {', '.join(sell['symbol'].tolist())}")

    # 统计
    print("\n" + "="*100)
    print("信号统计".center(100))
    print("="*100)

    recent_df = df[df['date'].isin(recent_dates)]
    signal_counts = recent_df.groupby(['symbol', 'signal']).size().unstack(fill_value=0)

    print(f"\n{signal_counts}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='CCI策略每日数据记录')
    parser.add_argument('--record', action='store_true', help='记录今日数据')
    parser.add_argument('--view', type=int, default=0, help='查看最近N天历史')

    args = parser.parse_args()

    if args.record:
        record_daily_data()
    elif args.view > 0:
        view_history(args.view)
    else:
        # 默认执行记录
        record_daily_data()


if __name__ == "__main__":
    main()
