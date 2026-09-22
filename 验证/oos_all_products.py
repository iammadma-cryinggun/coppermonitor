#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全品种 OOS 样本外验证
━━━━━━━━━━━━━━━━━━━
对12个国内期货品种统一做时间切分验证:
  - 前60% 训练期 (固定参数, 不调优)
  - 后40% 测试期 (从未见过的数据)
  - 参数 = 最优参数配置.py 的值 (全样本优化结果)
  - 判定: 测试期为正且保留训练期 >30% → edge 可能真实
"""
import pandas as pd
import numpy as np
import os, sys

sys.path.append(r'D:\期货数据\铜期货监控\CCI策略系统')
from 最优参数配置 import OPTIMAL_PARAMS
from validate_cci_and_copper import StrictBacktest

DATA_DIR = r'D:\期货数据\铜期货监控\global_futures_daily'

def get_data_path(symbol):
    if symbol == '铜':
        return os.path.join(DATA_DIR, '铜_LME_daily.csv')
    return os.path.join(DATA_DIR, f'{symbol}_daily.csv')

def load(symbol):
    df = pd.read_csv(get_data_path(symbol))
    # 兼容不同列名
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df

def run_split(symbol):
    df = load(symbol)
    params = OPTIMAL_PARAMS[symbol]
    n = len(df)
    split = int(n * 0.6)
    df_train = df.iloc[:split]
    df_test  = df.iloc[split:]

    bt_train = StrictBacktest(df_train, params)
    bt_test  = StrictBacktest(df_test, params)
    r_train  = bt_train.run()
    r_test   = bt_test.run()

    if r_train and r_test:
        retention = (r_test['total_return'] / r_train['total_return'] * 100
                     if abs(r_train['total_return']) > 0.01 else 0)
        return {
            '品种': symbol,
            '数据根数': n,
            '训练期': f"{df_train.index[0].date()}~{df_train.index[-1].date()}",
            '测试期': f"{df_test.index[0].date()}~{df_test.index[-1].date()}",

            '训练_收益%': round(r_train['total_return'], 1),
            '训练_DD%': round(r_train['max_dd'], 1),
            '训练_笔': r_train['total_trades'],
            '训练_胜率%': round(r_train['win_rate'], 1),

            '测试_收益%': round(r_test['total_return'], 1),
            '测试_DD%': round(r_test['max_dd'], 1),
            '测试_笔': r_test['total_trades'],
            '测试_胜率%': round(r_test['win_rate'], 1),

            '保留%': round(retention, 0),
            'edge判定': ('PASS' if r_test['total_return'] > 0 and retention > 30
                         else ('MARGINAL' if r_test['total_return'] > 0 else 'FAIL'))
        }
    return None

def main():
    print('=' * 100)
    print('全品种 OOS 样本外验证')
    print('=' * 100)
    print('参数 = 最优参数配置.py (全样本优化结果)')
    print('训练期 = 前60%   测试期 = 后40%')
    print('=' * 100)
    print()

    all_r = []
    for symbol in OPTIMAL_PARAMS:
        try:
            r = run_split(symbol)
            if r:
                all_r.append(r)
                status = '[PASS]' if r['edge判定'] == 'PASS' else '[FAIL]'
                print(f"  {symbol:<6} 训练{r['训练_收益%']:>+8.1f}% / 测试{r['测试_收益%']:>+8.1f}% "
                      f"(保留{r['保留%']:.0f}%)  训练{r['训练_笔']}笔 测试{r['测试_笔']}笔 "
                      f"{status}")
        except Exception as e:
            print(f"  {symbol:<6} [ERROR] {e}")

    if all_r:
        print('\n' + '=' * 100)
        print('汇总 (按测试期收益排序)')
        print('=' * 100)
        df_r = pd.DataFrame(all_r)
        df_r = df_r.sort_values('测试_收益%', ascending=False)
        cols = ['品种', '训练_收益%', '训练_DD%', '测试_收益%', '测试_DD%',
                '保留%', '训练_笔', '测试_笔', 'edge判定']
        print(df_r[cols].to_string(index=False))

        passes = [r for r in all_r if r['edge判定'] == 'PASS']
        fails  = [r for r in all_r if r['edge判定'] == 'FAIL']
        marginal = [r for r in all_r if r['edge判定'] == 'MARGINAL']

        print(f'\n结果: {len(passes)} PASS / {len(marginal)} MARGINAL / {len(fails)} FAIL (共{len(all_r)}品种)')
        if passes:
            print(f'  可考虑部署: {", ".join(r["品种"] for r in passes)}')
        if marginal:
            print(f'  边缘/需观察: {", ".join(r["品种"] for r in marginal)}')
        if fails:
            print(f'  不可部署:   {", ".join(r["品种"] for r in fails)}')

if __name__ == '__main__':
    main()
