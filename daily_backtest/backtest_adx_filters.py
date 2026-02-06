# -*- coding: utf-8 -*-
"""
完整回测验证：不同过滤条件的实战效果
对比分析：无过滤 vs EMA过滤 vs ADX过滤 vs 组合过滤
"""

import pandas as pd
import numpy as np
import os

os.chdir(r"D:\期货数据\铜期货监控\daily_backtest")

print("="*80)
print("完整回测验证：不同过滤条件的实战效果")
print("="*80)

# ============== 第一步：准备数据 ==============
print("\n正在准备数据...")

# 读取日线数据
df_daily = pd.read_csv('SN_沪锡_日线_10年.csv')
df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
df_daily = df_daily.rename(columns={
    '收盘价': 'close',
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low'
})

# 计算技术指标
def calculate_ema(df, fast=3, slow=15):
    """计算EMA"""
    df[f'ema_{fast}'] = df['close'].ewm(span=fast, adjust=False).mean()
    df[f'ema_{slow}'] = df['close'].ewm(span=slow, adjust=False).mean()
    return df

def calculate_adx(df, period=14):
    """计算ADX"""
    # True Range
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )

    # Directional Movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']

    df['plus_dm'] = np.where(
        (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
        df['up_move'],
        0
    )

    df['minus_dm'] = np.where(
        (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
        df['down_move'],
        0
    )

    # Smoothed
    df['atr'] = df['tr'].rolling(window=period).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])

    # DX and ADX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=period).mean()

    # Trend direction
    df['trend'] = 0
    df.loc[df['plus_di'] > df['minus_di'], 'trend'] = 1
    df.loc[df['minus_di'] > df['plus_di'], 'trend'] = -1

    return df

# 计算所有指标
df_daily = calculate_ema(df_daily, fast=3, slow=15)
df_daily = calculate_adx(df_daily, period=14)

# 删除NaN（前100天用于计算指标）
df_daily = df_daily.iloc[100:].reset_index(drop=True)

print(f"日线数据准备完成: {len(df_daily)}天")

# ============== 第二步：定义回测函数 ==============
def run_backtest_with_filter(df, filter_config, initial_capital=100000):
    """
    运行回测

    参数：
        df: 日线数据
        filter_config: 过滤条件配置
        initial_capital: 初始资金

    返回：
        dict: 回测结果
    """

    capital = initial_capital
    position = 0  # 0=空仓, 1=持仓
    entry_price = 0
    entry_date = None

    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        current_price = row['close']

        # 如果持仓中，检查出场条件
        if position == 1:
            # 出场条件1：止损2%
            if current_price <= entry_price * 0.98:
                pnl_pct = (current_price - entry_price) / entry_price * 100
                pnl = capital * pnl_pct / 100
                capital += pnl

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row['datetime'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'exit_reason': 'stop_loss'
                })

                position = 0
                entry_price = 0
                entry_date = None
                continue

            # 出场条件2：EMA反转（EMA3下穿EMA15）
            if row['ema_3'] < row['ema_15']:
                pnl_pct = (current_price - entry_price) / entry_price * 100
                pnl = capital * pnl_pct / 100
                capital += pnl

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row['datetime'],
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'exit_reason': 'trend_reversal'
                })

                position = 0
                entry_price = 0
                entry_date = None
                continue

            # 持仓中，继续
            continue

        # 如果空仓，检查入场条件
        if position == 0 and i < len(df) - 20:  # 确保有20天数据
            # 检查过滤条件
            can_enter = True

            # 条件1：EMA趋势向上
            if filter_config.get('require_ema_up', False):
                if row['ema_3'] <= row['ema_15']:
                    can_enter = False

            # 条件2：ADX强度
            if filter_config.get('require_adx', False):
                min_adx = filter_config.get('min_adx', 25)
                if row['adx'] < min_adx:
                    can_enter = False

            # 条件3：趋势方向
            if filter_config.get('require_trend_up', False):
                if row['trend'] != 1:
                    can_enter = False

            # 条件4：不追高（入场前5天涨幅 < 1%）
            if filter_config.get('no_chase', False):
                if i >= 5:
                    past_5_days = df.iloc[i-5:i]
                    price_change = (past_5_days['close'].iloc[-1] - past_5_days['close'].iloc[0]) / past_5_days['close'].iloc[0]
                    if price_change > 0.01:  # 1%
                        can_enter = False

            # 如果满足所有条件，入场
            if can_enter:
                position = 1
                entry_price = current_price
                entry_date = row['datetime']

    # 计算统计指标
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'final_capital': initial_capital,
            'trades': []
        }

    df_trades = pd.DataFrame(trades)

    winning_trades = df_trades[df_trades['pnl'] > 0]
    losing_trades = df_trades[df_trades['pnl'] <= 0]

    total_return = (capital - initial_capital) / initial_capital * 100
    win_rate = len(winning_trades) / len(trades) * 100

    return {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'final_capital': capital,
        'trades': df_trades
    }

# ============== 第三步：测试不同过滤条件 ==============
print(f"\n{'='*80}")
print("回测不同过滤条件的效果")
print(f"{'='*80}")

# 定义测试配置
test_configs = [
    {
        'name': '无过滤（原始策略）',
        'config': {}
    },
    {
        'name': '条件1：EMA向上',
        'config': {'require_ema_up': True}
    },
    {
        'name': '条件2：EMA向上 + 不追高',
        'config': {'require_ema_up': True, 'no_chase': True}
    },
    {
        'name': '条件3：EMA向上 + ADX>=25',
        'config': {'require_ema_up': True, 'require_adx': True, 'min_adx': 25}
    },
    {
        'name': '条件4：EMA向上 + ADX>=25 + 不追高',
        'config': {'require_ema_up': True, 'require_adx': True, 'min_adx': 25, 'no_chase': True}
    },
    {
        'name': '条件5：全部条件（EMA+ADX+方向+不追高）',
        'config': {
            'require_ema_up': True,
            'require_adx': True,
            'min_adx': 25,
            'require_trend_up': True,
            'no_chase': True
        }
    },
]

results = []

print(f"\n{'策略':<40} {'交易数':<8} {'胜率':<10} {'总收益':<12} {'最终资金':<12}")
print("-"*100)

for test in test_configs:
    result = run_backtest_with_filter(df_daily, test['config'])

    results.append({
        'name': test['name'],
        'config': test['config'],
        'result': result
    })

    print(f"{test['name']:<40} {result['total_trades']:<8} "
          f"{result['win_rate']:<10.1f}% {result['total_return']:<12.2f}% "
          f"{result['final_capital']:>10.0f}")

# ============== 第四步：详细分析最佳策略 ==============
print(f"\n{'='*80}")
print("最佳策略详细分析")
print(f"{'='*80}")

# 找出收益最高的策略
best_strategy = max(results, key=lambda x: x['result']['total_return'])

print(f"\n最佳策略: {best_strategy['name']}")
print(f"总交易数: {best_strategy['result']['total_trades']}")
print(f"盈利交易: {best_strategy['result']['winning_trades']}")
print(f"亏损交易: {best_strategy['result']['losing_trades']}")
print(f"胜率: {best_strategy['result']['win_rate']:.1f}%")
print(f"总收益: {best_strategy['result']['total_return']:.2f}%")
print(f"最终资金: {best_strategy['result']['final_capital']:.2f}")

# 详细分析盈利和亏损
if len(best_strategy['result']['trades']) > 0:
    winning = best_strategy['result']['trades'][best_strategy['result']['trades']['pnl'] > 0]
    losing = best_strategy['result']['trades'][best_strategy['result']['trades']['pnl'] <= 0]

    print(f"\n盈利交易统计:")
    print(f"  平均盈利: {winning['pnl_pct'].mean():.2f}%")
    print(f"  最大盈利: {winning['pnl_pct'].max():.2f}%")
    print(f"  最小盈利: {winning['pnl_pct'].min():.2f}%")

    print(f"\n亏损交易统计:")
    print(f"  平均亏损: {losing['pnl_pct'].mean():.2f}%")
    print(f"  最大亏损: {losing['pnl_pct'].min():.2f}%")
    print(f"  最小亏损: {losing['pnl_pct'].max():.2f}%")

# ============== 第五步：对比分析 ==============
print(f"\n{'='*80}")
print("策略对比分析")
print(f"{'='*80}")

baseline = results[0]['result']  # 无过滤

print(f"\n相对基准（无过滤）的改进：")
print(f"{'策略':<40} {'交易数变化':<15} {'胜率变化':<15} {'收益变化':<15}")
print("-"*100)

for item in results[1:]:  # 跳过第一个（基准）
    res = item['result']

    trades_change = res['total_trades'] - baseline['total_trades']
    win_rate_change = res['win_rate'] - baseline['win_rate']
    return_change = res['total_return'] - baseline['total_return']

    trades_str = f"{trades_change:+d}"
    win_rate_str = f"{win_rate_change:+.1f}%"
    return_str = f"{return_change:+.2f}%"

    print(f"{item['name']:<40} {trades_str:<15} {win_rate_str:<15} {return_str:<15}")

# ============== 第六步：最终结论 ==============
print(f"\n{'='*80}")
print("【最终结论】")
print(f"{'='*80}")

print(f"""
基于10年日线数据的完整回测验证：

【回测基准】
- 数据范围: 2016-02-15 ~ 2026-02-04
- 初始资金: 100,000元
- 交易品种: 沪锡期货
- 回测周期: 10年

【策略排名】
""")

# 按收益排序
results_sorted = sorted(results, key=lambda x: x['result']['total_return'], reverse=True)

for i, item in enumerate(results_sorted, 1):
    res = item['result']
    print(f"{i}. {item['name']}")
    print(f"   交易数: {res['total_trades']}, 胜率: {res['win_rate']:.1f}%, 收益: {res['total_return']:.2f}%")

print(f"\n【核心发现】")

# 找出胜率最高的策略
best_win_rate = max(results, key=lambda x: x['result']['win_rate'])
print(f"1. 最高胜率: {best_win_rate['name']} ({best_win_rate['result']['win_rate']:.1f}%)")

# 找出收益最高的策略
print(f"2. 最高收益: {best_strategy['name']} ({best_strategy['result']['total_return']:.2f}%)")

# 分析过滤条件的效果
print(f"\n【过滤条件的价值】")

no_filter = results[0]['result']
with_ema = results[1]['result']
with_adx = results[3]['result']
with_all = results[5]['result']

print(f"1. EMA过滤的效果:")
print(f"   交易数: {no_filter['total_trades']} -> {with_ema['total_trades']}")
print(f"   胜率: {no_filter['win_rate']:.1f}% -> {with_ema['win_rate']:.1f}% ({with_ema['win_rate']-no_filter['win_rate']:+.1f}%)")
print(f"   收益: {no_filter['total_return']:.2f}% -> {with_ema['total_return']:.2f}% ({with_ema['total_return']-no_filter['total_return']:+.2f}%)")

print(f"\n2. ADX过滤的效果:")
print(f"   交易数: {no_filter['total_trades']} -> {with_adx['total_trades']}")
print(f"   胜率: {no_filter['win_rate']:.1f}% -> {with_adx['win_rate']:.1f}% ({with_adx['win_rate']-no_filter['win_rate']:+.1f}%)")
print(f"   收益: {no_filter['total_return']:.2f}% -> {with_adx['total_return']:.2f}% ({with_adx['total_return']-no_filter['total_return']:+.2f}%)")

print(f"\n3. 全部过滤的效果:")
print(f"   交易数: {no_filter['total_trades']} -> {with_all['total_trades']}")
print(f"   胜率: {no_filter['win_rate']:.1f}% -> {with_all['win_rate']:.1f}% ({with_all['win_rate']-no_filter['win_rate']:+.1f}%)")
print(f"   收益: {no_filter['total_return']:.2f}% -> {with_all['total_return']:.2f}% ({with_all['total_return']-no_filter['total_return']:+.2f}%)")

# 判断哪种策略最好
if best_strategy['result']['total_return'] > no_filter['total_return']:
    print(f"\n【推荐】使用: {best_strategy['name']}")
    print(f"原因: 相比无过滤，收益提升{best_strategy['result']['total_return']-no_filter['total_return']:.2f}个百分点")
else:
    print(f"\n【推荐】使用: 无过滤（原始策略）")
    print(f"原因: 过滤条件反而降低了收益")

print("\n" + "="*80)
print("回测完成")
print("="*80)

# 保存最佳策略的交易记录
if len(best_strategy['result']['trades']) > 0:
    best_strategy['result']['trades'].to_csv(f'best_strategy_trades.csv', index=False)
    print(f"\n最佳策略的交易记录已保存: best_strategy_trades.csv")
