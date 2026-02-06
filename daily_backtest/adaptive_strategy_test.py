# -*- coding: utf-8 -*-
"""
"看人下菜碟"策略测试：根据GHE自适应切换交易模式
核心思路：用GHE判断市场状态，选择最合适的策略
"""

import pandas as pd
import numpy as np
import os

os.chdir(r"D:\期货数据\铜期货监控\daily_backtest")

print("="*80)
print("自适应策略切换系统测试")
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

# 计算所有需要的指标
def calculate_all_indicators(df):
    """计算所有技术指标"""
    # GHE
    def calculate_ghe(series, q=2, max_tau=20):
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

    # EMA
    df['ema_3'] = df['close'].ewm(span=3, adjust=False).mean()
    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()

    # ADX
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr'])
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=14).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 布林带
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # 计算GHE（需要200天窗口）
    ghe_values = []
    for i in range(200, len(df)):
        window_data = df.iloc[i-200:i]['close']
        ghe = calculate_ghe(window_data)
        ghe_values.append(ghe)

    # 前200天GHE设为NaN
    ghe_full = [np.nan] * 200 + ghe_values
    df['ghe'] = ghe_full

    return df

df_daily = calculate_all_indicators(df_daily)
df_daily = df_daily.iloc[200:].reset_index(drop=True)  # 删除前200天（用于计算GHE）

print(f"日线数据准备完成: {len(df_daily)}天")

# ============== 第二步：定义3种交易模式 ==============
def trend_mode_strategy(df, i, initial_capital=100000):
    """
    趋势模式策略（GHE > 0.45时使用）
    使用EMA顺势，不做空，只在回调时买入
    """
    if i < 20:
        return None

    # 入场条件：EMA(3) > EMA(15)
    if df.loc[i, 'ema_3'] <= df.loc[i, 'ema_15']:
        return None

    # 检查之前是否持仓（简化处理，假设每次1手）
    # 这里简化为：每次入场都记录
    entry_price = df.loc[i, 'close']
    entry_date = df.loc[i, 'datetime']

    # 止损：2%
    stop_loss_price = entry_price * 0.98

    # 寻找出场点（未来20天内）
    for j in range(i+1, min(i+21, len(df))):
        # 出场条件1：止损
        if df.loc[j, 'close'] <= stop_loss_price:
            exit_price = df.loc[j, 'close']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            return {
                'mode': 'trend',
                'entry_date': entry_date,
                'exit_date': df.loc[j, 'datetime'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': 'stop_loss'
            }

        # 出场条件2：趋势反转（EMA3下穿EMA15）
        if df.loc[j, 'ema_3'] < df.loc[j, 'ema_15']:
            exit_price = df.loc[j, 'close']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            return {
                'mode': 'trend',
                'entry_date': entry_date,
                'exit_date': df.loc[j, 'datetime'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': 'trend_reversal'
            }

        # 出场条件3：20天后强制平仓
        if j == i + 20:
            exit_price = df.loc[j, 'close']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            return {
                'mode': 'trend',
                'entry_date': entry_date,
                'exit_date': df.loc[j, 'datetime'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': 'timeout'
            }

    return None


def ranging_mode_strategy(df, i, initial_capital=100000):
    """
    震荡模式策略（GHE < 0.38时使用）
    使用RSI/布林带高抛低吸
    """
    if i < 20:
        return None

    # 入场条件：RSI < 30（超卖）
    if df.loc[i, 'rsi'] >= 30:
        return None

    # 检查价格是否接近布林带下轨
    bb_position = (df.loc[i, 'close'] - df.loc[i, 'bb_lower']) / (df.loc[i, 'bb_upper'] - df.loc[i, 'bb_lower'])

    if bb_position > 0.5:  # 不在下轨附近
        return None

    entry_price = df.loc[i, 'close']
    entry_date = df.loc[i, 'datetime']

    # 止损：2%
    stop_loss_price = entry_price * 0.98

    # 目标：RSI回到50或涨到布林带中轨
    for j in range(i+1, min(i+11, len(df))):  # 最多10天
        # 出场条件1：止损
        if df.loc[j, 'close'] <= stop_loss_price:
            exit_price = df.loc[j, 'close']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            return {
                'mode': 'ranging',
                'entry_date': entry_date,
                'exit_date': df.loc[j, 'datetime'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': 'stop_loss'
            }

        # 出场条件2：RSI回到50以上
        if df.loc[j, 'rsi'] >= 50:
            exit_price = df.loc[j, 'close']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            return {
                'mode': 'ranging',
                'entry_date': entry_date,
                'exit_date': df.loc[j, 'datetime'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': 'rsi_target'
            }

        # 出场条件3：价格回到布林带中轨
        if df.loc[j, 'close'] >= df.loc[j, 'bb_middle']:
            exit_price = df.loc[j, 'close']
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            return {
                'mode': 'ranging',
                'entry_date': entry_date,
                'exit_date': df.loc[j, 'datetime'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': 'bb_middle'
            }

    # 10天后强制平仓
    if i + 10 < len(df):
        exit_price = df.loc[i+10, 'close']
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        return {
            'mode': 'ranging',
            'entry_date': entry_date,
            'exit_date': df.loc[i+10, 'datetime'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'exit_reason': 'timeout'
        }

    return None


def adaptive_strategy(df, i, initial_capital=100000):
    """
    自适应策略：根据GHE值自动选择模式
    """
    ghe = df.loc[i, 'ghe']

    # 混沌模式：不交易
    if pd.isna(ghe) or (0.38 <= ghe <= 0.45):
        return None

    # 趋势模式：GHE > 0.45
    if ghe > 0.45:
        return trend_mode_strategy(df, i, initial_capital)

    # 震荡模式：GHE < 0.38
    if ghe < 0.38:
        return ranging_mode_strategy(df, i, initial_capital)

    return None


# ============== 第三步：回测3种策略 ==============
print(f"\n{'='*80}")
print("开始回测...")
print(f"{'='*80}")

strategies = {
    '趋势模式（只做EMA顺势）': lambda df, i: trend_mode_strategy(df, i) if df.loc[i, 'ghe'] > 0.45 else None,
    '震荡模式（只做RSI高抛低吸）': lambda df, i: ranging_mode_strategy(df, i) if df.loc[i, 'ghe'] < 0.38 else None,
    '自适应模式（根据GHE切换）': lambda df, i: adaptive_strategy(df, i),
}

results = {}

for strategy_name, strategy_func in strategies.items():
    print(f"\n正在回测: {strategy_name}...")

    trades = []
    for i in range(100, len(df_daily) - 20):  # 确保有足够的数据
        result = strategy_func(df_daily, i)

        if result is not None:
            trades.append(result)

    df_trades = pd.DataFrame(trades)

    if len(df_trades) == 0:
        print(f"  没有产生交易")
        results[strategy_name] = None
        continue

    # 计算统计指标
    winning = df_trades[df_trades['pnl_pct'] > 0]
    losing = df_trades[df_trades['pnl_pct'] <= 0]

    total_return = df_trades['pnl_pct'].sum()
    win_rate = len(winning) / len(df_trades) * 100

    results[strategy_name] = {
        'trades': df_trades,
        'total_trades': len(df_trades),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_return': df_trades['pnl_pct'].mean()
    }

    print(f"  交易数: {len(df_trades)}")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  总收益: {total_return:.2f}%")

# ============== 第四步：对比分析 ==============
print(f"\n{'='*80}")
print("策略对比分析")
print(f"{'='*80}")

print(f"\n{'策略':<30} {'交易数':<8} {'胜率':<10} {'总收益%':<12} {'平均收益%':<12}")
print("-"*80)

for strategy_name, result in results.items():
    if result is None:
        continue

    print(f"{strategy_name:<30} {result['total_trades']:<8} "
          f"{result['win_rate']:<10.1f}% {result['total_return']:<12.2f} "
          f"{result['avg_return']:<12.2f}")

# ============== 第五步：深入分析自适应策略 ==============
if '自适应模式（根据GHE切换）' in results:
    adaptive = results['自适应模式（根据GHE切换）']
    df_adaptive = adaptive['trades']

    print(f"\n{'='*80}")
    print("自适应策略详细分析")
    print(f"{'='*80}")

    # 按模式分组
    trend_trades = df_adaptive[df_adaptive['mode'] == 'trend']
    ranging_trades = df_adaptive[df_adaptive['mode'] == 'ranging']

    print(f"\n趋势模式交易:")
    print(f"  交易数: {len(trend_trades)}")
    if len(trend_trades) > 0:
        trend_win_rate = (trend_trades['pnl_pct'] > 0).sum() / len(trend_trades) * 100
        trend_return = trend_trades['pnl_pct'].sum()
        print(f"  胜率: {trend_win_rate:.1f}%")
        print(f"  总收益: {trend_return:.2f}%")
        print(f"  平均收益: {trend_trades['pnl_pct'].mean():.2f}%")

    print(f"\n震荡模式交易:")
    print(f"  交易数: {len(ranging_trades)}")
    if len(ranging_trades) > 0:
        ranging_win_rate = (ranging_trades['pnl_pct'] > 0).sum() / len(ranging_trades) * 100
        ranging_return = ranging_trades['pnl_pct'].sum()
        print(f"  胜率: {ranging_win_rate:.1f}%")
        print(f"  总收益: {ranging_return:.2f}%")
        print(f"  平均收益: {ranging_trades['pnl_pct'].mean():.2f}%")

# ============== 第六步：最终结论 ==============
print(f"\n{'='*80}")
print("【最终结论】")
print(f"{'='*80}")

print(f"""
基于10年日线数据的"看人下菜碟"策略测试：

策略                            交易数     胜率       总收益
────────────────────────────────────────────────────────────────────────────────
""")

for strategy_name, result in results.items():
    if result is None:
        print(f"{strategy_name:<30} {'无交易':<8} {'N/A':<10} {'N/A':<12}")
    else:
        print(f"{strategy_name:<30} {result['total_trades']:<8} "
              f"{result['win_rate']:<10.1f}% {result['total_return']:<12.2f}%")

print(f"""
核心发现：

1. GHE作为"环境感知层"的效果：
   - GHE > 0.45时，市场处于趋势状态，用EMA顺势
   - GHE < 0.38时，市场处于震荡状态，用RSI高抛低吸
   - GHE 0.38-0.45时，市场混乱，避免交易

2. 自适应切换的优势：
   - 根据市场状态选择合适的策略
   - 避免在震荡市用趋势策略（左右挨打）
   - 避免在趋势市用震荡策略（错过大行情）

3. 局限性：
   - 信号较少（需要GHE满足特定条件）
   - 切换可能有滞后性
   - 需要实时计算GHE（200天窗口）

4. 实战建议：
   如果自适应策略的胜率和收益都优于单一策略：
   → 采用自适应切换
   → 用GHE作为每日的"市场温度计"
   → 根据温度选择合适的衣服（策略）

   如果自适应策略表现不佳：
   → 坚持使用单一的最优策略（你现在的4小时+日线框架）
   → 用ADX和入场时机来过滤，而不是GHE
""")

print("\n" + "="*80)
print("测试完成")
print("="*80)

# 保存数据
for strategy_name, result in results.items():
    if result is not None and len(result['trades']) > 0:
        filename = f"adaptive_backtest_{strategy_name.replace('（', '_').replace('）', '').replace(' ', '_')}.csv"
        result['trades'].to_csv(filename, index=False)
        print(f"\n数据已保存: {filename}")
