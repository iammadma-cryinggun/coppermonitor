# -*- coding: utf-8 -*-
"""
纯碱SA0 - CCI+STC策略超快速参数优化
测试2880个参数组合，寻找最优参数
"""

import numpy as np
import pandas as pd
from datetime import datetime
import itertools

# ==================== 数据加载 ====================
def load_data():
    """加载纯碱SA0数据"""
    file_path = r"D:\期货数据\铜期货监控\CCI策略系统\processed_data\SA0_processed.csv"
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

# ==================== 技术指标计算 ====================
def calculate_cci(series, length=12):
    """计算CCI指标"""
    typical = (series['high'] + series['low'] + series['close']) / 3
    sma = typical.rolling(window=length).mean()
    mad = typical.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (typical - sma) / (0.015 * mad)
    return cci

def calculate_stc(df, fast_length=12, slow_length=26, length=10):
    """计算STC指标"""
    # 计算MACD
    ema_fast = df['close'].ewm(span=fast_length, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_length, adjust=False).mean()
    macd = ema_fast - ema_slow

    # 计算STO
    min_macd = macd.rolling(window=length).min()
    max_macd = macd.rolling(window=length).max()
    stoch_k = 100 * (macd - min_macd) / (max_macd - min_macd)

    # 计算STC (STO的平滑版本)
    stc = stoch_k.rolling(window=length).mean()
    return stc - 50  # 转换为以0为中心

def calculate_ma(series, length=5):
    """简单移动平均"""
    return series.rolling(window=length).mean()

# ==================== 回测引擎 ====================
def backtest_strategy(df, params):
    """
    回测单个参数组合

    参数:
        df: 数据DataFrame
        params: 参数字典

    返回:
        dict: 回测结果
    """
    # 提取参数
    cci_oversold = params['cci_oversold']
    cci_overbought = params['cci_overbought']
    cci_cross_max = params['cci_cross_max']
    stc_oversold = params['stc_oversold']
    stc_cross = params['stc_cross']
    vol_threshold = params['vol_threshold']
    cci_length = params['cci_length']
    ma_length = params['ma_length']

    # 计算指标
    df = df.copy()
    df['cci'] = calculate_cci(df, length=cci_length)
    df['ma'] = calculate_ma(df['close'], length=ma_length)
    df['stc'] = calculate_stc(df)

    # 计算量能过滤 (使用成交量的相对变化)
    df['vol_change'] = df['volume'].pct_change()
    df['vol_valid'] = df['vol_change'] >= vol_threshold

    # 初始化变量
    position = None  # None, 'long'
    entry_price = 0
    trades = []
    total_return = 0

    # 止盈止损参数
    take_profit_pct = 0.20  # 20%止盈
    stop_loss_pct = 0.04    # 4%止损

    # 遍历数据
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        current_cci = df['cci'].iloc[i]
        current_stc = df['stc'].iloc[i]
        prev_cci = df['cci'].iloc[i-1]
        current_vol_valid = df['vol_valid'].iloc[i]

        # 平仓逻辑
        if position == 'long':
            # 计算收益率
            pnl_pct = (current_price - entry_price) / entry_price

            # 止盈
            if pnl_pct >= take_profit_pct:
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return': pnl_pct,
                    'exit_reason': 'take_profit',
                    'exit_idx': i
                })
                total_return += pnl_pct
                position = None
                continue

            # 止损
            if pnl_pct <= -stop_loss_pct:
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return': pnl_pct,
                    'exit_reason': 'stop_loss',
                    'exit_idx': i
                })
                total_return += pnl_pct
                position = None
                continue

            # CCI超买平仓
            if current_cci >= cci_overbought:
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return': pnl_pct,
                    'exit_reason': 'cci_overbought',
                    'exit_idx': i
                })
                total_return += pnl_pct
                position = None
                continue

            # CCI死叉平仓 (从高位下穿cci_cross_max)
            if prev_cci >= cci_cross_max and current_cci < cci_cross_max:
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return': pnl_pct,
                    'exit_reason': 'cci_cross_down',
                    'exit_idx': i
                })
                total_return += pnl_pct
                position = None
                continue

        # 开仓逻辑
        if position is None:
            # CCI从超卖区域上穿
            cci_oversold_signal = prev_cci <= cci_oversold and current_cci > cci_oversold

            # STC同样从超卖区域上穿
            prev_stc = df['stc'].iloc[i-1]
            stc_oversold_signal = prev_stc <= stc_oversold and current_stc > stc_oversold

            # STC上穿确认线
            stc_cross_signal = prev_stc <= stc_cross and current_stc > stc_cross

            # 综合开仓条件
            if (cci_oversold_signal and
                stc_oversold_signal and
                stc_cross_signal and
                current_vol_valid):

                position = 'long'
                entry_price = current_price

    # 如果还有持仓，按最后价格平仓
    if position == 'long':
        final_price = df['close'].iloc[-1]
        pnl_pct = (final_price - entry_price) / entry_price
        trades.append({
            'entry_price': entry_price,
            'exit_price': final_price,
            'return': pnl_pct,
            'exit_reason': 'end_of_data',
            'exit_idx': len(df) - 1
        })
        total_return += pnl_pct

    # 计算统计指标
    num_trades = len(trades)
    if num_trades > 0:
        returns = [t['return'] for t in trades]
        avg_return = np.mean(returns)
        win_rate = len([r for r in returns if r > 0]) / num_trades
        max_return = max(returns)
        min_return = min(returns)
    else:
        avg_return = 0
        win_rate = 0
        max_return = 0
        min_return = 0

    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'avg_return': avg_return,
        'win_rate': win_rate,
        'max_return': max_return,
        'min_return': min_return,
        'params': params
    }

# ==================== 参数优化 ====================
def optimize_parameters():
    """参数优化主函数"""
    print("=" * 80)
    print("纯碱SA0 - CCI+STC策略超快速参数优化")
    print("=" * 80)

    # 加载数据
    print("\n[1/4] 加载数据...")
    df = load_data()
    print(f"数据加载完成: {len(df)} 条记录")
    print(f"时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")

    # 定义参数网格
    print("\n[2/4] 定义参数搜索空间...")
    param_grid = {
        'cci_oversold': [-70, -60, -50, -40, -30],
        'cci_overbought': [150, 170, 190, 210],
        'cci_cross_max': [140, 160, 180, 200],
        'stc_oversold': [-60, -50, -40],
        'stc_cross': [-140, -130, -120],
        'vol_threshold': [0.0, 0.3, 0.5, 0.7],
        'cci_length': [12],  # 固定
        'ma_length': [5]     # 固定
    }

    # 生成所有参数组合
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    total_combinations = len(param_combinations)
    print(f"总参数组合数: {total_combinations}")
    print(f"预期完成时间: ~{total_combinations * 0.5 / 60:.1f} 分钟")

    # 运行回测
    print("\n[3/4] 开始参数扫描...")
    print("-" * 80)

    results = []
    for idx, params in enumerate(param_combinations, 1):
        result = backtest_strategy(df, params)
        results.append(result)

        # 每50个组合打印一次进度
        if idx % 50 == 0:
            best_so_far = max(results, key=lambda x: x['total_return'])
            print(f"进度: {idx}/{total_combinations} ({idx/total_combinations*100:.1f}%) | "
                  f"当前最佳收益: {best_so_far['total_return']:.2%}")

    # 分析结果
    print("\n[4/4] 分析优化结果...")
    print("-" * 80)

    # 按总收益率排序
    results_sorted = sorted(results, key=lambda x: x['total_return'], reverse=True)

    # 老参数结果对比
    old_params = {
        'cci_oversold': -100,
        'cci_overbought': 100,
        'cci_cross_max': 100,
        'stc_oversold': -25,
        'stc_cross': -100,
        'vol_threshold': 0.0,
        'cci_length': 12,
        'ma_length': 5
    }

    old_result = backtest_strategy(df, old_params)
    old_return = old_result['total_return']
    print(f"\n老参数收益: {old_return:.2%}")
    print(f"老参数配置: {old_params}")

    # TOP 20 结果
    print("\n" + "=" * 80)
    print("TOP 20 最佳参数组合")
    print("=" * 80)
    print(f"{'排名':<4} {'收益率':<10} {'交易数':<8} {'胜率':<8} {'平均收益':<10} {'参数配置'}")
    print("-" * 120)

    for rank, result in enumerate(results_sorted[:20], 1):
        params = result['params']
        param_str = (f"CCI超卖={params['cci_oversold']}, "
                    f"超买={params['cci_overbought']}, "
                    f"死叉={params['cci_cross_max']}, "
                    f"STC超卖={params['stc_oversold']}, "
                    f"STC叉={params['stc_cross']}, "
                    f"量能={params['vol_threshold']}")

        improvement = result['total_return'] - old_return
        improvement_str = f" (提升{improvement:.2%})" if rank == 1 else ""

        print(f"{rank:<4} {result['total_return']:>8.2%}  "
              f"{result['num_trades']:<8} "
              f"{result['win_rate']:>7.1%}  "
              f"{result['avg_return']:>9.2%}  "
              f"{param_str}{improvement_str}")

    # 最佳参数详细分析
    best_result = results_sorted[0]
    print("\n" + "=" * 80)
    print("最佳参数组合详细分析")
    print("=" * 80)
    print(f"总收益率: {best_result['total_return']:.2%}")
    print(f"提升幅度: {(best_result['total_return'] - old_return):.2%} "
          f"(相对老参数{(best_result['total_return']/old_return - 1)*100:.1f}%)")
    print(f"交易次数: {best_result['num_trades']}")
    print(f"胜率: {best_result['win_rate']:.2%}")
    print(f"平均收益: {best_result['avg_return']:.2%}")
    print(f"最大单笔收益: {best_result['max_return']:.2%}")
    print(f"最大单笔亏损: {best_result['min_return']:.2%}")
    print("\n参数配置:")
    for key, value in best_result['params'].items():
        print(f"  {key}: {value}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = rf"D:\期货数据\铜期货监控\CCI策略系统\optimization_results\SA0_optimization_{timestamp}.csv"

    # 准备保存的数据
    results_df = pd.DataFrame([
        {
            'rank': rank + 1,
            'total_return': r['total_return'],
            'num_trades': r['num_trades'],
            'win_rate': r['win_rate'],
            'avg_return': r['avg_return'],
            'max_return': r['max_return'],
            'min_return': r['min_return'],
            'cci_oversold': r['params']['cci_oversold'],
            'cci_overbought': r['params']['cci_overbought'],
            'cci_cross_max': r['params']['cci_cross_max'],
            'stc_oversold': r['params']['stc_oversold'],
            'stc_cross': r['params']['stc_cross'],
            'vol_threshold': r['params']['vol_threshold'],
            'improvement': r['total_return'] - old_return
        }
        for rank, r in enumerate(results_sorted)
    ])

    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_file}")

    print("\n" + "=" * 80)
    print("优化完成!")
    print("=" * 80)

# ==================== 主程序 ====================
if __name__ == "__main__":
    optimize_parameters()
