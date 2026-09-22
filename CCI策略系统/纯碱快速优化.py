"""
纯碱(SA0) 快速参数优化 - 只测试核心参数组合
目标: 10-30分钟内完成
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

# 基础配置
SYMBOL = 'sa0'
MULTIPLIER = 20
COMMISSION_RATE = 0.0003
SLIPPAGE_RATE = 0.0002
INITIAL_CAPITAL = 100000
LEVERAGE = 2


def get_akshare_data(code):
    """从akshare获取完整历史数据"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def calculate_stc(close_prices, fast_period=23, slow_period=50, cycle_period=10):
    """计算STC指标"""
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    lowest_macd = macd.rolling(window=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period).max()
    range_macd = highest_macd - lowest_macd

    stc = pd.Series(index=close_prices.index, dtype=float)
    stc.iloc[:cycle_period] = 0

    for i in range(cycle_period, len(macd)):
        if range_macd.iloc[i] != 0:
            stc.iloc[i] = ((macd.iloc[i] - lowest_macd.iloc[i]) / range_macd.iloc[i] - 0.5) * 200
        else:
            stc.iloc[i] = stc.iloc[i-1] if i > cycle_period else 0

    stc_normalized = pd.Series(index=stc.index, dtype=float)
    lookback = 20
    for i in range(len(stc)):
        if i < lookback:
            stc_normalized.iloc[i] = stc.iloc[i]
        else:
            window = stc.iloc[i-lookback+1:i+1]
            low = window.min()
            high = window.max()
            if high != low:
                stc_normalized.iloc[i] = (stc.iloc[i] - low) / (high - low) * 100 - 50
            else:
                stc_normalized.iloc[i] = stc_normalized.iloc[i-1] if i > 0 else 0

    return stc_normalized


def run_backtest(df, params):
    """运行回测"""
    # 计算指标
    tp = (df['high'] + df['low'] + df['close']) / 3
    cci_length = params['cci_length']
    ma_length = params['ma_length']

    sma = tp.rolling(window=cci_length).mean()
    mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean()
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()

    df['stc'] = calculate_stc(df['close'])

    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    # 回测循环
    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        cci = current['cci']
        cci_ma = current['cci_ma']
        stc = current['stc']
        vol_ratio = current['vol_ratio']

        cci_prev = df.iloc[i-1]['cci'] if i > 0 else cci
        cci_ma_prev = df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

        # 计算权益
        if position > 0:
            unrealized_pnl = (current['close'] - entry_price) * position * MULTIPLIER
            current_equity = balance + unrealized_pnl
        else:
            current_equity = balance
        equity.append(current_equity)

        # 平仓条件
        if position > 0:
            exit_triggered = False

            # 1. 止损(-4%)
            if next_day['low'] <= entry_price * 0.96:
                stop_price = max(next_day['open'], entry_price * 0.96)
                exit_price = stop_price * (1 - SLIPPAGE_RATE)
                pnl = (exit_price - entry_price) * position * MULTIPLIER
                commission = exit_price * position * MULTIPLIER * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({'pnl': net_pnl, 'type': 'stop_loss'})
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # 2. 止盈(20%)
            if not exit_triggered and next_day['high'] >= entry_price * 1.20:
                take_profit_price = min(next_day['open'], entry_price * 1.20)
                exit_price = take_profit_price * (1 - SLIPPAGE_RATE)
                pnl = (exit_price - entry_price) * position * MULTIPLIER
                commission = exit_price * position * MULTIPLIER * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({'pnl': net_pnl, 'type': 'take_profit'})
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # 3. CCI超买平仓
            if not exit_triggered and cci > params['cci_overbought']:
                exit_price = next_day['close'] * (1 - SLIPPAGE_RATE)
                pnl = (exit_price - entry_price) * position * MULTIPLIER
                commission = exit_price * position * MULTIPLIER * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({'pnl': net_pnl, 'type': 'overbought'})
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # 4. CCI死叉平仓
            if not exit_triggered and (cci_prev >= cci_ma_prev and cci < cci_ma):
                exit_price = next_day['close'] * (1 - SLIPPAGE_RATE)
                pnl = (exit_price - entry_price) * position * MULTIPLIER
                commission = exit_price * position * MULTIPLIER * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({'pnl': net_pnl, 'type': 'death_cross'})
                position = 0
                entry_price = 0.0
                exit_triggered = True

        # 开仓逻辑
        if position == 0:
            open_signal = False

            if (cci < params['cci_oversold'] and
                stc >= params['stc_oversold']):
                open_signal = True

            elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                  cci <= params['cci_cross_max'] and
                  stc >= params['stc_cross']):
                open_signal = True

            if open_signal:
                if params['vol_threshold'] > 0 and not np.isnan(vol_ratio) and vol_ratio < params['vol_threshold']:
                    pass
                else:
                    entry_price = next_day['open'] * (1 + SLIPPAGE_RATE)
                    max_value = balance * 0.9 * LEVERAGE
                    qty = int(max_value / (entry_price * MULTIPLIER))
                    qty = max(1, qty)
                    commission = entry_price * qty * MULTIPLIER * COMMISSION_RATE
                    balance -= commission
                    position = qty

    # 计算结果
    trades_df = pd.DataFrame(trades)
    total_return = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    if len(trades_df) > 0:
        equity_series = pd.Series(equity)
        max_equity = equity_series.cummax()
        drawdown = (equity_series - max_equity) / max_equity * 100
        max_dd = drawdown.min()

        win_trades = trades_df[trades_df['pnl'] > 0]
        lose_trades = trades_df[trades_df['pnl'] < 0]

        win_rate = len(win_trades) / len(trades_df) * 100
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = lose_trades['pnl'].mean() if len(lose_trades) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        profit_factor = abs(win_trades['pnl'].sum() / lose_trades['pnl'].sum()) if lose_trades['pnl'].sum() != 0 else 0
        expected_value = trades_df['pnl'].mean()

        return {
            'total_return': total_return,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expected_value': expected_value,
            'total_trades': len(trades_df),
            'final_balance': balance,
        }
    else:
        return None


def quick_optimization(df):
    """快速优化 - 只测试核心参数"""
    print("\n" + "=" * 100)
    print("快速参数优化".center(100))
    print("=" * 100)

    # 固定参数（基于原值，减少搜索空间）
    fixed_params = {
        'cci_length': 12,
        'ma_length': 5,
    }

    # 待优化参数（核心影响参数）
    param_ranges = {
        'cci_oversold': [-70, -60, -50, -40, -30],  # 5个值
        'cci_overbought': [150, 170, 190, 210],  # 4个值
        'cci_cross_max': [140, 160, 180, 200],  # 4个值
        'stc_oversold': [-60, -50, -40],  # 3个值
        'stc_cross': [-140, -130, -120],  # 3个值
        'vol_threshold': [0.0, 0.3, 0.5, 0.7],  # 4个值
    }

    total_combos = (5 * 4 * 4 * 3 * 3 * 4)  # 2880个组合
    print(f"\n参数组合总数: {total_combos:,}")
    print(f"预计耗时: {total_combos * 0.5 / 60:.1f}分钟")
    print(f"优化策略: 粗筛 + 细筛")

    results = []
    count = 0

    print("\n开始优化...")
    start_time = datetime.now()

    for cci_os in param_ranges['cci_oversold']:
        for cci_ob in param_ranges['cci_overbought']:
            for cci_cm in param_ranges['cci_cross_max']:
                # 粗筛：快速过滤明显不合理的组合
                if cci_os >= -30 and cci_ob <= 150:
                    continue  # 超卖太高且超买太低，矛盾

                for stc_os in param_ranges['stc_oversold']:
                    for stc_cr in param_ranges['stc_cross']:
                        for vol_th in param_ranges['vol_threshold']:

                            params = {
                                'cci_length': fixed_params['cci_length'],
                                'ma_length': fixed_params['ma_length'],
                                'cci_oversold': cci_os,
                                'cci_overbought': cci_ob,
                                'cci_cross_max': cci_cm,
                                'stc_oversold': stc_os,
                                'stc_cross': stc_cr,
                                'vol_threshold': vol_th,
                            }

                            result = run_backtest(df.copy(), params)

                            if result:
                                results.append({
                                    'params': params,
                                    'metrics': result,
                                })
                                count += 1

                                if count % 100 == 0:
                                    elapsed = (datetime.now() - start_time).total_seconds()
                                    remaining = elapsed / count * (total_combos - count) / 60
                                    print(f"进度: {count}/{total_combos} ({count/total_combos*100:.1f}%) - 预计剩余: {remaining:.1f}分钟")

    # 按收益率排序
    results.sort(key=lambda x: x['metrics']['total_return'], reverse=True)

    elapsed_total = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n优化完成! 总耗时: {elapsed_total:.1f}分钟")

    return results


def print_results(results):
    """打印优化结果"""
    print("\n" + "=" * 100)
    print("优化结果 TOP 20")
    print("=" * 100)

    print(f"\n{'排名':<6} {'收益率':>10} {'回撤':>10} {'胜率':>8} {'盈亏比':>8} {'交易':>6} {'盈亏因子':>10}")
    print("-" * 100)

    for i, res in enumerate(results[:20], 1):
        p = res['params']
        m = res['metrics']
        print(f"{i:<6} {m['total_return']:>8.2f}% {m['max_dd']:>10.2f}% {m['win_rate']:>8.1f}% {m['win_loss_ratio']:>8.2f} {m['total_trades']:>6} {m['profit_factor']:>10.2f}")

    # 最优参数详细
    best = results[0]
    print("\n" + "=" * 100)
    print("最优参数配置")
    print("=" * 100)

    print(f"\n收益率: {best['metrics']['total_return']:.2f}%")
    print(f"最大回撤: {best['metrics']['max_dd']:.2f}%")
    print(f"胜率: {best['metrics']['win_rate']:.2f}%")
    print(f"盈亏比: {best['metrics']['win_loss_ratio']:.2f}")
    print(f"盈亏因子: {best['metrics']['profit_factor']:.2f}")
    print(f"交易次数: {best['metrics']['total_trades']}")
    print(f"期望收益: ${best['metrics']['expected_value']:.2f}/笔")

    print("\n详细参数:")
    print(f"  cci_length: {best['params']['cci_length']}")
    print(f"  ma_length: {best['params']['ma_length']}")
    print(f"  cci_oversold: {best['params']['cci_oversold']}")
    print(f"  cci_overbought: {best['params']['cci_overbought']}")
    print(f"  cci_cross_max: {best['params']['cci_cross_max']}")
    print(f"  stc_oversold: {best['params']['stc_oversold']}")
    print(f"  stc_cross: {best['params']['stc_cross']}")
    print(f"  vol_threshold: {best['params']['vol_threshold']}")

    # 对比老参数
    print("\n" + "=" * 100)
    print("与老参数对比")
    print("=" * 100)
    old_params = {
        'cci_length': 12,
        'ma_length': 5,
        'cci_oversold': -40,
        'cci_overbought': 190,
        'cci_cross_max': 180,
        'stc_oversold': -50,
        'stc_cross': -130,
        'vol_threshold': 0.50,
    }
    print(f"\n老参数收益: -25.83%")
    print(f"新参数收益: {best['metrics']['total_return']:.2f}%")
    print(f"提升: {best['metrics']['total_return'] - (-25.83):.2f}%")

    print("\n" + "=" * 100)


def main():
    print("=" * 100)
    print("纯碱(SA0) 快速参数优化".center(100))
    print("=" * 100)
    print(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 获取数据
    df = get_akshare_data(SYMBOL)
    if df is None:
        print("[FAILED] 数据获取失败")
        return

    print(f"\n数据量: {len(df)}条")
    print(f"时间范围: {df.index[0].date()} 至 {df.index[-1].date()}")

    # 运行快速优化
    results = quick_optimization(df)

    # 打印结果
    print_results(results)

    print("\n" + "=" * 100)
    print(f"优化完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
