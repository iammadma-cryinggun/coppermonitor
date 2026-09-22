"""
纯碱(SA0) 参数优化 - 重新寻找最优CCI+STC参数
基于当前市场环境重新优化，找出有效参数
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


def run_backtest(df, params, start_year=None, end_year=None):
    """运行回测"""

    # 筛选时间范围
    if start_year:
        df = df[df.index.year >= start_year].copy()
    if end_year:
        df = df[df.index.year <= end_year].copy()

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
    skipped_low_vol = 0

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
                    skipped_low_vol += 1
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


def grid_search(df, param_grid, start_year=None, end_year=None):
    """网格搜索最优参数"""
    results = []
    total = len(list(product(*param_grid.values())))

    print(f"\n参数组合总数: {total}")
    print(f"测试中...\n")

    count = 0
    for values in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), values))
        result = run_backtest(df.copy(), params, start_year, end_year)

        if result:
            results.append({
                'params': params,
                'metrics': result,
            })
            count += 1

            if count % 10 == 0 or count == total:
                print(f"进度: {count}/{total} ({count/total*100:.1f}%)")

    # 按收益率排序
    results.sort(key=lambda x: x['metrics']['total_return'], reverse=True)

    return results


def main():
    print("=" * 100)
    print("纯碱(SA0) 参数优化 - 网格搜索".center(100))
    print("=" * 100)
    print(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 获取数据
    df = get_akshare_data(SYMBOL)
    if df is None:
        print("[FAILED] 数据获取失败")
        return

    print(f"\n数据量: {len(df)}条")
    print(f"时间范围: {df.index[0].date()} 至 {df.index[-1].date()}")

    # 选择优化时间段
    print("\n请选择优化时间段:")
    print("  1. 全部数据 (2019-2025)")
    print("  2. 只用2023-2025年 (适应最新市场)")
    print("  3. 只用2019-2022年 (保持原盈利期)")

    # 默认用全部数据
    time_range = 'all'
    start_year = None
    end_year = None

    print(f"\n使用: 全部数据 (2019-2025)")

    # 定义参数搜索范围（精简版，减少搜索时间）
    print("\n定义参数搜索范围...")

    # CCI参数范围
    cci_length_range = [10, 12, 14, 16]
    ma_length_range = [3, 5, 7, 10]
    cci_oversold_range = [-60, -50, -40, -30]
    cci_overbought_range = [150, 170, 190, 210]
    cci_cross_max_range = [140, 160, 180, 200]

    # STC参数范围
    stc_oversold_range = [-60, -50, -40]
    stc_cross_range = [-140, -130, -120, -110]

    # 量能范围
    vol_threshold_range = [0.0, 0.3, 0.5, 0.7]

    param_grid = {
        'cci_length': cci_length_range,
        'ma_length': ma_length_range,
        'cci_oversold': cci_oversold_range,
        'cci_overbought': cci_overbought_range,
        'cci_cross_max': cci_cross_max_range,
        'stc_oversold': stc_oversold_range,
        'stc_cross': stc_cross_range,
        'vol_threshold': vol_threshold_range,
    }

    # 计算总组合数
    total_combinations = (len(cci_length_range) *
                        len(ma_length_range) *
                        len(cci_oversold_range) *
                        len(cci_overbought_range) *
                        len(cci_cross_max_range) *
                        len(stc_oversold_range) *
                        len(stc_cross_range) *
                        len(vol_threshold_range))

    print(f"\n警告: 参数组合总数 = {total_combinations:,}")
    print(f"预计耗时: {total_combinations * 0.5 / 60:.1f}分钟")

    if total_combinations > 10000:
        print("\n组合数过多，请先运行精简版或缩小范围")
        print("\n精简范围建议:")
        print("  cci_length: [12] (固定)")
        print("  ma_length: [5] (固定)")
        print("  只优化: cci_oversold, cci_overbought, cci_cross_max, stc参数, vol_threshold")

        # 使用精简范围
        response = input("\n是否使用精简范围? (y/n): ").strip().lower()
        if response == 'y':
            param_grid = {
                'cci_length': [12],
                'ma_length': [5],
                'cci_oversold': cci_oversold_range,
                'cci_overbought': cci_overbought_range,
                'cci_cross_max': cci_cross_max_range,
                'stc_oversold': stc_oversold_range,
                'stc_cross': stc_cross_range,
                'vol_threshold': vol_threshold_range,
            }

    # 开始搜索
    print("\n开始网格搜索...")
    results = grid_search(df, param_grid, start_year, end_year)

    # 显示结果
    print("\n" + "=" * 100)
    print("优化结果 TOP 10")
    print("=" * 100)

    print(f"\n{'排名':<6} {'收益率':>10} {'回撤':>10} {'胜率':>8} {'盈亏比':>8} {'交易数':>8}")
    print("-" * 80)

    for i, res in enumerate(results[:10], 1):
        p = res['params']
        m = res['metrics']
        print(f"{i:<6} {m['total_return']:>8.2f}% {m['max_dd']:>10.2f}% {m['win_rate']:>8.1f}% {m['win_loss_ratio']:>8.2f} {m['total_trades']:>8}")

    # 保存最优参数
    best = results[0]
    print("\n" + "=" * 100)
    print("最优参数配置")
    print("=" * 100)
    print(f"\n收益率: {best['metrics']['total_return']:.2f}%")
    print(f"最大回撤: {best['metrics']['max_dd']:.2f}%")
    print(f"胜率: {best['metrics']['win_rate']:.2f}%")
    print(f"盈亏比: {best['metrics']['win_loss_ratio']:.2f}")
    print(f"交易次数: {best['metrics']['total_trades']}")

    print("\n参数:")
    for key, value in best['params'].items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
