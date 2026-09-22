"""
纯碱(SA0) 完整历史回测 - akshare全部数据
使用1502条数据（2019-2026，约7年）
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# 纯碱最新参数配置
SA_PARAMS = {
    'code': 'sa0',
    'exchange': 'CZCE',
    'multiplier': 20,

    # CCI参数
    'cci_length': 12,
    'ma_length': 5,
    'cci_oversold': -40,
    'cci_overbought': 190,  # 超买平仓阈值
    'cci_cross_max': 180,

    # STC参数
    'stc_oversold': -50,
    'stc_cross': -130,

    # 量能过滤
    'vol_threshold': 0.50,

    # 论文因子
    'use_paper_factor': True,
    'paper_factor_type': 'gap',
    'gap_threshold': -0.15,  # 隔夜缺口因子
}

# 交易成本
COMMISSION_RATE = 0.0003  # 万三手续费
SLIPPAGE_RATE = 0.0002    # 万二滑点
INITIAL_CAPITAL = 100000
LEVERAGE = 2  # 2倍杠杆


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

        # 不限制数据量，使用全部数据
        print(f"获取{len(df)}条完整历史数据")
        print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
        print(f"数据跨度: {(df.index[-1] - df.index[0]).days / 365:.1f}年")

        return df
    except Exception as e:
        print(f"[ERROR] 获取数据失败: {e}")
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

    # 归一化到STC范围
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


def calculate_overnight_gap_factor(df, lookback=10):
    """计算隔夜缺口因子"""
    gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    high_max = df['high'].rolling(lookback).max()
    low_min = df['low'].rolling(lookback).min()
    true_range = high_max - low_min
    normalized_gap = gap / (true_range / df['close'])
    return normalized_gap


def run_backtest(df, params):
    """运行回测"""

    # 计算指标
    tp = (df['high'] + df['low'] + df['close']) / 3
    cci_length = params['cci_length']
    ma_length = params['ma_length']

    sma = tp.rolling(window=cci_length).mean()
    mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()

    df['stc'] = calculate_stc(df['close'])

    # 计算量比
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    # 计算论文因子
    if params.get('use_paper_factor', False):
        df['gap_factor'] = calculate_overnight_gap_factor(df)

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_date = None
    trades = []
    equity = []
    skipped_low_vol = 0

    print("\n开始回测...")

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        cci = current['cci']
        cci_ma = current['cci_ma']
        stc = current['stc']
        vol_ratio = current['vol_ratio']

        cci_prev = df.iloc[i-1]['cci'] if i > 0 else cci
        cci_ma_prev = df.iloc[i-1]['cci_ma'] if i > 0 else cci_ma

        # 计算当前权益
        if position > 0:
            unrealized_pnl = (current['close'] - entry_price) * position * params['multiplier']
            current_equity = balance + unrealized_pnl
        else:
            current_equity = balance
        equity.append(current_equity)

        # 平仓条件检查
        if position > 0:
            exit_triggered = False
            exit_price = None
            exit_type = None

            # 1. 止损（-4%）
            if next_day['low'] <= entry_price * 0.96:
                stop_price = max(next_day['open'], entry_price * 0.96)
                exit_price = stop_price * (1 - SLIPPAGE_RATE)
                pnl = (exit_price - entry_price) * position * params['multiplier']
                commission = exit_price * position * params['multiplier'] * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'stop_loss',
                    'hold_days': (next_day.name - entry_date).days,
                    'cci_on_entry': entry_cci,
                    'cci_on_exit': cci,
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # 2. CCI超买平仓（主要止盈）
            if not exit_triggered and cci > params['cci_overbought']:
                exit_price = next_day['close'] * (1 - SLIPPAGE_RATE)
                pnl = (exit_price - entry_price) * position * params['multiplier']
                commission = exit_price * position * params['multiplier'] * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'overbought',
                    'hold_days': (next_day.name - entry_date).days,
                    'cci_on_entry': entry_cci,
                    'cci_on_exit': cci,
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # 3. CCI死叉平仓
            if not exit_triggered and (cci_prev >= cci_ma_prev and cci < cci_ma):
                exit_price = next_day['close'] * (1 - SLIPPAGE_RATE)
                pnl = (exit_price - entry_price) * position * params['multiplier']
                commission = exit_price * position * params['multiplier'] * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance += net_pnl
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': 'death_cross',
                    'hold_days': (next_day.name - entry_date).days,
                    'cci_on_entry': entry_cci,
                    'cci_on_exit': cci,
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

        # 开仓逻辑
        if position == 0:
            open_signal = False

            # 超卖开仓（CCI + STC 双条件）
            if (cci < params['cci_oversold'] and
                stc >= params['stc_oversold']):
                open_signal = True

            # 金叉开仓（CCI + STC 双条件）
            elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                  cci <= params['cci_cross_max'] and
                  stc >= params['stc_cross']):
                open_signal = True

            if open_signal:
                # 论文因子过滤
                factor_pass = True
                if params.get('use_paper_factor', False):
                    gap_th = params.get('gap_threshold')
                    if gap_th is not None:
                        if current['gap_factor'] < gap_th:
                            factor_pass = False

                # 量能过滤
                if factor_pass:
                    if params['vol_threshold'] > 0 and not np.isnan(vol_ratio):
                        if vol_ratio < params['vol_threshold']:
                            skipped_low_vol += 1
                        else:
                            entry_price = next_day['open'] * (1 + SLIPPAGE_RATE)
                            max_value = balance * 0.9 * LEVERAGE
                            qty = int(max_value / (entry_price * params['multiplier']))
                            qty = max(1, qty)
                            commission = entry_price * qty * params['multiplier'] * COMMISSION_RATE
                            balance -= commission

                            position = qty
                            entry_date = next_day.name
                            entry_cci = cci
                else:
                    skipped_low_vol += 1

    # 计算最终指标
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

        # 平均持仓天数
        avg_hold_days = trades_df['hold_days'].mean()

        # 平仓类型统计
        exit_stats = trades_df['type'].value_counts()

        return {
            'total_return': total_return,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expected_value': expected_value,
            'total_trades': len(trades_df),
            'avg_hold_days': avg_hold_days,
            'exit_stats': exit_stats,
            'trades_df': trades_df,
            'skipped_low_vol': skipped_low_vol,
            'final_balance': balance,
        }
    else:
        return None


def print_results(result, params):
    """打印回测结果"""

    print("\n" + "=" * 100)
    print("纯碱(SA0) 完整历史回测结果".center(100))
    print("=" * 100)

    print(f"\n数据时间范围: {result['trades_df']['entry_date'].min()} 至 {result['trades_df']['exit_date'].max()}")

    print("\n" + "=" * 100)
    print("整体表现")
    print("=" * 100)
    print(f"  总收益率:     {result['total_return']:>10.2f}%")
    print(f"  最大回撤:     {result['max_dd']:>10.2f}%")
    print(f"  收益回撤比:   {result['total_return'] / abs(result['max_dd']):>10.2f}")
    print(f"  胜率:         {result['win_rate']:>10.2f}%")
    print(f"  盈亏比:       {result['win_loss_ratio']:>10.2f}")
    print(f"  盈亏因子:     {result['profit_factor']:>10.2f}")
    print(f"  期望收益:     ${result['expected_value']:>10.2f}/笔")
    print(f"  总交易次数:   {result['total_trades']:>10}笔")
    print(f"  平均持仓天数: {result['avg_hold_days']:>10.1f}天")
    print(f"  量能过滤跳过: {result['skipped_low_vol']:>10}次")

    print("\n" + "=" * 100)
    print("平仓类型统计")
    print("=" * 100)
    print(f"  {'类型':<15} {'次数':>8} {'占比':>10} {'平均盈亏':>12} {'总盈亏':>14}")
    print("-" * 80)

    type_map = {
        'stop_loss': '止损(-4%)',
        'overbought': 'CCI超买平仓',
        'death_cross': 'CCI死叉平仓'
    }

    for exit_type, count in result['exit_stats'].items():
        pct = count / result['total_trades'] * 100
        type_trades = result['trades_df'][result['trades_df']['type'] == exit_type]
        avg_pnl = type_trades['pnl'].mean()
        total_pnl = type_trades['pnl'].sum()
        print(f"  {type_map.get(exit_type, exit_type):<15} {count:>8} {pct:>9.1f}% {avg_pnl:>11.0f} {total_pnl:>13.0f}")

    print("\n" + "=" * 100)
    print("止盈止损规则")
    print("=" * 100)
    print("  1. 止损(-4%):    日内最低价 <= 开仓价 × 0.96")
    print("  2. CCI超买平仓:   CCI > 190 (主要止盈方式)")
    print("  3. CCI死叉平仓:   CCI从上往下穿越CCI_MA")

    print("\n" + "=" * 100)


def main():
    print("=" * 100)
    print("纯碱(SA0) 完整历史回测 - 全部数据".center(100))
    print("=" * 100)
    print(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # 获取数据
    print(f"\n[1/3] 获取纯碱完整数据...")
    df = get_akshare_data(SA_PARAMS['code'])

    if df is None:
        print("[FAILED] 数据获取失败")
        return

    # 运行回测
    print(f"\n[2/3] 运行回测...")
    result = run_backtest(df, SA_PARAMS)

    if result:
        # 输出结果
        print(f"\n[3/3] 输出结果...")
        print_results(result, SA_PARAMS)

        # 保存详细交易记录
        output_file = 'D:\\期货数据\\铜期货监控\\CCI策略系统\\纯碱完整回测详细记录.csv'
        result['trades_df'].to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细交易记录已保存至: {output_file}")
    else:
        print("\n[FAILED] 回测失败或无交易")

    print("\n" + "=" * 100)
    print(f"回测完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
