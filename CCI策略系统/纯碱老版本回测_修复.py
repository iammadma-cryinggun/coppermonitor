"""
纯碱(SA0) 老版本回测 - 使用老版本的完整逻辑
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

import pandas as pd
import numpy as np
from datetime import datetime

# ========================================
# 纯碱参数（来自OPTIMAL_PARAMS）
# ========================================
SA_PARAMS = {
    'code': 'sa0',
    'exchange': 'CZCE',
    'multiplier': 20,

    # CCI参数
    'cci_length': 12,
    'ma_length': 5,
    'cci_oversold': -40,
    'cci_overbought': 190,
    'cci_cross_max': 180,

    # STC参数
    'stc_oversold': -50,
    'stc_cross': -130,
}

# 交易成本（与老版本一致）
COMMISSION_RATE = 0.0003
SLIPPAGE_RATE = 0.0002
INITIAL_CAPITAL = 100000
LEVERAGE = 2

# 量能过滤阈值（老版本VOL_THRESHOLDS中纯碱的值）
VOL_THRESHOLD = 0.50  # 纯碱的量能阈值


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
        print(f"数据量: {len(df)}条")
        print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
        print(f"数据跨度: {(df.index[-1] - df.index[0]).days / 365:.1f}年")
        return df
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def calculate_stc(close_prices, fast_period=23, slow_period=50, cycle_period=10):
    """计算STC指标（与老版本一致）"""
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


def run_backtest(df, params, vol_threshold):
    """运行老版本回测（完整逻辑）"""

    # 计算指标
    print("计算指标...")
    tp = (df['high'] + df['low'] + df['close']) / 3
    cci_length = params['cci_length']
    ma_length = params['ma_length']

    sma = tp.rolling(window=cci_length).mean()
    mad = tp.rolling(window=cci_length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()

    df['stc'] = calculate_stc(df['close'])

    # 量比
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_date = None
    entry_cci = None
    trades = []
    equity = []
    skipped_low_vol = 0

    print("开始回测...")

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

        # 平仓条件检查（老版本完整逻辑）
        if position > 0:
            exit_triggered = False
            exit_price = None
            exit_type = None

            # 1. 止损（-4%）- 最高优先级
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

            # 2. 止盈（20%）- 第二优先级
            if not exit_triggered and next_day['high'] >= entry_price * 1.20:
                take_profit_price = min(next_day['open'], entry_price * 1.20)
                exit_price = take_profit_price * (1 - SLIPPAGE_RATE)
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
                    'type': 'take_profit',
                    'hold_days': (next_day.name - entry_date).days,
                    'cci_on_entry': entry_cci,
                    'cci_on_exit': cci,
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # 3. CCI超买平仓
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

            # 4. CCI死叉平仓
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

        # 开仓逻辑（老版本完整逻辑）
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
                # 量能过滤
                if vol_threshold > 0 and not np.isnan(vol_ratio) and vol_ratio < vol_threshold:
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


def print_results(result, params, vol_threshold):
    """打印详细回测结果"""

    print("\n" + "=" * 120)
    print("纯碱(SA0) 老版本回测结果 - 完整逻辑".center(120))
    print("=" * 120)

    if result is None:
        print("\n[FAILED] 回测失败或无交易")
        return

    print(f"\n数据时间范围: {result['trades_df']['entry_date'].min()} 至 {result['trades_df']['exit_date'].max()}")

    print("\n" + "=" * 120)
    print("整体表现")
    print("=" * 120)
    print(f"  总收益率:     {result['total_return']:>10.2f}%")
    print(f"  最大回撤:     {result['max_dd']:>10.2f}%")
    print(f"  收益回撤比:   {result['total_return'] / abs(result['max_dd']):>10.2f}")
    print(f"  胜率:         {result['win_rate']:>10.2f}%")
    print(f"  盈亏比:       {result['win_loss_ratio']:>10.2f}")
    print(f"  盈亏因子:     {result['profit_factor']:>10.2f}")
    print(f"  期望收益:     ${result['expected_value']:>10.2f}/笔")
    print(f"  总交易次数:   {result['total_trades']:>10}笔")
    print(f"  平均持仓天数: {result['avg_hold_days']:>10.1f}天")
    print(f"  跳过低量能: {result['skipped_low_vol']:>10}次")

    # 与文档对比
    print("\n" + "=" * 120)
    print("与文档预期对比")
    print("=" * 120)
    print(f"{'指标':<15} {'老版本结果':>15} {'文档预期':>15} {'差异':>15}")
    print("-" * 80)
    print(f"{'收益率':<15} {result['total_return']:>10.2f}%  {1075.48:>10.2f}%  {result['total_return'] - 1075.48:>10.2f}%")
    print(f"{'交易次数':<15} {result['total_trades']:>10}笔  {118:>10}笔  {result['total_trades'] - 118:>}笔")
    print(f"{'最大回撤':<15} {result['max_dd']:>10.2f}%  {-25.52:>10.2f}% {result['max_dd'] - (-25.52):>}%}")

    print("\n" + "=" * 120)
    print("平仓类型统计")
    print("=" * 120)
    print(f"  {'类型':<15} {'次数':>8} {'占比':>10} {'平均盈亏':>12} {'总盈亏':>14}")
    print("-" * 80)

    type_map = {
        'stop_loss': '止损(-4%)',
        'take_profit': '止盈(20%)',
        'overbought': 'CCI超买平仓',
        'death_cross': 'CCI死叉平仓'
    }

    for exit_type, count in result['exit_stats'].items():
        pct = count / result['total_trades'] * 100
        type_trades = result['trades_df'][result['trades_df']['type'] == exit_type]
        avg_pnl = type_trades['pnl'].mean()
        total_pnl = type_trades['pnl'].sum()
        print(f"  {type_map.get(exit_type, exit_type):<15} {count:>8} {pct:>9.1f}% {avg_pnl:>11.0f} {total_pnl:>13.0f}")

    print("\n" + "=" * 120)
    print("止盈止损规则（老版本完整）")
    print("=" * 120)
    print("  平仓优先级（从高到低）:")
    print("    1. 止损(-4%):        日内最低价 <= 开仓价 × 0.96")
    print("    2. 止盈(20%):        日内最高价 >= 开仓价 × 1.20")
    print("    3. CCI超买平仓:      CCI > 190")
    print("    4. CCI死叉平仓:      CCI从上往下穿越CCI_MA")

    print("\n" + "=" * 120)
    print(f"参数配置")
    print("=" * 120)
    print(f" CCI参数: length={params['cci_length']}, MA={params['ma_length']}")
    print(f" 超卖/超买: {params['cci_oversold']} / {params['cci_overbought']}")
    print(f" 金叉上限: {params['cci_cross_max']}")
    print(f" STC参数: oversold={params['stc_oversold']}, cross={params['stc_cross']}")
    print(f" 量能过滤: vol_ratio >= {vol_threshold}")
    print(f" 杠杆: {LEVERAGE}x, 乘数: {params['multiplier']}")

    print("\n" + "=" * 120)


def main():
    print("=" * 120)
    print("纯碱(SA0) 老版本回测 - 完整逻辑".center(120))
    print("=" * 120)
    print(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 获取数据
    print(f"\n[1/2] 获取纯碱数据...")
    df = get_akshare_data(SA_PARAMS['code'])

    if df is None:
        print("[FAILED] 数据获取失败")
        return

    # 运行回测
    print(f"\n[2/2] 运行回测...")
    result = run_backtest(df, SA_PARAMS, VOL_THRESHOLD)

    if result:
        # 输出结果
        print(f"\n[2/2] 输出结果...")
        print_results(result, SA_PARAMS, VOL_THRESHOLD)
    else:
        print("\n[FAILED] 回测失败或无交易")

    print("\n" + "=" * 120)
    print(f"回测完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)


if __name__ == "__main__":
    main()
