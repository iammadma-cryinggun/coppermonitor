# -*- coding: utf-8 -*-
"""
纯碱SA0 - S.C.C.系统 v4.0 狙击版
核心：全仓进出，吃完鱼身就跑（电梯盘专用）
"""
import pandas as pd
import numpy as np
from datetime import datetime

SYMBOL = 'sa0'
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
SLIPPAGE_RATE = 0.0002
INITIAL_CAPITAL = 100000
MARGIN_RATE = 0.10

# ==========================================
# v4.0 狙击版参数（全仓进出）
# ==========================================
PARAMS = {
    # 共享参数
    'cci_length': 12,
    'cci_ma_length': 5,
    'atr_length': 14,
    'vol_ma_period': 5,
    'risk_per_trade': 0.02,
    'multiplier': 20,

    # 做多参数：结构猎人狙击版v3（STC+CCI深度优化）
    'long_stc_oversold': -30,
    'long_stc_awake': -15,        # 提高到-15（要求更强的抬头）
    'long_stc_slope_min': 3,       # 提高到3（要求更陡的斜率）
    'long_cci_oversold': -100,
    'long_cci_zero_cross': -60,     # 提高到-60（要求更强的突破）
    'long_cci_confirm_days': 1,     # 保持1天确认
    'long_vol_threshold': 0.9,
    'long_require_bullish': True,
    'long_take_profit_atr': 2.0,
    'long_cci_exit_threshold': 100,
    'long_stop_loss_atr': 1.0,        # 进一步收紧到1.0倍ATR

    # 做空参数：重力加速器狙击版v3（STC+CCI深度优化）
    'short_stc_oversold': 20,       # 降低到20（更容易触发）
    'short_stc_awake': 0,           # 跌破0触发（更灵敏）
    'short_stc_slope_min': -3,       # 提高到-3（要求更陡的下坡）
    'short_cci_oversold': 60,       # 降低到60（更容易触发）
    'short_cci_zero_cross': 0,      # 跌破0就动手（中线破位）
    'short_cci_confirm_days': 0,
    'short_vol_threshold': 0.5,     # 提高到0.5（略微过滤）
    'short_require_bullish': False,
    'short_take_profit_atr': 1.5,
    'short_cci_exit_threshold': -100,
    'short_stop_loss_atr': 1.2,
}


def get_data():
    import akshare as ak
    df = ak.futures_main_sina(symbol=SYMBOL)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.dropna()
    df.set_index('date', inplace=True)
    return df


def calc_stc(df, fast_period=23, slow_period=50, cycle_period=10):
    """计算STC"""
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    lowest_macd = macd.rolling(window=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period).max()
    range_macd = highest_macd - lowest_macd

    stc = pd.Series(index=df.index, dtype=float)
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


def calc_atr(df, period=14):
    """计算ATR"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def run_backtest_v4_sniper(df, params):
    """v4.0 狙击版回测（全仓进出）"""

    # 计算指标
    print("计算指标...")

    df['stc'] = calc_stc(df)
    df['stc_prev'] = df['stc'].shift(1)
    df['stc_slope'] = df['stc'] - df['stc_prev']

    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=params['cci_length']).mean()
    mad = tp.rolling(window=params['cci_length']).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=params['cci_ma_length']).mean()
    df['cci_prev'] = df['cci'].shift(1)

    df['atr'] = calc_atr(df, params['atr_length'])

    df['vol_ma'] = df['volume'].rolling(window=params['vol_ma_period']).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['is_bullish_candle'] = df['close'] > df['open']

    # 交易状态变量
    balance = INITIAL_CAPITAL
    position = 0  # 正数=多头，负数=空头
    entry_price = 0.0
    entry_date = None
    entry_atr = 0.0
    position_type = None  # 'long' or 'short'

    trades = []
    equity = []

    print("开始回测...")

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        # 计算当前权益
        if position != 0:
            if position_type == 'long':
                unrealized_pnl = (current['close'] - entry_price) * abs(position) * params['multiplier']
            else:
                unrealized_pnl = (entry_price - current['close']) * abs(position) * params['multiplier']
            current_equity = balance + unrealized_pnl
        else:
            current_equity = balance
        equity.append(current_equity)

        # ========================================================
        # 平仓逻辑（全仓进出 - 狙击模式）
        # ========================================================
        if position != 0:
            # 计算浮盈（ATR倍数）
            if position_type == 'long':
                current_profit_atr = (current['close'] - entry_price) / entry_atr
            else:
                current_profit_atr = (entry_price - current['close']) / entry_atr

            # 获取当前方向的参数
            if position_type == 'long':
                take_profit_atr = params['long_take_profit_atr']
                cci_exit_threshold = params['long_cci_exit_threshold']
                stop_loss_atr = params['long_stop_loss_atr']
            else:
                take_profit_atr = params['short_take_profit_atr']
                cci_exit_threshold = params['short_cci_exit_threshold']
                stop_loss_atr = params['short_stop_loss_atr']

            exit_triggered = False
            exit_reason = None

            # -------------------------------------------
            # 1. 止盈（Take Profit） - 狙击全跑
            # -------------------------------------------
            # 条件A: 吃够了波动幅度（ATR止盈）
            tp_hit_atr = current_profit_atr > take_profit_atr

            # 条件B: 指标过热（CCI止盈）
            # 做多: CCI > 120; 做空: CCI < -120
            if position_type == 'long':
                tp_hit_cci = current['cci'] > cci_exit_threshold
            else:
                tp_hit_cci = current['cci'] < cci_exit_threshold

            if tp_hit_atr or tp_hit_cci:
                # 【全仓止盈】
                exit_price = current['close']
                if position_type == 'long':
                    pnl = (exit_price - entry_price) * abs(position) * params['multiplier']
                else:
                    pnl = (entry_price - exit_price) * abs(position) * params['multiplier']

                commission = exit_price * abs(position) * params['multiplier'] * COMMISSION_RATE * 2
                net_pnl = pnl - commission
                balance += net_pnl

                if tp_hit_atr:
                    exit_reason = f'atr_tp_{take_profit_atr}'
                else:
                    exit_reason = f'cci_tp_{cci_exit_threshold}'

                trades.append({
                    'direction': position_type,
                    'entry_date': entry_date,
                    'exit_date': current.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': exit_reason,
                    'hold_days': (current.name - entry_date).days,
                    'exit_qty': abs(position),
                })

                print(f"{position_type.upper()}止盈: {current.name.date()} 价格:{exit_price:.2f} CCI:{current['cci']:.2f} "
                      f"浮盈:{current_profit_atr:.2f}ATR 盈利:{net_pnl:.0f} ({exit_reason})")

                position = 0
                entry_price = 0.0
                position_type = None
                exit_triggered = True

            # -------------------------------------------
            # 2. 止损（Stop Loss） - 硬止损
            # -------------------------------------------
            if not exit_triggered:
                if position_type == 'long':
                    sl_price = entry_price - (stop_loss_atr * entry_atr)
                    hit_sl = next_day['low'] <= sl_price
                    if hit_sl:
                        exit_price = max(next_day['open'], sl_price)
                else:
                    sl_price = entry_price + (stop_loss_atr * entry_atr)
                    hit_sl = next_day['high'] >= sl_price
                    if hit_sl:
                        exit_price = min(next_day['open'], sl_price)

                if hit_sl:
                    if position_type == 'long':
                        pnl = (exit_price - entry_price) * abs(position) * params['multiplier']
                    else:
                        pnl = (entry_price - exit_price) * abs(position) * params['multiplier']

                    commission = exit_price * abs(position) * params['multiplier'] * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl

                    trades.append({
                        'direction': position_type,
                        'entry_date': entry_date,
                        'exit_date': next_day.name,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': f'stop_loss_{stop_loss_atr}atr',
                        'hold_days': (next_day.name - entry_date).days,
                        'exit_qty': abs(position),
                    })

                    print(f"{position_type.upper()}止损: {next_day.name.date()} 价格:{exit_price:.2f} "
                          f"止损:{sl_price:.2f} 亏损:{net_pnl:.0f}")

                    position = 0
                    entry_price = 0.0
                    position_type = None
                    exit_triggered = True

        # ========================================================
        # 做多开仓逻辑（狙击版）
        # ========================================================
        if position == 0:
            # STC: 从底部抬头
            stc_prev_oversold = (df.iloc[i-1]['stc'] < params['long_stc_oversold']) if i > 0 else False
            stc_current_awake = current['stc'] > params['long_stc_awake']
            stc_slope_strong = current['stc_slope'] > params['long_stc_slope_min']
            condition_stc = stc_prev_oversold and stc_current_awake and stc_slope_strong

            # 量能: 必须放量（提高到1.0倍）
            vol_sufficient = current['vol_ratio'] > params['long_vol_threshold']
            is_bullish = current['is_bullish_candle'] if params['long_require_bullish'] else True
            condition_vol = vol_sufficient and is_bullish

            # CCI: 1日确认（改快）
            if i > 0:
                cci_breakout = (current['cci'] > params['long_cci_zero_cross'] and
                               current['cci'] > current['cci_ma'])
                # 简化：只要CCI突破就行，不要求连续2天
                condition_cci = cci_breakout
            else:
                condition_cci = False

            if condition_stc and condition_vol and condition_cci:
                entry_price = next_day['open'] * (1 + SLIPPAGE_RATE)
                entry_atr = current['atr']
                stop_distance = entry_atr * params['long_stop_loss_atr']

                # 计算仓位
                risk_amount = balance * params['risk_per_trade']
                max_value = balance / MARGIN_RATE
                qty_by_risk = int(risk_amount / (stop_distance * params['multiplier']))
                qty_by_capital = int(max_value / (entry_price * params['multiplier']))
                qty = min(max(1, qty_by_risk), qty_by_capital)

                commission = entry_price * qty * params['multiplier'] * COMMISSION_RATE
                balance -= commission

                position = qty
                entry_date = next_day.name
                position_type = 'long'

                print(f"开多: {entry_date.date()} 价格:{entry_price:.2f} "
                      f"ATR止损:{entry_price - stop_distance:.2f} CCI:{current['cci']:.2f} "
                      f"STC:{current['stc']:.2f}(斜:{current['stc_slope']:.2f}) 量比:{current['vol_ratio']:.2f}")

        # ========================================================
        # 做空开仓逻辑（狙击版）
        # ========================================================
        if position == 0:
            # STC: 从高位掉头（修正：从80掉破70）
            stc_was_high = (df.iloc[i-1]['stc'] > params['short_stc_oversold']) if i > 0 else False
            stc_breakdown = current['stc'] < params['short_stc_awake']
            stc_slope_down = current['stc_slope'] < params['short_stc_slope_min']
            condition_stc = stc_was_high and stc_breakdown and stc_slope_down

            # 量能: 不需要放量
            vol_ok = current['vol_ratio'] > params['short_vol_threshold']

            # CCI: 从100跌破50（比原来快）
            cci_was_high = (df.iloc[i-1]['cci'] > params['short_cci_oversold']) if i > 0 else False
            cci_break_threshold = current['cci'] < params['short_cci_zero_cross']
            condition_cci = cci_was_high and cci_break_threshold

            # 不需要强制阴线
            if condition_stc and vol_ok and condition_cci:
                entry_price = next_day['open'] * (1 - SLIPPAGE_RATE)
                entry_atr = current['atr']
                stop_distance = entry_atr * params['short_stop_loss_atr']

                # 计算仓位
                risk_amount = balance * params['risk_per_trade']
                max_value = balance / MARGIN_RATE
                qty_by_risk = int(risk_amount / (stop_distance * params['multiplier']))
                qty_by_capital = int(max_value / (entry_price * params['multiplier']))
                qty = min(max(1, qty_by_risk), qty_by_capital)

                commission = entry_price * qty * params['multiplier'] * COMMISSION_RATE
                balance -= commission

                position = qty
                entry_date = next_day.name
                position_type = 'short'

                print(f"开空: {entry_date.date()} 价格:{entry_price:.2f} "
                      f"ATR止损:{entry_price + stop_distance:.2f} CCI:{current['cci']:.2f} "
                      f"STC:{current['stc']:.2f}(斜:{current['stc_slope']:.2f}) 量比:{current['vol_ratio']:.2f}")

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

        avg_hold_days = trades_df['hold_days'].mean()

        exit_stats = trades_df['type'].value_counts()

        long_trades = trades_df[trades_df['direction'] == 'long']
        short_trades = trades_df[trades_df['direction'] == 'short']

        long_pnl = long_trades['pnl'].sum() if len(long_trades) > 0 else 0
        short_pnl = short_trades['pnl'].sum() if len(short_trades) > 0 else 0

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
            'final_balance': balance,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'long_win_rate': len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0,
            'short_win_rate': len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0,
        }
    else:
        return None


def print_results(result, params):
    """打印详细回测结果"""

    print("\n" + "=" * 100)
    print("纯碱SA0 - S.C.C.系统 v4.0 狙击版".center(100))
    print("全仓进出，吃完鱼身就跑（电梯盘专用）".center(100))
    print("=" * 100)

    if result is None:
        print("\n[FAILED] 回测失败或无交易")
        return

    print(f"\n数据时间范围: {result['trades_df']['entry_date'].min()} 至 {result['trades_df']['exit_date'].max()}")

    print("\n" + "=" * 100)
    print("整体表现")
    print("=" * 100)
    print(f"  总收益率:     {result['total_return']:>10.2f}%")
    print(f"  最大回撤:     {result['max_dd']:>10.2f}%")
    if result['max_dd'] != 0:
        print(f"  收益回撤比:   {result['total_return'] / abs(result['max_dd']):>10.2f}")
    print(f"  胜率:         {result['win_rate']:>10.2f}%")
    print(f"  盈亏比:       {result['win_loss_ratio']:>10.2f}")
    print(f"  盈亏因子:     {result['profit_factor']:>10.2f}")
    print(f"  期望收益:     ${result['expected_value']:>10.2f}/笔")
    print(f"  总交易次数:   {result['total_trades']:>10}笔")
    print(f"  平均持仓天数: {result['avg_hold_days']:>10.1f}天")

    print("\n" + "=" * 100)
    print("多空不对称对比")
    print("=" * 100)
    print(f"  {'方向':<8} {'交易次数':>10} {'胜率':>10} {'总盈亏':>15}")
    print("-" * 50)
    print(f"  {'多头':<8} {result['long_trades']:>10}笔 {result['long_win_rate']:>9.2f}% {result['long_pnl']:>14.0f}元")
    print(f"  {'空头':<8} {result['short_trades']:>10}笔 {result['short_win_rate']:>9.2f}% {result['short_pnl']:>14.0f}元")

    print("\n" + "=" * 100)
    print("平仓类型统计")
    print("=" * 100)
    print(f"  {'类型':<30} {'次数':>8} {'占比':>10} {'平均盈亏':>12} {'总盈亏':>14}")
    print("-" * 90)

    type_map = {
        'atr_tp_2.0': '多ATR止盈(2.0倍）',
        'cci_tp_100': '多CCI止盈(>100）',
        'atr_tp_1.5': '空ATR止盈(1.5倍）',
        'cci_tp_-100': '空CCI止盈(<-100）',
        'stop_loss_1.0atr': '多ATR止损(1.0倍）',
        'stop_loss_1.2atr': '空ATR止损(1.2倍）',
    }

    for exit_type, count in result['exit_stats'].items():
        pct = count / result['total_trades'] * 100
        type_trades = result['trades_df'][result['trades_df']['type'] == exit_type]
        avg_pnl = type_trades['pnl'].mean()
        total_pnl = type_trades['pnl'].sum()
        print(f"  {type_map.get(exit_type, exit_type):<30} {count:>8} {pct:>9.1f}% {avg_pnl:>11.0f} {total_pnl:>13.0f}")

    print("\n" + "=" * 100)
    print("v4.0 狙击版核心逻辑")
    print("=" * 100)
    print("  【电梯盘哲学】")
    print("  - 纯碱脉冲式行情，吃完鱼身就跑")
    print("  - 不做时间的朋友，时间越长风险越大")
    print("  - 全仓进出，宁可错过，绝不做电梯")
    print("")
    print("  做多（结构猎人狙击v3 - STC+CCI深度优化）：")
    print("    STC: 从底部抬头(<-30 → >-15) + 斜率>3（更严格）")
    print("    量能: 必须0.9倍放量 + 必须阳线")
    print("    CCI: 突破-60（提高阈值）")
    print("    止盈: 2.0倍ATR 或 CCI>100 → 全跑")
    print("    止损: 1.0倍ATR（非常收紧）")
    print("")
    print("  做空（重力加速器狙击v3 - STC+CCI深度优化）：")
    print("    STC: 从高位掉头(>20 → <0) + 斜率<-3（更灵敏）")
    print("    量能: >0.5倍（略微过滤）")
    print("    CCI: 从60跌破0（中线破位）")
    print("    止盈: 1.5倍ATR 或 CCI<-100 → 全跑")
    print("    止损: 1.2倍ATR")

    print("\n" + "=" * 100)


def main():
    print("=" * 100)
    print("纯碱SA0 - S.C.C.系统 v4.0 狙击版".center(100))
    print("全仓进出，吃完鱼身就跑".center(100))
    print("=" * 100)
    print(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # 获取数据
    print(f"\n[1/2] 获取数据...")
    df = get_data()

    if df is None:
        print("[FAILED] 数据获取失败")
        return

    print(f"数据量: {len(df)}条")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")

    # 运行回测
    print(f"\n[2/2] 运行回测...")
    result = run_backtest_v4_sniper(df, PARAMS)

    if result:
        # 输出结果
        print(f"\n[2/2] 输出结果...")
        print_results(result, PARAMS)
    else:
        print("\n[FAILED] 回测失败或无交易")

    print("\n" + "=" * 100)
    print(f"回测完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
