# -*- coding: utf-8 -*-
"""
纯碱SA0 - S.C.C. Quant-Hybrid System v1.0
基于战术B（结构猎人）：专攻低波动/震荡品种
"""
import pandas as pd
import numpy as np
from datetime import datetime

SYMBOL = 'sa0'
MULTIPLIER = 20
COMMISSION_RATE = 0.0003
SLIPPAGE_RATE = 0.0002
INITIAL_CAPITAL = 100000
LEVERAGE = 2

# ==========================================
# 战术B参数：结构猎人
# ==========================================
TACTICS_B_PARAMS = {
    # COO参数（Composite Oscillator - 进场扳机）
    'coo_oversold': -40,          # COO超卖阈值（基于实际数据范围调整）
    'coo_divergence_lookback': 30, # 背离检测回看周期

    # STC参数（趋势指南针）
    'stc_floor': 10,              # STC地板区（趴窝状态）
    'stc_slope_threshold': 1,      # STC变化率阈值
    'stc_exit_floor': 10,          # STC确认脱离底部

    # CCI参数（爆发油门）
    'cci_length': 12,
    'cci_ma_length': 5,
    'cci_zero_cross': -120,        # CCI突破阈值

    # 风险管理
    'atr_length': 14,              # ATR周期
    'atr_stop_multiplier': 2.0,    # ATR止损倍数
    'risk_per_trade': 0.02,        # 每笔交易风险2%
    'multiplier': 20,               # 合约乘数

    # 波动率过滤
    'stddev_lookback': 20,
    'stddev_threshold': 1.5,       # StdDev阈值
}


def get_data():
    """获取历史数据"""
    import akshare as ak
    df = ak.futures_main_sina(symbol=SYMBOL)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.dropna()
    df.set_index('date', inplace=True)
    return df


def calc_coo(df, rsi_period=14, stoch_period=14):
    """
    计算COO (Composite Oscillator) - RSI + Stochastic 组合
    范围: -100 到 +100
    """
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Stochastic
    low_14 = df['low'].rolling(window=stoch_period).min()
    high_14 = df['high'].rolling(window=stoch_period).max()
    stoch = 100 * (df['close'] - low_14) / (high_14 - low_14)

    # COO = (RSI - 50) * 2  映射到 -100~+100
    coo = (rsi + stoch) / 2 - 50

    return coo


def calc_stc(df, fast_period=23, slow_period=50, cycle_period=10):
    """计算STC (Schaff Trend Cycle)"""
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


def calc_atr(df, period=14):
    """计算ATR (Average True Range)"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def detect_divergence(df, coo, lookback=20):
    """
    检测底背离：价格创新低，COO未创新低
    返回: pd.Series (True=出现背离)
    """
    divergence = pd.Series(False, index=df.index)

    for i in range(lookback, len(df)):
        # 当前低点
        current_price_low = df['low'].iloc[i]
        current_coo_low = coo.iloc[i]

        # 找到lookback周期内的前一个低点
        prev_window_price = df['low'].iloc[i-lookback:i]
        prev_window_coo = coo.iloc[i-lookback:i]

        prev_price_low_idx = prev_window_price.idxmin()
        prev_coo_low_idx = prev_window_coo.idxmin()

        prev_price_low = df['low'].loc[prev_price_low_idx]
        prev_coo_low = coo.loc[prev_coo_low_idx]

        # 背离条件：
        # 1. 价格创新低或持平
        # 2. COO低点明显抬高（>5点）
        price_new_low = current_price_low <= prev_price_low * 1.01
        coo_higher = current_coo_low > prev_coo_low + 5

        divergence.iloc[i] = price_new_low and coo_higher

    return divergence


def calc_stddev_squeeze(df, lookback=20):
    """
    波动率压缩检测
    返回: True = 噪音模式（StdDev过低）
    """
    # 计算价格变化率
    price_change = df['close'].pct_change()
    stddev = price_change.rolling(window=lookback).std() * 100

    return stddev


def run_backtest_tactics_b(df, params):
    """
    战术B回测：结构猎人
    - 不接飞刀（等待背离）
    - 波动率过滤（拒绝死水）
    - ATR动态止损
    """

    # 计算指标
    print("计算指标...")
    df['coo'] = calc_coo(df)
    df['stc'] = calc_stc(df)
    df['atr'] = calc_atr(df, params['atr_length'])
    df['stddev'] = calc_stddev_squeeze(df, params['stddev_lookback'])

    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=params['cci_length']).mean()
    mad = tp.rolling(window=params['cci_length']).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=params['cci_ma_length']).mean()

    # STC斜率
    df['stc_slope'] = df['stc'].diff()

    # 背离检测
    df['divergence'] = detect_divergence(df, df['coo'], params['coo_divergence_lookback'])

    # 交易循环
    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_date = None
    entry_atr_stop = 0.0
    trades = []
    equity = []
    skipped_noise = 0
    skipped_no_divergence = 0

    print("开始回测...")

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        # 计算当前权益
        if position > 0:
            unrealized_pnl = (current['close'] - entry_price) * position * params['multiplier']
            current_equity = balance + unrealized_pnl
        else:
            current_equity = balance
        equity.append(current_equity)

        # 平仓逻辑（只有止损，止盈用CCI超买）
        if position > 0:
            exit_triggered = False

            # ATR动态止损
            if next_day['low'] <= entry_atr_stop:
                stop_price = max(next_day['open'], entry_atr_stop)
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
                    'type': 'atr_stop',
                    'hold_days': (next_day.name - entry_date).days,
                    'exit_reason': f"ATR止损 @{entry_atr_stop:.2f}",
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # CCI超买平仓（CCI > 150）
            if not exit_triggered and current['cci'] > 150:
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
                    'type': 'cci_overbought',
                    'hold_days': (next_day.name - entry_date).days,
                    'exit_reason': f"CCI超买 {current['cci']:.2f}",
                })
                position = 0
                entry_price = 0.0
                exit_triggered = True

            # CCI死叉平仓
            if not exit_triggered:
                cci_prev = df.iloc[i-1]['cci'] if i > 0 else current['cci']
                cci_ma_prev = df.iloc[i-1]['cci_ma'] if i > 0 else current['cci_ma']
                if cci_prev >= cci_ma_prev and current['cci'] < current['cci_ma']:
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
                        'type': 'cci_death_cross',
                        'hold_days': (next_day.name - entry_date).days,
                        'exit_reason': f"CCI死叉 {current['cci']:.2f}",
                    })
                    position = 0
                    entry_price = 0.0
                    exit_triggered = True

        # 开仓逻辑（战术B：结构猎人）
        if position == 0:
            # 1. 波动率过滤：拒绝死水
            if current['stddev'] < params['stddev_threshold']:
                skipped_noise += 1
                continue

            # 2. COO进入超卖区
            coo_oversold = current['coo'] < params['coo_oversold']

            # 3. 背离确认
            divergence_confirmed = current['divergence']

            # 4. STC脱离底部
            stc_awake = current['stc'] > params['stc_exit_floor']

            # 5. STC斜率抬头
            stc_slope_up = current['stc_slope'] > params['stc_slope_threshold']

            # 6. CCI向0轴冲击
            cci_breakout = (current['cci'] > params['cci_zero_cross'] and
                          current['cci'] > current['cci_ma'])

            # 战术B完整条件（放宽：不严格要求背离）
            if (coo_oversold and stc_awake and cci_breakout):

                entry_price = next_day['open'] * (1 + SLIPPAGE_RATE)

                # ATR动态止损
                entry_atr_stop = entry_price - current['atr'] * params['atr_stop_multiplier']

                # 仓位计算（基于风险）
                risk_amount = balance * params['risk_per_trade']
                stop_distance = entry_price - entry_atr_stop
                max_value = balance * 0.9 * LEVERAGE

                # 方法1：基于风险计算
                qty_by_risk = int(risk_amount / (stop_distance * params['multiplier']))
                # 方法2：基于资金计算
                qty_by_capital = int(max_value / (entry_price * params['multiplier']))

                qty = min(max(1, qty_by_risk), qty_by_capital)

                commission = entry_price * qty * params['multiplier'] * COMMISSION_RATE
                balance -= commission

                position = qty
                entry_date = next_day.name

                print(f"开仓: {entry_date.date()} 价格:{entry_price:.2f} "
                      f"ATR止损:{entry_atr_stop:.2f} CCI:{current['cci']:.2f} "
                      f"STC:{current['stc']:.2f} COO:{current['coo']:.2f}")

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
            'skipped_noise': skipped_noise,
            'skipped_no_divergence': skipped_no_divergence,
            'final_balance': balance,
        }
    else:
        return None


def print_results(result, params):
    """打印详细回测结果"""

    print("\n" + "=" * 100)
    print("纯碱SA0 - S.C.C.系统 v1.0 战术B（结构猎人）".center(100))
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
    print(f"  收益回撤比:   {result['total_return'] / abs(result['max_dd']) if result['max_dd'] != 0 else 0:>10.2f}")
    print(f"  胜率:         {result['win_rate']:>10.2f}%")
    print(f"  盈亏比:       {result['win_loss_ratio']:>10.2f}")
    print(f"  盈亏因子:     {result['profit_factor']:>10.2f}")
    print(f"  期望收益:     ${result['expected_value']:>10.2f}/笔")
    print(f"  总交易次数:   {result['total_trades']:>10}笔")
    print(f"  平均持仓天数: {result['avg_hold_days']:>10.1f}天")

    print("\n" + "=" * 100)
    print("过滤统计")
    print("=" * 100)
    print(f"  跳过低波动（噪音）: {result['skipped_noise']:>10}次")
    print(f"  跳过无背离:        {result['skipped_no_divergence']:>10}次")

    print("\n" + "=" * 100)
    print("平仓类型统计")
    print("=" * 100)
    print(f"  {'类型':<20} {'次数':>8} {'占比':>10} {'平均盈亏':>12} {'总盈亏':>14}")
    print("-" * 80)

    type_map = {
        'atr_stop': 'ATR动态止损',
        'cci_overbought': 'CCI超买平仓',
        'cci_death_cross': 'CCI死叉平仓'
    }

    for exit_type, count in result['exit_stats'].items():
        pct = count / result['total_trades'] * 100
        type_trades = result['trades_df'][result['trades_df']['type'] == exit_type]
        avg_pnl = type_trades['pnl'].mean()
        total_pnl = type_trades['pnl'].sum()
        print(f"  {type_map.get(exit_type, exit_type):<20} {count:>8} {pct:>9.1f}% {avg_pnl:>11.0f} {total_pnl:>13.0f}")

    print("\n" + "=" * 100)
    print("参数配置 - 战术B（结构猎人）")
    print("=" * 100)
    print(f" COO超卖阈值: {params['coo_oversold']}")
    print(f" 背离检测周期: {params['coo_divergence_lookback']}")
    print(f" STC脱离底部阈值: > {params['stc_exit_floor']}")
    print(f" STC斜率阈值: > {params['stc_slope_threshold']}")
    print(f" CCI突破阈值: > {params['cci_zero_cross']}")
    print(f" ATR止损倍数: {params['atr_stop_multiplier']}")
    print(f" 每笔交易风险: {params['risk_per_trade']*100}%")
    print(f" StdDev噪音阈值: < {params['stddev_threshold']}")

    print("\n" + "=" * 100)


def main():
    print("=" * 100)
    print("纯碱SA0 - S.C.C.系统 v1.0".center(100))
    print("战术B（结构猎人）：专攻低波动/震荡品种".center(100))
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
    print(f"数据跨度: {(df.index[-1] - df.index[0]).days / 365:.1f}年")

    # 运行回测
    print(f"\n[2/2] 运行回测...")
    result = run_backtest_tactics_b(df, TACTICS_B_PARAMS)

    if result:
        # 输出结果
        print(f"\n[2/2] 输出结果...")
        print_results(result, TACTICS_B_PARAMS)

        # 保存详细记录
        output_file = r'D:\期货数据\铜期货监控\CCI策略系统\纯碱SCC系统详细记录.csv'
        result['trades_df'].to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细交易记录已保存至: {output_file}")
    else:
        print("\n[FAILED] 回测失败或无交易")

    print("\n" + "=" * 100)
    print(f"回测完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
