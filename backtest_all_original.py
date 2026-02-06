# -*- coding: utf-8 -*-
"""
批量回测22个期货品种 - 使用原始铜策略
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ========== 原始铜策略参数 ==========
EMA_FAST = 5
EMA_SLOW = 15
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RSI_FILTER = 45
RATIO_TRIGGER = 1.15
STC_LENGTH = 10
STC_FAST = 23
STC_SLOW = 50
STC_SELL_ZONE = 85
STOP_LOSS_PCT = 0.02
INITIAL_CAPITAL = 100000
MARGIN_RATIO = 0.15  # 保证金比例
CONTRACT_SIZE = 5     # 1手合约吨数（不同品种可能不同，这里简化为5）

def calculate_indicators(df):
    """计算技术指标（原始铜策略）"""
    # EMA
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    # MACD & Ratio
    exp1 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # STC
    stc_macd = df['close'].ewm(span=STC_FAST, adjust=False).mean() - \
               df['close'].ewm(span=STC_SLOW, adjust=False).mean()
    stoch_period = STC_LENGTH
    min_macd = stc_macd.rolling(window=stoch_period).min()
    max_macd = stc_macd.rolling(window=stoch_period).max()
    stoch_k = 100 * (stc_macd - min_macd) / (max_macd - min_macd).replace(0, np.nan)
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=3).mean()
    min_stoch_d = stoch_d.rolling(window=stoch_period).min()
    max_stoch_d = stoch_d.rolling(window=stoch_period).max()
    stc_raw = 100 * (stoch_d - min_stoch_d) / (max_stoch_d - min_stoch_d).replace(0, np.nan)
    stc_raw = stc_raw.fillna(50)
    df['stc'] = stc_raw.rolling(window=3).mean()

    # 预计算前值
    df['ratio_prev'] = df['ratio'].shift(1)
    df['stc_prev'] = df['stc'].shift(1)

    return df

def backtest(csv_file, future_name):
    """回测单个品种（使用原始铜策略逻辑）"""
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = calculate_indicators(df)

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    # 简化：不用期货保证金，直接用资金比例
    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # ==================== 入场条件 ====================
        # 趋势条件
        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < RATIO_TRIGGER)
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > RSI_FILTER

        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong

        buy_signal = sniper_signal or chase_signal

        # 计算仓位比例
        if df['ratio_prev'].iloc[i] > 0:
            if df['ratio_prev'].iloc[i] > 2.0:
                position_size = 2.0
            elif df['ratio_prev'].iloc[i] > 1.5:
                position_size = 1.5
            elif df['ratio_prev'].iloc[i] > 1.0:
                position_size = 1.2
            else:
                position_size = 1.0
        else:
            position_size = 1.0

        stop_loss = current['close'] * (1 - STOP_LOSS_PCT)

        # 买入（使用期货保证金制度）
        if buy_signal and position is None:
            entry_price = current['close']
            contract_value = entry_price * CONTRACT_SIZE
            margin_per_contract = contract_value * MARGIN_RATIO
            available_margin = capital * position_size
            contracts = int(available_margin / margin_per_contract)

            if contracts <= 0:
                continue

            position = {
                'entry_datetime': current['datetime'],
                'entry_price': entry_price,
                'contracts': contracts,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'entry_index': i,
                'signal_type': 'sniper' if sniper_signal else 'chase'
            }

        # ==================== 卖出逻辑 ====================
        elif position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 检查后续K线
            for j in range(position['entry_index'] + 1, len(df)):
                bar = df.iloc[j]

                # 1. 止损
                if bar['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                    exit_triggered = True

                # 2. STC卖出
                elif (df['stc_prev'].iloc[j] > STC_SELL_ZONE and
                      bar['stc'] < df['stc_prev'].iloc[j]):
                    exit_price = bar['close']
                    exit_reason = 'stc'
                    exit_triggered = True

                # 3. 趋势反转
                elif bar['ema_fast'] < bar['ema_slow']:
                    exit_price = bar['close']
                    exit_reason = 'trend'
                    exit_triggered = True

                if exit_triggered:
                    pnl = (exit_price - position['entry_price']) * position['contracts'] * CONTRACT_SIZE
                    capital += pnl

                    trades.append({
                        'entry_datetime': position['entry_datetime'],
                        'exit_datetime': bar['datetime'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'contracts': position['contracts'],
                        'position_size': position['position_size'],
                        'pnl': pnl,
                        'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                        'exit_reason': exit_reason,
                        'holding_bars': j - position['entry_index'],
                        'signal_type': position['signal_type']
                    })

                    position = None
                    break

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])

    total_pnl = trades_df['pnl'].sum()
    return_pct = total_pnl / INITIAL_CAPITAL * 100
    win_rate = winning_trades / total_trades * 100

    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    sniper_trades = trades_df[trades_df['signal_type'] == 'sniper']
    chase_trades = trades_df[trades_df['signal_type'] == 'chase']

    return {
        'name': future_name,
        'total_trades': total_trades,
        'sniper_trades': len(sniper_trades),
        'chase_trades': len(chase_trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'return_pct': return_pct,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_capital': capital,
        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
    }

def main():
    print("=" * 80)
    print("批量回测22个期货品种 - 原始铜策略")
    print(f"策略参数: EMA({EMA_FAST},{EMA_SLOW}) + RSI({RSI_FILTER}) + STC({STC_SELL_ZONE})")
    print(f"入场: sniper(5条件) 或 chase(EMA交叉+RSI)")
    print(f"出场: 止损{STOP_LOSS_PCT*100:.0f}% / STC>{STC_SELL_ZONE} / 趋势反转")
    print("=" * 80)

    data_dir = Path('futures_data_4h')
    csv_files = sorted(data_dir.glob('*.csv'))

    results = []

    for csv_file in csv_files:
        future_name = csv_file.stem.replace('_4hour', '')
        print(f"\n正在回测: {future_name}...")

        try:
            result = backtest(csv_file, future_name)
            if result:
                results.append(result)
                print(f"  [OK] 收益率: {result['return_pct']:+.2f}%, 胜率: {result['win_rate']:.1f}%, 交易: {result['total_trades']}")
        except Exception as e:
            print(f"  [FAIL] {e}")

    if not results:
        print("\n无有效结果")
        return

    # 排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)

    # 输出完整排名
    print("\n" + "=" * 80)
    print("回测结果排名（按收益率排序）")
    print("=" * 80)
    print(f"\n{'排名':<6} {'品种':<10} {'收益率':>10} {'胜率':>8} {'交易':>6} {'狙击':>4} {'追击':>4} {'盈亏比':>8}")
    print("-" * 80)

    for i, row in results_df.iterrows():
        rank = results_df.index.get_loc(i) + 1
        print(f"{rank:<6} {row['name']:<10} {row['return_pct']:>+8.2f}% {row['win_rate']:>7.1f}% {row['total_trades']:>6} {row['sniper_trades']:>4} {row['chase_trades']:>4} {row['profit_factor']:>8.2f}")

    # 统计
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)

    profitable = len(results_df[results_df['return_pct'] > 0])
    unprofitable = len(results_df[results_df['return_pct'] <= 0])

    print(f"\n总品种数:     {len(results_df)}")
    print(f"盈利品种:     {profitable}")
    print(f"亏损品种:     {unprofitable}")
    print(f"\n平均收益率:   {results_df['return_pct'].mean():+.2f}%")
    print(f"最高收益:     {results_df['return_pct'].max():+.2f}% ({results_df.iloc[0]['name']})")
    print(f"最低收益:     {results_df['return_pct'].min():+.2f}% ({results_df.iloc[-1]['name']})")
    print(f"收益率中位数: {results_df['return_pct'].median():+.2f}%")

    # 显示前5名详细信息
    print("\n" + "=" * 80)
    print("前5名详细信息")
    print("=" * 80)

    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n第{i+1}名: {row['name']}")
        print(f"  收益率:       {row['return_pct']:+.2f}%")
        print(f"  交易次数:     {row['total_trades']} (狙击:{row['sniper_trades']}, 追击:{row['chase_trades']})")
        print(f"  胜率:         {row['win_rate']:.1f}%")
        print(f"  盈利次数:     {row['winning_trades']}")
        print(f"  亏损次数:     {row['losing_trades']}")
        print(f"  平均盈利:     {row['avg_win']:+,.2f}")
        print(f"  平均亏损:     {row['avg_loss']:+,.2f}")
        print(f"  盈亏比:       {row['profit_factor']:.2f}")
        print(f"  总盈亏:       {row['total_pnl']:+,.2f}")
        print(f"  最终资金:     {row['final_capital']:,.2f}")

    # 用沪铜数据验证
    print("\n" + "=" * 80)
    print("沪铜详细验证")
    print("=" * 80)
    copper_result = results_df[results_df['name'] == '沪铜']
    if not copper_result.empty:
        row = copper_result.iloc[0]
        print(f"\n沪铜收益率: {row['return_pct']:+.2f}%")
        print(f"与原回测对比需要用相同数据文件验证")

if __name__ == '__main__':
    main()
