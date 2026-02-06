# -*- coding: utf-8 -*-
"""
===================================
沪铜策略回测系统 (基于真正的4小时K线)
===================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 策略参数（与实盘一致）
# ==========================================
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
INITIAL_CAPITAL = 100000  # 初始资金 (元)
MARGIN_RATIO = 0.15  # 保证金比例 15%

# ==========================================
# 数据加载
# ==========================================

def load_data(csv_path='沪铜4小时K线_1年.csv'):
    """加载4小时K线数据"""
    logger.info(f"加载数据: {csv_path}")

    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    logger.info(f"数据量: {len(df)} 条")
    logger.info(f"时间范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    return df

# ==========================================
# 技术指标计算
# ==========================================

def calculate_indicators(df):
    """计算技术指标"""
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

# ==========================================
# 信号检测
# ==========================================

def detect_signals(df):
    """检测交易信号"""
    signals = []

    for i in range(200, len(df)):  # 从第200根K线开始，确保指标足够
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 趋势条件
        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < RATIO_TRIGGER)
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > RSI_FILTER

        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong

        # 买入信号
        buy_signal = sniper_signal or chase_signal
        buy_reason = 'sniper' if sniper_signal else ('chase' if chase_signal else None)

        # 计算仓位（用上一根Ratio，实盘逻辑）
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

        signals.append({
            'index': i,
            'datetime': current['datetime'],
            'price': current['close'],
            'buy_signal': buy_signal,
            'signal_type': buy_reason,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'ratio': current['ratio'],
            'rsi': current['rsi'],
            'stc': current['stc']
        })

    return pd.DataFrame(signals)

# ==========================================
# 回测引擎
# ==========================================

def run_backtest(df, signals_df):
    """运行回测"""
    logger.info("=" * 80)
    logger.info("开始回测")
    logger.info("=" * 80)

    trades = []
    position = None
    capital = INITIAL_CAPITAL

    for idx, signal in signals_df.iterrows():
        # 买入逻辑
        if signal['buy_signal'] and position is None:
            entry_price = signal['price']
            position_size_pct = signal['position_size']  # 仓位比例 (1.0x, 1.5x, 2.0x)

            # 计算合约数量 (基于保证金制度)
            # 沪铜: 1手=5吨，保证金比例约15%
            contract_value = entry_price * 5  # 1手合约价值
            margin_per_contract = contract_value * MARGIN_RATIO  # 1手所需保证金

            # 可用保证金 = 资金 × 仓位比例
            available_margin = capital * position_size_pct

            # 可交易手数
            contracts = int(available_margin / margin_per_contract)

            if contracts <= 0:
                logger.warning(f"[跳过] 资金不足，无法开仓 | 价格: {entry_price:.0f} | "
                             f"可用资金: {available_margin:.0f} | 需要: {margin_per_contract:.0f}")
                continue

            position = {
                'entry_datetime': signal['datetime'],
                'entry_price': entry_price,
                'contracts': contracts,
                'position_size': position_size_pct,
                'stop_loss': signal['stop_loss'],
                'entry_index': int(signal['index'])
            }

            logger.info(f"[买入] {signal['datetime']} | 价格: {entry_price:.0f} | "
                       f"仓位: {position_size_pct:.1f}x | 合约: {contracts}手 | "
                       f"保证金: {margin_per_contract*contracts:.0f}")

        # 卖出逻辑（检查后续K线）
        elif position is not None:
            # 从入场位置开始查找卖出点
            start_idx = position['entry_index'] + 1

            for j in range(start_idx, len(df)):
                current_bar = df.iloc[j]

                # 检查止损
                if current_bar['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                    exit_datetime = current_bar['datetime']

                    pnl = (exit_price - position['entry_price']) * position['contracts'] * 5
                    capital += pnl

                    trade = {
                        'entry_datetime': position['entry_datetime'],
                        'exit_datetime': exit_datetime,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'contracts': position['contracts'],
                        'position_size': position['position_size'],
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                        'holding_bars': j - position['entry_index']
                    }
                    trades.append(trade)

                    logger.info(f"[止损卖出] {exit_datetime} | 价格: {exit_price:.0f} | "
                               f"盈亏: {pnl:.0f} ({pnl/capital*100:.2f}%)")

                    position = None
                    break

                # 检查STC卖出
                elif (df['stc_prev'].iloc[j] > STC_SELL_ZONE and
                      current_bar['stc'] < df['stc_prev'].iloc[j]):
                    exit_price = current_bar['close']
                    exit_reason = 'stc'
                    exit_datetime = current_bar['datetime']

                    pnl = (exit_price - position['entry_price']) * position['contracts'] * 5
                    capital += pnl

                    trade = {
                        'entry_datetime': position['entry_datetime'],
                        'exit_datetime': exit_datetime,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'contracts': position['contracts'],
                        'position_size': position['position_size'],
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                        'holding_bars': j - position['entry_index']
                    }
                    trades.append(trade)

                    logger.info(f"[STC卖出] {exit_datetime} | 价格: {exit_price:.0f} | "
                               f"盈亏: {pnl:.0f} ({pnl/capital*100:.2f}%)")

                    position = None
                    break

                # 检查趋势反转
                elif current_bar['ema_fast'] < current_bar['ema_slow']:
                    exit_price = current_bar['close']
                    exit_reason = 'trend'
                    exit_datetime = current_bar['datetime']

                    pnl = (exit_price - position['entry_price']) * position['contracts'] * 5
                    capital += pnl

                    trade = {
                        'entry_datetime': position['entry_datetime'],
                        'exit_datetime': exit_datetime,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'contracts': position['contracts'],
                        'position_size': position['position_size'],
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                        'holding_bars': j - position['entry_index']
                    }
                    trades.append(trade)

                    logger.info(f"[趋势卖出] {exit_datetime} | 价格: {exit_price:.0f} | "
                               f"盈亏: {pnl:.0f} ({pnl/capital*100:.2f}%)")

                    position = None
                    break

    return pd.DataFrame(trades), capital

# ==========================================
# 统计分析
# ==========================================

def analyze_results(trades_df, final_capital):
    """分析回测结果"""
    logger.info("\n" + "=" * 80)
    logger.info("回测结果统计")
    logger.info("=" * 80)

    if trades_df.empty:
        logger.info("无交易记录")
        return

    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])

    total_pnl = trades_df['pnl'].sum()
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0

    max_win = trades_df['pnl'].max()
    max_loss = trades_df['pnl'].min()

    avg_holding_bars = trades_df['holding_bars'].mean()

    initial_capital = INITIAL_CAPITAL
    total_return = (final_capital - initial_capital) / initial_capital * 100

    print(f"\n【资金】")
    print(f"  初始资金: {initial_capital:,.0f} 元")
    print(f"  最终资金: {final_capital:,.0f} 元")
    print(f"  总收益: {total_pnl:,.0f} 元 ({total_return:.2f}%)")

    print(f"\n【交易统计】")
    print(f"  总交易次数: {total_trades}")
    print(f"  盈利次数: {winning_trades} ({win_rate:.1f}%)")
    print(f"  亏损次数: {losing_trades}")

    print(f"\n【盈亏分析】")
    print(f"  平均盈利: {avg_win:,.0f} 元")
    print(f"  平均亏损: {avg_loss:,.0f} 元")
    print(f"  最大盈利: {max_win:,.0f} 元")
    print(f"  最大亏损: {max_loss:,.0f} 元")
    print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  盈亏比: N/A")

    print(f"\n【持仓分析】")
    print(f"  平均持仓K线数: {avg_holding_bars:.1f} (约{avg_holding_bars*4:.1f}小时)")

    print(f"\n【退出原因统计】")
    exit_reasons = trades_df['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/total_trades*100:.1f}%)")

    # 详细交易记录
    print(f"\n【详细交易记录】")
    print(trades_df.to_string(index=False))

# ==========================================
# 主程序
# ==========================================

def main():
    """主程序"""
    # 加载数据
    df = load_data()

    # 计算指标
    logger.info("计算技术指标...")
    df = calculate_indicators(df)

    # 检测信号
    logger.info("检测交易信号...")
    signals_df = detect_signals(df)

    buy_signals = signals_df[signals_df['buy_signal']]
    logger.info(f"发现 {len(buy_signals)} 个买入信号")

    # 运行回测
    trades_df, final_capital = run_backtest(df, signals_df)

    # 分析结果
    analyze_results(trades_df, final_capital)

    # 保存结果
    output_path = 'backtest_results.csv'
    if not trades_df.empty:
        trades_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"\n回测结果已保存: {output_path}")

if __name__ == "__main__":
    main()
