# -*- coding: utf-8 -*-
"""
回测程序 - LME铜

使用贝叶斯优化的参数进行回测
修复了所有已知的bug
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

# 加载数据
with open('../原始数据/data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

df = DATA_POOL['cu'].copy()

# 加载优化参数
try:
    with open('../优化结果/cu_optimized_params.json', 'r', encoding='utf-8') as f:
        opt_result = json.load(f)
    params = opt_result['params']
except:
    # 使用默认参数
    params = {
        'EMA_FAST': 10,
        'EMA_SLOW': 12,
        'MACD_FAST': 11,
        'MACD_SLOW': 23,
        'MACD_SIGNAL': 11,
        'RSI_PERIOD': 17,
        'RSI_FILTER': 47,
        'RATIO_TRIGGER': 1.229,
        'STC_LENGTH': 9,
        'STC_FAST': 24,
        'STC_SLOW': 44,
        'STC_SELL_ZONE': 72
    }

print("="*80)
print("LME铜回测系统")
print("="*80)
print(f"\n数据信息:")
print(f"  数据量: {len(df)}条")
print(f"  时间范围: {df.index[0].date()} ~ {df.index[-1].date()}")

print(f"\n策略参数:")
for k, v in params.items():
    print(f"  {k}: {v}")

# 固定参数
INITIAL_CAPITAL = 100000
MARGIN_RATIO = 0.11
CONTRACT_SIZE = 5
MAX_LEVERAGE = 3.0
MAX_DRAWDOWN_PCT = 50.0
MIN_CAPITAL_PCT = 0.3
STOP_LOSS_PCT = 0.03  # 调整为3%以提高胜率
COMMISSION_RATE = 0.0003  # 手续费 0.03%/边 (与 CCI 全品种严格回测.py 一致)
SLIPPAGE_RATE = 0.0002    # 滑点 0.02%/边

print(f"\n交易规则:")
print(f"  初始资金: {INITIAL_CAPITAL:,}")
print(f"  保证金率: {MARGIN_RATIO*100}%")
print(f"  合约单位: {CONTRACT_SIZE}吨/手")
print(f"  最大杠杆: {MAX_LEVERAGE}倍")
print(f"  最大回撤: {MAX_DRAWDOWN_PCT}%")
print(f"  固定止损: {STOP_LOSS_PCT*100}%")


def calculate_indicators(df, params):
    """计算技术指标"""
    df = df.copy()

    df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

    exp1 = df['close'].ewm(span=params['MACD_FAST'], adjust=False).mean()
    exp2 = df['close'].ewm(span=params['MACD_SLOW'], adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=params['MACD_SIGNAL'], adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'].abs() > 1e-10, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/params['RSI_PERIOD'], adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/params['RSI_PERIOD'], adjust=False).mean()
    loss_safe = loss.where(loss.abs() > 1e-10, 1e-10)
    rs = gain / loss_safe
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

    stc_macd = df['close'].ewm(span=params['STC_FAST'], adjust=False).mean() - \
               df['close'].ewm(span=params['STC_SLOW'], adjust=False).mean()
    stoch_period = params['STC_LENGTH']
    min_macd = stc_macd.rolling(window=stoch_period).min()
    max_macd = stc_macd.rolling(window=stoch_period).max()
    macd_range = max_macd - min_macd
    stoch_k = 100 * (stc_macd - min_macd) / macd_range.where(macd_range.abs() > 1e-10, np.nan)
    stoch_k = stoch_k.fillna(50)
    stoch_d = stoch_k.rolling(window=3).mean()
    min_stoch_d = stoch_d.rolling(window=stoch_period).min()
    max_stoch_d = stoch_d.rolling(window=stoch_period).max()
    stoch_range = max_stoch_d - min_stoch_d
    stc_raw = 100 * (stoch_d - min_stoch_d) / stoch_range.where(stoch_range.abs() > 1e-10, np.nan)
    stc_raw = stc_raw.fillna(50)
    df['stc'] = stc_raw.rolling(window=3).mean()

    return df


def backtest(df, params):
    """回测函数（修复所有bug）"""
    df = calculate_indicators(df, params)

    capital = INITIAL_CAPITAL
    position = None
    trades = []
    capital_history = [{'date': df.index[0], 'capital': capital}]

    peak_capital = INITIAL_CAPITAL

    def apply_slippage(price, direction):
        """方向: 'buy' 多付滑点, 'sell' 少收滑点"""
        return price * (1 + SLIPPAGE_RATE) if direction == 'buy' else price * (1 - SLIPPAGE_RATE)

    def net_pnl(entry_price, exit_price, contracts, direction='buy'):
        """含手续费+滑点的净盈亏。
        - 入场/退出价均为滑点后成交价 (apply_slippage 已应用)
        - 手续费: 双边各 0.03% × 各自名义价值
        """
        gross = (exit_price - entry_price) * contracts * CONTRACT_SIZE
        entry_notional = entry_price * contracts * CONTRACT_SIZE
        exit_notional = exit_price * contracts * CONTRACT_SIZE
        fee = (entry_notional + exit_notional) * COMMISSION_RATE
        return gross - fee

    # 注意: 信号在 bar i 收盘计算, 次bar (next_day) 开盘成交 —— 消除 look-ahead
    for i in range(200, len(df) - 1):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        next_day = df.iloc[i+1]

        # 风险管理
        current_drawdown = (capital - peak_capital) / peak_capital * 100

        if current_drawdown < -MAX_DRAWDOWN_PCT:
            if position is not None:
                exit_px = apply_slippage(current['close'], 'sell')
                pnl = net_pnl(position['entry_price'], exit_px, position['contracts'])
                capital += pnl
                trades.append({
                    'entry_date': position['entry_datetime'],
                    'exit_date': current.name,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_px,
                    'contracts': position['contracts'],
                    'leverage': position['leverage'],
                    'pnl': pnl,
                    'pnl_pct': (exit_px - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': 'force_close_max_dd'
                })
                position = None
            break

        if capital < INITIAL_CAPITAL * MIN_CAPITAL_PCT:
            if position is not None:
                exit_px = apply_slippage(current['close'], 'sell')
                pnl = net_pnl(position['entry_price'], exit_px, position['contracts'])
                capital += pnl
                trades.append({
                    'entry_date': position['entry_datetime'],
                    'exit_date': current.name,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_px,
                    'contracts': position['contracts'],
                    'leverage': position['leverage'],
                    'pnl': pnl,
                    'pnl_pct': (exit_px - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': 'force_close_low_capital'
                })
                position = None
            break

        # 持仓中：检查次bar (next_day) 的退出条件 —— 消除 look-ahead
        # 信号在 bar i 收盘确认, 退出在 next_day 执行
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 止损: next_day 最低价触发, 以 max(next_open, stop_loss) 成交(跳空不利)
            if next_day['low'] <= position['stop_loss']:
                exit_price = max(next_day['open'], position['stop_loss'])
                exit_price = apply_slippage(exit_price, 'sell')
                exit_reason = 'stop_loss'
                exit_triggered = True
            # STC 死叉 (bar i 收盘信号) → next_bar 收盘出场
            elif (prev['stc'] > params['STC_SELL_ZONE'] and
                  current['stc'] < prev['stc']):
                exit_price = apply_slippage(next_day['close'], 'sell')
                exit_reason = 'stc'
                exit_triggered = True
            # 趋势反转 (bar i 收盘信号) → next_bar 收盘出场
            elif current['ema_fast'] < current['ema_slow']:
                exit_price = apply_slippage(next_day['close'], 'sell')
                exit_reason = 'trend'
                exit_triggered = True

            if exit_triggered:
                pnl = net_pnl(position['entry_price'], exit_price, position['contracts'])
                capital += pnl

                if capital > peak_capital:
                    peak_capital = capital

                trades.append({
                    'entry_date': position['entry_datetime'],
                    'exit_date': next_day.name,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'leverage': position['leverage'],
                    'pnl': pnl,
                    'pnl_pct': (exit_price - position['entry_price']) / position['entry_price'] * 100,
                    'exit_reason': exit_reason
                })
                position = None

        capital_history.append({'date': next_day.name, 'capital': capital})

        # 无持仓：检查买入
        trend_up = current['ema_fast'] > current['ema_slow']
        ratio_safe = (0 < current['ratio'] < params['RATIO_TRIGGER'])
        ratio_shrinking = current['ratio'] < prev['ratio']
        turning_up = current['macd_dif'] > prev['macd_dif']
        is_strong = current['rsi'] > params['RSI_FILTER']
        ema_cross = (prev['ema_fast'] <= prev['ema_slow']) and (current['ema_fast'] > current['ema_slow'])

        sniper_signal = trend_up and ratio_safe and ratio_shrinking and turning_up and is_strong
        chase_signal = ema_cross and is_strong
        buy_signal = sniper_signal or chase_signal

        # 仓位计算
        if current['ratio'] > 0:
            if current['ratio'] > 2.0:
                base_position_size = 2.0
            elif current['ratio'] > 1.5:
                base_position_size = 1.5
            elif current['ratio'] > 1.0:
                base_position_size = 1.2
            else:
                base_position_size = 1.0
        else:
            base_position_size = 1.0

        # 资金状况判断（修复顺序）
        capital_ratio = capital / INITIAL_CAPITAL
        if capital_ratio < 0.3:
            base_position_size = 0
        elif capital_ratio < 0.5:
            base_position_size = base_position_size * 0.5

        # 限制最大杠杆
        position_size = min(base_position_size, MAX_LEVERAGE)

        # 买入: 信号在 bar i 收盘确认, 次bar (next_day) 开盘成交 —— 消除 look-ahead
        # 入场价 = next_day open + slippage; stop 基于真实入场价
        stop_loss = None  # 延迟到入场价确定后计算

        # 买入
        if buy_signal and position is None and position_size > 0:
            entry_price = apply_slippage(next_day['open'], 'buy')
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            contract_value = entry_price * CONTRACT_SIZE

            # 以损定仓法（真实期货交易逻辑）
            # 1. 确定每笔交易的风险比例
            # position_size作为风险放大系数：1.0x-2.0x
            # 基础风险2%，根据ratio调整到2%-4%
            base_risk_pct = 0.02  # 基础风险2%
            risk_pct = base_risk_pct * position_size  # 实际风险2%-4%

            # 2. 计算每笔最大可承受亏损
            max_loss_per_trade = capital * risk_pct

            # 3. 计算每手止损时的亏损
            loss_per_contract = entry_price * STOP_LOSS_PCT * CONTRACT_SIZE

            # 4. 计算可开手数
            contracts = int(max_loss_per_trade / loss_per_contract)

            if contracts <= 0:
                continue

            # 5. 计算实际杠杆
            total_notional = contracts * contract_value
            actual_leverage = total_notional / capital

            # 6. 验证杠杆不超标
            if actual_leverage > MAX_LEVERAGE:
                contracts = int((capital * MAX_LEVERAGE) / contract_value)
                if contracts <= 0:
                    continue
                total_notional = contracts * contract_value
                actual_leverage = total_notional / capital

            # 7. 验证保证金充足
            required_margin = total_notional * MARGIN_RATIO
            if required_margin > capital:
                contracts = int(capital / (contract_value * MARGIN_RATIO))
                if contracts <= 0:
                    continue
                total_notional = contracts * contract_value
                actual_leverage = total_notional / capital

            position = {
                'entry_datetime': next_day.name,
                'entry_price': entry_price,
                'contracts': contracts,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'entry_index': i + 1,
                'leverage': actual_leverage
            }

            if capital > peak_capital:
                peak_capital = capital

    return trades, capital_history


def main():
    """主函数"""
    print(f"\n开始回测...")
    print("="*80)

    # 运行回测
    trades, capital_history = backtest(df, params)

    if not trades:
        print("\n未产生交易")
        return

    trades_df = pd.DataFrame(trades)
    capital_df = pd.DataFrame(capital_history)

    # 统计分析
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    total_pnl = trades_df['pnl'].sum()
    return_pct = total_pnl / INITIAL_CAPITAL * 100

    capital_series = capital_df.set_index('date')['capital']
    peak = capital_series.cummax()
    drawdown = (capital_series - peak) / peak * 100
    max_drawdown = drawdown.min()

    print(f"\n{'='*80}")
    print("回测结果")
    print(f"{'='*80}")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}")
    print(f"最终资金: {capital_series.iloc[-1]:,.0f}")
    print(f"总盈亏: {total_pnl:,.0f}")
    print(f"总收益率: {return_pct:+.2f}%")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"\n交易统计:")
    print(f"  总交易: {len(trades_df)}次")
    print(f"  盈利: {len(winning_trades)}次")
    print(f"  亏损: {len(losing_trades)}次")
    print(f"  胜率: {len(winning_trades)/len(trades_df)*100:.1f}%")
    print(f"  平均盈利: {winning_trades['pnl'].mean():,.0f}" if len(winning_trades) > 0 else "  平均盈利: N/A")
    print(f"  平均亏损: {losing_trades['pnl'].mean():,.0f}" if len(losing_trades) > 0 else "  平均亏损: N/A")
    print(f"  最大盈利: {trades_df['pnl'].max():,.0f}")
    print(f"  最大亏损: {trades_df['pnl'].min():,.0f}")

    if len(winning_trades) > 0 and len(losing_trades) > 0:
        print(f"  盈亏比: {-winning_trades['pnl'].mean()/losing_trades['pnl'].mean():.2f}")

    # 杠杆分析
    print(f"\n杠杆分析:")
    print(f"  平均杠杆: {trades_df['leverage'].mean():.2f}倍")
    print(f"  最大杠杆: {trades_df['leverage'].max():.2f}倍")

    # 显示交易明细
    print(f"\n{'='*80}")
    print("交易明细")
    print(f"{'='*80}")
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 15)
    print(trades_df.to_string(index=False))

    # 保存结果
    trades_df.to_csv('../优化结果/交易明细.csv', index=False, encoding='utf-8-sig')
    capital_df.to_csv('../优化结果/资金曲线.csv', index=False, encoding='utf-8-sig')

    # 保存参数
    with open('../优化结果/使用参数.json', 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: 回测系统/优化结果/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
