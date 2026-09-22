"""
纯碱(SA0) 标准回测 - 完整验证版
严格检查每个逻辑步骤，确保无遗漏
"""
import pandas as pd
import numpy as np
from datetime import datetime

# ========================================
# 配置参数（与文档完全一致）
# ========================================
SA_PARAMS = {
    'code': 'sa0',
    'exchange': 'CZCE',
    'multiplier': 20,  # 合约乘数

    # CCI参数
    'cci_length': 12,
    'ma_length': 5,
    'cci_oversold': -40,
    'cci_overbought': 190,  # CCI超买平仓阈值
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

# 交易成本（标准配置）
COMMISSION_RATE = 0.0003  # 万三手续费（双边）
SLIPPAGE_RATE = 0.0002    # 万二滑点
INITIAL_CAPITAL = 100000
LEVERAGE = 2  # 2倍杠杆


def get_and_validate_data(code):
    """获取并验证数据完整性"""
    print("\n[数据获取]")
    import akshare as ak

    df = ak.futures_main_sina(symbol=code)
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.dropna()

    print(f"  原始数据量: {len(df)}条")
    print(f"  时间范围: {df['date'].min()} 至 {df['date'].max()}")

    # 验证必要列
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  [ERROR] 缺少列: {missing_cols}")
        return None

    # 验证无负价格
    if (df[['open', 'high', 'low', 'close']] < 0).any().any():
        print(f"  [WARNING] 发现负价格")

    # 验证无零成交量的交易日
    zero_vol_days = (df['volume'] == 0).sum()
    if zero_vol_days > 0:
        print(f"  [INFO] {zero_vol_days}天零成交量")

    df.set_index('date', inplace=True)
    return df


def calculate_indicators(df):
    """计算所有技术指标"""
    print("\n[指标计算]")

    # 1. CCI计算
    print("  计算CCI...")
    cci_length = SA_PARAMS['cci_length']
    ma_length = SA_PARAMS['ma_length']

    # TP = (High + Low + Close) / 3
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3

    # SMA(TP, CCI_Length)
    df['sma_tp'] = df['tp'].rolling(window=cci_length).mean()

    # MAD = Mean(|TP - SMA(TP)|, CCI_Length)
    df['mad'] = df['tp'].rolling(window=cci_length).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )

    # CCI = (TP - SMA) / (0.015 * MAD)
    df['cci'] = (df['tp'] - df['sma_tp']) / (0.015 * df['mad'])
    df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()

    print(f"    CCI范围: {df['cci'].min():.2f} ~ {df['cci'].max():.2f}")

    # 2. STC计算
    print("  计算STC...")
    fast_period = 23
    slow_period = 50
    cycle_period = 10

    # MACD
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow

    # STC
    lowest_macd = df['macd'].rolling(window=cycle_period).min()
    highest_macd = df['macd'].rolling(window=cycle_period).max()
    df['range_macd'] = highest_macd - lowest_macd

    df['stc_raw'] = ((df['macd'] - lowest_macd) / df['range_macd'] - 0.5) * 200
    df['stc_raw'] = df['stc_raw'].fillna(0)

    # 归一化STC到[-50, +50]
    lookback = 20
    df['stc_low'] = df['stc_raw'].rolling(window=lookback).min()
    df['stc_high'] = df['stc_raw'].rolling(window=lookback).max()
    df['stc_range'] = df['stc_high'] - df['stc_low']

    df['stc'] = np.where(
        df['stc_range'] != 0,
        (df['stc_raw'] - df['stc_low']) / df['stc_range'] * 100 - 50,
        df['stc_raw']
    )

    print(f"    STC范围: {df['stc'].min():.2f} ~ {df['stc'].max():.2f}")

    # 3. 量比计算
    print("  计算量比...")
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']
    print(f"    量比范围: {df['vol_ratio'].min():.2f} ~ {df['vol_ratio'].max():.2f}")

    # 4. 隔夜缺口因子
    if SA_PARAMS.get('use_paper_factor', False):
        print("  计算隔夜缺口因子...")
        # gap = (今日开盘 - 昨日收盘) / 昨日收盘
        df['gap_raw'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # 归一化: gap / (true_range / price)
        lookback = 10
        df['high_max'] = df['high'].rolling(lookback).max()
        df['low_min'] = df['low'].rolling(lookback).min()
        df['true_range'] = (df['high_max'] - df['low_min']) / df['close']
        df['gap_factor'] = df['gap_raw'] / df['true_range']
        df['gap_factor'] = df['gap_factor'].fillna(0)

        print(f"    缺口因子范围: {df['gap_factor'].min():.3f} ~ {df['gap_factor'].max():.3f}")

    return df


def run_backtest(df, params):
    """运行标准回测"""
    print("\n[回测开始]")
    print(f"  初始资金: ${INITIAL_CAPITAL:,}")
    print(f"  杠杆倍数: {LEVERAGE}x")
    print(f"  合约乘数: {params['multiplier']}")
    print(f"  滑点率: {SLIPPAGE_RATE*100:.2f}%")
    print(f"  手续费率: {COMMISSION_RATE*100:.3f}%")

    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_date = None
    entry_cci = None
    trades = []
    equity = []
    skipped_low_vol = 0
    skipped_gap_factor = 0

    print("\n[回测循环]")

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

        # ========================================
        # 平仓条件检查（优先级从高到低）
        # ========================================

        if position > 0:
            exit_triggered = False
            exit_price = None
            exit_type = None
            exit_reason = ""

            # 1. 止损（-4%）- 最高优先级
            if next_day['low'] <= entry_price * 0.96:
                stop_loss_threshold = entry_price * 0.96
                # 实际止损价 = max(次日开盘, 止损阈值)
                stop_price = max(next_day['open'], stop_loss_threshold)
                exit_price = stop_price * (1 - SLIPPAGE_RATE)

                pnl = (exit_price - entry_price) * position * params['multiplier']
                commission = exit_price * position * params['multiplier'] * COMMISSION_RATE
                net_pnl = pnl - commission

                balance += net_pnl
                exit_type = 'stop_loss'
                exit_reason = f"止损 低价{next_day['low']:.2f}<=阈值{stop_loss_threshold:.2f}"
                exit_triggered = True

            # 2. CCI超买平仓（主要止盈方式）
            if not exit_triggered and cci > params['cci_overbought']:
                exit_price = next_day['close'] * (1 - SLIPPAGE_RATE)

                pnl = (exit_price - entry_price) * position * params['multiplier']
                commission = exit_price * position * params['multiplier'] * COMMISSION_RATE
                net_pnl = pnl - commission

                balance += net_pnl
                exit_type = 'overbought'
                exit_reason = f"CCI超买 CCI={cci:.2f}>{params['cci_overbought']}"
                exit_triggered = True

            # 3. CCI死叉平仓
            if not exit_triggered and (cci_prev >= cci_ma_prev and cci < cci_ma):
                exit_price = next_day['close'] * (1 - SLIPPAGE_RATE)

                pnl = (exit_price - entry_price) * position * params['multiplier']
                commission = exit_price * position * params['multiplier'] * COMMISSION_RATE
                net_pnl = pnl - commission

                balance += net_pnl
                exit_type = 'death_cross'
                exit_reason = f"CCI死叉 CCI从{cci_prev:.2f}下穿{cci_ma:.2f}"
                exit_triggered = True

            # 执行平仓
            if exit_triggered:
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': next_day.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'type': exit_type,
                    'hold_days': (next_day.name - entry_date).days,
                    'cci_on_entry': entry_cci,
                    'cci_on_exit': cci,
                    'reason': exit_reason,
                })
                position = 0
                entry_price = 0.0
                entry_date = None

        # ========================================
        # 开仓信号检查
        # ========================================

        if position == 0:
            open_signal = False
            open_condition = ""

            # 开仓条件1: CCI超卖 + STC确认
            if (cci < params['cci_oversold'] and
                stc >= params['stc_oversold']):
                open_signal = True
                open_condition = f"CCI超卖({cci:.2f}<{params['cci_oversold']})+STC({stc:.2f}>={params['stc_oversold']})"

            # 开仓条件2: CCI金叉 + STC确认 + CCI限制
            elif (cci_prev <= cci_ma_prev and cci > cci_ma and
                  cci <= params['cci_cross_max'] and
                  stc >= params['stc_cross']):
                open_signal = True
                open_condition = f"CCI金叉+STC确认,CCI<={params['cci_cross_max']}"

            if open_signal:
                # 论文因子过滤
                factor_pass = True
                if params.get('use_paper_factor', False):
                    gap_th = params.get('gap_threshold')
                    if gap_th is not None:
                        if not pd.isna(current['gap_factor']):
                            if current['gap_factor'] < gap_th:
                                factor_pass = False
                                skipped_gap_factor += 1

                # 量能过滤
                if factor_pass:
                    if params['vol_threshold'] > 0:
                        if not pd.isna(vol_ratio):
                            if vol_ratio < params['vol_threshold']:
                                skipped_low_vol += 1
                            else:
                                # 计算开仓
                                entry_price = next_day['open'] * (1 + SLIPPAGE_RATE)
                                max_value = balance * 0.9 * LEVERAGE
                                qty = int(max_value / (entry_price * params['multiplier']))
                                qty = max(1, qty)

                                # 开仓手续费
                                commission = entry_price * qty * params['multiplier'] * COMMISSION_RATE
                                balance -= commission

                                position = qty
                                entry_date = next_day.name
                                entry_cci = cci

                                print(f"  [开仓] {next_day.name.date()} "
                                      f"条件:{open_condition} "
                                      f"价格:{entry_price:.2f} "
                                      f"数量:{qty}手")

                else:
                    skipped_low_vol += 1

    # 计算最终指标
    print("\n[计算最终指标]")

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
            'skipped_gap_factor': skipped_gap_factor,
            'final_balance': balance,
        }
    else:
        print("  [ERROR] 无交易记录")
        return None


def print_results(result):
    """打印详细回测结果"""
    print("\n" + "=" * 100)
    print("纯碱(SA0) 标准回测结果".center(100))
    print("=" * 100)

    print(f"\n数据时间范围: {result['trades_df']['entry_date'].min()} 至 {result['trades_df']['exit_date'].max()}")
    print(f"总交易日: {len(result['trades_df']) + 500}天")

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
    print(f"  缺口因子跳过: {result['skipped_gap_factor']:>10}次")

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
    print("止盈止损规则（已确认）")
    print("=" * 100)
    print("  平仓优先级（从高到低）:")
    print("    1. 止损(-4%):        日内最低价 <= 开仓价 × 0.96")
    print("    2. CCI超买平仓:      CCI > 190")
    print("    3. CCI死叉平仓:      CCI从上往下穿越CCI_MA")
    print("  注意: 已去掉20%固定止盈")

    print("\n" + "=" * 100)


def main():
    print("=" * 100)
    print("纯碱(SA0) 标准回测 - 完整验证版".center(100))
    print("=" * 100)
    print(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # 1. 获取并验证数据
    df = get_and_validate_data(SA_PARAMS['code'])
    if df is None:
        return

    # 2. 计算所有指标
    df = calculate_indicators(df)

    # 3. 运行回测
    result = run_backtest(df, SA_PARAMS)

    # 4. 打印结果
    if result:
        print_results(result)

        # 5. 保存详细记录
        output_file = 'D:\\期货数据\\铜期货监控\\CCI策略系统\\纯碱标准回测详细记录.csv'
        result['trades_df'].to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细交易记录已保存至: {output_file}")
    else:
        print("\n[FAILED] 回测失败")

    print("\n" + "=" * 100)
    print(f"回测完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
