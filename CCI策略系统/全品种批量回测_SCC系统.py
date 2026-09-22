# -*- coding: utf-8 -*-
"""
全品种批量回测系统 - S.C.C.策略
自动遍历所有期货品种，生成性能报告
"""
import pandas as pd
import numpy as np
from datetime import datetime
import akshare as ak

# 复用策略参数库
from 全自动_SCC系统 import STRATEGY_REPO, detect_market_regime, calc_stc, calc_atr

MULTIPLIER = 20  # 默认乘数，不同品种会不同
COMMISSION_RATE = 0.0001
SLIPPAGE_RATE = 0.0002
INITIAL_CAPITAL = 100000
MARGIN_RATE = 0.10

# 主力合约代码映射
SYMBOL_MAP = {
    # 有色金属
    'cu0': '铜', 'al0': '铝', 'zn0': '锌', 'pb0': '铅', 'ni0': '镍', 'sn0': '锡',
    # 贵金属
    'au0': '黄金', 'ag0': '白银',
    # 黑色
    'rb0': '螺纹钢', 'hc0': '热卷', 'i0': '铁矿石', 'j0': '焦炭', 'jm0': '焦煤',
    # 化工
    'sa0': '纯碱', 'fg0': '玻璃', 'ma0': '甲醇', 'ta0': 'PTA', 'l0': '塑料', 'pp0': 'PP', 'v0': 'PVC',
    # 能源
    'sc0': '原油', 'fu0': '燃料油', 'lu0': '沥青',
    # 农产品
    'c0': '玉米', 'cs0': '玉米淀粉', 'm0': '豆粕', 'y0': '豆油', 'p0': '棕榈油', 'a0': '豆一',
}


def get_data(symbol):
    """获取数据"""
    try:
        df = ak.futures_main_sina(symbol=symbol)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'settle']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.dropna()
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"    [ERROR] 获取数据失败: {e}")
        return None


def run_backtest(df, symbol):
    """回测单个品种"""
    # 计算指标
    df['stc'] = calc_stc(df)
    df['stc_prev'] = df['stc'].shift(1)
    df['stc_slope'] = df['stc'] - df['stc_prev']

    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=14).mean()
    mad = tp.rolling(window=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_prev'] = df['cci'].shift(1)

    df['atr'] = calc_atr(df)
    df['vol_ma'] = df['volume'].rolling(window=5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['is_bullish_candle'] = df['close'] > df['open']

    # 识别市场状态
    regime = detect_market_regime(df)
    params = STRATEGY_REPO[regime]

    if regime == 'TRASH':
        return None

    # 执行回测（简化版，只统计基本指标）
    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_date = None
    entry_atr = 0.0
    position_type = None

    trades = []
    equity = []

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        # 计算权益
        if position != 0:
            if position_type == 'long':
                unrealized_pnl = (current['close'] - entry_price) * abs(position) * MULTIPLIER
            else:
                unrealized_pnl = (entry_price - current['close']) * abs(position) * MULTIPLIER
            current_equity = balance + unrealized_pnl
        else:
            current_equity = balance
        equity.append(current_equity)

        # 平仓
        if position != 0:
            exit_triggered = False

            if position_type == 'long':
                current_profit_atr = (current['close'] - entry_price) / entry_atr
            else:
                current_profit_atr = (entry_price - current['close']) / entry_atr

            # 固定止盈
            if params['take_profit'] < 999:
                if current_profit_atr > params['take_profit']:
                    exit_price = current['close']
                    if position_type == 'long':
                        pnl = (exit_price - entry_price) * abs(position) * MULTIPLIER
                    else:
                        pnl = (entry_price - exit_price) * abs(position) * MULTIPLIER
                    commission = exit_price * abs(position) * MULTIPLIER * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl

                    trades.append({
                        'direction': position_type,
                        'entry_date': entry_date,
                        'exit_date': current.name,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'take_profit',
                        'hold_days': (current.name - entry_date).days,
                    })

                    position = 0
                    entry_price = 0.0
                    position_type = None
                    exit_triggered = True

            # ATR止损
            if not exit_triggered:
                if position_type == 'long':
                    sl_price = entry_price - (params['stop_loss'] * entry_atr)
                    hit_sl = next_day['low'] <= sl_price
                    if hit_sl:
                        exit_price = max(next_day['open'], sl_price)
                else:
                    sl_price = entry_price + (params['stop_loss'] * entry_atr)
                    hit_sl = next_day['high'] >= sl_price
                    if hit_sl:
                        exit_price = min(next_day['open'], sl_price)

                if hit_sl:
                    if position_type == 'long':
                        pnl = (exit_price - entry_price) * abs(position) * MULTIPLIER
                    else:
                        pnl = (entry_price - exit_price) * abs(position) * MULTIPLIER
                    commission = exit_price * abs(position) * MULTIPLIER * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl

                    trades.append({
                        'direction': position_type,
                        'entry_date': entry_date,
                        'exit_date': next_day.name,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'type': 'stop_loss',
                        'hold_days': (next_day.name - entry_date).days,
                    })

                    position = 0
                    entry_price = 0.0
                    position_type = None
                    exit_triggered = True

        # 开仓
        if position == 0:
            # 做空信号
            if params['short_enabled']:
                stc_trigger = (df.iloc[i-1]['stc'] > params['stc_entry']) if i > 0 else False
                stc_confirmed = current['stc'] < 0
                stc_slope_ok = current['stc_slope'] < params['stc_slope']
                vol_ok = current['vol_ratio'] > params['vol_filter']
                cci_trigger = (df.iloc[i-1]['cci'] > params['cci_entry']) if i > 0 else False
                cci_confirmed = current['cci'] < current['cci_prev']

                if stc_trigger and stc_confirmed and stc_slope_ok and vol_ok and cci_trigger and cci_confirmed:
                    entry_price = next_day['open'] * (1 - SLIPPAGE_RATE)
                    entry_atr = current['atr']
                    stop_distance = entry_atr * params['stop_loss']

                    risk_amount = balance * 0.02
                    max_value = balance / MARGIN_RATE
                    qty_by_risk = int(risk_amount / (stop_distance * MULTIPLIER))
                    qty_by_capital = int(max_value / (entry_price * MULTIPLIER))
                    qty = min(max(1, qty_by_risk), qty_by_capital)

                    commission = entry_price * qty * MULTIPLIER * COMMISSION_RATE
                    balance -= commission

                    position = qty
                    entry_date = next_day.name
                    position_type = 'short'

            # 做多信号
            if params['long_enabled']:
                stc_trigger = (df.iloc[i-1]['stc'] < -params['stc_entry']) if i > 0 else False
                stc_confirmed = current['stc'] > 0
                stc_slope_ok = current['stc_slope'] > abs(params['stc_slope'])
                vol_ok = current['vol_ratio'] > params['vol_filter']
                is_bullish = current['is_bullish_candle']
                cci_trigger = (df.iloc[i-1]['cci'] < -params['cci_entry']) if i > 0 else False
                cci_confirmed = current['cci'] > current['cci_prev']

                if stc_trigger and stc_confirmed and stc_slope_ok and vol_ok and is_bullish and cci_trigger and cci_confirmed:
                    entry_price = next_day['open'] * (1 + SLIPPAGE_RATE)
                    entry_atr = current['atr']
                    stop_distance = entry_atr * params['stop_loss']

                    risk_amount = balance * 0.02
                    max_value = balance / MARGIN_RATE
                    qty_by_risk = int(risk_amount / (stop_distance * MULTIPLIER))
                    qty_by_capital = int(max_value / (entry_price * MULTIPLIER))
                    qty = min(max(1, qty_by_risk), qty_by_capital)

                    commission = entry_price * qty * MULTIPLIER * COMMISSION_RATE
                    balance -= commission

                    position = qty
                    entry_date = next_day.name
                    position_type = 'long'

    # 计算结果
    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        return None

    total_return = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
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

    long_trades = trades_df[trades_df['direction'] == 'long']
    short_trades = trades_df[trades_df['direction'] == 'short']

    long_pnl = long_trades['pnl'].sum() if len(long_trades) > 0 else 0
    short_pnl = short_trades['pnl'].sum() if len(short_trades) > 0 else 0

    return {
        'symbol': symbol,
        'name': SYMBOL_MAP.get(symbol, symbol),
        'regime': regime,
        'total_return': total_return,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'win_loss_ratio': win_loss_ratio,
        'profit_factor': profit_factor,
        'total_trades': len(trades_df),
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_pnl': long_pnl,
        'short_pnl': short_pnl,
        'final_balance': balance,
    }


def main():
    print("=" * 100)
    print("全品种批量回测系统 - S.C.C.策略".center(100))
    print("=" * 100)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    results = []
    tested = 0
    success = 0

    # 遍历所有品种
    for symbol in sorted(SYMBOL_MAP.keys()):
        tested += 1
        print(f"\n[{tested}/{len(SYMBOL_MAP)}] 正在测试: {symbol} ({SYMBOL_MAP[symbol]}) ...")

        df = get_data(symbol)

        if df is None or len(df) < 100:
            print(f"    [SKIP] 数据不足或获取失败")
            continue

        print(f"    [数据] 时间范围: {df.index[0]} 至 {df.index[-1]} | 数据量: {len(df)}条")

        try:
            result = run_backtest(df, symbol)
            if result:
                results.append(result)
                success += 1
                print(f"    [结果] 收益: {result['total_return']:>8.2f}% | "
                      f"回撤: {result['max_dd']:>7.2f}% | "
                      f"交易: {result['total_trades']:>4}笔 | "
                      f"模式: {result['regime']}")
            else:
                print(f"    [SKIP] 无交易")
        except Exception as e:
            print(f"    [ERROR] 回测失败: {e}")

    # 输出汇总报告
    print("\n" + "=" * 100)
    print("全品种回测报告".center(100))
    print("=" * 100)

    if results:
        # 按收益率排序
        results_df = pd.DataFrame(results).sort_values('total_return', ascending=False)

        print(f"\n测试成功: {success}/{tested} 个品种\n")

        print(f"  {'品种代码':<8} {'品种名称':<10} {'模式':<15} {'收益率':>10} {'最大回撤':>10} "
              f"{'交易数':>8} {'胜率':>8} {'盈亏比':>8}")
        print("-" * 100)

        for _, r in results_df.iterrows():
            print(f"  {r['symbol']:<8} {r['name']:<10} {r['regime']:<15} {r['total_return']:>9.2f}% "
                  f"{r['max_dd']:>9.2f}% {r['total_trades']:>8}笔 "
                  f"{r['win_rate']:>7.1f}% {r['win_loss_ratio']:>8.2f}")

        # 统计各模式表现
        print("\n" + "=" * 100)
        print("策略模式统计".center(100))
        print("=" * 100)

        regime_stats = results_df.groupby('regime').agg({
            'symbol': 'count',
            'total_return': ['mean', 'std'],
            'max_dd': 'mean',
            'total_trades': 'sum',
        })

        print(f"\n  {'模式':<15} {'品种数':>8} {'平均收益':>12} {'收益标准差':>12} {'平均回撤':>12} {'总交易':>10}")
        print("-" * 80)

        for regime, data in regime_stats.iterrows():
            print(f"  {regime:<15} {int(data[('symbol', 'count')]):>8} {data[('total_return', 'mean')]:>11.2f}% "
                  f"{data[('total_return', 'std')]:>11.2f}% {data[('max_dd', 'mean')]:>11.2f}% "
                  f"{int(data[('total_trades', 'sum')]):>10}笔")

        # 找出最佳品种
        best = results_df.iloc[0]
        print("\n" + "=" * 100)
        print("最佳品种推荐".center(100))
        print("=" * 100)
        print(f"\n  🏆 {best['name']} ({best['symbol']})")
        print(f"     收益率: {best['total_return']:.2f}%")
        print(f"     最大回撤: {best['max_dd']:.2f}%")
        print(f"     交易次数: {best['total_trades']}笔")
        print(f"     策略模式: {best['regime']}")

    else:
        print("\n[FAILED] 没有品种通过回测")

    print("\n" + "=" * 100)
    print(f"回测完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # 保存详细结果
    if results:
        output_file = r'D:\期货数据\铜期货监控\CCI策略系统\全品种回测结果.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
