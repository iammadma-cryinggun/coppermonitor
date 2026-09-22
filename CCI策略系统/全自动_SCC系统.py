# -*- coding: utf-8 -*-
"""
全自动 S.C.C. 智能交易系统
核心功能：自动识别市场状态 -> 自动切换策略参数 -> 自动执行
无需人工干预，无需截图判断
"""
import pandas as pd
import numpy as np
from datetime import datetime

# ========================================================
# 1. 策略参数仓库 (自动调用)
# ========================================================
STRATEGY_REPO = {
    # 【模式A：空头狙击 (SNIPER_SHORT)】
    # 适用：纯碱 (高波动 + 下跌/震荡)
    'SNIPER_SHORT': {
        'desc': '高波-空头狙击',
        'long_enabled': False,      # 关多
        'short_enabled': True,       # 开空
        'stc_entry': 80,            # 高位死叉
        'stc_slope': -1.5,
        'cci_entry': 50,            # 超进
        'cci_period': 14,
        'vol_filter': 0.3,          # 只要有量就行
        'stop_loss': 1.5,
        'take_profit': 2.5,          # 固定止盈
        'use_ma_filter': False,
    },

    # 【模式B：多头突击 (SNIPER_LONG)】
    # 适用：铜 (高波动 + 上升趋势)
    'SNIPER_LONG': {
        'desc': '高波-多头突击',
        'long_enabled': True,        # 开多
        'short_enabled': False,      # 关空
        'stc_entry': 20,            # 低位金叉（20就干，不用等-20）
        'stc_slope': 1.5,
        'cci_entry': -50,           # 超卖回补
        'cci_period': 14,
        'vol_filter': 0.8,          # 上涨需要一点量
        'stop_loss': 1.5,
        'take_profit': 3.0,          # 趋势强，吃多点
        'use_ma_filter': False,
    },

    # 【模式C：稳健趋势 (TREND)】
    # 适用：黄金 (低波动 + 强趋势)
    'TREND': {
        'desc': '稳健趋势跟随',
        'long_enabled': True,
        'short_enabled': True,
        'stc_entry': 90,
        'stc_slope': -2,
        'cci_entry': 100,
        'cci_period': 20,
        'vol_filter': 1.0,
        'stop_loss': 2.0,
        'take_profit': 999,          # 无固定止盈
        'use_ma_filter': True,
    },

    # 【模式D：垃圾时间 (TRASH)】
    'TRASH': {
        'desc': '休眠模式',
        'long_enabled': False,
        'short_enabled': False
    }
}

# 全局交易参数
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
SLIPPAGE_RATE = 0.0002
INITIAL_CAPITAL = 100000
MARGIN_RATE = 0.10


# ========================================================
# 2. 核心大脑：自动状态识别 (Auto-Regime)
# ========================================================
def detect_market_regime(df):
    """
    数学化替代"人眼看盘"
    返回：'SNIPER_SHORT' | 'SNIPER_LONG' | 'TREND' | 'TRASH'
    """
    if len(df) < 60:
        return 'TRASH'

    # --- 计算体检指标 ---

    # 1. 波动率 (ATR Ratio)
    df['tr'] = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1)))
    df['atr'] = df['tr'].rolling(20).mean()
    current_price = df['close'].iloc[-1]
    current_atr = df['atr'].iloc[-1]

    vol_ratio = (current_atr / current_price) * 100

    # 2. 趋势强度 (ADX 简化版)
    ema20 = df['close'].ewm(span=20).mean().iloc[-1]
    ema60 = df['close'].ewm(span=60).mean().iloc[-1]
    trend_strength = abs(ema20 - ema60) / ema60 * 100

    # 3. 趋势方向
    is_bullish = ema20 > ema60  # 多头排列

    # 校准后的趋势度 (根据实测数据调整)
    # 纯碱是0.53，铜是4.44，所以我们把门槛定在 2.0
    is_strong_trend = trend_strength > 2.0

    print(f"  [系统自检] 波动:{vol_ratio:.2f}% | 强度:{trend_strength:.2f}% | 多头:{is_bullish}")

    # === 自动判定逻辑 ===

    # 1. 高波动 (妖股/疯牛)
    # 波动率 > 2.0%
    if vol_ratio > 2.0:
        if is_strong_trend and is_bullish:
            return 'SNIPER_LONG'  # 铜：高波+趋势上 -> 疯牛
        else:
            return 'SNIPER_SHORT'  # 纯碱：高波+趋势下/无 -> 疯狗

    # 2. 中波动 (贵族)
    # 波动率 0.8% - 2.0%
    elif vol_ratio > 0.8 and is_strong_trend:
        return 'TREND'

    # 3. 低波动 (垃圾时间)
    else:
        return 'TRASH'


def get_data(symbol):
    """获取数据"""
    import akshare as ak
    df = ak.futures_main_sina(symbol=symbol)
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


def auto_run_strategy(df, symbol):
    """自动执行策略"""
    print(f"\n正在分析品种: {symbol} ...")

    # A. 自动获取市场状态
    regime = detect_market_regime(df)
    params = STRATEGY_REPO[regime]

    print(f"  [AI决策] 识别为: 【{params['desc']}】 -> 自动加载参数")

    if regime == 'TRASH':
        print("  [操作] 波动率过低，跳过交易")
        return None

    # B. 准备数据 (计算指标)
    df['stc'] = calc_stc(df)
    df['stc_prev'] = df['stc'].shift(1)
    df['stc_slope'] = df['stc'] - df['stc_prev']

    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=params['cci_period']).mean()
    mad = tp.rolling(window=params['cci_period']).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_prev'] = df['cci'].shift(1)

    df['atr'] = calc_atr(df)
    df['vol_ma'] = df['volume'].rolling(window=5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['is_bullish_candle'] = df['close'] > df['open']

    # EMA过滤（趋势模式用）
    if params['use_ma_filter']:
        ema20 = df['close'].ewm(span=20).mean()
        df['trend_ok'] = df['close'] > ema20
    else:
        df['trend_ok'] = True  # 狙击模式不看均线

    # C. 执行信号 (根据 params 里的开关自动执行)
    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_date = None
    entry_atr = 0.0
    position_type = None

    trades = []
    equity = []

    print(f"  [回测开始] 模式: {regime} | 做多: {params['long_enabled']} | 做空: {params['short_enabled']}")

    for i in range(len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]

        # 计算当前权益
        if position != 0:
            if position_type == 'long':
                unrealized_pnl = (current['close'] - entry_price) * abs(position) * MULTIPLIER
            else:
                unrealized_pnl = (entry_price - current['close']) * abs(position) * MULTIPLIER
            current_equity = balance + unrealized_pnl
        else:
            current_equity = balance
        equity.append(current_equity)

        # 平仓逻辑
        if position != 0:
            exit_triggered = False

            # 计算浮盈（ATR倍数）
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

                    print(f"    {position_type.upper()}止盈: {current.name.date()} 盈利:{net_pnl:.0f}")
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

        # 开仓逻辑（根据 params 自动开关）
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

                    print(f"    开空: {entry_date.date()} 价格:{entry_price:.2f}")

            # 做多信号
            if params['long_enabled']:
                stc_trigger = (df.iloc[i-1]['stc'] < -params['stc_entry']) if i > 0 else False
                stc_confirmed = current['stc'] > 0
                stc_slope_ok = current['stc_slope'] > abs(params['stc_slope'])

                vol_ok = current['vol_ratio'] > params['vol_filter']
                is_bullish = current['is_bullish_candle']

                cci_trigger = (df.iloc[i-1]['cci'] < -params['cci_entry']) if i > 0 else False
                cci_confirmed = current['cci'] > current['cci_prev']

                trend_ok = current['trend_ok']

                if stc_trigger and stc_confirmed and stc_slope_ok and vol_ok and is_bullish and cci_trigger and cci_confirmed and trend_ok:
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

                    print(f"    开多: {entry_date.date()} 价格:{entry_price:.2f}")

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

    result = {
        'symbol': symbol,
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

    print(f"  [回测结果] 收益: {total_return:.2f}% | 回撤: {max_dd:.2f}% | 交易: {len(trades_df)}笔")

    return result


# ========================================================
# 主程序
# ========================================================
if __name__ == "__main__":
    print("=" * 100)
    print("全自动 S.C.C. 智能交易系统".center(100))
    print("自动识别市场状态 -> 自动切换策略参数 -> 自动执行".center(100))
    print("=" * 100)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # 品种列表
    symbols = ['sa0', 'cu0']

    results = []

    for s in symbols:
        print(f"\n{'='*100}")
        print(f"正在处理: {s}")
        print('='*100)

        df = get_data(s)
        print(f"  [数据加载] 时间范围: {df.index[0]} 至 {df.index[-1]} | 数据量: {len(df)}条")

        result = auto_run_strategy(df, s)

        if result:
            results.append(result)

    # 输出总结
    print("\n" + "=" * 100)
    print("全品种回测总结".center(100))
    print("=" * 100)
    print(f"  {'品种':<8} {'模式':<12} {'收益率':>10} {'最大回撤':>10} {'交易数':>8} {'做多':>6} {'做空':>6}")
    print("-" * 80)

    for r in results:
        print(f"  {r['symbol']:<8} {r['regime']:<12} {r['total_return']:>9.2f}% "
              f"{r['max_dd']:>9.2f}% {r['total_trades']:>8}笔 "
              f"{r['long_trades']:>6}笔 {r['short_trades']:>6}笔")

    print("\n" + "=" * 100)
    print(f"系统运行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
