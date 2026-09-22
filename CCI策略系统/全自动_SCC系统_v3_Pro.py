# -*- coding: utf-8 -*-
"""
全自动 S.C.C. 智能交易系统 v3.0 (Pro版)
核心升级：
1. 标准化 STC 算法 (Stochastic of Stochastic)
2. 完整的信号判定逻辑
3. 差异化风控模块
"""
import pandas as pd
import numpy as np
from datetime import datetime

# ========================================================
# 1. 策略参数仓库
# ========================================================
STRATEGY_REPO = {
    # 【模式A：空头狙击 (SNIPER_SHORT)】
    'SNIPER_SHORT': {
        'desc': '高波-空头狙击',
        'long_enabled': False,
        'short_enabled': True,
        'stc_entry': 80,          # STC高位死叉
        'stc_slope': -1.0,
        'cci_entry': 50,           # CCI跌破此值
        'cci_period': 14,
        'vol_filter': 0.5,
        'stop_loss': 1.5,
        'take_profit': 2.5,
        'use_ma_filter': False,
    },

    # 【模式B：多头突击 (SNIPER_LONG)】
    'SNIPER_LONG': {
        'desc': '高波-多头突击',
        'long_enabled': True,
        'short_enabled': False,
        'stc_entry': 20,          # STC低位金叉
        'stc_slope': 1.0,
        'cci_entry': -50,          # CCI突破此值
        'cci_period': 14,
        'vol_filter': 0.8,
        'stop_loss': 1.5,
        'take_profit': 3.0,
        'use_ma_filter': False,
    },

    # 【模式C：稳健趋势 (TREND)】
    'TREND': {
        'desc': '稳健趋势跟随',
        'long_enabled': True,
        'short_enabled': True,
        'stc_entry': 25,
        'stc_short_entry': 75,
        'stc_slope': -1.5,
        'cci_entry': 100,
        'cci_short_entry': -100,
        'cci_period': 20,
        'vol_filter': 1.0,
        'stop_loss': 2.0,
        'take_profit': 999,  # 无固定止盈
        'use_ma_filter': True,
    },

    # 【模式D：休眠 (TRASH)】
    'TRASH': {
        'desc': '休眠模式',
        'long_enabled': False,
        'short_enabled': False,
    }
}

# 全局交易参数
MULTIPLIER = 20
COMMISSION_RATE = 0.0001
SLIPPAGE_RATE = 0.0002
INITIAL_CAPITAL = 100000
MARGIN_RATE = 0.10
RISK_PER_TRADE = 0.02


# ========================================================
# 2. 标准化 STC 计算
# ========================================================
def calculate_stc_standard(df, period_fast=23, period_slow=50, k_period=10, d_period=3):
    """
    标准化 Schaff Trend Cycle 计算
    Stochastic of Stochastic: 对MACD的%K线做两次平滑归一化
    """
    # 1. MACD
    ema_fast = df['close'].ewm(span=period_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=period_slow, adjust=False).mean()
    macd = ema_fast - ema_slow

    # 2. First Stochastic (%K1)
    min_macd = macd.rolling(window=k_period).min()
    max_macd = macd.rolling(window=k_period).max()
    stoch_k1 = 100 * (macd - min_macd) / (max_macd - min_macd)
    stoch_k1 = stoch_k1.fillna(50)  # 填充初期值

    # 3. First Smoothing (%D1 of %K1)
    stoch_d1 = stoch_k1.ewm(span=d_period, adjust=False).mean()

    # 4. Second Stochastic (%K2 of %D1)
    min_stoch_d1 = stoch_d1.rolling(window=k_period).min()
    max_stoch_d1 = stoch_d1.rolling(window=k_period).max()
    stoch_k2 = 100 * (stoch_d1 - min_stoch_d1) / (max_stoch_d1 - min_stoch_d1)
    stoch_k2 = stoch_k2.fillna(50)

    # 5. Second Smoothing (%D2 of %K2) → Final STC
    stc = stoch_k2.ewm(span=d_period, adjust=False).mean()

    return stc


# ========================================================
# 3. 市场状态识别
# ========================================================
def detect_market_regime(df):
    """自动识别市场状态"""
    if len(df) < 60:
        return 'TRASH'

    # 计算ATR（用于波动率）
    tr = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1)))
    atr20 = tr.rolling(window=20).mean().iloc[-1]
    current_price = df['close'].iloc[-1]
    vol_ratio = (atr20 / current_price) * 100

    # 计算趋势方向和强度
    ema20 = df['close'].ewm(span=20).mean().iloc[-1]
    ema60 = df['close'].ewm(span=60).mean().iloc[-1]
    trend_strength = abs(ema20 - ema60) / ema60 * 100
    is_bullish = ema20 > ema60

    is_strong_trend = trend_strength > 2.0  # 修正后的阈值

    print(f"  [系统自检] 波动:{vol_ratio:.2f}% | 强度:{trend_strength:.2f}% | 多头:{is_bullish}")

    # 自动判定
    if vol_ratio > 2.0:
        return 'SNIPER_LONG' if is_bullish else 'SNIPER_SHORT'
    elif vol_ratio > 0.8 and is_strong_trend:
        return 'TREND'
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


# ========================================================
# 4. 指标计算
# ========================================================
def calculate_indicators(df, cci_period=14):
    """计算技术指标"""
    # 1. CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=cci_period).mean()
    mad = tp.rolling(window=cci_period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_prev'] = df['cci'].shift(1)

    # 2. STC（标准版）
    df['stc'] = calculate_stc_standard(df)

    # 3. ATR
    df['tr'] = np.maximum(df['high'] - df['low'], np.abs(df['high'] - df['close'].shift(1)))
    df['atr'] = df['tr'].rolling(14).mean()

    # 4. 成交量
    df['vol_ma'] = df['volume'].rolling(window=5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']

    # 5. 趋势过滤（TREND模式用）
    df['is_bullish_candle'] = df['close'] > df['open']
    if 'ma60' not in df.columns:
        df['ma60'] = df['close'].ewm(span=60).mean()

    return df


# ========================================================
# 5. 策略执行引擎
# ========================================================
def auto_run_strategy(df, symbol):
    """自动执行策略"""
    regime = detect_market_regime(df)
    params = STRATEGY_REPO[regime]

    print(f"  [AI决策] 识别为: 【{params['desc']}】 -> 自动加载参数")

    if regime == 'TRASH':
        print("  [操作] 市场状态不适合交易")
        return None

    # 动态计算指标
    df = calculate_indicators(df, cci_period=params['cci_period'])

    # 回测变量
    balance = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_atr = 0.0
    position_type = None

    trades = []

    print(f"  [回测开始] 模式: {regime} | 做多:{params['long_enabled']} | 做空:{params['short_enabled']}")

    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # 平仓逻辑
        if position != 0:
            profit_raw = (row['close'] - entry_price) * position
            profit_atr = profit_raw / entry_atr

            exit_triggered = False

            # 1. 固定止盈（SNIPER模式）
            if params['take_profit'] < 999:
                if profit_atr > params['take_profit']:
                    exit_price = row['close']
                    pnl = (exit_price - entry_price) * position * MULTIPLIER
                    commission = exit_price * abs(position) * MULTIPLIER * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl

                    trades.append({
                        'date': df.index[i],
                        'type': 'take_profit',
                        'pnl': net_pnl,
                    })

                    position = 0
                    exit_triggered = True

            # 2. 移动止损（TREND模式：跌破MA60）
            if not exit_triggered and params.get('use_ma_filter', False):
                if position > 0 and row['close'] < row['ma60']:
                    exit_price = row['close']
                    pnl = (exit_price - entry_price) * position * MULTIPLIER
                    commission = exit_price * abs(position) * MULTIPLIER * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl

                    trades.append({
                        'date': df.index[i],
                        'type': 'trend_stop',
                        'pnl': net_pnl,
                    })

                    position = 0
                    exit_triggered = True

            # 3. ATR止损（所有模式通用）
            if not exit_triggered:
                if profit_atr < -params['stop_loss']:
                    if position > 0:
                        stop_price = entry_price * (1 - params['stop_loss'])
                    else:
                        stop_price = entry_price * (1 + params['stop_loss'])

                    pnl = (stop_price - entry_price) * position * MULTIPLIER
                    commission = stop_price * abs(position) * MULTIPLIER * COMMISSION_RATE * 2
                    net_pnl = pnl - commission
                    balance += net_pnl

                    trades.append({
                        'date': df.index[i],
                        'type': 'atr_stop',
                        'pnl': net_pnl,
                    })

                    position = 0
                    exit_triggered = True

        # 开仓逻辑
        if position == 0:
            # 做空信号
            if params['short_enabled']:
                threshold = params['stc_entry']
                cond_stc = (prev['stc'] > threshold) and (row['stc'] < 0)
                cond_slope = row['stc'] < params['stc_slope']

                cci_threshold = params.get('stc_short_entry', params['cci_entry'])
                cond_cci = (prev['cci'] > cci_threshold) and (row['cci'] < cci_threshold)

                cond_vol = row['vol_ratio'] > params['vol_filter']

                if cond_stc and cond_slope and cond_cci and cond_vol:
                    position = -1
                    entry_price = row['close']
                    entry_atr = row['atr']

            # 做多信号
            if params['long_enabled']:
                threshold = params['stc_entry']
                cond_stc = (prev['stc'] < -threshold) and (row['stc'] > 0)
                cond_slope = row['stc'] > abs(params['stc_slope'])

                cci_threshold = params.get('stc_short_entry', params['cci_entry'])
                cond_cci = (prev['cci'] < cci_threshold) and (row['cci'] > cci_threshold)

                cond_vol = row['vol_ratio'] > params['vol_filter']
                cond_ma = True
                if params.get('use_ma_filter', False):
                    cond_ma = row['close'] > row['ma60']

                if cond_stc and cond_slope and cond_cci and cond_vol and cond_ma:
                    position = 1
                    entry_price = row['close']
                    entry_atr = row['atr']

    # 计算结果
    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df['pnl'].sum()
    total_return = (balance + total_pnl - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    print(f"  [回测结果] 交易:{len(trades_df)}笔 | 收益:{total_return:.2f}%")

    return {
        'symbol': symbol,
        'regime': regime,
        'total_return': total_return,
        'total_trades': len(trades_df),
        'trades_df': trades_df,
        'final_balance': balance + total_pnl,
    }


# ========================================================
# 6. 主程序
# ========================================================
if __name__ == "__main__":
    print("=" * 100)
    print("全自动 S.C.C. 智能交易系统 v3.0 (Pro版)".center(100))
    print("=" * 100)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

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
    print(f"  {'品种':<8} {'模式':<12} {'收益率':>10} {'交易数':>8}")
    print("-" * 50)

    for r in results:
        print(f"  {r['symbol']:<8} {r['regime']:<12} {r['total_return']:>9.2f}% {r['total_trades']:>8}笔")

    print("\n" + "=" * 100)
    print(f"系统运行完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
