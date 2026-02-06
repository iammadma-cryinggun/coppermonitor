# -*- coding: utf-8 -*-
"""
TOP 10期货品种做空完整回测
使用准确的资金管理、仓位控制、止损止盈逻辑
"""

import pandas as pd
import numpy as np
from pathlib import Path
from china_futures_fetcher import ChinaFuturesFetcher
from datetime import datetime
import json

# TOP 10品种配置（含优化后的做空参数）
TOP10_FUTURES_CONFIG = {
    'PTA': {
        'name': 'PTA',
        'code': 'TA',
        'exchange': 'CZCE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 10,
            'RSI_FILTER': 40,
            'RATIO_TRIGGER': 1.25,
            'STC_SELL_ZONE': 75,
        },
        'short_params': {
            'EMA_FAST': 7,
            'EMA_SLOW': 20,
            'RSI_FILTER': 45,
            'RATIO_TRIGGER': 1.05,
            'STC_BUY_ZONE': 15,
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '沪镍': {
        'name': '沪镍',
        'code': 'NI',
        'exchange': 'SHFE',
        'long_params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 20,
            'RSI_FILTER': 55,
            'RATIO_TRIGGER': 1.10,
            'STC_SELL_ZONE': 75,
        },
        'short_params': {
            'EMA_FAST': 5,
            'EMA_SLOW': 20,
            'RSI_FILTER': 45,
            'RATIO_TRIGGER': 1.05,
            'STC_BUY_ZONE': 15,
        },
        'contract_size': 1,
        'margin_rate': 0.12
    },
    '棕榈油': {
        'name': '棕榈油',
        'code': 'P',
        'exchange': 'DCE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 10,
            'RSI_FILTER': 45,
            'RATIO_TRIGGER': 1.20,
            'STC_SELL_ZONE': 75,
        },
        'short_params': {
            'EMA_FAST': 14,
            'EMA_SLOW': 15,
            'RSI_FILTER': 50,
            'RATIO_TRIGGER': 1.05,
            'STC_BUY_ZONE': 15,
        },
        'contract_size': 10,
        'margin_rate': 0.08
    },
    '纯碱': {
        'name': '纯碱',
        'code': 'SA',
        'exchange': 'CZCE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 20,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
        },
        'short_params': {
            'EMA_FAST': 10,
            'EMA_SLOW': 15,
            'RSI_FILTER': 50,
            'RATIO_TRIGGER': 1.05,
            'STC_BUY_ZONE': 15,
        },
        'contract_size': 20,
        'margin_rate': 0.08
    },
    'PVC': {
        'name': 'PVC',
        'code': 'V',
        'exchange': 'DCE',
        'long_params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 25,
            'RSI_FILTER': 55,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
        },
        'short_params': {
            'EMA_FAST': 5,
            'EMA_SLOW': 30,
            'RSI_FILTER': 45,
            'RATIO_TRIGGER': 1.05,
            'STC_BUY_ZONE': 15,
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '沪铜': {
        'name': '沪铜',
        'code': 'CU',
        'exchange': 'SHFE',
        'long_params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 20,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
        },
        'short_params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 15,
            'RSI_FILTER': 45,
            'RATIO_TRIGGER': 1.05,
            'STC_BUY_ZONE': 15,
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '豆粕': {
        'name': '豆粕',
        'code': 'M',
        'exchange': 'DCE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 20,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.25,
            'STC_SELL_ZONE': 75,
        },
        'short_params': {
            'EMA_FAST': 10,
            'EMA_SLOW': 15,
            'RSI_FILTER': 45,
            'RATIO_TRIGGER': 1.25,
            'STC_BUY_ZONE': 15,
        },
        'contract_size': 10,
        'margin_rate': 0.08
    },
    '沪锡': {
        'name': '沪锡',
        'code': 'SN',
        'exchange': 'SHFE',
        'long_params': {
            'EMA_FAST': 3,
            'EMA_SLOW': 10,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.25,
            'STC_SELL_ZONE': 75,
        },
        'short_params': {
            'EMA_FAST': 2,
            'EMA_SLOW': 10,
            'RSI_FILTER': 60,
            'RATIO_TRIGGER': 1.05,
            'STC_BUY_ZONE': 15,
        },
        'contract_size': 1,
        'margin_rate': 0.13
    },
    '沪铅': {
        'name': '沪铅',
        'code': 'PB',
        'exchange': 'SHFE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 10,
            'RSI_FILTER': 40,
            'RATIO_TRIGGER': 1.05,
            'STC_SELL_ZONE': 75,
        },
        'short_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 15,
            'RSI_FILTER': 40,
            'RATIO_TRIGGER': 1.05,
            'STC_BUY_ZONE': 15,
        },
        'contract_size': 5,
        'margin_rate': 0.08
    },
    '玻璃': {
        'name': '玻璃',
        'code': 'FG',
        'exchange': 'CZCE',
        'long_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 10,
            'RSI_FILTER': 35,
            'RATIO_TRIGGER': 1.10,
            'STC_SELL_ZONE': 75,
        },
        'short_params': {
            'EMA_FAST': 12,
            'EMA_SLOW': 15,
            'RSI_FILTER': 50,
            'RATIO_TRIGGER': 1.05,
            'STC_BUY_ZONE': 15,
        },
        'contract_size': 20,
        'margin_rate': 0.08
    }
}

def calculate_indicators(df, params):
    """计算技术指标"""
    df['ema_fast'] = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()

    exp1 = df['close'].ewm(span=params['EMA_FAST'], adjust=False).mean()
    exp2 = df['close'].ewm(span=params['EMA_SLOW'], adjust=False).mean()
    df['macd_dif'] = exp1 - exp2
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['ratio'] = np.where(df['macd_dea'] != 0, df['macd_dif'] / df['macd_dea'], 0)

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # STC指标
    def calculate_stc(series, fast_period=23, slow_period=50, period=10):
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        stoch_k = 100 * (macd - macd.rolling(window=period).min()) / \
                  (macd.rolling(window=period).max() - macd.rolling(window=period).min())
        stoch_d = stoch_k.ewm(span=period, adjust=False).mean()
        stoch_k_d = 100 * (stoch_k - stoch_d.rolling(window=period).min()) / \
                    (stoch_d.rolling(window=period).max() - stoch_d.rolling(window=period).min())
        stc = stoch_k_d.ewm(span=period, adjust=False).mean()
        return stc

    df['stc'] = calculate_stc(df['close'])
    df['stc_prev'] = df['stc'].shift(1)

    return df

def backtest_short_complete(code, name, params, contract_size, margin_rate):
    """
    完整做空回测（使用准确的资金管理）
    """
    # 获取数据
    fetcher = ChinaFuturesFetcher()
    df = fetcher.get_historical_data(code, days=300)

    if df is None or df.empty:
        return None

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = calculate_indicators(df, params)

    # 回测参数
    INITIAL_CAPITAL = 50000
    MAX_POSITION_RATIO = 0.80  # 最大仓位80%
    MAX_SINGLE_LOSS_PCT = 0.15  # 单笔最大亏损15%
    STOP_LOSS_PCT = 0.02  # 止损2%

    capital = INITIAL_CAPITAL
    position = None
    all_trades = []

    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # 持仓管理
        if position is not None:
            exit_triggered = False
            exit_price = None
            exit_reason = None

            # 止损检查（做空：高点触及止损价）
            if current['high'] >= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_triggered = True
                exit_reason = '止损'

            # 平空仓信号检查
            if not exit_triggered:
                stc_exit = (prev['stc'] < params['STC_BUY_ZONE']) and (current['stc'] > prev['stc'])
                trend_exit = current['ema_fast'] > current['ema_slow']

                if stc_exit:
                    exit_price = current['close']
                    exit_triggered = True
                    exit_reason = 'STC止盈'
                elif trend_exit:
                    exit_price = current['close']
                    exit_triggered = True
                    exit_reason = '趋势反转'

            # 执行平仓
            if exit_triggered:
                pnl = (position['entry_price'] - exit_price) * position['contracts'] * contract_size
                capital += pnl

                trade = {
                    'entry_datetime': position['entry_datetime'],
                    'exit_datetime': current['datetime'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'contracts': position['contracts'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / position['capital_at_entry']) * 100,
                    'capital_after': capital,
                    'exit_reason': exit_reason,
                    'margin_used': position['margin_used']
                }
                all_trades.append(trade)
                position = None
                continue

        # 开仓信号检查
        trend_down = current['ema_fast'] < current['ema_slow']
        ratio_safe = (-params['RATIO_TRIGGER'] < current['ratio'] < 0)
        ratio_falling = current['ratio'] < prev['ratio']
        turning_down = current['macd_dif'] < prev['macd_dif']
        is_weak = current['rsi'] < params['RSI_FILTER']

        sniper_short = trend_down and ratio_safe and ratio_falling and turning_down and is_weak
        ema_death_cross = (prev['ema_fast'] >= prev['ema_slow']) and (current['ema_fast'] < current['ema_slow'])
        chase_short = ema_death_cross and is_weak

        if (sniper_short or chase_short) and position is None:
            entry_price = current['close']
            stop_loss = entry_price * (1 + STOP_LOSS_PCT)

            # 计算手数（资金管理）
            contract_value = entry_price * contract_size
            margin_per_contract = contract_value * margin_rate
            potential_loss_per_contract = (stop_loss - entry_price) * contract_size

            max_contracts_by_margin = int((capital * MAX_POSITION_RATIO) / margin_per_contract)
            max_contracts_by_risk = int((capital * MAX_SINGLE_LOSS_PCT) / potential_loss_per_contract)
            contracts = min(max_contracts_by_margin, max_contracts_by_risk)

            if contracts > 0:
                margin_used = contracts * margin_per_contract

                position = {
                    'entry_datetime': current['datetime'],
                    'entry_price': entry_price,
                    'contracts': contracts,
                    'stop_loss': stop_loss,
                    'capital_at_entry': capital,
                    'margin_used': margin_used
                }

    if not all_trades:
        return None

    trades_df = pd.DataFrame(all_trades)

    total_pnl = trades_df['pnl'].sum()
    total_pnl_pct = (total_pnl / INITIAL_CAPITAL) * 100
    win_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = (win_trades / len(trades_df)) * 100

    return {
        'code': code,
        'name': name,
        'params': params,
        'total_pnl': total_pnl,
        'total_pnl_pct': total_pnl_pct,
        'trades': len(trades_df),
        'win_rate': win_rate,
        'final_capital': capital,
        'trades_df': trades_df
    }

def batch_backtest_all():
    """批量回测所有品种"""

    print("=" * 140)
    print("TOP 10期货品种做空完整回测".center(140))
    print("=" * 140)

    results = []

    for key, config in TOP10_FUTURES_CONFIG.items():
        print(f"\n{'='*140}")
        print(f"回测 {key} ({config['name']})")
        print(f"{'='*140}")

        result = backtest_short_complete(
            config['code'],
            config['name'],
            config['short_params'],
            config['contract_size'],
            config['margin_rate']
        )

        if result:
            results.append(result)

            print(f"\n参数: EMA({result['params']['EMA_FAST']},{result['params']['EMA_SLOW']}), "
                  f"RSI={result['params']['RSI_FILTER']}, RATIO={result['params']['RATIO_TRIGGER']}, "
                  f"STC={result['params']['STC_BUY_ZONE']}")
            print(f"结果: {result['total_pnl_pct']:+.2f}%, {result['trades']}笔, 胜率{result['win_rate']:.1f}%")
            print(f"资金: {50000:,.0f} → {result['final_capital']:,.0f} 元")
        else:
            print("  无交易记录")

    # 汇总结果
    print(f"\n\n{'='*140}")
    print("回测汇总")
    print(f"{'='*140}")

    print(f"\n{'品种':<10} {'代码':<8} {'EMA':<10} {'RSI':<8} {'RATIO':<10} {'STC':<8} {'收益率':<12} {'交易':<6} {'胜率':<10} {'盈亏金额':<15}")
    print(f"{'-'*140}")

    for r in results:
        ema_str = f"({r['params']['EMA_FAST']},{r['params']['EMA_SLOW']})"
        pnl_str = f"{r['total_pnl_pct']:+.2f}%"
        amount_str = f"{r['total_pnl']:+,.0f}元"

        print(f"{r['name']:<10} {r['params'].get('code', ''):<8} {ema_str:<10} "
              f"{r['params']['RSI_FILTER']:<8} "
              f"{r['params']['RATIO_TRIGGER']:<10.2f} "
              f"{r['params']['STC_BUY_ZONE']:<8} "
              f"{pnl_str:<12} "
              f"{r['trades']}笔{' ':<2} "
              f"{r['win_rate']:.1f}%{' ':<5} "
              f"{amount_str:<15}")

    return results

if __name__ == '__main__':
    batch_backtest_all()
