# -*- coding: utf-8 -*-
"""
纯碱SA0 - S.C.C.系统 多空不对称版
核心：做多=结构猎人，做空=重力加速器
"""
import pandas as pd
import numpy as np
from datetime import datetime

SYMBOL = 'sa0'
EXCHANGE = 'CZCE'
MULTIPLIER = 20
MIN_PRICE_TICK = 1
MARGIN_RATE = 0.10

COMMISSION_RATE = 0.0001
SLIPPAGE_TICKS = 0.5

# ==========================================
# 多空不对称参数
# ==========================================
ASYMMETRIC_PARAMS = {
    # ========================================
    # 做多参数：结构猎人
    # ========================================
    'long': {
        # STC: 从底部抬头
        'stc_oversold': -30,
        'stc_awake': -20,
        'stc_slope_min': 2,

        # CCI: 突破确认
        'cci_oversold': -100,
        'cci_zero_cross': -80,
        'cci_confirm_days': 2,  # 2日确认

        # 量能：必须放量（抵抗重力需要能量）
        'vol_threshold': 0.9,
        'require_bullish_candle': True,  # 必须阳线

        # 止盈：可以慢（上涨有惯性）
        'cci_first_target': 100,
        'profit_atr_multiple': 1.5,
        'half_exit_ratio': 0.5,

        # 止损
        'atr_stop_multiplier': 2.0,
    },

    # ========================================
    # 做空参数：重力加速器
    # ========================================
    'short': {
        # STC: 高位掉头（不等对称，只要跌破80就危险）
        'stc_overbought': -20,  # STC跌破-20（高位钝化后掉头）
        'stc_breakdown': -80,  # STC跌破-80（确认下跌）
        'stc_slope_min': -2,  # STC斜率<-2（快速掉头）

        # CCI: 跌破关键位（主跌浪特征）
        'cci_crash_threshold': -50,  # CCI跌破-50（主跌浪开始）
        'cci_crash_aggressive': -100,  # CCI跌破-100（激进）
        'cci_zero_cross': 0,  # CCI跌破0轴（中线破位）

        # 量能：不需要放量（重力自然下坠）
        # 只要不是"缩量滞跌"就行
        'vol_max_ratio': 3.0,  # vol_ratio不能>3（缩量滞跌）
        'vol_min_ratio': 0.5,  # vol_ratio不能<0.5（完全没量）

        # 止盈：必须快（暴力反弹危险）
        'cci_first_target': -150,  # 更敏感！CCI<-150就止盈50%
        'profit_atr_multiple': 1.0,  # 更快！浮盈1倍ATR就止盈
        'half_exit_ratio': 0.5,

        # 止损：更紧（空头怕逼空）
        'atr_stop_multiplier': 1.5,  # 更紧的止损
    },
}

# 共享参数
RISK_PER_TRADE = 0.02
CCI_OVERBOUGHT_LONG = 130
CCI_OVERSOLD_SHORT = -130  # 做空的超卖平仓（CCI<-130平空）
ATR_LENGTH = 14
VOL_MA_PERIOD = 5
CCI_LENGTH = 12
CCI_MA_LENGTH = 5

INITIAL_CAPITAL = 100000


class AsymmetricBacktest:
    """多空不对称回测引擎"""

    def __init__(self, df, params, capital):
        self.df = df.copy()
        self.params = params
        self.initial_capital = capital

        # 账户状态
        self.balance = capital
        self.position_long = 0
        self.position_short = 0
        self.avg_price_long = 0.0
        self.avg_price_short = 0.0
        self.entry_date_long = None
        self.entry_date_short = None
        self.entry_atr_long = 0.0
        self.entry_atr_short = 0.0
        self.has_taken_half_profit_long = False
        self.has_taken_half_profit_short = False

        # 交易记录
        self.trades = []
        self.equity = []

    def calc_margin_required(self, price, qty):
        contract_value = price * qty * MULTIPLIER
        margin = contract_value * MARGIN_RATE
        return margin

    def check_margin_sufficient(self, price, qty):
        margin_required = self.calc_margin_required(price, qty)
        return self.balance >= margin_required

    def open_long_position(self, i, price, reason=""):
        if self.position_long > 0:
            return False

        long_params = self.params['long']
        atr = self.df.iloc[i]['atr']

        max_qty_by_margin = int(self.balance / self.calc_margin_required(price, 1))
        stop_price = price - atr * long_params['atr_stop_multiplier']
        risk_per_contract = (price - stop_price) * MULTIPLIER
        risk_amount = self.balance * RISK_PER_TRADE
        max_qty_by_risk = int(risk_amount / risk_per_contract)
        qty = min(max_qty_by_margin, max_qty_by_risk)
        qty = max(1, qty)

        if not self.check_margin_sufficient(price, qty):
            return False

        margin_required = self.calc_margin_required(price, qty)
        commission = price * qty * MULTIPLIER * COMMISSION_RATE
        self.balance -= (margin_required + commission)

        self.position_long = qty
        self.avg_price_long = price
        self.entry_date_long = self.df.index[i]
        self.entry_atr_long = atr
        self.has_taken_half_profit_long = False

        return True

    def open_short_position(self, i, price, reason=""):
        if self.position_short > 0:
            return False

        short_params = self.params['short']
        atr = self.df.iloc[i]['atr']

        max_qty_by_margin = int(self.balance / self.calc_margin_required(price, 1))
        stop_price = price + atr * short_params['atr_stop_multiplier']
        risk_per_contract = (stop_price - price) * MULTIPLIER
        risk_amount = self.balance * RISK_PER_TRADE
        max_qty_by_risk = int(risk_amount / risk_per_contract)
        qty = min(max_qty_by_margin, max_qty_by_risk)
        qty = max(1, qty)

        if not self.check_margin_sufficient(price, qty):
            return False

        margin_required = self.calc_margin_required(price, qty)
        commission = price * qty * MULTIPLIER * COMMISSION_RATE
        self.balance -= (margin_required + commission)

        self.position_short = qty
        self.avg_price_short = price
        self.entry_date_short = self.df.index[i]
        self.entry_atr_short = atr
        self.has_taken_half_profit_short = False

        return True

    def close_long_position(self, i, price, exit_type, exit_reason=""):
        if self.position_long == 0:
            return

        pnl = (price - self.avg_price_long) * self.position_long * MULTIPLIER
        commission = price * self.position_long * MULTIPLIER * COMMISSION_RATE
        net_pnl = pnl - commission

        margin_returned = self.calc_margin_required(self.avg_price_long, self.position_long)
        self.balance += (margin_returned + net_pnl)

        self.trades.append({
            'direction': 'long',
            'entry_date': self.entry_date_long,
            'exit_date': self.df.index[i],
            'entry_price': self.avg_price_long,
            'exit_price': price,
            'pnl': net_pnl,
            'type': exit_type,
            'exit_reason': exit_reason,
            'hold_days': (self.df.index[i] - self.entry_date_long).days,
            'qty': self.position_long,
        })

        self.position_long = 0
        self.avg_price_long = 0.0
        self.entry_date_long = None
        self.has_taken_half_profit_long = False

    def close_short_position(self, i, price, exit_type, exit_reason=""):
        if self.position_short == 0:
            return

        pnl = (self.avg_price_short - price) * self.position_short * MULTIPLIER
        commission = price * self.position_short * MULTIPLIER * COMMISSION_RATE
        net_pnl = pnl - commission

        margin_returned = self.calc_margin_required(self.avg_price_short, self.position_short)
        self.balance += (margin_returned + net_pnl)

        self.trades.append({
            'direction': 'short',
            'entry_date': self.entry_date_short,
            'exit_date': self.df.index[i],
            'entry_price': self.avg_price_short,
            'exit_price': price,
            'exit_price': price,
            'pnl': net_pnl,
            'type': exit_type,
            'exit_reason': exit_reason,
            'hold_days': (self.df.index[i] - self.entry_date_short).days,
            'qty': self.position_short,
        })

        self.position_short = 0
        self.avg_price_short = 0.0
        self.entry_date_short = None
        self.has_taken_half_profit_short = False

    def run_backtest(self):
        print("开始多空不对称回测...")

        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]
            next_day = self.df.iloc[i + 1]

            # 更新权益
            total_equity = self.balance
            if self.position_long > 0:
                unrealized_pnl_long = (current['close'] - self.avg_price_long) * self.position_long * MULTIPLIER
                total_equity += unrealized_pnl_long
            elif self.position_short > 0:
                unrealized_pnl_short = (self.avg_price_short - current['close']) * self.position_short * MULTIPLIER
                total_equity += unrealized_pnl_short
            self.equity.append(total_equity)

            # ========================================
            # 做多信号：结构猎人
            # ========================================
            if self.position_long == 0:
                long_params = self.params['long']

                # STC: 从底部抬头
                stc_prev_oversold = (self.df.iloc[i - 1]['stc'] < long_params['stc_oversold']) if i > 0 else False
                stc_current_awake = current['stc'] > long_params['stc_awake']
                stc_slope_strong = current['stc_slope'] > long_params['stc_slope_min']
                condition_stc = stc_prev_oversold and stc_current_awake and stc_slope_strong

                # 量能：必须放量（抵抗重力需要能量）
                vol_sufficient = current['vol_ratio'] > long_params['vol_threshold']
                is_bullish = current['is_bullish_candle']
                condition_vol = vol_sufficient and is_bullish

                # CCI: 2日确认
                if i > 0:
                    cci_rising_prev = self.df.iloc[i - 1]['cci'] > self.df.iloc[i - 1]['cci_prev'] if i > 1 else False
                    cci_rising_current = current['cci'] > current['cci_prev']
                    cci_consecutive = cci_rising_prev and cci_rising_current
                else:
                    cci_consecutive = False

                cci_breakout = (current['cci'] > long_params['cci_zero_cross'] and
                               current['cci'] > current['cci_ma'])

                condition_cci = cci_breakout and cci_consecutive

                if condition_stc and condition_vol and condition_cci:
                    entry_price = current['close'] + (MIN_PRICE_TICK * SLIPPAGE_TICKS)
                    if self.open_long_position(i, entry_price, f"STC抬头{current['stc']:.2f}"):
                        print(f"开多: {current.name.date()} 价格:{entry_price:.2f} "
                              f"CCI:{current['cci']:.2f} STC:{current['stc']:.2f} "
                              f"斜:{current['stc_slope']:.2f} 量比:{current['vol_ratio']:.2f}")

            # ========================================
            # 做空信号：重力加速器
            # ========================================
            if self.position_short == 0:
                short_params = self.params['short']

                # STC: 高位掉头
                stc_broken_down = current['stc'] < short_params['stc_breakdown']
                stc_slope_down = current['stc_slope'] < short_params['stc_slope_min']
                condition_stc = stc_broken_down and stc_slope_down

                # 量能：不需要放量（只要不是缩量滞跌）
                vol_ok = (short_params['vol_min_ratio'] < current['vol_ratio'] < short_params['vol_max_ratio'])

                # CCI: 主跌浪特征
                cci_crash_mild = current['cci'] < short_params['cci_crash_threshold']
                cci_crash_aggressive = current['cci'] < short_params['cci_crash_aggressive']
                cci_break_zero = current['cci'] < short_params['cci_zero_cross']
                condition_cci = cci_crash_mild or cci_crash_aggressive or cci_break_zero

                # 不需要强制阴线（重力自然下坠）
                if condition_stc and vol_ok and condition_cci:
                    entry_price = current['close'] - (MIN_PRICE_TICK * SLIPPAGE_TICKS)
                    if self.open_short_position(i, entry_price, f"STC掉头{current['stc']:.2f}"):
                        print(f"开空: {current.name.date()} 价格:{entry_price:.2f} "
                              f"CCI:{current['cci']:.2f} STC:{current['stc']:.2f} "
                              f"斜:{current['stc_slope']:.2f} 量比:{current['vol_ratio']:.2f}")

            # ========================================
            # 多头平仓逻辑
            # ========================================
            if self.position_long > 0:
                long_params = self.params['long']
                exit_triggered = False

                # 50%分批止盈
                if not self.has_taken_half_profit_long:
                    profit_atr = (current['close'] - self.avg_price_long) / self.entry_atr_long

                    if current['cci'] > long_params['cci_first_target'] or profit_atr > long_params['profit_atr_multiple']:
                        exit_qty = int(self.position_long * long_params['half_exit_ratio'])
                        exit_qty = max(1, exit_qty)
                        exit_price = current['close'] - (MIN_PRICE_TICK * SLIPPAGE_TICKS)
                        pnl = (exit_price - self.avg_price_long) * exit_qty * MULTIPLIER
                        commission = exit_price * exit_qty * MULTIPLIER * COMMISSION_RATE
                        net_pnl = pnl - commission

                        margin_returned = self.calc_margin_required(self.avg_price_long, exit_qty)
                        self.balance += (margin_returned + net_pnl)

                        self.trades.append({
                            'direction': 'long',
                            'entry_date': self.entry_date_long,
                            'exit_date': current.name,
                            'entry_price': self.avg_price_long,
                            'exit_price': exit_price,
                            'pnl': net_pnl,
                            'type': 'half_profit_long',
                            'exit_reason': f'50%止盈CCI:{current["cci"]:.2f}',
                            'hold_days': (current.name - self.entry_date_long).days,
                            'qty': exit_qty,
                        })

                        print(f"多50%止盈: {current.name.date()} 盈利:{net_pnl:.0f}")
                        self.position_long -= exit_qty
                        self.has_taken_half_profit_long = True

                        if self.position_long == 0:
                            exit_triggered = True

                # 剩余仓位止损（移到成本价）
                if not exit_triggered and self.position_long > 0:
                    if next_day['low'] <= self.avg_price_long:
                        stop_price = max(next_day['open'], self.avg_price_long)
                        stop_price -= (MIN_PRICE_TICK * SLIPPAGE_TICKS)
                        self.close_long_position(i + 1, stop_price, 'atr_stop_long', '保本止损')
                        exit_triggered = True

                # CCI超买平仓
                if not exit_triggered and self.position_long > 0:
                    if current['cci'] > CCI_OVERBOUGHT_LONG:
                        exit_price = current['close'] - (MIN_PRICE_TICK * SLIPPAGE_TICKS)
                        self.close_long_position(i, exit_price, 'cci_overbought_long', f'CCI超买{current["cci"]:.2f}')
                        exit_triggered = True

                # CCI死叉平仓
                if not exit_triggered and self.position_long > 0 and i > 0:
                    cci_prev = self.df.iloc[i - 1]['cci']
                    cci_ma_prev = self.df.iloc[i - 1]['cci_ma']
                    if cci_prev >= cci_ma_prev and current['cci'] < current['cci_ma']:
                        exit_price = current['close'] - (MIN_PRICE_TICK * SLIPPAGE_TICKS)
                        self.close_long_position(i, exit_price, 'death_cross_long', f'CCI死叉{current["cci"]:.2f}')
                        exit_triggered = True

            # ========================================
            # 空头平仓逻辑（快进快出）
            # ========================================
            if self.position_short > 0:
                short_params = self.params['short']
                exit_triggered = False

                # 50%分批止盈（更敏感！）
                if not self.has_taken_half_profit_short:
                    profit_atr = (self.avg_price_short - current['close']) / self.entry_atr_short

                    # 更敏感的止盈触发条件
                    if current['cci'] < short_params['cci_first_target'] or profit_atr > short_params['profit_atr_multiple']:
                        exit_qty = int(self.position_short * short_params['half_exit_ratio'])
                        exit_qty = max(1, exit_qty)
                        exit_price = current['close'] + (MIN_PRICE_TICK * SLIPPAGE_TICKS)
                        pnl = (self.avg_price_short - exit_price) * exit_qty * MULTIPLIER
                        commission = exit_price * exit_qty * MULTIPLIER * COMMISSION_RATE
                        net_pnl = pnl - commission

                        margin_returned = self.calc_margin_required(self.avg_price_short, exit_qty)
                        self.balance += (margin_returned + net_pnl)

                        self.trades.append({
                            'direction': 'short',
                            'entry_date': self.entry_date_short,
                            'exit_date': current.name,
                            'entry_price': self.avg_price_short,
                            'exit_price': exit_price,
                            'pnl': net_pnl,
                            'type': 'half_profit_short',
                            'exit_reason': f'50%止盈CCI:{current["cci"]:.2f}',
                            'hold_days': (current.name - self.entry_date_short).days,
                            'qty': exit_qty,
                        })

                        print(f"空50%止盈: {current.name.date()} 盈利:{net_pnl:.0f}")
                        self.position_short -= exit_qty
                        self.has_taken_half_profit_short = True

                        if self.position_short == 0:
                            exit_triggered = True

                # 剩余仓位止损（更紧的止损）
                if not exit_triggered and self.position_short > 0:
                    if next_day['high'] >= self.avg_price_short:
                        stop_price = min(next_day['open'], self.avg_price_short)
                        stop_price += (MIN_PRICE_TICK * SLIPPAGE_TICKS)
                        self.close_short_position(i + 1, stop_price, 'atr_stop_short', '保本止损')
                        exit_triggered = True

                # CCI超卖平仓（防止逼空）
                if not exit_triggered and self.position_short > 0:
                    if current['cci'] < CCI_OVERSOLD_SHORT:
                        exit_price = current['close'] + (MIN_PRICE_TICK * SLIPPAGE_TICKS)
                        self.close_short_position(i, exit_price, 'cci_oversold_short', f'CCI超卖{current["cci"]:.2f}')
                        exit_triggered = True

                # CCI金叉平仓（快进快出）
                if not exit_triggered and self.position_short > 0 and i > 0:
                    cci_prev = self.df.iloc[i - 1]['cci']
                    cci_ma_prev = self.df.iloc[i - 1]['cci_ma']
                    if cci_prev <= cci_ma_prev and current['cci'] > current['cci_ma']:
                        exit_price = current['close'] + (MIN_PRICE_TICK * SLIPPAGE_TICKS)
                        self.close_short_position(i, exit_price, 'golden_cross_short', f'CCI金叉{current["cci"]:.2f}')
                        exit_triggered = True

    def get_results(self):
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) == 0:
            return None

        final_balance = self.balance
        total_return = (final_balance - self.initial_capital) / self.initial_capital * 100

        equity_series = pd.Series(self.equity)
        max_equity = equity_series.cummax()
        drawdown = (equity_series - max_equity) / max_equity * 100
        max_dd = drawdown.min()

        long_trades = trades_df[trades_df['direction'] == 'long']
        short_trades = trades_df[trades_df['direction'] == 'short']

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
            'final_balance': final_balance,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'long_win_rate': len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0,
            'short_win_rate': len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0,
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


def calc_stc(df):
    ema_fast = df['close'].ewm(span=23, adjust=False).mean()
    ema_slow = df['close'].ewm(span=50, adjust=False).mean()
    macd = ema_fast - ema_slow

    lowest_macd = macd.rolling(window=10).min()
    highest_macd = macd.rolling(window=10).max()
    range_macd = highest_macd - lowest_macd

    stc = pd.Series(index=df.index, dtype=float)
    stc.iloc[:10] = 0

    for i in range(10, len(macd)):
        if range_macd.iloc[i] != 0:
            stc.iloc[i] = ((macd.iloc[i] - lowest_macd.iloc[i]) / range_macd.iloc[i] - 0.5) * 200
        else:
            stc.iloc[i] = stc.iloc[i-1] if i > 10 else 0

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
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calc_indicators(df, params):
    print("计算指标...")

    df['stc'] = calc_stc(df)
    df['stc_prev'] = df['stc'].shift(1)
    df['stc_slope'] = df['stc'] - df['stc_prev']

    tp = (df['high'] + df['low'] + df['close']) / 3
    sma = tp.rolling(window=CCI_LENGTH).mean()
    mad = tp.rolling(window=CCI_LENGTH).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
    df['cci'] = (tp - sma) / (0.015 * mad)
    df['cci_ma'] = df['cci'].rolling(window=CCI_MA_LENGTH).mean()
    df['cci_prev'] = df['cci'].shift(1)

    df['atr'] = calc_atr(df, ATR_LENGTH)

    df['vol_ma'] = df['volume'].rolling(window=VOL_MA_PERIOD).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['is_bullish_candle'] = df['close'] > df['open']

    return df


def print_results(result):
    print("\n" + "=" * 100)
    print("纯碱SA0 - S.C.C.系统 多空不对称版".center(100))
    print("=" * 100)
    print("做多=结构猎人，做空=重力加速器".center(100))
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

    print("\n" + "=" * 100)
    print("多空不对称对比")
    print("=" * 100)
    print(f"  {'方向':<8} {'交易次数':>10} {'胜率':>10} {'总盈亏':>15}")
    print("-" * 80)
    print(f"  {'多头':<8} {result['long_trades']:>10}笔 {result['long_win_rate']:>9.2f}% {result['long_pnl']:>14.0f}元")
    print(f"  {'空头':<8} {result['short_trades']:>10}笔 {result['short_win_rate']:>9.2f}% {result['short_pnl']:>14.0f}元")

    print("\n" + "=" * 100)
    print("关键不对称参数")
    print("=" * 100)
    print("  做多（结构猎人）：")
    print("    STC: 从底部抬头(>-20) + 斜率>2")
    print("    量能: 必须放量(>0.9) + 必须阳线")
    print("    CCI: 突破-80 + 2日确认")
    print("    止盈: CCI>100 或 浮盈>1.5ATR")
    print("\n  做空（重力加速器）：")
    print("    STC: 跌破-80 + 斜率<-2（快速掉头）")
    print("    量能: 不需要放量(0.5<vol<3.0)")
    print("    CCI: 跌破-50/-100/0（主跌浪）")
    print("    止盈: CCI<-150 或 浮盈>1.0ATR（更敏感！）")

    print("\n" + "=" * 100)


def main():
    print("=" * 100)
    print("纯碱SA0 - S.C.C.系统 多空不对称版".center(100))
    print("做多=结构猎人，做空=重力加速器".center(100))
    print("=" * 100)
    print(f"回测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    print(f"\n[1/3] 获取数据...")
    df = get_data()

    if df is None:
        print("[FAILED] 数据获取失败")
        return

    print(f"数据量: {len(df)}条")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")

    print(f"\n[2/3] 计算指标...")
    df = calc_indicators(df, ASYMMETRIC_PARAMS)

    print(f"\n[3/3] 运行多空不对称回测...")
    backtest = AsymmetricBacktest(df, ASYMMETRIC_PARAMS, INITIAL_CAPITAL)
    backtest.run_backtest()

    print(f"\n输出结果...")
    result = backtest.get_results()

    if result:
        print_results(result)

        output_file = r'D:\期货数据\铜期货监控\CCI策略系统\纯碱多空不对称详细记录.csv'
        result['trades_df'].to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n详细交易记录已保存至: {output_file}")
    else:
        print("\n[FAILED] 回测失败或无交易")

    print("\n" + "=" * 100)
    print(f"回测完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)


if __name__ == "__main__":
    main()
