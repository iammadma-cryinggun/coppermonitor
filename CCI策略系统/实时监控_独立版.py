"""
CCI策略实时监控系统 - 完全独立版
===================================
所有配置内嵌在代码中，避免文件读取编码问题
"""
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys


# 所有配置参数（内嵌版本）
OPTIMAL_PARAMS = {
    'Silver': {
        'code': 'ag0',
        'cci_length': 15, 'ma_length': 15,
        'cci_oversold': -60, 'cci_overbought': 190,
        'cci_cross_max': 180, 'stc_oversold': -50, 'stc_cross': -100,
        'expected_return': 6711.5, 'max_dd': -40.4, 'win_rate': 51.1,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.35,
    },
    'Copper': {
        'code': 'cu0',
        'cci_length': 20, 'ma_length': 5,
        'cci_oversold': -100, 'cci_overbought': 100,
        'cci_cross_max': -20, 'stc_oversold': -20, 'stc_cross': -30,
        'expected_return': 1296.0, 'max_dd': -50.5, 'win_rate': 48.3,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.15,
    },
    'SodaAsh': {
        'code': 'sa0',
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -40, 'cci_overbought': 190,
        'cci_cross_max': 180, 'stc_oversold': -50, 'stc_cross': -130,
        'expected_return': 738.9, 'max_dd': -54.4, 'win_rate': 45.2,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.15,
    },
    'Nickel': {
        'code': 'ni0',
        'cci_length': 12, 'ma_length': 8,
        'cci_oversold': -40, 'cci_overbought': 190,
        'cci_cross_max': 180, 'stc_oversold': -60, 'stc_cross': -120,
        'expected_return': 224.4, 'max_dd': -26.3, 'win_rate': 46.7,
        'use_paper_factor': True, 'paper_factor_type': 'trend_quality', 'trend_quality_threshold': 0.25,
    },
    'Glass': {
        'code': 'fg0',
        'cci_length': 14, 'ma_length': 8,
        'cci_oversold': -60, 'cci_overbought': 100,
        'cci_cross_max': 150, 'stc_oversold': -40, 'stc_cross': -70,
        'expected_return': 979.6, 'max_dd': -41.0, 'win_rate': 45.4,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.2,
    },
    'Tin': {
        'code': 'sn0',
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -40, 'cci_overbought': 180,
        'cci_cross_max': 150, 'stc_oversold': -40, 'stc_cross': -130,
        'expected_return': 1581.7, 'max_dd': -38.9, 'win_rate': 52.9,
        'use_paper_factor': False,
    },
    'Aluminum': {
        'code': 'al0',
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -40, 'cci_overbought': 180,
        'cci_cross_max': 150, 'stc_oversold': -60, 'stc_cross': -120,
        'expected_return': 121.4, 'max_dd': -33.9, 'win_rate': 54.1,
        'use_paper_factor': True, 'paper_factor_type': 'trend_quality', 'trend_quality_threshold': 0.25,
    },
    'Lead': {
        'code': 'pb0',
        'cci_length': 20, 'ma_length': 5,
        'cci_oversold': -100, 'cci_overbought': 100,
        'cci_cross_max': -20, 'stc_oversold': -20, 'stc_cross': -30,
        'expected_return': 37.8, 'max_dd': -31.7, 'win_rate': 43.6,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.3,
    },
    'Cotton': {
        'code': 'cf0',
        'cci_length': 15, 'ma_length': 10,
        'cci_oversold': -100, 'cci_overbought': 150,
        'cci_cross_max': 120, 'stc_oversold': -50, 'stc_cross': -100,
        'expected_return': 549.8, 'max_dd': -28.5, 'win_rate': 51.7,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.1,
    },
    'Zinc': {
        'code': 'zn0',
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -80, 'cci_overbought': 150,
        'cci_cross_max': 120, 'stc_oversold': -40, 'stc_cross': -80,
        'expected_return': 229.3, 'max_dd': -46.4, 'win_rate': 45.9,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.15,
    },
    'Gold': {
        'code': 'au0',
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -40, 'cci_overbought': 160,
        'cci_cross_max': 150, 'stc_oversold': -60, 'stc_cross': -130,
        'expected_return': 2237.97, 'max_dd': -87.72, 'win_rate': 55.1,
        'use_paper_factor': False,
    },
    'Zinc': {
        'code': 'zn0',
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -80, 'cci_overbought': 150,
        'cci_cross_max': 120, 'stc_oversold': -40, 'stc_cross': -80,
        'expected_return': 229.3, 'max_dd': -46.4, 'win_rate': 45.9,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.15,
    },
    'Gold': {
        'code': 'au0',
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -40, 'cci_overbought': 160,
        'cci_cross_max': 150, 'stc_oversold': -60, 'stc_cross': -130,
        'expected_return': 2237.97, 'max_dd': -87.72, 'win_rate': 55.1,
        'use_paper_factor': False,
    },
    'Sugar': {
        'code': 'sr0',
        'cci_length': 15, 'ma_length': 5,
        'cci_oversold': -60, 'cci_overbought': 100,
        'cci_cross_max': 120, 'stc_oversold': -40, 'stc_cross': -60,
        'expected_return': 47.4, 'max_dd': -41.6, 'win_rate': 45.9,
        'use_paper_factor': False,
    },
}


def get_akshare_data(code, days=200):
    """Get real-time data"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)
        df = df.iloc[:, :5]
        df.columns = ['date', 'open', 'high', 'low', 'close']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.tail(days)
        df = df.dropna()
        return df
    except Exception as e:
        print(f"[ERROR] Get data failed {code}: {e}")
        return None


def calculate_stc(close_prices, fast_period=23, slow_period=50, cycle_period=10):
    """Calculate STC indicator"""
    ema_fast = close_prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow

    lowest_macd = macd.rolling(window=cycle_period).min()
    highest_macd = macd.rolling(window=cycle_period).max()
    range_macd = highest_macd - lowest_macd

    stc = pd.Series(index=close_prices.index, dtype=float)
    stc.iloc[:cycle_period] = 0
    for i in range(cycle_period, len(macd)):
        if range_macd.iloc[i] != 0:
            stc.iloc[i] = ((macd.iloc[i] - lowest_macd.iloc[i]) / range_macd.iloc[i] - 0.5) * 200
        else:
            stc.iloc[i] = stc.iloc[i-1] if i > cycle_period else 0

    # Normalize
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


def calculate_indicators(df, params):
    """Calculate all indicators"""
    try:
        # Calculate CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        cci_length = params['cci_length']
        ma_length = params['ma_length']

        sma = tp.rolling(window=cci_length).mean()
        mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - sma) / (0.015 * mad)
        df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()

        # Calculate STC
        df['stc'] = calculate_stc(df['close'])

        # Calculate paper factors (if enabled)
        if params.get('use_paper_factor', False):
            factor_type = params.get('paper_factor_type', '')

            if factor_type == 'gap':
                # Overnight gap factor
                gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
                high_max = df['high'].rolling(10).max()
                low_min = df['low'].rolling(10).min()
                true_range = high_max - low_min
                df['gap_factor'] = gap / (true_range / df['close'])
                df['factor_pass'] = df['gap_factor'] >= params['gap_threshold']

            elif factor_type == 'trend_quality':
                # Trend quality factor
                direction = (df['close'].diff() > 0).astype(int)
                direction_consistency = direction.rolling(10).mean()

                atr = (df['high'] - df['low']).rolling(14).mean()
                atr_ma = atr.rolling(60).mean()
                atr_ratio = atr / atr_ma
                volatility_quality = 1 / (atr_ratio + 0.1)

                vol_ratio = df['volume'] / df['volume'].rolling(5).mean()
                liquidity_quality = np.minimum(vol_ratio, 2) / 2

                df['trend_quality'] = direction_consistency * volatility_quality * liquidity_quality
                df['factor_pass'] = df['trend_quality'] >= params['trend_quality_threshold']

        return df
    except Exception as e:
        print(f"[ERROR] Calculate indicators failed: {e}")
        return None


def check_signals(data, params):
    """Check trading signals"""
    if data is None or len(data) < 2:
        return None

    current = data.iloc[-1]
    previous = data.iloc[-2]

    cci = current['cci']
    cci_ma = current['cci_ma']
    stc = current['stc']

    prev_cci = previous['cci']
    prev_cci_ma = previous['cci_ma']

    try:
        date_str = current.name.strftime('%Y-%m-%d') if hasattr(current.name, 'strftime') else str(current.name)
    except:
        date_str = str(current.name)

    signals = {
        'cci': round(cci, 2),
        'cci_ma': round(cci_ma, 2),
        'stc': round(stc, 2),
        'close': current['close'],
        'date': date_str,
        'oversold_signal': False,
        'golden_cross_signal': False,
        'overbought_warning': False,
        'death_cross_warning': False,
        'factor_enabled': params.get('use_paper_factor', False),
        'factor_pass': True,
    }

    # Check oversold signal
    if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
        signals['oversold_signal'] = True

    # Check golden cross signal
    if (prev_cci <= prev_cci_ma and cci > cci_ma and
        cci <= params['cci_cross_max'] and stc >= params['stc_cross']):
        signals['golden_cross_signal'] = True

    # Check overbought warning
    if cci > params['cci_overbought']:
        signals['overbought_warning'] = True

    # Check death cross warning
    if prev_cci >= prev_cci_ma and cci < cci_ma:
        signals['death_cross_warning'] = True

    # Paper factor filter
    if params.get('use_paper_factor', False) and (signals['oversold_signal'] or signals['golden_cross_signal']):
        if 'factor_pass' in data.columns:
            signals['factor_pass'] = current['factor_pass']
            if not signals['factor_pass']:
                # Factor not passed, cancel signals
                signals['oversold_signal'] = False
                signals['golden_cross_signal'] = False

    return signals


def format_signal_output(symbol, params, signals):
    """Format signal output"""
    if signals is None:
        return f"{symbol:<10} [No data]"

    output = []
    output.append(f"\n{'='*100}")
    output.append(f"{symbol} - {signals['date']}".center(100))
    output.append(f"{'='*100}")

    # Basic info
    output.append(f"\nCurrent Price: {signals['close']:.2f}")
    output.append(f"CCI: {signals['cci']:>8.2f}  |  CCI_MA: {signals['cci_ma']:>8.2f}  |  STC: {signals['stc']:>8.2f}")

    # Thresholds
    output.append(f"\nThresholds:")
    output.append(f"  Oversold: {params['cci_oversold']}  |  Overbought: {params['cci_overbought']}  |  Cross Max: {params['cci_cross_max']}")
    output.append(f"  STC Oversold: {params['stc_oversold']}  |  STC Cross: {params['stc_cross']}")

    # Paper factor status
    if signals['factor_enabled']:
        factor_type = params.get('paper_factor_type', 'None')
        output.append(f"\nPaper Factor: {factor_type}")
        factor_status = "[PASS]" if signals['factor_pass'] else "[FILTERED]"
        output.append(f"  Status: {factor_status}")

    # Trading signals
    output.append(f"\nTrading Signals:")

    has_signal = False

    if signals['oversold_signal']:
        output.append(f"  [BUY] CCI Oversold Signal! CCI={signals['cci']:.2f} < {params['cci_oversold']}, STC={signals['stc']:.2f}")
        has_signal = True

    if signals['golden_cross_signal']:
        output.append(f"  [BUY] CCI Golden Cross Signal!")
        has_signal = True

    if signals['overbought_warning']:
        output.append(f"  [WARNING] CCI Overbought! CCI={signals['cci']:.2f} > {params['cci_overbought']}")
        has_signal = True

    if signals['death_cross_warning']:
        output.append(f"  [WARNING] CCI Death Cross!")
        has_signal = True

    if not has_signal:
        output.append(f"  [WAIT] No clear trading signal")

    # Backtest performance
    output.append(f"\nBacktest Performance:")
    output.append(f"  Expected Return: {params['expected_return']:+.2f}%  |  Max DD: {params['max_dd']:.2f}%")
    output.append(f"  Win Rate: {params['win_rate']:.1f}%")

    return '\n'.join(output)


def monitor_all_symbols(symbols_list=None):
    """Monitor all symbols"""
    print("\n" + "="*120)
    print("CCI Strategy Real-time Monitoring System".center(120))
    print("="*120)
    print(f"Monitor Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120))
    print("="*120)

    if symbols_list is None:
        symbols_list = ['Silver', 'Copper', 'SodaAsh', 'Nickel', 'Glass', 'Tin', 'Zinc', 'Gold']

    results = []

    for i, symbol in enumerate(symbols_list, 1):
        print(f"\n[{i}/{len(symbols_list)}] Getting {symbol} data...", end=" ")
        sys.stdout.flush()

        params = OPTIMAL_PARAMS[symbol]
        code = params['code']

        # Get real-time data
        df = get_akshare_data(code, days=200)

        if df is None:
            print("[Failed]")
            continue

        print(f"[Success] {len(df)} bars")

        # Calculate indicators
        data = calculate_indicators(df, params)

        # Check signals
        signals = check_signals(data, params)

        if signals:
            results.append((symbol, params, signals))

            # Output details
            print(format_signal_output(symbol, params, signals))

        # Avoid rate limiting
        time.sleep(1)

    # Summary
    print("\n" + "="*120)
    print("Signal Summary".center(120))
    print("="*120)

    buy_signals = []
    wait_list = []

    for symbol, params, signals in results:
        if signals['oversold_signal'] or signals['golden_cross_signal']:
            if signals['factor_pass']:
                buy_signals.append((symbol, signals))
            else:
                wait_list.append((symbol, "Filtered"))
        elif signals['overbought_warning'] or signals['death_cross_warning']:
            wait_list.append((symbol, "Warning"))
        else:
            wait_list.append((symbol, "Wait"))

    # Display buy signals
    if buy_signals:
        print(f"\n[BUY SIGNALS] {len(buy_signals)} symbols:")
        for symbol, sig in buy_signals:
            signal_type = []
            if sig['oversold_signal']:
                signal_type.append("Oversold")
            if sig['golden_cross_signal']:
                signal_type.append("Golden Cross")
            print(f"  {symbol:<20} {', '.join(signal_type)}")

    # Display wait/filtered
    if wait_list:
        print(f"\n[WAIT/FILTERED] {len(wait_list)} symbols:")
        for symbol, reason in wait_list:
            print(f"  {symbol:<20} {reason}")

    print("\n" + "="*120)
    print(f"Monitor Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120))
    print("="*120)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CCI Strategy Monitoring System')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'all'],
                       help='Monitor mode: quick(top6), all')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Custom symbols')

    args = parser.parse_args()

    if args.mode == 'quick':
        monitor_all_symbols()
    elif args.mode == 'all':
        monitor_all_symbols()
    elif args.mode == 'custom' and args.symbols:
        # Map English names to Chinese
        name_map = {
            'Silver': 'Silver', 'Copper': 'Copper', 'SodaAsh': 'SodaAsh',
            'Nickel': 'Nickel', 'Glass': 'Glass', 'Tin': 'Tin',
            'Aluminum': 'Aluminum', 'Lead': 'Lead', 'Cotton': 'Cotton',
            'Zinc': 'Zinc', 'Gold': 'Gold', 'Sugar': 'Sugar'
        }
        symbols_list = [name_map.get(s, s) for s in args.symbols]
        monitor_all_symbols(symbols_list)
    else:
        monitor_all_symbols()
