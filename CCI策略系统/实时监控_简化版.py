"""
CCI策略实时监控系统 - 简化版（2026-02-12）
===================================
功能：实时监控所有品种的CCI指标和交易信号
更新：使用最新论文因子参数，移除所有可能导致编码问题的字符
"""
import pandas as pd
import numpy as np
from datetime import datetime
import time

# 导入完整配置
exec(open('最优参数_完整版_含代码.py', 'r', encoding='utf-8').read())


def get_akshare_data(code, days=200):
    """获取实时行情数据"""
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
    """计算STC指标"""
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

    # 归一化
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
    """计算所有指标"""
    try:
        # 计算CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        cci_length = params['cci_length']
        ma_length = params['ma_length']

        sma = tp.rolling(window=cci_length).mean()
        mad = tp.rolling(window=cci_length).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - sma) / (0.015 * mad)
        df['cci_ma'] = df['cci'].rolling(window=ma_length).mean()

        # 计算STC
        df['stc'] = calculate_stc(df['close'])

        # 计算论文因子（如果启用）
        if params.get('use_paper_factor', False):
            factor_type = params.get('paper_factor_type', '')

            if factor_type == 'gap':
                # 隔夜缺口因子
                gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
                high_max = df['high'].rolling(10).max()
                low_min = df['low'].rolling(10).min()
                true_range = high_max - low_min
                df['gap_factor'] = gap / (true_range / df['close'])
                df['factor_pass'] = df['gap_factor'] >= params['gap_threshold']

            elif factor_type == 'trend_quality':
                # 趋势质量因子
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
    """检查交易信号"""
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

    # 检查超卖开仓信号
    if cci < params['cci_oversold'] and stc >= params['stc_oversold']:
        signals['oversold_signal'] = True

    # 检查金叉开仓信号
    if (prev_cci <= prev_cci_ma and cci > cci_ma and
        cci <= params['cci_cross_max'] and stc >= params['stc_cross']):
        signals['golden_cross_signal'] = True

    # 检查超买警告
    if cci > params['cci_overbought']:
        signals['overbought_warning'] = True

    # 检查死叉警告
    if prev_cci >= prev_cci_ma and cci < cci_ma:
        signals['death_cross_warning'] = True

    # 论文因子过滤
    if params.get('use_paper_factor', False) and (signals['oversold_signal'] or signals['golden_cross_signal']):
        if 'factor_pass' in data.columns:
            signals['factor_pass'] = current['factor_pass']
            if not signals['factor_pass']:
                # 因子未通过，取消信号
                signals['oversold_signal'] = False
                signals['golden_cross_signal'] = False

    return signals


def format_signal_output(symbol, params, signals):
    """格式化信号输出"""
    if signals is None:
        return f"{symbol:<10} [No data]"

    output = []
    output.append(f"\n{'='*100}")
    output.append(f"{symbol} - {signals['date']}".center(100))
    output.append(f"{'='*100}")

    # 基本信息
    output.append(f"\nCurrent Price: {signals['close']:.2f}")
    output.append(f"CCI: {signals['cci']:>8.2f}  |  CCI_MA: {signals['cci_ma']:>8.2f}  |  STC: {signals['stc']:>8.2f}")

    # 阈值参考
    output.append(f"\nThresholds:")
    output.append(f"  Oversold: {params['cci_oversold']}  |  Overbought: {params['cci_overbought']}  |  Cross Max: {params['cci_cross_max']}")
    output.append(f"  STC Oversold: {params['stc_oversold']}  |  STC Cross: {params['stc_cross']}")

    # 论文因子状态
    if signals['factor_enabled']:
        factor_type = params.get('paper_factor_type', 'None')
        output.append(f"\nPaper Factor: {factor_type}")
        factor_status = "[PASS]" if signals['factor_pass'] else "[FILTERED] (signal blocked)"
        output.append(f"  Status: {factor_status}")

    # 交易信号
    output.append(f"\nTrading Signals:")

    has_signal = False

    if signals['oversold_signal']:
        output.append(f"  [BUY] CCI Oversold Signal! CCI={signals['cci']:.2f} < {params['cci_oversold']}, STC={signals['stc']:.2f}")
        has_signal = True

    if signals['golden_cross_signal']:
        output.append(f"  [BUY] CCI Golden Cross Signal! CCI crossed above CCI_MA")
        has_signal = True

    if signals['overbought_warning']:
        output.append(f"  [WARNING] CCI Overbought! CCI={signals['cci']:.2f} > {params['cci_overbought']}")
        has_signal = True

    if signals['death_cross_warning']:
        output.append(f"  [WARNING] CCI Death Cross! CCI crossed below CCI_MA")
        has_signal = True

    if not has_signal:
        output.append(f"  [WAIT] No clear trading signal")

    # 历史表现
    output.append(f"\nBacktest Performance:")
    output.append(f"  Expected Return: {params['expected_return']:+.2f}%  |  Max DD: {params['max_dd']:.2f}%")
    output.append(f"  Win Rate: {params['win_rate']:.1f}%")

    # 因子效果说明
    if 'improvement' in params:
        output.append(f"\nFactor Effect: {params['improvement']}")

    return '\n'.join(output)


def monitor_all_symbols(symbols_list=None):
    """监控所有品种"""
    print("\n" + "="*120)
    print("CCI Strategy Real-time Monitoring System".center(120))
    print("="*120)
    print(f"Monitor Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120))
    print("="*120)

    if symbols_list is None:
        symbols_list = list(OPTIMAL_PARAMS.keys())

    results = []

    for i, symbol in enumerate(symbols_list, 1):
        print(f"\n[{i}/{len(symbols_list)}] Getting {symbol} data...", end=" ")
        sys.stdout.flush()

        params = OPTIMAL_PARAMS[symbol]
        code = CODE_MAP[symbol]

        # 获取实时数据
        df = get_akshare_data(code, days=200)

        if df is None:
            print("[Failed]")
            continue

        print(f"[Success] {len(df)} bars")

        # 计算指标
        data = calculate_indicators(df, params)

        # 检查信号
        signals = check_signals(data, params)

        if signals:
            results.append((symbol, params, signals))

            # 输出详细信息
            print(format_signal_output(symbol, params, signals))

        # 避免请求过快
        time.sleep(1)

    # 汇总信号
    print("\n" + "="*120)
    print("Signal Summary".center(120))
    print("="*120)

    buy_signals = []
    sell_signals = []
    wait_list = []

    for symbol, params, signals in results:
        if signals['oversold_signal'] or signals['golden_cross_signal']:
            if signals['factor_pass']:
                buy_signals.append((symbol, signals))
            else:
                wait_list.append((symbol, "Factor Filtered"))
        elif signals['overbought_warning'] or signals['death_cross_warning']:
            sell_signals.append((symbol, signals))
        else:
            wait_list.append((symbol, "No Signal"))

    # 显示买入信号
    if buy_signals:
        print(f"\n[BUY SIGNALS] {len(buy_signals)} symbols:")
        for symbol, sig in buy_signals:
            signal_type = []
            if sig['oversold_signal']:
                signal_type.append(f"Oversold(CCI={sig['cci']:.2f})")
            if sig['golden_cross_signal']:
                signal_type.append("Golden Cross")
            print(f"  {symbol:<10} {', '.join(signal_type)}")

    # 显示观望/被过滤
    if wait_list:
        print(f"\n[WAIT/FILTERED] {len(wait_list)} symbols:")
        for symbol, sig, reason in wait_list:
            print(f"  {symbol:<10} {reason}")

    print("\n" + "="*120)
    print(f"Monitor Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120))
    print("="*120)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CCI Strategy Monitoring System')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'all', 'custom'],
                       help='Monitor mode: quick(top5), all, custom')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Custom symbols, e.g.: --symbols Silver Copper')

    args = parser.parse_args()

    if args.mode == 'quick':
        monitor_all_symbols()
    elif args.mode == 'all':
        monitor_all_symbols()
    elif args.mode == 'custom' and args.symbols:
        # 验证品种名称
        valid_symbols = [s for s in args.symbols if s in OPTIMAL_PARAMS]
        invalid_symbols = [s for s in args.symbols if s not in OPTIMAL_PARAMS]

        if invalid_symbols:
            print(f"\n[WARNING] Invalid symbols: {', '.join(invalid_symbols)}")
            print(f"Valid symbols: {', '.join(OPTIMAL_PARAMS.keys())}")

        if valid_symbols:
            monitor_all_symbols(valid_symbols)
        else:
            print("\n[ERROR] No valid symbols")
    else:
        # 默认快速监控
        monitor_all_symbols()


if __name__ == "__main__":
    main()
