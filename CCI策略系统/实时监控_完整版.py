"""
CCI策略实时监控系统 - 完整版（2026-02-12）
===================================
更新内容：
1. 添加完整的止盈止损信息
2. 使用中文名称
3. 确认所有12个品种
4. 详细的操作建议
"""
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sys

# 所有配置参数（内嵌版本）
OPTIMAL_PARAMS = {
    '白银': {
        'code': 'ag0', 'multiplier': 15,
        'cci_length': 15, 'ma_length': 15,
        'cci_oversold': -60, 'cci_overbought': 190,
        'cci_cross_max': 180, 'stc_oversold': -50, 'stc_cross': -100,
        'expected_return': 6711.5, 'max_dd': -40.4, 'win_rate': 51.1,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.35,
        'factor_effect': '极其显著', 'improvement': '+1700.4% (+34%)',
    },
    '铜': {
        'code': 'cu0', 'multiplier': 5,
        'cci_length': 20, 'ma_length': 5,
        'cci_oversold': -100, 'cci_overbought': 100,
        'cci_cross_max': -20, 'stc_oversold': -20, 'stc_cross': -30,
        'expected_return': 1296.0, 'max_dd': -50.5, 'win_rate': 48.3,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.15,
        'factor_effect': '显著', 'improvement': '+572.6% (+79%)',
    },
    '纯碱': {
        'code': 'sa0', 'multiplier': 20,
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -40, 'cci_overbought': 190,
        'cci_cross_max': 180, 'stc_oversold': -50, 'stc_cross': -130,
        'expected_return': 738.9, 'max_dd': -54.4, 'win_rate': 45.2,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.15,
        'factor_effect': '小幅', 'improvement': '+32.5% (+5%)',
    },
    '镍': {
        'code': 'ni0', 'multiplier': 1,
        'cci_length': 12, 'ma_length': 8,
        'cci_oversold': -40, 'cci_overbought': 190,
        'cci_cross_max': 180, 'stc_oversold': -60, 'stc_cross': -120,
        'expected_return': 224.4, 'max_dd': -26.3, 'win_rate': 46.7,
        'use_paper_factor': True, 'paper_factor_type': 'trend_quality', 'trend_quality_threshold': 0.25,
        'factor_effect': '中等', 'improvement': '+40.6% (+22%)',
    },
    '玻璃': {
        'code': 'fg0', 'multiplier': 20,
        'cci_length': 14, 'ma_length': 8,
        'cci_oversold': -60, 'cci_overbought': 100,
        'cci_cross_max': 150, 'stc_oversold': -40, 'stc_cross': -70,
        'expected_return': 979.6, 'max_dd': -41.0, 'win_rate': 45.4,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.2,
        'factor_effect': '显著', 'improvement': '+479.9% (+96%)',
    },
    '锡': {
        'code': 'sn0', 'multiplier': 1,
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -40, 'cci_overbought': 180,
        'cci_cross_max': 150, 'stc_oversold': -40, 'stc_cross': -130,
        'expected_return': 1581.7, 'max_dd': -38.9, 'win_rate': 52.9,
        'use_paper_factor': False,
        'factor_effect': '无效', 'improvement': '+0%',
    },
    '铝': {
        'code': 'al0', 'multiplier': 5,
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -40, 'cci_overbought': 180,
        'cci_cross_max': 150, 'stc_oversold': -60, 'stc_cross': -120,
        'expected_return': 121.4, 'max_dd': -33.9, 'win_rate': 54.1,
        'use_paper_factor': True, 'paper_factor_type': 'trend_quality', 'trend_quality_threshold': 0.25,
        'factor_effect': '中等', 'improvement': '+33.7% (+38%)',
    },
    '铅': {
        'code': 'pb0', 'multiplier': 5,
        'cci_length': 20, 'ma_length': 5,
        'cci_oversold': -100, 'cci_overbought': 100,
        'cci_cross_max': -20, 'stc_oversold': -20, 'stc_cross': -30,
        'expected_return': 37.8, 'max_dd': -31.7, 'win_rate': 43.6,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.3,
        'factor_effect': '小幅', 'improvement': '+8.8% (+30%)',
    },
    '棉花': {
        'code': 'cf0', 'multiplier': 5,
        'cci_length': 15, 'ma_length': 10,
        'cci_oversold': -100, 'cci_overbought': 150,
        'cci_cross_max': 120, 'stc_oversold': -50, 'stc_cross': -100,
        'expected_return': 549.8, 'max_dd': -28.5, 'win_rate': 51.7,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.1,
        'factor_effect': '显著', 'improvement': '+50.1% (+10%)',
    },
    '锌': {
        'code': 'zn0', 'multiplier': 5,
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -80, 'cci_overbought': 150,
        'cci_cross_max': 120, 'stc_oversold': -40, 'stc_cross': -80,
        'expected_return': 229.3, 'max_dd': -46.4, 'win_rate': 45.9,
        'use_paper_factor': True, 'paper_factor_type': 'gap', 'gap_threshold': -0.15,
        'factor_effect': '中等', 'improvement': '+47.3% (+26%)',
    },
    '黄金': {
        'code': 'au0', 'multiplier': 1000,
        'cci_length': 12, 'ma_length': 5,
        'cci_oversold': -40, 'cci_overbought': 160,
        'cci_cross_max': 150, 'stc_oversold': -60, 'stc_cross': -130,
        'expected_return': 2237.97, 'max_dd': -87.72, 'win_rate': 55.1,
        'use_paper_factor': False,
        'factor_effect': '无效', 'improvement': '+0%',
    },
    '糖': {
        'code': 'sr0', 'multiplier': 10,
        'cci_length': 15, 'ma_length': 5,
        'cci_oversold': -60, 'cci_overbought': 100,
        'cci_cross_max': 120, 'stc_oversold': -40, 'stc_cross': -60,
        'expected_return': 47.4, 'max_dd': -41.6, 'win_rate': 45.9,
        'use_paper_factor': True, 'paper_factor_type': 'gap_acceptance', 'gap_acceptance_threshold': 0.001,
        'factor_effect': '小幅', 'improvement': '+47.3%',
        'warning': '绝对收益太低，不推荐实盘',
    },
}

# 中文名称映射
NAME_MAP = {
    '白银': 'Silver', '铜': 'Copper', '纯碱': 'SodaAsh',
    '镍': 'Nickel', '玻璃': 'Glass', '锡': 'Tin',
    '铝': 'Aluminum', '铅': 'Lead', '棉花': 'Cotton',
    '锌': 'Zinc', '黄金': 'Gold', '糖': 'Sugar',
}

# 反向映射（英文到中文）
EN_TO_CN = {v: k for k, v in NAME_MAP.items()}


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
    close = current['close']

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
        'close': close,
        'date': date_str,
        'oversold_signal': False,
        'golden_cross_signal': False,
        'overbought_warning': False,
        'death_cross_warning': False,
        'factor_enabled': params.get('use_paper_factor', False),
        'factor_type': params.get('paper_factor_type', None),
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


def calculate_position_size(balance, entry_price, multiplier, max_leverage=2):
    """计算仓位大小"""
    risk_per_trade = balance * 0.9  # 每次交易使用90%资金
    max_value = risk_per_trade * max_leverage
    qty = int(max_value / (entry_price * multiplier))
    qty = max(1, qty)
    return qty


def format_signal_output(symbol_cn, params, signals):
    """格式化信号输出（包含完整止盈止损信息）"""
    if signals is None:
        return f"{symbol_cn:<10} [No data]"

    output = []
    output.append(f"\n{'='*100}")
    output.append(f"{symbol_cn} - {signals['date']}".center(100))
    output.append(f"{'='*100}")

    # 基本信息
    output.append(f"\n当前价格: {signals['close']:.2f}")
    output.append(f"CCI: {signals['cci']:>8.2f} | CCI_MA: {signals['cci_ma']:>8.2f} | STC: {signals['stc']:>8.2f}")

    # 阈值参数
    output.append(f"\n阈值参数:")
    output.append(f"  超卖: {params['cci_oversold']}  |  超买: {params['cci_overbought']}  |  金叉上限: {params['cci_cross_max']}")
    output.append(f"  STC超卖: {params['stc_oversold']}  |  STC金叉: {params['stc_cross']}")

    # 论文因子状态
    if signals['factor_enabled']:
        factor_type = signals['factor_type']
        output.append(f"\n论文因子: {factor_type}")
        if factor_type == 'gap':
            output.append(f"  隔夜缺口阈值: {params['gap_threshold']}")
        elif factor_type == 'trend_quality':
            output.append(f"  趋势质量阈值: {params['trend_quality_threshold']}")

        status = "[通过]" if signals['factor_pass'] else "[未通过]（信号被过滤）"
        output.append(f"  状态: {status}")

    # 交易信号
    output.append(f"\n交易信号:")

    has_buy_signal = False
    position_info = []

    if signals['oversold_signal']:
        output.append(f"  [买入] CCI超卖信号! CCI={signals['cci']:.2f} < {params['cci_oversold']}, STC={signals['stc']:.2f}")
        has_buy_signal = True
        position_info.append({
            'type': 'CCI超卖',
            'entry_reason': f"CCI={signals['cci']:.2f} < {params['cci_oversold']}"
        })

    if signals['golden_cross_signal']:
        output.append(f"  [买入] CCI金叉信号! CCI上穿CCI_MA")
        has_buy_signal = True
        position_info.append({
            'type': 'CCI金叉',
            'entry_reason': "CCI上穿CCI_MA"
        })

    if not has_buy_signal:
        # 无买入信号
        output.append(f"  [观望] 无明确买入信号")

    # 飀查其他信号
    if signals['overbought_warning']:
        output.append(f"  [注意] CCI超买警告! CCI={signals['cci']:.2f} > {params['cci_overbought']}")
        position_info.append({'type': 'CCI超买', 'reason': f"CCI={signals['cci']:.2f}超买"})

    if signals['death_cross_warning']:
        output.append(f"  [注意] CCI死叉警告! CCI下穿CCI_MA")
        position_info.append({'type': 'CCI死叉', 'reason': "CCI下穿CCI_MA"})

    # 如果有买入信号，计算详细的止盈止损信息
    if has_buy_signal:
        close = signals['close']
        multiplier = params['multiplier']

        # 假设使用10万资金
        account_balance = 100000

        # 计算仓位
        qty = calculate_position_size(account_balance, close, multiplier)

        position_value = close * qty * multiplier
        actual_capital = account_balance * 0.1  # 实际使用10%保证金

        # 止盈止损价格
        stop_loss_price = close * 0.96
        take_profit_price = close * 1.20

        # 每点价值
        point_value = multiplier

        output.append(f"\n【仓位与风控信息】")
        output.append(f"  假设资金: {account_balance:,.0f} 元")
        output.append(f"  合约乘数: {multiplier}")
        output.append(f"  开仓方向: 做多")
        output.append(f"  开仓价格: {close:.2f}")
        output.append(f"  建议仓位: {qty} 手")
        output.append(f"  占用资金: {position_value:,.2f} 元 ({position_value/account_balance*100:.1f}%)")
        output.append(f"")
        output.append(f"  止损价格: {stop_loss_price:.2f} (亏损 {(close-stop_loss_price)/close*100:.2f}%)")
        output.append(f"  止盈价格: {take_profit_price:.2f} (盈利 {(take_profit_price-close)/close*100:.2f}%)")
        output.append(f"")
        output.append(f"  最大止损: {position_value * 0.04:,.2f} 元 (4%)")
        output.append(f" 最大止盈: {position_value * 0.20:,.2f} 元 (20%)")
        output.append(f"  风险收益比: 1:5 (止损:止盈 = 4%:20%)")

    # 历史表现
    output.append(f"\n【历史回测表现】")
    output.append(f"  预期收益: {params['expected_return']:+.2f}%")
    output.append(f"  最大回撤: {params['max_dd']:.2f}%")
    output.append(f"  胜率: {params['win_rate']:.1f}%")

    if 'factor_effect' in params:
        output.append(f"  因子效果: {params['factor_effect']}")

    # 操作建议
    output.append(f"\n【操作建议】")

    if has_buy_signal:
        output.append(f"  [强烈建议] 开多仓")
        output.append(f"  仓位: {qty} 手 (建议用10%资金开1手)")
        output.append(f"  止损: {stop_loss_price:.2f} (收盘价跌破此价格止损)")
        output.append(f"  止盈: {take_profit_price:.2f} (收盘价突破此价格止盈)")
        output.append(f"  风控: 严格执行止损止盈，不可抗单！")

        # 根据因子效果给出建议强度
        if params.get('factor_effect', '') == '极其显著':
            output.append(f"  信心: 极高 (因子提升{params['improvement']})")
        elif params.get('factor_effect', '') == '显著':
            output.append(f"  信心: 高 (因子提升{params['improvement']})")
        elif params.get('factor_effect', '') == '中等':
            output.append(f"  信心: 中等 (因子提升{params['improvement']})")
        else:
            output.append(f"  信心: 一般")

    elif signals['overbought_warning']:
        output.append(f"  [建议] 暂不追高")
        output.append(f"  CCI已进入超买区域({signals['cci']:.2f})")
        output.append(f"  建议等待CCI回调后再入场")
        output.append(f"  或考虑逐步减仓")

    elif signals['death_cross_warning']:
        output.append(f"  [建议] 观望或平仓")
        output.append(f"  CCI已形成死叉")
        output.append(f"  建议持币观望或平仓")

    else:
        output.append(f"  [建议] 观望")
        output.append(f"  无明确交易信号")
        output.append(f"  建议耐心等待")

    return '\n'.join(output)


def monitor_all_symbols(symbols_list=None, show_positions=True):
    """监控所有品种"""
    print("\n" + "="*120)
    print("CCI策略实时监控系统 - 完整版".center(120))
    print("="*120)
    print(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120))
    print("="*120)

    # 使用所有12个品种
    if symbols_list is None:
        symbols_list = ['白银', '铜', '纯碱', '镍', '玻璃', '锡', '铝', '铅', '棉花', '锌', '黄金', '糖']

    results = []
    buy_signals = []

    for i, symbol in enumerate(symbols_list, 1):
        params = OPTIMAL_PARAMS.get(symbol)
        if not params:
            continue

        code = params['code']
        symbol_cn = symbol  # 中文名

        print(f"\n[{i}/{len(symbols_list)}] 获取 {symbol_cn} 数据...", end=" ")
        sys.stdout.flush()

        # 获取实时数据
        df = get_akshare_data(code, days=200)

        if df is None:
            print("[失败]")
            continue

        print(f"[成功] {len(df)} 条数据")

        # 计算指标
        data = calculate_indicators(df, params)

        # 检查信号
        signals = check_signals(data, params)

        if signals:
            results.append((symbol_cn, params, signals))

            # 输出详细信息
            print(format_signal_output(symbol_cn, params, signals))

            # 如果有买入信号，添加到列表
            if signals['oversold_signal'] or signals['golden_cross_signal']:
                buy_signals.append({
                    'symbol': symbol_cn,
                    'signals': signals,
                    'params': params
                })

        # 避免请求过快
        time.sleep(1)

    # 汇总买入信号
    if buy_signals and show_positions:
        print("\n" + "="*120)
        print("买入信号汇总".center(120))
        print("="*120)

        print(f"\n共发现 {len(buy_signals)} 个买入信号\n")

        for i, signal_info in enumerate(buy_signals, 1):
            symbol = signal_info['symbol']
            signals = signal_info['signals']
            params = signal_info['params']

            print(f"{i}. {symbol}")
            print(f"   信号类型: ", end="")

            signal_types = []
            if signals['oversold_signal']:
                signal_types.append(f"CCI超卖({signals['cci']:.2f})")
            if signals['golden_cross_signal']:
                signal_types.append("CCI金叉")

            print(", ".join(signal_types))

            # 显示因子状态
            if signals['factor_enabled']:
                factor_status = "通过" if signals['factor_pass'] else "被过滤"
                print(f"   论文因子: {factor_status}")

            # 显示预期效果
            print(f"   预期收益: {params['expected_return']:+.2f}%")
            print(f"   最大回撤: {params['max_dd']:.2f}%")
            print(f"   胜率: {params['win_rate']:.1f}%")

            # 显示因子效果
            if 'improvement' in params:
                print(f"   因子提升: {params['improvement']}")

            print()

        print("="*120)

    # 最终建议
        print("\n" + "="*120)
        print("综合操作建议".center(120))
        print("="*120)

        # 按优先级排序
        priority_signals = sorted(buy_signals,
                                   key=lambda x: (
                                       x['params'].get('factor_effect', ''),
                                       x['params']['expected_return']
                                   ),
                                   reverse=True)

        if priority_signals:
            print(f"\n【优先级排序】（因子效果 > 预期收益）\n")

            for i, signal_info in enumerate(priority_signals[:5], 1):
                symbol = signal_info['symbol']
                params = signal_info['params']

                priority = "高"
                if params.get('factor_effect') == '极其显著':
                    priority = "极高"
                elif params.get('factor_effect') == '显著':
                    priority = "高"
                elif params.get('factor_effect') == '中等':
                    priority = "中"
                else:
                    priority = "低"

                print(f"{i}. {symbol:<10} (优先级: {priority})")
                print(f"   预期收益: {params['expected_return']:+.2f}%")
                print(f"   最大回撤: {params['max_dd']:.2f}%")
                print()

        print("="*120)
        print(f"监控完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(120))
        print("="*120)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CCI策略实时监控系统 - 完整版')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'all', 'custom'],
                       help='监控模式: quick(前6), all(全部12个), custom(自定义)')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='自定义监控品种，例如: --symbols 白银 铜 纯碱')
    parser.add_argument('--no-position', action='store_true',
                       help='不显示仓位信息')

    args = parser.parse_args()

    if args.mode == 'quick':
        monitor_all_symbols(show_positions=not args.no_position)
    elif args.mode == 'all':
        monitor_all_symbols(show_positions=not args.no_position)
    elif args.mode == 'custom' and args.symbols:
        # 映射中文到英文
        valid_symbols = []
        for s in args.symbols:
            if s in OPTIMAL_PARAMS:
                valid_symbols.append(s)
            else:
                # 尝试反向映射
                if s in NAME_MAP:
                    valid_symbols.append(NAME_MAP[s])

        if not valid_symbols:
            print(f"\n[WARNING] 无效品种: {[s for s in args.symbols if s not in valid_symbols]}")
            print(f"有效品种: {', '.join(OPTIMAL_PARAMS.keys())}")

        if valid_symbols:
            monitor_all_symbols(valid_symbols, show_positions=not args.no_position)
        else:
            print("\n[ERROR] 没有有效品种")
    else:
        # 默认快速监控
        monitor_all_symbols(show_positions=not args.no_position)


if __name__ == "__main__":
    main()
