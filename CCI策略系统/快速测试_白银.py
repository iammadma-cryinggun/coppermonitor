"""
快速测试监控系统 - 只测试白银
"""
import sys
sys.path.append('D:\\期货数据\\铜期货监控\\CCI策略系统')

# 直接导入参数
from 最优参数配置_完整版_含代码 import (
    OPTIMAL_PARAMS,
    CODE_MAP
)

# 导入CCI计算
from cci_calculations import calculate_cci_tv, calculate_stc, f_normalize

import pandas as pd


def get_akshare_data(code, days=200):
    """获取实时行情数据"""
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=code)

        # 处理数据
        df = df.iloc[:, :5]
        df.columns = ['date', 'open', 'high', 'low', 'close']
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.tail(days)
        df = df.dropna()

        return df
    except Exception as e:
        print(f"[ERROR] 获取数据失败 {code}: {e}")
        return None


def calculate_indicators(df, cci_length, ma_length):
    """计算CCI和STC指标"""
    try:
        # 计算CCI
        data = calculate_cci_tv(df, cci_length=cci_length, ma_length=ma_length)

        # 计算STC
        stc_raw = calculate_stc(data, length=10, fast=23, slow=50, aaa=0.5)
        data['stc'] = f_normalize(stc_raw, 20, 80)

        return data
    except Exception as e:
        print(f"[ERROR] 计算指标失败: {e}")
        return None


def check_signals(data, params):
    """检查交易信号"""
    if data is None or len(data) < 2:
        return None

    # 最新数据
    current = data.iloc[-1]
    previous = data.iloc[-2]

    cci = current['cci']
    cci_ma = current['cci_ma']
    stc = current['stc']

    prev_cci = previous['cci']
    prev_cci_ma = previous['cci_ma']

    # 获取日期
    try:
        date_str = current.name.strftime('%Y-%m-%d') if hasattr(current.name, 'strftime') else str(current.name)
    except:
        date_str = data.index[-1].strftime('%Y-%m-%d') if hasattr(data.index[-1], 'strftime') else str(data.index[-1])

    signals = {
        'cci': round(cci, 2),
        'cci_ma': round(cci_ma, 2),
        'stc': round(stc, 2),
        'close': current['close'],
        'date': date_str,
        'oversold_signal': False,
        'golden_cross_signal': False,
        'overbought_warning': False,
        'death_cross_warning': False
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

    return signals


def format_signal_output(symbol, params, signals):
    """格式化信号输出"""
    if signals is None:
        return f"{symbol:<10} [no data]"

    output = []
    output.append(f"\n{'='*100}")
    output.append(f"{symbol} - {signals['date']}".center(100))
    output.append(f"{'='*100}")

    # 基本信息
    output.append(f"\nCurrent Price: {signals['close']:.2f}")
    output.append(f"CCI: {signals['cci']:>8.2f} | CCI_MA: {signals['cci_ma']:>8.2f} | STC: {signals['stc']:>8.2f}")

    # 阈值参考
    output.append(f"\nThresholds:")
    output.append(f"  Oversold: {params['cci_oversold']} | Overbought: {params['cci_overbought']} | Cross Max: {params['cci_cross_max']}")
    output.append(f"  STC Oversold: {params['stc_oversold']} | STC Cross: {params['stc_cross']}")

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
        output.append(f"  [NO SIGNAL] Wait for entry signal")

    # 历史表现
    output.append(f"\nBacktest Performance:")
    output.append(f"  Expected Return: {params['expected_return']:+.2f}% | Max DD: {params['max_dd']:.2f}%")
    output.append(f"  Win Rate: {params['win_rate']:.1f}%")

    # 因子效果
    if 'factor_effect' in params:
        output.append(f"\nFactor Effect: {params['factor_effect']}")

    return '\n'.join(output)


def main():
    print("="*100)
    print("Quick Test - Silver Monitoring".center(100))
    print("="*100)

    symbol = '白银'
    params = OPTIMAL_PARAMS[symbol]
    code = CODE_MAP[symbol]

    print(f"\nTesting {symbol}...")
    print(f"Code: {code}")
    print(f"Expected Return: {params['expected_return']:.2f}%")

    # 获取实时数据
    df = get_akshare_data(code, days=200)

    if df is None:
        print("Failed to get data!")
        return

    print(f"Data loaded: {len(df)} bars")

    # 计算指标
    data = calculate_indicators(df, params['cci_length'], params['ma_length'])

    # 检查信号
    signals = check_signals(data, params)

    # 输出结果
    output = format_signal_output(symbol, params, signals)
    print(output)

    print("\n" + "="*100)
    print("Test Complete".center(100))
    print("="*100)


if __name__ == "__main__":
    main()
