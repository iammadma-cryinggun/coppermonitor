# -*- coding: utf-8 -*-
"""
市场状态识别系统
判断当前是牛市、震荡市还是熊市
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_monitor import calculate_indicators

# 最优参数
BEST_PARAMS = {
    'EMA_FAST': 3,
    'EMA_SLOW': 15,
    'RSI_FILTER': 30,
    'RATIO_TRIGGER': 1.05,
    'STC_SELL_ZONE': 65,
    'STOP_LOSS_PCT': 0.02
}


def detect_market_regime(df, lookback=60):
    """
    识别市场状态

    Args:
        df: 包含价格和技术指标的数据
        lookback: 回看天数

    Returns:
        dict: {
            'regime': 'BULL'/'BEAR'/'RANGING',
            'trend_strength': 0-100,
            'volatility': 'LOW'/'MEDIUM'/'HIGH',
            'confidence': 0-100
        }
    """
    if len(df) < lookback:
        return {
            'regime': 'RANGING',
            'trend_strength': 50,
            'volatility': 'MEDIUM',
            'confidence': 50
        }

    recent = df.tail(lookback)

    # 1. 趋势判断（基于EMA）
    ema_fast = recent['ema_fast'].iloc[-1]
    ema_slow = recent['ema_slow'].iloc[-1]
    ema_trend = (ema_fast - ema_slow) / ema_slow * 100

    # 2. 趋势强度（基于价格变化率）
    price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100
    annualized_change = price_change / lookback * 250

    # 3. 波动率判断
    returns = recent['close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(250) * 100  # 年化波动率

    # 4. RSI判断
    current_rsi = recent['rsi'].iloc[-1]

    # 5. MACD判断
    macd_dif = recent['macd_dif'].iloc[-1]

    # 综合判断
    bullish_signals = 0
    bearish_signals = 0

    # 趋势信号
    if ema_trend > 0.5:  # 快线在慢线上方0.5%以上
        bullish_signals += 2
    elif ema_trend < -0.5:
        bearish_signals += 2

    # 价格变化率信号
    if annualized_change > 20:  # 年化涨幅超20%
        bullish_signals += 2
    elif annualized_change < -20:
        bearish_signals += 2
    elif annualized_change > 10:
        bullish_signals += 1
    elif annualized_change < -10:
        bearish_signals += 1

    # MACD信号
    if macd_dif > 0:
        bullish_signals += 1
    else:
        bearish_signals += 1

    # RSI信号
    if current_rsi > 60:
        bullish_signals += 1
    elif current_rsi < 40:
        bearish_signals += 1

    # 波动率分类
    if volatility > 40:
        volatility_level = 'HIGH'
    elif volatility > 25:
        volatility_level = 'MEDIUM'
    else:
        volatility_level = 'LOW'

    # 趋势强度（0-100）
    signal_diff = bullish_signals - bearish_signals
    trend_strength = 50 + signal_diff * 10  # 转换为0-100
    trend_strength = max(0, min(100, trend_strength))

    # 市场状态判断
    if bullish_signals >= 4 and bearish_signals <= 1:
        regime = 'BULL'
        confidence = min(100, 70 + signal_diff * 5)
    elif bearish_signals >= 4 and bullish_signals <= 1:
        regime = 'BEAR'
        confidence = min(100, 70 - signal_diff * 5)
    else:
        regime = 'RANGING'
        confidence = 50 + abs(signal_diff) * 5

    return {
        'regime': regime,
        'trend_strength': trend_strength,
        'volatility': volatility_level,
        'confidence': confidence,
        'annualized_change': annualized_change,
        'ema_trend': ema_trend,
        'current_rsi': current_rsi,
        'macd_dif': macd_dif,
        'volatility_value': volatility
    }


def analyze_historical_regimes(df):
    """分析历史市场状态分布"""
    df = calculate_indicators(df.copy(), BEST_PARAMS)

    regimes = []
    for i in range(200, len(df), 20):  # 每20天采样一次
        current_df = df.iloc[:i+1]
        regime_info = detect_market_regime(current_df, lookback=60)
        regime_info['date'] = df.iloc[i]['datetime']
        regimes.append(regime_info)

    regime_df = pd.DataFrame(regimes)

    return regime_df


def main():
    print("="*100)
    print("市场状态识别系统 - 沪锡10年分析")
    print("="*100)

    # 加载数据
    df = pd.read_csv('SN_沪锡_日线_10年.csv')
    df = df.rename(columns={
        '开盘价': 'open',
        '最高价': 'high',
        '最低价': 'low',
        '收盘价': 'close',
        '成交量': 'volume',
        '持仓量': 'hold'
    })
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 计算指标
    df = calculate_indicators(df.copy(), BEST_PARAMS)

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    # 分析历史状态
    print("\n[1] 分析历史市场状态分布...")
    regime_df = analyze_historical_regimes(df)

    print(f"\n市场状态分布:")
    regime_counts = regime_df['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(regime_df) * 100
        regime_name = {'BULL': '牛市', 'BEAR': '熊市', 'RANGING': '震荡市'}[regime]
        print(f"  {regime_name}: {count}天 ({pct:.1f}%)")

    # 当前状态
    print("\n[2] 当前市场状态...")
    current_regime = detect_market_regime(df, lookback=60)

    regime_name = {'BULL': '牛市', 'BEAR': '熊市', 'RANGING': '震荡市'}[current_regime['regime']]

    print(f"\n当前状态: {regime_name}")
    print(f"  置信度: {current_regime['confidence']:.1f}%")
    print(f"  趋势强度: {current_regime['trend_strength']:.1f}/100")
    print(f"  波动率: {current_regime['volatility']} ({current_regime['volatility_value']:.2f}%年化)")
    print(f"  年化收益率: {current_regime['annualized_change']:.2f}%")
    print(f"  EMA趋势: {current_regime['ema_trend']:.2f}%")
    print(f"  RSI: {current_regime['current_rsi']:.1f}")
    print(f"  MACD: {current_regime['macd_dif']:.2f}")

    # 策略推荐
    print("\n[3] 策略推荐...")

    print(f"\n基于当前{regime_name}状态，推荐策略:")

    if current_regime['regime'] == 'BULL':
        print(f"\n  [主策略] 纯做多策略")
        print(f"    - 理由: 牛市趋势明确，做多策略收益最大（803.75%）")
        print(f"    - 配置: 80%纯做多 + 20%观望")
        print(f"    - 止损: 2%")
        print(f"    - 止盈: STC指标")

    elif current_regime['regime'] == 'BEAR':
        print(f"\n  [主策略] LPPL双向策略（偏做空）")
        print(f"    - 理由: 熊市下行，做多机会少，LPPL高D值做空有效")
        print(f"    - 配置: 60%做空 + 20%做多 + 20%观望")
        print(f"    - LPPL阈值: D>=80%做空")

    else:  # RANGING
        print(f"\n  [主策略] LPPL双向策略（均衡）")
        print(f"    - 理由: 震荡市多空不明，LPPL双向最稳健（212.47%）")
        print(f"    - 配置: 40%做多 + 40%做空 + 20%观望")
        print(f"    - LPPL阈值: D<=20%做多, D>=80%做空")

    # 状态切换规则
    print("\n[4] 状态切换规则...")

    print(f"\n市场状态识别标准:")
    print(f"  牛市信号:")
    print(f"    - EMA快线持续在慢线上方 >0.5%")
    print(f"    - 年化收益率 >20%")
    print(f"    - MACD金叉")
    print(f"    - RSI >60")
    print(f"  需要>=4个信号确认牛市")

    print(f"\n  熊市信号:")
    print(f"    - EMA快线持续在慢线下方 <-0.5%")
    print(f"    - 年化收益率 <-20%")
    print(f"    - MACD死叉")
    print(f"    - RSI <40")
    print(f"  需要>=4个信号确认熊市")

    print(f"\n  震荡市信号:")
    print(f"    - 牛市和熊市信号都不明确")
    print(f"    - 年化收益率在-20%到20%之间")
    print(f"    - 波动率适中")

    # 历史状态切换
    print(f"\n[5] 历史状态切换分析...")

    regime_df['regime_name'] = regime_df['regime'].map({
        'BULL': '牛市',
        'BEAR': '熊市',
        'RANGING': '震荡市'
    })

    # 检测状态切换
    regime_df['regime_change'] = regime_df['regime'].ne(regime_df['regime'].shift())
    changes = regime_df[regime_df['regime_change']]

    print(f"\n历史状态切换: {len(changes)}次")

    if len(changes) > 0:
        print(f"\n重要切换点:")
        for idx, row in changes.head(10).iterrows():
            from_regime = regime_df.iloc[idx-1]['regime_name'] if idx > 0 else '起始'
            print(f"  {row['date'].date()}: {from_regime} → {row['regime_name']}")

    # 保存状态历史
    output_file = 'market_regime_history.csv'
    regime_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n市场状态历史已保存: {output_file}")

    print("\n" + "="*100)
    print("总结")
    print("="*100)

    print(f"\n市场状态系统已建立，可用于:")
    print(f"  1. 实时判断当前市场状态")
    print(f"  2. 根据状态切换策略（牛市纯做多，震荡/熊市LPPL双向）")
    print(f"  3. 动态调整仓位配置")
    print(f"\n建议每20天重新评估一次市场状态")

    print("="*100)


if __name__ == "__main__":
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    main()
