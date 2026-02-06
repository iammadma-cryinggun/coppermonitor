# -*- coding: utf-8 -*-
"""
用LPPL模型分析沪锡10年数据的趋势
检测泡沫和趋势反转信号
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os

# 添加LPPL模块路径
sys.path.insert(0, 'D:/lppls')

try:
    from lppls import lppls
    LPPL_AVAILABLE = True
except ImportError as e:
    print(f"警告: LPPL模块导入失败: {e}")
    print("请确保lppls已安装: pip install lppls")
    LPPL_AVAILABLE = False


def analyze_sn_with_lppl():
    """用LPPL分析沪锡趋势"""
    print("="*80)
    print("沪锡10年数据 - LPPL趋势分析")
    print("="*80)

    if not LPPL_AVAILABLE:
        print("\n[ERROR] LPPL模块不可用，无法继续分析")
        return

    # 读取数据
    csv_file = "SN_沪锡_日线_10年.csv"
    df = pd.read_csv(csv_file)

    # 转换列名
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

    print(f"\n数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    print(f"数据量: {len(df)} 条")

    # 准备LPPL数据
    # 将时间转换为ordinal
    time = [pd.Timestamp.toordinal(t) for t in df['datetime']]

    # 使用对数价格
    price = np.log(df['close'].values)

    # 创建观测数组
    observations = np.array([time, price])

    print(f"\n开始LPPL分析...")
    print("注意: LPPL分析可能需要较长时间（几分钟）")

    # 实例化LPPL模型
    lppls_model = lppls.LPPLS(observations=observations)

    # 尝试拟合模型
    try:
        MAX_SEARCHES = 25
        tc, m, w, a, b, c, c1, c2, O, D = lppls_model.fit(MAX_SEARCHES)

        print(f"\nLPPL拟合结果:")
        print(f"  tc (临界时间): {pd.Timestamp.fromordinal(int(tc))}")
        print(f"  m (超指数增长度): {m:.4f}")
        print(f"  w (振荡频率): {w:.4f}")
        print(f"  a (峰值对数价格): {a:.4f}")
        print(f"  b (幂律振幅): {b:.4f}")
        print(f"  c (振荡振幅): {c:.4f}")
        print(f"  O (拟合优度): {O:.4f}")
        print(f"  D (泡沫指标): {D:.4f}")

        # 判断泡沫状态
        if D > 0.5:
            print(f"\n[WARNING] 检测到泡沫信号！D值={D:.4f} > 0.5")
        elif D > 0.3:
            print(f"\n[CAUTION] 可能存在泡沫，D值={D:.4f}")
        else:
            print(f"\n[OK] 无明显泡沫信号，D值={D:.4f}")

        # 绘制拟合图
        print(f"\n生成拟合图...")
        lppls_model.plot_fit()
        plt.savefig('SN_lppl_fit.png', dpi=150, bbox_inches='tight')
        print(f"  图表已保存: SN_lppl_fit.png")

    except Exception as e:
        print(f"\n[ERROR] LPPL拟合失败: {e}")
        import traceback
        traceback.print_exc()

    # 计算置信度指标（滑动窗口）
    print(f"\n开始计算滑动窗口置信度指标...")
    print("注意: 这可能需要更长时间（10-30分钟）")

    try:
        res = lppls_model.mp_compute_nested_fits(
            workers=4,  # 使用4个worker
            window_size=120,  # 窗口大小120天
            smallest_window_size=30,
            outer_increment=1,
            inner_increment=5,
            max_searches=25,
        )

        print(f"\n置信度指标计算完成！")

        # 绘制置信度指标
        lppls_model.plot_confidence_indicators(res)
        plt.savefig('SN_lppl_confidence.png', dpi=150, bbox_inches='tight')
        print(f"  置信度图表已保存: SN_lppl_confidence.png")

        # 计算指标DataFrame
        res_df = lppls_model.compute_indicators(res)
        res_df.to_csv('SN_lppl_indicators.csv', index=False)
        print(f"  指标数据已保存: SN_lppl_indicators.csv")

        # 分析最近的泡沫信号
        recent = res_df.tail(30)
        bubble_signals = recent[recent['qualified'] > 0.5]

        print(f"\n最近30天的泡沫信号:")
        if len(bubble_signals) > 0:
            print(f"  检测到 {len(bubble_signals)} 天的泡沫信号")
            print(f"\n  日期             泡沫概率  临界时间tc")
            for _, row in bubble_signals.iterrows():
                tc_date = pd.Timestamp.fromordinal(int(row['tc']))
                print(f"  {row['time']}    {row['qualified']:.2f}      {tc_date.date()}")
        else:
            print(f"  最近30天无泡沫信号")

    except Exception as e:
        print(f"\n[ERROR] 置信度计算失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("LPPL分析完成")
    print("="*80)


if __name__ == "__main__":
    os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")
    analyze_sn_with_lppl()
