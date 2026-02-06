# -*- coding: utf-8 -*-
"""
测试：分析当前沪锡市场状态
"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir("D:\\期货数据\\铜期货监控\\daily_backtest")

from trend_prediction_tool import analyze_current_market, print_analysis_report

# 加载数据
print("加载沪锡10年数据...")
df = pd.read_csv('SN_沪锡_日线_10年.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print(f"数据范围: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
print(f"数据量: {len(df)} 天")

# 分析当前市场（最新数据）
print("\n分析当前市场状态...")
result = analyze_current_market(df)

# 打印报告
print_analysis_report(result)

# 保存结果到文件
if result['status'] == 'success':
    with open('current_market_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GHE+LPPL趋势预判报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"当前日期: {result['current_date']}\n")
        f.write(f"当前价格: {result['current_price']:.2f}\n")
        f.write(f"GHE值: {result['ghe']:.3f}\n")
        f.write(f"LPPL-D值: {result['lppl_d']:.3f}\n\n")

        pred = result['prediction']
        f.write(f"预测趋势: {pred['forecast']}\n")
        f.write(f"置信度: {pred['confidence']}%\n")
        f.write(f"样本数: {pred.get('sample_size', 'N/A')}\n\n")

        f.write(f"概率分布:\n")
        f.write(f"  强上升: {pred['probability']['STRONG_UP']:.1f}%\n")
        f.write(f"  强下降: {pred['probability']['STRONG_DOWN']:.1f}%\n")
        f.write(f"  横盘震荡: {pred['probability']['RANGING']:.1f}%\n\n")

        f.write(f"交易建议:\n")
        f.write(f"  {pred['action']}\n")

        if 'warning' in pred:
            f.write(f"\n注意事项:\n")
            f.write(f"  {pred['warning']}\n")

        if 'details' in pred:
            f.write(f"\n组合详情:\n")
            f.write(f"  {pred['details']}\n")

        f.write("\n" + "=" * 80 + "\n")

    print("\n报告已保存到: current_market_analysis.txt")

print("\n分析完成！")
