# -*- coding: utf-8 -*-
"""
历史镜像验证系统
不听模型怎么说，只看历史怎么做
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trend_prediction_tool import calculate_ghe, calculate_lppl_d
from datetime import timedelta
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

os.chdir(r"D:\期货数据\铜期货监控\daily_backtest")

print("="*80)
print("历史镜像验证系统")
print("="*80)

# ============== 第一步：构建历史GHE+LPPL数据库 ==============
print("\n正在构建历史GHE+LPPL数据库...")

df = pd.read_csv('SN_沪锡_日线_10年.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.rename(columns={'收盘价': 'close'})

window_size = 200

# 计算历史每一天的GHE和LPPL-D
ghe_history = []
lppl_history = []

for i in range(window_size, len(df)):
    window_data = df.iloc[i-window_size:i]
    ghe = calculate_ghe(window_data['close'])
    lppl = calculate_lppl_d(window_data['close'])
    ghe_history.append(ghe)
    lppl_history.append(lppl)

# 创建历史数据库
df_history = df.iloc[window_size:].reset_index(drop=True).copy()
df_history['ghe'] = ghe_history
df_history['lppl'] = lppl_history
df_history = df_history.dropna().reset_index(drop=True)

print(f"历史数据库构建完成：{len(df_history)}天")

# 保存历史数据库供后续使用
df_history[['datetime', 'close', 'ghe', 'lppl']].to_csv('ghe_lppl_history_database.csv', index=False)
print("历史数据库已保存：ghe_lppl_history_database.csv")

# ============== 第二步：验证函数 ==============
def verify_signal_by_mirror(current_ghe, current_lppl, df_db, ghe_tol=0.03, lppl_tol=0.15, min_matches=10):
    """
    通过历史镜像验证信号可靠性

    参数：
        current_ghe: 当前GHE值
        current_lppl: 当前LPPL-D值
        df_db: 历史数据库
        ghe_tol: GHE容差（±0.03）
        lppl_tol: LPPL-D容差（±0.15）
        min_matches: 最少匹配数量

    返回：
        dict: 验证结果
    """
    # 在历史中寻找相似的GHE+LPPL组合
    condition = (
        (abs(df_db['ghe'] - current_ghe) <= ghe_tol) &
        (abs(df_db['lppl'] - current_lppl) <= lppl_tol)
    )

    matches = df_db[condition].copy()

    if len(matches) < min_matches:
        return {
            'valid': False,
            'reason': f'历史相似案例不足（只有{len(matches)}个，需要至少{min_matches}个）',
            'matches_count': len(matches)
        }

    # 计算每个匹配案例的未来收益
    results = []

    for idx, match_row in matches.iterrows():
        # 找到这个匹配点在原数据框中的位置
        match_date = match_row['datetime']
        original_idx = df[df['datetime'] == match_date].index

        if len(original_idx) == 0:
            continue

        original_idx = original_idx[0]

        # 计算未来5/10/20天的收益
        for days in [5, 10, 20]:
            if original_idx + days < len(df):
                future_close = df.loc[original_idx + days, 'close']
                current_close = df.loc[original_idx, 'close']
                return_rate = (future_close - current_close) / current_close * 100

                results.append({
                    'match_date': match_date,
                    'future_days': days,
                    'return_pct': return_rate,
                    'ghe': match_row['ghe'],
                    'lppl': match_row['lppl']
                })

    if len(results) == 0:
        return {
            'valid': False,
            'reason': '无法计算未来收益（数据不足）',
            'matches_count': len(matches)
        }

    # 统计结果
    results_df = pd.DataFrame(results)

    # 按未来天数分组统计
    summary = {}

    for days in [5, 10, 20]:
        day_results = results_df[results_df['future_days'] == days]

        if len(day_results) > 0:
            win_count = (day_results['return_pct'] > 0).sum()
            total_count = len(day_results)
            win_rate = win_count / total_count * 100
            avg_return = day_results['return_pct'].mean()
            max_return = day_results['return_pct'].max()
            min_return = day_results['return_pct'].min()

            summary[f'{days}d'] = {
                'count': total_count,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'max_return': max_return,
                'min_return': min_return
            }

    return {
        'valid': True,
        'matches_count': len(matches),
        'summary': summary,
        'all_results': results_df
    }

# ============== 第三步：测试几个典型组合 ==============
print(f"\n{'='*80}")
print("测试几个典型组合的历史镜像验证")
print(f"{'='*80}")

# 测试组合
test_cases = [
    {'ghe': 0.35, 'lppl': 0.85, 'desc': '之前认为是"强上升"的组合'},
    {'ghe': 0.40, 'lppl': 0.75, 'desc': '中等GHE + 高LPPL'},
    {'ghe': 0.30, 'lppl': 0.25, 'desc': '低GHE + 低LPPL'},
    {'ghe': 0.50, 'lppl': 0.85, 'desc': '高GHE + 高LPPL（最危险组合）'},
    {'ghe': 0.38, 'lppl': 0.73, 'desc': '沪锡10年平均水平'},
]

for case in test_cases:
    print(f"\n{'─'*80}")
    print(f"测试组合：GHE={case['ghe']}, LPPL-D={case['lppl']}")
    print(f"描述：{case['desc']}")
    print(f"{'─'*80}")

    result = verify_signal_by_mirror(
        case['ghe'],
        case['lppl'],
        df_history,
        ghe_tol=0.03,
        lppl_tol=0.15,
        min_matches=5
    )

    if not result['valid']:
        print(f"验证结果：{result['reason']}")
        continue

    print(f"\n历史相似案例数：{result['matches_count']}个")
    print(f"\n{'未来周期':<10} {'案例数':<8} {'胜率':<10} {'平均收益':<12} {'最大收益':<12} {'最小收益':<12}")
    print("-"*80)

    for period, stats in result['summary'].items():
        print(f"{period:<10} {stats['count']:<8} "
              f"{stats['win_rate']:>6.1f}%    {stats['avg_return']:>8.2f}%      "
              f"{stats['max_return']:>8.2f}%      {stats['min_return']:>8.2f}%")

    # 给出明确的建议
    if '5d' in result['summary']:
        win_rate_5d = result['summary']['5d']['win_rate']
        avg_return_5d = result['summary']['5d']['avg_return']

        print(f"\n【实战建议】")
        if win_rate_5d >= 60 and avg_return_5d > 0:
            print(f"  [OK] 历史验证有效：5天胜率{win_rate_5d:.1f}%，平均收益{avg_return_5d:.2f}%")
            print(f"  建议：可以考虑做多")
        elif win_rate_5d <= 40 and avg_return_5d < 0:
            print(f"  [X] 历史验证无效：5天胜率{win_rate_5d:.1f}%，平均亏损{avg_return_5d:.2f}%")
            print(f"  建议：严格避免入场，或考虑做空")
        else:
            print(f"  [!] 历史验证模糊：5天胜率{win_rate_5d:.1f}%，平均收益{avg_return_5d:.2f}%")
            print(f"  建议：无明显优势，不建议单独使用此信号")

# ============== 第四步：分析当前市场状态 ==============
print(f"\n{'='*80}")
print("当前市场状态验证")
print(f"{'='*80}")

# 获取最新的GHE和LPPL值
latest_ghe = df_history.iloc[-1]['ghe']
latest_lppl = df_history.iloc[-1]['lppl']
latest_date = df_history.iloc[-1]['datetime']
latest_close = df_history.iloc[-1]['close']

print(f"\n当前日期：{latest_date}")
print(f"当前价格：{latest_close:.2f}")
print(f"当前GHE：{latest_ghe:.3f}")
print(f"当前LPPL-D：{latest_lppl:.3f}")

print(f"\n{'─'*80}")
print("历史镜像验证结果：")
print(f"{'─'*80}")

result = verify_signal_by_mirror(
    latest_ghe,
    latest_lppl,
    df_history,
    ghe_tol=0.03,
    lppl_tol=0.15,
    min_matches=10
)

if not result['valid']:
    print(f"[!] {result['reason']}")
else:
    print(f"\n历史相似案例数：{result['matches_count']}个")
    print(f"\n{'未来周期':<10} {'案例数':<8} {'胜率':<10} {'平均收益':<12} {'最大收益':<12} {'最小收益':<12}")
    print("-"*80)

    for period, stats in result['summary'].items():
        print(f"{period:<10} {stats['count']:<8} "
              f"{stats['win_rate']:>6.1f}%    {stats['avg_return']:>8.2f}%      "
              f"{stats['max_return']:>8.2f}%      {stats['min_return']:>8.2f}%")

    # 给出明确的建议
    print(f"\n{'='*80}")
    print("【最终实战建议】")
    print(f"{'='*80}")

    if '5d' in result['summary']:
        win_rate_5d = result['summary']['5d']['win_rate']
        avg_return_5d = result['summary']['5d']['avg_return']

        if win_rate_5d >= 60 and avg_return_5d > 0:
            print(f"[OK] 当前GHE+LPPL组合，历史验证有效")
            print(f"   - 5天胜率：{win_rate_5d:.1f}%")
            print(f"   - 平均收益：{avg_return_5d:.2f}%")
            print(f"   建议：可以考虑做多，仓位可适当加大")
        elif win_rate_5d <= 40 and avg_return_5d < 0:
            print(f"[X] 当前GHE+LPPL组合，历史验证危险")
            print(f"   - 5天胜率：{win_rate_5d:.1f}%")
            print(f"   - 平均亏损：{avg_return_5d:.2f}%")
            print(f"   建议：严格避免入场，或考虑做空")
        else:
            print(f"[!] 当前GHE+LPPL组合，历史验证模糊")
            print(f"   - 5天胜率：{win_rate_5d:.1f}%")
            print(f"   - 平均收益：{avg_return_5d:.2f}%")
            print(f"   建议：无明显优势，不建议单独使用GHE+LPPL作为入场依据")

    print(f"\n核心结论：")
    print(f"  不听模型怎么说，只看历史怎么做。")
    print(f"  历史数据是最诚实的裁判。")

print(f"\n{'='*80}")
print("验证完成")
print(f"{'='*80}")
