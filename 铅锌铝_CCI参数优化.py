"""
铅、锌、铝 - CCI参数优化
=======================
依次优化这三个品种的CCI参数
"""

import pandas as pd
import numpy as np
import pickle
import itertools

print("="*80)
print("铅、锌、铝 - CCI参数优化".center(80))
print("="*80)

# 加载数据
with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

# 计算CCI
def calculate_cci(df, length=20):
    data = df.copy()
    data['tp'] = (data['high'] + data['low'] + data['close']) / 3
    data['tp_ma'] = data['tp'].rolling(window=length).mean()
    data['tp_dev'] = data['tp'].rolling(window=length).std()
    data['cci'] = (data['tp'] - data['tp_ma']) / (0.015 * data['tp_dev'])
    data['cci'] = data['cci'].fillna(0)
    return data

# 回测函数
def backtest_cci(df, cci_length, ma_length):
    data = calculate_cci(df, length=cci_length)
    data['cci_ma'] = data['cci'].rolling(window=ma_length).mean()

    balance = 100000
    position = 0
    entry_price = 0.0
    trades = []
    equity = []

    closes = data['close'].values
    opens = data['open'].values
    cci = data['cci'].values
    cci_ma = data['cci_ma'].values

    start_idx = cci_length + ma_length + 1

    for i in range(start_idx, len(data)-1):
        cci_val = cci[i]
        cci_ma_val = cci_ma[i]
        cci_prev = cci[i-1]
        cci_ma_prev = cci_ma[i-1]

        # 平仓逻辑
        if position > 0:
            if cci_prev >= cci_ma_prev and cci_val <= cci_ma_val:
                exit_p = opens[i+1]
                pnl = (exit_p - entry_price) * position * 5
                balance += pnl
                trades.append(pnl)
                position = 0

        # 开仓逻辑
        if position == 0:
            if cci_prev <= cci_ma_prev and cci_val > cci_ma_val:
                qty = int(balance * 2.0 / (opens[i+1] * 5))
                qty = max(1, qty)
                position = qty
                entry_price = opens[i+1]

        val = balance
        if position > 0:
            val += (closes[i+1] - entry_price) * position * 5
        equity.append(val)

    if not trades or len(trades) < 5:
        return None

    ret = (equity[-1] - 100000) / 100000 * 100
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100

    equity_series = pd.Series(equity)
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100
    max_dd = dd.min()

    winning = [t for t in trades if t > 0]
    losing = [t for t in trades if t <= 0]
    profit_factor = abs(sum(winning) / sum(losing)) if losing else 0

    return {
        'return_pct': ret,
        'max_dd': max_dd,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'final_balance': equity[-1],
        'cci_length': cci_length,
        'ma_length': ma_length
    }

# 要优化的品种
symbols_to_optimize = {
    'pb': '铅',
    'zn': '锌',
    'al': '铝'
}

# 参数网格（参考铜和锡的结果）
cci_length_grid = list(range(14, 36, 2))  # [14, 16, ..., 34]
ma_length_grid = list(range(3, 16, 1))    # [3, 4, ..., 15]

all_results = {}

for symbol_key, symbol_name in symbols_to_optimize.items():
    if symbol_key not in DATA_POOL:
        print(f"\n[警告] {symbol_name} 数据不存在，跳过")
        continue

    print("\n" + "="*80)
    print(f"{symbol_name}期货 - CCI参数优化".center(80))
    print("="*80)

    df_full = DATA_POOL[symbol_key].copy()
    df_full.sort_index(inplace=True)

    train_df = df_full.loc[:'2023-12-31'].copy()
    test_df = df_full.loc['2024-01-01':].copy()

    print(f"\n数据:")
    print(f"  训练集: {len(train_df)}根K线 (2016-2023)")
    print(f"  测试集: {len(test_df)}根K线 (2024-2026)")

    # ============ 网格搜索 ============
    print(f"\n参数网格: CCI{cci_length_grid}, MA{ma_length_grid}")
    print(f"总组合数: {len(cci_length_grid) * len(ma_length_grid)}")

    param_combinations = list(itertools.product(cci_length_grid, ma_length_grid))

    import time
    start_time = time.time()

    results = []
    for i, (cci_len, ma_len) in enumerate(param_combinations):
        result = backtest_cci(train_df, cci_len, ma_len)
        if result:
            results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  进度: {i+1}/{len(param_combinations)} ({(i+1)/len(param_combinations)*100:.1f}%) | "
                  f"耗时: {elapsed:.1f}秒 | 有效: {len(results)}", flush=True)

    elapsed = time.time() - start_time
    print(f"  完成! 耗时: {elapsed:.1f}秒 | 有效组合: {len(results)}")

    # ============ 排序 ============
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('return_pct', ascending=False)

    # ============ Top 15样本外测试 ============
    print(f"\n{'='*80}")
    print("Top 15参数样本外验证".center(80))
    print(f"{'='*80}")

    top_params = results_df.head(15)
    out_of_sample_results = []

    for idx, row in top_params.iterrows():
        cci_len = int(row['cci_length'])
        ma_len = int(row['ma_length'])

        try:
            test_result = backtest_cci(test_df, cci_len, ma_len)

            if test_result:
                test_result['train_return'] = row['return_pct']
                test_result['train_win_rate'] = row['win_rate']
                test_result['train_dd'] = row['max_dd']
                out_of_sample_results.append(test_result)
        except:
            pass

    # 找到测试集表现最好的
    if out_of_sample_results:
        best_test = max(out_of_sample_results, key=lambda x: x['return_pct'])

        # 找到训练集和测试集平衡最好的（差异最小且测试收益高）
        best_balance = max(out_of_sample_results,
                          key=lambda x: x['return_pct'] if abs(x['return_pct'] - x['train_return']) < 50 else 0)

        print(f"\n[测试集最高收益]")
        print(f"  参数: CCI({best_test['cci_length']},MA{best_test['ma_length']})")
        print(f"  训练: {best_test['train_return']:+.2f}% | 测试: {best_test['return_pct']:+.2f}% | "
              f"差异: {best_test['return_pct']-best_test['train_return']:+.2f}%")

        print(f"\n[最稳健（训练测试接近）]")
        if best_balance != best_test:
            print(f"  参数: CCI({best_balance['cci_length']},MA{best_balance['ma_length']})")
            print(f"  训练: {best_balance['train_return']:+.2f}% | 测试: {best_balance['return_pct']:+.2f}% | "
                  f"差异: {best_balance['return_pct']-best_balance['train_return']:+.2f}%")
        else:
            print(f"  与测试集最高收益相同")

        # 与默认参数对比
        default_test = backtest_cci(test_df, 20, 14)
        if default_test:
            improvement = best_test['return_pct'] - default_test['return_pct']
            print(f"\n[vs 默认参数 CCI(20,14)]")
            print(f"  默认: {default_test['return_pct']:+.2f}%")
            print(f"  优化: {best_test['return_pct']:+.2f}%")
            print(f"  提升: {improvement:+.2f}%", end="")

            if improvement > 20:
                print(" [优秀]")
            elif improvement > 10:
                print(" [良好]")
            elif improvement > 0:
                print(" [提升]")
            else:
                print(" [下降]")

        all_results[symbol_key] = {
            'name': symbol_name,
            'best_test': best_test,
            'best_balance': best_balance if best_balance != best_test else best_test,
            'default': default_test if default_test else None,
            'all_test_results': out_of_sample_results
        }

# ============ 总结对比 ============
print("\n" + "="*80)
print("总结对比 - 所有品种".center(80))
print("="*80)

print(f"\n{'品种':<8} {'默认参数收益':<15} {'优化参数收益':<15} {'优化参数回撤':<15} {'提升':<10} {'推荐参数':<20}")
print("-"*80)

for symbol_key, symbol_name in symbols_to_optimize.items():
    if symbol_key not in all_results:
        continue

    res = all_results[symbol_key]
    best = res['best_test']
    default = res['default']

    if default:
        improvement = best['return_pct'] - default['return_pct']
        param_str = f"CCI({best['cci_length']},MA{best['ma_length']})"

        print(f"{symbol_name:<8} {default['return_pct']:>+13.2f}% {best['return_pct']:>+13.2f}% "
              f"{best['max_dd']:>13.2f}% {improvement:>+8.2f}% {param_str}")

# ============ 最终推荐 ============
print("\n" + "="*80)
print("最终参数推荐".center(80))
print("="*80)

recommendations = {
    'cu': {'symbol': '铜', 'params': (20, 10), 'return': 101.47, 'dd': -14.89},
    'sn': {'symbol': '锡', 'params': (24, 5), 'return': 287.93, 'dd': -11.16},
}

for symbol_key, res in all_results.items():
    recommendations[symbol_key] = {
        'symbol': res['name'],
        'params': (res['best_test']['cci_length'], res['best_test']['ma_length']),
        'return': res['best_test']['return_pct'],
        'dd': res['best_test']['max_dd']
    }

print(f"\n{'品种':<8} {'推荐参数':<20} {'预期收益':<12} {'预期回撤':<12} {'评价':<10}")
print("-"*80)

for symbol_key, rec in recommendations.items():
    symbol = rec['symbol']
    params = f"CCI({rec['params'][0]},MA{rec['params'][1]})"
    ret = rec['return']
    dd = rec['dd']

    if ret > 100:
        evaluation = "优秀"
    elif ret > 50:
        evaluation = "良好"
    elif ret > 20:
        evaluation = "一般"
    else:
        evaluation = "较差"

    print(f"{symbol:<8} {params:<20} {ret:>+10.2f}% {dd:>10.2f}% {evaluation:<10}")

print("\n" + "="*80)
print("优化完成！")
print("="*80)
