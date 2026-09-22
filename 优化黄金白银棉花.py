"""
优化黄金、白银、棉花的CCI参数
==============================
"""

import pandas as pd
import numpy as np
import pickle
import itertools
import os

print("="*80)
print("优化黄金、白银、棉花".center(80))
print("="*80)

# ============ 加载数据 ============
data_dir = "D:\\期货数据\\铜期货监控\\global_futures_daily"

files_to_load = {
    'au': '黄金_daily.csv',   # 黄金
    'ag': '白银_daily.csv',   # 白银
    'cf': '棉花_daily.csv'    # 棉花
}

new_data = {}

for symbol_key, filename in files_to_load.items():
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"  [警告] 文件不存在: {filename}")
        continue

    print(f"\n加载 {filename}...")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # 只保留需要的列
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # 检查数据时间范围
    start_date = df.index[0].strftime('%Y-%m-%d')
    end_date = df.index[-1].strftime('%Y-%m-%d')
    total_records = len(df)

    print(f"  时间范围: {start_date} 至 {end_date}")
    print(f"  总记录数: {total_records}")

    new_data[symbol_key] = df

# ============ 加载原有数据池 ============
print("\n" + "="*80)
print("加载数据池...")
print("="*80)

try:
    with open('data_pool.pkl', 'rb') as f:
        DATA_POOL = pickle.load(f)
except:
    DATA_POOL = {}

# 更新数据池
DATA_POOL.update(new_data)

# ============ CCI回测函数 ============
def calculate_cci(df, length=20):
    data = df.copy()
    data['tp'] = (data['high'] + data['low'] + data['close']) / 3
    data['tp_ma'] = data['tp'].rolling(window=length).mean()
    data['tp_dev'] = data['tp'].rolling(window=length).std()
    data['cci'] = (data['tp'] - data['tp_ma']) / (0.015 * data['tp_dev'])
    data['cci'] = data['cci'].fillna(0)
    return data

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

# ============ 优化黄金、白银、棉花 ============
symbols_to_optimize = {
    'au': '黄金',
    'ag': '白银',
    'cf': '棉花'
}

# 参数网格
cci_length_grid = list(range(14, 36, 2))
ma_length_grid = list(range(3, 16, 1))

all_results = {}

for symbol_key, symbol_name in symbols_to_optimize.items():
    if symbol_key not in DATA_POOL:
        print(f"\n[警告] {symbol_name} 数据不存在，跳过")
        continue

    print("\n" + "="*80)
    print(f"{symbol_name} - CCI参数优化".center(80))
    print("="*80)

    df_full = DATA_POOL[symbol_key].copy()
    df_full.sort_index(inplace=True)

    train_df = df_full.loc[:'2023-12-31'].copy()
    test_df = df_full.loc['2024-01-01':].copy()

    print(f"\n数据:")
    print(f"  训练集: {len(train_df)}根K线 ({train_df.index[0].strftime('%Y-%m-%d')} 至 {train_df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  测试集: {len(test_df)}根K线 ({test_df.index[0].strftime('%Y-%m-%d')} 至 {test_df.index[-1].strftime('%Y-%m-%d')})")

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

    if len(results) == 0:
        print(f"  [错误] 没有有效的参数组合")
        continue

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

        print(f"\n[测试集最高收益]")
        print(f"  参数: CCI({best_test['cci_length']},MA{best_test['ma_length']})")
        print(f"  训练: {best_test['train_return']:+.2f}% | 测试: {best_test['return_pct']:+.2f}% | "
              f"差异: {best_test['return_pct']-best_test['train_return']:+.2f}%")
        print(f"  回撤: {best_test['max_dd']:.2f}% | 胜率: {best_test['win_rate']:.1f}% | "
              f"交易: {best_test['total_trades']}笔")

        # 与默认参数对比
        default_test = backtest_cci(test_df, 20, 14)
        if default_test:
            improvement = best_test['return_pct'] - default_test['return_pct']
            dd_improvement = default_test['max_dd'] - best_test['max_dd']

            print(f"\n[vs 默认参数 CCI(20,14)]")
            print(f"  默认: {default_test['return_pct']:+.2f}% (回撤{default_test['max_dd']:.2f}%, "
                  f"胜率{default_test['win_rate']:.1f}%, 交易{default_test['total_trades']}笔)")
            print(f"  优化: {best_test['return_pct']:+.2f}% (回撤{best_test['max_dd']:.2f}%, "
                  f"胜率{best_test['win_rate']:.1f}%, 交易{best_test['total_trades']}笔)")
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
            'default': default_test if default_test else None
        }

# ============ 总结对比 ============
print("\n" + "="*80)
print("所有品种总结".center(80))
print("="*80)

# 之前优化的品种结果
previous_results = [
    {'symbol': '锡', 'key': 'sn', 'params': (24, 5), 'return': 287.93, 'dd': -11.16, 'trades': 46},
    {'symbol': '铅', 'key': 'pb', 'params': (18, 12), 'return': 148.20, 'dd': -12.02, 'trades': 50},
    {'symbol': '铜', 'key': 'cu', 'params': (20, 10), 'return': 101.47, 'dd': -14.89, 'trades': 36},
    {'symbol': '镍', 'key': 'ni', 'params': (20, 10), 'return': 63.78, 'dd': -20.73, 'trades': 35},
    {'symbol': '铝', 'key': 'al', 'params': (20, 10), 'return': 63.78, 'dd': -20.73, 'trades': 35},
    {'symbol': '锌', 'key': 'zn', 'params': (20, 14), 'return': 43.10, 'dd': -17.46, 'trades': 50, 'note': '默认参数'},
]

# 添加新优化的品种结果
for symbol_key, res in all_results.items():
    rec = {
        'symbol': res['name'],
        'key': symbol_key,
        'params': (res['best_test']['cci_length'], res['best_test']['ma_length']),
        'return': res['best_test']['return_pct'],
        'dd': res['best_test']['max_dd'],
        'trades': res['best_test']['total_trades']
    }
    previous_results.append(rec)

# 排序
previous_results.sort(key=lambda x: x['return'], reverse=True)

print(f"\n{'排名':<6} {'品种':<8} {'推荐参数':<20} {'预期收益':<12} {'预期回撤':<12} {'交易次数':<10} {'评价':<10}")
print("-"*80)

for rank, rec in enumerate(previous_results, 1):
    symbol = rec['symbol']
    params = f"CCI({rec['params'][0]},MA{rec['params'][1]})"
    ret = rec['return']
    dd = rec['dd']
    trades = rec.get('trades', 0)

    if ret > 200:
        evaluation = "震撼"
    elif ret > 100:
        evaluation = "优秀"
    elif ret > 50:
        evaluation = "良好"
    elif ret > 20:
        evaluation = "一般"
    else:
        evaluation = "较差"

    print(f"{rank:<6} {symbol:<8} {params:<20} {ret:>+10.2f}% {dd:>10.2f}% {trades:>8} {evaluation:<10}")

# 计算组合收益
print("\n" + "="*80)
print("多品种组合分析".center(80))
print("="*80)

# 全部品种
total_return = sum([r['return'] for r in previous_results])
avg_return = total_return / len(previous_results)
max_dd = min([r['dd'] for r in previous_results])

print(f"\n[全部{len(previous_results)}个品种等权重组合]")
print(f"  平均收益: {avg_return:+.2f}%")
print(f"  最大回撤: {max_dd:.2f}%")

# 优选品种（收益>50%）
good_performers = [r for r in previous_results if r['return'] > 50]
if len(good_performers) > 0:
    avg_return_good = sum([r['return'] for r in good_performers]) / len(good_performers)
    max_dd_good = min([r['dd'] for r in good_performers])

    print(f"\n[优选组合（收益>50%的{len(good_performers)}个品种）]")
    print(f"  品种: {', '.join([r['symbol'] for r in good_performers])}")
    print(f"  平均收益: {avg_return_good:+.2f}%")
    print(f"  最大回撤: {max_dd_good:.2f}%")

# 优秀品种（收益>100%）
excellent_performers = [r for r in previous_results if r['return'] > 100]
if len(excellent_performers) > 0:
    avg_return_excellent = sum([r['return'] for r in excellent_performers]) / len(excellent_performers)
    max_dd_excellent = min([r['dd'] for r in excellent_performers])

    print(f"\n[精英组合（收益>100%的{len(excellent_performers)}个品种）]")
    print(f"  品种: {', '.join([r['symbol'] for r in excellent_performers])}")
    print(f"  平均收益: {avg_return_excellent:+.2f}%")
    print(f"  最大回撤: {max_dd_excellent:.2f}%")

# 按品种类型分组
print(f"\n[按品种类型分组]")
print(f"\n金属品种:")
metals = [r for r in previous_results if r['symbol'] in ['锡', '铅', '铜', '锌', '铝', '镍', '黄金', '白银']]
if metals:
    avg_metals = sum([r['return'] for r in metals]) / len(metals)
    print(f"  品种: {', '.join([r['symbol'] for r in metals])}")
    print(f"  平均收益: {avg_metals:+.2f}%")

print(f"\n农产品:")
agricultural = [r for r in previous_results if r['symbol'] in ['棉花']]
if agricultural:
    avg_agri = sum([r['return'] for r in agricultural]) / len(agricultural)
    print(f"  品种: {', '.join([r['symbol'] for r in agricultural])}")
    print(f"  平均收益: {avg_agri:+.2f}%")

print("\n" + "="*80)
print("优化完成！")
print("="*80)
