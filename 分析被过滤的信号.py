"""
分析被ADX过滤的信号详情
"""
import pandas as pd
import numpy as np
import pickle

with open('data_pool.pkl', 'rb') as f:
    DATA_POOL = pickle.load(f)

# 计算ADX
def calculate_adx(df, length=14):
    data = df.copy()
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['tr'] = data['tr'].fillna(0)

    data['up_move'] = data['high'] - data['high'].shift(1)
    data['down_move'] = data['low'].shift(1) - data['low']

    data['plus_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0),
                                data['up_move'], 0)
    data['minus_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0),
                                 data['down_move'], 0)

    data['plus_dm'] = data['plus_dm'].fillna(0)
    data['minus_dm'] = data['minus_dm'].fillna(0)

    data['atr'] = data['tr'].rolling(window=length).mean()
    data['plus_di'] = 100 * (data['plus_dm'].rolling(window=length).mean() / data['atr'])
    data['minus_di'] = 100 * (data['minus_dm'].rolling(window=length).mean() / data['atr'])

    data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
    data['dx'] = data['dx'].fillna(0)
    data['adx'] = data['dx'].rolling(window=length).mean()

    return data

# 计算CCI
def calculate_cci(df, length=20):
    data = df.copy()
    data['tp'] = (data['high'] + data['low'] + data['close']) / 3
    data['tp_ma'] = data['tp'].rolling(window=length).mean()
    data['tp_dev'] = data['tp'].rolling(window=length).std()
    data['cci'] = (data['tp'] - data['tp_ma']) / (0.015 * data['tp_dev'])
    data['cci'] = data['cci'].fillna(0)
    return data

symbol_key = 'sn'
df_full = DATA_POOL[symbol_key].copy()
df_full.sort_index(inplace=True)
test_df = df_full.loc['2024-01-01':].copy()

data = calculate_cci(test_df, length=20)
data = calculate_adx(data, length=14)
data['cci_ma'] = data['cci'].rolling(window=14).mean()

start_idx = 20 + 14 + 1

print("="*80)
print("被ADX过滤器过滤的信号详情".center(80))
print("="*80)

filtered_signals = []
for i in range(start_idx, len(data)-1):
    cci_val = data['cci'].iloc[i]
    cci_ma_val = data['cci_ma'].iloc[i]
    cci_prev = data['cci'].iloc[i-1]
    cci_ma_prev = data['cci_ma'].iloc[i-1]
    adx_val = data['adx'].iloc[i]

    # 金叉但ADX<20
    if cci_prev <= cci_ma_prev and cci_val > cci_ma_val and adx_val <= 20:
        next_date = data.index[i+1]
        next_open = data['open'].iloc[i+1]

        # 计算这笔交易的最终结果
        entry_price = next_open
        holding = True
        exit_price = None
        exit_idx = None

        for j in range(i+1, len(data)-1):
            cci_val_j = data['cci'].iloc[j]
            cci_ma_val_j = data['cci_ma'].iloc[j]
            cci_prev_j = data['cci'].iloc[j-1]
            cci_ma_prev_j = data['cci_ma'].iloc[j-1]

            if cci_prev_j >= cci_ma_prev_j and cci_val_j <= cci_ma_val_j:
                exit_price = data['open'].iloc[j+1]
                exit_idx = j
                holding = False
                break

        if holding:
            # 没有死叉，用最后一天的价格
            exit_price = data['close'].iloc[-1]
            exit_idx = len(data) - 1

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        filtered_signals.append({
            '日期': next_date.strftime('%Y-%m-%d'),
            'ADX': adx_val,
            'CCI': cci_val,
            '入场价': entry_price,
            '出场价': exit_price,
            '收益%': pnl_pct,
            '持仓天数': exit_idx - i
        })

print(f"\n共发现 {len(filtered_signals)} 个被过滤的信号\n")
print("-"*80)
print(f"{'序号':<6} {'日期':<12} {'ADX':<8} {'CCI':<8} {'入场价':<10} {'出场价':<10} {'收益%':<10} {'持仓天数':<10}")
print("-"*80)

for i, sig in enumerate(filtered_signals, 1):
    profit_str = f"{sig['收益%']:+.2f}%"
    if sig['收益%'] > 0:
        profit_str = "\033[92m" + profit_str + "\033[0m"  # 绿色
    else:
        profit_str = "\033[91m" + profit_str + "\033[0m"  # 红色

    print(f"{i:<6} {sig['日期']:<12} {sig['ADX']:<8.2f} {sig['CCI']:<8.2f} "
          f"{sig['入场价']:<10.2f} {sig['出场价']:<10.2f} {profit_str:<16} {sig['持仓天数']:<10}")

# 统计
winning = len([s for s in filtered_signals if s['收益%'] > 0])
losing = len([s for s in filtered_signals if s['收益%'] <= 0])
avg_profit = np.mean([s['收益%'] for s in filtered_signals])
total_profit = sum([s['收益%'] for s in filtered_signals])

print("-"*80)
print(f"统计:")
print(f"  盈利信号: {winning}个 ({winning/len(filtered_signals)*100:.1f}%)")
print(f"  亏损信号: {losing}个 ({losing/len(filtered_signals)*100:.1f}%)")
print(f"  平均收益: {avg_profit:+.2f}%")
print(f"  累计收益: {total_profit:+.2f}%")

print("\n" + "="*80)
