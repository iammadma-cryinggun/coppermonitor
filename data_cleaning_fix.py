"""
数据清洗修复脚本
=================
根据审计报告修复以下问题：
1. OHLC逻辑异常（high < open, low > close）
2. 重新计算ATR
3. 标记极端事件期间
4. 清理无用列
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

print("="*80)
print("期货数据池 - 数据清洗修复".center(80))
print("="*80)

# 加载数据
DATA_PATH = 'data_pool.pkl'
print(f"\n正在读取: {DATA_PATH}")

with open(DATA_PATH, 'rb') as f:
    DATA_POOL = pickle.load(f)

print(f"品种数量: {len(DATA_POOL)}")

# 备份原始数据
backup_path = f'data_pool_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
print(f"\n备份原始数据到: {backup_path}")
with open(backup_path, 'wb') as f:
    pickle.dump(DATA_POOL, f)

# ============ 修复函数 ============
def fix_ohlc_logic(df):
    """修复OHLC逻辑：确保 high >= max(open,close), low <= min(open,close)"""
    df = df.copy()

    # 记录修复前的异常数
    before_high_open = (df['high'] < df['open']).sum()
    before_high_close = (df['high'] < df['close']).sum()
    before_low_open = (df['low'] > df['open']).sum()
    before_low_close = (df['low'] > df['close']).sum()
    total_before = before_high_open + before_high_close + before_low_open + before_low_close

    # 修复high
    df['high'] = df[['high', 'open', 'close']].max(axis=1)

    # 修复low
    df['low'] = df[['low', 'open', 'close']].min(axis=1)

    # 记录修复后
    after_issues = (df['high'] < df['open']).sum() + (df['high'] < df['close']).sum() + \
                   (df['low'] > df['open']).sum() + (df['low'] > df['close']).sum()

    return df, total_before, after_issues

def recalculate_atr(df, period=14):
    """重新计算ATR（使用SMA方法）"""
    df = df.copy()

    # 删除旧的ATR
    if 'atr' in df.columns:
        df = df.drop(columns=['atr'])
    if 'atr_pct' in df.columns:
        df = df = df.drop(columns=['atr_pct'])

    # 计算TR
    df['tr'] = np.maximum(df['high'] - df['low'],
                         np.abs(df['high'] - df['close'].shift(1)))
    df['tr'] = np.maximum(df['tr'], np.abs(df['low'] - df['close'].shift(1)))

    # 计算ATR（SMA）
    df['atr'] = df['tr'].rolling(window=period).mean()

    # 计算ATR百分比
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # 删除临时列
    df = df.drop(columns=['tr'])

    return df

def mark_extreme_events(df, symbol):
    """标记极端事件期间（如镍2022年3月）"""
    df = df.copy()

    # 添加极端事件标记列
    df['extreme_event'] = False

    if symbol == 'ni':
        # 镍2022年3月逼仓事件
        df.loc['2022-03-07':'2022-03-21', 'extreme_event'] = True
        print(f"    ⚠️  标记了2022年3月镍逼仓事件期间")

    return df

def cleanup_columns(df):
    """清理无用列"""
    df = df.copy()

    removed = []
    if 's' in df.columns and (df['s'] == 0).all():
        df = df.drop(columns=['s'])
        removed.append('s')

    if 'settlement' in df.columns and (df['settlement'] == 0).all():
        df = df.drop(columns=['settlement'])
        removed.append('settlement')

    return df, removed

# ============ 主修复流程 ============
print("\n" + "="*80)
print("开始修复数据...")
print("="*80)

FIXED_POOL = {}
fix_summary = []

for symbol in DATA_POOL.keys():
    print(f"\n处理品种: {symbol}")
    df = DATA_POOL[symbol].copy()

    # 1. 修复OHLC逻辑
    df, before, after = fix_ohlc_logic(df)
    print(f"  ✅ OHLC逻辑修复: {before}个异常 → {after}个异常")

    # 2. 重新计算ATR
    df = recalculate_atr(df)
    print(f"  ✅ ATR重新计算完成")

    # 3. 标记极端事件
    df = mark_extreme_events(df, symbol)

    # 4. 清理无用列
    df, removed = cleanup_columns(df)
    if removed:
        print(f"  ✅ 清理无用列: {', '.join(removed)}")

    FIXED_POOL[symbol] = df

    # 记录摘要
    fix_summary.append({
        'symbol': symbol,
        'rows': len(df),
        'ohlc_fixed': before,
        'columns_removed': removed
    })

# ============ 保存修复后的数据 ============
output_path = 'data_pool_fixed.pkl'
print(f"\n保存修复后的数据到: {output_path}")
with open(output_path, 'wb') as f:
    pickle.dump(FIXED_POOL, f)

# ============ 生成修复报告 ============
print("\n" + "="*80)
print("修复摘要")
print("="*80)

summary_df = pd.DataFrame(fix_summary)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("验证修复效果...")
print("="*80)

# 验证是否还有OHLC异常
print("\nOHLC逻辑验证:")
for symbol in FIXED_POOL.keys():
    df = FIXED_POOL[symbol]
    issues = (df['high'] < df['open']).sum() + \
             (df['high'] < df['close']).sum() + \
             (df['low'] > df['open']).sum() + \
             (df['low'] > df['close']).sum()

    status = "[OK]" if issues == 0 else "[FAIL]"
    print(f"  {symbol}: {status} (剩余{issues}个异常)")

print("\n" + "="*80)
print("[SUCCESS] 数据清洗修复完成！")
print("="*80)
print(f"\n原始数据备份: {backup_path}")
print(f"修复后数据: {output_path}")
print("\n建议下一步：")
print("  1. 修改回测脚本，将 'data_pool.pkl' 改为 'data_pool_fixed.pkl'")
print("  2. 重新运行FTT策略回测，对比修复前后的差异")
print("  3. 关注镍2022年3月期间的交易是否需要特殊处理")
print("="*80)
