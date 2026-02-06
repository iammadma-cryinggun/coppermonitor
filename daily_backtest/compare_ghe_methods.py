# -*- coding: utf-8 -*-
"""
对比两种GHE计算方法的差异
"""

import pandas as pd
import numpy as np

# 创建测试数据
test_series = pd.Series([
    100, 102, 101, 105, 108, 107, 110, 112, 115, 113,
    118, 120, 119, 122, 125, 123, 120, 118, 122, 125
])

print("="*60)
print("GHE计算方法对比")
print("="*60)

# ========================================
# 方法1：用户提供的方法（使用pandas shift）
# ========================================
def calculate_ghe_method1(series, q=2, max_tau=10):
    """
    用户提供的方法
    特点：使用pandas的shift()函数
    """
    series_log = np.log(series)
    taus = np.arange(1, max_tau + 1)
    Kq = []

    for tau in taus:
        # 使用shift
        diff = np.abs(series_log.shift(-tau) - series_log).dropna()
        Kq.append(np.mean(diff**q))

    log_taus = np.log(taus)
    log_Kq = np.log(Kq)
    h_q = np.polyfit(log_taus, log_Kq, 1)[0] / q

    return h_q, Kq, taus

# ========================================
# 方法2：我之前的方法（使用numpy切片）
# ========================================
def calculate_ghe_method2(series, q=2, max_tau=10):
    """
    我之前的方法
    特点：使用numpy切片
    """
    log_price = np.log(series.values)
    taus = np.arange(2, max_tau + 1)
    Kq = []

    for tau in taus:
        # 使用numpy切片
        diff = np.abs(log_price[tau:] - log_price[:-tau])
        Kq.append(np.mean(diff ** q))

    log_taus = np.log(taus)
    log_Kq = np.log(Kq)
    slope, _ = np.polyfit(log_taus, log_Kq, 1)
    H_q = slope / q

    return H_q, Kq, taus

# ========================================
# 对比测试
# ========================================

print("\n测试序列:")
print(test_series.tolist())

h1, Kq1, taus1 = calculate_ghe_method1(test_series, max_tau=10)
h2, Kq2, taus2 = calculate_ghe_method2(test_series, max_tau=10)

print(f"\n{'方法':<20} {'H值':<10} {'taus范围':<15} {'Kq数量':<10}")
print("-"*60)

print(f"{'方法1 (shift)':<20} {h1:<10.4f} {min(taus1)}-{max(taus1)}<{len(Kq1):<10}")
print(f"{'方法2 (切片)':<20} {h2:<10.4f} {min(taus2)}-{max(taus2)}<{len(Kq2):<10}")

print(f"\n差异: {abs(h1 - h2):.6f}")

# 详细对比
print("\n" + "="*60)
print("详细对比")
print("="*60)

print("\n1. tau的起始点:")
print(f"   方法1: tau从1开始")
print(f"   方法2: tau从2开始")
print(f"   → 影响: tau=1时，相邻数据点差分的意义不大")

print("\n2. diff计算方式:")
print(f"   方法1: series_log.shift(-tau) - series_log")
print(f"   方法2: log_price[tau:] - log_price[:-tau]")
print(f"   → 本质相同，但方法1用了dropna()")

print("\n3. 索引处理:")
print(f"   方法1: 返回Series，需要dropna()")
print(f"   方法2: 返回numpy array，自动对齐")

# ========================================
# 关键差异总结
# ========================================

print("\n" + "="*60)
print("关键差异总结")
print("="*60)

print("""
┌─────────────┬──────────────────┬──────────────────────────────┐
│   方面      │  方法1 (shift)   │  方法2 (切片)               │
├─────────────┼──────────────────┼──────────────────────────────┤
│ tau范围     │  1 到 max_tau    │  2 到 max_tau               │
│ 差分计算     │  shift(-tau)     │  [tau:] vs [:-tau]          │
│ 缺失值处理   │  自动dropna()    │  无缺失值                   │
│ 数据类型     │  pandas Series   │  numpy array               │
│ 代码可读性   │  更易理解        │  更数学化                    │
└─────────────┴──────────────────┴──────────────────────────────┘

论文通常建议：
- tau从2开始（避免tau=1的短期噪音）
- 使用numpy切片（更快，无需处理索引）
- 去除前几个tau的线性拟合（提高稳定性）

实际影响：
- 对于长序列（>100点），差异很小
- 对于短序列（<20点），可能略有不同
""")

# ========================================
# 推荐的改进方法（结合两者优点）
# ========================================

def calculate_ghe_optimized(series, q=2, min_tau=2, max_tau=20):
    """
    优化版GHE计算
    - tau从2开始（避免噪音）
    - 使用numpy切片（性能更好）
    - 返回更多信息（便于调试）
    """
    log_price = np.log(series.values)
    taus = np.arange(min_tau, max_tau + 1)
    Kq = []

    for tau in taus:
        diff = np.abs(log_price[tau:] - log_price[:-tau])
        Kq.append(np.mean(diff ** q))

    log_taus = np.log(taus)
    log_Kq = np.log(Kq)
    slope, intercept = np.polyfit(log_taus, log_Kq, 1)
    H_q = slope / q

    # 计算拟合优度R²
    Kq_pred = np.exp(intercept + slope * log_taus)
    ss_res = np.sum((np.array(Kq) - Kq_pred) ** 2)
    ss_tot = np.sum((np.array(Kq) - np.mean(Kq)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        'H_q': H_q,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'taus': taus,
        'Kq': Kq,
        'log_taus': log_taus,
        'log_Kq': log_Kq
    }

# 测试优化版本
result = calculate_ghe_optimized(test_series, min_tau=2, max_tau=10)

print("\n优化版本结果:")
print(f"  H值: {result['H_q']:.4f}")
print(f"  R²: {result['r_squared']:.4f}")
print(f"  拟合优度: {'好' if result['r_squared'] > 0.9 else '一般'}")

print("\n" + "="*60)
print("结论")
print("="*60)

print("""
两种方法的核心逻辑相同，都是：
1. 计算不同时间间隔的对数增量
2. 计算q阶矩
3. 对log(Kq)和log(tau)做线性回归
4. 斜率/q = H(q)

主要差异：
- tau起始点（1 vs 2）
- 实现方式（pandas vs numpy）

建议：
- 使用方法2（numpy切片），tau从2开始
- 这更符合论文的标准做法
- 对于实际数据，两种方法结果差异很小
""")

print("\n" + "="*60)
