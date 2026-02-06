import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import timedelta
import warnings
import sys
import io
warnings.filterwarnings('ignore')

# 修复Windows控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 数据预处理 ---
df = pd.read_csv('D:/期货数据/铜期货监控/daily_backtest/SN_沪锡_日线_10年.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')
df = df.set_index('datetime')

# 计算趋势指标 (MA60)
df['MA60'] = df['收盘价'].rolling(window=60).mean()

# --- 2. 定义 LPPL 核心函数 ---
def lppl_func(t, A, B, tc, m, C, omega, phi):
    dt = tc - t
    dt[dt <= 0] = 1e-8
    return A + B * (dt ** m) * (1 + C * np.cos(omega * np.log(dt) + phi))

def check_lppl_risk(price_series):
    """
    对给定的价格序列进行一次 LPPL 诊断
    返回: (is_risk, days_to_crash, params)
    """
    t_data = np.arange(len(price_series))
    log_price = np.log(price_series.values)
    last_day = t_data[-1]

    # 初始猜测
    p0 = [log_price.max(), -0.1, last_day + 20, 0.5, 0.05, 10.0, 0.0]
    # 宽松的约束
    bounds = (
        [-np.inf, -np.inf, last_day, 0.1, -1, 4.0, -2*np.pi],
        [np.inf, 0, last_day + 60, 1.5, 1, 15.0, 2*np.pi]
    )

    try:
        popt, _ = curve_fit(lppl_func, t_data, log_price, p0=p0, bounds=bounds, maxfev=2000)
        tc, m, omega = popt[2], popt[3], popt[5]

        days_to_crash = tc - last_day
        if (5 < days_to_crash < 60) and (0.1 < m < 0.95) and (4.0 < omega < 15.0):
            return True, days_to_crash, popt
        else:
            return False, 0, popt
    except Exception as e:
        return False, 0, None

# --- 3. 滚动回测 (Rolling Scan) ---
print("=" * 80)
print("趋势+风险双模系统：MA60趋势判断 + LPPL泡沫预警")
print("=" * 80)
print("\n开始滚动扫描 LPPL 风险信号...")

risk_signals = []
safe_periods = []
scan_dates = []

# 扫描最近 3 年的数据
scan_start_date = df.index[-750]
subset_df = df[df.index >= scan_start_date].copy()

window_size = 200
step = 5

for i in range(0, len(subset_df) - window_size, step):
    window_data = subset_df.iloc[i : i + window_size]
    current_date = window_data.index[-1]

    is_risk, days_left, params = check_lppl_risk(window_data['收盘价'])

    if is_risk:
        risk_signals.append({
            'date': current_date,
            'days_to_crash': days_left,
            'price': window_data['收盘价'].iloc[-1],
            'ma60': window_data['MA60'].iloc[-1],
            'params': params
        })

    if i % 50 == 0:
        print(f"扫描进度: {current_date.date()}...")

print(f"\n扫描完成！")
print(f"共检测到 {len(risk_signals)} 个LPPL风险信号")

# --- 4. 统计分析 ---
print("\n" + "=" * 80)
print("LPPL风险信号统计分析")
print("=" * 80)

if risk_signals:
    # 分析风险信号与趋势的关系
    trend_up_risks = [s for s in risk_signals if s['price'] > s['ma60']]
    trend_down_risks = [s for s in risk_signals if s['price'] <= s['ma60']]

    print(f"\n在上升趋势中(价格>MA60)的风险信号: {len(trend_up_risks)} 个")
    print(f"在下降趋势中(价格<=MA60)的风险信号: {len(trend_down_risks)} 个")

    # 找出密集预警区
    if len(risk_signals) > 0:
        dates = [s['date'] for s in risk_signals]
        print(f"\n最早预警日期: {min(dates).date()}")
        print(f"最晚预警日期: {max(dates).date()}")

        # 预警密度分析
        from collections import Counter
        # 按月统计
        monthly_counts = Counter([d.strftime('%Y-%m') for d in dates])
        top_months = monthly_counts.most_common(5)
        print(f"\n预警最密集的月份:")
        for month, count in top_months:
            print(f"  {month}: {count} 次")

        # 检查最近几个月的预警情况
        recent_month = df.index[-1].strftime('%Y-%m')
        recent_count = monthly_counts.get(recent_month, 0)
        print(f"\n{recent_month}月的预警次数: {recent_count}")

# --- 5. 绘图 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[2, 1])

# A. 主图：价格 + 趋势 + LPPL预警
ax1.plot(subset_df.index, subset_df['收盘价'], label='沪锡价格', color='black', alpha=0.6, linewidth=1)
ax1.plot(subset_df.index, subset_df['MA60'], label='MA60趋势线', color='blue', linewidth=1.5)

# 绿色背景 = 趋势向上
trend_up = subset_df['收盘价'] > subset_df['MA60']
ax1.fill_between(subset_df.index, subset_df['收盘价'].min(), subset_df['收盘价'].max(),
                 where=trend_up, color='green', alpha=0.1, label='多头安全区')

# 红色竖线 = LPPL风险信号
for signal in risk_signals:
    ax1.axvline(x=signal['date'], color='red', alpha=0.3, linewidth=2)

if risk_signals:
    ax1.axvline(x=risk_signals[0]['date'], color='red', alpha=0.3, linewidth=2, label='LPPL泡沫预警')

ax1.set_title('沪锡：MA60趋势跟踪 + LPPL泡沫风险预警系统', fontsize=14, fontweight='bold')
ax1.set_ylabel('价格', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# B. 副图：风险密度
# 计算每个月的风险信号数量
if risk_signals:
    risk_df = pd.DataFrame([{'date': s['date'], 'risk': 1} for s in risk_signals])
    risk_df = risk_df.set_index('date')
    risk_monthly = risk_df.resample('M').count()

    ax2.bar(risk_monthly.index, risk_monthly['risk'], color='red', alpha=0.5, label='月度风险信号次数')
    ax2.axhline(y=risk_monthly['risk'].mean(), color='orange', linestyle='--', label='平均风险水平')
    ax2.set_title('LPPL风险信号密度分布', fontsize=12)
    ax2.set_ylabel('信号次数', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('D:/期货数据/铜期货监控/daily_backtest/ma60_lppl_analysis.png', dpi=150, bbox_inches='tight')
print("\n图表已保存到: ma60_lppl_analysis.png")
plt.show()

# --- 6. 关键发现总结 ---
print("\n" + "=" * 80)
print("关键发现总结")
print("=" * 80)

if risk_signals:
    # 找出"绿地红旗"的典型时刻（趋势向上但LPPL预警）
    green_red_flags = [s for s in risk_signals if s['price'] > s['ma60']]

    if green_red_flags:
        print(f"\n[WARNING] '绿地红旗'信号（趋势向上但有泡沫风险）: {len(green_red_flags)} 次")
        print("\n最近5次'绿地红旗'信号:")
        for s in green_red_flags[-5:]:
            print(f"  {s['date'].date()} - 价格: {s['price']:.2f}, MA60: {s['ma60']:.2f}, "
                  f"预计崩盘: {s['days_to_crash']:.0f}天后")

    # 当前状态判断
    latest_price = df['收盘价'].iloc[-1]
    latest_ma60 = df['MA60'].iloc[-1]
    latest_date = df.index[-1]

    print(f"\n当前市场状态 ({latest_date.date()}):")
    print(f"  价格: {latest_price:.2f}")
    print(f"  MA60: {latest_ma60:.2f}")
    if latest_price > latest_ma60:
        print(f"  趋势: [UP] 上升趋势（价格>MA60）")
    else:
        print(f"  趋势: [DOWN] 下降趋势（价格<=MA60）")

    # 检查最近30天是否有LPPL预警
    recent_risks = [s for s in risk_signals if (latest_date - s['date']).days <= 30]
    if recent_risks:
        print(f"  LPPL风险: [WARNING] 最近30天检测到 {len(recent_risks)} 次泡沫预警")
    else:
        print(f"  LPPL风险: [SAFE] 最近30天无泡沫预警")

print("\n" + "=" * 80)
