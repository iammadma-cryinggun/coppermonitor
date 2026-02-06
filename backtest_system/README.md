# 期货策略完整回测验证系统

## 📁 目录结构

```
backtest_system/
├── data/               # 原始4小时K线数据
├── programs/           # 回测和优化程序
├── results/            # 回测结果输出
├── logs/               # 运行日志
├── config/             # 配置文件
└── run_backtest.py     # 主程序入口
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy akshare
```

### 2. 查看数据

```bash
ls data/
```

包含22个期货品种的4小时K线数据：
- 沪铜、沪铝、沪锌、沪镍、沪铅、沪锡
- 螺纹钢、热卷、铁矿石、焦煤、焦炭
- 甲醇、PP、PVC、纯碱、玻璃
- 豆粕、豆油、棕榈油、棉花、白糖、PTA

### 3. 运行完整回测验证

```bash
# 方式1: 使用主程序入口
python run_backtest.py

# 方式2: 直接运行优化程序
cd programs
python optimize_all_futures.py
```

## 📊 程序说明

### programs/optimize_all_futures.py
**功能**: 对所有22个品种进行参数优化

**参数空间**: 4^5 = 1024种组合
- EMA_FAST: [3, 5, 7, 10]
- EMA_SLOW: [10, 15, 20, 25]
- RSI_FILTER: [35, 40, 45, 50]
- RATIO_TRIGGER: [1.05, 1.10, 1.15, 1.20]
- STC_SELL_ZONE: [75, 80, 85, 90]

**输出**: results/{品种名}_full_results.csv

### programs/verify_optimized_results.py
**功能**: 验证优化后的参数效果

对比优化前后的收益率、胜率、最大回撤等指标

### programs/backtest_all_futures.py
**功能**: 使用统一参数回测所有品种

可以测试特定参数组合在所有品种上的表现

### programs/backtest.py
**功能**: 单品种基础回测程序

用于测试单个品种的策略表现

## 📈 输出结果

### results/ 目录

运行后会生成以下文件：

1. **{品种名}_full_results.csv** - 每个品种的详细优化结果
2. **all_futures_summary.csv** - 所有品种汇总结果
3. **backtest_report.txt** - 回测报告
4. **optimization_log.txt** - 优化日志

### 关键指标

- **总收益率** - 策略的累计收益率
- **年化收益率** - 折算成年化的收益率
- **胜率** - 盈利交易占比
- **最大回撤** - 最大亏损幅度
- **夏普比率** - 风险调整后收益
- **交易次数** - 总交易次数

## ⚙️ 配置文件

### config/parameters.json
策略默认参数配置

```json
{
  "EMA_FAST": 3,
  "EMA_SLOW": 20,
  "RSI_FILTER": 40,
  "RATIO_TRIGGER": 1.2,
  "STC_SELL_ZONE": 85,
  "STOP_LOSS_PCT": 0.02
}
```

## 📝 使用流程

### 完整验证流程

1. **参数优化**
   ```bash
   python programs/optimize_all_futures.py
   ```

2. **验证结果**
   ```bash
   python programs/verify_optimized_results.py
   ```

3. **对比分析**
   ```bash
   python programs/backtest_all_futures.py
   ```

4. **查看报告**
   ```bash
   cat results/backtest_report.txt
   ```

### 快速测试单个品种

```python
from programs.backtest import backtest_single_future

# 测试沪铜
result = backtest_single_future(
    data_file='data/沪铜_4hour.csv',
    params={
        'EMA_FAST': 3,
        'EMA_SLOW': 20,
        'RSI_FILTER': 40,
        'RATIO_TRIGGER': 1.2,
        'STC_SELL_ZONE': 85
    }
)
print(result)
```

## ⚠️ 注意事项

1. **数据质量**
   - 所有数据已修正时间戳（01:00, 09:00, 13:00, 21:00）
   - 数据来自 AkShare API
   - 部分品种有 .bak 备份文件

2. **参数说明**
   - 参数空间可根据需要调整
   - 固定参数：MACD(12,26,9), RSI(14), STC(10,23,50)
   - 止损默认为 2%

3. **性能优化**
   - 大规模参数搜索可能需要较长时间
   - 建议先用小范围参数空间测试
   - 可使用 multiprocessing 加速

## 📊 历史优化结果

### TOP 7 品种（按信号质量）

1. 沪镍 (NI) - 81.0分
2. 纯碱 (SA) - 78.6分
3. PVC (V) - 78.3分
4. 沪铜 (CU) - 77.0分
5. 沪锡 (SN) - 76.2分
6. 沪铅 (PB) - 73.3分
7. 玻璃 (FG) - 71.9分

---

**版本**: 1.0
**创建日期**: 2026-02-06
**数据范围**: 4小时K线
**品种数量**: 22个
