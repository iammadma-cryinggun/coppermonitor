# 期货多品种实盘监控系统

## 📋 功能特性

1. ✅ **多品种监控** - 同时监控7个优质期货品种
2. ✅ **独立参数** - 每个品种使用最优策略参数
3. ✅ **数据获取** - AkShare API实时获取4小时K线数据
4. ✅ **技术指标** - EMA, MACD, RSI, STC自动计算
5. ✅ **交易信号** - Sniper信号 + Chase信号
6. ✅ **持仓管理** - 自动保存/清除持仓状态（每品种独立）
7. ✅ **止损止盈** - 固定止损 + STC/趋势止盈
8. ✅ **Telegram通知** - 实时推送交易信号

## 🎯 监控品种（TOP 7）

1. **沪镍 (NI)** - 81.0分
2. **纯碱 (SA)** - 78.6分
3. **PVC (V)** - 78.3分
4. **沪铜 (CU)** - 77.0分
5. **沪锡 (SN)** - 76.2分
6. **沪铅 (PB)** - 73.3分
7. **玻璃 (FG)** - 71.9分

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置Telegram（可选）

```bash
# 复制配置模板
cp telegram.json.example telegram.json

# 编辑telegram.json，填入你的Bot Token和Chat ID
```

### 3. 运行程序

```bash
# 单次运行
python futures_monitor.py

# 定时运行（每4小时K线收盘后自动运行）
python futures_monitor.py --scheduled
```

## 📂 文件说明

- **futures_monitor.py** - 多品种主程序（7品种监控）
- **copper_monitor.py** - 单品种示例程序
- **china_futures_fetcher.py** - 数据获取模块
- **notifier.py** - Telegram通知模块
- **requirements.txt** - Python依赖包
- **telegram.json.example** - 配置模板

## ⏰ 定时运行

定时运行模式会自动在以下时间运行（确保K线数据已更新）：
- **01:30** - 夜盘收盘后
- **11:30** - 早盘收盘后
- **15:30** - 午盘收盘后
- **21:30** - 夜盘开始后

## 📊 日志文件

程序会在 `logs/` 目录自动创建：
- **multi_positions.json** - 所有品种持仓状态
- **multi_signals.json** - 所有交易信号记录
- **multi_tracking.csv** - 监控追踪记录
- **multi_replay_data.csv** - 详细复盘数据（OHLC+指标）
- **multi_monitor.log** - 运行日志

## ⚙️ 策略参数

**买入信号：**
- Sniper信号：Ratio收缩 + 趋势向上 + RSI强势
- Chase信号：EMA金叉 + RSI强势

**卖出信号：**
- 止损：价格跌破止损价（品种参数设定）
- STC止盈：STC从设定值以上回落
- 趋势反转：EMA死叉

## ⚠️ 注意事项

1. 本程序仅供参考，不构成投资建议
2. 实际交易请结合市场情况
3. 定期检查日志文件和持仓状态
4. 每个品种使用独立的最优参数

---

**版本**: 2.0
**更新日期**: 2026-02-06
**Python版本**: 3.8+
**监控品种**: 7个
