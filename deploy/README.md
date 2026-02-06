# 沪铜期货实盘监控系统

## 📋 功能特性

1. ✅ **数据获取** - AkShare API实时获取4小时K线数据
2. ✅ **技术指标** - EMA, MACD, RSI, STC自动计算
3. ✅ **交易信号** - Sniper信号 + Chase信号
4. ✅ **持仓管理** - 自动保存/清除持仓状态
5. ✅ **止损止盈** - 固定1.5%止损 + STC/趋势止盈
6. ✅ **Telegram通知** - 实时推送交易信号

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
python copper_monitor.py
```

## 📂 文件说明

- **copper_monitor.py** - 主程序（监控核心逻辑）
- **china_futures_fetcher.py** - 数据获取模块
- **notifier.py** - Telegram通知模块
- **requirements.txt** - Python依赖包
- **telegram.json.example** - 配置模板

## ⏰ 定时运行

建议每4小时运行一次：
- 09:05, 13:05, 17:05, 21:05

## 📊 日志文件

程序会在 `logs/` 目录自动创建：
- **signal_log.json** - 所有交易信号记录
- **position_status.json** - 当前持仓状态
- **strategy_monitor.log** - 运行日志

## ⚙️ 策略参数

**买入信号：**
- Sniper信号：Ratio收缩 + 趋势向上 + RSI强势
- Chase信号：EMA金叉 + RSI强势

**卖出信号：**
- 止损：价格跌破入场价1.5%
- STC止盈：STC从85以上回落
- 趋势反转：EMA死叉

## ⚠️ 注意事项

1. 本程序仅供参考，不构成投资建议
2. 实际交易请结合市场情况
3. 定期检查日志文件

---

**版本**: 1.0
**更新日期**: 2026-02-06
**Python版本**: 3.8+
