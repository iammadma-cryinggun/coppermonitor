# 沪铜期货实盘监控系统

## 📋 系统概述

实盘监控系统用于沪铜期货(CU)策略的自动化监控和通知。

**主要功能**:
1. ✅ 自动获取实时行情数据（AkShare API）
2. ✅ 运行策略生成交易信号
3. ✅ 发送Telegram通知（每4小时）
4. ✅ 记录所有买卖建议到日志
5. ✅ 跟踪策略表现 vs 实际表现

---

## 🚀 快速开始

### 1. 目录结构

```
D:\期货数据\铜期货监控\
├── copper_monitor.py         # 主监控脚本
├── china_futures_fetcher.py  # 数据获取模块
├── notifier.py               # Telegram通知模块
├── analyze.py                # 分析工具
├── run_monitor.bat           # 手动运行脚本
├── setup_task.bat            # 创建定时任务
├── logs/                     # 日志目录
│   ├── signal_log.json       # 信号日志
│   ├── position_status.json  # 持仓状态
│   ├── performance_tracking.csv  # 监控记录
│   └── strategy_monitor.log  # 运行日志
└── config/                   # 配置目录
    └── telegram.json         # Telegram配置
```

### 2. 配置Telegram通知

#### 步骤1: 创建Telegram Bot

1. 在Telegram中搜索 `@BotFather`
2. 发送 `/newbot` 命令
3. 按提示设置bot名称
4. 保存返回的 **Bot Token**（格式：`123456789:ABCdefGHIjklMNOpqrsTUVwxyz`）

#### 步骤2: 获取Chat ID

1. 在Telegram中搜索你的bot（用刚创建的用户名）
2. 发送任意消息给bot（如 `/start`）
3. 在浏览器访问：
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
   将 `<YOUR_BOT_TOKEN>` 替换为你的Token
4. 在返回的JSON中找到 `chat.id`（纯数字，如 `123456789`）

#### 步骤3: 填写配置文件

编辑 `config/telegram.json`:

```json
{
  "token": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
  "chat_id": "123456789"
}
```

#### 步骤4: 测试通知

```bash
cd "D:\期货数据\铜期货监控"
python notifier.py
```

如果配置正确，你会收到测试消息。

---

## ⏰ 设置定时任务

### 方法1: 自动创建（推荐）

1. 右键点击 `setup_task.bat`
2. 选择"以管理员身份运行"
3. 等待任务创建完成

任务将：
- 每4小时运行一次
- 时间点: 09:05, 13:05, 17:05, 21:05
- 使用SYSTEM账户运行

### 方法2: 手动创建

1. 按 `Win + R`，输入 `taskschd.msc` 打开任务计划程序
2. 右侧点击"创建基本任务"
3. 名称: `沪铜监控`
4. 触发器: 每4小时运行一次
5. 操作: 运行 `D:\期货数据\铜期货监控\run_monitor.bat`

---

## 📊 使用方法

### 手动运行监控

```bash
# 方法1: 双击批处理文件
run_monitor.bat

# 方法2: 命令行运行
cd "D:\期货数据\铜期货监控"
python copper_monitor.py
```

### 查看分析报告

```bash
cd "D:\期货数据\铜期货监控"
python analyze.py
```

输出示例：
```
================================================================================
沪铜策略 - 实盘跟踪分析
================================================================================

[1] 信号统计
--------------------------------------------------------------------------------
总信号数: 10
已执行: 8 (80.0%)
待处理: 1
已忽略: 1

买入信号: 6
卖出信号: 4

[2] 当前持仓
--------------------------------------------------------------------------------
持仓状态: 有持仓
  入场价: 82610
  入场时间: 2025-09-25 01:00:00
  仓位: 1.5x
  止损价: 80957
  持有天数: 132 天

[3] 监控记录
--------------------------------------------------------------------------------
监控次数: 156
时间范围: 2025-08-01 15:00:00 ~ 2026-02-03 15:00:00
```

---

## 📱 Telegram通知格式

### 有信号时

```
🟢 买入信号

📊 市场状态
• 时间: `2026-02-03 15:00:00`
• 价格: `104500`
• 趋势: `UP (normal)`
• Ratio: `0.846` (上一根: `0.791`)
• RSI: `56.5`
• STC: `33.1` (上一根: `28.2`)
• 波动率: `3.66%`

🟢 买入信号
• 类型: `sniper`
• 建议仓位: `1.2x`
• 止损价格: `102410`

💼 当前持仓: 空仓

_数据源: API_
---
🤖 沪铜策略实盘监控
```

### 无信号时

```
⚪ 监控报告

📊 市场状态
• 时间: `2026-02-03 15:00:00`
• 价格: `104500`
• 趋势: `UP (weak)`
...

📭 无交易信号，继续观望

---
🤖 沪铜策略实盘监控
```

---

## ⚙️ 配置说明

### 策略参数

编辑 `copper_monitor.py` 中的配置：

```python
# 策略参数（已优化）
EMA_FAST = 5
EMA_SLOW = 15
MACD_FAST = 12
MACD_SLOW = 26
RSI_FILTER = 45
RATIO_TRIGGER = 1.15
STC_SELL_ZONE = 85
STOP_LOSS_PCT = 0.02  # 止损2%

# 数据配置
FUTURES_CODE = 'CU'        # 沪铜
HISTORICAL_DAYS = 300      # 获取300天历史数据
```

### 通知配置

编辑 `config/telegram.json`:

```json
{
  "token": "YOUR_BOT_TOKEN",
  "chat_id": "YOUR_CHAT_ID"
}
```

---

## 📁 日志文件

### 1. 信号日志 `logs/signal_log.json`

记录所有交易信号：
```json
[
  {
    "timestamp": "2026-02-03T19:00:00",
    "signal_datetime": "2026-02-03 15:00:00",
    "signal": {...},
    "status": "pending",
    "actual_price": null,
    "actual_action": null,
    "notes": ""
  }
]
```

**状态**:
- `pending`: 待处理
- `executed`: 已执行
- `ignored`: 已忽略

### 2. 持仓状态 `logs/position_status.json`

当前持仓状态：
```json
{
  "holding": true,
  "entry_price": 82610,
  "entry_datetime": "2025-09-25 01:00:00",
  "position_size": 1.5,
  "stop_loss": 80957,
  "signal_id": "2025-09-25_01:00:00_chase"
}
```

### 3. 监控记录 `logs/performance_tracking.csv`

每次运行的完整记录：
```csv
datetime,price,ratio,rsi,stc,trend,buy_signal,sell_signal,holding,position_size,data_source,timestamp
2026-02-03 15:00:00,104500,0.85,56.5,33.1,up,False,False,False,0.0,API,2026-02-03 19:00:00
```

---

## 🔧 故障排查

### 1. Telegram通知未收到

**检查**:
- 配置文件是否正确填写（`config/telegram.json`）
- Bot是否已启动（发送 `/start` 给bot）
- Chat ID是否正确

**测试**:
```bash
python notifier.py
```

### 2. 定时任务未运行

**检查**:
- 任务是否已创建：`taskschd.msc` 查找"沪铜监控"
- 计算机是否在运行时间点开机
- 查看日志文件：`logs/strategy_monitor.log`

**手动运行**:
```bash
run_monitor.bat
```

### 3. 数据获取失败

**原因**:
- 网络连接问题
- AkShare API限流

**解决**:
- 系统会自动降级到CSV备用数据
- 检查日志文件查看详细错误

---

## 📈 性能优化建议

### 1. 运行时间

建议在K线收盘后5分钟运行：
- 09:05（09:00 K线收盘）
- 13:05（13:00 K线收盘）
- 17:05（17:00 K线收盘）
- 21:05（21:00 K线收盘）

### 2. 资金管理

即使策略给出2.0x仓位建议：
- 新手建议最多1.0x
- 有经验后可逐步提高到1.5x
- 2.0x仅用于超强趋势且风险可控时

### 3. 信号执行

- 收到Telegram通知后，先检查市场情况
- 使用限价单，避免市价单滑点
- 如果滑点>0.2%，考虑不执行
- 执行后更新 `logs/position_status.json`

---

## 📞 支持与帮助

### 查看日志

```bash
# Windows
notepad logs\strategy_monitor.log

# 或在PowerShell
Get-Content logs\strategy_monitor.log -Tail 50 -Wait
```

### 常用命令

```bash
# 删除定时任务
schtasks /Delete /TN "沪铜监控" /F

# 查看任务信息
schtasks /Query /TN "沪铜监控" /FO LIST /V

# 立即运行任务
schtasks /Run /TN "沪铜监控"
```

---

## 📝 更新日志

### v1.0 (2026-02-03)
- ✅ 实时API数据获取
- ✅ Telegram通知集成
- ✅ 自动定时任务
- ✅ 完整日志记录
- ✅ 策略逻辑验证（无未来函数）

---

**系统已就绪，可用于实盘监控！** 🚀
