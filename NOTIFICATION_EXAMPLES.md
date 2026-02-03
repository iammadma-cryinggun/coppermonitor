# 线上部署 - 完整通知示例

## 📱 Telegram通知格式

### 1. 买入信号通知（最详细）

```
🟢 买入信号 - SNIPER

━━━━━━━━━━━━━━━━━━━━

📊 市场状态
• 时间: `2026-02-03 15:00:00`
• 当前价格: `104500`
• 趋势: `UP` (normal)
• 波动率: `3.66%`

📈 技术指标
• Ratio: `0.846` (上一根: `0.791`)
• RSI: `56.5` (中性)
• STC: `33.1` (上一根: `28.2`)
• EMA_Fast: `103075`
• EMA_Slow: `102373`

━━━━━━━━━━━━━━━━━━━━

💰 交易计划

📍 开仓信息
• 入场价格: `104500`
• 建议仓位: `1.2x`
• 信号类型: `sniper` (狙击点)

🛡️ 风险控制
• 止损价格: `102410` (`2.00%`)
• 止损金额: `2088` 点/手

🎯 止盈目标
• 第一目标: `107632` (`3.00%`) ← 建议50%仓位
• 第二目标: `109500` (`4.78%`) ← 建议30%仓位
• 第三目标: `113318` (`8.43%`) ← 剩余20%

📊 风险收益比
• 风险: `2090` 点 (`2.00%`)
• 第一目标收益: `3132` 点 (`3.00%`)
• 盈亏比: `1:1.5`

━━━━━━━━━━━━━━━━━━━━

🔍 技术分析
• 趋势强度: `中等 ➡️`
• 信号质量: `良好 ⭐⭐`
• 狙击点入场：Ratio回缩+趋势向上+RSI强势，可靠性较高

━━━━━━━━━━━━━━━━━━━━

_数据源: API_ | _生成时间: 2026-02-03 19:15:00_

---
🤖 沪铜策略实盘监控
⚠️ 风险提示：仅供参考，实际交易请结合市场情况
```

---

### 2. 卖出信号通知（平仓结算）

```
🔴 卖出信号 - TREND

━━━━━━━━━━━━━━━━━━━━

💼 平仓信息

📍 交易结果
• 入场价格: `82610`
• 出场价格: `104500`
• 仓位大小: `1.5x`
• 持仓天数: `132` 天

💰 盈亏结算
✅ • 盈亏: `52830` 点
✅ • 盈亏率: `+42.72%`
✅ • 状态: `盈利`

━━━━━━━━━━━━━━━━━━━━

📊 当前市场状态
• 时间: `2026-02-03 15:00:00`
• 价格: `104500`
• 趋势: `UP` (weak)
• Ratio: `0.846`
• RSI: `56.5`
• STC: `33.1`

🔔 卖出原因
• 信号类型: `trend`
• 趋势反转，EMA死叉

━━━━━━━━━━━━━━━━━━━━

_数据源: API_ | _生成时间: 2026-02-03 19:15:00_

---
🤖 沪铜策略实盘监控
```

---

### 3. 监控更新通知（无信号）

```
⚪ 市场监控更新

━━━━━━━━━━━━━━━━━━━━

📊 当前市场状态
• 时间: `2026-02-03 15:00:00`
• 价格: `104500`
• 趋势: `UP` (weak)
• 波动率: `3.66%`

📈 技术指标
• Ratio: `0.846` (上一根: `0.791`)
  └─ 📈 上升
• RSI: `56.5` (中性)
• STC: `33.1` (上一根: `28.2`)
  └─ 📈 上升
• EMA_Fast: `103075`
• EMA_Slow: `102373`
  └─ 金叉 🟢

━━━━━━━━━━━━━━━━━━━━

💼 当前持仓: 空仓

━━━━━━━━━━━━━━━━━━━━

🔍 市场分析
• 趋势强度: `中等 ➡️`
• 信号状态: `趋势向上 🟢 | Ratio安全 ✅ | RSI强势 ✅`
• 操作建议: `等待Ratio回缩后的买入机会`

━━━━━━━━━━━━━━━━━━━━

_数据源: API_ | _生成时间: 2026-02-03 19:15:00_

---
🤖 沪铜策略实盘监控
```

---

### 4. 有持仓时的监控更新

```
⚪ 市场监控更新

━━━━━━━━━━━━━━━━━━━━

📊 当前市场状态
• 时间: `2026-02-03 15:00:00`
• 价格: `104500`
• 趋势: `UP` (weak)
• 波动率: `3.66%`

📈 技术指标
• Ratio: `0.846` (上一根: `0.791`)
  └─ 📈 上升
• RSI: `56.5` (中性)
• STC: `33.1` (上一根: `28.2`)
  └─ 📈 上升
• EMA_Fast: `103075`
• EMA_Slow: `102373`
  └─ 金叉 🟢

━━━━━━━━━━━━━━━━━━━━

💼 当前持仓
• 入场价: `100000`
• 当前价: `104500`
• 仓位: `1.5x`
• 持仓天数: `15` 天
• 止损价: `98000`
• 浮动盈亏: `+6750` 点 (`+4.50%`)
✅ 盈利

━━━━━━━━━━━━━━━━━━━━

🔍 市场分析
• 趋势强度: `中等 ➡️`
• 信号状态: `趋势向上 🟢 | Ratio安全 ✅ | RSI强势 ✅`
• 操作建议: `持仓中，继续持有`

━━━━━━━━━━━━━━━━━━━━

_数据源: API_ | _生成时间: 2026-02-03 19:15:00_

---
🤖 沪铜策略实盘监控
```

---

## 🔧 环境变量配置（线上部署）

### Linux/Mac

```bash
# 临时设置（当前终端会话）
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="987654321"

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export TELEGRAM_BOT_TOKEN="your_token"' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="your_chat_id"' >> ~/.bashrc
source ~/.bashrc

# 运行监控
python copper_monitor.py
```

### Windows (CMD)

```cmd
REM 临时设置
set TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
set TELEGRAM_CHAT_ID=987654321

REM 运行监控
python copper_monitor.py

REM 永久设置（系统环境变量）
REM 1. 右键"此电脑" -> 属性 -> 高级系统设置 -> 环境变量
REM 2. 新建用户变量:
REM    TELEGRAM_BOT_TOKEN = your_token
REM    TELEGRAM_CHAT_ID = your_chat_id
```

### Windows (PowerShell)

```powershell
# 临时设置
$env:TELEGRAM_BOT_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
$env:TELEGRAM_CHAT_ID = "987654321"

# 运行监控
python copper_monitor.py

# 永久设置（系统环境变量）
[System.Environment]::SetEnvironmentVariable('TELEGRAM_BOT_TOKEN', 'your_token', 'User')
[System.Environment]::SetEnvironmentVariable('TELEGRAM_CHAT_ID', 'your_chat_id', 'User')
```

---

## 🐳 Docker部署（完整示例）

### docker-compose.yml

```yaml
version: '3.8'

services:
  copper-monitor:
    build: .
    container_name: copper_monitor
    restart: unless-stopped

    environment:
      # Telegram配置（从.env读取）
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - TZ=Asia/Shanghai

    # 定时任务：每4小时运行一次
    command: >
      sh -c "
      echo '0 */4 * * * cd /app && python copper_monitor.py' > /tmp/crontab &&
      crond -f -l 2
      "

    volumes:
      - ./logs:/app/logs

    # 时区同步
    timezone: Asia/Shanghai
```

### .env文件

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
TZ=Asia/Shanghai
```

### 启动命令

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

---

## ⚙️ GitHub Actions部署

### .github/workflows/monitor.yml

```yaml
name: Copper Futures Monitor

on:
  schedule:
    # 每4小时运行一次
    - cron: '0 */4 * * *'
  workflow_dispatch:  # 支持手动触发

jobs:
  monitor:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install pandas numpy akshare requests

    - name: Run monitor
      env:
        # 从GitHub Secrets读取
        TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
        TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      run: |
        python copper_monitor.py

    - name: Upload logs
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: logs-$(date +%Y%m%d-%H%M%S)
        path: logs/
```

### 配置GitHub Secrets

1. 进入GitHub仓库
2. Settings → Secrets and variables → Actions
3. 点击 "New repository secret"
4. 添加以下Secrets:
   - Name: `TELEGRAM_BOT_TOKEN`, Value: `your_bot_token`
   - Name: `TELEGRAM_CHAT_ID`, Value: `your_chat_id`

---

## 📊 通知内容说明

### 买入信号包含：

✅ **市场状态**
- 当前价格、趋势、波动率
- Ratio、RSI、STC等关键指标
- EMA快线和慢线

✅ **交易计划**
- 入场价格、建议仓位
- 信号类型（狙击点/追涨）

✅ **风险控制**
- 止损价格、止损百分比
- 止损金额（点/手）

✅ **止盈目标**
- 三个目标价位
- 每个目标的百分比收益
- 分批止盈建议（50%、30%、20%）

✅ **风险收益比**
- 风险点数和百分比
- 收益点数和百分比
- 盈亏比（如1:1.5）

✅ **技术分析**
- 趋势强度（极强/强势/中等/弱势/极弱）
- 信号质量（优秀/良好/一般）
- 交易建议（狙击点/追涨）

---

### 卖出信号包含：

✅ **平仓信息**
- 入场价、出场价
- 仓位大小、持仓天数

✅ **盈亏结算**
- 盈亏点数
- 盈亏百分比
- 盈利/亏损状态

✅ **卖出原因**
- 信号类型（STC/Trend/止损）
- 详细原因说明

---

### 监控更新包含：

✅ **市场状态**
- 价格、趋势、波动率
- Ratio变化方向
- RSI状态（超买/超卖/中性）
- STC变化方向
- EMA金叉/死叉状态

✅ **持仓信息**（如有）
- 入场价、当前价
- 仓位、持仓天数
- 止损价
- 浮动盈亏

✅ **市场分析**
- 趋势强度
- 信号状态检查
- 操作建议

---

## 🎯 关键特性总结

1. **环境变量优先**：支持环境变量和配置文件两种方式
2. **详细通知**：开仓、止损、止盈、持仓盈亏全覆盖
3. **分批止盈**：提供3个目标价位和分批建议
4. **风险收益比**：自动计算盈亏比
5. **技术分析**：趋势强度、信号质量评估
6. **持仓跟踪**：实时显示浮动盈亏
7. **操作建议**：根据市场状态给出具体建议

---

## ✅ 部署检查清单

部署前请确认：

- [ ] 已获取Telegram Bot Token
- [ ] 已获取Chat ID
- [ ] 已配置环境变量或配置文件
- [ ] 已测试通知发送（`python notifier.py`）
- [ ] 已测试监控运行（`python copper_monitor.py`）
- [ ] 已配置定时任务或Docker
- [ ] 已验证通知内容格式
- [ ] 已设置日志持久化

---

**系统已完全支持线上部署，环境变量配置优先，通知内容详细完整！** 🚀
