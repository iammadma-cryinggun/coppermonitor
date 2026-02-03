# Zeabur部署说明

## 快速开始

1. **Fork本仓库到你的GitHub账号**

2. **在Zeabur上部署**
   - 访问 https://zeabur.com/dashboard
   - 点击 "New Service"
   - 选择 "Import from Git"
   - 输入你的仓库地址

3. **配置Secrets（必需）**
   - 在Zeabur控制台中点击服务
   - 进入 "Variables" 或 "Secrets" 设置
   - 添加以下两个Secrets：
     ```
     TELEGRAM_BOT_TOKEN = 你的Bot Token
     TELEGRAM_CHAT_ID = 你的Chat ID
     ```

4. **部署**
   - 点击 "Deploy"
   - 等待部署完成（约1-2分钟）

## 配置文件说明

- `zeabur.yml` - Zeabur主配置文件（服务类型、环境变量、构建命令）
- `requirements.txt` - Python依赖列表
- `run_continuous.py` - 持续运行脚本（每4小时自动运行）
- `copper_monitor.py` - 主监控脚本

## 运行逻辑

1. Zeabur从GitHub拉取代码
2. 根据 `requirements.txt` 安装依赖
3. 运行 `run_continuous.py`
4. 脚本每4小时自动运行一次监控
5. 发送Telegram通知

## 查看日志

在Zeabur控制台点击服务，可以查看实时日志输出。
