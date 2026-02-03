# 沪铜期货监控系统

Zeabur会自动读取这个Dockerfile来构建镜像

## 部署说明

1. 在Zeabur导入GitHub仓库
2. 配置环境变量（Secrets）：
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
3. 点击Deploy

## 启动命令

容器会自动运行：python run_continuous.py
每4小时监控一次并发送Telegram通知
