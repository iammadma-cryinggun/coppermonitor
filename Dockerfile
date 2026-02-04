# 使用Python 3.11作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY china_futures_fetcher.py .
COPY notifier.py .
COPY futures_monitor.py .

# 创建logs目录并设置权限
RUN mkdir -p /app/logs && chmod 777 /app/logs

# 设置环境变量（将在Zeabur控制台覆盖）
ENV TELEGRAM_BOT_TOKEN="" \
    TELEGRAM_CHAT_ID=""

# 健康检查（每10分钟检查一次）
HEALTHCHECK --interval=10m --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/logs/multi_tracking.csv') else 1)" || exit 1

# 定时运行多品种监控
CMD ["python", "futures_monitor.py", "--scheduled"]
