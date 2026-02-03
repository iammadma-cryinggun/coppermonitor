# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有项目文件
COPY *.py ./
COPY config/ ./config/

# 创建logs目录
RUN mkdir -p logs

# 设置环境变量（将在Zeabur控制台覆盖）
ENV TELEGRAM_BOT_TOKEN="" \
    TELEGRAM_CHAT_ID=""

# 持续运行监控
CMD ["python", "run_continuous.py"]
