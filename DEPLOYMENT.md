# 线上部署指南（Docker/云服务器）

## 📋 环境变量配置

### 方法1: Docker Compose（推荐）

创建 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  copper-monitor:
    build: .
    container_name: copper_monitor
    restart: unless-stopped

    # 环境变量配置
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - TZ=Asia/Shanghai

    # 定时任务（每4小时）
    command: >
      sh -c "
      echo '0 */4 * * * cd /app && python copper_monitor.py >> /app/logs/cron.log 2>&1' > /tmp/crontab &&
      crond -f -l 2 -L /app/logs/cron.log
      "

    volumes:
      - ./logs:/app/logs
      - ./config:/app/config:ro

    # 时区同步
    timezone: Asia/Shanghai
```

创建 `.env` 文件:

```bash
# Telegram配置
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321

# 时区
TZ=Asia/Shanghai
```

**启动**:
```bash
docker-compose up -d
docker-compose logs -f
```

---

### 方法2: Kubernetes ConfigMap

创建 `configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: copper-monitor-config
data:
  TELEGRAM_BOT_TOKEN: "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
  TELEGRAM_CHAT_ID: "987654321"
```

创建 `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: copper-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app: copper-monitor
  template:
    metadata:
      labels:
        app: copper-monitor
    spec:
      containers:
      - name: monitor
        image: your-registry/copper-monitor:latest
        envFrom:
        - configMapRef:
            name: copper-monitor-config
        volumeMounts:
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: logs
        emptyDir: {}
```

**部署**:
```bash
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
```

---

### 方法3: 云服务器（Systemd Timer）

#### 3.1 创建环境变量文件

```bash
# /etc/copper-monitor/env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```

#### 3.2 创建Systemd Service

`/etc/systemd/system/copper-monitor.service`:

```ini
[Unit]
Description=Copper Futures Monitor
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/opt/copper-monitor
EnvironmentFile=/etc/copper-monitor/env
ExecStart=/usr/bin/python3 copper_monitor.py
StandardOutput=append:/var/log/copper-monitor/monitor.log
StandardError=append:/var/log/copper-monitor/error.log

[Install]
WantedBy=multi-user.target
```

#### 3.3 创建Systemd Timer

`/etc/systemd/system/copper-monitor.timer`:

```ini
[Unit]
Description=Copper Futures Monitor Timer (Every 4 hours)
Requires=copper-monitor.service

[Timer]
OnCalendar=00/4:00
AccuracySec=5min
Persistent=true

[Install]
WantedBy=timers.target
```

**启动**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable copper-monitor.timer
sudo systemctl start copper-monitor.timer

# 查看状态
sudo systemctl list-timers
sudo journalctl -u copper-monitor -f
```

---

### 方法4: GitHub Actions（定时运行）

创建 `.github/workflows/monitor.yml`:

```yaml
name: Copper Futures Monitor

on:
  schedule:
    # 每4小时运行一次 (UTC时间)
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
        TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
        TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      run: |
        python copper_monitor.py

    - name: Upload logs
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: logs
        path: logs/
```

**配置Secrets**:
1. 进入GitHub仓库 Settings → Secrets and variables → Actions
2. 添加以下Secrets:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`

---

### 方法5: AWS Lambda + EventBridge

#### Lambda函数代码

```python
import os
import json

def lambda_handler(event, context):
    # 从环境变量读取配置
    os.environ['TELEGRAM_BOT_TOKEN'] = os.environ.get('TELEGRAM_BOT_TOKEN')
    os.environ['TELEGRAM_CHAT_ID'] = os.environ.get('TELEGRAM_CHAT_ID')

    # 导入并运行监控
    from copper_monitor import run_monitoring
    run_monitoring()

    return {
        'statusCode': 200,
        'body': json.dumps('Monitor completed')
    }
```

#### EventBridge规则

```json
{
  "RuleName": "copper-monitor-every-4-hours",
  "ScheduleExpression": "rate(4 hours)",
  "Targets": [{
    "Id": "1",
    "Arn": "arn:aws:lambda:region:account:function:copper-monitor",
    "Input": "{}"
  }]
}
```

---

## 🔐 环境变量说明

| 变量名 | 必需 | 说明 | 示例 |
|--------|------|------|------|
| `TELEGRAM_BOT_TOKEN` | ✅ | Telegram Bot Token | `123456789:ABCdefGHIjklMNOpqrsTUVwxyz` |
| `TELEGRAM_CHAT_ID` | ✅ | 接收消息的Chat ID | `987654321` |
| `TZ` | ❌ | 时区（可选） | `Asia/Shanghai` |

---

## 🐳 Docker部署示例

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY *.py ./

# 创建日志目录
RUN mkdir -p logs

# 设置环境变量（可被docker-compose覆盖）
ENV TELEGRAM_BOT_TOKEN=""
ENV TELEGRAM_CHAT_ID=""

# 定时任务（每4小时）
RUN echo '0 */4 * * * cd /app && python copper_monitor.py >> /app/logs/cron.log 2>&1' > /tmp/crontab

# 启动cron
CMD ["crond", "-f", "-l", "2", "-L", "/app/logs/cron.log"]
```

### requirements.txt

```
pandas>=1.5.0
numpy>=1.23.0
akshare>=1.9.0
requests>=2.28.0
```

### 构建和运行

```bash
# 构建镜像
docker build -t copper-monitor:latest .

# 运行容器
docker run -d \
  --name copper-monitor \
  -e TELEGRAM_BOT_TOKEN="your_token" \
  -e TELEGRAM_CHAT_ID="your_chat_id" \
  -v $(pwd)/logs:/app/logs \
  copper-monitor:latest

# 查看日志
docker logs -f copper-monitor
```

---

## 📊 监控和日志

### 查看日志

```bash
# Docker
docker logs -f copper-monitor

# Kubernetes
kubectl logs -f deployment/copper-monitor

# Systemd
sudo journalctl -u copper-monitor -f

# 直接文件
tail -f logs/strategy_monitor.log
```

### 健康检查

```bash
# 检查最后运行时间
stat logs/performance_tracking.csv

# 检查日志中的错误
grep ERROR logs/strategy_monitor.log

# 检查Telegram发送状态
grep "Telegram" logs/strategy_monitor.log | tail -20
```

---

## 🚀 快速部署检查清单

- [ ] 1. 设置环境变量（TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID）
- [ ] 2. 安装Python依赖（pandas, numpy, akshare, requests）
- [ ] 3. 测试通知发送：`python notifier.py`
- [ ] 4. 测试监控运行：`python copper_monitor.py`
- [ ] 5. 配置定时任务（Docker/K8s/Systemd）
- [ ] 6. 验证日志输出
- [ ] 7. 监控首次运行结果

---

## 🔧 故障排查

### 环境变量未生效

```bash
# 检查环境变量
echo $TELEGRAM_BOT_TOKEN
echo $TELEGRAM_CHAT_ID

# 或在Python中测试
python -c "import os; print(os.environ.get('TELEGRAM_BOT_TOKEN'))"
```

### Docker定时任务不运行

```bash
# 进入容器检查
docker exec -it copper-monitor bash
crontab -l
cat /tmp/crontab

# 手动运行测试
docker exec copper-monitor python copper_monitor.py
```

### Kubernetes Pod重启

```bash
# 查看Pod状态
kubectl get pods
kubectl describe pod copper-monitor-xxx

# 查看日志
kubectl logs copper-monitor-xxx --previous
```

---

## 📝 总结

**推荐部署方式**:
- 本地测试: 直接运行 + 环境变量
- 云服务器: Docker Compose
- 生产环境: Kubernetes
- 简单定时: GitHub Actions

**环境变量优先级**:
1. 环境变量 > 2. 配置文件

**关键点**:
- 所有敏感信息通过环境变量配置
- 持久化日志到本地volume
- 设置重试策略应对网络故障
