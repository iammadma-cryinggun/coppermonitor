FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cci_calculations.py .
COPY optimal_params.py .
COPY paper_trade.py .

RUN mkdir -p /data && chmod 777 /data

CMD ["python", "paper_trade.py", "daemon"]
