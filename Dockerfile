# Стадия 1: ставим зависимости
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Стадия 2: только для запуска
FROM python:3.10-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY tests/ ./tests/

RUN useradd -r -m -s /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
