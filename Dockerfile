# Multi-stage build for AI Assistant
FROM python:3.12-slim AS base

WORKDIR /app

# Cài đặt công cụ hệ thống tối thiểu
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt dependencies từ requirements.lock
COPY requirements.lock requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock

# Copy mã nguồn ứng dụng
COPY . .

# Expose FastAPI (8000) và Streamlit (8501)
EXPOSE 8000 8501

# Chạy ứng dụng FastAPI mặc định
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
