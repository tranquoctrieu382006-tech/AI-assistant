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

# Tạo non-root user 'appuser' với UID 1000 để tăng cường bảo mật container
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/user_uploads && \
    chown -R appuser:appuser /app

# Copy mã nguồn ứng dụng với quyền sở hữu appuser
COPY --chown=appuser:appuser . .

# Chuyển sang non-root user
USER appuser

# Expose FastAPI (8000) và Streamlit (8501)
EXPOSE 8000 8501

# Healthcheck định kỳ kiểm tra trạng thái sống của dịch vụ
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/live || exit 1

# Chạy ứng dụng FastAPI mặc định
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
