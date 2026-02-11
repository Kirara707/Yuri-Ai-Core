# ============================================
# Yuri AI Core — 多阶段构建 Dockerfile
# ============================================
# 阶段1: 依赖安装
# 阶段2: 运行时镜像（精简）
# ============================================

# ── 阶段 1: 构建依赖 ─────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── 阶段 2: 运行时 ───────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="Yuri AI Core Team"
LABEL description="百合文学智能分析平台"

WORKDIR /app

# 从构建阶段拷贝依赖
COPY --from=builder /install /usr/local

# 拷贝应用代码
COPY backend/ ./backend/
COPY script/ ./script/
COPY config.json .
COPY assets/ ./assets/

# 拷贝模型（如果需要在镜像内自包含）
# COPY models/ ./models/

# 创建非 root 用户
RUN groupadd -r yuri && useradd -r -g yuri yuri \
    && mkdir -p /app/logs /app/csv /app/txt \
    && chown -R yuri:yuri /app

USER yuri

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/system/health')" || exit 1

# 默认启动命令
CMD ["uvicorn", "backend.core.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
