"""
FastAPI 主应用入口

启动:
    uvicorn backend.core.api:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.core.middleware import RequestLoggingMiddleware
from backend.core.analysis_routes import router as analysis_router
from backend.core.system_routes import router as system_router
from backend.utils.config import settings
from backend.utils.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # ── 启动 ──
    setup_logging()
    logger.info(
        f"🚀 {settings.app.name} v{settings.app.version} 启动中 "
        f"(debug={settings.app.debug})"
    )

    # 初始化数据库
    try:
        from backend.db.models import init_db
        init_db()
        logger.info("数据库表已就绪")
    except Exception as e:
        logger.warning(f"数据库初始化跳过: {e}")

    yield

    # ── 关闭 ──
    logger.info("应用关闭")


# ── 创建 FastAPI 实例 ─────────────────────

app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description="百合文学智能分析平台 — 提供 BERT + LLM 多维度文本分析 API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── 中间件 ────────────────────────────────

app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路由挂载 ──────────────────────────────

app.include_router(analysis_router, prefix="/api/v1")
app.include_router(system_router, prefix="/api/v1")


@app.get("/", tags=["根"])
async def root():
    return {
        "name": settings.app.name,
        "version": settings.app.version,
        "docs": "/docs",
        "health": "/api/v1/system/health",
    }
