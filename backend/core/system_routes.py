"""
系统路由 /api/v1/system/*

GET  /health   — 健康检查
GET  /metrics  — 性能指标
GET  /breaker  — 熔断器状态
"""

from fastapi import APIRouter
from loguru import logger

from backend.models.schemas import HealthResponse, MetricsResponse
from backend.services.cache_service import CacheService
from backend.services.llm_service import get_breaker_status
from backend.services.metrics_service import metrics
from backend.utils.config import settings

router = APIRouter(prefix="/system", tags=["系统"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查 — 校验 Redis、DB、BERT 模型状态
    """
    cache = CacheService()
    redis_ok = cache.ping()

    # 数据库连通性
    db_ok = False
    try:
        from backend.db.models import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception as e:
        logger.debug(f"DB 健康检查失败: {e}")

    # BERT 模型是否已加载
    from backend.services.bert_service import _model
    bert_loaded = _model is not None

    overall = "ok" if redis_ok else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app.version,
        components={
            "redis": {"status": "up" if redis_ok else "down", "stats": cache.get_cache_stats()},
            "database": {"status": "up" if db_ok else "down"},
            "bert_model": {"status": "loaded" if bert_loaded else "not_loaded"},
            "llm_breaker": get_breaker_status(),
        },
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(minutes: int = 60):
    """
    性能指标 — 各模块 p50/p95/p99 延迟和成功率
    """
    return metrics.get_all_stats(minutes)


@router.get("/breaker")
async def breaker_status():
    """
    LLM 熔断器当前状态
    """
    return get_breaker_status()
