"""
pytest 全局配置与 fixtures
"""

import os
import sys
import pytest

# 确保项目根目录在 path 中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 设置测试环境变量（在导入 backend 之前）
os.environ.setdefault("APP_DEBUG", "true")
os.environ.setdefault("LLM_API_KEY", "test-key-not-real")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("POSTGRES_HOST", "localhost")


@pytest.fixture(scope="session")
def settings():
    """全局配置 fixture"""
    from backend.utils.config import settings
    return settings


@pytest.fixture
def cache_service():
    """Redis 缓存服务 fixture（需要 Redis 运行）"""
    from backend.services.cache_service import CacheService
    svc = CacheService()
    if not svc.ping():
        pytest.skip("Redis 不可用")
    return svc


@pytest.fixture
def metrics():
    """指标服务 fixture"""
    from backend.services.metrics_service import MetricsService
    return MetricsService()
