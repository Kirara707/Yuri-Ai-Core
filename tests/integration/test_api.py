"""
FastAPI 集成测试

使用 httpx.AsyncClient 测试 API 端点（不需要实际的 Redis/Celery）
"""

import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
    """创建 FastAPI 测试实例"""
    from backend.core.api import app
    return app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── 根路由 ────────────────────────────────

class TestRootRoute:
    @pytest.mark.asyncio
    async def test_root(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


# ── 系统路由 ──────────────────────────────

class TestSystemRoutes:
    @pytest.mark.asyncio
    async def test_metrics(self, client):
        response = await client.get("/api/v1/system/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data or "modules" in data

    @pytest.mark.asyncio
    async def test_breaker_status(self, client):
        response = await client.get("/api/v1/system/breaker")
        assert response.status_code == 200
        data = response.json()
        assert "state" in data


# ── 分析路由（参数校验）──────────────────

class TestAnalysisRoutes:
    @pytest.mark.asyncio
    async def test_submit_validation_error(self, client):
        """文本太短应返回 422"""
        response = await client.post(
            "/api/v1/analysis/submit",
            json={"text": "太短", "book_id": "1"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_progress_not_found(self, client):
        """查询不存在的任务进度"""
        response = await client.get("/api/v1/analysis/progress/nonexistent-id")
        # 应返回 PENDING（Celery 对未知 task 返回 PENDING）
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("pending", "running")
