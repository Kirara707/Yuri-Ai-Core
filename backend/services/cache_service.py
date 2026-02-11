"""
Redis 缓存服务

功能:
- 文本 MD5 哈希 → 结果缓存（30 天 TTL）
- 任务进度读写
- 任务结果读写

Key 结构:
    cache:text:{md5}         → JSON 分析结果      TTL 30 天
    task:progress:{task_id}  → JSON 进度信息      TTL 1 小时
    task:result:{task_id}    → JSON 最终结果      TTL 24 小时
"""

import json
import hashlib
from typing import Optional
from datetime import datetime

import redis
from loguru import logger

from backend.utils.config import settings


class _InMemoryCacheClient:
    """轻量级内存缓存，仅用于 Mock/降级场景"""

    def __init__(self):
        self._store: dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def get(self, name: str) -> Optional[str]:
        return self._store.get(name)

    def setex(self, name: str, ttl: int, value: str) -> None:
        self._store[name] = value

    def info(self, section: str = "memory") -> dict:
        return {"used_memory_human": "0B"}

    def dbsize(self) -> int:
        return len(self._store)


class CacheService:
    """Redis 缓存封装"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
    ):
        self._using_fallback = False
        self._client = self._create_client(host, port, db, password)

    def _create_client(
        self,
        host: str,
        port: int,
        db: int,
        password: str,
    ):
        if settings.llm.mock_mode or not settings.cache.enabled:
            logger.debug("CacheService: mock mode or cache disabled, using in-memory cache")
            self._using_fallback = True
            return _InMemoryCacheClient()

        try:
            client = redis.Redis(
                host=host or settings.redis.host,
                port=port or settings.redis.port,
                db=db if db is not None else settings.redis.db,
                password=password or settings.redis.password,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            client.ping()
            return client
        except Exception as exc:
            logger.warning(f"Redis 初始化失败，降级到内存缓存: {exc}")
            self._using_fallback = True
            return _InMemoryCacheClient()

    # ── 健康检查 ──────────────────────────

    def ping(self) -> bool:
        try:
            return self._client.ping()
        except Exception:
            return False

    # ── 文本缓存 ──────────────────────────

    @staticmethod
    def text_hash(text: str) -> str:
        """对文本生成 MD5 哈希"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_cached_result(self, text_hash: str) -> Optional[dict]:
        """查询文本分析缓存，返回 dict 或 None"""
        key = f"cache:text:{text_hash}"
        raw = self._client.get(key)
        if raw:
            logger.debug(f"缓存命中: {key}")
            return json.loads(raw)
        return None

    def set_cached_result(
        self, text_hash: str, result: dict, ttl: int = None
    ):
        """存储文本分析结果到缓存"""
        ttl = ttl or settings.cache.ttl
        key = f"cache:text:{text_hash}"
        self._client.setex(key, ttl, json.dumps(result, default=str, ensure_ascii=False))
        logger.debug(f"缓存写入: {key} TTL={ttl}s")

    # ── 任务进度 ──────────────────────────

    def get_progress(self, task_id: str) -> Optional[dict]:
        """读取任务进度"""
        key = f"task:progress:{task_id}"
        raw = self._client.get(key)
        return json.loads(raw) if raw else None

    def set_progress(
        self,
        task_id: str,
        *,
        status: str,
        progress: float,
        current_step: str = "",
        extra: dict = None,
        ttl: int = 3600,
    ):
        """更新任务进度"""
        key = f"task:progress:{task_id}"
        data = {
            "status": status,
            "progress": progress,
            "current_step": current_step,
            "updated_at": datetime.now().isoformat(),
        }
        if extra:
            data.update(extra)
        self._client.setex(key, ttl, json.dumps(data, default=str, ensure_ascii=False))

    # ── 任务结果 ──────────────────────────

    def get_task_result(self, task_id: str) -> Optional[dict]:
        """获取任务最终结果"""
        key = f"task:result:{task_id}"
        raw = self._client.get(key)
        return json.loads(raw) if raw else None

    def set_task_result(self, task_id: str, result: dict, ttl: int = 86400):
        """存储任务最终结果（默认保留 24 小时）"""
        key = f"task:result:{task_id}"
        self._client.setex(key, ttl, json.dumps(result, default=str, ensure_ascii=False))

    # ── 统计 ──────────────────────────────

    def get_cache_stats(self) -> dict:
        """返回 Redis 连接和缓存统计信息"""
        try:
            keys_count = self._client.dbsize()
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

        if self._using_fallback:
            return {
                "connected": False,
                "mode": "in-memory",
                "total_keys": keys_count,
            }

        info = self._client.info("memory")
        return {
            "connected": True,
            "used_memory_human": info.get("used_memory_human", "N/A"),
            "total_keys": keys_count,
        }
