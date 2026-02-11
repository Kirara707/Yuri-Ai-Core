"""
统一配置管理模块

支持从 .env 和 config.json 加载配置，.env 优先级更高。
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_legacy_config() -> dict:
    """加载旧版 config.json（向后兼容）"""
    config_path = os.path.join(BASE_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


_legacy = _load_legacy_config()


def _env(key: str, default=None, cast=None):
    """优先从环境变量获取，其次从 config.json 获取"""
    val = os.getenv(key)
    if val is None:
        val = _legacy.get(key, default)
    if val is None:
        return default
    if cast is not None:
        return cast(val)
    return val


# ────────────────────────────────────────
# 应用级配置
# ────────────────────────────────────────
@dataclass(frozen=True)
class AppConfig:
    version: str = _env("APP_VERSION", "2.0.0")
    debug: bool = _env("DEBUG", "False").lower() in ("true", "1", "yes") if isinstance(_env("DEBUG", "False"), str) else False
    environment: str = _env("ENVIRONMENT", "development")


# ────────────────────────────────────────
# Redis 配置
# ────────────────────────────────────────
@dataclass(frozen=True)
class RedisConfig:
    host: str = _env("REDIS_HOST", "127.0.0.1")
    port: int = _env("REDIS_PORT", 6379, int)
    db: int = _env("REDIS_DB", 0, int)
    password: Optional[str] = _env("REDIS_PASSWORD") or None

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


# ────────────────────────────────────────
# PostgreSQL 配置
# ────────────────────────────────────────
@dataclass(frozen=True)
class PostgresConfig:
    host: str = _env("POSTGRES_HOST", "127.0.0.1")
    port: int = _env("POSTGRES_PORT", 5432, int)
    user: str = _env("POSTGRES_USER", "yuri")
    password: str = _env("POSTGRES_PASSWORD", "yuri_password")
    database: str = _env("POSTGRES_DB", "yuri_ai_db")

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# ────────────────────────────────────────
# Celery 配置
# ────────────────────────────────────────
@dataclass(frozen=True)
class CeleryConfig:
    broker_url: str = _env("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
    result_backend: str = _env("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/1")
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: tuple = ("json",)
    timezone: str = "Asia/Shanghai"
    task_time_limit: int = 30 * 60      # 30 分钟硬超时
    task_soft_time_limit: int = 25 * 60  # 25 分钟软超时
    worker_prefetch_multiplier: int = 1
    worker_max_tasks_per_child: int = 100


# ────────────────────────────────────────
# LLM API 配置
# ────────────────────────────────────────
@dataclass(frozen=True)
class LLMConfig:
    api_key: str = _env("MOONSHOT_API_KEY") or _env("api_key", "")
    model: str = _env("LLM_MODEL", "kimi-k2-0905-preview")
    base_url: str = _env("LLM_BASE_URL", "https://api.moonshot.cn/v1")
    timeout: int = _env("LLM_TIMEOUT", 60, int)
    max_retries: int = _env("LLM_MAX_RETRIES", 5, int)
    retry_base_delay: float = _env("LLM_RETRY_BASE_DELAY", 1.0, float)
    max_threads_dialogue: int = _env("LLM_MAX_THREADS_DIALOGUE", 7, int)
    max_threads_verb: int = _env("LLM_MAX_THREADS_VERB", 20, int)


# ────────────────────────────────────────
# BERT 模型配置
# ────────────────────────────────────────
@dataclass(frozen=True)
class BERTConfig:
    model_path: str = _env("BERT_MODEL_PATH") or _env("bert_checkpoint", "./models/checkpoint-47200")
    max_len: int = _env("BERT_MAX_LEN") or _env("bert_max_len", 512, int)
    stride: int = _env("BERT_STRIDE") or _env("bert_stride", 128, int)
    batch_size: int = _env("BERT_BATCH_SIZE") or _env("bert_batch_size", 16, int)
    device: str = _env("BERT_DEVICE", "cuda")

    @property
    def abs_model_path(self) -> str:
        return os.path.abspath(os.path.join(BASE_DIR, self.model_path))


# ────────────────────────────────────────
# 日志配置
# ────────────────────────────────────────
@dataclass(frozen=True)
class LogConfig:
    level: str = _env("LOG_LEVEL", "INFO")
    log_dir: str = _env("LOG_DIR", "./logs")
    format: str = _env("LOG_FORMAT", "json")
    retention_days: int = _env("LOG_RETENTION_DAYS", 30, int)

    @property
    def abs_log_dir(self) -> str:
        return os.path.abspath(os.path.join(BASE_DIR, self.log_dir))


# ────────────────────────────────────────
# 缓存配置
# ────────────────────────────────────────
@dataclass(frozen=True)
class CacheConfig:
    ttl: int = _env("CACHE_TTL", 2592000, int)  # 30 天
    enabled: bool = _env("CACHE_ENABLED", "true").lower() in ("true", "1", "yes") if isinstance(_env("CACHE_ENABLED", "true"), str) else True


# ────────────────────────────────────────
# 熔断器配置
# ────────────────────────────────────────
@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = _env("CIRCUIT_BREAKER_FAILURE_THRESHOLD", 10, int)
    recovery_timeout: int = _env("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", 300, int)
    success_threshold: int = _env("CIRCUIT_BREAKER_SUCCESS_THRESHOLD", 2, int)


# ────────────────────────────────────────
# 路径配置（兼容旧 config.json）
# ────────────────────────────────────────
@dataclass(frozen=True)
class PathConfig:
    txt_test_dir: str = os.path.join(BASE_DIR, _legacy.get("txt_test_dir", "./txt_test"))
    txt_test_cleaned_dir: str = os.path.join(BASE_DIR, _legacy.get("txt_test_cleaned_dir", "./assets/txt_test_cleaned"))
    csv_cut_dialogue_dir: str = os.path.join(BASE_DIR, _legacy.get("csv_cut_dialogue_dir", "./csv/cut_dialogue"))
    csv_prediction_dir: str = os.path.join(BASE_DIR, _legacy.get("csv_prediction_dir", "./csv/prediction"))
    csv_weighted_dir: str = os.path.join(BASE_DIR, "csv", "weighted")
    csv_result_dir: str = os.path.join(BASE_DIR, "csv", "result")
    stopwords_file: str = os.path.join(BASE_DIR, _legacy.get("stopwords_file", "./assets/stopwords.txt"))
    verbword_file: str = os.path.join(BASE_DIR, _legacy.get("verbword_file", "./assets/verbword.txt"))


# ════════════════════════════════════════
# 全局单例汇总
# ════════════════════════════════════════
class Settings:
    """所有配置的统一入口"""
    app = AppConfig()
    redis = RedisConfig()
    postgres = PostgresConfig()
    celery = CeleryConfig()
    llm = LLMConfig()
    bert = BERTConfig()
    log = LogConfig()
    cache = CacheConfig()
    circuit_breaker = CircuitBreakerConfig()
    paths = PathConfig()


settings = Settings()
