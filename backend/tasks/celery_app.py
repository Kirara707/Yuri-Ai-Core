"""
Celery 应用配置与实例化

Broker: Redis
Backend: Redis
"""

from celery import Celery
from backend.utils.config import settings

celery_app = Celery(
    "yuri_ai_core",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,           # 24 小时后过期
    task_soft_time_limit=1800,      # 30 分钟软超时
    task_time_limit=3600,           # 60 分钟硬超时
    task_routes={
        "backend.tasks.analysis.*": {"queue": "analysis"},
    },
)

# 自动发现任务模块
celery_app.autodiscover_tasks(["backend.tasks"])
