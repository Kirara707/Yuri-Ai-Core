"""
结构化日志配置

输出格式:
- 控制台: 彩色人类可读
- 文件:   JSON 格式（兼容 ELK/Splunk）

每个请求自动注入 request_id、耗时等上下文信息。
"""

import os
import sys
import json
from datetime import datetime
from contextvars import ContextVar
from loguru import logger

# ── 请求级上下文变量 ──────────────────────
request_context: ContextVar[dict] = ContextVar("request_context", default={})


def _json_serializer(record) -> str:
    """将 loguru record 序列化为 JSON 一行"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    # 合并请求上下文（request_id, method, path …）
    ctx = request_context.get()
    if ctx:
        log_entry.update(ctx)
    # 异常信息
    if record["exception"] is not None:
        log_entry["exception"] = str(record["exception"])
    return json.dumps(log_entry, ensure_ascii=False, default=str) + "\n"


def setup_logging(log_dir: str = "logs", level: str = "INFO", fmt: str = "json"):
    """
    初始化日志系统

    Args:
        log_dir:  日志文件存放目录
        level:    最低日志级别
        fmt:      文件日志格式 "json" | "text"
    """
    # 移除 loguru 默认 handler
    logger.remove()

    # 1) 控制台 —— 始终使用彩色人类可读格式
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # 2) 文件 —— JSON 格式 or 纯文本
    os.makedirs(log_dir, exist_ok=True)

    if fmt == "json":
        logger.add(
            os.path.join(log_dir, "yuri_ai_{time:YYYY-MM-DD}.log"),
            format=_json_serializer,
            level="DEBUG",
            rotation="00:00",       # 每天午夜轮换
            retention="30 days",    # 保留 30 天
            compression="gz",      # 旧日志压缩
            enqueue=True,          # 异步写入，不阻塞主线程
        )
    else:
        logger.add(
            os.path.join(log_dir, "yuri_ai_{time:YYYY-MM-DD}.log"),
            level="DEBUG",
            rotation="00:00",
            retention="30 days",
            compression="gz",
            enqueue=True,
        )

    logger.info(f"日志系统初始化完成 | 级别={level} 格式={fmt} 目录={log_dir}")
    return logger
