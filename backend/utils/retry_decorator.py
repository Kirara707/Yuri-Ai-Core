"""
指数退避重试装饰器

支持同步和异步函数，可配置最大重试次数、基础延迟、退避因子、最大延迟。
用于处理 LLM API 的 Rate Limit、网络超时等瞬时故障。

重试延迟序列（默认参数）:
  失败 1 → 等 1s → 失败 2 → 等 2s → 失败 3 → 等 4s → ... → 上限 32s

使用示例:
    @exponential_backoff_retry(max_retries=5, exceptions=(ConnectionError, TimeoutError))
    async def call_api():
        ...
"""

import asyncio
import functools
import time
from typing import Callable, Any, Tuple, Type
from loguru import logger


def exponential_backoff_retry(
    max_retries: int = 5,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 32.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable = None,
):
    """
    指数退避重试装饰器

    Args:
        max_retries:    最大重试次数（不含首次调用）
        base_delay:     首次重试等待时间（秒）
        backoff_factor: 每次重试的延迟倍增因子
        max_delay:      最大延迟上限（秒）
        exceptions:     需要重试的异常类型元组
        on_retry:       每次重试前的回调函数 on_retry(attempt, exc, delay)
    """

    def decorator(func: Callable) -> Callable:

        # ── 异步版本 ──────────────────────
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        logger.error(
                            f"[Retry] {func.__name__} 全部 {max_retries+1} 次尝试失败 | "
                            f"最终错误: {exc}"
                        )
                        raise
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(
                        f"[Retry] {func.__name__} 第 {attempt+1} 次失败, "
                        f"{delay:.1f}s 后重试 | 错误: {exc}"
                    )
                    if on_retry:
                        on_retry(attempt + 1, exc, delay)
                    await asyncio.sleep(delay)
            raise last_exc  # pragma: no cover

        # ── 同步版本 ──────────────────────
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        logger.error(
                            f"[Retry] {func.__name__} 全部 {max_retries+1} 次尝试失败 | "
                            f"最终错误: {exc}"
                        )
                        raise
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(
                        f"[Retry] {func.__name__} 第 {attempt+1} 次失败, "
                        f"{delay:.1f}s 后重试 | 错误: {exc}"
                    )
                    if on_retry:
                        on_retry(attempt + 1, exc, delay)
                    time.sleep(delay)
            raise last_exc  # pragma: no cover

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
