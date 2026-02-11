"""
请求日志中间件

功能:
- 注入唯一 request_id
- 记录请求耗时与状态码
- 将 request_id 注入 loguru 上下文
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger

from backend.utils.logging_config import request_context


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """为每个 HTTP 请求注入 request_id 并记录访问日志"""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request_context.set(request_id)
        request.state.request_id = request_id

        t0 = time.time()
        path = request.url.path

        try:
            response = await call_next(request)
            duration = (time.time() - t0) * 1000

            logger.info(
                f"{request.method} {path} → {response.status_code} "
                f"({duration:.1f}ms) [rid={request_id}]"
            )
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as e:
            duration = (time.time() - t0) * 1000
            logger.error(
                f"{request.method} {path} → 500 ({duration:.1f}ms) "
                f"[rid={request_id}] error={e}"
            )
            raise
