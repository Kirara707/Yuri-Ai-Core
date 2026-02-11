"""
熔断器（Circuit Breaker）

防止级联故障：当远程服务连续失败超过阈值时，自动打开熔断器，
拒绝后续请求并走降级逻辑，一段时间后尝试半开恢复。

状态转移:
    CLOSED ──(连续 N 次失败)──→ OPEN ──(超时后)──→ HALF_OPEN
      ▲                                                │
      └───────────(连续 M 次成功)──────────────────────┘
      ▲                          │
      └──────────(1 次失败)──────┘  → 重新 OPEN

使用示例:
    breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=300)

    if breaker.can_execute():
        try:
            result = call_api()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            if not breaker.can_execute():
                result = fallback()
    else:
        result = fallback()
"""

import enum
import threading
from datetime import datetime, timedelta
from loguru import logger


class CircuitState(enum.Enum):
    CLOSED = "closed"        # 正常运行，允许所有请求
    OPEN = "open"            # 故障模式，拒绝请求
    HALF_OPEN = "half_open"  # 试探模式，允许有限请求


class CircuitBreaker:
    """线程安全的熔断器"""

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 10,
        success_threshold: int = 2,
        recovery_timeout: int = 300,
    ):
        """
        Args:
            name:              熔断器名称（用于日志）
            failure_threshold: 连续失败 N 次后打开熔断器
            success_threshold: HALF_OPEN 状态下连续成功 M 次恢复 CLOSED
            recovery_timeout:  OPEN 后等待多少秒尝试 HALF_OPEN
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: datetime | None = None
        self._lock = threading.Lock()

    # ── 公开接口 ────────────────────────

    def can_execute(self) -> bool:
        """判断当前是否允许执行请求"""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._last_failure_time and (
                    datetime.now() - self._last_failure_time
                    > timedelta(seconds=self.recovery_timeout)
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        f"🟡 [{self.name}] 熔断器 → HALF_OPEN（尝试恢复）"
                    )
                    return True
                return False

            # HALF_OPEN
            return True

    def record_success(self):
        """记录一次成功调用"""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
                    logger.info(
                        f"🟢 [{self.name}] 熔断器 → CLOSED（恢复正常）"
                    )

    def record_failure(self):
        """记录一次失败调用"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                # 半开状态下失败，立即重新打开
                self._state = CircuitState.OPEN
                logger.error(
                    f"🔴 [{self.name}] 熔断器 → OPEN（半开状态失败，回退）"
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(
                    f"🔴 [{self.name}] 熔断器 → OPEN（连续失败 {self._failure_count} 次）"
                )

    # ── 状态查询 ────────────────────────

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def get_status(self) -> dict:
        """返回熔断器的完整状态快照"""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": (
                self._last_failure_time.isoformat()
                if self._last_failure_time
                else None
            ),
        }

    def reset(self):
        """手动重置熔断器到 CLOSED"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info(f"🔄 [{self.name}] 熔断器已手动重置")
