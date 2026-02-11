"""
工具层单元测试

测试对象:
- 重试装饰器
- 熔断器
- 配置加载
"""

import time
import pytest
from backend.utils.retry_decorator import exponential_backoff_retry
from backend.utils.circuit_breaker import CircuitBreaker


# ── 重试装饰器 ────────────────────────────

class TestRetryDecorator:
    def test_success_no_retry(self):
        call_count = 0

        @exponential_backoff_retry(max_retries=3, base_delay=0.01)
        def success_fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert success_fn() == "ok"
        assert call_count == 1

    def test_retry_then_success(self):
        call_count = 0

        @exponential_backoff_retry(max_retries=3, base_delay=0.01)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("临时错误")
            return "recovered"

        assert fail_twice() == "recovered"
        assert call_count == 3

    def test_all_retries_exhausted(self):
        @exponential_backoff_retry(max_retries=2, base_delay=0.01)
        def always_fail():
            raise RuntimeError("永久错误")

        with pytest.raises(RuntimeError, match="永久错误"):
            always_fail()


# ── 熔断器 ────────────────────────────────

class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.can_execute() is True
        assert cb.state == "CLOSED"

    def test_opens_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.can_execute() is False

    def test_half_open_after_recovery(self):
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"

        time.sleep(0.15)
        assert cb.can_execute() is True
        assert cb.state == "HALF_OPEN"

    def test_recovery_on_success(self):
        cb = CircuitBreaker("test", failure_threshold=2, success_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)

        cb.can_execute()  # → HALF_OPEN
        cb.record_success()
        assert cb.state == "CLOSED"

    def test_get_status(self):
        cb = CircuitBreaker("test_status")
        status = cb.get_status()
        assert status["name"] == "test_status"
        assert status["state"] == "CLOSED"
        assert "failure_count" in status

    def test_reset(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        assert cb.state == "OPEN"
        cb.reset()
        assert cb.state == "CLOSED"


# ── 指标服务 ──────────────────────────────

class TestMetricsService:
    def test_record_and_stats(self, metrics):
        for i in range(10):
            metrics.record("test_module", latency_ms=100 + i * 10, success=True)

        stats = metrics.get_module_stats("test_module")
        assert stats["sample_size"] == 10
        assert stats["success_rate"] == 1.0
        assert stats["avg_ms"] > 0

    def test_timed_context(self, metrics):
        with metrics.timed("test_timed"):
            time.sleep(0.01)

        stats = metrics.get_module_stats("test_timed")
        assert stats["sample_size"] == 1
        assert stats["min_ms"] >= 9  # 至少 ~10ms

    def test_empty_module(self, metrics):
        stats = metrics.get_module_stats("nonexistent")
        assert stats["sample_size"] == 0

    def test_get_all_stats(self, metrics):
        metrics.record("mod_a", 50.0)
        metrics.record("mod_b", 100.0)
        all_stats = metrics.get_all_stats()
        assert "modules" in all_stats


# ── 配置加载 ──────────────────────────────

class TestConfig:
    def test_settings_exists(self, settings):
        assert settings.app.name is not None
        assert settings.redis.host is not None

    def test_bert_config(self, settings):
        assert settings.bert.max_len > 0
        assert settings.bert.stride > 0
