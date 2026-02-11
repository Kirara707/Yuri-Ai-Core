"""
性能指标收集服务

追踪各模块（BERT / LLM_dialogue / LLM_verb / cache 等）的耗时和成功率，
提供 p50 / p95 / p99 等统计数据，供 /api/v1/metrics 端点展示。
"""

import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class LatencyRecord:
    module: str
    latency_ms: float
    success: bool
    timestamp: datetime


class MetricsService:
    """线程安全的性能指标聚合器"""

    def __init__(self, max_records: int = 10000):
        self._max = max_records
        self._records: List[LatencyRecord] = []
        self._lock = threading.Lock()

    # ── 记录 ──────────────────────────────

    def record(self, module: str, latency_ms: float, success: bool = True):
        """记录一次模块调用的延迟"""
        with self._lock:
            self._records.append(
                LatencyRecord(
                    module=module,
                    latency_ms=latency_ms,
                    success=success,
                    timestamp=datetime.now(),
                )
            )
            # 滑动窗口，保留最新记录
            if len(self._records) > self._max:
                self._records = self._records[-self._max :]

    def timed(self, module: str):
        """
        上下文管理器，用于自动计时

        with metrics.timed("bert_infer"):
            result = bert_service.infer_text(text)
        """
        return _TimedContext(self, module)

    # ── 统计 ──────────────────────────────

    def get_module_stats(self, module: str, minutes: int = 60) -> dict:
        """
        获取指定模块最近 N 分钟的统计数据

        Returns:
            {
                "module": "bert_infer",
                "min_ms": 100.0,
                "max_ms": 500.0,
                "avg_ms": 250.5,
                "p50_ms": 240.0,
                "p95_ms": 400.0,
                "p99_ms": 450.0,
                "success_count": 98,
                "fail_count": 2,
                "success_rate": 0.98,
                "sample_size": 100,
                "time_window_minutes": 60
            }
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)

        with self._lock:
            recent = [
                r
                for r in self._records
                if r.module == module and r.timestamp >= cutoff
            ]

        if not recent:
            return {
                "module": module,
                "sample_size": 0,
                "message": "暂无数据",
            }

        latencies = sorted(r.latency_ms for r in recent)
        n = len(latencies)
        successes = sum(1 for r in recent if r.success)

        def percentile(pct: float) -> float:
            idx = min(int(n * pct), n - 1)
            return round(latencies[idx], 2)

        return {
            "module": module,
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "avg_ms": round(sum(latencies) / n, 2),
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
            "success_count": successes,
            "fail_count": n - successes,
            "success_rate": round(successes / n, 4),
            "sample_size": n,
            "time_window_minutes": minutes,
        }

    def get_all_stats(self, minutes: int = 60) -> dict:
        """获取所有已知模块的统计数据"""
        with self._lock:
            modules = sorted(set(r.module for r in self._records))

        return {
            "timestamp": datetime.now().isoformat(),
            "modules": {
                m: self.get_module_stats(m, minutes) for m in modules
            },
        }


# ── 计时上下文管理器 ──────────────────────


class _TimedContext:
    def __init__(self, svc: MetricsService, module: str):
        self._svc = svc
        self._module = module
        self._t0 = 0.0
        self._success = True

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.time() - self._t0) * 1000
        self._svc.record(self._module, elapsed, success=(exc_type is None))
        return False  # 不吞异常


# ── 全局单例 ──────────────────────────────
metrics = MetricsService()
