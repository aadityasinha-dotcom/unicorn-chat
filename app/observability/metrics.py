"""Lightweight in-process metrics.

Counters + a bounded latency reservoir, exposed as JSON at ``GET /metrics``.
This is deliberately simple: the intent is to show *what* we would measure.
In production the same ``Metrics`` interface would be backed by Prometheus /
OpenTelemetry, and Langfuse (see ``tracing.py``) already receives per-turn
traces with token usage.
"""
from __future__ import annotations

import statistics
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Metrics:
    counters: Counter = field(default_factory=Counter)
    _latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def inc(self, name: str, value: int = 1, **labels: str) -> None:
        key = name if not labels else name + "{" + ",".join(f"{k}={v}" for k, v in sorted(labels.items())) + "}"
        with self._lock:
            self.counters[key] += value

    def observe_latency(self, ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(ms)

    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self._latencies_ms)
            counters = dict(sorted(self.counters.items()))
        latency = {"count": len(lat)}
        if lat:
            srt = sorted(lat)
            latency.update(
                avg_ms=round(statistics.fmean(lat), 1),
                p50_ms=round(srt[len(srt) // 2], 1),
                p95_ms=round(srt[min(len(srt) - 1, int(len(srt) * 0.95))], 1),
                max_ms=round(srt[-1], 1),
            )
        queries = counters.get("queries_total", 0)
        errors = sum(v for k, v in counters.items() if k.startswith("errors_total"))
        rejected = sum(v for k, v in counters.items() if k.startswith("rejected_inputs_total"))
        derived = {
            "clarification_rate": round(counters.get("clarifications_total", 0) / queries, 3) if queries else 0.0,
            "error_rate": round(errors / queries, 3) if queries else 0.0,
            "rejected_rate": round(rejected / queries, 3) if queries else 0.0,
        }
        return {"started_at": self.started_at, "counters": counters, "latency": latency, "derived": derived}


metrics = Metrics()
