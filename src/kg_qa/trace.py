"""在线问答阶段轨迹和耗时记录，不记录模型思维链。"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, TypeVar

from .schemas import TraceEvent

T = TypeVar("T")


class TraceCollector:
    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or f"qa-{uuid.uuid4().hex}"
        self.events: list[TraceEvent] = []

    def run(
        self,
        stage: str,
        callback: Callable[[], T],
        *,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        on_event: Callable[[TraceEvent], None] | None = None,
    ) -> T:
        started = time.perf_counter()
        self.events.append(TraceEvent(stage=stage, status="STARTED", summary=summary, metadata=metadata or {}))
        if on_event:
            on_event(self.events[-1])
        try:
            result = callback()
        except Exception as exc:
            duration = round((time.perf_counter() - started) * 1000, 2)
            self.events.append(TraceEvent(
                stage=stage, status="FAILED", duration_ms=duration,
                summary=str(exc), metadata={"error_type": type(exc).__name__},
            ))
            if on_event:
                on_event(self.events[-1])
            raise
        duration = round((time.perf_counter() - started) * 1000, 2)
        self.events.append(TraceEvent(
            stage=stage, status="COMPLETED", duration_ms=duration,
            summary=summary, metadata=metadata or {},
        ))
        if on_event:
            on_event(self.events[-1])
        return result

