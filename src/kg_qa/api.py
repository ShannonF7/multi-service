"""KG-QA HTTP 路由。"""

from __future__ import annotations

import json
from queue import Queue
from threading import Thread
from typing import Any, Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .schemas import QARequest, QAResponse
from .service import QAService

router = APIRouter(prefix="/kg-qa", tags=["KG-QA"])
service = QAService()


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


@router.post("/query", response_model=QAResponse)
def query(payload: QARequest) -> QAResponse:
    return service.run(payload)


def _sse(item: dict[str, Any]) -> str:
    return f"data: {json.dumps(item, ensure_ascii=False, default=str)}\n\n"


@router.post("/stream")
def stream(payload: QARequest) -> StreamingResponse:
    """以 SSE 推送真实阶段状态，随后逐段推送最终答案。"""
    queue: Queue[dict[str, Any] | None] = Queue()

    def on_trace(event: Any) -> None:
        queue.put({"type": "trace", "event": _dump(event)})

    def on_answer_delta(delta: str) -> None:
        if delta:
            queue.put({"type": "answer_delta", "delta": str(delta)})

    def worker() -> None:
        try:
            result = service.run(
                payload,
                on_trace=on_trace,
                on_answer_delta=on_answer_delta,
            )
            queue.put({"type": "result", "data": _dump(result)})
        except Exception as exc:
            queue.put({"type": "error", "message": str(exc)})
        finally:
            queue.put(None)

    Thread(target=worker, name="kg-qa-stream", daemon=True).start()

    def generate() -> Iterator[str]:
        while True:
            item = queue.get()
            if item is None:
                break
            yield _sse(item)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

