"""Small fail-open HTTP client for the isolated Qwen-VL Reranker service."""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def score_documents(
    *,
    query: dict[str, Any],
    documents: list[dict[str, Any]],
    instruction: str,
    timeout: float | None = None,
) -> list[float]:
    """Score text/image evidence, returning an empty list when unavailable.

    Reranking is an enrichment signal only.  A stopped or overloaded isolated
    service therefore never makes a GrowthRun fail.
    """
    if not query or not documents:
        return []
    endpoint = os.getenv("QWEN_VL_RERANKER_URL", "http://127.0.0.1:7010").rstrip("/") + "/score"
    body = json.dumps(
        {
            "instruction": instruction,
            "query": query,
            "documents": documents,
            "fps": 1.0,
            "max_frames": 8,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout or float(os.getenv("QWEN_VL_RERANKER_TIMEOUT", "12"))) as response:
            payload = json.loads(response.read().decode("utf-8"))
        scores = payload.get("scores") or []
        return [max(0.0, min(1.0, float(item))) for item in scores[: len(documents)]]
    except (OSError, ValueError, TypeError, URLError) as exc:
        logger.warning("multimodal reranker skipped: %s", exc)
        return []

