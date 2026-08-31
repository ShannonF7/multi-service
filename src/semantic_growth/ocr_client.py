"""调用独立 PaddleOCR 服务的容错客户端。

输入为待识别图片列表，输出按 asset_id 索引的 OCR 结果；服务不可用时返回空字典，
由调用方决定重试，不阻塞文本证据消费。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def extract_ocr_batch(items: list[dict[str, Any]], *, timeout: float | None = None) -> dict[int, dict[str, Any]]:
    """批量调用本地 OCR 服务。

    输入：包含 asset_id/image 的图片列表；输出：按 asset_id 索引的结果字典。
    网络或服务异常时返回空字典并记录日志，不抛出异常，保证文本增长链可继续。
    """
    if not items:
        return {}
    endpoint = os.getenv("GROWTH_OCR_URL", "http://127.0.0.1:7011").rstrip("/") + "/extract"
    body = json.dumps({"items": items[:16]}, ensure_ascii=False).encode("utf-8")
    request = Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout or float(os.getenv("GROWTH_OCR_TIMEOUT", "45"))) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result: dict[int, dict[str, Any]] = {}
        for item in payload.get("items") or []:
            if item.get("asset_id") is None:
                continue
            result[int(item["asset_id"])] = dict(item)
        return result
    except (OSError, ValueError, TypeError, URLError) as exc:
        logger.warning("growth OCR skipped: %s", exc)
        return {}

