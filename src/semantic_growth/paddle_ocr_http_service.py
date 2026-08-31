"""调用独立 PaddleOCR 环境的本地 HTTP 适配器。

输入为图片 URL 或本地路径列表，输出保留原始 OCR 文本、过滤后的文本、
识别框坐标、行置信度和模型版本。该服务只负责识别，不写业务数据库。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit


MEDIA_BASE_URL = os.getenv("OCR_MEDIA_BASE_URL", "http://ai.smartoptiks.cn").rstrip("/")
MODEL_CACHE_DIR = Path(os.getenv("OCR_CACHE_DIR", "/tmp/growth_ocr_cache"))
MIN_SCORE = float(os.getenv("OCR_MIN_SCORE", "0.55"))
_ocr: Any = None
_ocr_error: str | None = None
_lock = threading.Lock()


def _get_ocr() -> Any:
    global _ocr, _ocr_error
    if _ocr is not None:
        return _ocr
    with _lock:
        if _ocr is not None:
            return _ocr
        try:
            from paddleocr import PaddleOCR

            _ocr = PaddleOCR(
                text_detection_model_name=os.getenv("OCR_DET_MODEL", "PP-OCRv5_server_det"),
                text_recognition_model_name=os.getenv("OCR_REC_MODEL", "PP-OCRv5_server_rec"),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            _ocr_error = None
            return _ocr
        except Exception as exc:  # pragma: no cover - depends on OCR runtime
            _ocr_error = f"{type(exc).__name__}: {exc}"
            raise


def _resolve_url(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("/"):
        return urljoin(MEDIA_BASE_URL + "/", value.lstrip("/"))
    return value


def _download(value: str, asset_id: Any) -> Path:
    raw = str(value or "").strip()
    if raw.startswith("/") and Path(raw).exists():
        return Path(raw)
    source = _resolve_url(raw)
    if not source:
        raise ValueError("image url is empty")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlsplit(source).path).suffix.lower()
    allowed_suffixes = {".bmp", ".dib", ".jpeg", ".jpg", ".png", ".webp", ".pbm", ".pgm", ".ppm", ".pnm", ".sr", ".ras", ".tiff", ".tif", ".pdf"}
    if suffix not in allowed_suffixes:
        suffix = ".jpg"
    path = MODEL_CACHE_DIR / f"{asset_id}_{hashlib.sha1(source.encode()).hexdigest()[:12]}{suffix}"
    if path.exists() and path.stat().st_size:
        return path
    fd, temp_name = tempfile.mkstemp(dir=str(MODEL_CACHE_DIR), suffix=".tmp")
    os.close(fd)
    try:
        request = urllib.request.Request(source, headers={"User-Agent": "growth-ocr/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            Path(temp_name).write_bytes(response.read())
        Path(temp_name).replace(path)
        return path
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _bbox(value: Any) -> list[float] | None:
    """把 PaddleOCR 的四点框或四值框归一化为 [x1,y1,x2,y2]。"""
    try:
        points = value.tolist() if hasattr(value, "tolist") else value
        if isinstance(points, (list, tuple)) and len(points) == 4:
            if all(isinstance(item, (int, float)) for item in points):
                return [float(item) for item in points]
            flat = [
                float(coord)
                for point in points
                if isinstance(point, (list, tuple)) and len(point) >= 2
                for coord in point[:2]
            ]
            if len(flat) >= 4:
                return [min(flat[0::2]), min(flat[1::2]), max(flat[0::2]), max(flat[1::2])]
    except (TypeError, ValueError):
        return None
    return None


def _parse(result: Any) -> dict[str, Any]:
    """解析 PaddleOCR 输出并保留可追溯的文本框信息。

    输入：PaddleOCR predict 的单页结果；输出：原始/清洗文本、分数、
    ocr_blocks 框列表和模型版本。低于阈值的行不会进入清洗文本和候选抽取，
    但会保留在 ocr_raw_text 以便后续人工复核。
    """
    page = result[0] if isinstance(result, list) and result else result
    if not isinstance(page, dict):
        return {"ocr_text": "", "ocr_raw_text": "", "ocr_blocks": [], "max_score": 0.0, "status": "no_text", "model": "paddleocr"}
    texts = list(page.get("rec_texts") or [])
    scores = list(page.get("rec_scores") or [])
    boxes = list(page.get("rec_boxes") or page.get("dt_polys") or page.get("rec_polys") or [])
    raw_texts: list[str] = []
    kept: list[str] = []
    kept_scores: list[float] = []
    blocks: list[dict[str, Any]] = []
    for index, value in enumerate(texts):
        text_value = str(value or "").strip()
        if not text_value:
            continue
        raw_texts.append(text_value)
        score = float(scores[index]) if index < len(scores) and scores[index] is not None else 0.0
        if score >= MIN_SCORE:
            kept.append(text_value)
            kept_scores.append(score)
            blocks.append({
                "text": text_value,
                "score": score,
                "bbox": _bbox(boxes[index]) if index < len(boxes) else None,
                "order": len(blocks),
            })
    return {
        "ocr_text": "\n".join(kept),
        "ocr_raw_text": "\n".join(raw_texts),
        "ocr_blocks": blocks,
        "max_score": max((float(item or 0) for item in scores), default=0.0),
        "mean_score": (sum(kept_scores) / len(kept_scores)) if kept_scores else 0.0,
        "min_score": min(kept_scores, default=0.0),
        "line_count": len(kept),
        "status": "ok" if kept else "no_text",
        "model": os.getenv("OCR_MODEL_VERSION", "paddleocr"),
    }


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json(404, {"error": "not_found"})
            return
        self._json(200, {"ok": True, "loaded": _ocr is not None, "error": _ocr_error})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/extract":
            self._json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            items = payload.get("items") or []
            if not isinstance(items, list) or len(items) > 16:
                self._json(422, {"error": "items must contain at most 16 images"})
                return
            ocr = _get_ocr()
            output = []
            for item in items:
                asset_id = item.get("asset_id")
                try:
                    image_path = _download(str(item.get("image") or ""), asset_id)
                    # PaddleOCR's predictor is not thread-safe; serialize inference
                    # while still allowing the HTTP server to accept requests.
                    with _lock:
                        prediction = ocr.predict(str(image_path))
                    output.append({"asset_id": asset_id, **_parse(prediction)})
                except Exception as exc:
                    output.append({"asset_id": asset_id, "status": "error", "error": str(exc)[:500]})
            self._json(200, {"items": output})
        except Exception as exc:
            self._json(503, {"error": str(exc)[:1000]})

    def log_message(self, *_args: Any) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(os.getenv("OCR_PORT", "7011"))), Handler).serve_forever()
