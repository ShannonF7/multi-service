"""Loopback OCR adapter using the existing paddle_ocr environment."""

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


def _parse(result: Any) -> dict[str, Any]:
    page = result[0] if isinstance(result, list) and result else result
    if not isinstance(page, dict):
        return {"ocr_text": "", "max_score": 0.0, "status": "no_text"}
    texts = list(page.get("rec_texts") or [])
    scores = list(page.get("rec_scores") or [])
    kept = []
    kept_scores = []
    for index, value in enumerate(texts):
        value = str(value or "").strip()
        score = float(scores[index]) if index < len(scores) and scores[index] is not None else 0.0
        if value and score >= MIN_SCORE:
            kept.append(value)
            kept_scores.append(score)
    return {
        "ocr_text": "\n".join(kept),
        "max_score": max((float(item or 0) for item in scores), default=0.0),
        "mean_score": (sum(kept_scores) / len(kept_scores)) if kept_scores else 0.0,
        "min_score": min(kept_scores, default=0.0),
        "line_count": len(kept),
        "status": "ok" if kept else "no_text",
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
