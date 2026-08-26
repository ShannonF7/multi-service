"""Isolated Qwen3-VL-Reranker service.

This process intentionally runs outside the main 7001 NLP environment.  It is
bound to loopback and only scores supplied query/document pairs; it does not
write candidates or publish graph facts.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(
    os.getenv(
        "QWEN_VL_RERANKER_MODEL",
        "/home/zhangbi/Zhangbi_Traveler/multimodal/Qwen3-VL-Reranker-2B",
    )
).expanduser()
MODEL_ROOT = MODEL_PATH
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

_model: Any = None
_model_error: Optional[str] = None
_model_lock = threading.Lock()

app = FastAPI(title="Qwen3-VL Reranker", version="0.1.0")


class ScoreRequest(BaseModel):
    instruction: str = "Retrieve evidence relevant to the user's claim."
    query: Dict[str, Any]
    documents: List[Dict[str, Any]] = Field(min_length=1, max_length=32)
    fps: float = Field(default=1.0, ge=0.1, le=8.0)
    max_frames: int = Field(default=8, ge=1, le=32)


def _get_model() -> Any:
    global _model, _model_error
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            # Import lazily so /health remains useful when the model is still
            # loading and so this module can be imported by deployment checks.
            from scripts.qwen3_vl_reranker import Qwen3VLReranker

            kwargs: Dict[str, Any] = {}
            # bfloat16 is supported by the dedicated environment and reduces
            # memory pressure on the 2B model.  Keep attention implementation
            # at the model default unless explicitly requested.
            import torch

            if torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.bfloat16
            _model = Qwen3VLReranker(model_name_or_path=str(MODEL_PATH), **kwargs)
            _model_error = None
            return _model
        except Exception as exc:  # pragma: no cover - hardware/model dependent
            _model_error = f"{type(exc).__name__}: {exc}"
            raise


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "loaded": _model is not None,
        "error": _model_error,
    }


@app.post("/score")
def score(request: ScoreRequest) -> Dict[str, Any]:
    query = request.query or {}
    if not any(query.get(key) for key in ("text", "image", "video")):
        raise HTTPException(status_code=422, detail="query requires text, image, or video")
    for index, document in enumerate(request.documents):
        if not any(document.get(key) for key in ("text", "image", "video")):
            raise HTTPException(
                status_code=422,
                detail=f"document[{index}] requires text, image, or video",
            )

    try:
        model = _get_model()
        outputs = model.process(
            {
                "instruction": request.instruction,
                "query": query,
                "documents": request.documents,
                "fps": request.fps,
                "max_frames": request.max_frames,
            }
        )
        scores = [float(value.item() if hasattr(value, "item") else value) for value in outputs]
        return {
            "scores": scores,
            "count": len(scores),
            "model": MODEL_PATH.name,
            "device": str(getattr(model, "device", "unknown")),
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - hardware/model dependent
        raise HTTPException(status_code=503, detail=f"reranker unavailable: {exc}") from exc
