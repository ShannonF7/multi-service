"""Persist batched OCR results for synchronized image assets."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.semantic_growth.ocr_client import extract_ocr_batch


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


def _content_hash(row: dict[str, Any], ocr_text: str) -> str:
    base = str(row.get("content_hash") or row.get("file_hash") or row.get("url") or "")
    return hashlib.sha256(f"{base}|ocr:{ocr_text}".encode("utf-8")).hexdigest()


def process_image_ocr_batch(
    *,
    source_scenic_id: str,
    asset_ids: list[str] | None = None,
    limit: int = 16,
) -> dict[str, Any]:
    """OCR assets that do not yet have durable OCR text.

    The select and update are separate transactions deliberately: PaddleOCR is
    an external process and must not hold an AI_DB transaction open while it
    downloads and analyses images.
    """
    normalized_ids = [str(value).strip() for value in (asset_ids or []) if str(value).strip()]
    capped_limit = max(1, min(int(limit or 16), 16))
    with ai_session_scope() as db:
        where = [
            "source_scenic_id=:source_scenic_id",
            "asset_type='image'",
            "coalesce(url, '') <> ''",
            "coalesce(ocr_text, '') = ''",
        ]
        params: dict[str, Any] = {
            "source_scenic_id": str(source_scenic_id),
            "limit": capped_limit,
        }
        if normalized_ids:
            where.append("source_asset_id = any(:asset_ids)")
            params["asset_ids"] = normalized_ids
        rows = [
            dict(row)
            for row in db.execute(
                text(
                    """
                    select id, source_asset_id, url, file_hash, content_hash,
                           coalesce(metadata, '{}'::jsonb) as metadata
                    from node_assets
                    where """
                    + " and ".join(where)
                    + " order by id asc limit :limit"
                ),
                params,
            ).mappings().all()
        ]

    if not rows:
        return {"status": "completed", "source_scenic_id": str(source_scenic_id), "items": []}

    extracted = extract_ocr_batch(
        [{"asset_id": row["id"], "image": row.get("url")} for row in rows]
    )
    items: list[dict[str, Any]] = []
    with ai_session_scope() as db:
        for row in rows:
            result = dict(extracted.get(int(row["id"])) or {})
            status = str(result.get("status") or "error").upper()
            ocr_text = str(result.get("ocr_text") or "").strip()
            if status == "OK" and ocr_text:
                next_status = "SUCCEEDED"
                error = ""
            elif status == "NO_TEXT":
                next_status = "NO_TEXT"
                error = ""
            else:
                next_status = "FAILED"
                error = str(result.get("error") or "OCR service returned no result")[:4000]
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    "ocr_status": next_status,
                    "ocr_model": str(result.get("model") or "paddleocr")[:128],
                    "ocr_max_score": float(result.get("max_score") or 0.0),
                    "ocr_mean_score": float(result.get("mean_score") or 0.0),
                    "ocr_min_score": float(result.get("min_score") or 0.0),
                    "ocr_line_count": int(result.get("line_count") or 0),
                    "ocr_error": error,
                }
            )
            update_params = {
                "id": int(row["id"]),
                "ocr_text": ocr_text,
                "file_hash": row.get("file_hash"),
                "content_hash": _content_hash(row, ocr_text) if ocr_text else row.get("content_hash"),
                "metadata": _json(metadata),
            }
            db.execute(
                text(
                    """
                    update node_assets
                    set ocr_text=:ocr_text,
                        content_hash=:content_hash,
                        metadata=cast(:metadata as jsonb),
                        updated_at=now()
                    where id=:id
                    """
                ),
                update_params,
            )
            items.append(
                {
                    "asset_id": row["source_asset_id"],
                    "status": "OK" if next_status == "SUCCEEDED" else next_status,
                    "ocr_text": ocr_text,
                    "max_score": float(result.get("max_score") or 0.0),
                    "mean_score": float(result.get("mean_score") or 0.0),
                    "min_score": float(result.get("min_score") or 0.0),
                    "line_count": int(result.get("line_count") or 0),
                    "model": str(result.get("model") or "paddleocr"),
                    "error": error,
                }
            )
    return {
        "status": "completed",
        "source_scenic_id": str(source_scenic_id),
        "items": items,
    }


def process_image_ocr_urls(items: list[dict[str, Any]], *, limit: int = 16) -> dict[str, Any]:
    """OCR image URLs that are not yet bound to a B-side node asset."""
    batch = [
        {"asset_id": item.get("asset_id"), "image": item.get("image")}
        for item in items
        if isinstance(item, dict) and item.get("asset_id") is not None and item.get("image")
    ][: max(1, min(int(limit or 16), 16))]
    extracted = extract_ocr_batch(batch)
    result_items = []
    for item in batch:
        result = dict(extracted.get(int(item["asset_id"])) or {})
        status = str(result.get("status") or "error").upper()
        ocr_text = str(result.get("ocr_text") or "").strip()
        if status == "OK" and ocr_text:
            status = "OK"
        elif status != "NO_TEXT":
            status = "ERROR"
        result_items.append(
            {
                "asset_id": item["asset_id"],
                "status": status,
                "ocr_text": ocr_text,
                "max_score": float(result.get("max_score") or 0.0),
                "mean_score": float(result.get("mean_score") or 0.0),
                "min_score": float(result.get("min_score") or 0.0),
                "line_count": int(result.get("line_count") or 0),
                "model": str(result.get("model") or "paddleocr"),
                "error": str(result.get("error") or "")[:4000],
            }
        )
    return {"status": "completed", "items": result_items}
