"""Stable independence keys for evidence units and source groups."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def canonical_page_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    if not parts.netloc:
        return raw.rstrip("/")
    query = [(key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def source_independence_key(metadata: dict[str, Any] | None) -> str:
    metadata = metadata or {}
    source_type = str(metadata.get("source_type") or metadata.get("asset_type") or "").lower()
    nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    if source_type.startswith("image") or metadata.get("asset_id") is not None:
        asset_id = metadata.get("asset_id") or nested.get("asset_id") or metadata.get("source_asset_id")
        if asset_id is not None:
            return f"image:{asset_id}"
    document_id = (
        metadata.get("source_doc_id") or metadata.get("document_id")
        or metadata.get("source_id") or nested.get("source_doc_id")
        or nested.get("document_id")
    )
    if document_id and (source_type.startswith("domain") or metadata.get("chunk_id") is not None):
        return f"document:{document_id}"
    url = canonical_page_url(metadata.get("source_url") or metadata.get("url"))
    if url:
        return f"web:{url}"
    if document_id:
        return f"document:{document_id}"
    return str(metadata.get("evidence_unit_uid") or metadata.get("source_id") or "unknown")
