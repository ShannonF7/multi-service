"""Lightweight router for A端一键入库.

This module is intentionally separated from src.rag.router. The historical
router also owns hybrid-search startup logic and BGE model loading; the one-click
sync endpoints only need schema validation plus AI_DB writes, so importing this
router keeps the main Travel API startup cheap and predictable.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException

from src.rag.schemas import ScenicSyncPayload, ScenicSyncResponse, SyncJobStatusResponse
from src.rag.service.sync_service import get_job_status, sync_scenic_service
from src.rag.service.image_ocr_service import process_image_ocr_batch, process_image_ocr_urls

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/sync/scenic", response_model=ScenicSyncResponse)
async def sync_scenic(payload: ScenicSyncPayload):
    """Receive one scenic-area payload from A side and upsert AI_DB RAG data."""
    try:
        return await sync_scenic_service(payload)
    except Exception as exc:
        logger.error("Scenic sync failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sync/jobs/{job_id}", response_model=SyncJobStatusResponse)
async def sync_job_status(job_id: str):
    """Return status and recent events for one A端一键入库 job."""
    result = await get_job_status(job_id)
    if result.get("status") == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="sync job not found")
    return result


@router.post("/sync/image-ocr")
def sync_image_ocr(payload: dict = Body(default={} )):
    """Batch OCR synchronized image assets and persist the result in AI_DB."""
    source_scenic_id = str(payload.get("source_scenic_id") or "").strip()
    if not source_scenic_id:
        raise HTTPException(status_code=400, detail="source_scenic_id is required")
    asset_ids = payload.get("asset_ids")
    if asset_ids is not None and not isinstance(asset_ids, list):
        raise HTTPException(status_code=400, detail="asset_ids must be a list")
    try:
        return process_image_ocr_batch(
            source_scenic_id=source_scenic_id,
            asset_ids=asset_ids,
            limit=int(payload.get("limit") or 16),
        )
    except Exception as exc:
        logger.error("Image OCR sync failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/sync/image-ocr/urls")
def sync_image_ocr_urls(payload: dict = Body(default={} )):
    """OCR image URLs that are not represented by a bound node asset."""
    items = payload.get("items")
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="items must be a list")
    try:
        return process_image_ocr_urls(items, limit=int(payload.get("limit") or 16))
    except Exception as exc:
        logger.error("Image URL OCR failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
