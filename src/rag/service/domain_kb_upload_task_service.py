"""Chunked upload tasks for domain pre-knowledge files."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.rag.service.domain_kb_service import UPLOAD_ROOT, _safe_filename, ingest_domain_kb_files
from src.rag.service.embedding_job_service import enqueue_domain_kb_embedding_jobs


TASK_ROOT = UPLOAD_ROOT / "_upload_tasks"
TASK_ROOT.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_dir(upload_id: str) -> Path:
    upload_id = str(upload_id or "").strip()
    if not upload_id or "/" in upload_id or "\\" in upload_id or ".." in upload_id:
        raise ValueError("invalid upload_id")
    return TASK_ROOT / upload_id


def _task_path(upload_id: str) -> Path:
    return _task_dir(upload_id) / "task.json"


def _read_task(upload_id: str) -> dict[str, Any]:
    path = _task_path(upload_id)
    if not path.exists():
        raise FileNotFoundError("upload task not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_task(task: dict[str, Any]) -> dict[str, Any]:
    task["updated_at"] = _now()
    path = _task_path(str(task["upload_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return task


def _chunk_files(task: dict[str, Any]) -> list[Path]:
    chunks_dir = _task_dir(str(task["upload_id"])) / "chunks"
    return sorted(chunks_dir.glob("*.part"))


def _refresh_upload_progress(task: dict[str, Any]) -> dict[str, Any]:
    files = _chunk_files(task)
    task["received_chunks"] = len(files)
    task["uploaded_bytes"] = sum(path.stat().st_size for path in files if path.exists())
    total_chunks = int(task.get("total_chunks") or 0)
    if total_chunks:
        task["progress"] = min(100, int(task["received_chunks"] * 100 / total_chunks))
    return task


def create_upload_task(
    *,
    source_scenic_id: str,
    filename: str,
    total_size: int,
    total_chunks: int,
    source_scenic_pk: int | None = None,
    scenic_name: str | None = None,
    submitted_by: str | None = None,
) -> dict[str, Any]:
    source_scenic_id = str(source_scenic_id or "").strip()
    if not source_scenic_id:
        raise ValueError("source_scenic_id is required")
    if total_chunks < 1:
        raise ValueError("total_chunks must be greater than 0")
    upload_id = uuid.uuid4().hex
    safe_name = _safe_filename(filename)
    task = {
        "upload_id": upload_id,
        "source_scenic_id": source_scenic_id,
        "source_scenic_pk": source_scenic_pk,
        "scenic_name": scenic_name or "",
        "submitted_by": submitted_by or "",
        "filename": safe_name,
        "original_filename": filename or safe_name,
        "total_size": int(total_size or 0),
        "total_chunks": int(total_chunks),
        "received_chunks": 0,
        "uploaded_bytes": 0,
        "progress": 0,
        "status": "UPLOADING",
        "stage": "uploading",
        "message": "uploading",
        "result": None,
        "error_message": "",
        "created_at": _now(),
        "updated_at": _now(),
    }
    (_task_dir(upload_id) / "chunks").mkdir(parents=True, exist_ok=True)
    return _write_task(task)


def save_upload_chunk(upload_id: str, chunk_index: int, content: bytes) -> dict[str, Any]:
    task = _read_task(upload_id)
    if task.get("status") not in {"UPLOADING", "UPLOADED"}:
        raise ValueError(f"task status is {task.get('status')}, cannot upload chunk")
    total_chunks = int(task.get("total_chunks") or 0)
    chunk_index = int(chunk_index)
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise ValueError("chunk_index out of range")
    chunks_dir = _task_dir(upload_id) / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunks_dir / f"{chunk_index:08d}.part"
    chunk_path.write_bytes(content or b"")
    task = _refresh_upload_progress(task)
    if int(task.get("received_chunks") or 0) >= total_chunks:
        task["status"] = "UPLOADED"
        task["stage"] = "uploaded"
        task["message"] = "all chunks received"
        task["progress"] = 100
    return _write_task(task)


def assemble_upload_task(upload_id: str) -> dict[str, Any]:
    task = _read_task(upload_id)
    total_chunks = int(task.get("total_chunks") or 0)
    chunks_dir = _task_dir(upload_id) / "chunks"
    missing = [idx for idx in range(total_chunks) if not (chunks_dir / f"{idx:08d}.part").exists()]
    if missing:
        raise ValueError(f"missing chunks: {missing[:10]}")
    assembled_dir = _task_dir(upload_id) / "assembled"
    assembled_dir.mkdir(parents=True, exist_ok=True)
    file_path = assembled_dir / str(task.get("filename") or "upload")
    with file_path.open("wb") as out:
        for idx in range(total_chunks):
            with (chunks_dir / f"{idx:08d}.part").open("rb") as part:
                shutil.copyfileobj(part, out, length=1024 * 1024)
    task["assembled_path"] = str(file_path)
    task["status"] = "PROCESSING"
    task["stage"] = "queued"
    task["message"] = "queued for processing"
    task["progress"] = 100
    return _write_task(task)


def process_upload_task(upload_id: str) -> dict[str, Any]:
    task = _read_task(upload_id)
    try:
        task["status"] = "PROCESSING"
        task["stage"] = "extracting"
        task["message"] = "extracting and chunking"
        _write_task(task)

        file_path = Path(str(task.get("assembled_path") or ""))
        if not file_path.exists():
            task = assemble_upload_task(upload_id)
            file_path = Path(str(task.get("assembled_path") or ""))
        content = file_path.read_bytes()
        result = ingest_domain_kb_files(
            source_scenic_id=str(task.get("source_scenic_id") or ""),
            source_scenic_pk=task.get("source_scenic_pk"),
            scenic_name=task.get("scenic_name") or None,
            submitted_by=task.get("submitted_by") or None,
            files=[(str(task.get("filename") or file_path.name), content)],
        )
        source_ids = [item.get("doc_id") for item in result.get("files", []) if item.get("chunks", 0) > 0]
        if source_ids:
            task["stage"] = "embedding_queued"
            task["message"] = "embedding jobs queued"
            result["embedding_jobs"] = enqueue_domain_kb_embedding_jobs(str(task.get("source_scenic_id") or ""), source_ids)
        task["status"] = "COMPLETED"
        task["stage"] = "completed"
        task["message"] = "completed"
        task["result"] = result
        task["error_message"] = ""
    except Exception as exc:
        task["status"] = "FAILED"
        task["stage"] = "failed"
        task["message"] = "failed"
        task["error_message"] = str(exc)
    return _write_task(task)


def get_upload_task(upload_id: str) -> dict[str, Any]:
    task = _read_task(upload_id)
    if task.get("status") in {"UPLOADING", "UPLOADED"}:
        task = _refresh_upload_progress(task)
        _write_task(task)
    return task


def list_upload_tasks(source_scenic_id: str, *, limit: int = 50) -> dict[str, Any]:
    source_scenic_id = str(source_scenic_id or "").strip()
    items: list[dict[str, Any]] = []
    for path in sorted(TASK_ROOT.glob("*/task.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if source_scenic_id and str(task.get("source_scenic_id") or "") != source_scenic_id:
            continue
        items.append(task)
        if len(items) >= limit:
            break
    return {"items": items, "total": len(items)}
