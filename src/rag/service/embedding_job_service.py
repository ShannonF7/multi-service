"""Domain KB embedding job queue service."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.rag.dependencies import ai_session_scope

RAG_DIR = Path(__file__).resolve().parents[1]
MIGRATION_FILE = RAG_DIR / "migrations" / "20260707_domain_kb_embedding_jobs.sql"
_SCHEMA_READY = False


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _execute_statements(db: Session, raw: str) -> None:
    current: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            sql = "\n".join(current).rstrip(";").strip()
            if sql:
                db.execute(text(sql))
            current = []
    if current:
        sql = "\n".join(current).strip()
        if sql:
            db.execute(text(sql))


def apply_embedding_job_schema(db: Session) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    if not MIGRATION_FILE.exists():
        raise RuntimeError(f"Missing migration file: {MIGRATION_FILE}")
    # Serialize migration checks across API workers. CREATE INDEX IF NOT EXISTS
    # still takes relation locks; concurrent migration attempts can deadlock with
    # job enqueue/claim traffic on a hot production queue.
    db.execute(text("select pg_advisory_xact_lock(hashtext('domain_kb_embedding_jobs_schema'))"))
    _execute_statements(db, MIGRATION_FILE.read_text(encoding="utf-8"))
    _SCHEMA_READY = True


def _is_transient_lock_error(exc: Exception) -> bool:
    text_value = str(exc).lower()
    return (
        "deadlock" in text_value
        or "deadlockdetected" in text_value
        or "40p01" in text_value
        or "lock timeout" in text_value
        or "locktimeout" in text_value
        or "55p03" in text_value
    )


def enqueue_domain_kb_embedding_jobs(source_scenic_id: str, source_ids: list[str], *, priority: int = 100) -> dict[str, Any]:
    source_ids = [str(x or "").strip() for x in source_ids if str(x or "").strip()]
    if not source_scenic_id or not source_ids:
        return {"queued": 0, "source_scenic_id": source_scenic_id, "source_ids": []}

    # Apply schema in its own short transaction, not inside the bulk enqueue loop.
    with ai_session_scope() as db:
        apply_embedding_job_schema(db)

    queued = 0
    skipped_running = 0
    skipped_locked = 0
    missing_chunks = 0
    errors: list[dict[str, str]] = []

    for source_id in source_ids:
        try:
            with ai_session_scope() as db:
                # Keep each source in a small transaction. Lock order is always:
                # job row first, chunks second. Workers follow the same order.
                db.execute(text("set local lock_timeout = '1000ms'"))
                total_chunks = db.execute(
                    text("""
                        select count(*) from knowledge_chunks
                        where source_scenic_id=:source_scenic_id
                          and source_type='domain_kb'
                          and source_id=:source_id
                    """),
                    {"source_scenic_id": source_scenic_id, "source_id": source_id},
                ).scalar() or 0
                if not total_chunks:
                    missing_chunks += 1
                    continue

                inserted = db.execute(
                    text("""
                        insert into domain_kb_embedding_jobs (
                            job_id, source_scenic_id, source_id, status, priority, attempts,
                            total_chunks, embedded_chunks, error_message, metadata, created_at, updated_at
                        ) values (
                            cast(:job_id as uuid), :source_scenic_id, :source_id, 'PENDING', :priority, 0,
                            :total_chunks, 0, null, cast(:metadata as jsonb), now(), now()
                        )
                        on conflict (source_scenic_id, source_id) do nothing
                        returning id
                    """),
                    {
                        "job_id": str(uuid.uuid4()),
                        "source_scenic_id": source_scenic_id,
                        "source_id": source_id,
                        "priority": priority,
                        "total_chunks": int(total_chunks),
                        "metadata": _json({}),
                    },
                ).mappings().first()

                if inserted:
                    should_mark_pending = True
                else:
                    row = db.execute(
                        text("""
                            update domain_kb_embedding_jobs
                            set status = 'PENDING',
                                priority = :priority,
                                attempts = 0,
                                total_chunks = :total_chunks,
                                embedded_chunks = 0,
                                error_message = null,
                                worker_id = null,
                                device = null,
                                started_at = null,
                                finished_at = null,
                                updated_at = now()
                            where source_scenic_id=:source_scenic_id
                              and source_id=:source_id
                              and status <> 'RUNNING'
                            returning id
                        """),
                        {
                            "source_scenic_id": source_scenic_id,
                            "source_id": source_id,
                            "priority": priority,
                            "total_chunks": int(total_chunks),
                        },
                    ).mappings().first()
                    should_mark_pending = bool(row)
                    if not row:
                        skipped_running += 1

                if should_mark_pending:
                    db.execute(
                        text("""
                            update knowledge_chunks
                            set metadata = jsonb_set(coalesce(metadata, '{}'::jsonb), '{embedding_status}', '"pending"'::jsonb, true)
                            where source_scenic_id=:source_scenic_id
                              and source_type='domain_kb'
                              and source_id=:source_id
                        """),
                        {"source_scenic_id": source_scenic_id, "source_id": source_id},
                    )
                    queued += 1
        except OperationalError as exc:
            if _is_transient_lock_error(exc):
                skipped_locked += 1
                errors.append({"source_id": source_id, "error": str(exc)[:300]})
                continue
            raise

    return {
        "queued": queued,
        "source_scenic_id": source_scenic_id,
        "source_ids": source_ids,
        "skipped_running": skipped_running,
        "skipped_locked": skipped_locked,
        "missing_chunks": missing_chunks,
        "errors": errors[:20],
    }



def _is_deadlock_error(exc: Exception) -> bool:
    text_value = str(exc).lower()
    return _is_transient_lock_error(exc)


def claim_next_embedding_job(*, worker_id: str, device: str = "", retries: int = 3) -> dict[str, Any] | None:
    for attempt in range(max(1, retries)):
        try:
            with ai_session_scope() as db:
                # Schema migrations are applied when jobs are enqueued/listed. Avoid
                # running DDL in every worker polling transaction; concurrent DDL plus
                # row claiming is a common source of deadlocks.
                picked = db.execute(
                    text("""
                        select id
                        from domain_kb_embedding_jobs
                        where status in ('PENDING', 'FAILED')
                          and attempts < max_attempts
                        order by priority asc, created_at asc
                        for update skip locked
                        limit 1
                    """),
                ).mappings().first()
                if not picked:
                    return None
                row = db.execute(
                    text("""
                        update domain_kb_embedding_jobs
                        set status='RUNNING', attempts=attempts+1, worker_id=:worker_id, device=:device,
                            started_at=now(), updated_at=now(), error_message=null
                        where id=:job_pk
                        returning id, job_id, source_scenic_id, source_id, attempts, total_chunks
                    """),
                    {"worker_id": worker_id, "device": device, "job_pk": int(picked["id"])},
                ).mappings().first()
            return dict(row) if row else None
        except OperationalError as exc:
            if not _is_deadlock_error(exc) or attempt >= retries - 1:
                raise
            time.sleep(0.2 * (attempt + 1))
    return None

def complete_embedding_job(job_pk: int, *, status: str, embedded_chunks: int = 0, model_name: str = "", error_message: str = "") -> None:
    with ai_session_scope() as db:
        db.execute(
            text("""
                update domain_kb_embedding_jobs
                set status=:status,
                    embedded_chunks=:embedded_chunks,
                    model_name=:model_name,
                    error_message=:error_message,
                    finished_at=now(),
                    updated_at=now()
                where id=:job_pk
            """),
            {
                "job_pk": job_pk,
                "status": status,
                "embedded_chunks": embedded_chunks,
                "model_name": model_name,
                "error_message": error_message[:2000],
            },
        )


def list_embedding_jobs(source_scenic_id: str, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    with ai_session_scope() as db:
        apply_embedding_job_schema(db)
        total = db.execute(
            text("select count(*) from domain_kb_embedding_jobs where source_scenic_id=:source_scenic_id"),
            {"source_scenic_id": source_scenic_id},
        ).scalar() or 0
        rows = db.execute(
            text("""
                select job_id::text, source_scenic_id, source_id, status, attempts, max_attempts,
                       device, total_chunks, embedded_chunks, model_name, error_message,
                       created_at, started_at, finished_at, updated_at
                from domain_kb_embedding_jobs
                where source_scenic_id=:source_scenic_id
                order by created_at desc
                limit :limit offset :offset
            """),
            {"source_scenic_id": source_scenic_id, "limit": limit, "offset": offset},
        ).mappings().all()
    return {"items": [dict(row) for row in rows], "total": int(total)}


def process_one_embedding_job(*, worker_id: str, device: str = "") -> dict[str, Any] | None:
    job = claim_next_embedding_job(worker_id=worker_id, device=device)
    if not job:
        return None
    try:
        if device:
            import os
            os.environ["EMBEDDING_DEVICE"] = device
        from src.rag.service.embedding_service import embed_domain_kb_document

        result = embed_domain_kb_document(job["source_scenic_id"], job["source_id"])
        status = "SUCCESS" if result.get("status") == "done" else "FAILED"
        complete_embedding_job(
            int(job["id"]),
            status=status,
            embedded_chunks=int(result.get("embedded") or 0),
            model_name=str(result.get("model_name") or ""),
            error_message=str(result.get("error") or ""),
        )
        return {"job": job, "result": result, "status": status}
    except Exception as exc:
        complete_embedding_job(int(job["id"]), status="FAILED", error_message=str(exc))
        return {"job": job, "result": {"status": "failed", "error": str(exc)}, "status": "FAILED"}
