"""Semantic batch completion job persistence and aggregation."""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.service.evidence_store import apply_semantic_completion_schema
from src.rag.service.semantic_candidate_store import apply_semantic_candidate_schema

RUNNING_STATUSES = {"PENDING", "RUNNING"}
DONE_STATUSES = {"DONE", "COMPLETED"}
CONFLICT_CLASSES = {"conflicting", "scope_mismatch", "entity_ambiguity", "value_conflict", "temporal_conflict", "exclusive_relation_conflict"}
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_ADVISORY_LOCK_NAME = "semantic_completion_schema"


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def apply_semantic_batch_schema(db) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        apply_semantic_completion_schema(db)
        apply_semantic_candidate_schema(db)
        try:
            db.execute(
                text("select pg_advisory_xact_lock(hashtext(:lock_name))"),
                {"lock_name": _SCHEMA_ADVISORY_LOCK_NAME},
            )
            db.execute(text("""
                create table if not exists semantic_batch_jobs (
                    id bigserial primary key,
                    batch_uid text not null unique,
                    source_scenic_id text not null,
                    scope text,
                    status text not null default 'RUNNING',
                    total_nodes integer not null default 0,
                    node_ids jsonb not null default '[]'::jsonb,
                    metadata jsonb not null default '{}'::jsonb,
                    created_by text,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now(),
                    finished_at timestamptz
                )
            """))
            db.execute(text("""
                create table if not exists semantic_batch_job_items (
                    id bigserial primary key,
                    batch_id bigint not null references semantic_batch_jobs(id) on delete cascade,
                    job_id bigint not null,
                    source_node_id text not null,
                    source_node_name text,
                    metadata jsonb not null default '{}'::jsonb,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now(),
                    unique(batch_id, job_id)
                )
            """))
            db.execute(text("create index if not exists idx_semantic_batch_jobs_scenic on semantic_batch_jobs(source_scenic_id, created_at desc)"))
            db.execute(text("create index if not exists idx_semantic_batch_items_batch on semantic_batch_job_items(batch_id)"))
            db.execute(text("create index if not exists idx_semantic_batch_items_job on semantic_batch_job_items(job_id)"))
            db.commit()
        except Exception:
            db.rollback()
            raise
        _SCHEMA_READY = True


def create_semantic_batch_job(payload: dict[str, Any], *, created_by: str | None = None) -> dict[str, Any]:
    source_scenic_id = str(payload.get("source_scenic_id") or payload.get("scenic_id") or "").strip()
    if not source_scenic_id:
        raise ValueError("source_scenic_id is required")
    node_ids = payload.get("node_ids") or []
    if not isinstance(node_ids, list):
        node_ids = [node_ids]
    node_ids = [str(item) for item in node_ids if str(item).strip()]
    total_nodes = int(payload.get("total_nodes") or len(node_ids) or 0)
    scope = str(payload.get("scope") or payload.get("batch_scope") or "").strip() or None
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    batch_uid = str(payload.get("batch_uid") or uuid.uuid4().hex)
    with ai_session_scope() as db:
        apply_semantic_batch_schema(db)
        row = db.execute(
            text("""
                insert into semantic_batch_jobs (
                    batch_uid, source_scenic_id, scope, status, total_nodes,
                    node_ids, metadata, created_by, created_at, updated_at
                ) values (
                    :batch_uid, :source_scenic_id, :scope, 'RUNNING', :total_nodes,
                    cast(:node_ids as jsonb), cast(:metadata as jsonb), :created_by, now(), now()
                )
                returning id, batch_uid, source_scenic_id, scope, status, total_nodes,
                          node_ids, metadata, created_by, created_at, updated_at, finished_at
            """),
            {
                "batch_uid": batch_uid,
                "source_scenic_id": source_scenic_id,
                "scope": scope,
                "total_nodes": total_nodes,
                "node_ids": _json(node_ids),
                "metadata": _json(metadata),
                "created_by": created_by,
            },
        ).mappings().first()
        return _hydrate_batch(db, dict(row))


def attach_semantic_batch_item(batch_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    job_id = int(payload.get("job_id") or payload.get("id") or 0)
    source_node_id = str(payload.get("source_node_id") or "").strip()
    if not job_id:
        raise ValueError("job_id is required")
    if not source_node_id:
        raise ValueError("source_node_id is required")
    with ai_session_scope() as db:
        apply_semantic_batch_schema(db)
        batch = db.execute(text("select * from semantic_batch_jobs where id=:id"), {"id": int(batch_id)}).mappings().first()
        if not batch:
            raise LookupError("batch job not found")
        db.execute(
            text("""
                insert into semantic_batch_job_items (
                    batch_id, job_id, source_node_id, source_node_name, metadata, created_at, updated_at
                ) values (
                    :batch_id, :job_id, :source_node_id, :source_node_name, cast(:metadata as jsonb), now(), now()
                )
                on conflict (batch_id, job_id) do update set
                    source_node_id = excluded.source_node_id,
                    source_node_name = excluded.source_node_name,
                    metadata = excluded.metadata,
                    updated_at = now()
            """),
            {
                "batch_id": int(batch_id),
                "job_id": job_id,
                "source_node_id": source_node_id,
                "source_node_name": payload.get("source_node_name") or payload.get("node_name") or "",
                "metadata": _json(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
            },
        )
        row = db.execute(text("select * from semantic_batch_jobs where id=:id"), {"id": int(batch_id)}).mappings().first()
        return _hydrate_batch(db, dict(row))


def get_semantic_batch_job(batch_id: int) -> dict[str, Any] | None:
    with ai_session_scope() as db:
        apply_semantic_batch_schema(db)
        row = db.execute(text("select * from semantic_batch_jobs where id=:id"), {"id": int(batch_id)}).mappings().first()
        return _hydrate_batch(db, dict(row)) if row else None


def list_semantic_batch_jobs(*, source_scenic_id: str | None = None, status: str | None = None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))
    where = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if source_scenic_id:
        where.append("source_scenic_id = :source_scenic_id")
        params["source_scenic_id"] = str(source_scenic_id)
    if status:
        where.append("status = :status")
        params["status"] = str(status).upper()
    where_sql = " where " + " and ".join(where) if where else ""
    with ai_session_scope() as db:
        apply_semantic_batch_schema(db)
        total = db.execute(text("select count(*) as n from semantic_batch_jobs" + where_sql), params).mappings().first()["n"]
        rows = db.execute(
            text("select * from semantic_batch_jobs" + where_sql + " order by created_at desc, id desc limit :limit offset :offset"),
            params,
        ).mappings().all()
        return {"items": [_hydrate_batch(db, dict(row), include_items=False) for row in rows], "total": int(total or 0)}


def _batch_items(db, batch_id: int) -> list[dict[str, Any]]:
    rows = db.execute(
        text("""
            select i.id, i.batch_id, i.job_id, i.source_node_id, i.source_node_name,
                   i.metadata, i.created_at, i.updated_at,
                   j.status as job_status, j.progress, j.current_stage, j.error_message,
                   j.candidate_count, j.conflict_count, j.created_at as job_created_at,
                   j.finished_at as job_finished_at
            from semantic_batch_job_items i
            left join semantic_completion_jobs j on j.id = i.job_id
            where i.batch_id = :batch_id
            order by i.created_at asc, i.id asc
        """),
        {"batch_id": int(batch_id)},
    ).mappings().all()
    return [dict(row) for row in rows]


def _hydrate_batch(db, row: dict[str, Any], *, include_items: bool = True) -> dict[str, Any]:
    items = _batch_items(db, int(row["id"]))
    total_nodes = int(row.get("total_nodes") or 0)
    submitted = len(items)
    queued = max(total_nodes - submitted, 0)
    completed = sum(1 for item in items if str(item.get("job_status") or "").upper() in DONE_STATUSES)
    failed = sum(1 for item in items if str(item.get("job_status") or "").upper() == "FAILED")
    running = sum(1 for item in items if str(item.get("job_status") or "").upper() in RUNNING_STATUSES) + queued
    job_ids = [int(item["job_id"]) for item in items if item.get("job_id")]
    candidate_count = 0
    conflict_count = 0
    if job_ids:
        counts = db.execute(
            text("""
                select count(*) as candidate_count,
                       count(*) filter (
                           where status = 'CONFLICT'
                              or conflict_class in ('conflicting','scope_mismatch','entity_ambiguity','value_conflict','temporal_conflict','exclusive_relation_conflict')
                       ) as conflict_count
                from semantic_claim_candidates
                where job_id = any(:job_ids)
            """),
            {"job_ids": job_ids},
        ).mappings().first()
        candidate_count = int(counts["candidate_count"] or 0)
        conflict_count = int(counts["conflict_count"] or 0)
    denom = max(total_nodes, submitted, 1)
    progress = int(round(((completed + failed) / denom) * 100))
    if total_nodes == 0 and submitted == 0:
        status = "EMPTY"
    elif completed + failed >= denom and failed:
        status = "PARTIAL_FAILED"
    elif completed >= denom:
        status = "COMPLETED"
    elif running or submitted:
        status = "RUNNING"
    else:
        status = "PENDING"
    finished_at_sql = "now()" if status in {"COMPLETED", "PARTIAL_FAILED"} else "null"
    db.execute(
        text("""
            update semantic_batch_jobs
            set status=:status, updated_at=now(), finished_at=case when :finished then coalesce(finished_at, now()) else null end
            where id=:id
        """),
        {"id": int(row["id"]), "status": status, "finished": status in {"COMPLETED", "PARTIAL_FAILED"}},
    )
    result = dict(row)
    result.update({
        "status": status,
        "progress": progress,
        "submitted_count": submitted,
        "queued_count": queued,
        "running_count": running,
        "completed_count": completed,
        "failed_count": failed,
        "candidate_count": candidate_count,
        "conflict_count": conflict_count,
    })
    if include_items:
        result["items"] = items
    return result
