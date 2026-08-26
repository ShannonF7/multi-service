"""Async semantic completion job service with DB-backed worker leasing."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.schemas import SemanticCompleteRequest
from src.rag.service.evidence_store import (
    apply_semantic_completion_schema,
    list_semantic_completion_questions,
    list_semantic_evidence_items,
    update_semantic_completion_question_stats,
)
from src.rag.service.semantic_candidate_store import list_semantic_candidates, list_semantic_candidate_groups
from src.rag.service.semantic_completion_service import complete_semantic_service
from src.rag.service.gap_status_service import list_semantic_gap_status

logger = logging.getLogger(__name__)
PIPELINE_VERSION = "semantic-completion-v2"
DEFAULT_MAX_ATTEMPTS = 3


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _payload_dict(payload: SemanticCompleteRequest) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    return payload.dict()


def _build_job_key(payload: SemanticCompleteRequest) -> str:
    metadata = payload.metadata or {}
    source_scope = metadata.get("source_scope") or metadata.get("evidence_source_scope") or []
    canonical = {
        "scenic_id": str(payload.scenic_id),
        "source_node_id": str(payload.node.source_node_id),
        "target_fields": sorted(str(x).strip() for x in (payload.target_fields or []) if str(x).strip()),
        "relation_intents": sorted(str(x).strip() for x in (payload.relation_intents or []) if str(x).strip()),
        "message": str(payload.message or "").strip(),
        "subgraph_depth": int(payload.subgraph_depth or 0),
        "graph_scope": str((payload.graph_context or {}).get("scope") or ""),
        "source_scope": source_scope,
        "retrieval_scope": str(metadata.get("retrieval_scope") or ""),
        "completion_mode": str(metadata.get("completion_mode") or metadata.get("job_mode") or "quick").lower(),
    }
    return hashlib.sha256(_json(canonical).encode("utf-8")).hexdigest()[:32]


def _resolve_ids(payload: SemanticCompleteRequest, db) -> tuple[int, int]:
    scenic_row = db.execute(
        text("select id from scenic_areas where source_scenic_id = :sid order by id limit 1"),
        {"sid": str(payload.scenic_id)},
    ).mappings().first()
    if scenic_row:
        scenic_id = int(scenic_row["id"])
    else:
        scenic_insert = db.execute(
            text(
                """
                insert into scenic_areas (source_scenic_id, name, description, metadata)
                values (:sid, :name, :description, cast(:metadata as jsonb))
                returning id
                """
            ),
            {
                "sid": str(payload.scenic_id),
                "name": payload.node.scenic_name or str(payload.scenic_id),
                "description": "semantic completion auto-created domain shell",
                "metadata": _json({"source": "semantic_completion_job"}),
            },
        ).mappings().first()
        scenic_id = int(scenic_insert["id"])

    node_row = db.execute(
        text(
            """
            select id
            from semantic_nodes
            where scenic_id = :scenic_id and source_node_id = :source_node_id
            order by id
            limit 1
            """
        ),
        {"scenic_id": scenic_id, "source_node_id": str(payload.node.source_node_id)},
    ).mappings().first()
    if node_row:
        return scenic_id, int(node_row["id"])

    node_insert = db.execute(
        text(
            """
            insert into semantic_nodes (
                scenic_id, source_scenic_id, source_node_id, parent_source_node_id,
                node_name, node_type, description, properties,
                source_table, source_pk, source_title, created_at, updated_at
            ) values (
                :scenic_id, :source_scenic_id, :source_node_id, :parent_source_node_id,
                :node_name, :node_type, :description, cast(:properties as jsonb),
                'semantic_completion_job', :source_pk, :source_title, now(), now()
            )
            returning id
            """
        ),
        {
            "scenic_id": scenic_id,
            "source_scenic_id": str(payload.scenic_id),
            "source_node_id": str(payload.node.source_node_id),
            "parent_source_node_id": str(getattr(payload.node, "parent_source_node_id", "") or "") or None,
            "node_name": str(payload.node.name or payload.node.source_node_id or ""),
            "node_type": str(payload.node.node_type or ""),
            "description": str(payload.node.description or ""),
            "properties": _json({"auto_created_for": "semantic_completion_job"}),
            "source_pk": str(payload.node.source_node_id),
            "source_title": str(payload.node.name or payload.node.source_node_id or ""),
        },
    ).mappings().first()
    logger.warning("auto-created semantic node shell for completion job: scenic=%s node=%s", payload.scenic_id, payload.node.source_node_id)
    return scenic_id, int(node_insert["id"])


def _normalize_mode_payload(payload: SemanticCompleteRequest) -> dict[str, Any]:
    request_payload = _payload_dict(payload)
    metadata = request_payload.setdefault("metadata", {}) or {}
    mode = str(metadata.get("completion_mode") or metadata.get("job_mode") or "quick").strip().lower()
    if mode == "fast":
        mode = "quick"
    metadata["completion_mode"] = mode
    depth = int(request_payload.get("subgraph_depth") or 0)
    retrieval_scope = "self_web" if depth == 0 else ("domain" if depth < 0 else "subgraph")
    metadata["retrieval_scope"] = retrieval_scope

    if mode == "quick_web":
        metadata.setdefault("question_batch_size", 3)
        metadata.setdefault("web_search_budget_seconds", 60)
        metadata.setdefault("web_limit_per_question", 5)
        metadata.setdefault("max_candidates_per_field", 2)
        metadata.setdefault("candidate_value_policy", "group_recommend_alternatives")
        metadata["source_scope"] = ["provided_evidence", "web_search", "web_extractor"]
        metadata["web_search_policy"] = "always"
        request_payload["use_web_search"] = True
        request_payload["use_web_extractor"] = True
        request_payload["max_web_results"] = 5
    elif mode in {"deep", "web", "full", "batch"}:
        metadata.setdefault("question_batch_size", 8)
        metadata.setdefault("graph_discovery_max_questions", 3)
        metadata.setdefault("web_search_budget_seconds", 180)
        metadata.setdefault("domain_kb_limit_per_question", 8)
        metadata.setdefault("web_limit_per_question", 8)
        metadata.setdefault("web_search_policy", "always")
        metadata.setdefault("web_query_variants_per_question", 2)
        metadata.setdefault("evidence_limit_per_question", 12)
        metadata.setdefault("extractor_chunks_per_question", 5)
        metadata.setdefault("verify_existing_facts", True)
        metadata.setdefault("max_candidates_per_field", 5)
        metadata.setdefault("candidate_value_policy", "group_recommend_alternatives")
        if retrieval_scope == "self_web":
            metadata["source_scope"] = ["provided_evidence", "web_search", "web_extractor"]
            metadata["web_search_policy"] = "always"
        else:
            metadata.setdefault("source_scope", ["provided_evidence", "domain_kb", "web_search", "web_extractor"])
        request_payload["use_web_search"] = True
        request_payload["use_web_extractor"] = True
        request_payload["max_web_results"] = max(1, min(int(request_payload.get("max_web_results") or metadata.get("web_limit_per_question") or 5), 10))
    elif mode == "standard":
        metadata.setdefault("question_batch_size", 5)
        metadata.setdefault("web_search_budget_seconds", 0)
        metadata.setdefault("domain_kb_limit_per_question", 6)
        metadata.setdefault("max_candidates_per_field", 3)
        metadata.setdefault("candidate_value_policy", "group_recommend_alternatives")
        metadata["source_scope"] = ["provided_evidence"] if retrieval_scope == "self_web" else ["provided_evidence", "domain_kb"]
        request_payload["use_web_search"] = False
        request_payload["use_web_extractor"] = False
    else:
        metadata.setdefault("question_batch_size", 3)
        metadata.setdefault("web_search_budget_seconds", 0)
        metadata.setdefault("domain_kb_limit_per_question", 4)
        metadata.setdefault("max_candidates_per_field", 2)
        metadata.setdefault("candidate_value_policy", "group_recommend_alternatives")
        metadata["source_scope"] = ["provided_evidence"] if retrieval_scope == "self_web" else metadata.get("source_scope", ["provided_evidence", "domain_kb"])
        request_payload["use_web_search"] = False
        request_payload["use_web_extractor"] = False

    request_payload["metadata"] = metadata
    return request_payload


def create_or_get_semantic_completion_job(payload: SemanticCompleteRequest, *, created_by: str | None = None) -> dict[str, Any]:
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        scenic_id, node_id = _resolve_ids(payload, db)
        request_payload = _normalize_mode_payload(payload)
        normalized_payload = SemanticCompleteRequest.parse_obj(request_payload)
        job_key = _build_job_key(normalized_payload)
        max_attempts = int((request_payload.get("metadata") or {}).get("max_attempts") or DEFAULT_MAX_ATTEMPTS)

        row = db.execute(
            text(
                """
                select id, trace_id, status, progress, current_stage,
                       question_count, evidence_count, candidate_count, conflict_count,
                       source_scenic_id, source_node_id,
                       attempt_count, max_attempts, locked_by, heartbeat_at, lease_expires_at,
                       created_at, started_at, finished_at, error_message, last_error_message
                from semantic_completion_jobs
                where source_scenic_id = :source_scenic_id
                  and source_node_id = :source_node_id
                  and job_key = :job_key
                  and status in ('PENDING', 'RUNNING')
                order by created_at desc
                limit 1
                """
            ),
            {"source_scenic_id": str(payload.scenic_id), "source_node_id": str(payload.node.source_node_id), "job_key": job_key},
        ).mappings().first()
        if row:
            result = dict(row)
            result["reused"] = True
            return result

        trace_id = str((payload.metadata or {}).get("trace_id") or (payload.metadata or {}).get("request_id") or uuid.uuid4().hex[:12])
        inserted = db.execute(
            text(
                """
                insert into semantic_completion_jobs (
                    trace_id, scenic_id, node_id, source_scenic_id, source_node_id,
                    job_key, status, progress, current_stage, request_payload,
                    max_attempts, pipeline_version, created_by, created_at, updated_at
                ) values (
                    :trace_id, :scenic_id, :node_id, :source_scenic_id, :source_node_id,
                    :job_key, 'PENDING', 0, 'queued', cast(:request_payload as jsonb),
                    :max_attempts, :pipeline_version, :created_by, now(), now()
                )
                returning id, trace_id, status, progress, current_stage,
                          question_count, evidence_count, candidate_count, conflict_count,
                          source_scenic_id, source_node_id,
                          attempt_count, max_attempts, locked_by, heartbeat_at, lease_expires_at,
                          created_at, started_at, finished_at, error_message, last_error_message
                """
            ),
            {
                "trace_id": trace_id,
                "scenic_id": scenic_id,
                "node_id": node_id,
                "source_scenic_id": str(payload.scenic_id),
                "source_node_id": str(payload.node.source_node_id),
                "job_key": job_key,
                "request_payload": _json(request_payload),
                "max_attempts": max(1, max_attempts),
                "pipeline_version": PIPELINE_VERSION,
                "created_by": created_by,
            },
        ).mappings().first()
        result = dict(inserted or {})
        result["reused"] = False
        return result


def claim_next_semantic_completion_job(*, worker_id: str, lease_seconds: int = 900) -> dict[str, Any] | None:
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        row = db.execute(
            text(
                """
                with picked as (
                    select id
                    from semantic_completion_jobs
                    where status = 'PENDING'
                      and coalesce(cancel_requested, false) = false
                      and coalesce(attempt_count, 0) < coalesce(max_attempts, 3)
                      and (next_retry_at is null or next_retry_at <= now())
                    order by created_at asc
                    for update skip locked
                    limit 1
                )
                update semantic_completion_jobs j
                set status = 'RUNNING',
                    progress = greatest(coalesce(progress, 0), 5),
                    current_stage = 'claimed',
                    attempt_count = coalesce(attempt_count, 0) + 1,
                    locked_by = :worker_id,
                    locked_at = now(),
                    heartbeat_at = now(),
                    lease_expires_at = now() + (:lease_seconds || ' seconds')::interval,
                    worker_version = :worker_version,
                    started_at = coalesce(started_at, now()),
                    updated_at = now(),
                    error_message = null
                from picked
                where j.id = picked.id
                returning j.*
                """
            ),
            {"worker_id": worker_id, "lease_seconds": int(lease_seconds), "worker_version": PIPELINE_VERSION},
        ).mappings().first()
        return dict(row) if row else None


def recover_stale_semantic_completion_jobs(*, worker_id: str | None = None) -> int:
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        retry_count = int(db.execute(
            text(
                """
                update semantic_completion_jobs
                set status = 'PENDING',
                    current_stage = 'recovered',
                    locked_by = null,
                    locked_at = null,
                    heartbeat_at = null,
                    lease_expires_at = null,
                    next_retry_at = now(),
                    last_error_code = 'LEASE_EXPIRED',
                    last_error_message = 'Worker lease expired; job returned to pending queue',
                    updated_at = now()
                where status = 'RUNNING'
                  and lease_expires_at is not null
                  and lease_expires_at < now()
                  and coalesce(attempt_count, 0) < coalesce(max_attempts, 3)
                """
            )
        ).rowcount or 0)
        failed_count = int(db.execute(
            text(
                """
                update semantic_completion_jobs
                set status = 'FAILED',
                    progress = 100,
                    current_stage = 'failed',
                    last_error_code = 'LEASE_EXPIRED_MAX_ATTEMPTS',
                    last_error_message = 'Worker lease expired and max attempts reached',
                    error_message = 'Worker lease expired and max attempts reached',
                    finished_at = now(),
                    updated_at = now()
                where status = 'RUNNING'
                  and lease_expires_at is not null
                  and lease_expires_at < now()
                  and coalesce(attempt_count, 0) >= coalesce(max_attempts, 3)
                """
            )
        ).rowcount or 0)
        if retry_count or failed_count:
            logger.warning("semantic job recovery by %s: retry=%s failed=%s", worker_id, retry_count, failed_count)
        return retry_count + failed_count


def heartbeat_semantic_completion_job(job_id: int, *, worker_id: str | None = None, stage: str | None = None, progress: int | None = None, lease_seconds: int = 900) -> None:
    assignments = ["heartbeat_at = now()", "lease_expires_at = now() + (:lease_seconds || ' seconds')::interval", "updated_at = now()"]
    params: dict[str, Any] = {"id": int(job_id), "lease_seconds": int(lease_seconds)}
    if worker_id:
        assignments.append("locked_by = :worker_id")
        params["worker_id"] = worker_id
    if stage:
        assignments.append("current_stage = :stage")
        params["stage"] = stage
    if progress is not None:
        assignments.append("progress = greatest(coalesce(progress, 0), :progress)")
        params["progress"] = int(progress)
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        db.execute(text("update semantic_completion_jobs set " + ", ".join(assignments) + " where id = :id"), params)


def _candidate_to_fill_item(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row or {})
    if item.get("claim_type") == "property":
        item.setdefault("name", item.get("predicate"))
        item.setdefault("key", item.get("predicate"))
        item.setdefault("value", item.get("object_value") or item.get("object_name"))
    elif item.get("claim_type") == "relation":
        item.setdefault("relation_type", item.get("predicate"))
        item.setdefault("target_name", item.get("object_name") or item.get("object_value"))
        item.setdefault("target_type", item.get("object_type"))
    item.setdefault("candidate_id", item.get("id"))
    item.setdefault("quote", item.get("quote") or "")
    item.setdefault("source_url", item.get("source_url") or "")
    return item


def _hydrate_job_result(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row or {})
    if str(result.get("status") or "").upper() not in {"DONE", "COMPLETED"}:
        return result
    job_id = int(result["id"])
    trace_id = str(result.get("trace_id") or "")
    evidence = list_semantic_evidence_items(job_id=job_id, limit=300).get("items") or []
    candidates = list_semantic_candidates(job_id=job_id, limit=300).get("items") or []
    groups = list_semantic_candidate_groups(job_id=job_id, limit=300).get("items") or []
    questions = list_semantic_completion_questions(job_id=job_id, limit=300).get("items") or []
    gaps = list_semantic_gap_status(job_id=job_id, limit=300).get("items") or []
    prop_items = [_candidate_to_fill_item(item) for item in candidates if str(item.get("claim_type") or "") == "property"]
    rel_items = [_candidate_to_fill_item(item) for item in candidates if str(item.get("claim_type") or "") == "relation"]
    summary = f"Semantic completion done: evidence {len(evidence)}, candidates {len(candidates)}, groups {len(groups)}."
    result["result_data"] = {
        "mode": "evidence_claims_async",
        "trace_id": trace_id,
        "summary": summary,
        "answer": summary,
        "questions": questions,
        "sources": evidence,
        "evidence": evidence,
        "evidence_chunks": evidence,
        "candidate_claims": candidates,
        "candidate_groups": groups,
        "gap_status": gaps,
        "template_fill": {"properties": prop_items, "relations": rel_items},
        "discoveries": {"properties": [], "entities": [], "relations": [], "facts": [], "conflicts": []},
        "candidates": {"properties": prop_items, "relations": rel_items, "entities": []},
        "bridge": {"job_id": job_id, "trace_id": trace_id},
    }
    return result


def _job_select_sql(extra_columns: str = "") -> str:
    return f"""
        select id, trace_id, scenic_id, node_id, source_scenic_id, source_node_id,
               job_key, status, progress, current_stage, request_payload,
               question_count, evidence_count, candidate_count, conflict_count,
               attempt_count, max_attempts, locked_by, heartbeat_at, lease_expires_at,
               next_retry_at, cancel_requested,
               error_message, last_error_code, last_error_message,
               created_by, created_at, started_at, finished_at, updated_at
               {extra_columns}
        from semantic_completion_jobs
    """


def get_semantic_completion_job(job_id: int) -> dict[str, Any] | None:
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        row = db.execute(text(_job_select_sql() + " where id = :id"), {"id": int(job_id)}).mappings().first()
        return _hydrate_job_result(dict(row)) if row else None


def get_latest_semantic_completion_job(*, source_scenic_id: str, source_node_id: str) -> dict[str, Any] | None:
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        row = db.execute(
            text(_job_select_sql() + " where source_scenic_id = :source_scenic_id and source_node_id = :source_node_id order by created_at desc limit 1"),
            {"source_scenic_id": str(source_scenic_id), "source_node_id": str(source_node_id)},
        ).mappings().first()
        return _hydrate_job_result(dict(row)) if row else None




def list_semantic_completion_jobs(
    *,
    source_scenic_id: str | None = None,
    source_node_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))
    where: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if source_scenic_id:
        where.append("source_scenic_id = :source_scenic_id")
        params["source_scenic_id"] = str(source_scenic_id)
    if source_node_id:
        where.append("source_node_id = :source_node_id")
        params["source_node_id"] = str(source_node_id)
    if status:
        where.append("status = :status")
        params["status"] = str(status).upper()
    where_sql = " where " + " and ".join(where) if where else ""
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        total = db.execute(text("select count(*) as n from semantic_completion_jobs" + where_sql), params).mappings().first()["n"]
        rows = db.execute(
            text(_job_select_sql() + where_sql + " order by created_at desc, id desc limit :limit offset :offset"),
            params,
        ).mappings().all()
        return {"items": [_hydrate_job_result(dict(row)) for row in rows], "total": int(total or 0)}


def _mark_failure(job_id: int, exc: Exception) -> None:
    message = str(exc)[:4000]
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        row = db.execute(
            text("select attempt_count, max_attempts from semantic_completion_jobs where id=:id"),
            {"id": int(job_id)},
        ).mappings().first()
        attempts = int((row or {}).get("attempt_count") or 0)
        max_attempts = int((row or {}).get("max_attempts") or DEFAULT_MAX_ATTEMPTS)
        if attempts < max_attempts:
            db.execute(
                text(
                    """
                    update semantic_completion_jobs
                    set status = 'PENDING',
                        current_stage = 'retry_wait',
                        progress = least(coalesce(progress, 0), 95),
                        error_message = :error_message,
                        last_error_code = 'JOB_EXCEPTION',
                        last_error_message = :error_message,
                        locked_by = null,
                        locked_at = null,
                        heartbeat_at = null,
                        lease_expires_at = null,
                        next_retry_at = now() + (:delay_seconds || ' seconds')::interval,
                        updated_at = now()
                    where id = :id
                    """
                ),
                {"id": int(job_id), "error_message": message, "delay_seconds": min(300, 20 * max(1, attempts))},
            )
        else:
            db.execute(
                text(
                    """
                    update semantic_completion_jobs
                    set status = 'FAILED',
                        progress = 100,
                        current_stage = 'failed',
                        error_message = :error_message,
                        last_error_code = 'JOB_EXCEPTION',
                        last_error_message = :error_message,
                        locked_by = null,
                        heartbeat_at = null,
                        lease_expires_at = null,
                        finished_at = now(),
                        updated_at = now()
                    where id = :id
                    """
                ),
                {"id": int(job_id), "error_message": message},
            )


async def run_semantic_completion_job(job_id: int, *, worker_id: str | None = None, already_claimed: bool = False) -> None:
    started_at = time.time()
    if not already_claimed:
        worker = worker_id or f"direct-{uuid.uuid4().hex[:8]}"
        with ai_session_scope() as db:
            apply_semantic_completion_schema(db)
            db.execute(
                text(
                    """
                    update semantic_completion_jobs
                    set status = 'RUNNING',
                        progress = greatest(coalesce(progress, 0), 5),
                        current_stage = 'direct_run',
                        attempt_count = coalesce(attempt_count, 0) + 1,
                        locked_by = :worker_id,
                        locked_at = now(),
                        heartbeat_at = now(),
                        lease_expires_at = now() + interval '900 seconds',
                        worker_version = :worker_version,
                        started_at = coalesce(started_at, now()),
                        updated_at = now(),
                        error_message = null
                    where id = :id and status in ('PENDING', 'RUNNING')
                    """
                ),
                {"id": int(job_id), "worker_id": worker, "worker_version": PIPELINE_VERSION},
            )
            worker_id = worker

    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        row = db.execute(
            text("select id, trace_id, request_payload from semantic_completion_jobs where id = :id limit 1"),
            {"id": int(job_id)},
        ).mappings().first()
        if not row:
            logger.error("semantic completion job not found: %s", job_id)
            return
        trace_id = str(row["trace_id"])
        request_payload = dict(row["request_payload"] or {})

    try:
        heartbeat_semantic_completion_job(job_id, worker_id=worker_id, stage="running", progress=10)
        payload = SemanticCompleteRequest.parse_obj(request_payload)
        response = await complete_semantic_service(payload, trace_id_override=trace_id, job_id=int(job_id))
        evidence_count = len(response.evidence_chunks or [])
        candidate_count = int(list_semantic_candidates(job_id=int(job_id), limit=1).get("total") or 0)
        conflict_count = len(response.conflicts or [])
        question_count = int(list_semantic_completion_questions(job_id=int(job_id), limit=1).get("total") or 0)
        try:
            update_semantic_completion_question_stats(job_id=int(job_id))
        except Exception as exc:
            logger.warning("update semantic question stats failed: %s", exc, exc_info=True)
        with ai_session_scope() as db:
            apply_semantic_completion_schema(db)
            db.execute(
                text(
                    """
                    update semantic_completion_jobs
                    set status = 'DONE',
                        progress = 100,
                        current_stage = 'done',
                        question_count = :question_count,
                        evidence_count = :evidence_count,
                        candidate_count = :candidate_count,
                        conflict_count = :conflict_count,
                        locked_by = null,
                        heartbeat_at = null,
                        lease_expires_at = null,
                        finished_at = now(),
                        updated_at = now(),
                        error_message = null,
                        last_error_code = null,
                        last_error_message = null
                    where id = :id
                    """
                ),
                {
                    "id": int(job_id),
                    "question_count": int(question_count),
                    "evidence_count": int(evidence_count),
                    "candidate_count": int(candidate_count),
                    "conflict_count": int(conflict_count),
                },
            )
    except Exception as exc:
        elapsed = round((time.time() - started_at) * 1000, 2)
        logger.error("semantic completion job failed: %s elapsed_ms=%s", exc, elapsed, exc_info=True)
        _mark_failure(int(job_id), exc)

