from __future__ import annotations

import hashlib
import json
import threading
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.service.semantic_candidate_store import apply_semantic_candidate_schema

MIGRATION_DIR = Path(__file__).parent / "migrations"
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _statements(raw: str) -> list[str]:
    statements, current = [], []
    for line in raw.splitlines():
        if line.strip().startswith("--"):
            continue
        current.append(line)
        if line.strip().endswith(";"):
            statements.append("\n".join(current).rstrip(";"))
            current = []
    if current and "".join(current).strip():
        statements.append("\n".join(current))
    return statements


def ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        with ai_session_scope() as db:
            apply_semantic_candidate_schema(db)
            db.execute(text("select pg_advisory_xact_lock(hashtext(:name))"), {"name": "semantic_growth_stage1_schema"})
            for migration_file in sorted(MIGRATION_DIR.glob("*.sql")):
                for statement in _statements(migration_file.read_text(encoding="utf-8")):
                    db.execute(text(statement))
        _SCHEMA_READY = True


def create_run(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_schema()
    growth_run_id = str(payload.get("growth_run_id") or f"growth-{uuid.uuid4().hex[:20]}")
    thread_id = str(payload.get("thread_id") or growth_run_id)
    seed_node_ids = [str(value) for value in payload.get("seed_node_ids") or []]
    params = {
        "growth_run_id": growth_run_id,
        "thread_id": thread_id,
        "domain_id": str(payload["domain_id"]),
        "scenic_id": str(payload.get("scenic_id") or "") or None,
        "seed_node_ids": _json(seed_node_ids),
        "max_iterations": int(payload.get("max_iterations") or 1),
        "budget": int(payload.get("budget") or 1),
        "created_by": str(payload.get("created_by") or "") or None,
        "metadata": _json({
            "mock_candidate": payload.get("mock_candidate") or {},
            "growth_track": "OPEN_DISCOVERY",
            "domain_schema": payload.get("domain_schema") or {},
            "discovery_budget": int(payload.get("max_evidence_per_run") or 500),
            "image_discovery_budget": int(payload.get("max_image_evidence_per_run") or 32),
            "extraction_concurrency": int(payload.get("extraction_concurrency") or 4),
            "review_budget": int(payload.get("budget") or 10),
            "consumer_version": "growth-open-v2",
        }),
    }
    with ai_session_scope() as db:
        row = db.execute(text("""
            insert into semantic_growth_runs (
                growth_run_id, thread_id, domain_id, scenic_id, seed_node_ids,
                max_iterations, budget, created_by, metadata
            ) values (
                :growth_run_id, :thread_id, :domain_id, :scenic_id, cast(:seed_node_ids as jsonb),
                :max_iterations, :budget, :created_by, cast(:metadata as jsonb)
            ) on conflict (growth_run_id) do nothing returning *
        """), params).mappings().first()
        if row is None:
            row = db.execute(text("select * from semantic_growth_runs where growth_run_id = :growth_run_id"), params).mappings().one()
    return dict(row)


def update_publication_sync_status(
    publication_batch_id: str,
    *,
    status: str,
    error: str | None = None,
    affected_scope: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    ensure_schema()
    normalized = str(status or "").strip().upper()
    if normalized not in {"GRAPH_SYNC_PENDING", "PUBLISHED", "GRAPH_SYNC_FAILED"}:
        raise ValueError("invalid publication sync status")
    with ai_session_scope() as db:
        row = db.execute(
            text(
                """
                update semantic_growth_publication_records
                set status=:status,
                    error=:error,
                    affected_scope=case when :affected_scope is not null
                        then cast(:affected_scope as jsonb) else affected_scope end,
                    updated_at=now(),
                    published_at=case when :status='PUBLISHED' then now() else published_at end
                where publication_batch_id=:publication_batch_id
                returning *
                """
            ),
            {
                "publication_batch_id": str(publication_batch_id),
                "status": normalized,
                "error": str(error or "")[:4000] or None,
                "affected_scope": _json(affected_scope) if isinstance(affected_scope, list) else None,
            },
        ).mappings().first()
    return dict(row) if row else None


def record_publication_result(growth_run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    ensure_schema()
    payload = payload if isinstance(payload, dict) else {}
    status = str(payload.get("status") or "").strip().upper()
    warning = str(payload.get("warning") or "").strip() or None
    error = str(payload.get("error") or "").strip() or None
    if not status:
        status = "SYNC_PENDING" if warning else ("PUBLISHED" if int(payload.get("published") or 0) else "NOT_REQUIRED")
    candidate_ids = [int(value) for value in (payload.get("candidate_ids") or []) if str(value).isdigit()]
    affected_scope = payload.get("affected_scope") if isinstance(payload.get("affected_scope"), list) else []
    params = {
        "growth_run_id": str(growth_run_id),
        "publication_batch_id": str(payload.get("batch_id") or "") or None,
        "status": status,
        "candidate_ids": _json(candidate_ids),
        "affected_scope": _json(affected_scope),
        "published_candidate_count": int(payload.get("published") or 0),
        "warning": warning,
        "error": error,
    }
    with ai_session_scope() as db:
        row = db.execute(
            text(
                """
                insert into semantic_growth_publication_records (
                    growth_run_id, publication_batch_id, status, candidate_ids,
                    affected_scope, published_candidate_count, warning, error,
                    attempt_count, published_at, updated_at
                ) values (
                    :growth_run_id, :publication_batch_id, :status,
                    cast(:candidate_ids as jsonb), cast(:affected_scope as jsonb),
                    :published_candidate_count, :warning, :error, 1,
                    case when :status='PUBLISHED' then now() else null end, now()
                )
                on conflict (growth_run_id) do update set
                    publication_batch_id=coalesce(excluded.publication_batch_id, semantic_growth_publication_records.publication_batch_id),
                    status=excluded.status,
                    candidate_ids=excluded.candidate_ids,
                    affected_scope=excluded.affected_scope,
                    published_candidate_count=excluded.published_candidate_count,
                    warning=excluded.warning,
                    error=excluded.error,
                    attempt_count=semantic_growth_publication_records.attempt_count+1,
                    published_at=case when excluded.status='PUBLISHED' then now() else semantic_growth_publication_records.published_at end,
                    updated_at=now()
                returning *
                """
            ),
            params,
        ).mappings().one()
    return dict(row)


def get_run(growth_run_id: str) -> dict[str, Any] | None:
    ensure_schema()
    with ai_session_scope() as db:
        row = db.execute(text("select * from semantic_growth_runs where growth_run_id = :id"), {"id": growth_run_id}).mappings().first()
    return dict(row) if row else None


def set_run_status(
    growth_run_id: str,
    status: str,
    stop_reason: str | None = None,
    *,
    status_reason_code: str | None = None,
    failed_opportunity_count: int | None = None,
    warning_codes: list[str] | None = None,
) -> None:
    with ai_session_scope() as db:
        db.execute(text("""
            update semantic_growth_runs set status = :status, stop_reason = :stop_reason,
                status_reason_code = coalesce(:status_reason_code, status_reason_code),
                failed_opportunity_count = coalesce(:failed_opportunity_count, failed_opportunity_count),
                warning_codes = coalesce(cast(:warning_codes as jsonb), warning_codes),
                updated_at = now(),
                finished_at = case when :status in ('COMPLETED', 'REJECTED', 'FAILED', 'NO_CHANGE') then now() else finished_at end
            where growth_run_id = :id
        """), {
            "id": growth_run_id,
            "status": status,
            "stop_reason": stop_reason,
            "status_reason_code": status_reason_code,
            "failed_opportunity_count": failed_opportunity_count,
            "warning_codes": _json(warning_codes) if warning_codes is not None else None,
        })


def record_step(growth_run_id: str, step_name: str, output: dict[str, Any], opportunity_id: str | None = None) -> None:
    with ai_session_scope() as db:
        db.execute(text("""
            insert into semantic_growth_step_records (
                growth_run_id, opportunity_id, step_name, status, output_ref, finished_at
            ) values (:id, :opportunity_id, :step_name, 'SUCCESS', cast(:output as jsonb), now())
        """), {"id": growth_run_id, "opportunity_id": opportunity_id, "step_name": step_name, "output": _json(output)})


def ensure_mock_opportunity(state: dict[str, Any]) -> dict[str, Any]:
    growth_run_id = state["growth_run_id"]
    node_id = (state.get("seed_node_ids") or [None])[0]
    dedupe_key = f"{growth_run_id}:stage1_mock"
    opportunity_id = f"opp-{hashlib.sha256(dedupe_key.encode()).hexdigest()[:20]}"
    with ai_session_scope() as db:
        row = db.execute(text("""
            insert into semantic_growth_opportunities (
                opportunity_id, growth_run_id, node_id, opportunity_type,
                target_property, reason, status, dedupe_key, metadata
            ) values (
                :opportunity_id, :growth_run_id, :node_id, 'stage1_mock',
                '阶段一验证属性', '验证持久化人工中断恢复闭环', 'PROCESSING', :dedupe_key, '{}'::jsonb
            ) on conflict (dedupe_key) do update set updated_at = now() returning *
        """), {"opportunity_id": opportunity_id, "growth_run_id": growth_run_id, "node_id": node_id, "dedupe_key": dedupe_key}).mappings().one()
    return dict(row)


def ensure_mock_candidate(state: dict[str, Any], opportunity: dict[str, Any]) -> int:
    growth_run_id = state["growth_run_id"]
    run = get_run(growth_run_id) or {}
    mock = (run.get("metadata") or {}).get("mock_candidate") or {}
    source_node_id = str(mock.get("source_node_id") or opportunity.get("node_id") or "growth-stage1-node")
    predicate = str(mock.get("predicate") or "阶段一验证属性")
    value = str(mock.get("value") or "LangGraph人工中断恢复成功")
    candidate_uid = hashlib.sha256(f"{growth_run_id}|{opportunity['opportunity_id']}|{predicate}|{value}".encode()).hexdigest()
    params = {
        "candidate_uid": candidate_uid, "growth_run_id": growth_run_id, "domain_id": state["domain_id"],
        "source_node_id": source_node_id, "claim_id": f"claim-{opportunity['opportunity_id']}",
        "predicate": predicate, "value": value, "opportunity_id": opportunity["opportunity_id"],
        "quote": str(mock.get("quote") or "阶段一模拟证据，仅用于验证人工审核恢复闭环。"),
        "raw_payload": _json(mock), "metadata": _json({
            "growth_run_id": growth_run_id, "growth_iteration": state.get("iteration", 0),
            "growth_opportunity_id": opportunity["opportunity_id"], "trigger_type": "stage1_mock",
            "source_modalities": ["mock"], "langgraph_thread_id": state["thread_id"],
        }),
    }
    with ai_session_scope() as db:
        row = db.execute(text("""
            insert into semantic_claim_candidates (
                candidate_uid, trace_id, run_id, source_scenic_id, source_node_id,
                claim_id, claim_type, candidate_type, predicate, object_value,
                retrieval_source, source_id, source_title, quote, confidence,
                evidence_score, evidence_status, status, raw_payload, metadata,
                risk_level, publication_policy
            ) values (
                :candidate_uid, :growth_run_id, :growth_run_id, :domain_id, :source_node_id,
                :claim_id, 'property', 'growth_stage1_mock', :predicate, :value,
                'growth_mock', :opportunity_id, 'LangGraph stage 1 mock evidence', :quote, 1.0,
                1.0, 'SUPPORTED', 'PENDING', cast(:raw_payload as jsonb), cast(:metadata as jsonb),
                'LOW', 'REVIEW_REQUIRED'
            ) on conflict (candidate_uid) do update set updated_at = now() returning id
        """), params).mappings().one()
        candidate_id = int(row["id"])
        db.execute(text("""
            insert into semantic_growth_candidate_links (growth_run_id, opportunity_id, candidate_id, iteration)
            values (:id, :opportunity_id, :candidate_id, :iteration) on conflict do nothing
        """), {"id": growth_run_id, "opportunity_id": opportunity["opportunity_id"], "candidate_id": candidate_id, "iteration": int(state.get("iteration") or 0)})
    return candidate_id


def finish_review(growth_run_id: str, action: str, candidate_ids: list[str], payload: dict[str, Any]) -> None:
    status = {"accept": "ADOPTED", "modify": "ADOPTED", "reject": "REJECTED"}[action]
    with ai_session_scope() as db:
        db.execute(text("""
            update semantic_claim_candidates set status = :status, reviewed_by = :reviewed_by,
                review_note = :review_note, reviewed_at = now(), updated_at = now()
            where id = any(:candidate_ids)
        """), {"status": status, "reviewed_by": str(payload.get("reviewer_id") or payload.get("reviewed_by") or "growth-stage1"), "review_note": str(payload.get("review_note") or "LangGraph stage 1 resume"), "candidate_ids": [int(value) for value in candidate_ids]})
        db.execute(text("""
            update semantic_growth_opportunities set status = :status, updated_at = now()
            where growth_run_id = :id
        """), {"status": "COMPLETED" if action != "reject" else "REJECTED", "id": growth_run_id})
    set_run_status(growth_run_id, "COMPLETED" if action != "reject" else "REJECTED", f"review_{action}")


def get_run_detail(growth_run_id: str) -> dict[str, Any] | None:
    run = get_run(growth_run_id)
    if not run:
        return None
    with ai_session_scope() as db:
        opportunities = [dict(row) for row in db.execute(text("select * from semantic_growth_opportunities where growth_run_id = :id order by id"), {"id": growth_run_id}).mappings().all()]
        steps = [dict(row) for row in db.execute(text("select * from semantic_growth_step_records where growth_run_id = :id order by id"), {"id": growth_run_id}).mappings().all()]
        candidates = [dict(row) for row in db.execute(text("""
            select c.* from semantic_claim_candidates c
            join semantic_growth_candidate_links l on l.candidate_id = c.id
            where l.growth_run_id = :id order by c.id
        """), {"id": growth_run_id}).mappings().all()]
        try:
            evidence_bindings = [dict(row) for row in db.execute(text("""
                select b.*, e.source_type, e.source_title, e.source_url,
                       e.content as evidence_content, e.metadata as evidence_metadata
                from semantic_growth_candidate_evidence_bindings b
                join semantic_growth_evidence_units e on e.id=b.evidence_unit_id
                where b.growth_run_id=:id order by b.candidate_id, b.id
            """), {"id": growth_run_id}).mappings().all()]
            by_candidate = {}
            for binding in evidence_bindings:
                by_candidate.setdefault(int(binding["candidate_id"]), []).append(binding)
            for candidate in candidates:
                candidate["growth_evidence"] = by_candidate.get(int(candidate["id"]), [])
            lineage_rows = [dict(row) for row in db.execute(text("""
                select b.candidate_id, b.raw_claim_id, b.evidence_unit_id,
                       b.source_independence_key, b.support_role, b.evidence_score,
                       b.metadata as binding_metadata,
                       rc.subject_text, rc.subject_type, rc.claim_type,
                       rc.raw_predicate, rc.object_text, rc.object_type,
                       rc.temporal_role, rc.quote, rc.confidence,
                       rc.extraction_pass, rc.status as raw_claim_status,
                       rc.metadata as raw_claim_metadata,
                       e.source_type, e.source_title, e.source_url,
                       e.content as evidence_content, e.metadata as evidence_metadata
                from semantic_growth_candidate_evidence_bindings b
                left join semantic_growth_raw_claims rc on rc.id=b.raw_claim_id
                join semantic_growth_evidence_units e on e.id=b.evidence_unit_id
                where b.growth_run_id=:id
                order by b.candidate_id, b.id
            """), {"id": growth_run_id}).mappings().all()]
            lineage_by_candidate = {}
            for row in lineage_rows:
                lineage_by_candidate.setdefault(int(row["candidate_id"]), []).append(row)
            for candidate in candidates:
                candidate["growth_lineage"] = lineage_by_candidate.get(int(candidate["id"]), [])
            fact_evidence_bindings = [dict(row) for row in db.execute(text("""
                select b.*, e.source_type, e.source_title, e.source_url,
                       e.content as evidence_content
                from semantic_growth_fact_evidence_bindings b
                join semantic_growth_evidence_units e on e.id=b.evidence_unit_id
                where b.growth_run_id=:id order by b.id
            """), {"id": growth_run_id}).mappings().all()]
        except Exception:
            evidence_bindings = []
            fact_evidence_bindings = []
        try:
            dependencies = [dict(row) for row in db.execute(text("""
                select d.*,
                       uc.status as upstream_candidate_status,
                       un.status as upstream_node_candidate_status
                from semantic_growth_candidate_dependencies d
                left join semantic_claim_candidates uc on uc.id=d.upstream_candidate_id
                left join semantic_node_candidates un on un.id=d.upstream_node_candidate_id
                where d.growth_run_id=:id
                order by d.id
            """), {"id": growth_run_id}).mappings().all()]
        except Exception:
            dependencies = []
        try:
            node_candidates = [dict(row) for row in db.execute(text("""
                select n.*
                from semantic_node_candidates n
                where n.id = any(
                    select d.upstream_node_candidate_id
                    from semantic_growth_candidate_dependencies d
                    where d.growth_run_id=:id
                      and d.upstream_node_candidate_id is not null
                )
                order by n.id
            """), {"id": growth_run_id}).mappings().all()]
        except Exception:
            node_candidates = []
        try:
            publication = db.execute(
                text("select * from semantic_growth_publication_records where growth_run_id=:id"),
                {"id": growth_run_id},
            ).mappings().first()
            publication = dict(publication) if publication else None
        except Exception:
            publication = None
    metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
    return {
        "run": run,
        "opportunities": opportunities,
        "steps": steps,
        "candidates": candidates,
        "evidence_bindings": evidence_bindings,
        "fact_evidence_bindings": fact_evidence_bindings,
        "dependencies": dependencies,
        "node_candidates": node_candidates,
        "publication": publication,
        "affected_scope": metadata.get("g5_affected_scope") or [],
    }



def list_runs(domain_id=None, status=None, limit=50):
    """List growth runs, optionally filtered by domain_id and status."""
    ensure_schema()
    conditions = []
    params = {"limit": int(limit)}
    if domain_id:
        conditions.append("domain_id = :domain_id")
        params["domain_id"] = str(domain_id)
    if status:
        conditions.append("status = :status")
        params["status"] = str(status)
    where = ("where " + " and ".join(conditions)) if conditions else ""
    query = "select * from semantic_growth_runs " + where + " order by created_at desc limit :limit"
    with ai_session_scope() as db:
        rows = db.execute(text(query), params).mappings().all()
    return [dict(row) for row in rows]


def set_run_paused(growth_run_id):
    """Pause a growth run. Only allowed for RUNNING or WAITING_REVIEW runs."""
    with ai_session_scope() as db:
        row = db.execute(
            text("""
                update semantic_growth_runs
                set status = 'PAUSED', stop_reason = 'manual_pause', updated_at = now()
                where growth_run_id = :id and status in ('RUNNING', 'WAITING_REVIEW')
                returning *
            """),
            {"id": str(growth_run_id)},
        ).mappings().first()
    return dict(row) if row else None


def set_run_cancelled(growth_run_id):
    """Cancel a growth run. Only allowed for non-terminal runs."""
    with ai_session_scope() as db:
        row = db.execute(
            text("""
                update semantic_growth_runs
                set status = 'CANCELLED', stop_reason = 'manual_cancel',
                    updated_at = now(), finished_at = now()
                where growth_run_id = :id
                  and status not in ('COMPLETED', 'REJECTED', 'CANCELLED')
                returning *
            """),
            {"id": str(growth_run_id)},
        ).mappings().first()
    return dict(row) if row else None


def create_opportunity(
    growth_run_id,
    node_id,
    opportunity_type="template_gap",
    target_property=None,
    target_relation=None,
    reason="",
    metadata=None,
    status="PROCESSING",
):
    """Create a real (non-mock) growth opportunity for Phase 2+."""
    import hashlib

    dedupe_key_parts = [
        str(growth_run_id), str(node_id), str(opportunity_type),
        str(target_property or ""), str(target_relation or ""),
    ]
    dedupe_key_raw = "|".join(dedupe_key_parts)
    dedupe_key = hashlib.sha256(dedupe_key_raw.encode()).hexdigest()[:40]
    opportunity_id = "opp-" + hashlib.sha256(dedupe_key.encode()).hexdigest()[:16]

    params = {
        "opportunity_id": opportunity_id,
        "growth_run_id": str(growth_run_id),
        "node_id": str(node_id),
        "opportunity_type": str(opportunity_type),
        "target_property": target_property,
        "target_relation": target_relation,
        "reason": str(reason),
        "status": str(status),
        "dedupe_key": dedupe_key,
        "metadata": _json(metadata or {}),
    }
    with ai_session_scope() as db:
        row = db.execute(
            text("""
                insert into semantic_growth_opportunities (
                    opportunity_id, growth_run_id, node_id, opportunity_type,
                    target_property, target_relation, reason, status, dedupe_key, metadata
                ) values (
                    :opportunity_id, :growth_run_id, :node_id, :opportunity_type,
                    :target_property, :target_relation, :reason, :status, :dedupe_key,
                    cast(:metadata as jsonb)
                )
                on conflict (dedupe_key) do update
                    set updated_at = now(),
                        status = excluded.status,
                        reason = excluded.reason,
                        target_property = excluded.target_property,
                        target_relation = excluded.target_relation
                returning *
            """),
            params,
        ).mappings().one()
    return dict(row)


def finish_opportunity(opportunity_id: str, *, status: str, error: str | None = None) -> None:
    with ai_session_scope() as db:
        db.execute(
            text(
                """
                update semantic_growth_opportunities
                set status=:status, updated_at=now()
                where opportunity_id=:id
                """
            ),
            {"id": str(opportunity_id), "status": str(status)},
        )


def create_candidate_link(growth_run_id, opportunity_id, candidate_id, iteration=1):
    """Link a semantic candidate to a growth run and opportunity."""
    with ai_session_scope() as db:
        db.execute(
            text("""
                insert into semantic_growth_candidate_links (
                    growth_run_id, opportunity_id, candidate_id, iteration
                ) values (:growth_run_id, :opportunity_id, :candidate_id, :iteration)
                on conflict do nothing
            """),
            {
                "growth_run_id": str(growth_run_id),
                "opportunity_id": str(opportunity_id),
                "candidate_id": int(candidate_id),
                "iteration": int(iteration),
            },
        )
