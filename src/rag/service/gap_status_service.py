"""Gap status persistence for semantic completion questions."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.schemas import CandidateClaim, EvidenceChunk, SemanticCompleteRequest
from src.rag.service.evidence_store import apply_semantic_completion_schema
from src.rag.service.semantic_candidate_store import apply_semantic_candidate_schema

CONFLICT_CLASSES = {"conflicting", "scope_mismatch", "entity_ambiguity", "value_conflict", "temporal_conflict", "exclusive_relation_conflict"}


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _question_key(question: Any) -> str:
    return str(getattr(question, "question_id", "") or "unknown")


def _status_for(question_id: str, chunks: list[EvidenceChunk], claims: list[CandidateClaim], groups: list[dict[str, Any]]) -> tuple[str, str | None, dict[str, Any]]:
    q_chunks = [item for item in chunks if str(getattr(item, "question_id", "") or "") == question_id]
    q_claims = [item for item in claims if str(getattr(item, "question_id", "") or "") == question_id]
    q_groups = [item for item in groups if str(item.get("question_id") or "") == question_id]
    conflict_class = None
    for group in q_groups:
        cclass = str(group.get("conflict_class") or "")
        if cclass in CONFLICT_CLASSES:
            conflict_class = cclass
            return "conflicted", conflict_class, {"reason": "candidate_group_conflict"}
    if q_claims:
        statuses = {str(getattr(item, "evidence_status", "") or "") for item in q_claims}
        if statuses <= {"unsupported"}:
            return "weak_evidence", "unsupported", {"reason": "only_unsupported_candidates"}
        if "supported" in statuses or "weak" in statuses:
            return "pending_review", None, {"reason": "has_reviewable_candidates"}
        return "pending_review", None, {"reason": "has_candidates"}
    if not q_chunks:
        return "no_evidence", None, {"reason": "no_evidence"}
    best_score = max((float(getattr(item, "final_evidence_score", 0.0) or 0.0) for item in q_chunks), default=0.0)
    if best_score >= 0.45:
        return "needs_more_evidence", None, {"reason": "evidence_without_candidate", "best_evidence_score": round(best_score, 3)}
    return "weak_evidence", "weak_evidence", {"reason": "weak_evidence_without_candidate", "best_evidence_score": round(best_score, 3)}


def update_semantic_gap_status(
    *,
    payload: SemanticCompleteRequest,
    trace_id: str,
    job_id: int | None,
    questions: list[Any],
    chunks: list[EvidenceChunk],
    claims: list[CandidateClaim],
    candidate_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    if not questions:
        return {"updated": 0}
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        apply_semantic_candidate_schema(db)
        updated = 0
        for question in questions:
            qid = _question_key(question)
            status, conflict_class, reason_meta = _status_for(qid, chunks, claims, candidate_groups)
            evidence_count = sum(1 for item in chunks if str(getattr(item, "question_id", "") or "") == qid)
            candidate_count = sum(1 for item in claims if str(getattr(item, "question_id", "") or "") == qid)
            db.execute(
                text(
                    """
                    insert into semantic_gap_status (
                        source_scenic_id, source_node_id, target_kind, target_field,
                        relation_intent, temporal_role, status, job_id, trace_id,
                        question_id, evidence_count, candidate_count, conflict_class,
                        metadata, updated_at
                    ) values (
                        :source_scenic_id, :source_node_id, :target_kind, :target_field,
                        :relation_intent, :temporal_role, :status, :job_id, :trace_id,
                        :question_id, :evidence_count, :candidate_count, :conflict_class,
                        cast(:metadata as jsonb), now()
                    )
                    on conflict (source_scenic_id, source_node_id, target_kind, target_field, relation_intent, temporal_role)
                    do update set
                        status = excluded.status,
                        job_id = excluded.job_id,
                        trace_id = excluded.trace_id,
                        question_id = excluded.question_id,
                        evidence_count = excluded.evidence_count,
                        candidate_count = excluded.candidate_count,
                        conflict_class = excluded.conflict_class,
                        metadata = excluded.metadata,
                        updated_at = now()
                    """
                ),
                {
                    "source_scenic_id": str(payload.scenic_id),
                    "source_node_id": str(payload.node.source_node_id),
                    "target_kind": str(getattr(question, "target_kind", "") or "fact"),
                    "target_field": getattr(question, "target_field", None),
                    "relation_intent": getattr(question, "relation_intent", None),
                    "temporal_role": getattr(question, "temporal_role", None),
                    "status": status,
                    "job_id": int(job_id) if job_id is not None else None,
                    "trace_id": trace_id,
                    "question_id": qid,
                    "evidence_count": int(evidence_count),
                    "candidate_count": int(candidate_count),
                    "conflict_class": conflict_class,
                    "metadata": _json({
                        "query_text": getattr(question, "query_text", None),
                        "priority": getattr(question, "priority", None),
                        **reason_meta,
                    }),
                },
            )
            updated += 1
    return {"updated": updated}


def list_semantic_gap_status(
    *,
    source_scenic_id: str | None = None,
    source_node_id: str | None = None,
    job_id: int | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    limit = max(1, min(int(limit or 100), 500))
    offset = max(0, int(offset or 0))
    where: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if source_scenic_id:
        where.append("source_scenic_id = :source_scenic_id")
        params["source_scenic_id"] = str(source_scenic_id)
    if source_node_id:
        where.append("source_node_id = :source_node_id")
        params["source_node_id"] = str(source_node_id)
    if job_id is not None:
        where.append("job_id = :job_id")
        params["job_id"] = int(job_id)
    if status:
        where.append("status = :status")
        params["status"] = str(status)
    where_sql = " where " + " and ".join(where) if where else ""
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        apply_semantic_candidate_schema(db)
        total = db.execute(text("select count(*) as n from semantic_gap_status" + where_sql), params).mappings().first()["n"]
        rows = db.execute(
            text(
                """
                select id, source_scenic_id, source_node_id, target_kind, target_field,
                       relation_intent, temporal_role, status, job_id, trace_id, question_id,
                       evidence_count, candidate_count, conflict_class, metadata, created_at, updated_at
                from semantic_gap_status
                """ + where_sql + " order by updated_at desc, id desc limit :limit offset :offset"
            ),
            params,
        ).mappings().all()
    return {"items": [dict(row) for row in rows], "total": int(total or 0)}
