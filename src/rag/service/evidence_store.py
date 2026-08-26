"""Persistence helpers for semantic completion evidence items."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.schemas import EvidenceChunk, SemanticCompleteRequest

RAG_DIR = Path(__file__).resolve().parents[1]
MIGRATION_FILES = [
    RAG_DIR / "migrations" / "20260707_semantic_claim_candidates.sql",
    RAG_DIR / "migrations" / "20260711_semantic_completion_jobs.sql",
]
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_ADVISORY_LOCK_NAME = "semantic_completion_schema"


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def apply_semantic_completion_schema(db) -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        try:
            db.execute(
                text("select pg_advisory_xact_lock(hashtext(:lock_name))"),
                {"lock_name": _SCHEMA_ADVISORY_LOCK_NAME},
            )
            for migration_file in MIGRATION_FILES:
                raw = migration_file.read_text(encoding="utf-8")
                statements: list[str] = []
                current: list[str] = []
                for line in raw.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("--"):
                        continue
                    current.append(line)
                    if stripped.endswith(";"):
                        statements.append("\n".join(current).rstrip(";"))
                        current = []
                if current:
                    statements.append("\n".join(current))
                for stmt in statements:
                    sql = stmt.strip()
                    if sql:
                        db.execute(text(sql))
            db.commit()
        except Exception:
            db.rollback()
            raise
        _SCHEMA_READY = True


def persist_semantic_completion_questions(
    *,
    trace_id: str,
    job_id: int,
    payload: SemanticCompleteRequest,
    questions: list[Any],
) -> dict[str, Any]:
    if not questions:
        return {"saved": 0}
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        saved = 0
        for question in questions:
            db.execute(
                text(
                    """
                    insert into semantic_completion_questions (
                        job_id, trace_id, source_scenic_id, source_node_id,
                        question_id, target_kind, target_field, relation_intent, temporal_role,
                        query_text, search_terms, priority, status, metadata, updated_at
                    ) values (
                        :job_id, :trace_id, :source_scenic_id, :source_node_id,
                        :question_id, :target_kind, :target_field, :relation_intent, :temporal_role,
                        :query_text, cast(:search_terms as jsonb), :priority, 'planned', cast(:metadata as jsonb), now()
                    )
                    on conflict (job_id, question_id) do update set
                        target_kind = excluded.target_kind,
                        target_field = excluded.target_field,
                        relation_intent = excluded.relation_intent,
                        temporal_role = excluded.temporal_role,
                        query_text = excluded.query_text,
                        search_terms = excluded.search_terms,
                        priority = excluded.priority,
                        metadata = excluded.metadata,
                        updated_at = now()
                    """
                ),
                {
                    "job_id": int(job_id),
                    "trace_id": trace_id,
                    "source_scenic_id": str(payload.scenic_id),
                    "source_node_id": str(payload.node.source_node_id),
                    "question_id": str(getattr(question, "question_id", "") or "unknown"),
                    "target_kind": str(getattr(question, "target_kind", "") or "fact"),
                    "target_field": getattr(question, "target_field", None),
                    "relation_intent": getattr(question, "relation_intent", None),
                    "temporal_role": getattr(question, "temporal_role", None),
                    "query_text": str(getattr(question, "query_text", "") or ""),
                    "search_terms": _json(getattr(question, "search_terms", []) or []),
                    "priority": int(getattr(question, "priority", 50) or 50),
                    "metadata": _json(getattr(question, "metadata", {}) or {}),
                },
            )
            saved += 1
        return {"saved": saved}


def update_semantic_completion_question_stats(*, job_id: int) -> dict[str, Any]:
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        db.execute(
            text(
                """
                update semantic_completion_questions q
                set evidence_count = coalesce(e.n, 0),
                    candidate_count = coalesce(c.n, 0),
                    status = case
                        when coalesce(c.n, 0) > 0 then 'has_candidates'
                        when coalesce(e.n, 0) > 0 then 'has_evidence'
                        else 'no_evidence'
                    end,
                    updated_at = now()
                from (
                    select q2.id,
                           (select count(*) from semantic_evidence_items e where e.job_id=q2.job_id and e.question_id=q2.question_id) as n
                    from semantic_completion_questions q2
                    where q2.job_id=:job_id
                ) e,
                (
                    select q3.id,
                           (select count(*) from semantic_claim_candidates c where c.job_id=q3.job_id and c.question_id=q3.question_id) as n
                    from semantic_completion_questions q3
                    where q3.job_id=:job_id
                ) c
                where q.job_id=:job_id and q.id=e.id and q.id=c.id
                """
            ),
            {"job_id": int(job_id)},
        )
        total = db.execute(text("select count(*) from semantic_completion_questions where job_id=:job_id"), {"job_id": int(job_id)}).scalar() or 0
        return {"updated": int(total)}


def list_semantic_completion_questions(
    *,
    job_id: int | None = None,
    trace_id: str | None = None,
    source_scenic_id: str | None = None,
    source_node_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    limit = max(1, min(int(limit or 100), 500))
    offset = max(0, int(offset or 0))
    where: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if job_id is not None:
        where.append("job_id = :job_id")
        params["job_id"] = int(job_id)
    if trace_id:
        where.append("trace_id = :trace_id")
        params["trace_id"] = str(trace_id)
    if source_scenic_id:
        where.append("source_scenic_id = :source_scenic_id")
        params["source_scenic_id"] = str(source_scenic_id)
    if source_node_id:
        where.append("source_node_id = :source_node_id")
        params["source_node_id"] = str(source_node_id)
    where_sql = " where " + " and ".join(where) if where else ""
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        total = db.execute(text("select count(*) as n from semantic_completion_questions" + where_sql), params).mappings().first()["n"]
        rows = db.execute(
            text(
                """
                select id, job_id, trace_id, source_scenic_id, source_node_id,
                       question_id, target_kind, target_field, relation_intent, temporal_role,
                       query_text, search_terms, priority, status, evidence_count, candidate_count,
                       metadata, created_at, updated_at
                from semantic_completion_questions
                """ + where_sql + " order by priority desc, id asc limit :limit offset :offset"
            ),
            params,
        ).mappings().all()
    return {"items": [dict(row) for row in rows], "total": int(total or 0)}


def persist_semantic_evidence_items(
    *,
    trace_id: str,
    job_id: int | None,
    payload: SemanticCompleteRequest,
    chunks: list[EvidenceChunk],
) -> dict[str, Any]:
    if not chunks:
        return {"saved": 0, "source_to_evidence_id": {}}

    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        source_row = db.execute(
            text(
                """
                select sa.id as scenic_id, sn.id as node_id
                from scenic_areas sa
                join semantic_nodes sn on sn.scenic_id = sa.id and sn.source_node_id = :source_node_id
                where sa.source_scenic_id = :source_scenic_id
                order by sn.id
                limit 1
                """
            ),
            {
                "source_scenic_id": str(payload.scenic_id),
                "source_node_id": str(payload.node.source_node_id),
            },
        ).mappings().first()
        if not source_row:
            raise ValueError("node not found for semantic evidence persistence")

        scenic_id = int(source_row["scenic_id"])
        node_id = int(source_row["node_id"])
        source_to_evidence_id: dict[str, int] = {}
        saved = 0
        for chunk in chunks:
            row = db.execute(
                text(
                    """
                    insert into semantic_evidence_items (
                        trace_id, job_id,
                        scenic_id, node_id, source_scenic_id, source_node_id,
                        question_id,
                        target_kind, target_field, relation_intent, temporal_role,
                        query_text,
                        source_type, source_title, source_url, source_doc_id, chunk_id, page_no,
                        quote, content,
                        retrieval_score, rerank_score, source_weight,
                        provenance_type, retrieval_method, authority_class, source_authority_score,
                        final_evidence_score,
                        metadata
                    ) values (
                        :trace_id, :job_id,
                        :scenic_id, :node_id, :source_scenic_id, :source_node_id,
                        :question_id,
                        :target_kind, :target_field, :relation_intent, :temporal_role,
                        :query_text,
                        :source_type, :source_title, :source_url, :source_doc_id, :chunk_id, :page_no,
                        :quote, :content,
                        :retrieval_score, :rerank_score, :source_weight,
                        :provenance_type, :retrieval_method, :authority_class, :source_authority_score,
                        :final_evidence_score,
                        cast(:metadata as jsonb)
                    )
                    returning id
                    """
                ),
                {
                    "trace_id": trace_id,
                    "job_id": int(job_id) if job_id is not None else None,
                    "scenic_id": scenic_id,
                    "node_id": node_id,
                    "source_scenic_id": str(payload.scenic_id),
                    "source_node_id": str(payload.node.source_node_id),
                    "question_id": str(chunk.question_id or "unknown"),
                    "target_kind": str(chunk.target_kind or "fact"),
                    "target_field": chunk.target_field,
                    "relation_intent": chunk.relation_intent,
                    "temporal_role": chunk.temporal_role,
                    "query_text": str(chunk.query_text or ""),
                    "source_type": str(chunk.source_type or "unknown"),
                    "source_title": chunk.title,
                    "source_url": chunk.source_url,
                    "source_doc_id": chunk.source_doc_id,
                    "chunk_id": chunk.chunk_id,
                    "page_no": chunk.page_no,
                    "quote": chunk.quote,
                    "content": chunk.content,
                    "retrieval_score": chunk.retrieval_score,
                    "rerank_score": chunk.rerank_score,
                    "source_weight": chunk.source_weight,
                    "provenance_type": (getattr(chunk, "metadata", {}) or {}).get("provenance_type") if isinstance(getattr(chunk, "metadata", {}), dict) else None,
                    "retrieval_method": (getattr(chunk, "metadata", {}) or {}).get("retrieval_method") if isinstance(getattr(chunk, "metadata", {}), dict) else None,
                    "authority_class": (getattr(chunk, "metadata", {}) or {}).get("authority_class") if isinstance(getattr(chunk, "metadata", {}), dict) else None,
                    "source_authority_score": (getattr(chunk, "metadata", {}) or {}).get("source_authority_score") if isinstance(getattr(chunk, "metadata", {}), dict) else None,
                    "final_evidence_score": chunk.final_evidence_score,
                    "metadata": _json({
                        "source_id": chunk.source_id,
                        "source_doc_id": chunk.source_doc_id,
                        "chunk_id": chunk.chunk_id,
                        "page_no": chunk.page_no,
                    }),
                },
            ).mappings().first()
            evidence_id = int(row["id"]) if row else None
            if evidence_id is not None:
                source_to_evidence_id[str(chunk.source_id)] = evidence_id
                saved += 1

        return {"saved": saved, "source_to_evidence_id": source_to_evidence_id}


def _clean_source_url(value: Any) -> str | None:
    url = str(value or "").strip()
    if not url or url.startswith("domain-kb://"):
        return None
    return url


def _row_to_evidence(row: Any) -> dict[str, Any]:
    data = dict(row or {})
    data["source_url"] = _clean_source_url(data.get("source_url"))
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    data["metadata"] = metadata
    data["source"] = {
        "type": data.get("source_type"),
        "title": data.get("source_title"),
        "url": data.get("source_url"),
        "doc_id": data.get("source_doc_id"),
        "chunk_id": data.get("chunk_id"),
        "page_no": data.get("page_no"),
    }
    return data


def list_semantic_evidence_items(
    *,
    job_id: int | None = None,
    trace_id: str | None = None,
    source_scenic_id: str | None = None,
    source_node_id: str | None = None,
    question_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    limit = max(1, min(int(limit or 100), 500))
    offset = max(0, int(offset or 0))
    where: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if job_id is not None:
        where.append("job_id = :job_id")
        params["job_id"] = int(job_id)
    if trace_id:
        where.append("trace_id = :trace_id")
        params["trace_id"] = str(trace_id)
    if source_scenic_id:
        where.append("source_scenic_id = :source_scenic_id")
        params["source_scenic_id"] = str(source_scenic_id)
    if source_node_id:
        where.append("source_node_id = :source_node_id")
        params["source_node_id"] = str(source_node_id)
    if question_id:
        where.append("question_id = :question_id")
        params["question_id"] = str(question_id)
    where_sql = " where " + " and ".join(where) if where else ""
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        total = db.execute(text("select count(*) as n from semantic_evidence_items" + where_sql), params).mappings().first()["n"]
        rows = db.execute(
            text(
                """
                select id, trace_id, job_id, source_scenic_id, source_node_id,
                       question_id, target_kind, target_field, relation_intent, temporal_role,
                       query_text, source_type, source_title, source_url, source_doc_id,
                       chunk_id, page_no, quote, content, retrieval_score, rerank_score,
                       source_weight, provenance_type, retrieval_method, authority_class,
                       source_authority_score, final_evidence_score, metadata, created_at
                from semantic_evidence_items
                """ + where_sql + " order by created_at desc, id desc limit :limit offset :offset"
            ),
            params,
        ).mappings().all()
    return {"items": [_row_to_evidence(row) for row in rows], "total": int(total or 0)}


def get_candidate_evidence(candidate_id: int) -> dict[str, Any] | None:
    with ai_session_scope() as db:
        apply_semantic_completion_schema(db)
        row = db.execute(
            text("select id, evidence_ids from semantic_claim_candidates where id=:id"),
            {"id": int(candidate_id)},
        ).mappings().first()
        if not row:
            return None
        evidence_ids = row.get("evidence_ids") or []
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        evidence_ids = [int(item) for item in evidence_ids if str(item or "").isdigit()]
        if not evidence_ids:
            return {"candidate_id": int(candidate_id), "evidence": []}
        rows = db.execute(
            text(
                """
                select id, trace_id, job_id, source_scenic_id, source_node_id,
                       question_id, target_kind, target_field, relation_intent, temporal_role,
                       query_text, source_type, source_title, source_url, source_doc_id,
                       chunk_id, page_no, quote, content, retrieval_score, rerank_score,
                       source_weight, provenance_type, retrieval_method, authority_class,
                       source_authority_score, final_evidence_score, metadata, created_at
                from semantic_evidence_items
                where id = any(:evidence_ids)
                order by created_at desc, id desc
                """
            ),
            {"evidence_ids": evidence_ids},
        ).mappings().all()
    return {"candidate_id": int(candidate_id), "evidence": [_row_to_evidence(row) for row in rows]}


def get_knowledge_chunk(chunk_id: int) -> dict[str, Any] | None:
    with ai_session_scope() as db:
        row = db.execute(
            text(
                """
                select id, source_scenic_id, source_type, source_id, source_title,
                       title, content, source_url, evidence_text, metadata, created_at
                from knowledge_chunks
                where id=:chunk_id
                limit 1
                """
            ),
            {"chunk_id": int(chunk_id)},
        ).mappings().first()
    if not row:
        return None
    data = dict(row)
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    source_url = _clean_source_url(data.get("source_url"))
    page_no = metadata.get("page_no") or metadata.get("page") or metadata.get("chunk_index")
    try:
        page_no = int(page_no) if page_no not in (None, "") else None
    except Exception:
        page_no = None
    return {
        "chunk_id": int(data["id"]),
        "doc_id": data.get("source_id") or metadata.get("doc_id"),
        "doc_title": data.get("source_title") or data.get("title") or metadata.get("filename"),
        "page_no": page_no,
        "content": data.get("content"),
        "source_file": {
            "source_scenic_id": data.get("source_scenic_id"),
            "source_type": data.get("source_type"),
            "doc_id": data.get("source_id") or metadata.get("doc_id"),
            "filename": metadata.get("filename") or data.get("source_title"),
            "url": source_url,
            "storage_path": metadata.get("storage_path"),
        },
        "metadata": metadata,
    }

