"""Persistent candidate pool for evidence-first semantic completion."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.rag.dependencies import ai_session_scope
from src.rag.schemas import CandidateClaim, ClaimConflict, SemanticCompleteRequest, EvidenceChunk
from src.rag.service.value_normalization_service import canonical_predicate
from src.rag.service.source_independence_service import source_independence_key
from src.rag.service.claim_evidence_fusion_service import TRUST_VERSION, fuse_evidence
from src.rag.service.conflict_classification_service import (
    classify_candidate_group,
    is_exclusive_relation,
    is_multi_value_property,
    _same_value,
)

RAG_DIR = Path(__file__).resolve().parents[1]
MIGRATION_FILES = [
    RAG_DIR / "migrations" / "20260707_semantic_claim_candidates.sql",
    RAG_DIR / "migrations" / "20260711_semantic_completion_jobs.sql",
    RAG_DIR / "migrations" / "20260711_candidate_grouping.sql",
    RAG_DIR / "migrations" / "20260717_entity_resolution_risk.sql",
    RAG_DIR / "migrations" / "20260826_unified_claim_identity.sql",
]
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_ADVISORY_LOCK_NAME = "semantic_completion_schema"


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _execute_statements(db: Session, statements: Iterable[str]) -> None:
    for stmt in statements:
        sql = stmt.strip()
        if sql:
            db.execute(text(sql))


def apply_semantic_candidate_schema(db: Session) -> None:
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
                _execute_statements(db, statements)
            db.commit()
        except Exception:
            db.rollback()
            raise
        _SCHEMA_READY = True


def _claim_dict(claim: CandidateClaim) -> dict[str, Any]:
    if hasattr(claim, "model_dump"):
        return claim.model_dump(mode="json")
    return claim.dict()


def _chunk_map(chunks: list[EvidenceChunk]) -> dict[str, EvidenceChunk]:
    return {str(chunk.source_id): chunk for chunk in chunks if getattr(chunk, "source_id", None)}


def _resolve_source_title(claim: CandidateClaim, chunks_by_id: dict[str, EvidenceChunk]) -> str:
    chunk = chunks_by_id.get(str(claim.source_id or ""))
    return str(getattr(chunk, "title", "") or getattr(chunk, "source", "") or "")[:512]


def _retrieval_source_for_claim(claim: CandidateClaim, chunks_by_id: dict[str, EvidenceChunk]) -> str:
    chunk = chunks_by_id.get(str(claim.source_id or ""))
    source_type = str(getattr(chunk, "source_type", "") or "").strip()
    if source_type.startswith("domain_kb"):
        return "domain_kb"
    if source_type:
        return source_type[:64]
    return "web" if claim.source_url else "local_evidence"


def _has_local_evidence(claim: CandidateClaim, chunks_by_id: dict[str, EvidenceChunk]) -> bool:
    if claim.evidence_ids:
        return True
    chunk = chunks_by_id.get(str(claim.source_id or ""))
    return bool(chunk and (getattr(chunk, "chunk_id", None) or getattr(chunk, "source_doc_id", None)))


def _source_weight_metadata(claim: CandidateClaim) -> dict[str, Any]:
    meta = claim.metadata or {}
    return {
        "source_authority_score": meta.get("source_authority_score"),
        "source_weight": meta.get("source_weight"),
        "provenance_type": meta.get("provenance_type"),
        "retrieval_method": meta.get("retrieval_method"),
        "authority_class": meta.get("authority_class"),
        "target_node_id": meta.get("target_node_id"),
        "target_node_candidate_id": meta.get("target_node_candidate_id"),
        "entity_resolution_status": meta.get("entity_resolution_status"),
        "possible_nodes": meta.get("possible_nodes") or [],
        "raw_type": getattr(claim, "raw_type", None) or meta.get("raw_type"),
        "suggested_type": getattr(claim, "suggested_type", None) or meta.get("suggested_type"),
        "type_confidence": getattr(claim, "type_confidence", 0.0) or meta.get("type_confidence") or 0.0,
        "risk_level": getattr(claim, "risk_level", None) or meta.get("risk_level"),
        "publication_policy": getattr(claim, "publication_policy", None) or meta.get("publication_policy"),
        "score_components": getattr(claim, "score_components", None) or meta.get("score_components") or {},
    }


def _ensure_scenic(db: Session, payload: SemanticCompleteRequest) -> int | None:
    source_scenic_id = str(payload.scenic_id or "").strip()
    if not source_scenic_id:
        return None
    row = db.execute(
        text("select id from scenic_areas where source_scenic_id = :sid order by id limit 1"),
        {"sid": source_scenic_id},
    ).mappings().first()
    if row:
        return int(row["id"])
    result = db.execute(
        text(
            """
            insert into scenic_areas (source_scenic_id, name, description, metadata)
            values (:sid, :name, :description, cast(:metadata as jsonb))
            returning id
            """
        ),
        {
            "sid": source_scenic_id,
            "name": payload.node.scenic_name or source_scenic_id,
            "description": "semantic completion auto-created domain shell",
            "metadata": _json({"source": "semantic_completion", "a_scenic_id": source_scenic_id}),
        },
    ).mappings().first()
    return int(result["id"]) if result else None


def _schema_values(payload: SemanticCompleteRequest, key: str) -> set[str]:
    schema = (payload.metadata or {}).get("domain_schema") or {}
    values: set[str] = set()
    if key == "properties":
        schema_map = schema.get("schema_map") or {}
        for fields in schema_map.values():
            for item in fields or []:
                text_value = str(item or "").strip()
                if text_value:
                    values.add(text_value)
    elif key == "relations":
        relation_intents = schema.get("relation_intents") or {}
        for rels in relation_intents.values():
            for item in rels or []:
                if isinstance(item, dict):
                    for sub_key in ("label", "code"):
                        text_value = str(item.get(sub_key) or "").strip()
                        if text_value:
                            values.add(text_value)
                else:
                    text_value = str(item or "").strip()
                    if text_value:
                        values.add(text_value)
    return values


def _candidate_type(payload: SemanticCompleteRequest, claim: CandidateClaim) -> str:
    predicate = str(claim.predicate or "").strip()
    if claim.claim_type == "property":
        template_props = set(payload.target_fields or []) | _schema_values(payload, "properties")
        return "template_property" if predicate in template_props else "discovered_property"
    if claim.claim_type == "relation":
        template_rels = set(payload.relation_intents or []) | _schema_values(payload, "relations")
        return "template_relation" if predicate in template_rels else "discovered_relation"
    if claim.claim_type == "entity":
        return "discovered_entity"
    return "discovered_fact"


def _claim_value(claim: CandidateClaim) -> str:
    if claim.claim_type == "relation":
        return str(claim.normalized_value or claim.display_value or claim.object_name or claim.object_value or "").strip()
    return str(claim.normalized_value or claim.display_value or claim.object_value or claim.object_name or "").strip()


def _conflict_key(payload: SemanticCompleteRequest, claim: CandidateClaim) -> str:
    predicate_key = canonical_predicate(claim.predicate or "", temporal_role=claim.temporal_role)
    return "|".join([
        str(payload.scenic_id or ""),
        str(claim.subject_node_id or payload.node.source_node_id or ""),
        str(claim.claim_type or ""),
        str(predicate_key or claim.predicate or ""),
        str(claim.temporal_role or ""),
    ])


def _candidate_uid(payload: SemanticCompleteRequest, trace_id: str, claim: CandidateClaim) -> str:
    source_identity = str(
        claim.source_url
        or claim.source_id
        or ",".join(str(value) for value in (claim.evidence_ids or []))
        or ""
    ).strip().casefold()
    predicate = canonical_predicate(
        claim.predicate or "",
        temporal_role=claim.temporal_role,
    ) or str(claim.predicate or "").strip()
    return _hash({
        "source_scenic_id": payload.scenic_id,
        "source_node_id": claim.subject_node_id or payload.node.source_node_id,
        "claim_type": claim.claim_type,
        "predicate": predicate.casefold(),
        "value": _claim_value(claim).casefold(),
        "target_node_id": (claim.metadata or {}).get("target_node_id"),
        "target_node_candidate_id": (claim.metadata or {}).get("target_node_candidate_id"),
        "source_identity": source_identity,
    })


def _claim_source_key(claim: CandidateClaim) -> str:
    metadata = claim.metadata if isinstance(claim.metadata, dict) else {}
    return source_independence_key({
        "source_type": metadata.get("source_type") or metadata.get("provenance_type"),
        "source_doc_id": metadata.get("source_doc_id") or metadata.get("doc_id"),
        "source_url": claim.source_url or metadata.get("source_url"),
        "source_id": claim.source_id,
        "chunk_id": metadata.get("chunk_id"),
        "evidence_unit_uid": claim.source_id,
    })


def _completion_fusion_binding(claim: CandidateClaim, chunks_by_id: dict[str, EvidenceChunk]) -> dict[str, Any]:
    """Build the shared fusion input without changing completion ranking yet."""
    metadata = claim.metadata if isinstance(claim.metadata, dict) else {}
    chunk = chunks_by_id.get(str(claim.source_id or ""))
    content = str(getattr(chunk, "content", "") or "")
    quote = str(claim.quote or getattr(chunk, "quote", "") or "").strip()
    if quote and content:
        evidence_locality = 1.0 if quote in content else 0.5
        quote_coverage = 1.0 if quote in content else min(1.0, len(quote) / max(1, len(content)))
    else:
        evidence_locality = 0.0
        quote_coverage = 0.0
    return {
        "source_independence_key": _claim_source_key(claim),
        "source_type": getattr(chunk, "source_type", None) or metadata.get("source_type") or metadata.get("provenance_type") or "unknown",
        "source_quality": metadata.get("source_authority_score") or metadata.get("source_weight") or getattr(chunk, "source_weight", None),
        "retrieval_relevance": metadata.get("retrieval_relevance") or getattr(chunk, "rerank_score", None) or getattr(chunk, "retrieval_score", None) or getattr(chunk, "score", None),
        "quote_coverage": metadata.get("quote_coverage", quote_coverage),
        "evidence_locality": metadata.get("evidence_locality", evidence_locality),
        "entailment_score": metadata.get("entailment_score"),
        "extraction_confidence": metadata.get("extraction_confidence", claim.confidence),
        "entity_resolution_confidence": metadata.get("entity_resolution_confidence", claim.confidence),
    }


def _claim_allows_multi_value(payload: SemanticCompleteRequest, claim: CandidateClaim) -> bool:
    predicate = canonical_predicate(claim.predicate or "", temporal_role=claim.temporal_role).strip()
    if claim.claim_type == "property":
        if claim.temporal_role:
            return False
        return is_multi_value_property(predicate, payload)
    if claim.claim_type == "relation":
        return not is_exclusive_relation(predicate, payload)
    return False


def _ensure_candidate_group_keys(payload: SemanticCompleteRequest, claim: CandidateClaim, conflict_key: str) -> tuple[str, str]:
    group_key = str(getattr(claim, "candidate_group_key", None) or "").strip()
    if not group_key:
        group_key = _hash({"candidate_group_key": conflict_key})[:32]
        setattr(claim, "candidate_group_key", group_key)
    value_key = str(getattr(claim, "value_group_key", None) or "").strip()
    if not value_key:
        value_key = _hash({"candidate_group_key": group_key, "value": _claim_value(claim)})[:32]
        setattr(claim, "value_group_key", value_key)
    return group_key, value_key


_BOOLEAN_CLAUSE_CUES = (
    "锚定", "落实", "推动", "打造", "建成", "构建", "营造", "承担", "获得",
    "努力", "矢志", "做出", "服务", "坚持", "开展", "形成", "入选", "存在",
    "达到", "突破", "培养", "建设",
)


def _malformed_claim_reason(claim: CandidateClaim) -> str | None:
    predicate = str(claim.predicate or "").strip()
    value = str(_claim_value(claim) or "").strip()
    if value not in {"是", "否"}:
        if claim.claim_type == "property" and any(cue in predicate for cue in _BOOLEAN_CLAUSE_CUES):
            if value == predicate or (len(value) >= 4 and value in predicate and len(predicate) > len(value) + 1):
                return "CLAUSE_AS_PROPERTY_SELF_VALUE"
            if not re.search(r"[0-9]|年|月|日|亿元|万元|平方米|公里|项|人|号|次|%",
                             value):
                return "ACTION_CLAUSE_AS_PROPERTY"
        return None
    if predicate.startswith(("是否", "有无")):
        return None
    if any(mark in predicate for mark in ("“", "”", '"', "‘", "’")):
        return "MALFORMED_PREDICATE_BOOLEAN_VALUE"
    if any(cue in predicate for cue in _BOOLEAN_CLAUSE_CUES) or len(predicate) > 12:
        return "CLAUSE_AS_PREDICATE_BOOLEAN_VALUE"
    return None


def _status_for_claim(claim: CandidateClaim, forced_conflict: bool = False) -> str:
    if forced_conflict or claim.status == "conflict_candidate":
        return "CONFLICT"
    if claim.status == "duplicate":
        return "DUPLICATE"
    if claim.status == "low_evidence":
        return "LOW_EVIDENCE"
    if claim.status in {"adoptable", "needs_review"}:
        return "PENDING"
    return str(claim.status or "PENDING").upper()


def _existing_conflict_claim_ids(conflicts: list[ClaimConflict]) -> set[str]:
    return {str(item.claim_id) for item in conflicts if getattr(item, "claim_id", None)}


def _meaningful_conflict_ids(conflicts: list[ClaimConflict]) -> set[str]:
    ids: set[str] = set()
    for item in conflicts:
        if not item.claim_id or str(getattr(item, "conflict_type", "") or "") not in {"conflicting", "scope_mismatch", "entity_ambiguity"}:
            continue
        existing = str(getattr(item, "existing_value", "") or getattr(item, "existing_target", "") or "").strip()
        candidate = str(getattr(item, "candidate_value", "") or getattr(item, "candidate_target", "") or "").strip()
        if existing and candidate and _same_value(existing, candidate):
            continue
        ids.add(str(item.claim_id))
    return ids


def persist_semantic_candidates(
    payload: SemanticCompleteRequest,
    *,
    trace_id: str,
    job_id: int | None = None,
    claims: list[CandidateClaim],
    conflicts: list[ClaimConflict],
    chunks: list[EvidenceChunk],
) -> dict[str, Any]:
    if not claims:
        return {"saved": 0, "conflict_groups": 0}

    graph_context = payload.graph_context or {}
    graph_scope = str(graph_context.get("scope") or "")
    chunks_by_id = _chunk_map(chunks)
    conflict_ids = _meaningful_conflict_ids(conflicts)

    key_values: dict[str, set[str]] = {}
    key_sources: dict[str, set[str]] = {}
    key_requires_single_value: dict[str, bool] = {}
    for claim in claims:
        key = _conflict_key(payload, claim)
        _ensure_candidate_group_keys(payload, claim, key)
        value = _claim_value(claim)
        source = _claim_source_key(claim)
        if value:
            key_values.setdefault(key, set()).add(value)
        if source:
            key_sources.setdefault(key, set()).add(source)
        if key not in key_requires_single_value:
            key_requires_single_value[key] = not _claim_allows_multi_value(payload, claim)

    run_conflict_keys = {
        key for key, values in key_values.items()
        if len(values) > 1 and key_requires_single_value.get(key, True)
    }
    run_multi_value_keys = {
        key for key, values in key_values.items()
        if len(values) > 1 and not key_requires_single_value.get(key, True)
    }

    # Compute trust-v1 per canonical value as a shadow result.  Candidate
    # recommendation still uses the existing score until the UI/thresholds are
    # explicitly migrated, but every completion candidate now carries the same
    # explainable fusion contract as growth candidates.
    fusion_bindings: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for claim in claims:
        if not claim.quote:
            continue
        identity = (_conflict_key(payload, claim), _claim_value(claim))
        if identity[1]:
            fusion_bindings.setdefault(identity, []).append(_completion_fusion_binding(claim, chunks_by_id))
    fusion_by_claim_id: dict[str, dict[str, Any]] = {}
    for claim in claims:
        identity = (_conflict_key(payload, claim), _claim_value(claim))
        fusion_by_claim_id[str(claim.claim_id)] = fuse_evidence(fusion_bindings.get(identity, []))

    saved = 0
    conflict_group_ids: set[str] = set()
    saved_group_claims: list[CandidateClaim] = []
    with ai_session_scope() as db:
        apply_semantic_candidate_schema(db)
        scenic_pk = _ensure_scenic(db, payload)
        for claim in claims:
            if not claim.quote:
                continue
            if not claim.source_url and not _has_local_evidence(claim, chunks_by_id):
                continue
            key = _conflict_key(payload, claim)
            _ensure_candidate_group_keys(payload, claim, key)
            if key in run_multi_value_keys and not getattr(claim, "conflict_class", None):
                setattr(claim, "conflict_class", "multi_value")
                setattr(claim, "gap_status", "pending_review")
            forced_conflict = str(getattr(claim, "conflict_class", "") or "") in {"conflicting", "scope_mismatch", "entity_ambiguity"} or key in run_conflict_keys or str(claim.claim_id) in conflict_ids
            conflict_group = _hash({"conflict_key": key})[:32] if forced_conflict else ""
            if conflict_group:
                conflict_group_ids.add(conflict_group)
            candidate_uid = _candidate_uid(payload, trace_id, claim)
            ctype = _candidate_type(payload, claim)
            raw = _claim_dict(claim)
            source_meta = _source_weight_metadata(claim)
            fusion = fusion_by_claim_id.get(str(claim.claim_id)) or fuse_evidence([])
            malformed_reason = _malformed_claim_reason(claim)
            metadata = {
                "a_user_id": (payload.metadata or {}).get("user_id"),
                "a_username": (payload.metadata or {}).get("username"),
                "domain_schema_present": bool((payload.metadata or {}).get("domain_schema")),
                "creates_entity": bool(claim.claim_type == "relation" and claim.object_name),
                "graph_scope": graph_scope,
                "source_endpoint": (payload.metadata or {}).get("source_endpoint"),
                "canonical_predicate": claim.metadata.get("canonical_predicate") if isinstance(claim.metadata, dict) else None,
                "raw_value": claim.raw_value,
                "normalized_value": claim.normalized_value,
                "display_value": claim.display_value,
                "temporal_role": claim.temporal_role,
                "candidate_group_key": getattr(claim, "candidate_group_key", None),
                "claim_quality": "INVALIDATED" if malformed_reason else "VALID",
                "invalid_reason": malformed_reason,
                "value_group_key": getattr(claim, "value_group_key", None),
                "conflict_class": getattr(claim, "conflict_class", None),
                "gap_status": getattr(claim, "gap_status", None),
                "source_authority_score": source_meta.get("source_authority_score"),
                "source_weight": source_meta.get("source_weight"),
                "provenance_type": source_meta.get("provenance_type"),
                "retrieval_method": source_meta.get("retrieval_method"),
                "authority_class": source_meta.get("authority_class"),
                "target_node_id": source_meta.get("target_node_id"),
                "target_node_candidate_id": source_meta.get("target_node_candidate_id"),
                "entity_resolution_status": source_meta.get("entity_resolution_status"),
                "possible_nodes": source_meta.get("possible_nodes") or [],
                "raw_type": source_meta.get("raw_type"),
                "suggested_type": source_meta.get("suggested_type"),
                "type_confidence": source_meta.get("type_confidence") or 0.0,
                "risk_level": source_meta.get("risk_level"),
                "publication_policy": source_meta.get("publication_policy"),
                "score_components": source_meta.get("score_components") or {},
                "allow_multi_value": _claim_allows_multi_value(payload, claim),
                "value_policy": "multi_value" if _claim_allows_multi_value(payload, claim) else "single_value",
                "is_recommended": False,
                "recommendation_rank": None,
                "source_independence_key": _claim_source_key(claim),
                "trust_v1_shadow": fusion,
            }
            claim_status = "INVALIDATED" if malformed_reason else _status_for_claim(claim, forced_conflict)
            row = db.execute(
                text(
                    """
                    insert into semantic_claim_candidates (
                        candidate_uid, trace_id, run_id, scenic_id, source_scenic_id,
                        source_node_id, subject_name, subject_type, graph_scope, retrieval_source,
                        claim_id, claim_type, candidate_type, predicate, object_value,
                        object_name, object_type, source_id, source_title, source_url, quote,
                        confidence, evidence_score, evidence_status, status,
                        job_id, question_id, evidence_ids, recommend_score, support_status,
                        candidate_group_key, value_group_key, conflict_class, gap_status,
                        source_authority_score, source_weight, provenance_type, retrieval_method, authority_class,
                        target_node_id, target_node_candidate_id, entity_resolution_status, possible_nodes,
                        raw_type, suggested_type, type_confidence, risk_level, publication_policy, score_components,
                        conflict_key, conflict_group, raw_payload, metadata,
                        canonical_claim_key, conflict_scope_key, trust_version,
                        trust_components, final_trust_score, updated_at
                    ) values (
                        :candidate_uid, :trace_id, :run_id, :scenic_id, :source_scenic_id,
                        :source_node_id, :subject_name, :subject_type, :graph_scope, :retrieval_source,
                        :claim_id, :claim_type, :candidate_type, :predicate, :object_value,
                        :object_name, :object_type, :source_id, :source_title, :source_url, :quote,
                        :confidence, :evidence_score, :evidence_status, :status,
                        :job_id, :question_id, cast(:evidence_ids as jsonb), :recommend_score, :support_status,
                        :candidate_group_key, :value_group_key, :conflict_class, :gap_status,
                        :source_authority_score, :source_weight, :provenance_type, :retrieval_method, :authority_class,
                        :target_node_id, :target_node_candidate_id, :entity_resolution_status, cast(:possible_nodes as jsonb),
                        :raw_type, :suggested_type, :type_confidence, :risk_level, :publication_policy, cast(:score_components as jsonb),
                        :conflict_key, :conflict_group, cast(:raw_payload as jsonb), cast(:metadata as jsonb),
                        :canonical_claim_key, :conflict_scope_key, :trust_version,
                        cast(:trust_components as jsonb), :final_trust_score, now()
                    )
                    on conflict (candidate_uid) do update set
                        status = excluded.status,
                        confidence = excluded.confidence,
                        evidence_score = excluded.evidence_score,
                        evidence_status = excluded.evidence_status,
                        question_id = excluded.question_id,
                        evidence_ids = excluded.evidence_ids,
                        recommend_score = excluded.recommend_score,
                        support_status = excluded.support_status,
                        candidate_group_key = excluded.candidate_group_key,
                        value_group_key = excluded.value_group_key,
                        conflict_class = excluded.conflict_class,
                        gap_status = excluded.gap_status,
                        source_authority_score = excluded.source_authority_score,
                        source_weight = excluded.source_weight,
                        provenance_type = excluded.provenance_type,
                        retrieval_method = excluded.retrieval_method,
                        authority_class = excluded.authority_class,
                        target_node_id = excluded.target_node_id,
                        target_node_candidate_id = excluded.target_node_candidate_id,
                        entity_resolution_status = excluded.entity_resolution_status,
                        possible_nodes = excluded.possible_nodes,
                        raw_type = excluded.raw_type,
                        suggested_type = excluded.suggested_type,
                        type_confidence = excluded.type_confidence,
                        risk_level = excluded.risk_level,
                        publication_policy = excluded.publication_policy,
                        score_components = excluded.score_components,
                        conflict_group = excluded.conflict_group,
                        raw_payload = excluded.raw_payload,
                        metadata = excluded.metadata,
                        canonical_claim_key = excluded.canonical_claim_key,
                        conflict_scope_key = excluded.conflict_scope_key,
                        trust_version = excluded.trust_version,
                        trust_components = excluded.trust_components,
                        final_trust_score = excluded.final_trust_score,
                        updated_at = now()
                    returning id
                    """
                ),
                {
                    "candidate_uid": candidate_uid,
                    "trace_id": trace_id,
                    "run_id": trace_id,
                    "scenic_id": scenic_pk,
                    "source_scenic_id": str(payload.scenic_id),
                    "source_node_id": str(claim.subject_node_id or payload.node.source_node_id or ""),
                    "subject_name": claim.subject_name or payload.node.name,
                    "subject_type": payload.node.node_type,
                    "graph_scope": graph_scope,
                    "retrieval_source": _retrieval_source_for_claim(claim, chunks_by_id),
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type,
                    "candidate_type": ctype,
                    "predicate": claim.predicate,
                    "object_value": claim.object_value,
                    "object_name": claim.object_name,
                    "object_type": claim.object_type,
                    "source_id": claim.source_id,
                    "source_title": _resolve_source_title(claim, chunks_by_id),
                    "source_url": claim.source_url,
                    "quote": claim.quote,
                    "confidence": claim.confidence,
                    "evidence_score": claim.evidence_score,
                    "evidence_status": claim.evidence_status,
                    "status": claim_status,
                    "job_id": int(job_id) if job_id is not None else None,
                    "question_id": claim.question_id,
                    "evidence_ids": _json(claim.evidence_ids or []),
                    "recommend_score": float(claim.recommend_score or 0.0),
                    "support_status": str(claim.support_status or "needs_more_evidence"),
                    "candidate_group_key": getattr(claim, "candidate_group_key", None),
                    "value_group_key": getattr(claim, "value_group_key", None),
                    "conflict_class": getattr(claim, "conflict_class", None),
                    "gap_status": getattr(claim, "gap_status", None),
                    "source_authority_score": source_meta.get("source_authority_score"),
                    "source_weight": source_meta.get("source_weight"),
                    "provenance_type": source_meta.get("provenance_type"),
                    "retrieval_method": source_meta.get("retrieval_method"),
                    "authority_class": source_meta.get("authority_class"),
                    "target_node_id": source_meta.get("target_node_id"),
                    "target_node_candidate_id": source_meta.get("target_node_candidate_id"),
                    "entity_resolution_status": source_meta.get("entity_resolution_status"),
                    "possible_nodes": _json(source_meta.get("possible_nodes") or []),
                    "raw_type": source_meta.get("raw_type"),
                    "suggested_type": source_meta.get("suggested_type"),
                    "type_confidence": float(source_meta.get("type_confidence") or 0.0),
                    "risk_level": source_meta.get("risk_level"),
                    "publication_policy": source_meta.get("publication_policy"),
                    "score_components": _json(source_meta.get("score_components") or {}),
                    "conflict_key": key,
                    "conflict_group": conflict_group,
                    "raw_payload": _json(raw),
                    "metadata": _json(metadata),
                    "canonical_claim_key": (claim.metadata or {}).get("canonical_claim_key"),
                    "conflict_scope_key": (claim.metadata or {}).get("conflict_scope_key"),
                    "trust_version": TRUST_VERSION,
                    "trust_components": _json(fusion),
                    "final_trust_score": float(fusion.get("evidence_support_score") or claim.recommend_score or 0.0),
                },
            ).mappings().first()
            candidate_id = int(row["id"]) if row else None
            claim.metadata = dict(claim.metadata or {})
            claim.metadata.update({
                "candidate_id": candidate_id,
                "candidate_uid": candidate_uid,
                "candidate_type": ctype,
                "conflict_group": conflict_group,
                "conflict_key": key,
                "candidate_group_key": getattr(claim, "candidate_group_key", None),
                "value_group_key": getattr(claim, "value_group_key", None),
                "conflict_class": getattr(claim, "conflict_class", None),
                "gap_status": getattr(claim, "gap_status", None),
                "source_authority_score": source_meta.get("source_authority_score"),
                "source_weight": source_meta.get("source_weight"),
                "provenance_type": source_meta.get("provenance_type"),
                "retrieval_method": source_meta.get("retrieval_method"),
                "authority_class": source_meta.get("authority_class"),
                "source_independence_key": _claim_source_key(claim),
                "trust_v1_shadow": fusion,
            })
            if candidate_id is not None:
                saved_group_claims.append(claim)
            saved += 1

        grouped_claims: dict[str, list[CandidateClaim]] = {}
        for claim in saved_group_claims:
            group_key = str(getattr(claim, "candidate_group_key", "") or "")
            if group_key:
                grouped_claims.setdefault(group_key, []).append(claim)
        for group_key, group_claims_for_key in grouped_claims.items():
            ranked = sorted(group_claims_for_key, key=lambda item: float(item.recommend_score or 0.0), reverse=True)
            best = ranked[0]
            values = {_claim_value(item) for item in group_claims_for_key if _claim_value(item)}
            sources = {_claim_source_key(item) for item in group_claims_for_key if _claim_source_key(item)}
            value_sources: dict[str, set[str]] = {}
            value_candidate_uids: dict[str, list[str]] = {}
            source_records: dict[str, dict[str, Any]] = {}
            value_fusion: dict[str, dict[str, Any]] = {}
            for item in group_claims_for_key:
                value = _claim_value(item)
                source_key = _claim_source_key(item)
                if value and source_key:
                    value_sources.setdefault(value, set()).add(source_key)
                if value and isinstance(item.metadata, dict) and item.metadata.get("candidate_uid"):
                    value_candidate_uids.setdefault(value, []).append(str(item.metadata.get("candidate_uid")))
                if value and value not in value_fusion:
                    value_fusion[value] = fusion_by_claim_id.get(str(item.claim_id)) or fuse_evidence([])
                if source_key:
                    source_records[source_key] = {
                        "source_key": source_key,
                        "source_id": item.source_id,
                        "source_url": item.source_url,
                        "source_title": _resolve_source_title(item, chunks_by_id),
                        "retrieval_source": _retrieval_source_for_claim(item, chunks_by_id),
                        "evidence_ids": item.evidence_ids or [],
                    }
            group_info = classify_candidate_group(group_claims_for_key, payload)
            allow_multi_value = any(_claim_allows_multi_value(payload, item) for item in group_claims_for_key)
            if allow_multi_value:
                recommended_by_value: dict[str, CandidateClaim] = {}
                for item in ranked:
                    value = _claim_value(item)
                    if value and value not in recommended_by_value:
                        recommended_by_value[value] = item
                recommended_claims = list(recommended_by_value.values())
            else:
                recommended_claims = [best]
            recommended_uids = {
                str(item.metadata.get("candidate_uid"))
                for item in recommended_claims
                if isinstance(item.metadata, dict) and item.metadata.get("candidate_uid")
            }
            recommendation_rows = []
            alternative_rows = []
            for rank, item in enumerate(ranked, start=1):
                uid = str(item.metadata.get("candidate_uid")) if isinstance(item.metadata, dict) else ""
                value = _claim_value(item)
                row = {
                    "candidate_uid": uid,
                    "claim_id": item.claim_id,
                    "value": value,
                    "recommend_score": round(float(item.recommend_score or 0.0), 3),
                    "source_count": len(value_sources.get(value, set())),
                    "source_keys": sorted(value_sources.get(value, set())),
                    "rank": rank,
                }
                if uid in recommended_uids:
                    recommendation_rows.append(row)
                else:
                    alternative_rows.append(row)
                db.execute(
                    text(
                        """
                        update semantic_claim_candidates
                        set metadata = coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                            updated_at = now()
                        where candidate_uid = :candidate_uid
                        """
                    ),
                    {
                        "candidate_uid": uid,
                        "patch": _json({
                            "is_recommended": uid in recommended_uids,
                            "recommendation_rank": rank,
                            "same_value_source_count": len(value_sources.get(value, set())),
                            "same_value_sources": sorted(value_sources.get(value, set())),
                            "source_independence_key": _claim_source_key(item),
                            "trust_v1_shadow": fusion_by_claim_id.get(str(item.claim_id)) or fuse_evidence([]),
                            "group_candidate_count": len(group_claims_for_key),
                            "group_distinct_value_count": len(values),
                            "group_source_count": len(sources),
                        }),
                    },
                )
            db.execute(
                text(
                    """
                    insert into semantic_candidate_groups (
                        candidate_group_key, trace_id, job_id, scenic_id, source_scenic_id,
                        source_node_id, question_id, claim_type, predicate, temporal_role,
                        conflict_class, gap_status, candidate_count, distinct_value_count,
                        source_count, best_candidate_uid, recommend_score, risk_level, publication_policy, score_components, metadata, updated_at
                    ) values (
                        :candidate_group_key, :trace_id, :job_id, :scenic_id, :source_scenic_id,
                        :source_node_id, :question_id, :claim_type, :predicate, :temporal_role,
                        :conflict_class, :gap_status, :candidate_count, :distinct_value_count,
                        :source_count, :best_candidate_uid, :recommend_score, :risk_level, :publication_policy, cast(:score_components as jsonb), cast(:metadata as jsonb), now()
                    )
                    on conflict (candidate_group_key) do update set
                        trace_id = excluded.trace_id,
                        job_id = excluded.job_id,
                        source_node_id = excluded.source_node_id,
                        question_id = excluded.question_id,
                        claim_type = excluded.claim_type,
                        predicate = excluded.predicate,
                        temporal_role = excluded.temporal_role,
                        conflict_class = excluded.conflict_class,
                        gap_status = excluded.gap_status,
                        candidate_count = excluded.candidate_count,
                        distinct_value_count = excluded.distinct_value_count,
                        source_count = excluded.source_count,
                        best_candidate_uid = excluded.best_candidate_uid,
                        recommend_score = excluded.recommend_score,
                        risk_level = excluded.risk_level,
                        publication_policy = excluded.publication_policy,
                        score_components = excluded.score_components,
                        metadata = excluded.metadata,
                        updated_at = now()
                    """
                ),
                {
                    "candidate_group_key": group_key,
                    "trace_id": trace_id,
                    "job_id": int(job_id) if job_id is not None else None,
                    "scenic_id": scenic_pk,
                    "source_scenic_id": str(payload.scenic_id),
                    "source_node_id": str(payload.node.source_node_id or ""),
                    "question_id": best.question_id,
                    "claim_type": best.claim_type,
                    "predicate": (best.metadata.get("canonical_predicate") or best.predicate) if isinstance(best.metadata, dict) else best.predicate,
                    "temporal_role": best.temporal_role,
                    "conflict_class": str(group_info.get("conflict_class") or getattr(best, "conflict_class", None) or "insufficient"),
                    "gap_status": str(group_info.get("gap_status") or getattr(best, "gap_status", None) or "needs_review"),
                    "candidate_count": len(group_claims_for_key),
                    "distinct_value_count": len(values),
                    "source_count": len(sources),
                    "best_candidate_uid": best.metadata.get("candidate_uid") if isinstance(best.metadata, dict) else None,
                    "recommend_score": float(group_info.get("recommend_score") or best.recommend_score or 0.0),
                    "risk_level": getattr(best, "risk_level", None),
                    "publication_policy": getattr(best, "publication_policy", None),
                    "score_components": _json(getattr(best, "score_components", None) or {}),
                    "metadata": _json({
                        "values": sorted(values),
                        "sources": sorted(sources),
                        "value_sources": {key: sorted(items) for key, items in value_sources.items()},
                        "value_candidate_uids": value_candidate_uids,
                        "source_records": list(source_records.values()),
                        "value_trust_v1_shadow": value_fusion,
                        "claim_ids": [item.claim_id for item in group_claims_for_key],
                        "allow_multi_value": allow_multi_value,
                        "value_policy": "multi_value" if allow_multi_value else "single_value",
                        "recommended_candidates": recommendation_rows,
                        "alternatives": alternative_rows,
                        "result_contract_version": "p3-candidate-policy-v1",
                    }),
                },
            )

        for group in conflict_group_ids:
            matching_key = ""
            group_claims = []
            for claim in claims:
                key = _conflict_key(payload, claim)
                if _hash({"conflict_key": key})[:32] == group:
                    matching_key = key
                    group_claims.append(claim)
            values = {_claim_value(c) for c in group_claims if _claim_value(c)}
            db.execute(
                text(
                    """
                    insert into semantic_conflict_groups (
                        conflict_group, conflict_key, trace_id, scenic_id, source_scenic_id,
                        source_node_id, claim_type, predicate, conflict_type,
                        candidate_count, distinct_value_count, status, summary, metadata, updated_at
                    ) values (
                        :conflict_group, :conflict_key, :trace_id, :scenic_id, :source_scenic_id,
                        :source_node_id, :claim_type, :predicate, :conflict_type,
                        :candidate_count, :distinct_value_count, 'PENDING', :summary, cast(:metadata as jsonb), now()
                    )
                    on conflict (conflict_group) do update set
                        candidate_count = excluded.candidate_count,
                        distinct_value_count = excluded.distinct_value_count,
                        summary = excluded.summary,
                        metadata = excluded.metadata,
                        updated_at = now()
                    """
                ),
                {
                    "conflict_group": group,
                    "conflict_key": matching_key,
                    "trace_id": trace_id,
                    "scenic_id": scenic_pk,
                    "source_scenic_id": str(payload.scenic_id),
                    "source_node_id": str(payload.node.source_node_id or ""),
                    "claim_type": group_claims[0].claim_type if group_claims else "unknown",
                    "predicate": group_claims[0].predicate if group_claims else "",
                    "conflict_type": str(getattr(group_claims[0], "conflict_class", None) or "conflicting") if group_claims else "conflicting",
                    "candidate_count": len(group_claims),
                    "distinct_value_count": len(values),
                    "summary": f"\u5019\u9009\u5b58\u5728 {len(values)} \u4e2a\u4e0d\u540c\u8bf4\u6cd5",
                    "metadata": _json({"values": sorted(values), "trace_id": trace_id}),
                },
            )
    return {"saved": saved, "conflict_groups": len(conflict_group_ids)}



def list_semantic_candidates(
    *,
    source_scenic_id: str | None = None,
    source_node_id: str | None = None,
    trace_id: str | None = None,
    job_id: int | None = None,
    status: str | None = None,
    risk_level: str | None = None,
    publication_policy: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    limit = max(1, min(int(limit or 100), 500))
    offset = max(0, int(offset or 0))
    where = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if source_scenic_id:
        where.append("source_scenic_id = :source_scenic_id")
        params["source_scenic_id"] = str(source_scenic_id)
    if source_node_id:
        where.append("source_node_id = :source_node_id")
        params["source_node_id"] = str(source_node_id)
    if trace_id:
        where.append("trace_id = :trace_id")
        params["trace_id"] = str(trace_id)
    if job_id is not None:
        where.append("job_id = :job_id")
        params["job_id"] = int(job_id)
    if status:
        where.append("status = :status")
        params["status"] = str(status).upper()
    if risk_level:
        where.append("risk_level = :risk_level")
        params["risk_level"] = str(risk_level).upper()
    if publication_policy:
        where.append("publication_policy = :publication_policy")
        params["publication_policy"] = str(publication_policy).upper()
    where_sql = " where " + " and ".join(where) if where else ""
    # The same source can emit the same semantic claim more than once (for
    # example one raw extraction and one normalized extraction).  The review
    # queue must expose one row for that identity, while preserving the
    # preferred terminal/reviewed row when it exists.
    identity_partition = """
        source_scenic_id,
        source_node_id,
        lower(trim(coalesce(claim_type, ''))),
        lower(trim(coalesce(predicate, ''))),
        lower(trim(coalesce(object_value, ''))),
        lower(trim(coalesce(object_name, ''))),
        lower(trim(coalesce(target_node_id::text, ''))),
        lower(trim(coalesce(target_node_candidate_id::text, ''))),
        lower(trim(coalesce(source_url, source_id, source_title, '')))
    """
    ranked_from = """
        select c.*,
               row_number() over (
                   partition by {identity_partition}
                   order by
                       case
                           when upper(coalesce(status, '')) in ('ADOPTED', 'PUBLISHED') then 0
                           when upper(coalesce(status, '')) in ('REJECTED', 'INVALIDATED') then 2
                           else 1
                       end,
                       updated_at desc nulls last,
                       created_at desc nulls last,
                       id desc
               ) as _dedupe_rank
        from semantic_claim_candidates c
    """.format(identity_partition=identity_partition)
    ranked_sql = ranked_from + where_sql
    columns = """
        id, candidate_uid, trace_id, run_id, source_scenic_id, source_node_id,
        subject_name, subject_type, graph_scope, retrieval_source,
        claim_id, claim_type, candidate_type, predicate, object_value,
        object_name, object_type, source_id, source_title, source_url, quote,
        confidence, evidence_score, evidence_status, status,
        job_id, question_id, evidence_ids, recommend_score, support_status,
        candidate_group_key, value_group_key, conflict_class, gap_status,
        source_authority_score, source_weight, provenance_type, retrieval_method, authority_class,
        target_node_id, target_node_candidate_id, entity_resolution_status, possible_nodes,
        raw_type, suggested_type, type_confidence, risk_level, publication_policy, score_components,
        conflict_key, conflict_group, metadata, created_at, updated_at
    """
    with ai_session_scope() as db:
        apply_semantic_candidate_schema(db)
        total = db.execute(
            text("select count(*) as n from (" + ranked_sql + ") ranked where _dedupe_rank = 1"),
            params,
        ).mappings().first()["n"]
        rows = db.execute(
            text(
                "select " + columns + " from (" + ranked_sql + ") ranked "
                "where _dedupe_rank = 1 order by created_at desc nulls last, id desc "
                "limit :limit offset :offset"
            ),
            params,
        ).mappings().all()
        return {"items": [dict(row) for row in rows], "total": int(total or 0)}



def list_semantic_candidate_groups(
    *,
    source_scenic_id: str | None = None,
    source_node_id: str | None = None,
    trace_id: str | None = None,
    job_id: int | None = None,
    gap_status: str | None = None,
    conflict_class: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    limit = max(1, min(int(limit or 100), 500))
    offset = max(0, int(offset or 0))
    where = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if source_scenic_id:
        where.append("source_scenic_id = :source_scenic_id")
        params["source_scenic_id"] = str(source_scenic_id)
    if source_node_id:
        where.append("source_node_id = :source_node_id")
        params["source_node_id"] = str(source_node_id)
    if trace_id:
        where.append("trace_id = :trace_id")
        params["trace_id"] = str(trace_id)
    if job_id is not None:
        where.append("job_id = :job_id")
        params["job_id"] = int(job_id)
    if gap_status:
        where.append("gap_status = :gap_status")
        params["gap_status"] = str(gap_status)
    if conflict_class:
        where.append("conflict_class = :conflict_class")
        params["conflict_class"] = str(conflict_class)
    where_sql = " where " + " and ".join(where) if where else ""
    with ai_session_scope() as db:
        apply_semantic_candidate_schema(db)
        total = db.execute(text("select count(*) as n from semantic_candidate_groups" + where_sql), params).mappings().first()["n"]
        rows = db.execute(
            text(
                """
                select id, candidate_group_key, trace_id, job_id, source_scenic_id,
                       source_node_id, question_id, claim_type, predicate, temporal_role,
                       conflict_class, gap_status, candidate_count, distinct_value_count,
                       source_count, best_candidate_uid, recommend_score, risk_level, publication_policy, score_components, metadata,
                       created_at, updated_at
                from semantic_candidate_groups
                """ + where_sql + " order by updated_at desc, id desc limit :limit offset :offset"
            ),
            params,
        ).mappings().all()
    return {"items": [dict(row) for row in rows], "total": int(total or 0)}


VALID_CANDIDATE_STATUSES = {
    "PENDING",
    "BLOCKED_BY_DEPENDENCY",
    "CONFLICT",
    "ADOPTED",
    "REJECTED",
    "INVALIDATED",
    "ARCHIVED",
    "LOW_EVIDENCE",
    "DUPLICATE",
    "PUBLISHED",
}


def _normalize_candidate_status(status: str) -> str:
    normalized = str(status or "").strip().upper()
    if normalized not in VALID_CANDIDATE_STATUSES:
        raise ValueError("invalid candidate status")
    return normalized


def update_semantic_candidate_status(candidate_id: int, *, status: str, reviewed_by: str | None = None, review_note: str | None = None, object_value: str | None = None) -> dict[str, Any] | None:
    normalized = _normalize_candidate_status(status)
    with ai_session_scope() as db:
        apply_semantic_candidate_schema(db)
        row = db.execute(
            text(
                """
                update semantic_claim_candidates
                set status = :status,
                    object_value = coalesce(:object_value, object_value),
                    reviewed_by = :reviewed_by,
                    review_note = :review_note,
                    reviewed_at = now(),
                    updated_at = now()
                where id = :id
                returning id, candidate_uid, trace_id, source_scenic_id, source_node_id,
                          claim_type, candidate_type, predicate, object_value, object_name,
                          source_url, quote, status, candidate_group_key, value_group_key,
                          conflict_class, gap_status, conflict_group, metadata, updated_at
                """
            ),
            {"id": int(candidate_id), "status": normalized, "object_value": object_value, "reviewed_by": reviewed_by, "review_note": review_note},
        ).mappings().first()
        return dict(row) if row else None


def update_semantic_candidate_status_batch(
    candidate_ids: list[int],
    *,
    status: str,
    reviewed_by: str | None = None,
    review_note: str | None = None,
) -> dict[str, Any]:
    normalized = _normalize_candidate_status(status)
    try:
        normalized_ids = list(dict.fromkeys(int(candidate_id) for candidate_id in candidate_ids))
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate_ids must contain integers") from exc
    if not normalized_ids or any(candidate_id <= 0 for candidate_id in normalized_ids):
        raise ValueError("candidate_ids must contain positive integers")
    if len(normalized_ids) > 1000:
        raise ValueError("candidate_ids exceeds limit 1000")

    with ai_session_scope() as db:
        apply_semantic_candidate_schema(db)
        rows = db.execute(
            text(
                """
                update semantic_claim_candidates
                set status = :status,
                    reviewed_by = :reviewed_by,
                    review_note = :review_note,
                    reviewed_at = now(),
                    updated_at = now()
                where id = any(:candidate_ids)
                returning id, candidate_uid, trace_id, source_scenic_id, source_node_id,
                          claim_type, candidate_type, predicate, object_value, object_name,
                          source_url, quote, status, candidate_group_key, value_group_key,
                          conflict_class, gap_status, conflict_group, metadata, updated_at
                """
            ),
            {
                "candidate_ids": normalized_ids,
                "status": normalized,
                "reviewed_by": reviewed_by,
                "review_note": review_note,
            },
        ).mappings().all()

    items = [dict(row) for row in rows]
    updated_ids = {int(item["id"]) for item in items}
    return {
        "status": normalized,
        "requested_count": len(normalized_ids),
        "updated_count": len(items),
        "candidate_ids": sorted(updated_ids),
        "missing_candidate_ids": [candidate_id for candidate_id in normalized_ids if candidate_id not in updated_ids],
        "items": items,
    }
