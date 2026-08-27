from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.service.value_normalization_service import canonical_predicate, normalize_text_value
from src.rag.service.claim_evidence_fusion_service import fuse_evidence
from src.rag.service.claim_policy_service import (
    RELATION_ALIASES,
    default_property_policy,
    default_relation_policy,
)


# G2 policy defaults are centralized so completion and growth share one
# cardinality/exclusivity contract; domain schema values override at runtime.


def _relation_key(value: Any) -> str:
    raw = str(value or "").strip()
    return canonical_predicate(RELATION_ALIASES.get(raw, raw))


def _property_policy(predicate: str, temporal_role: str | None = None) -> dict[str, str]:
    """Resolve the shared property policy used by KG-delta classification."""
    return default_property_policy(predicate, temporal_role)


def _relation_policy(predicate: str) -> dict[str, str]:
    """Resolve the shared relation policy used by KG-delta classification."""
    return default_relation_policy(predicate)



_BOOLEAN_CLAUSE_CUES = (
    "锚定", "落实", "推动", "打造", "建成", "构建", "营造", "承担", "获得",
    "努力", "矢志", "做出", "服务", "坚持", "开展", "形成", "入选", "存在",
    "达到", "突破", "培养", "建设",
)


def _malformed_claim_reason(item: dict[str, Any]) -> str | None:
    """Reject a complete action clause used as predicate with 是/否."""
    predicate = str(item.get("canonical_predicate") or item.get("raw_predicate") or "").strip()
    value = str(item.get("object_text") or item.get("normalized_value") or "").strip()
    if value not in {"是", "否"}:
        if item.get("claim_type") == "property" and any(cue in predicate for cue in _BOOLEAN_CLAUSE_CUES):
            if value == predicate or (len(value) >= 4 and value in predicate and len(predicate) > len(value) + 1):
                return "CLAUSE_AS_PROPERTY_SELF_VALUE"
            if not re.search(r"[0-9]|年|月|日|亿元|万元|平方米|公里|项|人|号|次|%", value):
                return "ACTION_CLAUSE_AS_PROPERTY"
        return None
    if predicate.startswith(("是否", "有无")):
        return None
    if any(mark in predicate for mark in ("“", "”", '"', "‘", "’")):
        return "MALFORMED_PREDICATE_BOOLEAN_VALUE"
    if any(cue in predicate for cue in _BOOLEAN_CLAUSE_CUES) or len(predicate) > 12:
        return "CLAUSE_AS_PREDICATE_BOOLEAN_VALUE"
    return None


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value or 0.0)))


def _resolution_score(resolution: dict[str, Any] | None) -> float:
    resolution = resolution or {}
    status = str(resolution.get("status") or "").upper()
    base = {"EXACT": 1.0, "ALIAS_MATCH": 0.95, "SEMANTIC_MATCH": 0.82, "AMBIGUOUS": 0.25, "NEW_ENTITY": 0.45}.get(status, 0.35)
    top1 = resolution.get("vector_top1_score")
    margin = resolution.get("vector_margin")
    if status == "SEMANTIC_MATCH" and top1 is not None:
        base = 0.6 * base + 0.3 * _clamp(top1) + 0.1 * _clamp(float(margin or 0.0) / 0.2)
    return _clamp(base)


def _trust_components(item: dict[str, Any], operation: str, *, image_only: bool = False) -> dict[str, Any]:
    subject_score = _resolution_score(item.get("subject_resolution"))
    target_score = _resolution_score(item.get("target_resolution")) if item.get("target_resolution") else subject_score
    entity_resolution = min(subject_score, target_score)
    bindings: list[dict[str, Any]] = []
    for support in item.get("supporting_claims") or []:
        unit = support.get("evidence_unit") or {}
        quote = str(support.get("quote") or "")
        content = str(unit.get("content") or "")
        retrieval_raw = max(
            float(unit.get("retrieval_score") or 0.0),
            float(unit.get("rerank_score") or 0.0),
            float(unit.get("score") or 0.0),
        )
        bindings.append(
            {
                "source_independence_key": support.get("source_independence_key"),
                "source_type": unit.get("source_type"),
                "source_quality": unit.get("source_authority") or 0.5,
                "retrieval_relevance": min(1.0, retrieval_raw / 5.0),
                "quote_coverage": 1.0 if quote and quote in content else 0.45 if quote else 0.0,
                "evidence_locality": 1.0 if unit.get("source_url") or unit.get("source_doc_id") else 0.65,
                "entailment_score": support.get("entailment_score"),
                "extraction_confidence": support.get("confidence"),
                "entity_resolution_confidence": entity_resolution,
                "evidence_text": content,
                "claim_subject": item.get("subject_name") or item.get("subject_text"),
                "claim_predicate": item.get("canonical_predicate") or item.get("predicate"),
                "claim_value": item.get("normalized_value") or item.get("object_text") or item.get("object_name"),
            }
        )
    fusion = fuse_evidence(bindings)
    conflict_risk = 1.0 if operation == "CONFLICT" else (
        0.75 if any(
            str((item.get(key) or {}).get("status") or "").upper() == "AMBIGUOUS"
            for key in ("subject_resolution", "target_resolution")
        ) else 0.55 if operation in {"UPDATE", "DEPRECATE"} else 0.0
    )
    if image_only:
        conflict_risk = max(conflict_risk, 0.35)
    trust = _clamp(
        0.55 * float(fusion["evidence_support_score"])
        + 0.20 * entity_resolution
        + 0.15 * float(fusion["source_quality"])
        + 0.10 * float(fusion["cross_source_support"])
        - 0.10 * conflict_risk
    )
    return {
        "evidence_count": int(fusion["evidence_count"]),
        "independent_source_count": int(fusion["independent_source_count"]),
        "source_group_count": int(fusion["source_group_count"]),
        "extraction_confidence": round(max(
            (float(item.get("confidence") or 0.0) for item in item.get("supporting_claims") or []),
            default=0.0,
        ), 4),
        "evidence_support_score": round(float(fusion["evidence_support_score"]), 4),
        "entity_resolution_confidence": round(entity_resolution, 4),
        "source_quality_score": round(float(fusion["source_quality"]), 4),
        "cross_source_support": round(float(fusion["cross_source_support"]), 4),
        "conflict_risk": round(_clamp(conflict_risk), 4),
        "semantic_trust_score": round(trust, 4),
        "trust_version": fusion["trust_version"],
    }

def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _published_property_values(properties: Any, predicate: str) -> list[str]:
    if not isinstance(properties, dict):
        return []
    values: list[str] = []
    for key, value in properties.items():
        if canonical_predicate(str(key)) != predicate:
            continue
        raw_values = value if isinstance(value, list) else [value]
        values.extend(normalize_text_value(str(item or "")).casefold() for item in raw_values if str(item or "").strip())
    return values


def classify_kg_deltas(
    *, source_scenic_id: str, aggregated_claims: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Classify normalized claims as EXISTS, ADD, or CONFLICT against formal facts.

    Input is the aggregated claim list produced by the growth discovery graph;
    output is a KG-delta list consumed by ``persist_kg_deltas``.
    """
    subject_ids = {
        str(item.get("subject_resolution", {}).get("node_id") or "")
        for item in aggregated_claims
        if item.get("subject_resolution", {}).get("node_id")
    }
    with ai_session_scope() as db:
        node_rows = {
            str(row["source_node_id"]): dict(row)
            for row in db.execute(
                text(
                    """
                    select source_node_id, properties from semantic_nodes
                    where source_scenic_id=:sid and source_node_id=any(:node_ids)
                    """
                ),
                {"sid": str(source_scenic_id), "node_ids": list(subject_ids) or ["__none__"]},
            ).mappings().all()
        }
        edge_rows = [
            dict(row)
            for row in db.execute(
                text(
                    """
                    select source_node_id, target_node_id, relation_type, relation_label
                    from semantic_edges where source_scenic_id=:sid and source_node_id=any(:node_ids)
                    """
                ),
                {"sid": str(source_scenic_id), "node_ids": list(subject_ids) or ["__none__"]},
            ).mappings().all()
        ]
    result: list[dict[str, Any]] = []
    for item in aggregated_claims:
        subject = item["subject_resolution"]
        target = item.get("target_resolution") or {}
        predicate = str(item["canonical_predicate"] or "")
        quote = str(item.get("quote") or "")
        operation = "ADD"
        reason = "new_fact_for_existing_entity"
        if subject.get("status") == "AMBIGUOUS" or target.get("status") == "AMBIGUOUS":
            operation, reason = "CONFLICT", "entity_resolution_ambiguous"
        elif subject.get("status") == "NEW_ENTITY" or target.get("status") == "NEW_ENTITY":
            operation, reason = "MINT_ADD", "new_entity_required"
        elif item["claim_type"] == "property":
            source_node_id = str(subject.get("node_id") or "")
            published = _published_property_values((node_rows.get(source_node_id) or {}).get("properties"), predicate)
            candidate_value = str(item.get("normalized_value") or "")
            if candidate_value and candidate_value in published:
                operation, reason = "EXISTS", "published_property_same_value"
            elif not published:
                operation, reason = "ADD", "new_property_for_existing_entity"
            elif DEPRECATE_CUES.search(quote):
                operation, reason = "DEPRECATE", "evidence_explicitly_deprecates_existing_fact"
            elif UPDATE_CUES.search(quote):
                operation, reason = "UPDATE", "evidence_explicitly_replaces_existing_value"
            elif _property_policy(predicate, item.get("temporal_role"))["conflict_policy"] != "exclusive":
                operation, reason = "ADD", "multi_value_property_can_coexist"
            else:
                operation, reason = "CONFLICT", "exclusive_property_has_different_value"
        else:
            source_node_id = str(subject.get("node_id") or "")
            target_node_id = str(target.get("node_id") or "")
            same_predicate = [
                row for row in edge_rows
                if str(row.get("source_node_id") or "") == source_node_id
                and _relation_key(row.get("relation_label") or row.get("relation_type")) == _relation_key(predicate)
            ]
            same_target = bool(target_node_id and any(
                str(row.get("target_node_id") or "") == target_node_id
                for row in same_predicate
            ))
            if same_target:
                # Exact fact identity always wins. Do not let the
                # cardinality branch below overwrite EXISTS with ADD.
                operation, reason = "EXISTS", "published_relation_same_target"
            else:
                relation_policy = _relation_policy(predicate)
                if not same_predicate:
                    operation, reason = "ADD", "new_relation_for_existing_entity"
                elif relation_policy["conflict_policy"] != "exclusive":
                    operation, reason = "ADD", "coexistable_relation_target"
                elif UPDATE_CUES.search(quote):
                    operation, reason = "UPDATE", "evidence_explicitly_replaces_relation_target"
                else:
                    operation, reason = "CONFLICT", "exclusive_relation_has_different_target"
        delta_policy = _property_policy(predicate, item.get("temporal_role")) if item["claim_type"] == "property" else _relation_policy(predicate)
        result.append({**item, "update_operation": operation, "delta_reason": reason, "delta_policy": delta_policy})
    return result


def _bind_existing_fact(
    db: Any, *, growth_run_id: str, source_scenic_id: str, item: dict[str, Any], trust_components: dict[str, float] | None = None
) -> None:
    subject = item["subject_resolution"]
    target = item.get("target_resolution") or {}
    for support in item["supporting_claims"]:
        unit = support["evidence_unit"]
        binding_uid = _hash(
            [
                source_scenic_id,
                item["claim_type"],
                str(subject.get("node_id") or ""),
                item["canonical_predicate"],
                item.get("normalized_value") if item["claim_type"] == "property" else "",
                str(target.get("node_id") or ""),
                str(item.get("temporal_role") or ""),
                int(unit["id"]),
            ]
        )
        db.execute(
            text(
                """
                insert into semantic_growth_fact_evidence_bindings (
                    binding_uid, growth_run_id, source_scenic_id, fact_kind, source_node_id,
                    predicate, normalized_value, target_node_id, temporal_role,
                    evidence_unit_id, raw_claim_id, source_independence_key,
                    evidence_score, metadata
                ) values (
                    :binding_uid, :growth_run_id, :sid, :fact_kind, :source_node_id,
                    :predicate, :normalized_value, :target_node_id, :temporal_role,
                    :evidence_unit_id, :raw_claim_id, :source_key,
                    :evidence_score, cast(:metadata as jsonb)
                ) on conflict do nothing
                """
            ),
            {
                "binding_uid": binding_uid,
                "growth_run_id": str(growth_run_id),
                "sid": str(source_scenic_id),
                "fact_kind": item["claim_type"],
                "source_node_id": str(subject.get("node_id") or ""),
                "predicate": item["canonical_predicate"],
                "normalized_value": item.get("normalized_value") if item["claim_type"] == "property" else "",
                "target_node_id": str(target.get("node_id") or "") or None,
                "temporal_role": str(item.get("temporal_role") or "") or None,
                "evidence_unit_id": int(unit["id"]),
                "raw_claim_id": int(support["id"]),
                "source_key": support["source_independence_key"],
                "evidence_score": float(support.get("confidence") or 0.0) * float(unit.get("source_authority") or 0.5),
                "metadata": json.dumps({"delta_reason": item["delta_reason"], "trust_components": trust_components or {}}, ensure_ascii=False),
            },
        )


def persist_kg_deltas(
    *, growth_run_id: str, source_scenic_id: str, classified_claims: list[dict[str, Any]]
) -> dict[str, Any]:
    """Persist G2 delta results and bind EXISTS claims to formal facts.

    Input is the output of ``classify_kg_deltas``; output contains operation
    counts, candidate IDs, trust/risk summaries, and evidence bindings.
    """
    candidate_ids: list[int] = []
    exists_count = 0
    operation_counts: dict[str, int] = {}
    low_evidence_count = 0
    trust_scores: list[float] = []
    trust_risk_counts: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    with ai_session_scope() as db:
        for item in classified_claims:
            operation = item["update_operation"]
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
            subject = item["subject_resolution"]
            target = item.get("target_resolution") or {}
            unit = item["evidence_unit"]
            image_only = all(
                str(support["evidence_unit"].get("source_type") or "") == "image"
                for support in item["supporting_claims"]
            )
            trust_components = _trust_components(item, operation, image_only=image_only)
            trust_scores.append(float(trust_components["semantic_trust_score"]))
            risk_bucket = "HIGH" if image_only or operation == "CONFLICT" or operation == "MINT_ADD" or trust_components["semantic_trust_score"] < 0.55 else ("MEDIUM" if trust_components["semantic_trust_score"] < 0.75 else "LOW")
            trust_risk_counts[risk_bucket] = trust_risk_counts.get(risk_bucket, 0) + 1
            if operation == "EXISTS":
                _bind_existing_fact(
                    db,
                    growth_run_id=growth_run_id,
                    source_scenic_id=source_scenic_id,
                    item=item,
                    trust_components=trust_components,
                )
                exists_count += 1
                db.execute(
                    text("update semantic_growth_raw_claims set status='BOUND_TO_EXISTING', updated_at=now() where id=any(:ids)"),
                    {"ids": item["raw_claim_ids"]},
                )
                continue
            source_node_id = str(subject.get("node_id") or f"__new__:{subject.get('node_candidate_id')}")
            source_identity = str(
                unit.get("source_url")
                or unit.get("source_title")
                or unit.get("source_id")
                or ""
            ).strip().casefold()
            stable_value = str(
                item.get("normalized_value")
                or item.get("object_text")
                or ""
            ).strip().casefold()
            candidate_uid = _hash([
                source_scenic_id,
                source_node_id,
                item["claim_type"],
                item["canonical_predicate"],
                stable_value,
                str(target.get("node_id") or ""),
                str(target.get("node_candidate_id") or ""),
                source_identity,
            ])
            candidate_group_key = _hash([source_scenic_id, source_node_id, item["claim_type"], item["canonical_predicate"]])[:32]
            value_group_key = item["aggregation_key"][:32]
            malformed_reason = _malformed_claim_reason(item)
            metadata = {
                "growth_run_id": growth_run_id,
                "discovery_track": "OPEN_DISCOVERY",
                "update_operation": operation,
                "claim_quality": "INVALIDATED" if malformed_reason else "VALID",
                "invalid_reason": malformed_reason,
                "delta_reason": item["delta_reason"],
                "delta_policy": item.get("delta_policy") or {},
                "aggregation_key": item["aggregation_key"],
                "raw_claim_ids": item["raw_claim_ids"],
                "evidence_unit_ids": item["evidence_unit_ids"],
                "independent_source_count": item["independent_source_count"],
                "raw_predicates": list(dict.fromkeys(str(x["raw_predicate"]) for x in item["supporting_claims"])),
                "predicate_resolution_status": item["predicate_resolution_status"],
                "subject_resolution": subject,
                "target_resolution": target,
                "subject_node_candidate_id": subject.get("node_candidate_id"),
                "target_node_candidate_id": target.get("node_candidate_id"),
                "trust_components": trust_components,
                "entity_resolution": {
                    "subject": {
                        "status": subject.get("status"),
                        "method": subject.get("resolution_method"),
                        "top1_score": subject.get("vector_top1_score"),
                        "top2_score": subject.get("vector_top2_score"),
                        "margin": subject.get("vector_margin"),
                        "candidates": subject.get("possible_nodes") or subject.get("vector_candidates") or [],
                    },
                    "target": {
                        "status": target.get("status"),
                        "method": target.get("resolution_method"),
                        "top1_score": target.get("vector_top1_score"),
                        "top2_score": target.get("vector_top2_score"),
                        "margin": target.get("vector_margin"),
                        "candidates": target.get("possible_nodes") or target.get("vector_candidates") or [],
                    },
                },
            }
            if image_only:
                low_evidence_count += 1
            # UPDATE/DEPRECATE are explicit operations, not automatically a
            # contradiction. They remain reviewable PENDING candidates.
            status = "INVALIDATED" if malformed_reason else ("CONFLICT" if operation == "CONFLICT" else "PENDING")
            existing_candidate_id = db.execute(
                text(
                    """
                    select id
                    from semantic_claim_candidates
                    where source_scenic_id=:sid
                      and canonical_claim_key=:canonical_claim_key
                      and upper(coalesce(status, 'PENDING')) not in ('ADOPTED','PUBLISHED','REJECTED','INVALIDATED')
                    order by id
                    limit 1
                    """
                ),
                {
                    "sid": str(source_scenic_id),
                    "canonical_claim_key": item.get("canonical_claim_key") or item["aggregation_key"],
                },
            ).scalar()
            if existing_candidate_id:
                db.execute(
                    text(
                        """
                        update semantic_claim_candidates
                        set confidence=greatest(confidence, :confidence),
                            evidence_score=greatest(evidence_score, :evidence_score),
                            metadata=coalesce(metadata, '{}'::jsonb) || cast(:metadata as jsonb),
                            canonical_claim_key=:canonical_claim_key,
                            conflict_scope_key=:conflict_scope_key,
                            trust_version=:trust_version,
                            trust_components=coalesce(trust_components, '{}'::jsonb) || cast(:trust_components as jsonb),
                            final_trust_score=greatest(coalesce(final_trust_score, 0), :final_trust_score),
                            independent_source_count=greatest(independent_source_count, :independent_source_count),
                            updated_at=now()
                        where id=:id
                        """
                    ),
                    {
                        "id": int(existing_candidate_id),
                        "confidence": float(item.get("confidence") or 0.0),
                        "evidence_score": float(item.get("confidence") or 0.0) * float(unit.get("source_authority") or 0.5),
                        "metadata": json.dumps(metadata, ensure_ascii=False),
                        "canonical_claim_key": item.get("canonical_claim_key") or item["aggregation_key"],
                        "conflict_scope_key": item.get("conflict_scope_key") or candidate_group_key,
                        "trust_version": "trust-v1",
                        "trust_components": json.dumps(trust_components, ensure_ascii=False),
                        "final_trust_score": float(trust_components["semantic_trust_score"]),
                        "independent_source_count": int(item["independent_source_count"]),
                    },
                )
                row = {"id": int(existing_candidate_id)}
            else:
                row = db.execute(
                    text(
                        """
                        insert into semantic_claim_candidates (
                            candidate_uid, trace_id, run_id, source_scenic_id, source_node_id,
                            subject_name, subject_type, graph_scope, retrieval_source,
                            claim_id, claim_type, candidate_type, predicate, object_value,
                            object_name, object_type, target_source_node_id, source_id,
                            source_title, source_url, quote, confidence, evidence_score,
                            evidence_status, status, evidence_ids, recommend_score, support_status,
                            candidate_group_key, value_group_key, conflict_class, gap_status,
                            provenance_type, retrieval_method, target_node_id,
                            target_node_candidate_id, entity_resolution_status, possible_nodes,
                            raw_type, suggested_type, type_confidence, risk_level,
                            publication_policy, score_components, conflict_key, conflict_group,
                            raw_payload, metadata, update_operation, aggregation_key,
                            independent_source_count, canonical_claim_key, conflict_scope_key,
                            trust_version, trust_components, final_trust_score, updated_at
                        ) values (
                            :candidate_uid, :trace_id, :run_id, :sid, :source_node_id,
                            :subject_name, :subject_type, 'evidence_unit', 'provided_evidence',
                            :claim_id, :claim_type, :candidate_type, :predicate, :object_value,
                            :object_name, :object_type, :target_source_node_id, :source_id,
                            :source_title, :source_url, :quote, :confidence, :evidence_score,
                            :evidence_status, :status, '[]'::jsonb, :recommend_score, :support_status,
                            :candidate_group_key, :value_group_key, :conflict_class, 'pending_review',
                            'growth_evidence_unit', 'open_discovery', :target_node_id,
                            :target_node_candidate_id, :entity_resolution_status, cast(:possible_nodes as jsonb),
                            :raw_type, :suggested_type, :type_confidence, :risk_level,
                            'MANUAL_REVIEW', cast(:score_components as jsonb), :conflict_key, :conflict_group,
                            cast(:raw_payload as jsonb), cast(:metadata as jsonb), :update_operation,
                            :aggregation_key, :independent_source_count, :canonical_claim_key,
                            :conflict_scope_key, :trust_version, cast(:trust_components as jsonb),
                            :final_trust_score, now()
                        ) on conflict (candidate_uid) do update set
                            confidence=greatest(semantic_claim_candidates.confidence, excluded.confidence),
                            evidence_score=greatest(semantic_claim_candidates.evidence_score, excluded.evidence_score),
                            metadata=semantic_claim_candidates.metadata || excluded.metadata,
                            update_operation=excluded.update_operation,
                            aggregation_key=excluded.aggregation_key,
                            canonical_claim_key=excluded.canonical_claim_key,
                            conflict_scope_key=excluded.conflict_scope_key,
                            trust_version=excluded.trust_version,
                            trust_components=semantic_claim_candidates.trust_components || excluded.trust_components,
                            final_trust_score=greatest(
                                coalesce(semantic_claim_candidates.final_trust_score, 0),
                                coalesce(excluded.final_trust_score, 0)
                            ),
                            independent_source_count=greatest(semantic_claim_candidates.independent_source_count, excluded.independent_source_count),
                            status=case when semantic_claim_candidates.status in ('ADOPTED','PUBLISHED','REJECTED','INVALIDATED')
                                        then semantic_claim_candidates.status else excluded.status end,
                            updated_at=now()
                        returning id
                        """
                    ),
                    {
                        "candidate_uid": candidate_uid,
                        "trace_id": f"{growth_run_id}:open:{item['aggregation_key'][:12]}",
                        "run_id": growth_run_id,
                        "sid": str(source_scenic_id),
                        "source_node_id": source_node_id,
                        "subject_name": item["subject_text"],
                        "subject_type": item.get("subject_type") or subject.get("node_type") or None,
                        "claim_id": f"open-{item['aggregation_key'][:16]}",
                        "claim_type": item["claim_type"],
                        "candidate_type": "discovered_entity" if operation == "MINT_ADD" else "discovered_fact",
                        "predicate": item["canonical_predicate"],
                        "object_value": item["object_text"] if item["claim_type"] == "property" else None,
                        "object_name": item["object_text"] if item["claim_type"] == "relation" else None,
                        "object_type": item.get("object_type") or target.get("node_type") or None,
                        "target_source_node_id": target.get("node_id"),
                        "source_id": unit["evidence_unit_uid"],
                        "source_title": unit.get("source_title"),
                        "source_url": unit.get("source_url"),
                        "quote": item["quote"],
                        "confidence": float(item.get("confidence") or 0.0),
                        "evidence_score": float(item.get("confidence") or 0.0) * float(unit.get("source_authority") or 0.5),
                        "evidence_status": "LOW_EVIDENCE" if image_only else "SUPPORTED",
                        "support_status": "weak" if image_only else "supported",
                        "status": status,
                        "recommend_score": trust_components["semantic_trust_score"],
                        "candidate_group_key": candidate_group_key,
                        "value_group_key": value_group_key,
                        "conflict_class": "conflicting" if status == "CONFLICT" else None,
                        "target_node_id": target.get("node_id"),
                        "target_node_candidate_id": target.get("node_candidate_id"),
                        "entity_resolution_status": target.get("status") if item["claim_type"] == "relation" else subject.get("status"),
                        "possible_nodes": json.dumps((target or subject).get("possible_nodes") or [], ensure_ascii=False),
                        "raw_type": item.get("object_type") or None,
                        "suggested_type": target.get("node_type") or item.get("object_type") or None,
                        "type_confidence": float(item.get("confidence") or 0.0),
                        "risk_level": "HIGH" if image_only or status == "CONFLICT" or operation == "MINT_ADD" or trust_components["semantic_trust_score"] < 0.55 else ("MEDIUM" if trust_components["semantic_trust_score"] < 0.75 else "LOW"),
                        "score_components": json.dumps(
                            {
                                **trust_components,
                                "source_count": float(item["independent_source_count"]),
                            },
                            ensure_ascii=False,
                        ),
                        "conflict_key": candidate_group_key,
                        "conflict_group": candidate_group_key if status == "CONFLICT" else None,
                        "raw_payload": json.dumps(
                            {
                                "raw_claim_ids": item["raw_claim_ids"],
                                "raw_predicates": metadata["raw_predicates"],
                                "temporal_role": item.get("temporal_role"),
                            },
                            ensure_ascii=False,
                        ),
                        "metadata": json.dumps(metadata, ensure_ascii=False),
                        "update_operation": operation,
                        "aggregation_key": item["aggregation_key"],
                        "canonical_claim_key": item.get("canonical_claim_key") or item["aggregation_key"],
                        "conflict_scope_key": item.get("conflict_scope_key") or candidate_group_key,
                        "trust_version": "trust-v1",
                        "trust_components": json.dumps(trust_components, ensure_ascii=False),
                        "final_trust_score": float(trust_components["semantic_trust_score"]),
                        "independent_source_count": int(item["independent_source_count"]),
                    },
                ).mappings().one()
            candidate_id = int(row["id"])
            candidate_ids.append(candidate_id)
            for support in item["supporting_claims"]:
                support_unit = support["evidence_unit"]
                db.execute(
                    text(
                        """
                        insert into semantic_growth_candidate_evidence_bindings (
                            growth_run_id, candidate_id, evidence_unit_id, raw_claim_id,
                            source_independence_key, support_role, evidence_score, metadata
                        ) values (
                            :growth_run_id, :candidate_id, :unit_id, :raw_claim_id,
                            :source_key, 'SUPPORTS', :score, '{}'::jsonb
                        ) on conflict do nothing
                        """
                    ),
                    {
                        "growth_run_id": growth_run_id,
                        "candidate_id": candidate_id,
                        "unit_id": int(support_unit["id"]),
                        "raw_claim_id": int(support["id"]),
                        "source_key": support["source_independence_key"],
                        "score": float(support.get("confidence") or 0.0) * float(support_unit.get("source_authority") or 0.5),
                    },
                )
            db.execute(
                text(
                    """
                    update semantic_claim_candidates c
                    set independent_source_count=s.source_count,
                        evidence_score=greatest(c.evidence_score, s.max_score),
                        metadata=coalesce(c.metadata, '{}'::jsonb) ||
                            jsonb_build_object(
                                'evidence_unit_ids', s.evidence_unit_ids,
                                'independent_source_count', s.source_count
                            ),
                        updated_at=now()
                    from (
                        select candidate_id,
                               count(distinct source_independence_key)::integer as source_count,
                               max(evidence_score) as max_score,
                               jsonb_agg(distinct evidence_unit_id) as evidence_unit_ids
                        from semantic_growth_candidate_evidence_bindings
                        where candidate_id=:candidate_id
                        group by candidate_id
                    ) s
                    where c.id=s.candidate_id
                    """
                ),
                {"candidate_id": candidate_id},
            )
            db.execute(
                text("update semantic_growth_raw_claims set status='CANDIDATE_PERSISTED', updated_at=now() where id=any(:ids)"),
                {"ids": item["raw_claim_ids"]},
            )
    return {
        "candidate_ids": list(dict.fromkeys(candidate_ids)),
        "exists_count": exists_count,
        "operation_counts": operation_counts,
        "low_evidence_count": low_evidence_count,
        "average_trust_score": round(sum(trust_scores) / len(trust_scores), 4) if trust_scores else 0.0,
        "trust_score_sum": round(sum(trust_scores), 4),
        "trust_scored_count": len(trust_scores),
        "trust_risk_counts": trust_risk_counts,
        "classified_count": len(classified_claims),
    }
