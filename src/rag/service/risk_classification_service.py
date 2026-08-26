"""Explainable recommendation scoring and publication risk policy."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.rag.schemas import CandidateClaim, SemanticCompleteRequest


HIGH_CONFLICT_CLASSES = {"conflicting", "scope_mismatch", "entity_ambiguity"}
LOW_IMPACT_PROPERTY_HINTS = ("简介", "描述", "特色", "特征", "功能", "用途", "说明", "备注")
CORE_FACT_HINTS = ("时间", "年代", "时期", "位置", "地址", "坐标", "面积", "高度", "级别", "归属")


def _source_key(claim: CandidateClaim) -> str:
    evidence = ",".join(str(item) for item in (claim.evidence_ids or []))
    return str(claim.source_url or evidence or claim.source_id or "").strip()


def _schema_values(payload: SemanticCompleteRequest, kind: str) -> set[str]:
    schema = (payload.metadata or {}).get("domain_schema") or {}
    values: set[str] = set()
    if kind == "property":
        for key in (schema.get("properties") or {}):
            values.add(str(key))
        for fields in (schema.get("schema_map") or {}).values():
            values.update(str(item) for item in (fields or []))
    else:
        relations = schema.get("relations") or {}
        values.update(str(key) for key in relations)
        for items in (schema.get("relation_intents") or {}).values():
            for item in items or []:
                if isinstance(item, dict):
                    values.update(str(item.get(key) or "") for key in ("label", "code"))
                else:
                    values.add(str(item))
    return {value.strip() for value in values if value.strip()}


def _schema_node_types(payload: SemanticCompleteRequest) -> set[str]:
    schema = (payload.metadata or {}).get("domain_schema") or {}
    values = schema.get("node_types") or schema.get("type_labels") or {}
    if isinstance(values, dict):
        values = values.keys()
    return {str(value).strip() for value in (values or []) if str(value).strip()}


def apply_recommendation_and_risk(
    payload: SemanticCompleteRequest,
    claims: list[CandidateClaim],
    candidate_groups: list[dict[str, Any]],
) -> dict[str, int]:
    claims_by_group: dict[str, list[CandidateClaim]] = defaultdict(list)
    for claim in claims:
        if claim.candidate_group_key:
            claims_by_group[str(claim.candidate_group_key)].append(claim)
    records = {str(item.get("candidate_group_key") or ""): item for item in candidate_groups}
    property_schema = _schema_values(payload, "property")
    relation_schema = _schema_values(payload, "relation")
    node_types = _schema_node_types(payload)
    counts = defaultdict(int)

    for group_key, group_claims in claims_by_group.items():
        record = records.get(group_key, {})
        conflict_class = str(record.get("conflict_class") or group_claims[0].conflict_class or "unsupported")
        source_count = len({_source_key(claim) for claim in group_claims if _source_key(claim)})
        multi_source_support = min(1.0, source_count / 3.0)
        for claim in group_claims:
            meta = dict(claim.metadata or {})
            source_authority = float(meta.get("source_authority_score") or 0.0)
            retrieval_relevance = float(meta.get("retrieval_relevance") or 0.0)
            evidence_support = float(claim.evidence_score or 0.0)
            model_confidence = max(0.0, min(float(claim.confidence or 0.0), 1.0))
            graph_consistency = 0.85
            if conflict_class in HIGH_CONFLICT_CLASSES:
                graph_consistency = 0.15
            elif claim.status == "duplicate":
                graph_consistency = 1.0
            elif conflict_class in {"weak_evidence", "unsupported"}:
                graph_consistency = 0.45
            components = {
                "source_authority": source_authority,
                "retrieval_relevance": retrieval_relevance,
                "evidence_support": evidence_support,
                "multi_source_support": multi_source_support,
                "graph_consistency": graph_consistency,
                "model_confidence": model_confidence,
            }
            score = (
                0.25 * source_authority
                + 0.20 * retrieval_relevance
                + 0.20 * evidence_support
                + 0.15 * multi_source_support
                + 0.10 * graph_consistency
                + 0.10 * model_confidence
            )
            penalty = 0.0
            if conflict_class == "weak_evidence":
                penalty += 0.12
            elif conflict_class == "unsupported":
                penalty += 0.35
            elif conflict_class == "entity_ambiguity":
                penalty += 0.25
            elif conflict_class == "conflicting":
                penalty += 0.30
            elif conflict_class == "scope_mismatch":
                penalty += 0.35
            claim.recommend_score = round(max(0.0, min(1.0, score - penalty)), 3)
            claim.score_components = components
            meta.update({"score_components": components, "score_penalty": round(penalty, 3)})
            claim.metadata = meta

        best = max(group_claims, key=lambda item: float(item.recommend_score or 0.0))
        predicate = str(best.metadata.get("canonical_predicate") or best.predicate or "")
        in_schema = predicate in (property_schema if best.claim_type == "property" else relation_schema)
        suggested_type = str(getattr(best, "suggested_type", None) or best.metadata.get("suggested_type") or "").strip()
        new_type = bool(suggested_type and node_types and suggested_type not in node_types)
        discovered = str(best.metadata.get("discovery_scope") or "") == "open" or not in_schema
        core_fact = any(hint in predicate for hint in CORE_FACT_HINTS)
        low_impact = best.claim_type == "property" and any(hint in predicate for hint in LOW_IMPACT_PROPERTY_HINTS)
        supported = all(str(claim.evidence_status or "") == "supported" for claim in group_claims)

        if conflict_class in HIGH_CONFLICT_CLASSES or new_type or (discovered and best.claim_type == "relation"):
            risk_level = "HIGH"
        elif conflict_class == "unsupported" or core_fact and conflict_class != "same_value":
            risk_level = "HIGH"
        elif supported and source_count >= 2 and conflict_class in {"same_value", "compatible"} and in_schema and low_impact:
            risk_level = "LOW"
        else:
            risk_level = "MEDIUM"
        publication_policy = {"LOW": "AUTO_PUBLISH", "MEDIUM": "BATCH_REVIEW", "HIGH": "MANUAL_REVIEW"}[risk_level]
        counts[risk_level.lower()] += 1
        for claim in group_claims:
            claim.risk_level = risk_level
            claim.publication_policy = publication_policy
            claim.metadata.update({"risk_level": risk_level, "publication_policy": publication_policy})
        if record:
            record["risk_level"] = risk_level
            record["publication_policy"] = publication_policy
            record["recommend_score"] = best.recommend_score
            record.setdefault("metadata", {})["score_components"] = best.score_components
    return dict(counts)
