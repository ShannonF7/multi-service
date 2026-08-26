"""Unified conflict classification for semantic completion candidates."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import unicodedata
from typing import Any

from src.rag.schemas import CandidateClaim, ClaimConflict, SemanticCompleteRequest
from src.rag.service.value_normalization_service import canonical_predicate

EXCLUSIVE_RELATIONS = {
    "\u4f4d\u4e8e", "\u5c5e\u4e8e", "\u5f52\u5c5e", "\u4e0a\u7ea7\u533a\u57df", "\u6240\u5c5e\u666f\u533a", "\u7236\u7ea7", "\u4e0b\u4f0f\u4e8e", "\u4e0a\u8986\u4e8e"
}

MULTI_VALUE_PROPERTIES = {
    "\u7b80\u4ecb", "\u63cf\u8ff0", "\u63cf\u8ff0\u7b80\u4ecb", "\u6982\u8ff0", "\u4ecb\u7ecd", "\u6458\u8981", "\u7279\u8272", "\u7279\u5f81", "\u6587\u732e\u8bb0\u8f7d",
    "\u5386\u53f2\u6cbf\u9769", "\u5730\u8d28\u6210\u56e0", "\u5730\u5f62\u7279\u5f81", "\u5730\u8c8c\u7279\u5f81", "\u7ec4\u6210", "\u6784\u6210", "\u5ca9\u6027",
    "\u8363\u8a89", "\u4f5c\u54c1", "\u6d3b\u52a8", "\u529f\u80fd", "\u7528\u9014", "\u529f\u80fd\u7528\u9014", "\u5907\u6ce8", "\u8bf4\u660e"
}

MULTI_VALUE_KEYWORDS = ("\u7b80\u4ecb", "\u63cf\u8ff0", "\u6982\u8ff0", "\u4ecb\u7ecd", "\u6458\u8981", "\u7279\u5f81", "\u7279\u8272", "\u8bb0\u8f7d", "\u6cbf\u9769", "\u6210\u56e0", "\u7ec4\u6210", "\u6784\u6210", "\u8bf4\u660e", "\u5907\u6ce8", "\u529f\u80fd", "\u7528\u9014")
TEMPORAL_CONFLICT_ROLES = {"construction_time", "current_status_time", "protection_time"}
COMPATIBLE_TEMPORAL_ROLES = {"renovation_time", "legend_time"}
CONFLICT_CLASSES = {"conflicting", "scope_mismatch", "entity_ambiguity"}


def claim_value(claim: CandidateClaim) -> str:
    if claim.claim_type == "relation":
        return str(claim.normalized_value or claim.display_value or claim.object_name or claim.object_value or "").strip()
    return str(claim.normalized_value or claim.display_value or claim.object_value or claim.object_name or "").strip()


def _comparison_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return "".join(char for char in text if char.isalnum())


def _same_value(left: str, right: str) -> bool:
    left_key = _comparison_key(left)
    right_key = _comparison_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    shorter, longer = sorted((left_key, right_key), key=len)
    if len(shorter) >= 16 and shorter in longer and len(shorter) / len(longer) >= 0.9:
        return True
    return min(len(left_key), len(right_key)) >= 16 and SequenceMatcher(None, left_key, right_key).ratio() >= 0.96


def _source_key(claim: CandidateClaim) -> str:
    evidence = ",".join(str(x) for x in (claim.evidence_ids or []))
    return str(claim.source_url or evidence or claim.source_id or "").strip()


def _schema(payload: SemanticCompleteRequest) -> dict[str, Any]:
    schema = (payload.metadata or {}).get("domain_schema") or {}
    return schema if isinstance(schema, dict) else {}


def _property_config(payload: SemanticCompleteRequest | None, predicate: str) -> dict[str, Any]:
    if not payload:
        return {}
    props = _schema(payload).get("properties") if isinstance(_schema(payload).get("properties"), dict) else {}
    config = props.get(predicate) if isinstance(props, dict) else None
    return config if isinstance(config, dict) else {}


def _relation_config(payload: SemanticCompleteRequest | None, predicate: str) -> dict[str, Any]:
    if not payload:
        return {}
    schema = _schema(payload)
    rels = schema.get("relations") if isinstance(schema.get("relations"), dict) else {}
    if predicate in rels and isinstance(rels[predicate], dict):
        return rels[predicate]
    relation_intents = schema.get("relation_intents") if isinstance(schema.get("relation_intents"), dict) else {}
    for items in relation_intents.values():
        for item in items or []:
            if isinstance(item, dict) and predicate in {str(item.get("label") or ""), str(item.get("code") or "")}:
                return item
    return {}


def is_multi_value_property(predicate: str, payload: SemanticCompleteRequest | None = None) -> bool:
    pred = canonical_predicate(predicate or "").strip()
    config = _property_config(payload, pred)
    if str(config.get("cardinality") or "").lower() == "multi":
        return True
    if str(config.get("conflict_policy") or "").lower() in {"compatible", "multi_value"}:
        return True
    if pred in MULTI_VALUE_PROPERTIES:
        return True
    return any(key in pred for key in MULTI_VALUE_KEYWORDS)


def is_exclusive_relation(predicate: str, payload: SemanticCompleteRequest | None = None) -> bool:
    pred = canonical_predicate(predicate or "").strip()
    config = _relation_config(payload, pred)
    if str(config.get("cardinality") or "").lower() == "single":
        return True
    if str(config.get("conflict_policy") or "").lower() == "exclusive":
        return True
    return pred in EXCLUSIVE_RELATIONS


def _is_temporal_group(claims: list[CandidateClaim]) -> bool:
    return bool(claims and (claims[0].temporal_role or "") and claims[0].claim_type == "property")


def _is_exclusive_group(claims: list[CandidateClaim], payload: SemanticCompleteRequest | None = None) -> bool:
    if not claims:
        return False
    first = claims[0]
    predicate = canonical_predicate(first.predicate or "", temporal_role=first.temporal_role).strip()
    if first.claim_type == "relation":
        return is_exclusive_relation(predicate, payload)
    if first.claim_type == "property":
        if _is_temporal_group(claims):
            return str(first.temporal_role or "") in TEMPORAL_CONFLICT_ROLES
        return not is_multi_value_property(predicate, payload)
    return False


def _base_evidence_class(claims: list[CandidateClaim]) -> str | None:
    if not claims:
        return "missing"
    has_supported = any(str(c.evidence_status or "") == "supported" for c in claims)
    has_weak = any(str(c.evidence_status or "") == "weak" for c in claims)
    if not (has_supported or has_weak):
        return "unsupported"
    if has_weak and not has_supported:
        return "weak_evidence"
    return None


def _distinct_values(claims: list[CandidateClaim]) -> list[str]:
    values: list[str] = []
    for claim in claims:
        value = claim_value(claim)
        if value and not any(_same_value(value, existing) for existing in values):
            values.append(value)
    return values


def classify_candidate_group(claims: list[CandidateClaim], payload: SemanticCompleteRequest | None = None) -> dict[str, Any]:
    if not claims:
        return {"conflict_class": "unsupported", "gap_status": "no_evidence", "candidate_count": 0, "distinct_value_count": 0, "source_count": 0, "best_claim_id": None, "recommend_score": 0.0}
    values = _distinct_values(claims)
    sources = {_source_key(claim) for claim in claims if _source_key(claim)}
    best = max(claims, key=lambda c: float(c.recommend_score or 0.0))
    statuses = {
        str(getattr(claim, "entity_resolution_status", None) or (claim.metadata or {}).get("entity_resolution_status") or "")
        for claim in claims
    }
    scope_mismatch = any(bool((claim.metadata or {}).get("scope_mismatch")) for claim in claims)
    evidence_class = _base_evidence_class(claims)
    if scope_mismatch:
        conflict_class = "scope_mismatch"
        gap_status = "conflicted"
    elif "AMBIGUOUS" in statuses:
        conflict_class = "entity_ambiguity"
        gap_status = "conflicted"
    elif evidence_class == "unsupported":
        conflict_class = "unsupported"
        gap_status = "weak_evidence"
    elif evidence_class == "weak_evidence":
        conflict_class = "weak_evidence"
        gap_status = "pending_review"
    elif len(values) <= 1:
        conflict_class = "same_value"
        gap_status = "pending_review"
    elif _is_temporal_group(claims):
        role = str(claims[0].temporal_role or "")
        if role in COMPATIBLE_TEMPORAL_ROLES:
            conflict_class = "compatible"
            gap_status = "pending_review"
        elif _is_exclusive_group(claims, payload):
            conflict_class = "conflicting"
            gap_status = "conflicted"
        else:
            conflict_class = "compatible"
            gap_status = "pending_review"
    elif _is_exclusive_group(claims, payload):
        conflict_class = "conflicting"
        gap_status = "conflicted"
    elif claims[0].claim_type == "property" and is_multi_value_property(claims[0].predicate or "", payload):
        conflict_class = "multi_value"
        gap_status = "pending_review"
    elif claims[0].claim_type == "relation":
        conflict_class = "multi_value"
        gap_status = "pending_review"
    else:
        conflict_class = "compatible"
        gap_status = "pending_review"
    return {"conflict_class": conflict_class, "gap_status": gap_status, "candidate_count": len(claims), "distinct_value_count": len(values), "source_count": len(sources), "best_claim_id": best.claim_id, "recommend_score": round(float(best.recommend_score or 0.0), 3), "values": sorted(values), "sources": sorted(sources)}


def compare_with_existing_graph(payload: SemanticCompleteRequest, claims: list[CandidateClaim]) -> list[ClaimConflict]:
    conflicts: list[ClaimConflict] = []
    for claim in claims:
        pred = canonical_predicate(claim.predicate or "", temporal_role=claim.temporal_role).strip()
        new_value = claim_value(claim)
        if not pred or not new_value:
            continue
        if claim.claim_type == "property":
            for existing in payload.existing_properties:
                existing_key = canonical_predicate(existing.key or "", temporal_role=claim.temporal_role).strip()
                if existing_key != pred:
                    continue
                old = str(existing.value or "").strip()
                if not old or _same_value(old, new_value):
                    if _same_value(old, new_value):
                        claim.status = "duplicate"
                    continue
                if is_multi_value_property(pred, payload) and not claim.temporal_role:
                    continue
                conflict_type = "conflicting"
                claim.status = "conflict_candidate"
                conflicts.append(ClaimConflict(conflict_type=conflict_type, claim_id=claim.claim_id, predicate=pred, existing_value=old, candidate_value=new_value, reason="candidate differs from existing property"))
        elif claim.claim_type == "relation":
            for existing in payload.existing_relations:
                existing_type = canonical_predicate(existing.relation_type or "").strip()
                if existing_type != pred:
                    continue
                old = str(existing.target_name or "").strip()
                if old and _same_value(old, new_value):
                    claim.status = "duplicate"
                elif old and new_value and is_exclusive_relation(pred, payload):
                    claim.status = "conflict_candidate"
                    conflicts.append(ClaimConflict(conflict_type="conflicting", claim_id=claim.claim_id, predicate=pred, existing_target=old, candidate_target=new_value, reason="candidate differs from existing exclusive relation"))
    return conflicts


def compare_candidate_groups(payload: SemanticCompleteRequest, claims: list[CandidateClaim]) -> list[ClaimConflict]:
    grouped: dict[tuple[str, str, str, str], list[CandidateClaim]] = defaultdict(list)
    for claim in claims:
        pred = canonical_predicate(claim.predicate or "", temporal_role=claim.temporal_role).strip()
        key = (str(claim.subject_node_id or payload.node.source_node_id or ""), str(claim.claim_type or ""), pred, str(claim.temporal_role or ""))
        grouped[key].append(claim)
    conflicts: list[ClaimConflict] = []
    for (_, claim_type, pred, _), group in grouped.items():
        info = classify_candidate_group(group, payload)
        ctype = str(info.get("conflict_class") or "")
        if ctype not in CONFLICT_CLASSES:
            continue
        for claim in group:
            claim.status = "conflict_candidate"
            conflicts.append(ClaimConflict(conflict_type=ctype, claim_id=claim.claim_id, predicate=pred, candidate_value=claim_value(claim) if claim_type != "relation" else None, candidate_target=claim_value(claim) if claim_type == "relation" else None, reason="conflicting values in candidate group"))
    return conflicts


def classify_conflicts(payload: SemanticCompleteRequest, claims: list[CandidateClaim]) -> list[ClaimConflict]:
    conflicts = compare_with_existing_graph(payload, claims)
    existing_ids = {str(item.claim_id) for item in conflicts if item.claim_id}
    for item in compare_candidate_groups(payload, claims):
        if str(item.claim_id) not in existing_ids:
            conflicts.append(item)
            existing_ids.add(str(item.claim_id))
    return conflicts
