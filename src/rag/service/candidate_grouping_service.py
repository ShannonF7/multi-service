"""Group semantic completion candidates by question, predicate, and value."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from src.rag.schemas import CandidateClaim, ClaimConflict, SemanticCompleteRequest
from src.rag.service.conflict_classification_service import claim_value, classify_candidate_group
from src.rag.service.value_normalization_service import canonical_predicate


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def candidate_group_key(payload: SemanticCompleteRequest, claim: CandidateClaim) -> str:
    predicate = canonical_predicate(claim.predicate or "", temporal_role=claim.temporal_role).strip()
    claim_type = str(claim.claim_type or "")
    source_node_id = str(claim.subject_node_id or payload.node.source_node_id or "")
    if claim_type == "entity":
        return _hash({
            "domain": str(payload.scenic_id or ""),
            "normalized_name": claim.normalized_value or claim.object_name or claim.object_value or "",
            "suggested_type": claim.suggested_type or claim.object_type or "",
        })[:32]
    return _hash({
        "source_scenic_id": str(payload.scenic_id or ""),
        "source_node_id": source_node_id,
        "claim_type": claim_type,
        "predicate": predicate or str(claim.predicate or ""),
        "temporal_role": str(claim.temporal_role or "") if claim_type == "property" else "",
    })[:32]


def value_group_key(payload: SemanticCompleteRequest, claim: CandidateClaim) -> str:
    return _hash({
        "candidate_group_key": candidate_group_key(payload, claim),
        "value": claim_value(claim),
    })[:32]


def assign_candidate_group_keys(
    payload: SemanticCompleteRequest,
    claims: list[CandidateClaim],
) -> None:
    for claim in claims:
        key = candidate_group_key(payload, claim)
        vkey = value_group_key(payload, claim)
        claim.candidate_group_key = key
        claim.value_group_key = vkey
        claim.metadata = dict(claim.metadata or {})
        claim.metadata.update({
            "candidate_group_key": key,
            "value_group_key": vkey,
            "group_value": claim_value(claim),
        })


def annotate_candidate_groups(    payload: SemanticCompleteRequest,
    claims: list[CandidateClaim],
    conflicts: list[ClaimConflict] | None = None,
) -> list[dict[str, Any]]:
    assign_candidate_group_keys(payload, claims)
    groups: dict[str, list[CandidateClaim]] = defaultdict(list)
    for claim in claims:
        groups[str(claim.candidate_group_key)].append(claim)

    conflict_claim_ids = {str(item.claim_id) for item in (conflicts or []) if getattr(item, "claim_id", None)}
    records: list[dict[str, Any]] = []
    for key, group_claims in groups.items():
        info = classify_candidate_group(group_claims, payload)
        if conflict_claim_ids.intersection({str(c.claim_id) for c in group_claims}):
            same_value_group = int(info.get("distinct_value_count") or 0) <= 1 and int(info.get("source_count") or 0) <= 1
            if not same_value_group and str(info.get("conflict_class") or "") not in {"conflicting", "scope_mismatch", "entity_ambiguity"}:
                info["conflict_class"] = "conflicting"
            if not same_value_group:
                info["gap_status"] = "conflicted"
        best_claim_id = info.get("best_claim_id")
        best_claim = next((c for c in group_claims if c.claim_id == best_claim_id), group_claims[0])
        predicate = canonical_predicate(best_claim.predicate or "", temporal_role=best_claim.temporal_role).strip() or best_claim.predicate
        for claim in group_claims:
            claim.conflict_class = str(info.get("conflict_class") or "insufficient")
            claim.gap_status = str(info.get("gap_status") or "needs_review")
            claim.metadata = dict(claim.metadata or {})
            claim.metadata.update({
                "conflict_class": claim.conflict_class,
                "gap_status": claim.gap_status,
                "group_candidate_count": info.get("candidate_count"),
                "group_distinct_value_count": info.get("distinct_value_count"),
                "group_source_count": info.get("source_count"),
            })
        records.append({
            "candidate_group_key": key,
            "question_id": best_claim.question_id,
            "claim_type": best_claim.claim_type,
            "predicate": predicate,
            "temporal_role": best_claim.temporal_role,
            "conflict_class": info.get("conflict_class"),
            "gap_status": info.get("gap_status"),
            "candidate_count": info.get("candidate_count"),
            "distinct_value_count": info.get("distinct_value_count"),
            "source_count": info.get("source_count"),
            "best_claim_id": best_claim_id,
            "recommend_score": info.get("recommend_score"),
            "metadata": {
                "values": info.get("values") or [],
                "sources": info.get("sources") or [],
                "claim_ids": [claim.claim_id for claim in group_claims],
            },
        })
    return records
