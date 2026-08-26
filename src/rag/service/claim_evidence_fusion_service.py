"""Explainable evidence support fusion shared by completion and growth."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


TRUST_VERSION = "trust-v1"
DEFAULT_WEIGHTS = {
    "source_quality": 0.20,
    "retrieval_relevance": 0.10,
    "quote_coverage": 0.15,
    "evidence_locality": 0.10,
    "entailment_score": 0.25,
    "extraction_confidence": 0.10,
    "entity_resolution_confidence": 0.10,
}


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def evidence_support_score(
    evidence: dict[str, Any], weights: dict[str, float] | None = None
) -> float:
    weights = weights or DEFAULT_WEIGHTS
    numerator = 0.0
    denominator = 0.0
    for name, weight in weights.items():
        if evidence.get(name) is None:
            continue
        numerator += _clamp(evidence.get(name)) * float(weight)
        denominator += float(weight)
    return round(_clamp(numerator / denominator if denominator else 0.0), 4)


def fuse_evidence(
    bindings: Iterable[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Collapse chunks within a source group, then fuse independent groups.

    The first version intentionally uses the strongest support per source
    group. This prevents chunk count from acting as a proxy for confidence.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    enriched: list[dict[str, Any]] = []
    for binding in bindings:
        item = dict(binding or {})
        score = evidence_support_score(item, weights)
        item["evidence_support_score"] = score
        source_key = str(item.get("source_independence_key") or item.get("source_key") or "unknown")
        groups[source_key].append(score)
        enriched.append(item)
    source_group_scores = {key: max(scores) for key, scores in groups.items()}
    scores = list(source_group_scores.values())
    source_types = {
        str(item.get("source_type") or "unknown")
        for item in enriched
        if item.get("source_type")
    }
    source_count = len(source_group_scores)
    source_quality = max(
        (_clamp(item.get("source_quality")) for item in enriched),
        default=0.0,
    )
    fused = sum(scores) / len(scores) if scores else 0.0
    cross_source = min(1.0, source_count / 3.0)
    if len(source_types) > 1:
        cross_source = min(1.0, cross_source + 0.1)
    return {
        "evidence_count": len(enriched),
        "independent_source_count": source_count,
        "source_group_count": source_count,
        "source_group_scores": source_group_scores,
        "source_quality": round(source_quality, 4),
        "evidence_support_score": round(_clamp(fused), 4),
        "cross_source_support": round(_clamp(cross_source), 4),
        "trust_version": TRUST_VERSION,
    }
