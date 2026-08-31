"""补全与自增长共用的可解释证据支持度融合服务。

本模块只负责把同一 CanonicalClaim 的证据绑定按来源族去重并融合，输入来自
补全链或自增长链，输出供候选审核、发布审计和 GrowthRun 统计使用；不负责抽取、
候选状态流转或正式事实发布。
"""

from __future__ import annotations

from collections import defaultdict
import re
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


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def lexical_entailment_baseline(
    evidence_text: Any, *, subject: Any = None, predicate: Any = None, value: Any = None
) -> float | None:
    """计算词面蕴含基线分数。

    输入：证据原文，以及可选的主体、谓词、值；输出：0~1 的可解释支持度，
    空证据返回 None。调用方是 fuse_evidence，该函数不是 NLI 模型，
    只用于在原文显式出现声明内容时提高分数，不能替代后续语义校验。
    """
    text = _compact_text(evidence_text)
    if not text:
        return None
    subject_match = bool(_compact_text(subject) and _compact_text(subject) in text)
    value_match = bool(_compact_text(value) and _compact_text(value) in text)
    predicate_match = bool(_compact_text(predicate) and _compact_text(predicate) in text)
    if value_match and subject_match:
        return 0.9 if predicate_match else 0.8
    if value_match:
        return 0.7
    if subject_match and predicate_match:
        return 0.45
    return 0.15


def evidence_support_score(
    evidence: dict[str, Any], weights: dict[str, float] | None = None
) -> float:
    """按可解释分量计算单条证据支持度。

    输入：证据分量字典和可选权重；输出：0~1 的支持度。缺失分量不参与分母，
    因此不会因字段缺失而隐式降低分数；调用方为 fuse_evidence。
    """
    weights = weights or DEFAULT_WEIGHTS
    numerator = 0.0
    denominator = 0.0
    for name, weight in weights.items():
        if evidence.get(name) is None:
            continue
        numerator += _clamp(evidence.get(name)) * float(weight)
        denominator += float(weight)
    return round(_clamp(numerator / denominator if denominator else 0.0), 4)


def semantic_trust_score(
    fusion: dict[str, Any],
    *,
    entity_resolution_confidence: Any = 0.0,
    conflict_risk: Any = 0.0,
) -> float:
    """根据融合结果计算统一的事实可信度。

    输入：fuse_evidence 返回的融合结果，以及实体消歧置信度、冲突风险；
    输出：0~1 的派生可信度。该函数只计算分数，不写数据库，也不改变候选状态。
    """
    entity_score = _clamp(entity_resolution_confidence)
    risk = _clamp(conflict_risk)
    score = (
        0.55 * _clamp(fusion.get("evidence_support_score"))
        + 0.20 * entity_score
        + 0.15 * _clamp(fusion.get("source_quality"))
        + 0.10 * _clamp(fusion.get("cross_source_support"))
        - 0.10 * risk
    )
    return round(_clamp(score), 4)


def fuse_evidence(
    bindings: Iterable[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """按来源族去重后融合证据支持度。

    输入：同一 CanonicalClaim 的证据绑定；输出：来源族数量、支持度和可信度分量。
    同一来源族只取最高支持度；不同独立来源使用 noisy-OR 累积，避免重复
    chunk 虚增，同时让真正独立的来源提升事实支持。
    """
    groups: dict[str, list[float]] = defaultdict(list)
    enriched: list[dict[str, Any]] = []
    for binding in bindings:
        item = dict(binding or {})
        if item.get("entailment_score") is None:
            item["entailment_score"] = lexical_entailment_baseline(
                item.get("evidence_text"),
                subject=item.get("claim_subject"),
                predicate=item.get("claim_predicate"),
                value=item.get("claim_value"),
            )
        score = evidence_support_score(item, weights)
        item["evidence_support_score"] = score
        source_key = str(
            item.get("source_independence_key")
            or item.get("source_family_id")
            or item.get("source_key")
            or f"binding:{len(enriched)}"
        )
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
    # noisy-OR：独立来源逐步增强支持度，同一来源族已在上面折叠。
    residual = 1.0
    for score in scores:
        residual *= 1.0 - _clamp(score)
    fused = 1.0 - residual if scores else 0.0
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
