"""Value normalization for semantic completion candidates."""

from __future__ import annotations

import re
from typing import Any

from src.rag.schemas import CandidateClaim, SemanticCompleteRequest

PROPERTY_ALIAS_MAP = {
    "\u529f\u80fd": "\u529f\u80fd\u7528\u9014",
    "\u7528\u9014": "\u529f\u80fd\u7528\u9014",
    "\u4f5c\u7528": "\u529f\u80fd\u7528\u9014",
    "\u6765\u6e90": "\u6765\u6e90\u51fa\u5904",
    "\u51fa\u5904": "\u6765\u6e90\u51fa\u5904",
    "\u63cf\u8ff0": "\u63cf\u8ff0\u7b80\u4ecb",
    "\u7b80\u4ecb": "\u63cf\u8ff0\u7b80\u4ecb",
    "\u6982\u8ff0": "\u63cf\u8ff0\u7b80\u4ecb",
    "\u4f4d\u7f6e": "\u7a7a\u95f4\u4f4d\u7f6e",
    "\u5730\u5740": "\u7a7a\u95f4\u4f4d\u7f6e",
    "located_in": "\u7a7a\u95f4\u4f4d\u7f6e",
    "founded_year": "\u6210\u7acb\u5e74\u4efd",
    "original_name": "\u539f\u540d",
    "former_name": "\u66fe\u7528\u540d",
    "merged_with": "\u5408\u5e76\u5bf9\u8c61",
    "merged_year": "\u5408\u5e76\u5e74\u4efd",
    "211_project_member": "211\u5de5\u7a0b\u9ad8\u6821",
    "double_first_class_university": "\u53cc\u4e00\u6d41\u5efa\u8bbe\u9ad8\u6821",
    "historical_significance": "\u5386\u53f2\u5730\u4f4d",
    "motto": "\u6821\u8bad",
    "cultural_characteristics": "\u6587\u5316\u7279\u8d28",
    "parent_organization": "\u6240\u5c5e\u673a\u6784",
    "status": "\u8ba4\u5b9a\u72b6\u6001",
    "mission_or_focus": "\u529e\u5b66\u5b97\u65e8",
}

TEMPORAL_VALUE_MAP = {
    "\u5510": "\u5510\u4ee3",
    "\u5510\u671d": "\u5510\u4ee3",
    "\u5510\u738b\u671d": "\u5510\u4ee3",
    "\u76db\u5510": "\u5510\u4ee3",
    "\u4e2d\u5510": "\u5510\u4ee3",
    "\u665a\u5510": "\u5510\u4ee3\u665a\u671f",
    "\u5b8b": "\u5b8b\u4ee3",
    "\u5b8b\u671d": "\u5b8b\u4ee3",
    "\u5317\u5b8b": "\u5317\u5b8b\u65f6\u671f",
    "\u5357\u5b8b": "\u5357\u5b8b\u65f6\u671f",
    "\u5143": "\u5143\u4ee3",
    "\u5143\u671d": "\u5143\u4ee3",
    "\u660e": "\u660e\u4ee3",
    "\u660e\u671d": "\u660e\u4ee3",
    "\u6e05": "\u6e05\u4ee3",
    "\u6e05\u671d": "\u6e05\u4ee3",
    "\u660e\u6e05": "\u660e\u6e05\u65f6\u671f",
    "\u968b\u5510": "\u968b\u5510\u65f6\u671f",
    "\u6e05\u672b": "\u6e05\u4ee3\u665a\u671f",
    "\u6c11\u56fd": "\u6c11\u56fd\u65f6\u671f",
    "\u4e2d\u534e\u4eba\u6c11\u5171\u548c\u56fd": "\u73b0\u4ee3",
    "\u65b0\u4e2d\u56fd": "\u73b0\u4ee3",
}

TEMPORAL_ROLE_HINTS = {
    "construction_time",
    "renovation_time",
    "current_status_time",
    "legend_time",
    "protection_time",
}

TIME_RANGE_PATTERNS = [
    (re.compile(r"(\d{3,4})\s*(?:-|\u2014|\u81f3|\u5230)\s*(\d{3,4})\s*\u5e74?"), "year_range"),
    (re.compile(r"(\d{3,4})\s*\u5e74"), "year"),
]

UNCERTAIN_HINTS = (
    "\u7ea6", "\u5927\u7ea6", "\u5de6\u53f3", "\u53ef\u80fd", "\u636e\u4f20",
    "\u76f8\u4f20", "\u7591\u4e3a", "\u6216", "\u7ea6\u4e3a",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", _clean(value)).strip()


def _strip_noise(value: str) -> str:
    text = _normalize_whitespace(value)
    text = re.sub(r"^[\uff1a:,\uff0c\u3001\u3002\uff1b;\s]+", "", text)
    text = re.sub(r"[\u3002\uff1b;\uff0c,\u3001\s]+$", "", text)
    text = re.sub(r"[\uff08(](?:\u6570\u636e|\u8d44\u6599|\u6765\u6e90|\u622a\u81f3|\u53c2\u89c1|\u89c1)[^\uff09)]*[\uff09)]$", "", text).strip()
    return text


def canonical_predicate(predicate: str, *, temporal_role: str | None = None) -> str:
    text = _clean(predicate)
    if temporal_role in TEMPORAL_ROLE_HINTS:
        return text or "\u65f6\u95f4"
    return PROPERTY_ALIAS_MAP.get(text, text)


def normalize_temporal_value(value: str) -> tuple[str, dict[str, Any]]:
    text = _strip_noise(value)
    if not text:
        return text, {"certainty": "unknown"}
    compact = re.sub(r"\s+", "", text)
    certainty = "uncertain" if any(hint in compact for hint in UNCERTAIN_HINTS) else "certain"
    cleaned = compact
    for hint in UNCERTAIN_HINTS:
        cleaned = cleaned.replace(hint, "")
    normalized = TEMPORAL_VALUE_MAP.get(cleaned, TEMPORAL_VALUE_MAP.get(text, text))
    time_start = None
    time_end = None
    granularity = None
    for pattern, kind in TIME_RANGE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        granularity = kind
        if kind == "year_range":
            time_start = int(match.group(1))
            time_end = int(match.group(2))
            normalized = f"{time_start}-{time_end}\u5e74"
        else:
            time_start = int(match.group(1))
            time_end = time_start
            normalized = f"{time_start}\u5e74"
        break
    return normalized, {
        "time_text": text,
        "time_normalized": normalized,
        "time_start": time_start,
        "time_end": time_end,
        "time_granularity": granularity or ("period" if normalized else None),
        "certainty": certainty,
    }


def normalize_text_value(value: str) -> str:
    text = _strip_noise(value)
    if not text:
        return text
    text = re.sub(r"[\uff08(][^\uff09)]*[\uff09)]$", "", text).strip()
    return text


def _domain_property_config(payload: SemanticCompleteRequest, predicate: str) -> dict[str, Any]:
    schema = (payload.metadata or {}).get("domain_schema") or {}
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if predicate in props and isinstance(props[predicate], dict):
        return props[predicate]
    return {}


def normalize_candidate_claims(payload: SemanticCompleteRequest, claims: list[CandidateClaim]) -> list[CandidateClaim]:
    for claim in claims:
        predicate = _clean(claim.predicate)
        temporal_role = _clean(claim.temporal_role)
        canonical = canonical_predicate(predicate, temporal_role=temporal_role or None)
        raw_value = _clean(claim.object_value or claim.object_name)
        normalized_value = raw_value
        normalization_mode = "text"
        extra: dict[str, Any] = {}
        prop_config = _domain_property_config(payload, canonical or predicate)
        value_type = str(prop_config.get("value_type") or "").lower()
        if claim.claim_type == "property":
            is_temporal = temporal_role in TEMPORAL_ROLE_HINTS or value_type == "time" or predicate in {"\u65f6\u671f", "\u5386\u53f2\u65f6\u671f", "\u5e74\u4ee3", "\u65f6\u95f4", "\u5efa\u9020\u65f6\u95f4", "\u59cb\u5efa\u65f6\u95f4", "\u4fee\u7f2e\u65f6\u95f4"}
            if is_temporal:
                normalized_value, extra = normalize_temporal_value(raw_value)
                normalization_mode = "temporal"
            else:
                normalized_value = normalize_text_value(raw_value)
        elif claim.claim_type == "relation":
            normalized_value = normalize_text_value(raw_value)
            normalization_mode = "relation_entity_name"
        else:
            normalized_value = normalize_text_value(raw_value)

        display_value = normalized_value or raw_value
        claim.raw_value = raw_value or None
        claim.normalized_value = normalized_value or None
        claim.display_value = display_value or None
        if claim.claim_type == "property":
            claim.object_value = display_value or claim.object_value
        elif claim.claim_type == "relation":
            claim.object_name = display_value or claim.object_name
        claim.metadata = dict(claim.metadata or {})
        claim.metadata["raw_predicate"] = predicate
        if completion_mode := str((payload.metadata or {}).get("completion_mode") or ""):
            if completion_mode == "growth_g2":
                claim.predicate = canonical
        claim.metadata.update({
            "canonical_predicate": canonical,
            "raw_value": claim.raw_value,
            "normalized_value": claim.normalized_value,
            "display_value": claim.display_value,
            "normalization_mode": normalization_mode,
            "normalization_version": "value-normalization-v2",
            "temporal_role": temporal_role or None,
            "alias_source": predicate if canonical != predicate else None,
            **extra,
        })
    return claims
