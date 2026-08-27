"""Schema policy defaults shared by completion and growth KG-delta code.

The domain payload can override these defaults through its ``properties`` and
``relations`` schema maps.  The constants below are deliberately centralized
so conflict behavior is not silently different between the two pipelines.
"""

from __future__ import annotations

from typing import Any

from src.rag.service.value_normalization_service import canonical_predicate

PROPERTY_POLICIES: dict[str, dict[str, str]] = {
    "\u522b\u540d": {"cardinality": "multi", "conflict_policy": "append"},
    "\u7b80\u79f0": {"cardinality": "multi", "conflict_policy": "append"},
    "\u8363\u8a89": {"cardinality": "multi", "conflict_policy": "append"},
    "\u7279\u8272": {"cardinality": "multi", "conflict_policy": "append"},
    "\u63cf\u8ff0": {"cardinality": "multi", "conflict_policy": "append"},
    "\u7b80\u4ecb": {"cardinality": "multi", "conflict_policy": "append"},
    "\u5386\u53f2\u6cbf\u9769": {"cardinality": "multi", "conflict_policy": "append"},
    "\u4f5c\u54c1": {"cardinality": "multi", "conflict_policy": "append"},
    "\u6d3b\u52a8": {"cardinality": "multi", "conflict_policy": "append"},
    "\u529f\u80fd": {"cardinality": "multi", "conflict_policy": "append"},
    "\u7528\u9014": {"cardinality": "multi", "conflict_policy": "append"},
    "\u6982\u8ff0": {"cardinality": "multi", "conflict_policy": "append"},
    "\u4ecb\u7ecd": {"cardinality": "multi", "conflict_policy": "append"},
    "\u6458\u8981": {"cardinality": "multi", "conflict_policy": "append"},
    "\u6587\u732e\u8bb0\u8f7d": {"cardinality": "multi", "conflict_policy": "append"},
    "\u5730\u8d28\u6210\u56e0": {"cardinality": "multi", "conflict_policy": "append"},
    "\u5730\u5f62\u7279\u5f81": {"cardinality": "multi", "conflict_policy": "append"},
    "\u5730\u8c8c\u7279\u5f81": {"cardinality": "multi", "conflict_policy": "append"},
    "\u7ec4\u6210": {"cardinality": "multi", "conflict_policy": "append"},
    "\u6784\u6210": {"cardinality": "multi", "conflict_policy": "append"},
    "\u5ca9\u6027": {"cardinality": "multi", "conflict_policy": "append"},
    "\u529f\u80fd\u7528\u9014": {"cardinality": "multi", "conflict_policy": "append"},
    "\u5907\u6ce8": {"cardinality": "multi", "conflict_policy": "append"},
    "\u8bf4\u660e": {"cardinality": "multi", "conflict_policy": "append"},
}

RELATION_POLICIES: dict[str, dict[str, str]] = {
    "\u7a7a\u95f4\u4f4d\u7f6e": {"cardinality": "single", "conflict_policy": "exclusive"},
    "\u5f52\u5c5e": {"cardinality": "single", "conflict_policy": "exclusive"},
    "\u4e0a\u7ea7\u533a\u57df": {"cardinality": "single", "conflict_policy": "exclusive"},
    "\u6240\u5c5e\u666f\u533a": {"cardinality": "single", "conflict_policy": "exclusive"},
    "\u96b6\u5c5e": {"cardinality": "single", "conflict_policy": "exclusive"},
    "\u6240\u5c5e\u673a\u6784": {"cardinality": "single", "conflict_policy": "exclusive"},
    "\u7236\u7ea7": {"cardinality": "single", "conflict_policy": "exclusive"},
    "\u4e0b\u4f0f\u4e8e": {"cardinality": "single", "conflict_policy": "exclusive"},
    "\u4e0a\u8986\u4e8e": {"cardinality": "single", "conflict_policy": "exclusive"},
}

RELATION_ALIASES = {
    "located_in": "\u7a7a\u95f4\u4f4d\u7f6e",
    "\u6240\u5728\u5730": "\u7a7a\u95f4\u4f4d\u7f6e",
    "\u4f4d\u7f6e": "\u7a7a\u95f4\u4f4d\u7f6e",
    "\u4f4d\u4e8e": "\u7a7a\u95f4\u4f4d\u7f6e",
    "\u5750\u843d\u4e8e": "\u7a7a\u95f4\u4f4d\u7f6e",
    "parent_organization": "\u6240\u5c5e\u673a\u6784",
    "\u96b6\u5c5e\u673a\u6784": "\u6240\u5c5e\u673a\u6784",
    "\u5408\u5e76\u7ec4\u6210\u5355\u4f4d": "\u5408\u5e76\u5bf9\u8c61",
}

EXCLUSIVE_RELATIONS = frozenset(RELATION_POLICIES)
MULTI_VALUE_PROPERTIES = frozenset(PROPERTY_POLICIES)
MULTI_VALUE_KEYWORDS = ("\u7b80\u4ecb", "\u63cf\u8ff0", "\u6982\u8ff0", "\u4ecb\u7ecd", "\u6458\u8981", "\u7279\u5f81", "\u7279\u8272", "\u8bb0\u8f7d", "\u6cbf\u9769", "\u6210\u56e0", "\u7ec4\u6210", "\u6784\u6210", "\u8bf4\u660e", "\u5907\u6ce8", "\u529f\u80fd", "\u7528\u9014")
TEMPORAL_CONFLICT_ROLES = frozenset({"construction_time", "current_status_time", "protection_time"})
COMPATIBLE_TEMPORAL_ROLES = frozenset({"renovation_time", "legend_time"})


def relation_key(value: Any) -> str:
    """Return the canonical relation label used by conflict policy lookup."""
    raw = str(value or "").strip()
    return canonical_predicate(RELATION_ALIASES.get(raw, raw))


def default_property_policy(predicate: str, temporal_role: str | None = None) -> dict[str, str]:
    """Return the default property cardinality/conflict policy."""
    if str(temporal_role or "") in COMPATIBLE_TEMPORAL_ROLES:
        return {"cardinality": "multi", "conflict_policy": "append"}
    canonical = canonical_predicate(str(predicate or "").strip())
    if canonical in PROPERTY_POLICIES:
        return PROPERTY_POLICIES[canonical]
    if any(keyword in canonical for keyword in MULTI_VALUE_KEYWORDS):
        return {"cardinality": "multi", "conflict_policy": "append"}
    return {"cardinality": "single", "conflict_policy": "exclusive"}


def default_relation_policy(predicate: str) -> dict[str, str]:
    """Return the default relation cardinality/conflict policy."""
    return RELATION_POLICIES.get(relation_key(predicate), {"cardinality": "multi", "conflict_policy": "append"})
