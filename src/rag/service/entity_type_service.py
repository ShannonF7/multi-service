"""Shared entity-type normalization and conservative inference."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


TYPE_ALIASES = {
    "poi": "poi", "place": "poi", "地点": "poi", "场所": "poi",
    "building": "building", "建筑": "building",
    "region": "region", "location": "region", "区域": "region",
    "person": "person", "人物": "person",
    "object": "object", "物品": "object", "文物": "object", "artifact": "object",
    "facility": "facility", "设施": "facility",
    "organization": "organization", "organisation": "organization", "机构": "organization",
    "school": "organization", "大学": "organization",
    "event": "event", "事件": "event",
    "route": "route", "路线": "route",
    "concept": "concept", "概念": "concept",
    "program": "program", "项目": "program",
    "scenicarea": "scenicarea", "景区": "scenicarea",
    "entity": "", "unresolved": "", "unknown": "", "": "",
}

GENERIC_TYPES = {"", "poi", "entity", "unresolved", "unknown", "thing", "object"}
ORG_PREDICATES = {"所属机构", "隶属机构", "parent_organization", "合并组成单位", "merged_with", "前身", "predecessor"}
PERSON_PREDICATES = {"建造者", "设计者", "相关人物", "notable_person", "人物"}
ORG_SUFFIXES = ("大学", "学院", "研究院", "研究所", "实验室", "办公室", "委员会", "集团", "公司", "基金会", "协会", "部门", "处", "中心", "机构")
BUILDING_SUFFIXES = ("楼", "馆", "庙", "塔", "殿", "寺")


def normalize_entity_type(value: Any) -> str:
    raw = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    if "=" in raw:
        code, _label = raw.split("=", 1)
        if code.strip().startswith(("type_", "domain_")) or code.strip() == "ce5":
            raw = code.strip()
    if not raw:
        return ""
    return TYPE_ALIASES.get(raw, raw[:64])


def infer_entity_type(raw_type: Any, *, predicate: Any = "", mention: Any = "", quote: Any = "") -> tuple[str, float, str]:
    """Return (type, confidence, method); never invents a type for ambiguous names."""
    raw = normalize_entity_type(raw_type)
    if raw not in GENERIC_TYPES:
        return raw, 0.9, "EXPLICIT_TYPE"
    pred = unicodedata.normalize("NFKC", str(predicate or "")).strip()
    name = unicodedata.normalize("NFKC", str(mention or "")).strip()
    if pred in ORG_PREDICATES or any(name.endswith(suffix) for suffix in ORG_SUFFIXES):
        return "organization", 0.86, "DETERMINISTIC_ORGANIZATION_CUE"
    if pred in PERSON_PREDICATES:
        return "person", 0.82, "DETERMINISTIC_PERSON_PREDICATE"
    if any(name.endswith(suffix) for suffix in BUILDING_SUFFIXES):
        return "building", 0.72, "DETERMINISTIC_BUILDING_SUFFIX"
    # A quote is intentionally not used for free-form semantic guessing.
    return raw, 0.0, "UNRESOLVED_GENERIC_TYPE"
