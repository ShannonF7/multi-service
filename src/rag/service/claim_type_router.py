"""Conservative routing of extracted claims into graph or background lanes."""

from __future__ import annotations

from typing import Any

BOOLEAN_VALUES = {"是", "否", "true", "false", "True", "False"}
ACTION_CUES = ("落实", "推动", "开展", "实施", "坚持", "承担", "服务", "形成", "完成")
GOAL_CUES = ("目标", "建设", "建成", "打造", "发展", "提升", "愿景", "规划")
POLICY_CUES = ("政策", "方针", "战略", "方案", "意见", "规定")
EVENT_CUES = ("始建", "建于", "成立", "建成", "重修", "维修", "发生", "举办")


def _schema_boolean(predicate: str, schema: dict[str, Any] | None) -> bool:
    schema = schema or {}
    properties = schema.get("properties") or {}
    config = properties.get(predicate) if isinstance(properties, dict) else None
    if isinstance(config, dict) and str(config.get("value_type") or "").lower() in {"bool", "boolean"}:
        return True
    if isinstance(config, str) and config.lower() in {"bool", "boolean"}:
        return True
    return predicate in set(str(item) for item in (schema.get("boolean_properties") or []))


def background_role(predicate: str, raw_text: str = "") -> str:
    text = f"{predicate} {raw_text}"
    if any(cue in text for cue in GOAL_CUES):
        return "GOAL"
    if any(cue in text for cue in POLICY_CUES):
        return "POLICY"
    if any(cue in text for cue in EVENT_CUES):
        return "EVENT"
    if any(cue in text for cue in ACTION_CUES):
        return "ACTION"
    return "OTHER"


def route_claim(
    *,
    claim_type: str,
    predicate: str,
    value: str,
    raw_text: str = "",
    schema: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return GRAPH PROPERTY/RELATION or BACKGROUND with a semantic role."""
    normalized_type = str(claim_type or "").strip().upper()
    predicate = str(predicate or "").strip()
    value = str(value or "").strip()
    if normalized_type == "RELATION":
        return {"claim_type": "RELATION", "semantic_role": ""}
    if normalized_type not in {"PROPERTY", ""}:
        return {"claim_type": "BACKGROUND", "semantic_role": background_role(predicate, raw_text)}
    if value in BOOLEAN_VALUES and not (
        _schema_boolean(predicate, schema)
        or predicate.startswith(("是否", "有无"))
        or (len(predicate) <= 12 and not any(cue in predicate for cue in ACTION_CUES + GOAL_CUES + POLICY_CUES))
    ):
        return {"claim_type": "BACKGROUND", "semantic_role": background_role(predicate, raw_text)}
    if any(cue in predicate for cue in ACTION_CUES + GOAL_CUES + POLICY_CUES) and value in BOOLEAN_VALUES:
        return {"claim_type": "BACKGROUND", "semantic_role": background_role(predicate, raw_text)}
    return {"claim_type": "PROPERTY", "semantic_role": ""}

