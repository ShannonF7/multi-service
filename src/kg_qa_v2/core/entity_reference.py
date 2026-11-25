"""问题中的实体引用解析：把表述、定语和上下文引用拆开。

本模块只做轻量、可审计的引用解析，不直接访问图谱，也不替代实体对齐。
"""
from __future__ import annotations

import re
from typing import Iterable

from ..models import ContextSnapshot, EntityQualifier, EntityReference


_CURRENT_LOCATION_RE = re.compile(
    r"(?:我(?:现在|当前)?在(?:哪里|哪儿|哪)|当前位置|当前所在|我所在位置|我这里|我附近)"
)
_CONTEXTUAL_RE = re.compile(
    r"(它|他|她|这个|那个|这里|那里|该楼|该校|该位置|上一个|刚才那个)"
)
_LEADING_RE = re.compile(
    r"^(?:请问|请帮我|帮我|请给我|给我|我想知道|我想了解|介绍一下|告诉我|关于|那么|然后|接着)\s*"
)
_HIERARCHY_MARKER_RE = re.compile(
    r"(?:大学|学院|校区|景区|园区|教学区|行政楼|教学楼|宿舍楼|实验楼|办公楼|"
    r"图书馆|体育馆|楼|座|层)"
)
_IMAGE_DEICTIC_RE = re.compile(r"^(?:这张|这张图片|这张照片|图片|照片|该图|这幅图|此图)$")


_PREFIXES = (
    "中国", "中华人民共和国", "山西省", "山西", "太原市", "太原",
    "晋中市", "晋中", "介休市", "介休",
)


def build_entity_references(
    question: str,
    targets: Iterable[str] | None,
    *,
    intent: str,
    context: ContextSnapshot,
    needs_current_location: bool = False,
) -> list[EntityReference]:
    """从槽位目标生成统一实体引用。

    顺序是：当前位置问题 → 上下文代词 → 显式目标。显式目标即使带有
    多级领域/校区/楼层定语，也保留原表述，同时提供可用于召回的核心名称。
    """
    text = str(question or "").strip()
    if needs_current_location or (intent == "LOCATION" and _CURRENT_LOCATION_RE.search(text)):
        return [EntityReference(
            surface="当前位置",
            core_mention="",
            qualifiers=[],
            reference_type="CURRENT_LOCATION",
        )]

    values = [str(item).strip() for item in (targets or []) if str(item).strip()]
    contextual_surface = _contextual_surface(text)
    if contextual_surface:
        core = context.last_entities[0] if context.last_entities else ""
        resolved = None
        if core and context.current_node_id and (
            not context.current_node_name or _same(core, context.current_node_name)
        ):
            resolved = context.current_node_id
        return [EntityReference(
            surface=contextual_surface,
            core_mention=core or contextual_surface,
            qualifiers=[EntityQualifier(type="context", text="沿用上一轮对象")] if core else [],
            reference_type="CONTEXTUAL",
            resolved_node_id=resolved,
        )]

    references: list[EntityReference] = []
    for value in values:
        surface = _clean_surface(value)
        core, qualifiers = _split_mention(surface)
        references.append(EntityReference(
            surface=surface,
            core_mention=core or surface,
            qualifiers=qualifiers,
            reference_type="EXPLICIT" if core or surface else "UNKNOWN",
        ))
    return _dedupe_references(references)


def is_image_deictic_reference(reference: EntityReference) -> bool:
    """判断引用是否只是“这张/这张图片”等图像指代，不应作为文本实体检索词。"""
    values = (reference.surface, reference.core_mention)
    return any(_IMAGE_DEICTIC_RE.fullmatch(_clean_surface(value)) for value in values if value)


def recall_terms(reference: EntityReference) -> list[str]:
    """返回从强到弱的实体召回词，不包含描述或事实文本。"""
    values: list[str] = []
    for value in (
        reference.surface,
        reference.core_mention,
        *[item.text for item in reference.qualifiers if item.type != "context"],
    ):
        value = _clean_surface(value)
        if value and len(value) >= 2 and value not in values:
            values.append(value)
    return values[:8]


def _clean_surface(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"^[\"'“”‘’（）()【】\[\]、，。！？?；;:：\s]+", "", value)
    value = re.sub(r"[\"'“”‘’（）()【】\[\]、，。！？?；;:：\s]+$", "", value)
    value = _LEADING_RE.sub("", value)
    value = re.sub(r"^的+", "", value).strip()
    value = re.sub(r"(?:在哪里|在哪儿|在哪|的位置|的地址|的图片|图片|照片)$", "", value).strip()
    return value


def _split_mention(surface: str) -> tuple[str, list[EntityQualifier]]:
    value = _clean_surface(surface)
    if not value:
        return "", []

    qualifiers: list[EntityQualifier] = []
    cursor = 0
    last_end = 0
    for match in _HIERARCHY_MARKER_RE.finditer(value):
        segment = value[cursor:match.end()].strip()
        if segment:
            qualifiers.append(EntityQualifier(type=_qualifier_type(segment), text=segment))
        cursor = match.end()
        last_end = match.end()
    tail = value[last_end:].strip() if last_end else value
    tail = re.sub(r"^的+", "", tail).strip()
    if not last_end:
        for prefix in _PREFIXES:
            if tail.startswith(prefix) and len(tail) - len(prefix) >= 2:
                qualifiers.append(EntityQualifier(type="geography", text=prefix))
                tail = tail[len(prefix):].strip()
                break
    if not tail and qualifiers:
        tail = qualifiers.pop().text
    core = tail or value

    # “太原理工大学明向校区行政楼B座财务部”这类输入中，marker
    # 可能已经把末尾核心名称吃进了上一段；从最后一段中再做一次兜底。
    if qualifiers and core in {item.text for item in qualifiers} and len(value) > len(core):
        core = value[len(value) - len(core):]
    return core, _dedupe_qualifiers(qualifiers)


def _qualifier_type(value: str) -> str:
    if any(token in value for token in ("大学", "学院")):
        return "organization"
    if any(token in value for token in ("省", "市")):
        return "geography"
    if any(token in value for token in ("校区", "景区", "园区", "教学区")):
        return "area"
    return "hierarchy"


def _contextual_surface(text: str) -> str | None:
    match = _CONTEXTUAL_RE.search(text)
    return match.group(1) if match else None


def _same(left: str, right: str) -> bool:
    return re.sub(r"\s+", "", str(left)).casefold() == re.sub(r"\s+", "", str(right)).casefold()


def _dedupe_qualifiers(values: list[EntityQualifier]) -> list[EntityQualifier]:
    result: list[EntityQualifier] = []
    seen: set[str] = set()
    for item in values:
        key = item.text.casefold()
        if key and key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _dedupe_references(values: list[EntityReference]) -> list[EntityReference]:
    result: list[EntityReference] = []
    seen: set[tuple[str, str]] = set()
    for item in values:
        key = (item.reference_type, item.core_mention.casefold() or item.surface.casefold())
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result
