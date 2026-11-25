"""Qwen Structured Output 槽位抽取适配器及安全降级。"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from ..models import ContextSnapshot, QueryRequest, QuerySpec, ResponseRequirements, RouteDecision
from .entity_reference import build_entity_references
from .prompts import build_slot_prompt
from .request_catalog import ATTRIBUTE_ALIASES, attribute_for_text, markers_for


class QwenSlotExtractor:
    def __init__(self, client: Optional[Callable[..., Any]] = None) -> None:
        self.client = client

    def extract(self, question: str, *, route: RouteDecision, context: ContextSnapshot) -> QuerySpec:
        if self.client is None:
            return fallback_slots(question, route=route, context=context)
        try:
            raw = self.client(
                prompt=build_slot_prompt(
                    question,
                    route_json=route.model_dump_json(),
                    context_json=context.model_dump_json(),
                ),
                response_format="json",
            )
            payload = _normalize_slot_payload(_as_dict(raw))
            allowed_slots = {
                key: payload[key]
                for key in (
                    "entity_refs", "targets", "requests", "attribute", "start", "end", "time_range",
                    "comparison_targets", "constraints", "scope", "question_rewrite", "response_requirements",
                )
                if key in payload
            }
            spec = QuerySpec.model_validate({
                **allowed_slots,
                "intent": route.intent,
                "route": route.route,
                "source": "jev+qwen",
                "context_dependent": route.context_dependent,
                "context_dependency": route.context_dependency,
                "needs_current_location": route.needs_current_location,
                "needs_clarification": route.needs_clarification,
            })
            spec = _attach_entity_references(question, spec, context)

            # 结构化模型可能返回合法但空的 JSON。此时不能把“抽取成功”
            # 等同于“没有查询对象”，否则后续不会进入已发布图谱候选召回。
            if route.route == "KNOWLEDGE" and (not spec.entity_refs or not spec.requests):
                fallback = fallback_slots(question, route=route, context=context)
                updates = {}
                used_entity_fallback = not spec.entity_refs and bool(fallback.entity_refs)
                if used_entity_fallback:
                    updates.update({"targets": fallback.targets, "entity_refs": fallback.entity_refs})
                if not spec.requests and fallback.requests:
                    updates.update({"requests": fallback.requests, "attribute": spec.attribute or fallback.attribute})
                if updates:
                    updates.update({
                        "question_rewrite": spec.question_rewrite or fallback.question_rewrite,
                        "context_dependent": spec.context_dependent or fallback.context_dependent,
                    })
                    if used_entity_fallback:
                        updates["source"] = "jev+qwen+fallback"
                    spec = spec.model_copy(update=updates)
            return spec
        except Exception:
            return fallback_slots(question, route=route, context=context)


def fallback_slots(question: str, *, route: RouteDecision, context: ContextSnapshot) -> QuerySpec:
    text = str(question or "").strip()
    targets = _extract_targets(text, route.intent)
    context_dependent = route.context_dependent or bool(
        re.search(r"^(?:他|她|它|这个|那个|这里|那里|该楼|该校)|(?:他|她|它|这个|那个|这里|那里|该楼|该校)", text)
    )
    if route.intent == "LOCATION" and route.needs_current_location:
        targets = []
    if context.last_entities and _is_implicit_context_attribute_question(text):
        targets = [context.last_entities[0]]
        context_dependent = True
    if context_dependent and context.last_entities:
        targets = [context.last_entities[0]]
    start, end = _extract_path_endpoints(text)
    comparison_targets = _extract_comparison_targets(text)
    intent = route.intent
    if intent == "UNKNOWN":
        intent = "DIRECT_FACT" if route.route == "KNOWLEDGE" else "CASUAL"
    requests = _extract_requests(text, intent)
    primary_attribute = next(
        (item.attribute for item in requests if item.request_type == "ATTRIBUTE" and item.attribute),
        "location" if intent == "LOCATION" else _infer_attribute(text, intent),
    )
    spec = QuerySpec(
        intent=intent,
        route=route.route,
        targets=targets,
        requests=requests,
        attribute=primary_attribute,
        start=start,
        end=end,
        comparison_targets=comparison_targets,
        response_requirements=_response_requirements_for(requests, route_intent=intent),
        needs_graph=route.needs_graph,
        needs_text=route.needs_text,
        needs_image=route.needs_image,
        needs_path=route.needs_path,
        context_dependent=context_dependent,
        context_dependency=route.context_dependency,
        needs_current_location=route.needs_current_location,
        needs_clarification=route.needs_clarification,
        question_rewrite=_rewrite(text, [], context) if context_dependent else _rewrite(text, targets, context),
        confidence=route.confidence,
        source="fallback",
    )
    return _attach_entity_references(text, spec, context)


def _extract_requests(text: str, intent: str) -> list[QueryRequest]:
    """将自然语言问题转换为可独立执行的请求；规则只负责语义解析，不判断证据是否存在。"""
    value = str(text or "").strip()
    found: list[tuple[int, QueryRequest]] = []

    def add(
        position: int,
        request_type: str,
        attribute: str | None,
        surface: str,
        *,
        target: str | None = None,
    ) -> None:
        if any(item.attribute == attribute and item.request_type == request_type for _, item in found):
            return
        found.append((position, QueryRequest(
            request_type=request_type,
            attribute=attribute,
            surface=surface.strip() or value,
            target=target,
        )))

    def segment(position: int, *, include_after: bool = True) -> str:
        begin = max(0, value.rfind("。", 0, position) + 1)
        if not include_after:
            return value[begin:position]
        end = len(value)
        for separator in ("？", "?", "！", "!", "。", "；", ";", "，", ","):
            candidate = value.find(separator, position)
            if candidate >= 0:
                end = min(end, candidate + 1)
        return value[begin:end]

    for attribute, aliases in ATTRIBUTE_ALIASES.items():
        for alias in aliases:
            position = value.casefold().find(alias.casefold())
            if position >= 0:
                add(position, "ATTRIBUTE", attribute, segment(position))
                break

    identification_question = (
        any(token in value for token in ("这张", "这幅", "此图", "上传图片", "上传照片"))
        and any(marker in value for marker in ("哪里", "在哪", "是什么"))
    )
    for request_type in ("NEARBY", "VISUAL_APPEARANCE", "IMAGE_RETRIEVAL", "LOCATION"):
        if request_type == "IMAGE_RETRIEVAL" and identification_question:
            continue
        matches = [
            (value.find(marker), marker)
            for marker in markers_for(request_type)
            if value.find(marker) >= 0
        ]
        if not matches:
            continue
        position, marker = min(matches, key=lambda item: (item[0], -len(item[1])))
        target_intent = "LIST" if request_type == "NEARBY" else "IMAGE" if request_type != "LOCATION" else "LOCATION"
        target = (_extract_targets(value, target_intent) or [None])[0]
        attribute = "location" if request_type == "LOCATION" else "image" if request_type == "IMAGE_RETRIEVAL" else None
        add(position, request_type, attribute, segment(position), target=target)

    if intent == "IMAGE" and identification_question:
        add(0, "IMAGE_IDENTIFICATION", "image", value)

    if intent == "HIERARCHY" and not any(item.request_type == "HIERARCHY" for _, item in found):
        add(0, "HIERARCHY", "hierarchy", value)
    if intent == "PATH" and not any(item.request_type == "PATH" for _, item in found):
        add(0, "PATH", "path", value)
    if not found and intent == "DIRECT_FACT":
        attribute = attribute_for_text(value) or _infer_attribute(value, intent)
        if attribute:
            add(0, "ATTRIBUTE", attribute, value)
    if not found and intent == "LOCATION":
        add(0, "LOCATION", "location", value)
    return [item for _, item in sorted(found, key=lambda pair: pair[0])]



def _extract_targets(text: str, intent: str) -> list[str]:
    cleaned = re.sub(r"[，。！？?；;、]+", " ", text).strip()
    cleaned = re.sub(r"^(?:给我|请给我|我想看|我想要看|给我看看)\s*", "", cleaned).strip()
    markers = list({
        marker
        for request_type in ("NEARBY", "VISUAL_APPEARANCE", "IMAGE_RETRIEVAL", "LOCATION")
        for marker in markers_for(request_type)
    }) + [
        "有哪些", "是什么", "为什么", "怎么去", "如何到达", "是谁",
        "的图片", "图片", "照片", "图像", "电话", "联系方式",
    ]
    if intent == "IMAGE":
        markers.extend(("长什么样", "长啥样"))
    candidates = [
        (cleaned.find(marker), marker)
        for marker in markers
        if cleaned.find(marker) >= 0
    ]
    for _, marker in sorted(candidates, key=lambda item: (item[0], -len(item[1]))):
        value = cleaned.split(marker, 1)[0]
        value = re.sub(r"^(?:请问|请介绍一下|介绍一下|告诉我|我想知道|那么|然后|接着|关于)", "", value).strip()
        value = re.sub(r"(?:的)?(?:图片|照片|图像)$", "", value).strip()
        value = re.sub(r"[的之]\s*$", "", value).strip()
        if 2 <= len(value) <= 64 and value not in {"我", "这里", "此处"}:
            return [value]
    return []



def _is_implicit_context_attribute_question(text: str) -> bool:
    compact = re.sub(r"[，。！？?；;、\s]+", "", str(text or ""))
    return bool(re.fullmatch(
        r"(?:管理员|负责人|联系人|电话|联系方式|邮箱|名称|归谁|谁负责)(?:是谁|是多少|是什么)?",
        compact,
    ))


def _infer_attribute(text: str, intent: str) -> str | None:
    if intent == "LOCATION":
        return "location"
    return attribute_for_text(text)



def _extract_path_endpoints(text: str) -> tuple[str | None, str | None]:
    cleaned = re.sub(r"[，。！？?；;、]+", " ", text).strip()
    match = re.search(r"从\s*(.+?)\s*到\s*(.+?)(?:怎么去|怎么走|如何到达|的路线|路线|$)", cleaned)
    return (match.group(1).strip(), match.group(2).strip()) if match else (None, None)


def _extract_comparison_targets(text: str) -> list[str]:
    match = re.search(r"(?:比较|对比)\s*(.+?)\s*(?:和|与|、)\s*(.+?)(?:的|有什么|区别|差异|$)", text)
    if not match:
        return []
    return [value.strip() for value in match.groups() if value.strip()]


def _rewrite(text: str, targets: list[str], context: ContextSnapshot) -> str:
    if targets:
        return text
    if context.last_entities and re.search(r"(?:他|她|它|这个|那个|这里|那里|该楼|接着|然后)", text):
        return f"关于{context.last_entities[0]}，{text}"
    return text


def _response_requirements_for(
    requests: list[QueryRequest],
    *,
    route_intent: str,
) -> ResponseRequirements:
    request_types = {item.request_type for item in requests}
    return ResponseRequirements(
        text=True,
        images=bool(request_types & {"VISUAL_APPEARANCE", "IMAGE_RETRIEVAL", "IMAGE_IDENTIFICATION"})
            or str(route_intent).upper() == "IMAGE",
        map=False,
    )


def _normalize_slot_payload(payload: dict[str, Any]) -> dict[str, Any]:
    requests = payload.get("requests")
    if isinstance(requests, list):
        normalized = []
        for item in requests:
            if not isinstance(item, dict):
                continue
            value = dict(item)
            if "request_type" not in value and "type" in value:
                value["request_type"] = value.pop("type")
            normalized.append(value)
        payload = dict(payload)
        payload["requests"] = normalized
    return payload


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError("Slot client must return a dict, JSON object, or Pydantic model")


def _attach_entity_references(
    question: str,
    spec: QuerySpec,
    context: ContextSnapshot,
) -> QuerySpec:
    # Question Understanding 已经返回正式 EntityReference 时优先使用模型
    # 的结构化结果；仅在旧客户端或空结果时使用本地轻量解析兜底。
    references = spec.entity_refs or build_entity_references(
        question,
        spec.targets,
        intent=spec.intent,
        context=context,
        needs_current_location=spec.needs_current_location,
    )
    targets = [
        item.core_mention or item.surface
        for item in references
        if item.reference_type not in {"CURRENT_LOCATION", "UNKNOWN"}
    ]
    return spec.model_copy(update={
        "targets": list(dict.fromkeys(targets)),
        "entity_refs": references,
        "attribute": spec.attribute or _infer_attribute(question, spec.intent),
        "context_dependent": spec.context_dependent or any(
            item.reference_type == "CONTEXTUAL" for item in references
        ),
    })
