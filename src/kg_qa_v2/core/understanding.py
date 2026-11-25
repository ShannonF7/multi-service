"""统一的问题理解入口：Jev Router + Qwen Structured Slots。"""

from __future__ import annotations

from typing import Any, Callable, Optional
import re
import time

from ..models import ContextSnapshot, QuerySpec, ResponseRequirements, UnderstandingResult
from .intent_router import JevRouter
from .request_catalog import can_fast_parse
from .slots import QwenSlotExtractor


class QuestionUnderstandingService:
    """将供应商模型隔离在两个窄适配器后面。"""

    def __init__(
        self,
        *,
        router_client: Optional[Callable[..., Any]] = None,
        slot_client: Optional[Callable[..., Any]] = None,
        decision_model: Optional[Any] = None,
    ) -> None:
        self.decision_model = decision_model
        self.router = JevRouter(router_client or decision_model)
        self.slots = QwenSlotExtractor(slot_client)

    @classmethod
    def from_env(cls) -> "QuestionUnderstandingService":
        from ..adapters.qwen_client import build_qwen_slot_client_from_env
        from .decision_model import build_decision_model_from_env

        return cls(
            decision_model=build_decision_model_from_env(),
            slot_client=build_qwen_slot_client_from_env(),
        )

    def understand(
        self,
        question: str,
        *,
        context: ContextSnapshot,
        has_image: bool = False,
        target_modality: str | None = None,
        timings: Optional[dict[str, int]] = None,
    ) -> UnderstandingResult:
        started = time.perf_counter()
        route = self.router.route(
            question,
            has_image=has_image,
            target_modality=target_modality,
            context=context,
            timings=timings,
        )
        if timings is not None:
            timings["router_ms"] = int((time.perf_counter() - started) * 1000)
        slot_started = time.perf_counter()
        # 闲聊没有实体、属性或检索槽位；路由确认后直接结束理解阶段。
        # 这样不会再为“你好”额外调用一次结构化槽位模型。
        if route.route == "CASUAL":
            from .slots import fallback_slots
            spec = fallback_slots(question, route=route, context=context)
        elif (
            route.source == "fallback"
            and self.router.last_runtime_meta.get("router_path") == "fast_path"
            and can_fast_parse(question)
            and (self.decision_model is not None or self.slots.client is None)
        ):
            # Fast Router 已经给出保守任务类型；短问题用本地确定性解析。
            from .slots import fallback_slots
            spec = fallback_slots(question, route=route, context=context)
        elif (
            route.source == "fallback"
            and route.intent == "DIRECT_FACT"
            and context.last_entities
            and re.search(r"(是谁|管理员|负责人|联系人|电话|联系方式|邮箱|名称|归谁|谁负责)", str(question or ""))
        ):
            # 短属性追问直接继承上一轮对象，避免再次调用槽位模型误判或延迟。
            from .slots import fallback_slots
            spec = fallback_slots(question, route=route, context=context)
        else:
            spec = self.slots.extract(question, route=route, context=context)
        spec = derive_execution_requirements(
            spec,
            question=question,
            route=route,
            has_image=has_image,
            target_modality=target_modality,
        )
        if timings is not None:
            timings["slot_ms"] = int((time.perf_counter() - slot_started) * 1000)
        return UnderstandingResult(
            route=route,
            query_spec=spec,
            runtime_meta=dict(self.router.last_runtime_meta or {}),
        )


def derive_execution_requirements(
    spec: QuerySpec,
    *,
    question: str,
    route: Any,
    has_image: bool,
    target_modality: str | None,
) -> QuerySpec:
    """由程序根据语义结果编译执行需求，模型不得直接决定检索开关。"""
    intent = spec.intent
    is_casual = spec.route == "CASUAL" or intent == "CASUAL"
    text = str(question or "")
    request_types = {item.request_type for item in spec.requests}
    needs_image = (not is_casual) and (
        intent == "IMAGE"
        or has_image
        or str(target_modality or "").upper() == "IMAGE"
        or bool(request_types & {"VISUAL_APPEARANCE", "IMAGE_RETRIEVAL", "IMAGE_IDENTIFICATION"})
        or any(token in text for token in ("图片", "照片", "图像", "配图", "长什么样", "长啥样", "外观", "现场图"))
    )
    needs_path = (not is_casual) and intent == "PATH"
    needs_graph = spec.route == "KNOWLEDGE" and not is_casual
    needs_text = (not is_casual) and intent in {"EXPLAIN", "COMPARE"}
    needs_current_location = (not is_casual) and (
        route.context_dependency == "LOCATION" or route.needs_current_location
    )
    return spec.model_copy(update={
        "needs_graph": needs_graph,
        "needs_text": needs_text,
        "needs_image": needs_image,
        "needs_path": needs_path,
        "response_requirements": spec.response_requirements.model_copy(update={
            "images": bool(spec.response_requirements.images or needs_image),
            "text": True,
        }),
        "context_dependent": bool(spec.context_dependent or route.context_dependent),
        "context_dependency": "NONE" if is_casual else route.context_dependency,
        "needs_current_location": needs_current_location,
        "needs_clarification": bool(route.needs_clarification or spec.needs_clarification),
    })
