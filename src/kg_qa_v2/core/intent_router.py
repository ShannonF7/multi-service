"""Router：优先使用 Jev，失败时使用保守规则降级。"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Optional

from ..models import ContextSnapshot, RouteDecision
from .request_catalog import can_fast_parse


class JevRouter:
    """Jev 的窄接口适配器。

    client 由部署侧注入，避免把供应商 SDK、密钥和网络调用写进业务链路。
    """

    def __init__(self, client: Optional[Callable[..., Any]] = None) -> None:
        self.client = client
        self.last_runtime_meta: dict[str, Any] = {}

    def route(self, question: str, *, has_image: bool, target_modality: str | None,
              context: ContextSnapshot, timings: Optional[dict[str, int]] = None) -> RouteDecision:
        started = time.perf_counter()
        # 规则 fast path 不会调用模型，先清理上一轮模型调用的计时，避免旧 TTFT 泄漏到本轮。
        client_timing = getattr(self.client, "last_timing", None)
        if isinstance(client_timing, dict):
            client_timing.clear()
        client_meta = getattr(self.client, "last_meta", None)
        if isinstance(client_meta, dict):
            client_meta.clear()
        self.last_runtime_meta = {}

        def finish(decision: RouteDecision, *, path: str) -> RouteDecision:
            if timings is not None:
                timings["router_total_ms"] = int((time.perf_counter() - started) * 1000)
                client_timing = getattr(self.client, "last_timing", {}) or {}
                for key in (
                    "router_queue_ms",
                    "router_ttft_ms",
                    "router_generation_ms",
                    "router_http_ms",
                    "router_parse_ms",
                    "router_attempt_count",
                    "router_model_calls",
                ):
                    if key in client_timing and client_timing[key] is not None:
                        timings[key] = int(client_timing[key])
            self.last_runtime_meta = {
                "router_path": path,
                "router_source": decision.source,
            }
            meta = getattr(self.client, "last_meta", None)
            if isinstance(meta, dict):
                self.last_runtime_meta.update({str(key): value for key, value in meta.items()})
            return decision

        # 明确问候语无需启动 7B 路由模型，直接走闲聊分支。
        if _is_obvious_casual(question) and _supports_fast_path(self.client):
            return finish(fallback_route(question, has_image=has_image, context=context), path="fast_path")
        # 对明确的单跳事实/位置问法使用保守规则，避免每个简单问题都等待 7B Router。
        # 复杂比较、解释和长问题仍交给决策模型，保持 Router 的扩展能力。
        if _is_fast_rule_question(question) and _supports_fast_path(self.client):
            return finish(fallback_route(question, has_image=has_image, context=context), path="fast_path")
        if self.client is None:
            return finish(fallback_route(question, has_image=has_image, context=context), path="fallback")
        try:
            state = {
                    "question": question,
                    "has_image": has_image,
                    "target_modality": target_modality or "TEXT",
                    "domain_id": context.domain_id,
                    "current_node_id": context.current_node_id,
                    "location_node_id": context.location_node_id,
                    "last_entities": context.last_entities,
                    "pending_candidates": context.pending_candidates,
                    "session_summary": context.session_summary,
                    "last_question": context.last_question,
                    "recent_messages": context.recent_messages[-8:],
                    "pending_clarification": context.pending_clarification,
                }
            raw = self.client.decide("router", state) if hasattr(self.client, "decide") else self.client(state=state)
            payload = _as_dict(raw)
            source = getattr(self.client, "provider", "jev")
            return finish(normalize_route_decision(
                payload,
                question=question,
                has_image=has_image,
                target_modality=target_modality,
                context=context,
                source=source,
            ), path="decision_model")
        except Exception:
            return finish(fallback_route(question, has_image=has_image, context=context), path="fallback")


def _is_obvious_casual(question: str) -> bool:
    compact = re.sub(r"[\s，。！？?；;、]+", "", str(question or "")).casefold()
    if compact in {"你好", "您好", "嗨", "hello", "hi", "谢谢", "感谢", "再见", "你是谁", "你叫什么"}:
        return True
    return bool(re.search(r"^(?:我|今天|最近)?(?:好|有点|太|真的)?(?:累|困|疲惫|难过|开心|郁闷|烦|焦虑|紧张)(?:啊|呀|了|呢)?$", compact)) or any(marker in compact for marker in (
        "讲个笑话", "说个笑话", "讲笑话", "说笑话", "讲个故事", "说个故事",
        "陪我聊聊", "聊聊天", "逗我一下", "你会唱歌吗", "唱首歌",
    )) or compact in {"无聊", "有点无聊", "随便说点什么"}


def _is_default_local_client(client: Any) -> bool:
    # 测试 transport、函数式注入和托管 Jev 都必须保留原有调用行为。
    return (
        client is not None
        and getattr(client, "provider", "") == "local"
        and getattr(client, "transport", None) is None
    )


def _supports_fast_path(client: Any) -> bool:
    """生产本地级联和无模型降级可走规则；测试注入仍保留原有调用契约。"""
    return client is None or bool(getattr(client, "supports_fast_path", False)) or _is_default_local_client(client)


def _is_fast_rule_question(question: str) -> bool:
    text = str(question or "").strip()
    compact = re.sub(r"[\s，。！？?；;、]+", "", text)
    if not text or len(text) > 80:
        return False
    if any(token in text for token in ("比较", "区别", "差异", "为什么", "原因", "详细介绍", "分析一下", "是否", "能不能")):
        return False
    if can_fast_parse(text):
        return True
    return any(token in compact for token in (
        "面积", "多大", "多少平方米",
        "属于", "上级", "下级", "组成", "包含", "有几个", "哪些", "路线", "怎么去",
    ))


def fallback_route(question: str, *, has_image: bool, context: ContextSnapshot) -> RouteDecision:
    return normalize_route_decision(
        _fallback_payload(question, has_image=has_image, context=context),
        question=question,
        has_image=has_image,
        target_modality=None,
        context=context,
        source="fallback",
    )


def _fallback_payload(question: str, *, has_image: bool, context: ContextSnapshot) -> dict[str, Any]:
    text = str(question or "").strip()
    compact = re.sub(r"[\s，。！？?；;、]+", "", text).casefold()
    if _is_obvious_casual(text):
        return {"intent": "CASUAL", "confidence": 0.98}
    if context.pending_candidates and _looks_like_clarification(text):
        return {
            "intent": "UNKNOWN",
            "context_dependency": "PENDING_CLARIFICATION",
            "needs_clarification": True,
            "confidence": 0.86,
        }
    if re.search(r"^(?:我|这里|此处|当前位置|我现在|我目前).*(?:在哪|哪里|什么位置)", compact):
        return {
            "intent": "LOCATION",
            "context_dependency": "LOCATION",
            "confidence": 0.96,
        }
    if has_image or any(token in text for token in ("图片", "照片", "图像", "配图")):
        return {"intent": "IMAGE", "context_dependency": "IMAGE", "confidence": 0.90}
    if any(token in text for token in ("长啥样", "长什么样", "什么样子", "外观", "样貌", "现场图", "给我看看")):
        return {"intent": "IMAGE", "context_dependency": "NONE", "confidence": 0.90}
    if any(token in text for token in ("附近有什么", "附近有", "周边有什么", "周围有什么")):
        return {"intent": "LIST", "context_dependency": "NONE", "confidence": 0.90}
    if any(token in text for token in ("从", "到", "怎么去", "如何到达", "路线", "经过")):
        return {"intent": "PATH", "confidence": 0.80}
    if any(token in text for token in ("哪些", "有几个", "列出", "分别")):
        hint = "LIST"
    elif any(token in text for token in ("属于", "组成", "包含", "上级", "下级")):
        hint = "HIERARCHY"
    elif any(token in text for token in ("哪里", "在哪", "位于", "地址", "附近")):
        hint = "LOCATION"
    elif any(token in text for token in ("为什么", "原因", "作用", "意义", "详细介绍", "背景", "原理")):
        hint = "EXPLAIN"
    else:
        hint = "DIRECT_FACT"
    return {"intent": hint, "confidence": 0.70}


def _looks_like_clarification(text: str) -> bool:
    return bool(re.search(
        r"(?:第?\d+个|第?[一二三四五六七八九十]+个|[一二三四五六七八九十两三四五六七八九十]+楼|"
        r"[一二三四五六七八九十两]+层|\d+楼|\d+层|这个|那个|左边|右边)",
        text,
    ))


def normalize_route_decision(
    value: dict[str, Any],
    *,
    question: str,
    has_image: bool,
    target_modality: str | None,
    context: ContextSnapshot,
    source: str = "fallback",
) -> RouteDecision:
    intents = {"CASUAL", "DIRECT_FACT", "LOCATION", "HIERARCHY", "PATH", "IMAGE", "COMPARE", "EXPLAIN", "LIST", "UNKNOWN"}
    intent = str(value.get("intent") or "UNKNOWN").upper()
    if intent not in intents:
        intent = "UNKNOWN"
    dependency = str(value.get("context_dependency") or "").upper()
    if dependency not in {"NONE", "SESSION", "LOCATION", "PENDING_CLARIFICATION", "IMAGE"}:
        dependency = "NONE"
    if context.pending_candidates and _looks_like_clarification(question):
        dependency = "PENDING_CLARIFICATION"
    text = str(question or "")
    compact = re.sub(r"[\s，。！？?；;、]+", "", text)
    needs_current_location = dependency == "LOCATION" or bool(
        re.search(r"^(?:我|这里|此处|当前位置|我现在|我目前).*(?:在哪|哪里|什么位置)", compact)
    )
    if intent == "CASUAL":
        route = "CASUAL"
    elif dependency == "PENDING_CLARIFICATION":
        route = "CLARIFICATION"
    else:
        route = "KNOWLEDGE"
    is_casual = route == "CASUAL"
    needs_image = (not is_casual) and (
        intent == "IMAGE" or has_image or str(target_modality or "").upper() == "IMAGE"
        or any(token in text for token in ("图片", "照片", "图像", "配图", "长什么样", "长啥样", "外观", "现场图"))
    )
    needs_path = (not is_casual) and intent == "PATH"
    needs_graph = route == "KNOWLEDGE" and intent != "CASUAL"
    needs_text = (not is_casual) and intent in {"EXPLAIN", "COMPARE"}
    return RouteDecision(
        route=route,
        intent=intent,
        confidence=_clamp(value.get("confidence", 0.0)),
        context_dependency="NONE" if is_casual else dependency,
        needs_graph=needs_graph,
        needs_text=needs_text,
        needs_image=needs_image,
        needs_path=needs_path,
        context_dependent=(not is_casual and dependency != "NONE"),
        needs_current_location=(not is_casual and needs_current_location),
        needs_clarification=(route == "CLARIFICATION"),
        source=source if source in {"jev", "local", "fallback"} else "jev",
    )


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError("Router client must return a dict, JSON object, or Pydantic model")


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0
