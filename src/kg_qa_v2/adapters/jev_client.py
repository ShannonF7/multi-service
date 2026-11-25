"""TypeSafe 官方 Python SDK 的 Jev 适配器。

业务层只消费归一化字典，SDK 对象和供应商响应结构隔离在本模块内。
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional


INTENTS = (
    "CASUAL", "DIRECT_FACT", "LOCATION", "HIERARCHY", "PATH",
    "IMAGE", "COMPARE", "EXPLAIN", "LIST", "UNKNOWN",
)


class TypeSafeJevClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-latest",
        timeout: float = 8.0,
        sdk_client: Any = None,
        sdk_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not api_key:
            raise ValueError("JEV_API_KEY or TYPESAFE_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._sdk_client = sdk_client
        self._sdk_factory = sdk_factory
        self.last_usage: Dict[str, Any] = {}

    def __call__(self, *, state: dict[str, Any], **_: Any) -> dict[str, Any]:
        client = self._get_sdk_client()
        response = client.system_one(
            state=state,
            model=self.model,
            questions=self.questions(),
        )
        normalized = self.normalize(response)
        self.last_usage = normalized.pop("usage", {})
        return normalized

    def decide(self, task: str, payload: dict[str, Any]) -> dict[str, Any]:
        """按统一 DecisionModel 协议执行路由、对齐或引用校验。"""
        if task == "router":
            return self(state=payload)
        client = self._get_sdk_client()
        response = client.system_one(
            state=payload,
            model=self.model,
            questions=self.questions_for_task(task, payload),
        )
        result = self.normalize_task(response, task)
        self.last_usage = result.pop("usage", {})
        return result

    def _get_sdk_client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        factory = self._sdk_factory
        if factory is None:
            try:
                from typesafe_sdk import TypeSafeClient
            except ImportError as exc:
                raise RuntimeError("未安装官方 typesafe-sdk，请安装后再启用 Jev") from exc
            factory = TypeSafeClient
        self._sdk_client = factory(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
        )
        return self._sdk_client

    def close(self) -> None:
        if self._sdk_client is not None and hasattr(self._sdk_client, "close"):
            self._sdk_client.close()

    @staticmethod
    def questions() -> dict[str, dict[str, Any]]:
        return {
            "intent": {
                "type": "choice",
                "instructions": "选择用户问题的唯一主意图，不要根据检索结果改变意图。",
                "criteria": {intent: intent for intent in INTENTS},
            },
            "context_dependency": {
                "type": "choice",
                "instructions": "选择当前问题对上下文的唯一依赖类型。",
                "criteria": {
                    "NONE": "不依赖会话上下文",
                    "SESSION": "依赖上一轮实体、代词、简称或省略",
                    "LOCATION": "依赖用户当前定位",
                    "PENDING_CLARIFICATION": "是在回答待确认候选",
                    "IMAGE": "主要依赖本轮上传图片",
                },
            },
        }

    @classmethod
    def questions_for_task(cls, task: str, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if task == "entity_alignment":
            candidates = payload.get("candidates") or []
            criteria = {str(item.get("node_id")): str(item.get("name") or item.get("node_id")) for item in candidates}
            criteria["UNKNOWN"] = "上下文不足，无法确定"
            return {
                "selected_node_id": {
                    "type": "choice",
                    "instructions": "只能选择给定候选的 node_id；无法确定时选择 UNKNOWN。",
                    "criteria": criteria,
                }
            }
        if task == "citation_validation":
            return {
                "status": {
                    "type": "choice",
                    "instructions": "判断答案句是否被给定证据支持。",
                    "criteria": {
                        "ENTAILED": "证据充分支持",
                        "PARTIAL": "证据仅部分支持",
                        "NOT_ENTAILED": "证据不支持",
                    },
                }
            }
        raise ValueError(f"不支持的 Jev 决策任务：{task}")

    @classmethod
    def normalize_task(cls, response: Any, task: str) -> dict[str, Any]:
        choices = getattr(response, "choices", {}) or {}
        if isinstance(response, dict):
            answers = response.get("answers") or {}
            answer = answers.get("selected_node_id" if task == "entity_alignment" else "status") or {}
            usage = dict(response.get("usage") or {})
            value = answer.get("choice") or "UNKNOWN"
            confidence = float(answer.get("confidence") or 0.0)
        else:
            key = "selected_node_id" if task == "entity_alignment" else "status"
            answer = choices.get(key)
            value = getattr(answer, "choice", "UNKNOWN") or "UNKNOWN"
            confidence = float(getattr(answer, "confidence", 0.0) or 0.0)
            usage_obj = getattr(response, "usage", None)
            usage = {
                key: value for key, value in {
                    "input_tokens": getattr(usage_obj, "input_tokens", None),
                    "output_tokens": getattr(usage_obj, "output_tokens", None),
                }.items() if value is not None
            }
        if task == "entity_alignment":
            return {"selected_node_id": str(value), "confidence": max(0.0, min(confidence, 1.0)), "usage": usage}
        status = str(value).upper()
        if status not in {"ENTAILED", "PARTIAL", "NOT_ENTAILED"}:
            status = "NOT_ENTAILED"
        return {"status": status, "confidence": max(0.0, min(confidence, 1.0)), "usage": usage}

    @classmethod
    def normalize(cls, response: Any) -> dict[str, Any]:
        """兼容官方 SDK typed response 和测试用原始字典。"""
        if isinstance(response, dict):
            return cls._normalize_dict(response)
        choices = getattr(response, "choices", {}) or {}
        nouls = getattr(response, "nouls", {}) or {}
        intent_answer = choices.get("intent")
        intent = str(getattr(intent_answer, "choice", "UNKNOWN") or "UNKNOWN").upper()
        if intent not in INTENTS:
            intent = "UNKNOWN"
        confidence = float(getattr(intent_answer, "confidence", 0.0) or 0.0)
        dependency_answer = choices.get("context_dependency")
        dependency = str(getattr(dependency_answer, "choice", "NONE") or "NONE").upper()

        def noul(name: str) -> float:
            answer = nouls.get(name)
            return float(getattr(answer, "noul", 0.0) or 0.0)

        usage = getattr(response, "usage", None)
        usage_value = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        } if usage is not None else {}
        return cls._build_decision(
            intent=intent,
            context_dependency=dependency,
            confidence=confidence,
            usage={key: value for key, value in usage_value.items() if value is not None},
        )

    @classmethod
    def _normalize_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        answers = data.get("answers") or {}
        intent_answer = answers.get("intent") or {}
        intent = str(intent_answer.get("choice") or "UNKNOWN").upper()
        if intent not in INTENTS:
            intent = "UNKNOWN"
        dependency = str((answers.get("context_dependency") or {}).get("choice") or "NONE").upper()
        return cls._build_decision(
            intent=intent,
            context_dependency=dependency,
            confidence=float(intent_answer.get("confidence") or 0.0),
            usage=dict(data.get("usage") or {}),
        )

    @staticmethod
    def _build_decision(*, intent: str, context_dependency: str, confidence: float, usage: dict[str, Any]) -> dict[str, Any]:
        if context_dependency not in {"NONE", "SESSION", "LOCATION", "PENDING_CLARIFICATION", "IMAGE"}:
            context_dependency = "NONE"
        return {
            "intent": intent,
            "confidence": max(0.0, min(confidence, 1.0)),
            "context_dependency": "NONE" if intent == "CASUAL" else context_dependency,
            "usage": usage,
        }


def build_jev_client_from_env() -> Optional[TypeSafeJevClient]:
    api_key = os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        return None
    try:
        timeout = float(os.getenv("JEV_TIMEOUT", "8"))
    except ValueError:
        timeout = 8.0
    return TypeSafeJevClient(
        api_key=api_key,
        base_url=os.getenv("JEV_BASE_URL", os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai")),
        model=os.getenv("JEV_MODEL", os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-latest")),
        timeout=timeout,
    )
