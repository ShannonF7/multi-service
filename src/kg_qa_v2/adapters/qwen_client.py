"""Qwen Structured Output 的 DashScope 客户端。"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from ..core.prompts import build_slot_system_prompt


class QwenStructuredSlotClient:
    def __init__(self, *, api_key: str, model: str = "qwen-flash", generation_call: Optional[Callable[..., Any]] = None) -> None:
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY or LLM_API_KEY is required")
        self.api_key = api_key
        self.model = model
        self.generation_call = generation_call

    def __call__(self, *, prompt: str, response_format: str = "json", **_: Any) -> dict[str, Any]:
        if response_format != "json":
            raise ValueError("Qwen slot client only accepts JSON response format")
        call = self.generation_call
        if call is None:
            from dashscope import Generation
            call = Generation.call
        response = call(
            api_key=self.api_key,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": build_slot_system_prompt(),
                },
                {"role": "user", "content": prompt},
            ],
            result_format="message",
            response_format={"type": "json_object"},
            enable_search=False,
        )
        content = self._extract_content(response)
        value = json.loads(content)
        if not isinstance(value, dict):
            raise TypeError("Qwen structured slot response must be an object")
        return value

    @staticmethod
    def _extract_content(response: Any) -> str:
        output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
        choices = output.get("choices") if isinstance(output, dict) else getattr(output, "choices", None)
        first = (choices or [None])[0]
        message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content).strip()
        return str(content or "").strip()


def build_qwen_slot_client_from_env() -> Optional[QwenStructuredSlotClient]:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    return QwenStructuredSlotClient(
        api_key=api_key,
        model=os.getenv("KG_QA_QWEN_SLOT_MODEL", os.getenv("KG_QA_LLM_MODEL", "qwen-flash")),
    )
