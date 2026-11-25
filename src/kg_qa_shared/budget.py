"""QA v2 的上下文预算工具。"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field


class BudgetSection(BaseModel):
    name: str
    text: str = ""
    priority: int = 0
    required: bool = False
    max_tokens: Optional[int] = None


class BudgetResult(BaseModel):
    text: str = ""
    selected: dict[str, str] = Field(default_factory=dict)
    used_tokens: int = 0
    budget_tokens: int = 0
    truncated: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)


class BudgetedHistory(BaseModel):
    summary: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    used_tokens: int = 0
    budget_tokens: int = 0
    truncated: list[str] = Field(default_factory=list)
    dropped_message_count: int = 0
    summary_truncated: bool = False


class ContextBudgetManager:
    """为问题、会话历史和摘要分配输入预算。

    max_context_tokens 是模型总窗口；reserved_output_tokens 为答案预留；
    safety_tokens 用于系统提示和序列化开销。可注入真实 tokenizer。
    """

    def __init__(
        self,
        *,
        max_context_tokens: int = 2048,
        reserved_output_tokens: int = 256,
        safety_tokens: int = 64,
        tokenizer: Any = None,
    ) -> None:
        self.max_context_tokens = max(256, int(max_context_tokens))
        self.reserved_output_tokens = max(0, int(reserved_output_tokens))
        self.safety_tokens = max(0, int(safety_tokens))
        self.tokenizer = tokenizer

    @classmethod
    def from_env(cls) -> "ContextBudgetManager":
        return cls(
            max_context_tokens=int(os.getenv("KG_QA_CONTEXT_MAX_TOKENS", "2048")),
            reserved_output_tokens=int(os.getenv("KG_QA_CONTEXT_RESERVED_OUTPUT_TOKENS", "256")),
            safety_tokens=int(os.getenv("KG_QA_CONTEXT_SAFETY_TOKENS", "64")),
        )

    @property
    def input_budget_tokens(self) -> int:
        return max(64, self.max_context_tokens - self.reserved_output_tokens - self.safety_tokens)

    def estimate_tokens(self, value: Any) -> int:
        text = str(value or "")
        if not text:
            return 0
        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer(text, add_special_tokens=False)
                input_ids = encoded.get("input_ids", encoded) if isinstance(encoded, dict) else encoded
                return max(1, len(input_ids))
            except Exception:
                pass
        non_ascii = sum(1 for char in text if ord(char) >= 0x2E80)
        return max(1, non_ascii + math.ceil((len(text) - non_ascii) / 4))

    def truncate_text(self, value: Any, max_tokens: int) -> tuple[str, bool]:
        text = str(value or "")
        limit = max(0, int(max_tokens))
        if not text or self.estimate_tokens(text) <= limit:
            return text, False
        if limit <= 2:
            return text[:limit], True
        marker = "…（已按上下文预算压缩）…"
        content_budget = max(1, limit - self.estimate_tokens(marker))
        head_budget = max(1, int(content_budget * 0.65))
        head = self._take_tokens(text, head_budget, from_end=False)
        tail = self._take_tokens(text, max(1, content_budget - head_budget), from_end=True)
        return f"{head}{marker}{tail}", True

    def pack(
        self,
        sections: Iterable[BudgetSection],
        *,
        extra_reserved_tokens: int = 0,
    ) -> BudgetResult:
        values = list(sections)
        budget = max(1, self.input_budget_tokens - max(0, int(extra_reserved_tokens)))
        selected: dict[str, str] = {}
        truncated: list[str] = []
        dropped: list[str] = []
        remaining = budget
        ranked = sorted(
            enumerate(values),
            key=lambda item: (not item[1].required, -int(item[1].priority), item[0]),
        )
        for _, section in ranked:
            if not section.text:
                continue
            full_tokens = self.estimate_tokens(section.text)
            allowed = remaining
            if section.max_tokens is not None:
                allowed = min(allowed, max(0, int(section.max_tokens)))
            if full_tokens <= allowed:
                selected[section.name] = section.text
                remaining -= full_tokens
                continue
            if allowed > 0:
                clipped, was_clipped = self.truncate_text(section.text, allowed)
                if clipped:
                    selected[section.name] = clipped
                    remaining = max(0, remaining - self.estimate_tokens(clipped))
                    if was_clipped:
                        truncated.append(section.name)
                    continue
            dropped.append(section.name)
        ordered_text = [selected[item.name] for item in values if item.name in selected]
        return BudgetResult(
            text="\n\n".join(ordered_text),
            selected=selected,
            used_tokens=max(0, budget - remaining),
            budget_tokens=budget,
            truncated=truncated,
            dropped=dropped,
        )

    def fit_history(
        self,
        *,
        question: str,
        summary: str = "",
        messages: Iterable[dict[str, Any]] = (),
        pending: str = "",
    ) -> BudgetedHistory:
        values = [item for item in messages if isinstance(item, dict)]
        sections: list[BudgetSection] = []
        if summary:
            sections.append(BudgetSection(
                name="summary", text=summary, priority=20, required=True, max_tokens=256,
            ))
        if pending:
            sections.append(BudgetSection(
                name="pending", text=pending, priority=90, required=True, max_tokens=160,
            ))
        for index, item in enumerate(values):
            content = str(item.get("content") or "")
            if not content:
                continue
            role = str(item.get("role") or "")
            sections.append(BudgetSection(
                name=f"message:{index}",
                text=f"{role}：{content}" if role else content,
                priority=100 + index,
                required=index == len(values) - 1,
            ))
        result = self.pack(sections, extra_reserved_tokens=self.estimate_tokens(question))
        kept_messages = [item for index, item in enumerate(values) if f"message:{index}" in result.selected]
        return BudgetedHistory(
            summary=result.selected.get("summary", ""),
            messages=kept_messages,
            used_tokens=result.used_tokens,
            budget_tokens=result.budget_tokens,
            truncated=result.truncated,
            dropped_message_count=max(0, len(values) - len(kept_messages)),
            summary_truncated="summary" in result.truncated,
        )

    def _take_tokens(self, text: str, limit: int, *, from_end: bool) -> str:
        chars = reversed(text) if from_end else iter(text)
        output: list[str] = []
        used = 0.0
        for char in chars:
            cost = 1.0 if ord(char) >= 0x2E80 else 0.25
            if used + cost > max(0, limit):
                break
            output.append(char)
            used += cost
        value = "".join(output)
        return value[::-1] if from_end else value
