"""QA v2 的四层记忆管理器。

提供工作记忆、会话消息、实体状态、会话摘要和长期语义记忆的最小读写边界。
具体 Redis/PostgreSQL 适配器由主应用注入，本模块不直接连接数据库。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel, Field

from ..models import ContextSnapshot
from src.kg_qa_shared.budget import ContextBudgetManager
from .memory import ContextResolver, MemoryContext, build_summary_contract


class MemoryStore(Protocol):
    """生产存储层需要实现的窄接口。"""

    def load_session(self, *, session_id: str, user_id: Optional[str], domain_id: str) -> dict[str, Any]:
        ...

    def save_session(self, *, session_id: str, user_id: Optional[str], domain_id: str, value: dict[str, Any]) -> None:
        ...

    def search_long_term(self, *, user_id: str, domain_id: str, query: str, limit: int) -> list[dict[str, Any]]:
        ...


class ConversationMessage(BaseModel):
    role: str
    content: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MemoryUpdateResult(BaseModel):
    session_id: str
    message_count: int
    compressed: bool = False
    removed_message_count: int = 0
    summary: Optional[str] = None
    display_event: Optional[dict[str, Any]] = None


class MemoryManager:
    """实现按需读取和阈值压缩，不把所有历史消息塞给模型。"""

    def __init__(
        self,
        *,
        store: MemoryStore,
        max_messages: int = 12,
        max_chars: int = 12000,
        keep_recent_messages: int = 6,
        summarizer: Optional[Callable[[list[ConversationMessage]], str]] = None,
        budget_manager: Optional[ContextBudgetManager] = None,
    ) -> None:
        self.store = store
        self.max_messages = max(2, int(max_messages))
        self.max_chars = max(1000, int(max_chars))
        self.keep_recent_messages = max(2, min(int(keep_recent_messages), self.max_messages))
        self.summarizer = summarizer
        self.budget_manager = budget_manager or ContextBudgetManager.from_env()

    def resolve(
        self,
        *,
        question: str,
        domain_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        current_location: Optional[dict[str, Any]] = None,
        has_image: bool = False,
        need_long_term: bool = False,
        long_term_query: Optional[str] = None,
    ) -> MemoryContext:
        raw: dict[str, Any] = {}
        if session_id:
            raw = self.store.load_session(
                session_id=session_id,
                user_id=user_id,
                domain_id=domain_id,
            ) or {}
        resolver = ContextResolver(
            session_reader=lambda **_: raw,
            long_term_reader=lambda **kwargs: self.store.search_long_term(
                **kwargs,
                limit=10,
            ),
            budget_manager=self.budget_manager,
        )
        return resolver.resolve(
            question=question,
            domain_id=domain_id,
            session_id=session_id,
            user_id=user_id,
            request_id=request_id,
            current_location=current_location,
            has_image=has_image,
            need_long_term=need_long_term,
            long_term_query=long_term_query,
        )

    @staticmethod
    def to_snapshot(
        memory: MemoryContext,
        *,
        session_id: Optional[str],
        user_id: Optional[str],
    ) -> ContextSnapshot:
        return ContextResolver.to_snapshot(memory, session_id=session_id, user_id=user_id)

    def update_after_turn(
        self,
        *,
        session_id: str,
        user_id: Optional[str],
        domain_id: str,
        question: str,
        answer: str,
        entity_state: Optional[dict[str, Any]] = None,
    ) -> MemoryUpdateResult:
        raw = self.store.load_session(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
        ) or {}
        messages = [
            ConversationMessage.model_validate(item)
            for item in (raw.get("messages") or [])
            if isinstance(item, dict)
        ]
        messages.extend([
            ConversationMessage(role="user", content=str(question or "").strip()),
            ConversationMessage(role="assistant", content=str(answer or "").strip()),
        ])
        previous_count = len(messages)
        compressed = False
        removed = 0
        summary = ((raw.get("session_summary") or {}).get("summary"))
        if _needs_compression(messages, max_messages=self.max_messages, max_chars=self.max_chars):
            old_messages = messages[:-self.keep_recent_messages]
            messages = messages[-self.keep_recent_messages:]
            summary_text = self.summarizer(old_messages) if self.summarizer else _compact_summary(old_messages)
            previous_summary = dict(raw.get("session_summary") or {})
            structured_summary = build_summary_contract(
                old_messages,
                previous=previous_summary,
                summary=summary_text,
            )
            summary = structured_summary["summary"]
            compressed = True
            removed = len(old_messages)
        session_summary = dict(raw.get("session_summary") or {})
        if summary:
            session_summary["summary"] = summary
            session_summary["version"] = int(session_summary.get("version") or 0) + (1 if compressed else 0)
            if compressed:
                session_summary.update(structured_summary)
                session_summary["version"] = int(session_summary.get("version") or 0) + 1
        value = {
            **raw,
            "messages": [item.model_dump() for item in messages],
            "entity_state": entity_state or raw.get("entity_state") or {},
            "session_summary": session_summary,
        }
        self.store.save_session(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
            value=value,
        )
        display_event = None
        if compressed:
            display_event = {
                "event": "memory_compressed",
                "message": "为保持上下文连贯，小V已整理较早的对话记录。",
                "summary": summary or "",
                "removed_message_count": removed,
            }
        return MemoryUpdateResult(
            session_id=session_id,
            message_count=len(messages),
            compressed=compressed,
            removed_message_count=removed,
            summary=summary,
            display_event=display_event,
        )


def _needs_compression(
    messages: list[ConversationMessage],
    *,
    max_messages: int,
    max_chars: int,
) -> bool:
    return len(messages) > max_messages or sum(len(item.content) for item in messages) > max_chars


def _compact_summary(messages: list[ConversationMessage]) -> str:
    parts = []
    for item in messages:
        content = " ".join(item.content.split())
        if content:
            parts.append(f"{item.role}：{content[:180]}")
    compact = "；".join(parts)
    return ContextBudgetManager.from_env().truncate_text(compact, 640)[0]
