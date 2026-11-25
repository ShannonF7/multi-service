"""四层记忆的读取边界与 Context Resolver。

本模块只负责按需读取和组装上下文，不负责摘要模型、数据库写入或检索答案。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from src.kg_qa_shared.budget import ContextBudgetManager
from src.kg_qa_shared.memory import build_summary_contract
from ..models import ContextSnapshot


class WorkingMemory(BaseModel):
    request_id: Optional[str] = None
    question: str = ""
    domain_id: str = ""
    current_location: Optional[Dict[str, Any]] = None
    has_image: bool = False


class EntityState(BaseModel):
    active_entity: Optional[str] = None
    active_node_id: Optional[str] = None
    active_domain_id: Optional[str] = None
    active_hierarchy_path: List[str] = Field(default_factory=list)
    active_node_type: Optional[str] = None
    last_focus: Optional[str] = None
    updated_at: Optional[str] = None
    recent_entities: List[str] = Field(default_factory=list)
    pending_candidates: List[Dict[str, Any]] = Field(default_factory=list)


class SessionSummary(BaseModel):
    summary: Optional[str] = None
    goal: Optional[str] = None
    confirmed_entities: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    unresolved_items: List[str] = Field(default_factory=list)
    important_facts: List[str] = Field(default_factory=list)
    version: int = 0


class LongTermMemoryItem(BaseModel):
    memory_id: str
    content: str
    memory_type: str = "semantic"
    domain_id: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MemoryContext(BaseModel):
    working: WorkingMemory
    entity_state: EntityState = Field(default_factory=EntityState)
    session_summary: SessionSummary = Field(default_factory=SessionSummary)
    long_term: List[LongTermMemoryItem] = Field(default_factory=list)
    last_question: Optional[str] = None
    recent_messages: List[Dict[str, Any]] = Field(default_factory=list)
    pending_clarification: Dict[str, Any] = Field(default_factory=dict)
    budget_report: Dict[str, Any] = Field(default_factory=dict)


class ContextResolver:
    """把存储层数据裁剪为本轮需要的上下文。"""

    def __init__(
        self,
        *,
        session_reader: Optional[Callable[..., Dict[str, Any]]] = None,
        long_term_reader: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        budget_manager: Optional[ContextBudgetManager] = None,
    ) -> None:
        self.session_reader = session_reader
        self.long_term_reader = long_term_reader
        self.budget_manager = budget_manager or ContextBudgetManager.from_env()

    def resolve(
        self,
        *,
        question: str,
        domain_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        current_location: Optional[Dict[str, Any]] = None,
        has_image: bool = False,
        need_long_term: bool = False,
        long_term_query: Optional[str] = None,
    ) -> MemoryContext:
        raw = self.session_reader(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
        ) if self.session_reader and session_id else {}
        raw = raw or {}
        entity_state = EntityState.model_validate(raw.get("entity_state") or {})
        pending = dict(raw.get("pending_clarification") or {})
        if pending and not entity_state.pending_candidates:
            entity_state.pending_candidates = list(pending.get("pending_candidates") or [])
        summary_raw = raw.get("session_summary") or {}
        summary = (
            SessionSummary(summary=summary_raw)
            if isinstance(summary_raw, str)
            else SessionSummary.model_validate(summary_raw)
        )
        recent_messages = [
            item for item in list(raw.get("recent_messages") or raw.get("messages") or [])
            if isinstance(item, dict)
        ]
        pending_text = ""
        if pending:
            pending_text = "待确认对象：" + "、".join(
                str(item.get("name") or item.get("label") or "")
                for item in entity_state.pending_candidates
                if isinstance(item, dict)
            )
        budgeted = self.budget_manager.fit_history(
            question=question,
            summary=summary.summary or "",
            messages=recent_messages,
            pending=pending_text,
        )
        if budgeted.summary != (summary.summary or ""):
            summary.summary = budgeted.summary or None
        long_term: List[LongTermMemoryItem] = []
        if need_long_term and self.long_term_reader and user_id:
            values = self.long_term_reader(
                user_id=user_id,
                domain_id=domain_id,
                query=long_term_query or question,
            ) or []
            long_term = [
                LongTermMemoryItem.model_validate(item)
                for item in values
                if not item.get("domain_id") or item.get("domain_id") == domain_id
            ]
        working = WorkingMemory(
            request_id=request_id,
            question=question,
            domain_id=domain_id,
            current_location=current_location,
            has_image=has_image,
        )
        return MemoryContext(
            working=working,
            entity_state=entity_state,
            session_summary=summary,
            long_term=long_term,
            last_question=str(raw.get("last_question") or "") or None,
            recent_messages=budgeted.messages,
            pending_clarification=pending,
            budget_report={
                "used_tokens": budgeted.used_tokens,
                "budget_tokens": budgeted.budget_tokens,
                "dropped_message_count": budgeted.dropped_message_count,
                "truncated": list(budgeted.truncated),
                "summary_truncated": budgeted.summary_truncated,
            },
        )

    @staticmethod
    def to_snapshot(memory: MemoryContext, *, session_id: Optional[str], user_id: Optional[str]) -> ContextSnapshot:
        """只把问题理解所需字段暴露给后续模块。"""
        location = memory.working.current_location or {}
        return ContextSnapshot(
            domain_id=memory.working.domain_id,
            session_id=session_id,
            user_id=user_id,
            current_node_id=memory.entity_state.active_node_id,
            current_node_name=memory.entity_state.active_entity,
            location_node_id=location.get("node_id"),
            location_node_name=location.get("name"),
            location_in_domain=location.get("location_in_domain"),
            location_domain_name=location.get("domain_name"),
            distance_to_domain_m=location.get("distance_to_domain_m"),
            domain_scope_radius_m=location.get("domain_scope_radius_m"),
            last_entities=list(dict.fromkeys([
                *memory.entity_state.recent_entities,
                *memory.session_summary.confirmed_entities,
            ]))[:5],
            pending_candidates=memory.entity_state.pending_candidates,
            pending_clarification=memory.pending_clarification,
            long_term_memory=[item.model_dump() for item in memory.long_term],
            recent_messages=list(memory.recent_messages),
            last_question=memory.last_question,
            session_summary=memory.session_summary.summary,
        )

