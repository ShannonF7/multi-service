"""KG-QA V1 与 V2 共用的短期会话、摘要和长期画像记忆。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from threading import RLock
from typing import Any

from src.kg_qa_shared.memory import build_summary_contract


@dataclass
class ConversationContext:
    """一次问答会话的内存上下文。

    输入：会话标识、摘要、消息轮次、用户画像和澄清状态。
    输出：供问答服务读取或更新的会话上下文对象。
    逻辑：统一承载单轮问答需要的会话状态，并可转换为兼容字典。
    """
    session_id: str
    user_id: str = ""
    domain_id: str = ""
    summary: str = ""
    summary_data: dict[str, Any] | None = None
    long_term_memory: dict[str, Any] | None = None
    turns: list[dict[str, Any]] | None = None
    needs_compression: bool = False
    compressed: bool = False
    pending_clarification: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.turns is None:
            self.turns = []
        if self.long_term_memory is None:
            self.long_term_memory = {}
        if self.summary_data is None:
            self.summary_data = {}

    def to_dict(self) -> dict[str, Any]:
        """将会话对象转换为供上下文读取器消费的字典。

        输入：当前会话上下文。
        输出：包含摘要、记忆、消息和澄清状态的字典。
        逻辑：把问答轮次展开为标准 role/content 消息列表。
        """
        messages: list[dict[str, str]] = []
        for turn in self.turns or []:
            if turn.get("user"):
                messages.append({"role": "user", "content": str(turn["user"])})
            if turn.get("assistant"):
                messages.append({"role": "assistant", "content": str(turn["assistant"])})
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "domain_id": self.domain_id,
            "summary": self.summary,
            "summary_data": dict(self.summary_data or {}),
            "long_term_memory": dict(self.long_term_memory or {}),
            "turns": list(self.turns or []),
            "messages": messages,
            "compressed": self.compressed,
            "pending_clarification": dict(self.pending_clarification or {}) if self.pending_clarification else None,
        }


class ConversationSessionStore:
    """管理短期对话窗口、历史摘要和长期画像。

    输入：会话标识及问答轮次、实体和澄清状态。
    输出：会话上下文，并在配置启用时持久化到数据库。
    逻辑：保留近期轮次，压缩较早对话并更新用户画像。
    """

    def __init__(self, *, max_turns: int | None = None, keep_turns: int | None = None) -> None:
        configured_max = int(os.getenv("KG_QA_SESSION_MAX_TURNS", "8"))
        self.max_turns = max(2, max_turns or configured_max)
        self.keep_turns = max(1, min(keep_turns or self.max_turns // 2, self.max_turns - 1))
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    @staticmethod
    def _key(session_id: str, user_id: str = "", domain_id: str = "") -> str:
        return "|".join((str(user_id or ""), str(domain_id or ""), str(session_id or "")))

    def open(
        self,
        session_id: str | None,
        *,
        user_id: str | None = None,
        domain_id: str | None = None,
    ) -> ConversationContext:
        key = self._key(str(session_id or "").strip(), str(user_id or "").strip(), str(domain_id or "").strip())
        if not str(session_id or "").strip():
            return ConversationContext(session_id="", user_id=str(user_id or ""), domain_id=str(domain_id or ""))
        with self._lock:
            state = self._sessions.setdefault(key, {
                "summary": "",
                "summary_data": {},
                "long_term_memory": {},
                "turns": [],
                "pending_clarification": None,
            })
            turns = list(state["turns"])
            return ConversationContext(
                session_id=str(session_id or "").strip(),
                user_id=str(user_id or "").strip(),
                domain_id=str(domain_id or "").strip(),
                summary=str(state.get("summary") or ""),
                summary_data=dict(state.get("summary_data") or {}),
                long_term_memory=dict(state.get("long_term_memory") or {}),
                turns=turns,
                needs_compression=len(turns) >= self.max_turns,
                pending_clarification=(
                    dict(state.get("pending_clarification") or {})
                    if state.get("pending_clarification") else None
                ),
            )

    def _update_profile(self, state: dict[str, Any], entities: list[Any]) -> None:
        profile = dict(state.get("long_term_memory") or {})
        known = list(profile.get("known_entities") or [])
        for item in entities:
            value = item.get("name") if isinstance(item, dict) else item
            value = str(value or "").strip()
            if value and value not in known:
                known.append(value)
        profile["known_entities"] = known[-30:]
        profile["turn_count"] = int(profile.get("turn_count") or 0) + 1
        if known:
            profile["last_entity"] = known[-1]
        state["long_term_memory"] = profile

    def compress(self, context: ConversationContext) -> ConversationContext:
        if not context.session_id or not context.needs_compression:
            return context
        key = self._key(context.session_id, context.user_id, context.domain_id)
        with self._lock:
            state = self._sessions.setdefault(key, {
                "summary": "",
                "summary_data": {},
                "long_term_memory": {},
                "turns": [],
                "pending_clarification": None,
            })
            turns = list(state["turns"])
            old_turns = turns[:-self.keep_turns]
            compact_lines = [
                f"用户：{item.get('user', '')}\n小V：{item.get('assistant', '')}"
                for item in old_turns
            ]
            previous = str(state.get("summary") or "").strip()
            compact = "\n".join(line for line in compact_lines if line.strip())
            summary_messages = []
            for item in old_turns:
                if item.get("user"):
                    summary_messages.append({"role": "user", "content": str(item.get("user"))})
                if item.get("assistant"):
                    summary_messages.append({"role": "assistant", "content": str(item.get("assistant"))})
            summary_data = build_summary_contract(
                summary_messages,
                previous=dict(state.get("summary_data") or {}),
                summary="\n".join(value for value in (previous, compact) if value),
            )
            state["summary_data"] = summary_data
            state["summary"] = str(summary_data.get("summary") or "")
            state["turns"] = turns[-self.keep_turns:]
            return ConversationContext(
                session_id=context.session_id,
                user_id=context.user_id,
                domain_id=context.domain_id,
                summary=state["summary"],
                summary_data=dict(state.get("summary_data") or {}),
                long_term_memory=dict(state.get("long_term_memory") or {}),
                turns=list(state["turns"]),
                compressed=True,
                pending_clarification=(
                    dict(state.get("pending_clarification") or {})
                    if state.get("pending_clarification") else None
                ),
            )

    def append(
        self,
        context: ConversationContext,
        user: str,
        assistant: str,
        entities: list[Any] | None = None,
        active_entity: dict[str, Any] | None = None,
    ) -> None:
        if not context.session_id:
            return
        names = []
        entity_states = []
        for item in entities or []:
            if isinstance(item, dict):
                entity_states.append(dict(item))
                value = item.get("name")
            else:
                value = item
            value = str(value or "").strip()
            if value and value not in names:
                names.append(value)
        if active_entity:
            entity_states.append(dict(active_entity))
            value = str(active_entity.get("name") or "").strip()
            if value and value not in names:
                names.append(value)
        key = self._key(context.session_id, context.user_id, context.domain_id)
        with self._lock:
            state = self._sessions.setdefault(key, {
                "summary": context.summary,
                "summary_data": dict(context.summary_data or {}),
                "long_term_memory": dict(context.long_term_memory or {}),
                "turns": [],
                "pending_clarification": None,
            })
            state["turns"].append({
                "user": str(user or ""),
                "assistant": str(assistant or ""),
                "entities": names,
                "entity_states": entity_states,
            })
            self._update_profile(state, names)


    def set_pending_clarification(self, context: ConversationContext, payload: dict[str, Any]) -> None:
        if not context.session_id:
            return
        key = self._key(context.session_id, context.user_id, context.domain_id)
        with self._lock:
            state = self._sessions.setdefault(key, {
                "summary": context.summary,
                "summary_data": dict(context.summary_data or {}),
                "long_term_memory": dict(context.long_term_memory or {}),
                "turns": list(context.turns or []),
                "pending_clarification": None,
            })
            state["pending_clarification"] = dict(payload or {})

    def clear_pending_clarification(self, context: ConversationContext) -> None:
        if not context.session_id:
            return
        key = self._key(context.session_id, context.user_id, context.domain_id)
        with self._lock:
            state = self._sessions.get(key)
            if state is not None:
                state["pending_clarification"] = None


class PersistentConversationSessionStore(ConversationSessionStore):
    """把会话窗口和用户画像持久化到 AI/RAG PostgreSQL。

    输入：基类提供的会话上下文及更新操作。
    输出：可跨进程读取的会话和用户画像状态。
    逻辑：保留内存存储行为，并通过数据库读写实现持久化。
    """

    _schema_ready = False
    _db_broken = False

    def _ensure_schema(self, db: Any) -> None:
        if self._schema_ready:
            return
        from sqlalchemy import text
        db.execute(text(
            """
            CREATE TABLE IF NOT EXISTS kg_qa_conversation_memory (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(128) NOT NULL,
                user_id VARCHAR(128) NOT NULL DEFAULT '',
                domain_id VARCHAR(128) NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                summary_data JSONB NOT NULL DEFAULT '{}'::jsonb,
                long_term_memory JSONB NOT NULL DEFAULT '{}'::jsonb,
                turns JSONB NOT NULL DEFAULT '[]'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (session_id, user_id, domain_id)
            )
            """
        ))
        db.execute(text(
            """
            ALTER TABLE kg_qa_conversation_memory
            ADD COLUMN IF NOT EXISTS summary_data JSONB NOT NULL DEFAULT '{}'::jsonb
            """
        ))
        db.execute(text(
            """
            ALTER TABLE kg_qa_conversation_memory
            ADD COLUMN IF NOT EXISTS pending_clarification JSONB NOT NULL DEFAULT '{}'::jsonb
            """
        ))
        db.execute(text(
            """
            CREATE TABLE IF NOT EXISTS kg_qa_user_memory (
                id BIGSERIAL PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL,
                domain_id VARCHAR(128) NOT NULL DEFAULT '',
                long_term_memory JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (user_id, domain_id)
            )
            """
        ))
        self._schema_ready = True

    def _disable_database(self) -> None:
        self._db_broken = True

    def open(
        self,
        session_id: str | None,
        *,
        user_id: str | None = None,
        domain_id: str | None = None,
    ) -> ConversationContext:
        context = super().open(session_id, user_id=user_id, domain_id=domain_id)
        if not context.session_id or self._db_broken:
            return context
        try:
            from src.rag.dependencies import ai_session_scope
            from sqlalchemy import text
            with ai_session_scope() as db:
                self._ensure_schema(db)
                row = db.execute(text(
                    """
                    SELECT summary, summary_data, long_term_memory, turns, pending_clarification
                    FROM kg_qa_conversation_memory
                    WHERE session_id=:session_id AND user_id=:user_id AND domain_id=:domain_id
                    """
                ), {
                    "session_id": context.session_id,
                    "user_id": context.user_id,
                    "domain_id": context.domain_id,
                }).mappings().first()
                user_row = db.execute(text(
                    """
                    SELECT long_term_memory
                    FROM kg_qa_user_memory
                    WHERE user_id=:user_id AND domain_id=:domain_id
                    """
                ), {"user_id": context.user_id, "domain_id": context.domain_id}).mappings().first()
            if not row:
                return context
            profile = dict(user_row["long_term_memory"] or {}) if user_row else {}
            profile.update(dict(row["long_term_memory"] or {}))
            turns = list(row["turns"] or [])
            key = self._key(context.session_id, context.user_id, context.domain_id)
            with self._lock:
                self._sessions[key] = {
                    "summary": str(row["summary"] or ""),
                    "summary_data": dict(row["summary_data"] or {}),
                    "long_term_memory": profile,
                    "turns": turns,
                    "pending_clarification": (
                        dict(row["pending_clarification"] or {})
                        if row["pending_clarification"] else None
                    ),
                }
            return ConversationContext(
                session_id=context.session_id,
                user_id=context.user_id,
                domain_id=context.domain_id,
                summary=str(row["summary"] or ""),
                summary_data=dict(row["summary_data"] or {}),
                long_term_memory=profile,
                turns=turns,
                needs_compression=len(turns) >= self.max_turns,
                pending_clarification=(
                    dict(row["pending_clarification"] or {})
                    if row["pending_clarification"] else None
                ),
            )
        except Exception:
            self._disable_database()
            return context

    def _persist(self, context: ConversationContext) -> None:
        if not context.session_id or self._db_broken:
            return
        key = self._key(context.session_id, context.user_id, context.domain_id)
        with self._lock:
            state = self._sessions.get(key) or {
                "summary": context.summary,
                "summary_data": dict(context.summary_data or {}),
                "long_term_memory": dict(context.long_term_memory or {}),
                "turns": list(context.turns or []),
                "pending_clarification": (
                    dict(context.pending_clarification or {})
                    if context.pending_clarification else None
                ),
            }
        try:
            from src.rag.dependencies import ai_session_scope
            from sqlalchemy import text
            with ai_session_scope() as db:
                self._ensure_schema(db)
                db.execute(text(
                    """
                    INSERT INTO kg_qa_conversation_memory
                        (session_id, user_id, domain_id, summary, summary_data, long_term_memory, turns, pending_clarification, updated_at)
                    VALUES (:session_id, :user_id, :domain_id, :summary, CAST(:summary_data AS jsonb), CAST(:long_term_memory AS jsonb), CAST(:turns AS jsonb), CAST(:pending_clarification AS jsonb), NOW())
                    ON CONFLICT (session_id, user_id, domain_id) DO UPDATE SET
                        summary=EXCLUDED.summary,
                        summary_data=EXCLUDED.summary_data,
                        long_term_memory=EXCLUDED.long_term_memory,
                        turns=EXCLUDED.turns,
                        pending_clarification=EXCLUDED.pending_clarification,
                        updated_at=NOW()
                    """
                ), {
                    "session_id": context.session_id,
                    "user_id": context.user_id,
                    "domain_id": context.domain_id,
                    "summary": state.get("summary") or "",
                    "summary_data": json.dumps(state.get("summary_data") or {}, ensure_ascii=False),
                    "long_term_memory": json.dumps(state.get("long_term_memory") or {}, ensure_ascii=False),
                    "turns": json.dumps(state.get("turns") or [], ensure_ascii=False),
                    "pending_clarification": json.dumps(
                        state.get("pending_clarification") or {}, ensure_ascii=False
                    ),
                })
                if context.user_id:
                    db.execute(text(
                        """
                        INSERT INTO kg_qa_user_memory (user_id, domain_id, long_term_memory, updated_at)
                        VALUES (:user_id, :domain_id, CAST(:long_term_memory AS jsonb), NOW())
                        ON CONFLICT (user_id, domain_id) DO UPDATE SET
                            long_term_memory=EXCLUDED.long_term_memory,
                            updated_at=NOW()
                        """
                    ), {
                        "user_id": context.user_id,
                        "domain_id": context.domain_id,
                        "long_term_memory": json.dumps(state.get("long_term_memory") or {}, ensure_ascii=False),
                    })
        except Exception:
            self._disable_database()

    def append(
        self,
        context: ConversationContext,
        user: str,
        assistant: str,
        entities: list[Any] | None = None,
        active_entity: dict[str, Any] | None = None,
    ) -> None:
        super().append(context, user, assistant, entities=entities, active_entity=active_entity)
        self._persist(context)

    def compress(self, context: ConversationContext) -> ConversationContext:
        compact = super().compress(context)
        self._persist(compact)
        return compact
 
    def set_pending_clarification(self, context: ConversationContext, payload: dict[str, Any]) -> None:
        super().set_pending_clarification(context, payload)
        self._persist(context)

    def clear_pending_clarification(self, context: ConversationContext) -> None:
        super().clear_pending_clarification(context)
        self._persist(context)


def build_default_session_store() -> ConversationSessionStore:
    enabled = os.getenv("KG_QA_PERSISTENT_MEMORY_ENABLED", "1").lower()
    return PersistentConversationSessionStore() if enabled not in {"0", "false", "no", "off"} else ConversationSessionStore()
