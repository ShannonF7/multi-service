"""KG-QA 会话摘要的稳定结构化契约。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .budget import ContextBudgetManager


def build_summary_contract(
    messages: list[Any],
    *,
    previous: Optional[Dict[str, Any]] = None,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    """根据已有对话生成兼容旧摘要文本的结构化摘要。

    输入：对话消息、可选的上一版摘要字段和可选摘要文本。
    输出：包含摘要、目标、主题、实体、待解决事项、事实和版本号的字典。
    逻辑：从对话中提取确定性摘要字段，并按上下文预算截断摘要文本。
    """
    previous = dict(previous or {})

    def _value(item: Any, key: str) -> str:
        """读取字典或对象中的消息字段，并转换为字符串。"""
        if isinstance(item, dict):
            return str(item.get(key) or "")
        return str(getattr(item, key, "") or "")

    user_messages: list[str] = [
        _value(item, "content").strip()
        for item in messages
        if _value(item, "role") == "user" and _value(item, "content").strip()
    ]
    assistant_messages: list[str] = [
        _value(item, "content").strip()
        for item in messages
        if _value(item, "role") == "assistant" and _value(item, "content").strip()
    ]
    compact: str = summary or "；".join(
        f"{_value(item, 'role')}：{' '.join(_value(item, 'content').split())}"
        for item in messages
        if _value(item, "content").strip()
    )
    compact, _ = ContextBudgetManager.from_env().truncate_text(compact, 640)
    return {
        "summary": compact,
        "goal": user_messages[-1][:240] if user_messages else previous.get("goal"),
        "topics": list(previous.get("topics") or [])[:8],
        "confirmed_entities": list(previous.get("confirmed_entities") or [])[:30],
        "unresolved_items": list(previous.get("unresolved_items") or [])[:8],
        "important_facts": [item[:240] for item in assistant_messages[-6:]],
        "version": int(previous.get("version") or 0),
    }
