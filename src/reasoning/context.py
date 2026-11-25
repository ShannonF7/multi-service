"""会话上下文与指代消解。

本模块只消费已经整理好的会话视图，不访问数据库。
"""

from dataclasses import dataclass, field
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ConversationContextView:
    """供推理层读取的最小会话视图。"""

    session_id: str = ""
    summary: str = ""
    long_term_memory: Mapping[str, Any] = field(default_factory=dict)
    turns: Tuple[Mapping[str, Any], ...] = ()
    pending_clarification: Optional[Mapping[str, Any]] = None
    location_node_id: Optional[str] = None
    location_node_name: Optional[str] = None

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "ConversationContextView":
        """把 session store 的字典安全转换为只读视图。"""

        if not value:
            return cls()
        turns = value.get("turns") or ()
        memory = value.get("long_term_memory") or {}
        return cls(
            session_id=str(value.get("session_id") or ""),
            summary=str(value.get("summary") or ""),
            long_term_memory=memory,
            turns=tuple(turns),
            pending_clarification=value.get("pending_clarification"),
            location_node_id=_as_node_id(value.get("location_node_id")),
            location_node_name=_clean(value.get("location_node_name")),
        )

    def recent_entity_names(self) -> Tuple[str, ...]:
        """按最近使用顺序返回上下文中的实体名称。"""

        names: List[str] = []
        for turn in reversed(self.turns):
            for key in ("entity_name", "target_name", "entity", "target"):
                name = _clean(turn.get(key)) if isinstance(turn, Mapping) else ""
                if name and name not in names:
                    names.append(name)
        last_entity = self.long_term_memory.get("last_entity")
        if isinstance(last_entity, Mapping):
            name = _clean(last_entity.get("name"))
        else:
            name = _clean(last_entity)
        if name and name not in names:
            names.append(name)
        return tuple(names)


@dataclass(frozen=True)
class PendingSelection:
    """一次待用户选择的候选项。"""

    candidate: Mapping[str, Any]
    index: int


@dataclass(frozen=True)
class ResolvedContext:
    """上下文解析结果，供逻辑查询构造使用。"""

    original_question: str
    standalone_question: str
    active_entity_name: Optional[str] = None
    active_entity_id: Optional[str] = None
    active_entity_source: Optional[str] = None
    used_context: bool = False
    used_location: bool = False
    pending_selection: Optional[PendingSelection] = None
    pending_mention: Optional[str] = None


_CURRENT_LOCATION_RE = re.compile(
    r"(?:请问|请|你知道|你能告诉我)?\s*"
    r"(?:我(?:现在|当前)?(?:的)?位置|当前位置|我所在的位置|"
    r"我在哪(?:里)?|我现在在哪(?:里)?|当前位置是什么位置)"
    r"(?:呢|吗)?[？?。！!，,]*$"
)
_CHOICE_NUMBER_RE = re.compile(r"^(?:第\s*)?([1-9][0-9]*)\s*(?:个|项|条|号)?$")
_FLOOR_CHOICE_RE = re.compile(r"^(?:第\s*)?([一二三四五六七八九十百0-9]+)\s*(?:楼|层)(?:的|那个|这一层)?$")


def _clean(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def _as_node_id(value: Any) -> Optional[str]:
    text = str(value).strip() if value is not None else ""
    return text or None


def is_current_location_question(question: str) -> bool:
    """判断是否是实时位置问题，避免把“我现在”当成节点名称。"""

    text = re.sub(r"\s+", "", question or "")
    return bool(_CURRENT_LOCATION_RE.fullmatch(text))


def _chinese_number(value: str) -> Optional[int]:
    if value.isdigit():
        return int(value)
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "零": 0}
    if value in digits:
        return digits[value]
    if value == "十":
        return 10
    if len(value) == 2 and value[0] == "十" and value[1] in digits:
        return 10 + digits[value[1]]
    if len(value) == 2 and value[1] == "十" and value[0] in digits:
        return digits[value[0]] * 10
    if len(value) == 3 and value[1] == "十" and value[0] in digits and value[2] in digits:
        return digits[value[0]] * 10 + digits[value[2]]
    return None


def resolve_pending_selection(
    question: str,
    pending: Optional[Mapping[str, Any]],
) -> Optional[PendingSelection]:
    """把“二楼的”“第 2 个”等回答解析为上一轮候选项。"""

    if not pending:
        return None
    candidates = pending.get("candidates") or []
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return None
    text = re.sub(r"\s+", "", question or "").strip("。！？!?，, ")
    match = _CHOICE_NUMBER_RE.fullmatch(text)
    index = int(match.group(1)) if match else None
    if index is None:
        floor = _FLOOR_CHOICE_RE.fullmatch(text)
        if floor:
            index = _chinese_number(floor.group(1))
    if index is None or index < 1 or index > len(candidates):
        return None
    candidate = candidates[index - 1]
    if not isinstance(candidate, Mapping):
        return None
    return PendingSelection(candidate=candidate, index=index)


def _candidate_value(candidate: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if candidate.get(key) not in (None, ""):
            return candidate.get(key)
    return None


def resolve_context(
    question: str,
    context: Optional[ConversationContextView] = None,
) -> ResolvedContext:
    """解析当前位置、待选项和最近一轮实体，生成独立问题。"""

    view = context or ConversationContextView()
    original = (question or "").strip()
    if is_current_location_question(original):
        return ResolvedContext(
            original_question=original,
            standalone_question=original,
            active_entity_id=view.location_node_id,
            active_entity_name=view.location_node_name,
            active_entity_source="LOCATION" if view.location_node_id or view.location_node_name else None,
            used_location=bool(view.location_node_id or view.location_node_name),
        )

    pending = resolve_pending_selection(original, view.pending_clarification)
    if pending:
        candidate = pending.candidate
        name = _clean(_candidate_value(candidate, "name", "label", "title"))
        node_id = _as_node_id(_candidate_value(candidate, "node_id", "id"))
        base_question = _clean(view.pending_clarification.get("question")) or original
        standalone = f"{base_question}（已选择：{name or pending.index}）"
        return ResolvedContext(
            original_question=original,
            standalone_question=standalone,
            active_entity_name=name,
            active_entity_id=node_id,
            active_entity_source="PENDING_SELECTION",
            used_context=True,
            pending_selection=pending,
        )

    recent = view.recent_entity_names()
    pronoun = bool(re.search(r"^(它|这个|那个|该处|这里|那里)(?:的|是|在|位于|有|呢|吗|？|\?|$)", original) or re.search(r"[，,。；;\s](它|这个|那个|该处|这里|那里)(?:的|是|在|位于|有|呢|吗|？|\?|$)", original))
    if pronoun and recent:
        name = recent[0]
        standalone = f"关于{name}，{original}"
        return ResolvedContext(
            original_question=original,
            standalone_question=standalone,
            active_entity_name=name,
            active_entity_source="CONTEXT",
            used_context=True,
        )

    if view.location_node_id or view.location_node_name:
        return ResolvedContext(
            original_question=original,
            standalone_question=original,
            active_entity_id=view.location_node_id,
            active_entity_name=view.location_node_name,
            active_entity_source="LOCATION",
            used_location=True,
        )

    return ResolvedContext(original_question=original, standalone_question=original)
