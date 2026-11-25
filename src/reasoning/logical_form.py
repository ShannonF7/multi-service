"""问答逻辑形式的数据契约。

这里的对象是后续图谱执行器与答案生成器之间的稳定边界。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple


Intent = Literal[
    "CHAT",
    "CURRENT_LOCATION",
    "LOCATION",
    "HIERARCHY",
    "RELATION",
    "PATH",
    "AGGREGATION",
    "COMPARISON",
    "TEMPORAL",
    "IMAGE",
    "EXPLANATION",
    "FACT",
]

ReasoningMode = Literal[
    "NONE",
    "DIRECT_PROPERTY",
    "DIRECT_RELATION",
    "HIERARCHY",
    "CURRENT_LOCATION",
    "MULTI_HOP",
    "PATH_SEARCH",
    "AGGREGATION",
    "COMPARISON",
    "TEMPORAL",
    "IMAGE_EVIDENCE",
]

MentionSource = Literal["EXPLICIT", "CONTEXT", "PENDING_SELECTION", "LOCATION"]


@dataclass(frozen=True)
class TargetReference:
    """一个需要在当前领域图谱中解析的目标引用。"""

    mention: str
    node_id: Optional[str] = None
    canonical_name: Optional[str] = None
    source: MentionSource = "EXPLICIT"


@dataclass(frozen=True)
class LogicalQuery:
    """问题理解后的结构化逻辑查询。"""

    question: str
    standalone_question: str
    domain_id: str
    intent: Intent
    relation: Optional[str] = None
    targets: Tuple[TargetReference, ...] = ()
    constraints: Dict[str, Any] = field(default_factory=dict)
    reasoning_mode: ReasoningMode = "NONE"
    confidence: float = 0.0
    needs_clarification: bool = False
    clarification: Optional[str] = None
    used_context: bool = False
    used_location: bool = False

    @property
    def requires_knowledge(self) -> bool:
        """闲聊不进入知识检索，其余意图进入知识链路。"""

        return self.intent != "CHAT"
