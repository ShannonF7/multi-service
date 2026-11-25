"""受控问答推理核心。

本包只负责把用户问题与会话上下文编译为结构化逻辑查询和推理计划，
不直接访问数据库、不调用模型，也不修改业务数据。
"""

from .executor import ReasoningExecutionResult, ReasoningExecutor
from .context import (
    ConversationContextView,
    PendingSelection,
    ResolvedContext,
    resolve_context,
)
from .logical_form import LogicalQuery, ReasoningMode, TargetReference
from .planner import ReasoningPlan, build_reasoning_plan
from .query_understanding import understand_question

__all__ = [
    "ReasoningExecutionResult",
    "ReasoningExecutor",
    "ConversationContextView",
    "PendingSelection",
    "ResolvedContext",
    "resolve_context",
    "LogicalQuery",
    "ReasoningMode",
    "TargetReference",
    "ReasoningPlan",
    "build_reasoning_plan",
    "understand_question",
]
