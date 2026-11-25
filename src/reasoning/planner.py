"""受控推理计划编译器。

计划只允许使用白名单动作与固定资源上限，避免把自然语言直接拼成任意查询。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .logical_form import LogicalQuery


ActionType = str
_ALLOWED_ACTIONS = frozenset(
    {
        "ENTITY_RESOLUTION",
        "CURRENT_LOCATION_RESOLUTION",
        "GRAPH_NODE_DETAIL",
        "GRAPH_RELATION_LOOKUP",
        "GRAPH_NEIGHBORHOOD",
        "GRAPH_PATH_SEARCH",
        "RULE_EVALUATION",
        "DOCUMENT_EVIDENCE_SEARCH",
        "IMAGE_EVIDENCE_SEARCH",
        "BUILD_ANSWER_EVIDENCE",
    }
)


@dataclass(frozen=True)
class ReasoningStep:
    """一个可执行的受控推理步骤。"""

    action: ActionType
    target_node_ids: Tuple[str, ...] = ()
    relation: Optional[str] = None
    query: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class ReasoningPlan:
    """完整的推理计划及资源边界。"""

    domain_id: str
    intent: str
    mode: str
    steps: Tuple[ReasoningStep, ...]
    max_hops: int = 0
    max_nodes: int = 0
    max_edges: int = 0
    stop_conditions: Tuple[str, ...] = ()
    requires_clarification: bool = False


def _ids(query: LogicalQuery) -> Tuple[str, ...]:
    return tuple(str(item.node_id) for item in query.targets if item.node_id is not None)


def _step(
    action: ActionType,
    query: LogicalQuery,
    *,
    reason: str,
    target_ids: Tuple[str, ...] = (),
    relation: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> ReasoningStep:
    if action not in _ALLOWED_ACTIONS:
        raise ValueError("推理动作不在白名单中")
    return ReasoningStep(
        action=action,
        target_node_ids=target_ids,
        relation=relation,
        query=query.standalone_question if action == "ENTITY_RESOLUTION" else None,
        params=dict(params or {}),
        reason=reason,
    )


def build_reasoning_plan(query: LogicalQuery) -> ReasoningPlan:
    """把逻辑查询转换为固定边界的图谱与证据检索计划。"""

    if query.intent == "CHAT":
        return ReasoningPlan(
            domain_id=query.domain_id,
            intent=query.intent,
            mode=query.reasoning_mode,
            steps=(),
            stop_conditions=("闲聊不访问知识库",),
        )

    steps = []
    target_ids = _ids(query)
    if query.intent == "CURRENT_LOCATION" and not target_ids:
        steps.append(
            _step(
                "CURRENT_LOCATION_RESOLUTION",
                query,
                reason="先解析用户实时位置，再决定是否继续查图谱",
                params={"require_location": True},
            )
        )
        return ReasoningPlan(
            domain_id=query.domain_id,
            intent=query.intent,
            mode=query.reasoning_mode,
            steps=tuple(steps),
            max_hops=0,
            max_nodes=1,
            max_edges=0,
            stop_conditions=(
                "实时位置解析成功后重新规划图谱查询",
                "没有实时位置时停止，不访问证据库",
            ),
            requires_clarification=query.needs_clarification,
        )

    if not target_ids and query.targets:
        steps.append(
            _step(
                "ENTITY_RESOLUTION",
                query,
                reason="在当前领域的已发布图谱中解析问题中的完整名称",
                params={"published_only": True, "domain_scoped": True, "max_candidates": 5},
            )
        )

    if target_ids:
        ids = target_ids
        if query.intent in {"LOCATION", "FACT", "TEMPORAL", "IMAGE"}:
            steps.append(
                _step(
                    "GRAPH_NODE_DETAIL",
                    query,
                    reason="读取目标节点属性与已发布关系",
                    target_ids=ids,
                    relation=query.relation,
                    params={"published_only": True, "parent_only": query.intent == "LOCATION"},
                )
            )
        elif query.intent == "HIERARCHY":
            steps.append(
                _step(
                    "GRAPH_NEIGHBORHOOD",
                    query,
                    reason="读取目标节点的一跳层级邻域",
                    target_ids=ids,
                    relation=query.relation,
                    params={"published_only": True, "depth": 1, "direction": "both"},
                )
            )
        elif query.intent == "PATH":
            steps.append(
                _step(
                    "GRAPH_PATH_SEARCH",
                    query,
                    reason="在层级与可通行关系上搜索有限跳数路径",
                    target_ids=ids,
                    relation=query.relation,
                    params={"published_only": True, "max_hops": 4, "max_paths": 3},
                )
            )
        else:
            steps.append(
                _step(
                    "GRAPH_RELATION_LOOKUP",
                    query,
                    reason="读取目标节点的直接关系与必要邻域",
                    target_ids=ids,
                    relation=query.relation,
                    params={"published_only": True, "depth": 1, "max_edges": 30},
                )
            )

    if query.intent in {"EXPLANATION", "RELATION", "PATH"}:
        steps.append(
            _step(
                "RULE_EVALUATION",
                query,
                reason="对已取回的层级、关系和属性执行可解释规则",
                params={"max_rules": 12},
            )
        )

    if query.intent == "IMAGE":
        steps.append(
            _step(
                "IMAGE_EVIDENCE_SEARCH",
                query,
                reason="只返回与目标节点关联且已发布的少量图片",
                target_ids=target_ids,
                params={"published_only": True, "limit": 3},
            )
        )
    else:
        steps.append(
            _step(
                "DOCUMENT_EVIDENCE_SEARCH",
                query,
                reason="补充目标节点关联的已发布文本证据",
                target_ids=target_ids,
                params={"published_only": True, "limit": 8},
            )
        )

    steps.append(
        _step(
            "BUILD_ANSWER_EVIDENCE",
            query,
            reason="合并可验证事实并生成可悬浮查看的引用",
            target_ids=target_ids,
            params={"max_items": 8, "require_source": True},
        )
    )
    return ReasoningPlan(
        domain_id=query.domain_id,
        intent=query.intent,
        mode=query.reasoning_mode,
        steps=tuple(steps),
        max_hops=4 if query.intent == "PATH" else 1,
        max_nodes=40,
        max_edges=80,
        stop_conditions=(
            "达到足以回答问题的最短路径后停止",
            "不得超过最大跳数、节点数和边数",
            "没有新增可验证事实时停止",
        ),
        requires_clarification=query.needs_clarification,
    )


def allowed_actions() -> Tuple[str, ...]:
    """返回执行器可接受的动作白名单。"""

    return tuple(sorted(_ALLOWED_ACTIONS))
