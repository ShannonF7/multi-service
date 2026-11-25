"""推理计划执行器。

本模块把新推理层的白名单计划适配到现有已发布图谱与证据服务。
它不生成答案、不写数据库，返回结构化结果供上层答案生成器使用。
"""

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.kg_qa.entity_linker import EntityLinker
from src.kg_qa.evidence import fuse_evidence
from src.kg_qa.retriever import DefaultRetriever
from src.kg_qa.schemas import (
    DomainContext,
    EntityCandidate,
    LinkedEntity,
    QueryPlan,
    QueryStep,
    QARequest,
)

from .logical_form import LogicalQuery, TargetReference
from .planner import ReasoningPlan, build_reasoning_plan


PhaseCallback = Callable[[str, str, Optional[float], str, Dict[str, Any]], None]


def _run_phase(
    callback: Optional[PhaseCallback],
    stage: str,
    action: Callable[[], Any],
    summary: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """执行一个阶段，并把真实阶段状态回调给上层轨迹。"""

    if callback:
        callback(stage, "STARTED", None, summary, metadata or {})
    started = perf_counter()
    try:
        result = action()
    except Exception as exc:
        duration = round((perf_counter() - started) * 1000, 2)
        if callback:
            callback(stage, "FAILED", duration, str(exc), {"error_type": type(exc).__name__})
        raise
    duration = round((perf_counter() - started) * 1000, 2)
    if callback:
        callback(stage, "COMPLETED", duration, summary, metadata or {})
    return result


@dataclass
class ReasoningExecutionResult:
    """一次计划执行的结构化结果。"""

    query: LogicalQuery
    plan: ReasoningPlan
    entities: List[LinkedEntity]
    graph_results: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]]
    diagnostics: List[Dict[str, Any]]
    evidence: list[Any]
    clarification: Optional[str] = None
    warnings: List[str] | None = None
    legacy_plan: Optional[QueryPlan] = None

    def __post_init__(self) -> None:
        """保证调用方始终拿到可迭代的告警列表。"""

        if self.warnings is None:
            self.warnings = []


def _candidate_from_target(target: TargetReference) -> LinkedEntity:
    """把已由上下文确定的节点转换成统一实体结果。"""

    return LinkedEntity(
        mention=target.mention,
        status="LINKED",
        node_id=target.node_id,
        name=target.canonical_name or target.mention,
        candidates=[
            EntityCandidate(
                node_id=target.node_id or "",
                name=target.canonical_name or target.mention,
                match_type="CONTEXT",
            )
        ],
    )


def _replace_targets(
    query: LogicalQuery,
    linked: List[LinkedEntity],
) -> LogicalQuery:
    """把实体链接结果回填到逻辑查询，保留原目标顺序。"""

    by_mention = {item.mention: item for item in linked}
    targets: list[TargetReference] = []
    for target in query.targets:
        item = by_mention.get(target.mention)
        if item and item.status == "LINKED":
            targets.append(
                replace(
                    target,
                    node_id=item.node_id,
                    canonical_name=item.name or target.canonical_name or target.mention,
                )
            )
        else:
            targets.append(target)
    return replace(query, targets=tuple(targets))


def _candidate_label(candidate: EntityCandidate) -> str:
    """给同名候选补充所在层级，避免用户无法区分。"""

    path = [
        str(value).strip()
        for value in (candidate.hierarchy_path or [])
        if str(value).strip()
    ]
    if path and path[-1] == candidate.name:
        path = path[:-1]
    label = candidate.name or candidate.node_id
    return f"{label}（{'·'.join(path)}）" if path else label


def _clarification_text(entities: List[LinkedEntity]) -> Optional[str]:
    """生成不暴露内部字段的用户选择提示。"""

    groups = []
    for entity in entities:
        if entity.status != "AMBIGUOUS":
            continue
        options = [
            f"{index}. {_candidate_label(candidate)}"
            for index, candidate in enumerate(entity.candidates, 1)
            if candidate.name
        ]
        if options:
            groups.append(f"{entity.mention}：" + chr(10) + chr(10).join(options))
    return ("请确认你指的是哪个位置或部门：" + chr(10) + chr(10).join(groups)) if groups else None


def _to_query_plan(query: LogicalQuery, plan: ReasoningPlan) -> QueryPlan:
    """把新动作映射为旧检索器可执行的受控 QueryPlan。"""

    steps: list[QueryStep] = []
    for item in plan.steps:
        if item.action == "GRAPH_NODE_DETAIL":
            for node_id in item.target_node_ids:
                steps.append(
                    QueryStep(
                        type="GRAPH_NODE_DETAIL",
                        target_node_id=node_id,
                        params={
                            "relation_mode": "parent_only"
                            if item.params.get("parent_only")
                            else "all"
                        },
                        reason=item.reason,
                    )
                )
        elif item.action == "GRAPH_RELATION_LOOKUP":
            for node_id in item.target_node_ids:
                steps.append(
                    QueryStep(
                        type="GRAPH_NODE_DETAIL",
                        target_node_id=node_id,
                        params={"relation_mode": "all"},
                        reason=item.reason,
                    )
                )
        elif item.action == "GRAPH_NEIGHBORHOOD" and item.target_node_ids:
            steps.append(
                QueryStep(
                    type="GRAPH_NEIGHBORHOOD",
                    target_node_id=item.target_node_ids[0],
                    params={
                        "depth": min(int(item.params.get("depth") or 1), plan.max_hops or 1),
                        "limit": min(int(item.params.get("limit") or 20), plan.max_nodes),
                    },
                    reason=item.reason,
                )
            )
        elif item.action == "GRAPH_PATH_SEARCH" and len(item.target_node_ids) >= 2:
            steps.append(
                QueryStep(
                    type="GRAPH_PATH",
                    target_node_ids=list(item.target_node_ids[:2]),
                    params={"max_depth": min(int(item.params.get("max_hops") or 4), plan.max_hops)},
                    reason=item.reason,
                )
            )
        elif item.action == "DOCUMENT_EVIDENCE_SEARCH":
            steps.append(
                QueryStep(
                    type="TEXT_SEARCH",
                    query=query.standalone_question,
                    params={"limit": min(int(item.params.get("limit") or 8), 20)},
                    reason=item.reason,
                )
            )
        elif item.action == "IMAGE_EVIDENCE_SEARCH":
            steps.append(
                QueryStep(
                    type="IMAGE_SEARCH",
                    query=query.standalone_question,
                    params={"limit": min(int(item.params.get("limit") or 3), 20)},
                    reason=item.reason,
                )
            )
    return QueryPlan(
        steps=steps,
        max_hops=plan.max_hops,
        max_nodes=plan.max_nodes,
        max_edges=plan.max_edges,
    )


class ReasoningExecutor:
    """执行实体解析、图谱检索和证据融合。"""

    def __init__(
        self,
        *,
        entity_linker: Optional[EntityLinker] = None,
        retriever: Optional[DefaultRetriever] = None,
    ) -> None:
        self.entity_linker = entity_linker or EntityLinker()
        self.retriever = retriever or DefaultRetriever()

    def run(
        self,
        query: LogicalQuery,
        request: QARequest,
        context: DomainContext,
        on_phase: Optional[PhaseCallback] = None,
    ) -> ReasoningExecutionResult:
        """执行一次受控计划，歧义或未解析时停止在检索前。"""

        initial_plan = build_reasoning_plan(query)
        initial_legacy_plan = _to_query_plan(query, initial_plan)
        if query.intent == "CHAT":
            return ReasoningExecutionResult(
                query=query,
                plan=initial_plan,
                entities=[],
                graph_results=[],
                retrieval_results=[],
                diagnostics=[],
                evidence=[],
                legacy_plan=initial_legacy_plan,
            )

        if query.intent == "CURRENT_LOCATION" and not query.targets:
            return ReasoningExecutionResult(
                query=query,
                plan=initial_plan,
                entities=[],
                graph_results=[],
                retrieval_results=[],
                diagnostics=[],
                evidence=[],
                legacy_plan=initial_legacy_plan,
                warnings=["未收到可用的实时位置，未访问图谱和证据库"],
            )

        linked: list[LinkedEntity] = []

        def link_entities() -> None:
            mentions = [target.mention for target in query.targets if target.node_id is None]
            if mentions:
                linked.extend(self.entity_linker.link(context, mentions))
            linked.extend(
                _candidate_from_target(target)
                for target in query.targets
                if target.node_id is not None
            )

        _run_phase(
            on_phase,
            "ENTITY_LINKING",
            link_entities,
            "在当前领域的已发布图谱中解析问题目标",
        )

        ambiguous = [item for item in linked if item.status == "AMBIGUOUS"]
        if ambiguous:
            return ReasoningExecutionResult(
                query=query,
                plan=initial_plan,
                entities=linked,
                graph_results=[],
                retrieval_results=[],
                diagnostics=[],
                evidence=[],
                clarification=_clarification_text(ambiguous),
                legacy_plan=initial_legacy_plan,
                warnings=["存在多个同名或相近位置，已暂停检索"],
            )

        unresolved = [item for item in linked if item.status == "UNRESOLVED"]
        resolved_entities = [item for item in linked if item.status == "LINKED"]
        if unresolved and not resolved_entities:
            names = "、".join(item.mention for item in unresolved)
            return ReasoningExecutionResult(
                query=query,
                plan=initial_plan,
                entities=linked,
                graph_results=[],
                retrieval_results=[],
                diagnostics=[],
                evidence=[],
                legacy_plan=initial_legacy_plan,
                warnings=[f"未找到已发布位置：{names}"],
            )

        effective_query = _replace_targets(query, resolved_entities)
        effective_plan = _run_phase(
            on_phase,
            "QUERY_PLANNING",
            lambda: build_reasoning_plan(effective_query),
            "根据已链接节点编译受控图谱与证据计划",
        )
        executable_plan = _to_query_plan(effective_query, effective_plan)
        raw = _run_phase(
            on_phase,
            "RETRIEVAL",
            lambda: self.retriever.execute(context, request, executable_plan),
            "查询当前领域的已发布图谱与领域资料",
        )
        graph_results = list(raw.get("graph_results") or [])
        retrieval_results = list(raw.get("retrieval_results") or [])
        evidence = _run_phase(
            on_phase,
            "EVIDENCE_FUSION",
            lambda: fuse_evidence(
                context.graph_domain_id,
                graph_results,
                retrieval_results,
                limit=request.max_evidence,
            ),
            "合并并保留可定位的图谱、文档和图片证据",
        )
        warnings = [
            f"未找到已发布位置：{item.mention}"
            for item in unresolved
        ]
        return ReasoningExecutionResult(
            query=effective_query,
            plan=effective_plan,
            entities=linked,
            graph_results=graph_results,
            retrieval_results=retrieval_results,
            diagnostics=list(raw.get("diagnostics") or []),
            evidence=evidence,
            legacy_plan=executable_plan,
            warnings=warnings,
        )
