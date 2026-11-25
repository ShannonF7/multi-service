"""受控查询计划，不允许模型直接生成任意 Cypher。"""

from __future__ import annotations

from .schemas import DomainContext, InputAnalysis, LinkedEntity, QueryPlan, QueryStep, QARequest


def _needs_spatial_neighborhood(message: str) -> bool:
    text = str(message or "")
    return any(marker in text for marker in ("附近", "周边", "周围", "旁边", "相邻", "里面有什么", "内部有什么", "包含哪些"))


def build_query_plan(
    request: QARequest,
    analysis: InputAnalysis,
    entities: list[LinkedEntity],
    context: DomainContext | None = None,
) -> QueryPlan:
    """根据意图生成有限动作集合，所有动作都有固定参数上限。"""
    if "CHAT" in analysis.knowledge_intents:
        return QueryPlan(steps=[], max_hops=0, max_nodes=0, max_edges=0)

    steps: list[QueryStep] = []
    linked_ids = [item.node_id for item in entities if item.status == "LINKED" and item.node_id]
    context_node_id = request.current_node_id or (context.location_node_id if context else None)
    intents = set(analysis.knowledge_intents)
    location_only = "SPATIAL" in intents and not _needs_spatial_neighborhood(request.message) and not (intents & {"LIST", "RELATION", "MULTI_HOP", "EXPLANATION", "HIERARCHY"})
    detail_ids = linked_ids or ([str(context_node_id)] if context_node_id and ("SPATIAL" in intents or not analysis.mentions) else [])
    for node_id in detail_ids:
        steps.append(QueryStep(
            type="GRAPH_NODE_DETAIL",
            target_node_id=node_id,
            params={"relation_mode": "parent_only" if location_only else "all"},
            reason="读取目标位置本身的正式信息" if location_only else "读取已链接位置的正式事实和关系",
        ))
    needs_neighbors = bool(intents & {"LIST", "RELATION", "MULTI_HOP", "EXPLANATION", "HIERARCHY"}) or ("SPATIAL" in intents and _needs_spatial_neighborhood(request.message))
    if request.scope in {"SUBGRAPH", "NODE"} or needs_neighbors:
        node_id = linked_ids[0] if linked_ids else context_node_id
        if node_id:
            steps.append(QueryStep(type="GRAPH_NEIGHBORHOOD", target_node_id=node_id, params={"depth": 2 if "MULTI_HOP" in intents else 1, "limit": 20}, reason="回答关系、邻近或列表问题时展开受限邻域"))
    if "MULTI_HOP" in intents and len(linked_ids) >= 2:
        steps.append(QueryStep(type="GRAPH_PATH", target_node_ids=linked_ids[:2], params={"max_depth": 2}, reason="回答多跳问题时查找两个位置之间的最短路径"))
    if analysis.has_text and analysis.target_modality == "TEXT":
        steps.append(QueryStep(type="TEXT_SEARCH", query=request.message.strip(), params={"limit": min(request.max_evidence, 8)}, reason="用领域知识库补充图谱中的详细说明和原文证据"))
    if analysis.target_modality == "IMAGE" or analysis.has_image or "IMAGE" in intents:
        steps.append(QueryStep(type="IMAGE_SEARCH", query=request.message.strip() or None, params={"limit": min(request.max_evidence, 8)}, reason="召回领域内相关图片作为答案证据或展示资源"))

    unique: list[QueryStep] = []
    seen: set[tuple] = set()
    for step in steps:
        key = (step.type, step.target_node_id, tuple(step.target_node_ids), step.query)
        if key not in seen:
            seen.add(key)
            unique.append(step)
    return QueryPlan(steps=unique, max_hops=2, max_nodes=20, max_edges=40)
