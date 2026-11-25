"""查询计划执行器：只调用已有 RAG/图谱服务，不复制存储逻辑。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol

from .schemas import DomainContext, QueryPlan, QARequest


class GraphGateway(Protocol):
    def node_detail(self, domain_id: str, node_id: str) -> Dict[str, Any]: ...
    def neighborhood(self, domain_id: str, node_id: str, depth: int, limit: int) -> Dict[str, Any]: ...
    def path(self, domain_id: str, source_node_id: str, target_node_id: str, max_depth: int) -> Dict[str, Any]: ...


class DefaultGraphGateway:
    def node_detail(self, domain_id: str, node_id: str) -> Dict[str, Any]:
        from src.rag.service.graph_discovery_service import get_published_node_detail

        return get_published_node_detail(domain_id, node_id)

    def neighborhood(self, domain_id: str, node_id: str, depth: int, limit: int) -> Dict[str, Any]:
        from src.rag.service.graph_discovery_service import get_published_neighborhood

        return get_published_neighborhood(domain_id, node_id, depth=depth, limit=limit)

    def path(self, domain_id: str, source_node_id: str, target_node_id: str, max_depth: int) -> Dict[str, Any]:
        from src.rag.service.graph_discovery_service import find_published_shortest_path

        return find_published_shortest_path(domain_id, source_node_id, target_node_id, max_depth=max_depth)


class DefaultRetriever:
    def __init__(
        self,
        graph: Optional[GraphGateway] = None,
        text_search: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        unified_search: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.graph = graph or DefaultGraphGateway()
        self.text_search = text_search or self._default_text_search
        self.unified_search = unified_search or self._default_unified_search

    @staticmethod
    def _default_text_search(source_scenic_id: str, query: str, limit: int) -> List[Dict[str, Any]]:
        from src.rag.service.domain_kb_service import search_domain_kb

        return search_domain_kb(
            source_scenic_id, query, limit=limit,
            retrieval_mode="hybrid", rerank_mode="separate", include_images=False,
        )

    @staticmethod
    def _default_unified_search(**kwargs: Any) -> Dict[str, Any]:
        from src.rag.service.unified_retrieval_service import search_unified

        return search_unified(**kwargs)

    def execute(self, context: DomainContext, request: QARequest, plan: QueryPlan) -> Dict[str, Any]:
        graph_results: list[dict[str, Any]] = []
        retrieval_results: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for step in plan.steps:
            if step.type == "GRAPH_NODE_DETAIL" and step.target_node_id:
                detail = self.graph.node_detail(context.graph_domain_id, step.target_node_id)
                relation_mode = step.params.get("relation_mode")
                if relation_mode == "parent_only":
                    detail = dict(detail or {})
                    node_id = str(detail.get("node_id") or "")
                    relations = [
                        item for item in (detail.get("relations") or [])
                        if str(item.get("target_node_id") or "") == node_id
                    ]
                    parent_ids = {str(item.get("source_node_id") or "") for item in relations}
                    detail["relations"] = relations
                    detail["neighbors"] = [
                        item for item in (detail.get("neighbors") or [])
                        if str(item.get("node_id") or "") in parent_ids
                    ]
                    detail["total_relations"] = len(relations)
                elif step.params.get("include_relations") is False:
                    detail = dict(detail or {})
                    detail["relations"] = []
                    detail["neighbors"] = []
                    detail["total_relations"] = 0
                graph_results.append(detail)
            elif step.type == "GRAPH_NEIGHBORHOOD" and step.target_node_id:
                graph_results.append(self.graph.neighborhood(
                    context.graph_domain_id, step.target_node_id,
                    depth=min(int(step.params.get("depth") or 1), plan.max_hops),
                    limit=plan.max_nodes,
                ))
            elif step.type == "GRAPH_PATH" and len(step.target_node_ids) >= 2:
                graph_results.append(self.graph.path(
                    context.graph_domain_id, step.target_node_ids[0], step.target_node_ids[1],
                    max_depth=plan.max_hops,
                ))
            elif step.type == "TEXT_SEARCH" and step.query:
                retrieval_results.extend(self.text_search(
                    context.source_scenic_id, step.query, int(step.params.get("limit") or 8),
                ) or [])
            elif step.type == "IMAGE_SEARCH":
                if request.image_path:
                    result = self.unified_search(
                        source_scenic_id=context.source_scenic_id,
                        image_path=request.image_path,
                        intent="IMAGE_TO_IMAGE" if request.target_modality == "IMAGE" else "IMAGE_TO_TEXT",
                        limit=int(step.params.get("limit") or 8),
                    )
                elif step.query:
                    result = self.unified_search(
                        source_scenic_id=context.source_scenic_id,
                        query_text=step.query,
                        intent="TEXT_TO_IMAGE",
                        limit=int(step.params.get("limit") or 8),
                    )
                else:
                    result = {"items": [], "diagnostics": {"reason": "image_path_missing"}}
                retrieval_results.extend(result.get("items") or [])
                diagnostics.append(result.get("diagnostics") or {})
        return {
            "graph_results": graph_results,
            "retrieval_results": retrieval_results,
            "diagnostics": diagnostics,
        }
