"""已发布图谱的 QA v2 适配器。

本模块只复用现有只读图谱服务，不自行拼接 Cypher；所有查询都携带领域和
published_only 约束，并把图谱结果转换成 QA v2 的 Evidence。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from ..core.entity import PublishedCandidateSearch
from ..core.entity_reference import is_image_deictic_reference
from ..core.retrieval import Evidence
from ..core.request_catalog import graph_predicates


class PublishedGraphGateway:
    """把现有 PublishedEntity 服务接到 QA v2 的两个窄端口。"""

    def __init__(
        self,
        *,
        search_entities: Optional[Callable[..., Any]] = None,
        node_detail: Optional[Callable[..., Any]] = None,
        facts: Optional[Callable[..., Any]] = None,
        neighborhood: Optional[Callable[..., Any]] = None,
        nearby: Optional[Callable[..., Any]] = None,
        shortest_path: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._search_entities = search_entities or self._default_search
        self._node_detail = node_detail or self._default_detail
        self._facts = facts or self._default_facts
        self._neighborhood = neighborhood or self._default_neighborhood
        self._nearby = nearby or self._default_nearby
        self._shortest_path = shortest_path or self._default_path

    def search(
        self,
        *,
        domain_id: str,
        mention: str,
        current_node_id: Optional[str] = None,
        location_node_id: Optional[str] = None,
        limit: int = 8,
        published_only: bool = True,
    ) -> dict[str, Any]:
        if not published_only:
            raise ValueError("QA v2 只允许搜索已发布图谱")
        result = self._search_entities(domain_id, mention, limit=limit)
        if isinstance(result, dict):
            identity_items = result.get("identity_items")
            related_items = result.get("related_items")
            items = result.get("items", [])
            if identity_items is None:
                identity_items = [
                    item for item in items
                    if isinstance(item, dict)
                    and item.get("match_type") not in {"semantic_description", "semantic_evidence"}
                ]
                related_items = [
                    item for item in items
                    if isinstance(item, dict)
                    and item.get("match_type") in {"semantic_description", "semantic_evidence"}
                ]
            return {
                "candidates": list(identity_items or []),
                "related_entities": list(related_items or []),
            }
        return {"candidates": list(result or []), "related_entities": []}

    def retrieve(self, **kwargs: Any) -> dict[str, Any]:
        if not kwargs.get("published_only", True):
            raise ValueError("QA v2 只允许检索已发布图谱")
        domain_id = str(kwargs["domain_id"])
        step = kwargs["step"]
        params = step.params or {}
        operation = params.get("operation", "node_detail")
        query_spec = kwargs.get("spec") or kwargs.get("query_spec")
        has_uploaded_image = bool(
            str((query_spec.constraints.get("image_path") if query_spec else "") or "").strip()
        )
        has_explicit_text_target = any(
            ref.reference_type == "EXPLICIT"
            and bool((ref.core_mention or ref.surface).strip())
            and not is_image_deictic_reference(ref)
            for ref in (query_spec.entity_refs if query_spec else [])
        )
        image_only_request = bool(
            query_spec
            and query_spec.intent == "IMAGE"
            and has_uploaded_image
            and not has_explicit_text_target
        )
        # Uploaded-image queries without a named node must not fall back to GPS/session nodes.
        node_id = params.get("node_id")
        if operation == "current_location":
            # 当前定位必须来自本次 GPS/定位上下文，不能回退到上一轮 active entity。
            node_id = node_id or kwargs["context"].location_node_id
            if not node_id:
                context = kwargs["context"]
                if context.location_in_domain is False:
                    domain_name = context.location_domain_name or context.domain_id
                    return {"evidence": [Evidence(
                        evidence_id=f"graph:{domain_id}:current-location:out-of-scope",
                        source_type="graph",
                        content=f"你当前不在“{domain_name}”的范围内。",
                        score=1.0,
                        metadata={
                            "location_scope": "OUTSIDE",
                            "domain_name": domain_name,
                            "distance_to_domain_m": context.distance_to_domain_m,
                            "domain_scope_radius_m": context.domain_scope_radius_m,
                        },
                    )]}
                return {"evidence": [], "warnings": ["当前请求没有可用的领域内定位"]}
        elif not node_id and not image_only_request:
            node_id = kwargs["context"].location_node_id or kwargs["context"].current_node_id

        if operation == "path":
            start_node_id = params.get("start_node_id")
            end_node_id = params.get("end_node_id")
            if not start_node_id or not end_node_id:
                return {"evidence": [], "warnings": ["路径查询缺少已对齐的起点或终点"]}
            result = self._shortest_path(
                domain_id,
                str(start_node_id),
                str(end_node_id),
                max_depth=int(params.get("max_hops") or 4),
            )
        elif operation == "neighborhood":
            if not node_id:
                return {"evidence": [], "warnings": ["层级查询缺少已对齐节点"]}
            result = self._neighborhood(
                domain_id,
                str(node_id),
                depth=min(int(params.get("depth") or 1), 3),
                limit=min(int(kwargs.get("spec").constraints.get("max_nodes", 80)), 300),
            )
        elif operation == "nearby":
            if not node_id:
                return {"evidence": [], "warnings": ["周边查询缺少已对齐节点"]}
            result = self._nearby(
                domain_id,
                str(node_id),
                radius_m=max(1.0, min(float(params.get("radius_m") or 300.0), 5000.0)),
                limit=max(1, min(int(params.get("limit") or 20), 100)),
                category=str(params.get("category") or "").strip() or None,
            )
        else:
            if not node_id:
                return {"evidence": [], "warnings": ["节点查询缺少已对齐节点"]}
            result = self._node_detail(domain_id, str(node_id))

        evidence = graph_result_to_evidence(domain_id, result)
        resolved_candidates = query_spec.constraints.get("resolved_candidates") if query_spec else []
        for candidate in resolved_candidates or []:
            if not isinstance(candidate, dict):
                continue
            path = [str(item).strip() for item in (candidate.get("hierarchy_path") or []) if str(item).strip()]
            name = str(candidate.get("name") or "").strip()
            node_id = str(candidate.get("node_id") or "").strip()
            if name and len(path) > 1:
                evidence.append(Evidence(
                    evidence_id=f"graph:{domain_id}:{node_id}:hierarchy",
                    source_type="graph",
                    content=f"{name}的层级位置为：{' · '.join(path)}",
                    score=0.96,
                    node_id=node_id or None,
                    metadata={"root": {"name": name, "node_id": node_id}, "hierarchy_path": path},
                ))
        return {"evidence": evidence}

    def retrieve_facts(self, **kwargs: Any) -> dict[str, Any]:
        if not kwargs.get("published_only", True):
            raise ValueError("QA v2 只允许检索已发布图谱")
        domain_id = str(kwargs["domain_id"])
        step = kwargs["step"]
        params = step.params or {}
        node_id = params.get("node_id")
        if not node_id:
            return {"evidence": [], "warnings": ["属性查询缺少已对齐节点"]}
        result = self._facts(domain_id, str(node_id), predicates=_fact_predicates(params.get("requests") or []))
        return {"evidence": graph_result_to_evidence(domain_id, result)}

    @staticmethod
    def _default_facts(domain_id: str, node_id: str, *, predicates: list[str]) -> dict[str, Any]:
        from src.rag.service.graph_discovery_service import get_published_node_facts
        return get_published_node_facts(domain_id, node_id, predicates=predicates)

    @staticmethod
    def _default_search(domain_id: str, mention: str, *, limit: int) -> dict[str, Any]:
        from src.rag.service.graph_discovery_service import search_published_entities

        return search_published_entities(domain_id, mention, limit=limit)

    @staticmethod
    def _default_detail(domain_id: str, node_id: str) -> dict[str, Any]:
        from src.rag.service.graph_discovery_service import get_published_node_detail

        return get_published_node_detail(domain_id, node_id)

    @staticmethod
    def _default_neighborhood(domain_id: str, node_id: str, *, depth: int, limit: int) -> dict[str, Any]:
        from src.rag.service.graph_discovery_service import get_published_neighborhood

        return get_published_neighborhood(domain_id, node_id, depth=depth, limit=limit)

    @staticmethod
    def _default_nearby(
        domain_id: str,
        node_id: str,
        *,
        radius_m: float,
        limit: int,
        category: Optional[str],
    ) -> dict[str, Any]:
        from src.rag.service.graph_discovery_service import get_published_nearby_nodes

        return get_published_nearby_nodes(
            domain_id,
            node_id,
            radius_m=radius_m,
            limit=limit,
            category=category,
        )

    @staticmethod
    def _default_path(domain_id: str, start_node_id: str, end_node_id: str, *, max_depth: int) -> dict[str, Any]:
        from src.rag.service.graph_discovery_service import find_published_shortest_path

        return find_published_shortest_path(
            domain_id,
            start_node_id,
            end_node_id,
            max_depth=max_depth,
        )


def build_published_graph_ports() -> tuple[PublishedCandidateSearch, PublishedGraphGateway]:
    """构造实体候选搜索和图谱证据检索端口。"""
    gateway = PublishedGraphGateway()
    return PublishedCandidateSearch(gateway.search), gateway


def graph_result_to_evidence(domain_id: str, result: dict[str, Any]) -> list[Evidence]:
    """把已发布节点详情、事实、关系和图片转成可引用证据。"""
    if not isinstance(result, dict) or not result.get("found", True):
        return []

    values: list[Evidence] = []
    root = result.get("root") or {}
    node_id = _text(result.get("node_id"), root.get("node_id"))
    title = _text(root.get("name"), node_id)

    lng, lat = root.get("lng"), root.get("lat")
    assets = [item for item in (root.get("assets") or []) if isinstance(item, dict)]
    profile_parts = []
    if lng not in (None, "") and lat not in (None, ""):
        profile_parts.append(f"{title}的坐标为经度 {lng}、纬度 {lat}")
    if assets:
        profile_parts.append(f"该位置有 {len(assets)} 张已发布图片")
    if profile_parts:
        first_url = _asset_url(assets[0]) if assets else None
        values.append(Evidence(
            evidence_id=f"graph:{domain_id}:{node_id}:profile",
            source_type="graph",
            content="；".join(profile_parts),
            score=0.98,
            node_id=node_id or None,
            asset_id=_text(assets[0].get("asset_id")) if assets else None,
            source_url=first_url,
            metadata={"root": root, "assets": assets, "coordinates": {"lng": lng, "lat": lat}},
        ))

    description = _text(root.get("description"))
    if description:
        values.append(Evidence(
            evidence_id=f"graph:{domain_id}:{node_id}:description",
            source_type="graph",
            content=description,
            score=0.94,
            node_id=node_id or None,
            metadata={"root": root},
        ))

    for fact in result.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        key, value = _text(fact.get("key"), fact.get("formal_property_id")), _text(fact.get("value"))
        if not key or not value:
            continue
        provenance = _provenance(fact)
        values.append(Evidence(
            evidence_id=f"graph:{domain_id}:{node_id}:fact:{key}",
            source_type="graph",
            content=_text(provenance.get("quote")) or f"{key}：{value}",
            score=0.91,
            node_id=node_id or None,
            source_url=_text(provenance.get("source_url")) or None,
            metadata={"fact": fact, "provenance": provenance},
        ))

    for relation in result.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        source = _text(relation.get("source_name"), title)
        target = _text(relation.get("target_name"))
        predicate = _text(relation.get("relation_label"), relation.get("relation_code"))
        if not source or not target or not predicate:
            continue
        values.append(Evidence(
            evidence_id=f"graph:{domain_id}:relation:{source}:{predicate}:{target}",
            source_type="graph",
            content=f"{source} {predicate} {target}",
            score=0.88,
            node_id=node_id or None,
            source_url=_text(_provenance(relation).get("source_url")) or None,
            metadata={"relation": relation, "provenance": _provenance(relation)},
        ))

    for node in result.get("nodes") or result.get("neighbors") or []:
        if not isinstance(node, dict):
            continue
        neighbor_id, neighbor_name = _text(node.get("node_id")), _text(node.get("name"))
        if not neighbor_id or not neighbor_name:
            continue
        distance_m = node.get("distance_m")
        if distance_m not in (None, ""):
            content = f"{title}附近约{distance_m}米处有：{neighbor_name}"
            score = 0.90
        else:
            content = f"{title}的相关位置包括：{neighbor_name}"
            score = 0.80
        values.append(Evidence(
            evidence_id=f"graph:{domain_id}:neighbor:{neighbor_id}",
            source_type="graph",
            content=content,
            score=score,
            node_id=neighbor_id,
            metadata={"node": node, "root_node_id": node_id, "distance_m": distance_m},
        ))

    path_nodes = result.get("nodes") or []
    if result.get("path") or (result.get("source_node_id") and len(path_nodes) >= 2):
        names = [_text(item.get("name"), item.get("node_id")) for item in path_nodes if isinstance(item, dict)]
        path_text = _text(result.get("path")) or " → ".join(name for name in names if name)
        if path_text:
            values.append(Evidence(
                evidence_id=f"graph:{domain_id}:path:{_text(result.get('source_node_id'), node_id)}:{_text(result.get('target_node_id'))}",
                source_type="graph",
                content=path_text,
                score=0.86,
                node_id=node_id or None,
                metadata={"path": result},
            ))
    return values


def _provenance(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("provenance_json")
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return value[0] if value and isinstance(value[0], dict) else {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    return {}


def _asset_url(item: dict[str, Any]) -> Optional[str]:
    for key in ("image_url", "url", "source_url", "asset_url"):
        value = _text(item.get(key))
        if value:
            return value
    return None


def _text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _fact_predicates(requests: list[dict[str, Any]]) -> list[str]:
    attributes = [
        str(item.get("attribute") or "").strip()
        for item in requests
        if isinstance(item, dict) and str(item.get("attribute") or "").strip()
    ]
    return graph_predicates(attributes)
