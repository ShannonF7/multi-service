"""领域内实体链接：先使用已有图谱搜索，再做可解释的唯一性判断。"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Protocol

from .schemas import DomainContext, EntityCandidate, LinkedEntity


class EntityGateway(Protocol):
    def search(self, domain_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]: ...


def _normalize(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(char for char in value if char.isalnum())


_ENTITY_SYNONYMS = {"财务室": "财务部", "财务办公室": "财务部"}
_QUALIFIER_MARKERS = ("东部教学区", "西部教学区", "行政楼", "生活区", "教学区", "校区", "座", "层")


def _query_variants(mention: str) -> list[str]:
    value = str(mention or "").strip()
    variants = [value]
    synonym = _ENTITY_SYNONYMS.get(value)
    if synonym:
        variants.append(synonym)
    for marker in ("的", "中", "内", "里", "附近"):
        if marker in value:
            tail = value.rsplit(marker, 1)[-1].strip()
            if len(tail) >= 2:
                variants.append(tail)
    for marker in _QUALIFIER_MARKERS:
        if marker in value:
            tail = value.rsplit(marker, 1)[-1].strip()
            if len(tail) >= 2:
                variants.append(tail)
    return list(dict.fromkeys(item for item in variants if 2 <= len(item) <= 100))


class DefaultEntityGateway:
    """适配现有 PublishedEntity 搜索服务，不复制图谱查询逻辑。"""

    def search(self, domain_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        from src.rag.service.graph_discovery_service import search_published_entities

        result = search_published_entities(domain_id, query, limit=limit)
        return list(result.get("items") or [])


class EntityLinker:
    def __init__(self, gateway: Optional[EntityGateway] = None) -> None:
        self._gateway = gateway or DefaultEntityGateway()

    def link(self, context: DomainContext, mentions: List[str]) -> List[LinkedEntity]:
        linked: List[LinkedEntity] = []
        for mention in mentions[:5]:
            raw_candidates: list[Dict[str, Any]] = []
            for query in _query_variants(mention):
                raw_candidates.extend(self._gateway.search(context.graph_domain_id, query, limit=100))
                if raw_candidates:
                    break
            deduped_candidates: list[Dict[str, Any]] = []
            seen_candidate_ids: set[str] = set()
            for item in raw_candidates:
                candidate_id = str(item.get("node_id") or "")
                if candidate_id and candidate_id not in seen_candidate_ids:
                    seen_candidate_ids.add(candidate_id)
                    deduped_candidates.append(item)
            candidates = [
                EntityCandidate(
                    node_id=str(item.get("node_id") or ""),
                    name=str(item.get("name") or ""),
                    node_type=item.get("node_type"),
                    match_type=item.get("match_type"),
                    score=item.get("match_score", item.get("score")),
                    parent_name=item.get("parent_name"),
                    grandparent_name=item.get("grandparent_name"),
                    hierarchy_path=[
                        str(value).strip() for value in (item.get("hierarchy_path") or [])
                        if value is not None and str(value).strip() and str(value).strip().casefold() not in {"none", "null"}
                    ],
                    distance_m=item.get("distance_m"),
                )
                for item in deduped_candidates
                if str(item.get("node_id") or "").strip() and str(item.get("name") or "").strip()
            ]
            if len(candidates) > 1:
                mention_norm = _normalize(mention)
                qualifier_scores = [
                    sum(1 for part in (item.hierarchy_path or [])[:-1] if _normalize(part) and _normalize(part) in mention_norm)
                    for item in candidates
                ]
                best_score = max(qualifier_scores or [0])
                if best_score:
                    candidates = [item for item, score in zip(candidates, qualifier_scores) if score == best_score]
            exact = [item for item in candidates if _normalize(item.name) == _normalize(mention) or _normalize(item.name) == _normalize(_query_variants(mention)[-1])]
            root_matches = [
                item for item in candidates
                if len(item.hierarchy_path or []) <= 1
                and _normalize(mention)
                and _normalize(mention) in _normalize(item.name)
            ]
            if len(root_matches) == 1:
                chosen = root_matches[0]
                linked.append(LinkedEntity(
                    mention=mention, status="LINKED", node_id=chosen.node_id,
                    name=chosen.name, node_type=chosen.node_type, candidates=candidates,
                ))
                continue
            if len(exact) == 1:
                chosen = exact[0]
                linked.append(LinkedEntity(
                    mention=mention, status="LINKED", node_id=chosen.node_id,
                    name=chosen.name, node_type=chosen.node_type, candidates=candidates,
                ))
            elif len(candidates) == 1:
                chosen = candidates[0]
                linked.append(LinkedEntity(
                    mention=mention, status="LINKED", node_id=chosen.node_id,
                    name=chosen.name, node_type=chosen.node_type, candidates=candidates,
                ))
            elif candidates:
                location_name = str(getattr(context, "location_node_name", "") or "").strip()
                location_matches = [
                    item for item in candidates
                    if location_name and location_name in (item.hierarchy_path or [])
                ]
                if len(location_matches) == 1:
                    chosen = location_matches[0]
                    linked.append(LinkedEntity(
                        mention=mention, status="LINKED", node_id=chosen.node_id,
                        name=chosen.name, node_type=chosen.node_type, candidates=candidates,
                    ))
                    continue
                linked.append(LinkedEntity(mention=mention, status="AMBIGUOUS", candidates=candidates))
            else:
                linked.append(LinkedEntity(mention=mention, status="UNRESOLVED"))
        return linked
