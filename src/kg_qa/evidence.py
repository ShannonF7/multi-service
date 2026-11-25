"""把图谱、文本和图片结果统一为可定位、可引用的 EvidenceItem。"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping

from .schemas import EvidenceItem


def _text(*values: Any) -> str:
    for value in values:
        value = str(value or "").strip()
        if value:
            return value
    return ""


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, list):
            return dict(parsed[0]) if parsed and isinstance(parsed[0], Mapping) else {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return dict(value[0])
    return {}


def _page_no(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _provenance(item: Mapping[str, Any]) -> Dict[str, Any]:
    raw = item.get("provenance_json")
    if isinstance(raw, list):
        return _as_dict(raw)
    return _as_dict(raw)


def _graph_location(
    *,
    node_id: str,
    title: str,
    item: Mapping[str, Any] | None = None,
    root_description: bool = False,
) -> Dict[str, Any]:
    item = item or {}
    provenance = _provenance(item)
    source_title = _text(provenance.get("source_title"), item.get("source_title"), title)
    source_url = _text(provenance.get("source_url"), item.get("source_url")) or None
    page_no = _page_no(provenance.get("page_no", item.get("page_no")))
    evidence_ids = provenance.get("evidence_ids") or item.get("evidence_ids") or []
    if not isinstance(evidence_ids, list):
        evidence_ids = [evidence_ids]
    evidence_ids = [_text(value) for value in evidence_ids if _text(value)]
    if root_description:
        location_label = f"已发布图谱节点 · {title} · 节点 {node_id}"
    elif source_url and page_no is not None:
        location_label = f"{source_title} · 第 {page_no} 页"
    elif source_url:
        location_label = source_title or "网页来源"
    elif source_title:
        location_label = source_title
    elif evidence_ids:
        location_label = f"图谱证据 · {', '.join(evidence_ids[:3])}"
    else:
        location_label = f"已发布图谱节点 · {title} · 节点 {node_id}"
    return {
        "source_url": source_url,
        "page_no": page_no,
        "source_label": source_title or None,
        "location_label": location_label,
        "provenance": provenance,
        "evidence_ids": evidence_ids,
    }


def _graph_evidence(domain_id: str, result: Dict[str, Any]) -> Iterable[EvidenceItem]:
    node_id = _text(result.get("node_id"), result.get("source_node_id"))
    root = result.get("root") or {}
    title = _text(root.get("name"), node_id)
    description = _text(root.get("description"))
    assets = [item for item in (root.get("assets") or []) if isinstance(item, Mapping) and _text(item.get("image_url"))]
    lng = root.get("lng")
    lat = root.get("lat")
    has_coordinates = lng not in (None, "") and lat not in (None, "")
    if has_coordinates or assets:
        profile_parts = []
        if has_coordinates:
            profile_parts.append(f"{title}的已发布节点坐标：经度 {lng}，纬度 {lat}")
        if assets:
            profile_parts.append(f"该节点关联 {len(assets)} 张已发布图片")
        first_asset = assets[0] if assets else {}
        yield EvidenceItem(
            evidence_id=f"graph:{domain_id}:{node_id}:profile",
            kind="graph_fact",
            content="；".join(profile_parts),
            title=title,
            source_id=node_id or None,
            source_url=_text(first_asset.get("image_url")) or None,
            image_url=_text(first_asset.get("image_url")) or None,
            source_label="已发布图谱节点",
            location_label=f"已发布图谱节点 · {title}",
            score=0.98,
            metadata={"domain_id": domain_id, "node": root, "assets": assets, "coordinates": {"lng": lng, "lat": lat}},
        )
    if description:
        location = _graph_location(node_id=node_id, title=title, root_description=True)
        yield EvidenceItem(
            evidence_id=f"graph:{domain_id}:{node_id}:description",
            kind="graph_fact",
            content=description,
            title=title,
            source_id=node_id or None,
            source_url=location["source_url"],
            page_no=location["page_no"],
            source_label="已发布图谱",
            location_label=location["location_label"],
            score=0.92,
            metadata={"domain_id": domain_id, "node": root, "location": location},
        )
    for fact in result.get("facts") or []:
        key = _text(fact.get("formal_property_id"), fact.get("key"))
        value = _text(fact.get("value"))
        if not key or not value:
            continue
        location = _graph_location(node_id=node_id, title=title, item=fact)
        provenance = location["provenance"]
        yield EvidenceItem(
            evidence_id=f"graph:{domain_id}:{node_id}:fact:{key}",
            kind="graph_fact",
            content=_text(provenance.get("quote"), f"{key}：{value}"),
            title=title,
            source_id=node_id or None,
            source_url=location["source_url"],
            page_no=location["page_no"],
            source_label=location["source_label"],
            location_label=location["location_label"],
            score=0.90,
            metadata={
                "domain_id": domain_id,
                "fact": fact,
                "location": location,
                "source_evidence_ids": location["evidence_ids"],
            },
        )
    for relation in result.get("relations") or []:
        source = _text(relation.get("source_name"), root.get("name"))
        target = _text(relation.get("target_name"))
        predicate = _text(relation.get("relation_label"), relation.get("relation_code"))
        if not source or not predicate or not target:
            continue
        location = _graph_location(node_id=node_id, title=title, item=relation)
        yield EvidenceItem(
            evidence_id=f"graph:{domain_id}:relation:{source}:{predicate}:{target}",
            kind="graph_fact",
            content=f"{source} {predicate} {target}",
            title=title,
            source_id=node_id or None,
            source_url=location["source_url"],
            page_no=location["page_no"],
            source_label=location["source_label"],
            location_label=location["location_label"],
            score=0.88,
            metadata={"domain_id": domain_id, "relation": relation, "location": location},
        )
    if result.get("path"):
        yield EvidenceItem(
            evidence_id=f"graph:{domain_id}:path:{node_id}",
            kind="graph_fact",
            content=_text(result.get("path")),
            title="图谱路径",
            source_id=node_id or None,
            source_label="已发布图谱路径",
            location_label=f"已发布图谱路径 · 节点 {node_id}",
            score=0.86,
            metadata={"domain_id": domain_id, "path": result.get("path")},
        )
    elif result.get("source_node_id") and result.get("target_node_id") and result.get("nodes"):
        names = [_text(item.get("name"), item.get("node_id")) for item in result.get("nodes") or []]
        names = [name for name in names if name]
        if names:
            yield EvidenceItem(
                evidence_id=f"graph:{domain_id}:path:{result['source_node_id']}:{result['target_node_id']}",
                kind="graph_fact",
                content=" -> ".join(names),
                title="图谱路径",
                source_id=_text(result.get("source_node_id")) or None,
                source_label="已发布图谱路径",
                location_label=f"已发布图谱路径 · {result['source_node_id']} → {result['target_node_id']}",
                score=0.86,
                metadata={"domain_id": domain_id, "path": result},
            )


def _retrieval_evidence(item: Dict[str, Any], index: int) -> EvidenceItem | None:
    asset_id = item.get("asset_id")
    is_image = asset_id is not None or "image" in _text(item.get("source_type")).lower()
    content = _text(item.get("content"), item.get("evidence_text"), item.get("quote"), item.get("caption"), item.get("title"))
    if not content:
        return None
    source_id = _text(item.get("source_id"), item.get("source_doc_id"), item.get("chunk_id"), asset_id) or None
    kind = "image" if is_image else "text"
    title = _text(item.get("title"), item.get("source_title"), item.get("source")) or None
    page_no = _page_no(item.get("page_no"))
    if title and page_no is not None:
        location_label = f"{title} · 第 {page_no} 页"
    elif title:
        location_label = title
    elif source_id:
        location_label = f"{kind} 检索结果 · {source_id}"
    else:
        location_label = f"{kind} 检索结果"
    image_url = _text(item.get("image_url"), item.get("thumbnail_url"), item.get("asset_url"), item.get("url"), item.get("source_url")) or None
    return EvidenceItem(
        evidence_id=f"{kind}:{source_id or index}",
        kind=kind,
        content=content,
        title=title,
        source_id=source_id,
        source_url=_text(item.get("source_url")) or None,
        image_url=image_url if kind == "image" else None,
        page_no=page_no,
        source_label=title,
        location_label=location_label,
        score=float(item.get("score") or item.get("final_evidence_score") or 0.0),
        metadata={"source_type": item.get("source_type"), "raw": item},
    )


def fuse_evidence(
    domain_id: str,
    graph_results: List[Dict[str, Any]],
    retrieval_results: List[Dict[str, Any]],
    limit: int = 8,
) -> List[EvidenceItem]:
    values: list[EvidenceItem] = list(item for result in graph_results for item in _graph_evidence(domain_id, result))
    for index, item in enumerate(retrieval_results):
        normalized = _retrieval_evidence(item, index)
        if normalized:
            values.append(normalized)

    unique: dict[str, EvidenceItem] = {}
    for item in values:
        previous = unique.get(item.evidence_id)
        if previous is None or item.score > previous.score:
            unique[item.evidence_id] = item
    return sorted(unique.values(), key=lambda item: item.score, reverse=True)[:max(1, min(limit, 20))]

