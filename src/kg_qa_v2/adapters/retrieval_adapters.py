"""QA v2 的文本/图片检索适配器。

只复用现有领域知识库检索服务，不在本模块直接操作数据库。适配器将旧服务
返回的领域文本和图片结果统一转换为 QA v2 Evidence。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.retrieval import Evidence


class DomainKBTextRetriever:
    """调用现有领域文本混合检索，并转换为文本 Evidence。"""

    def __init__(self, search_fn: Optional[Callable[..., Any]] = None) -> None:
        self._search_fn = search_fn or self._default_search

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        domain_id = str(kwargs["domain_id"])
        question = str(kwargs.get("question") or "").strip()
        spec = kwargs["query_spec"]
        step = kwargs["step"]
        query = question or " ".join(spec.targets)
        if not query:
            return {"evidence": []}
        limit = max(1, min(int(step.params.get("limit") or 8), 20))
        raw_items = self._search_fn(
            domain_id,
            query,
            limit=limit,
            retrieval_mode="hybrid",
            rerank_mode="separate",
            include_images=False,
        )
        return {"evidence": [_text_evidence(domain_id, item) for item in (raw_items or [])]}

    @staticmethod
    def _default_search(
        domain_id: str,
        query: str,
        *,
        limit: int,
        retrieval_mode: str,
        rerank_mode: str,
        include_images: bool,
    ) -> list[dict[str, Any]]:
        from src.rag.service.domain_kb_service import search_domain_kb

        return search_domain_kb(
            domain_id,
            query,
            limit=limit,
            retrieval_mode=retrieval_mode,
            rerank_mode=rerank_mode,
            include_images=include_images,
        )


class DomainKBImageRetriever:
    """调用现有 Qwen-VL 图片向量检索，并转换为图片 Evidence。"""

    def __init__(
        self,
        *,
        text_search_fn: Optional[Callable[..., Any]] = None,
        image_search_fn: Optional[Callable[..., Any]] = None,
        node_detail_fn: Optional[Callable[..., Any]] = None,
        image_rerank_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._text_search_fn = text_search_fn or self._default_text_search
        self._image_search_fn = image_search_fn or self._default_image_search
        self._node_detail_fn = node_detail_fn or self._default_node_detail
        self._image_rerank_fn = image_rerank_fn or self._default_image_rerank

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        domain_id = str(kwargs["domain_id"])
        question = str(kwargs.get("question") or "").strip()
        spec = kwargs["query_spec"]
        step = kwargs["step"]
        limit = max(1, min(int(step.params.get("limit") or 3), 20))
        image_path = str(
            step.params.get("image_path")
            or spec.constraints.get("image_path")
            or ""
        ).strip()
        node_id = str(step.params.get("node_id") or "").strip()
        warnings: list[str] = []
        raw_items: list[dict[str, Any]] = []

        if image_path:
            if not Path(image_path).is_file():
                return {"evidence": [], "warnings": ["图片查询文件不存在或不可读"]}
            raw_items = list(self._image_search_fn(domain_id, image_path, limit=limit) or [])
        elif node_id:
            # 视觉问题优先读取已解析节点绑定的已发布图片，避免把“东门长啥样”
            # 当成普通文本去搜，导致图片和目标节点漂移。
            try:
                detail = self._node_detail_fn(domain_id, node_id)
                assets = ((detail or {}).get("root") or {}).get("assets") or []
                for asset in assets[:limit]:
                    if isinstance(asset, dict):
                        item = dict(asset)
                        item.setdefault("node_id", node_id)
                        item.setdefault("score", 1.0)
                        raw_items.append(item)
            except Exception as exc:
                warnings.append(f"已解析节点图片读取失败：{str(exc)[:200]}")
        if not raw_items and not image_path:
            query = str(step.params.get("image_query") or "").strip() or question or " ".join(spec.targets)
            if not query:
                return {"evidence": [], "warnings": warnings}
            raw_items = list(
                self._text_search_fn(
                    domain_id,
                    query,
                    limit=limit,
                ) or []
            )
        node_bound_assets = bool(node_id and raw_items and not image_path)
        combined_location_visual = any(
            str(item.request_type).upper() == "LOCATION"
            for item in getattr(spec, "requests", [])
        )
        skip_bound_rerank = node_bound_assets and combined_location_visual
        if len(raw_items) > 1 and not skip_bound_rerank and not any(
            str(item.get("rerank_status") or "") == "completed"
            for item in raw_items
            if isinstance(item, dict)
        ):
            query_value = {"image": image_path} if image_path else {
                "text": str(step.params.get("image_query") or "").strip() or question
            }
            raw_items = list(self._image_rerank_fn(
                query=query_value,
                items=raw_items,
                instruction=(
                    "判断查询与候选图片是否对应同一地点、建筑或入口；"
                    "优先保留视觉和语义都匹配的已发布图片。"
                ),
            ) or [])
        raw_items = raw_items[:limit]
        evidence = [_image_evidence(domain_id, item) for item in raw_items]
        # 上传图片先用向量 Top1 确认节点，再读取该节点的已发布图谱详情。
        if image_path and raw_items:
            top_item = raw_items[0] if isinstance(raw_items[0], dict) else {}
            top_node_id = str(
                top_item.get("source_node_id") or top_item.get("node_id") or ""
            ).strip()
            if top_node_id:
                try:
                    detail = self._node_detail_fn(domain_id, top_node_id)
                    if isinstance(detail, dict) and detail.get("found", True):
                        from .graph_adapters import graph_result_to_evidence

                        evidence.extend(graph_result_to_evidence(domain_id, detail))
                        node_name = str(
                            (detail.get("root") or {}).get("name")
                            or top_item.get("node_name")
                            or ""
                        ).strip()
                        if node_name and evidence:
                            evidence[0].content = (
                                f"上传图片最相似的已发布图片对应节点为“{node_name}”"
                            )
                            evidence[0].metadata.update({
                                "matched_node_id": top_node_id,
                                "matched_node_name": node_name,
                                "match_rank": 1,
                            })
                    else:
                        warnings.append("图片相似结果未对应到已发布图谱节点")
                except Exception as exc:
                    warnings.append(f"图片 Top1 节点详情查询失败：{str(exc)[:200]}")
            else:
                warnings.append("图片相似结果缺少图谱节点标识")
        return {"evidence": evidence, "warnings": warnings}

    @staticmethod
    def _default_text_search(domain_id: str, query: str, *, limit: int) -> list[dict[str, Any]]:
        from src.rag.service.image_embedding_service import search_domain_kb_image_embeddings

        candidate_limit = min(max(int(limit) * 4, 12), 20)
        return search_domain_kb_image_embeddings(domain_id, query, limit=candidate_limit)

    @staticmethod
    def _default_image_search(domain_id: str, image_path: str, *, limit: int) -> list[dict[str, Any]]:
        from src.rag.service.image_embedding_service import search_domain_kb_image_by_path

        return search_domain_kb_image_by_path(domain_id, image_path, limit=limit)

    @staticmethod
    def _default_image_rerank(
        *,
        query: dict[str, Any],
        items: list[dict[str, Any]],
        instruction: str,
    ) -> list[dict[str, Any]]:
        from src.rag.service.image_embedding_service import rerank_image_candidates

        return rerank_image_candidates(
            query,
            items,
            instruction=instruction,
        )

    @staticmethod
    def _default_node_detail(domain_id: str, node_id: str) -> dict[str, Any]:
        from src.rag.service.graph_discovery_service import get_published_node_detail

        return get_published_node_detail(domain_id, node_id)


def build_domain_kb_retrievers() -> tuple[DomainKBTextRetriever, DomainKBImageRetriever]:
    """构造生产环境使用的文本和图片检索端口。"""
    return DomainKBTextRetriever(), DomainKBImageRetriever()


def _text_evidence(domain_id: str, item: Any) -> Evidence:
    value = dict(item) if isinstance(item, dict) else {"content": str(item)}
    content = str(value.get("content") or value.get("quote") or "").strip()
    source_doc_id = str(value.get("source_doc_id") or "").strip()
    chunk_id = str(value.get("chunk_id") or "").strip()
    key = chunk_id or hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    return Evidence(
        evidence_id=f"text:{domain_id}:{source_doc_id}:{key}",
        source_type="text",
        content=content,
        score=_score(value),
        source_url=_optional_text(
            value.get("source_url")
            or value.get("image_url")
            or value.get("proxy_image_url")
            or value.get("url")
        ),
        metadata={"raw_source_type": value.get("source_type"), **value},
    )


def _image_evidence(domain_id: str, item: Any) -> Evidence:
    value = dict(item) if isinstance(item, dict) else {"content": str(item)}
    content = str(value.get("content") or value.get("quote") or value.get("title") or "图片证据").strip()
    asset_id = _optional_text(value.get("asset_id"))
    key = asset_id or _optional_text(value.get("source_url")) or hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    return Evidence(
        evidence_id=f"image:{domain_id}:{key}",
        source_type="image",
        content=content,
        score=_score(value),
        node_id=_optional_text(value.get("node_id") or value.get("source_node_id")),
        asset_id=asset_id,
        source_url=_optional_text(
            value.get("source_url")
            or value.get("image_url")
            or value.get("proxy_image_url")
            or value.get("url")
        ),
        metadata={"raw_source_type": value.get("source_type"), **value},
    )


def _score(value: dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(value.get("score") or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None
