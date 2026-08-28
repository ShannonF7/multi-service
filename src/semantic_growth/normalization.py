from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections import defaultdict
from typing import Any
from urllib.parse import urljoin

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.service.value_normalization_service import canonical_predicate, normalize_text_value
from src.semantic_growth.multimodal_reranker_client import score_documents

logger = logging.getLogger(__name__)

G3_VERSION = "growth-g3-v2"
M1_VERSION = "growth-m1-v1"
_ALIAS_KEYS = {"alias", "aliases", "别名", "曾用名", "简称", "又名", "英文名"}


def _resolve_media_url(value: Any) -> str:
    raw = str(value or "").strip()
    if raw.startswith(("http://", "https://")):
        return raw
    if raw.startswith("/"):
        base = os.getenv("QWEN_VL_MEDIA_BASE_URL", "http://ai.smartoptiks.cn").rstrip("/")
        return urljoin(base + "/", raw.lstrip("/"))
    return raw


def normalize_entity_name(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return "".join(char for char in normalized if char.isalnum())


def _text_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("value") or value.get("values") or value.get("display_value")
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_text_values(item))
        return values
    return [item.strip() for item in re.split(r"[,，、;/；|]", str(value or "")) if item.strip()]


def _aliases(properties: Any) -> set[str]:
    if not isinstance(properties, dict):
        return set()
    aliases: set[str] = set()
    for key, value in properties.items():
        if str(key or "").strip().casefold() in _ALIAS_KEYS:
            aliases.update(normalize_entity_name(item) for item in _text_values(value))
    return {item for item in aliases if item}


def _node_indexes(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aliases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = normalize_entity_name(row.get("node_name"))
        if key:
            exact[key].append(row)
        for alias in _aliases(row.get("properties")):
            aliases[alias].append(row)
    return exact, aliases


def _entity_recall(raw_name: str, exact: dict[str, list[dict[str, Any]]], aliases: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    key = normalize_entity_name(raw_name)
    matches = exact.get(key) or aliases.get(key) or []
    match_type = "EXACT_MATCH" if exact.get(key) else ("ALIAS_MATCH" if aliases.get(key) else None)
    candidates = [
        {
            "target_node_id": str(item.get("source_node_id") or ""),
            "name": item.get("node_name"),
            "node_type": item.get("node_type"),
            "match_type": match_type,
        }
        for item in matches
    ]
    return {
        "match_type": match_type,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "decision": "AUTO_EXACT_RECALL" if len(candidates) == 1 and match_type == "EXACT_MATCH" else (
            "AUTO_ALIAS_RECALL" if len(candidates) == 1 and match_type == "ALIAS_MATCH" else (
                "REVIEW_REQUIRED" if candidates else "NO_GRAPH_MATCH"
            )
        ),
    }


def _graph_context_score(
    subject_node_id: str,
    target_node_id: str | None,
    nodes_by_id: dict[str, dict[str, Any]],
) -> float:
    """Score structural compatibility without treating it as proof."""
    if not target_node_id:
        return 0.0
    subject = nodes_by_id.get(str(subject_node_id)) or {}
    target = nodes_by_id.get(str(target_node_id)) or {}
    subject_parent = str(subject.get("parent_source_node_id") or "")
    target_parent = str(target.get("parent_source_node_id") or "")
    if subject_parent and subject_parent == str(target_node_id):
        return 1.0
    if target_parent and target_parent == str(subject_node_id):
        return 1.0
    if subject_parent and subject_parent == target_parent:
        return 0.55
    return 0.25


def _joint_resolution_score(
    entity: dict[str, Any],
    vector_recall: list[dict[str, Any]],
    *,
    evidence_score: float,
    graph_score: float,
    subject_node_id: str,
    nodes_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    decision = str(entity.get("decision") or "NO_GRAPH_MATCH")
    deterministic_score = {
        "AUTO_EXACT_RECALL": 1.0,
        "AUTO_ALIAS_RECALL": 0.85,
        "REVIEW_REQUIRED": 0.35,
        "NO_GRAPH_MATCH": 0.0,
    }.get(decision, 0.0)
    vector_scores = [
        1.0 / (1.0 + max(0.0, float(item.get("distance") or 0.0)))
        for item in vector_recall
        if item.get("distance") is not None
    ]
    vector_score = max(vector_scores, default=0.0)
    target_id = None
    candidates = entity.get("candidates") or []
    if len(candidates) == 1:
        target_id = str(candidates[0].get("target_node_id") or "") or None
    structural_score = graph_score if graph_score else _graph_context_score(subject_node_id, target_id, nodes_by_id)
    normalized_evidence = max(0.0, min(1.0, float(evidence_score or 0.0)))
    joint_score = round(
        0.45 * deterministic_score
        + 0.20 * normalized_evidence
        + 0.20 * structural_score
        + 0.15 * vector_score,
        3,
    )
    if decision in {"AUTO_EXACT_RECALL", "AUTO_ALIAS_RECALL"} and joint_score >= 0.7:
        rerank_decision = "REVIEW_READY_DETERMINISTIC"
    elif vector_score > 0 and joint_score >= 0.45:
        rerank_decision = "REVIEW_REQUIRED_VECTOR_SUPPORT"
    else:
        rerank_decision = "REVIEW_REQUIRED"
    return {
        "deterministic_score": round(deterministic_score, 3),
        "evidence_score": round(normalized_evidence, 3),
        "graph_context_score": round(structural_score, 3),
        "vector_recall_score": round(vector_score, 3),
        "joint_score": joint_score,
        "rerank_decision": rerank_decision,
        "policy": "RECALL_ONLY_HUMAN_REVIEW",
    }


def _vector_recall(source_scenic_id: str, query: str, *, limit: int = 3) -> list[dict[str, Any]]:
    if not query:
        return []
    try:
        with ai_session_scope() as db:
            count = db.execute(
                text(
                    """
                    select count(*)
                    from text_embeddings te
                    join knowledge_chunks kc on kc.id = te.chunk_id
                    where kc.source_scenic_id=:source_scenic_id
                      and kc.source_type='domain_kb'
                    """
                ),
                {"source_scenic_id": str(source_scenic_id)},
            ).scalar_one()
        if not count:
            return []
        from src.rag.service.embedding_service import search_text_embedding_chunks

        rows = search_text_embedding_chunks(str(source_scenic_id), query, limit=limit)
        return [
            {
                "chunk_id": item.get("chunk_id"),
                "source_doc_id": item.get("source_doc_id"),
                "distance": round(float(item.get("distance") or 0.0), 6),
                "score": round(float(item.get("score") or 0.0), 6),
            }
            for item in rows
            if item.get("chunk_id") is not None
        ]
    except Exception as exc:
        logger.warning("G3 vector recall skipped for %s: %s", source_scenic_id, exc)
        return []


def _multimodal_evidence(
    row: dict[str, Any],
    *,
    subject_name: str,
    assets: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Rerank existing image evidence for a text candidate.

    Images are evidence augmentation only in M1: this function never creates a
    candidate, resolves an entity, or changes a review status.
    """
    documents: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    for asset in assets:
        image = _resolve_media_url(asset.get("url") or asset.get("source_url"))
        if not image:
            continue
        text_parts = [
            str(asset.get("title") or "").strip(),
            str(asset.get("caption") or "").strip(),
            str(asset.get("ocr_text") or "").strip(),
        ]
        document_text = " ".join(part for part in text_parts if part)[:3000]
        documents.append({"text": document_text or None, "image": image})
        asset_rows.append(
            {
                "asset_id": asset.get("id"),
                "source_node_id": asset.get("source_node_id"),
                "title": asset.get("title"),
                "caption": asset.get("caption"),
                "image": image,
            }
        )
    if not documents:
        return None

    predicate = str(row.get("predicate") or "").strip()
    value = str(row.get("object_value") or row.get("object_name") or "").strip()
    quote = str(row.get("quote") or "").strip()
    query_text = " ".join(part for part in (subject_name, predicate, value, quote) if part)[:3000]
    scores = score_documents(
        query={"text": query_text},
        documents=documents,
        instruction="Rank image evidence that directly supports the supplied knowledge claim.",
    )
    if not scores:
        return {
            "version": M1_VERSION,
            "status": "UNAVAILABLE",
            "asset_count": len(asset_rows),
            "policy": "EVIDENCE_AUGMENTATION_REVIEW_ONLY",
        }
    ranked = []
    for asset, score in zip(asset_rows, scores):
        ranked.append({**asset, "score": round(float(score), 4)})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return {
        "version": M1_VERSION,
        "status": "SCORED",
        "model": "Qwen3-VL-Reranker-2B",
        "asset_count": len(ranked),
        "image_support_score": ranked[0]["score"] if ranked else 0.0,
        "assets": ranked,
        "policy": "EVIDENCE_AUGMENTATION_REVIEW_ONLY",
    }
def normalize_candidate_batch(
    *,
    source_scenic_id: str,
    candidate_ids: list[int | str],
    vector_limit: int = 3,
    max_vector_queries: int = 6,
    max_multimodal_queries: int = 3,
) -> dict[str, Any]:
    """补充 G3 召回和 M1 证据信号。

    输入为领域和候选 ID 预算；输出为更新数量、召回数量和错误。由 graph.normalize_candidates 调用，
    只更新 metadata，不改审核状态、目标节点或实体合并结果。
    """
    ids = list(dict.fromkeys(int(item) for item in candidate_ids if str(item).isdigit()))
    if not ids:
        return {"candidate_count": 0, "updated_count": 0, "vector_recall_count": 0, "errors": []}

    try:
        with ai_session_scope() as db:
            candidates = [
                dict(row)
                for row in db.execute(
                    text(
                        """
                        select id, source_node_id, claim_type, predicate, object_value,
                               object_name, object_type, entity_resolution_status,
                               evidence_score, confidence, quote,
                               possible_nodes, metadata
                        from semantic_claim_candidates
                        where id = any(:ids) and source_scenic_id=:source_scenic_id
                        """
                    ),
                    {"ids": ids, "source_scenic_id": str(source_scenic_id)},
                ).mappings().all()
            ]
            nodes = [
                dict(row)
                for row in db.execute(
                    text(
                        """
                        select source_node_id, node_name, node_type,
                               parent_source_node_id, properties
                        from semantic_nodes
                        where source_scenic_id=:source_scenic_id
                        order by id
                        limit 5000
                        """
                    ),
                    {"source_scenic_id": str(source_scenic_id)},
                ).mappings().all()
            ]
            assets: list[dict[str, Any]] = []
            source_node_ids = list({str(row.get("source_node_id")) for row in candidates if row.get("source_node_id")})
            if source_node_ids:
                try:
                    assets = [
                        dict(row)
                        for row in db.execute(
                            text(
                                """
                                select id, source_node_id, url, title, caption, ocr_text, source_url
                                from node_assets
                                where source_scenic_id=:source_scenic_id
                                  and asset_type='image'
                                  and source_node_id = any(:source_node_ids)
                                order by id desc
                                limit 300
                                """
                            ),
                            {"source_scenic_id": str(source_scenic_id), "source_node_ids": source_node_ids},
                        ).mappings().all()
                    ]
                except Exception as exc:
                    logger.warning("M1 image evidence lookup skipped for %s: %s", source_scenic_id, exc)
    except Exception as exc:
        logger.warning("G3 candidate load failed: %s", exc)
        return {"candidate_count": len(ids), "updated_count": 0, "vector_recall_count": 0, "errors": [str(exc)]}

    exact, aliases = _node_indexes(nodes)
    nodes_by_id = {
        str(row.get("source_node_id") or ""): row
        for row in nodes
        if row.get("source_node_id")
    }
    patches: list[tuple[int, dict[str, Any]]] = []
    errors: list[str] = []
    vector_recall_count = 0
    vector_queries = 0
    multimodal_query_count = 0
    multimodal_scored_count = 0
    vector_cache: dict[str, list[dict[str, Any]]] = {}
    assets_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for asset in assets:
        assets_by_node[str(asset.get("source_node_id") or "")].append(asset)

    for row in candidates:
        try:
            predicate = str(row.get("predicate") or "").strip()
            canonical = canonical_predicate(predicate)
            raw_value = str(row.get("object_value") or row.get("object_name") or "").strip()
            normalized = normalize_text_value(raw_value)
            entity = {}
            vector_recall: list[dict[str, Any]] = []
            if str(row.get("claim_type") or "") == "relation":
                entity = _entity_recall(str(row.get("object_name") or row.get("object_value") or ""), exact, aliases)
                if entity.get("candidate_count", 0) != 1:
                    query = str(row.get("object_name") or row.get("object_value") or "").strip()
                    if query in vector_cache:
                        vector_recall = vector_cache[query]
                    elif vector_queries < max(0, int(max_vector_queries)):
                        vector_recall = _vector_recall(
                            str(source_scenic_id),
                            query,
                            limit=vector_limit,
                        )
                        vector_cache[query] = vector_recall
                        vector_queries += 1
            vector_recall_count += 1 if vector_recall else 0
            rerank = _joint_resolution_score(
                entity,
                vector_recall,
                evidence_score=float(row.get("evidence_score") or 0.0),
                graph_score=0.0,
                subject_node_id=str(row.get("source_node_id") or ""),
                nodes_by_id=nodes_by_id,
            )
            entity_decision = str(entity.get("decision") or "")
            merge_decision = {
                "AUTO_EXACT_RECALL": "AUTO_EXACT_MATCH",
                "AUTO_ALIAS_RECALL": "AUTO_ALIAS_MATCH",
                "REVIEW_REQUIRED": "HUMAN_REVIEW_AMBIGUOUS",
            }.get(entity_decision, "HUMAN_REVIEW_REQUIRED")
            patch = {
                "g3_normalization": {
                    "version": G3_VERSION,
                    "canonical_predicate": canonical,
                    "raw_value": raw_value or None,
                    "normalized_value": normalized or None,
                    "value_mode": "text",
                    "entity_recall": entity or None,
                    "vector_recall": vector_recall,
                    "vector_role": "evidence_recall_only",
                    "merge_decision": merge_decision,
                    "resolution_basis": entity_decision or "NO_GRAPH_MATCH",
                    "joint_resolution": rerank,
                }
            }
            if multimodal_query_count < max(0, int(max_multimodal_queries)):
                node_key = str(row.get("source_node_id") or "")
                multimodal_assets = assets_by_node.get(node_key) or []
                if multimodal_assets:
                    subject_name = str((nodes_by_id.get(node_key) or {}).get("node_name") or node_key)
                    multimodal = _multimodal_evidence(
                        row,
                        subject_name=subject_name,
                        assets=multimodal_assets[:3],
                    )
                    multimodal_query_count += 1
                    if multimodal:
                        patch["multimodal_evidence"] = multimodal
                        if multimodal.get("status") == "SCORED":
                            multimodal_scored_count += 1
            patches.append((int(row["id"]), patch))
        except Exception as exc:
            errors.append(f"{row.get('id')}: {exc}")

    if patches:
        with ai_session_scope() as db:
            for candidate_id, patch in patches:
                db.execute(
                    text(
                        """
                        update semantic_claim_candidates
                        set metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                            updated_at=now()
                        where id=:candidate_id and source_scenic_id=:source_scenic_id
                        """
                    ),
                    {
                        "candidate_id": candidate_id,
                        "source_scenic_id": str(source_scenic_id),
                        "patch": json.dumps(patch, ensure_ascii=False),
                    },
                )

    return {
        "candidate_count": len(candidates),
        "updated_count": len(patches),
        "vector_recall_count": vector_recall_count,
        "vector_query_count": vector_queries,
        "multimodal_query_count": multimodal_query_count,
        "multimodal_scored_count": multimodal_scored_count,
        "errors": errors,
    }
