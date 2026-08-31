"""候选实体解析、归一化与聚合服务。

文件用途：将开放发现得到的实体提及与正式图谱节点对齐，并生成可审计的统一候选。
输入：GrowthRun 原始声明、领域节点及其属性/别名。
输出：实体解析结果、CanonicalClaim 聚合结果和 KG Delta 所需的判定信息。
逻辑：先精确/别名匹配，再用 BGE 向量召回和跨模态 Reranker 排序；向量只负责召回，
最终仍由分数间隔、类型与上下文规则决定是否合并，避免相似度直接制造错误节点。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import unicodedata
from collections import defaultdict
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.service.value_normalization_service import canonical_predicate, normalize_text_value
from src.rag.service.entity_type_service import infer_entity_type
from src.rag.service.claim_contract import CanonicalClaim
from src.rag.service.claim_identity_service import claim_keys
from src.rag.service.source_independence_service import source_independence_key

logger = logging.getLogger(__name__)
_VECTOR_INDEX_CACHE: dict[str, list[dict[str, Any]]] = {}
_VECTOR_INDEX_SIGNATURES: dict[str, str] = {}
_VECTOR_INDEX_LOCK = threading.Lock()
_VECTOR_MIN_SCORE = 0.84
_VECTOR_MIN_MARGIN = 0.08


def _normalized_node_type(value: Any) -> str:
    """规范化节点类型用于匹配。输入可为中文名称、英文编码或 code=label；输出稳定的比较键。"""
    raw = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    if "=" in raw:
        code, _label = raw.split("=", 1)
        if code.strip().startswith(("type_", "domain_")) or code.strip() == "ce5":
            raw = code.strip()
    aliases = {
        "poi": "poi", "景点": "poi", "地点": "poi",
        "region": "region", "区域": "region", "地区": "region",
        "organization": "organization", "机构": "organization", "学校": "organization",
        "person": "person", "人物": "person", "building": "building", "建筑": "building",
        "facility": "facility", "设施": "facility", "event": "event", "事件": "event",
        "route": "route", "路线": "route", "program": "program", "项目": "program",
        "concept": "concept", "概念": "concept", "time_literal": "time_literal", "时间": "time_literal",
    }
    return aliases.get(raw, raw)


def _type_compatible(raw_type: Any, candidate_type: Any, candidate_label: Any = None) -> bool:
    """判断抽取类型与正式节点类型是否兼容。

    B 端节点保存的是管理员生成的编码（如 type_xxx），而模型可能返回中文标签；
    candidate_label 由领域 schema 注册表提供，只有编码与标签确实对应时才放行。
    """
    raw = _normalized_node_type(raw_type)
    candidate = _normalized_node_type(candidate_type)
    label = _normalized_node_type(candidate_label)
    return not raw or not candidate or raw == candidate or (label and raw == label)


ALIAS_KEYS = {"alias", "aliases", "别名", "曾用名", "简称", "又名", "英文名"}


def _node_index_text(row: dict[str, Any]) -> str:
    """拼接节点向量索引文本。输入为节点行；输出包含名称、别名、类型和截断描述的中文文本。"""
    properties = row.get("properties") if isinstance(row.get("properties"), dict) else {}
    alias_values: list[str] = []
    for key, value in properties.items():
        key_text = str(key or "").casefold()
        if key_text in ALIAS_KEYS or any(token in key_text for token in ("alias", "鍒悕", "鏇剧敤", "绠€绉?")):
            alias_values.extend(_values(value))
    parts = [
        str(row.get("node_name") or ""),
        *alias_values,
        str(row.get("node_type") or ""),
        str(row.get("node_type_label") or ""),
    ]
    description = str(row.get("description") or "").strip()
    if description:
        parts.append(description[:240])
    return " ".join(item.strip() for item in parts if item and item.strip())


def _load_or_build_persisted_index(
    source_scenic_id: str,
    rows: list[dict[str, Any]],
    *,
    embed_texts: Any,
    np: Any,
) -> list[dict[str, Any]]:
    """读取或生成节点向量索引。

    输入：领域节点、文本向量函数和 NumPy 模块。输出：带节点标识、类型、父节点和向量的索引列表。由 ``_vector_entity_recall`` 调用；只写入 G3 专用向量表。
    """
    texts = [_node_index_text(row) for row in rows]
    row_by_id = {str(row.get("source_node_id") or ""): (row, text_value) for row, text_value in zip(rows, texts)}
    hashes = {node_id: _hash(text_value) for node_id, (_, text_value) in row_by_id.items()}
    loaded: dict[str, dict[str, Any]] = {}
    try:
        with ai_session_scope() as db:
            db_rows = db.execute(
                text(
                    """
                    select source_node_id, label_text, content_hash, embedding::text as embedding_text
                    from semantic_growth_entity_embeddings
                    where source_scenic_id=:sid and source_node_id=any(:node_ids)
                    """
                ),
                {"sid": str(source_scenic_id), "node_ids": list(row_by_id) or ["__none__"]},
            ).mappings().all()
        for db_row in db_rows:
            node_id = str(db_row.get("source_node_id") or "")
            if not node_id or db_row.get("content_hash") != hashes.get(node_id):
                continue
            embedding_text = str(db_row.get("embedding_text") or "").strip().strip("[]")
            vector = np.fromstring(embedding_text, sep=",", dtype=np.float32)
            if vector.size:
                loaded[node_id] = {"vector": vector.tolist(), "label_text": db_row.get("label_text") or ""}
    except Exception as exc:
        logger.info("G3 persisted entity index unavailable, using lazy memory index: %s", exc)

    missing_ids = [node_id for node_id in row_by_id if node_id not in loaded]
    if missing_ids:
        missing_texts = [row_by_id[node_id][1] for node_id in missing_ids]
        vectors = embed_texts(missing_texts, batch_size=16)
        for node_id, vector, label_text in zip(missing_ids, vectors, missing_texts):
            loaded[node_id] = {"vector": vector, "label_text": label_text}
        try:
            from src.rag.service.embedding_service import to_pgvector, MODEL_NAME
            with ai_session_scope() as db:
                for node_id in missing_ids:
                    row, label_text = row_by_id[node_id]
                    db.execute(
                        text(
                            """
                            insert into semantic_growth_entity_embeddings
                                (source_scenic_id, source_node_id, label_text, content_hash, embedding, model_name, updated_at)
                            values (:sid, :node_id, :label_text, :content_hash, cast(:embedding as vector), :model_name, now())
                            on conflict (source_scenic_id, source_node_id) do update set
                                label_text=excluded.label_text,
                                content_hash=excluded.content_hash,
                                embedding=excluded.embedding,
                                model_name=excluded.model_name,
                                updated_at=now()
                            """
                        ),
                        {
                            "sid": str(source_scenic_id),
                            "node_id": node_id,
                            "label_text": label_text,
                            "content_hash": hashes[node_id],
                            "embedding": to_pgvector(loaded[node_id]["vector"]),
                            "model_name": MODEL_NAME,
                        },
                    )
        except Exception as exc:
            logger.info("G3 entity vector persistence skipped: %s", exc)
    index = []
    for node_id, (row, label_text) in row_by_id.items():
        entry = loaded.get(node_id)
        if not entry:
            continue
        index.append({
            "node_id": node_id,
            "name": row.get("node_name"),
            "node_type": row.get("node_type"),
            "node_type_label": row.get("node_type_label"),
            "parent_node_id": str(row.get("parent_source_node_id") or ""),
            "text": label_text,
            "vector": entry["vector"],
        })
    return index


def _rerank_entity_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    raw_type: str = "",
    context_node_id: str | None = None,
) -> list[dict[str, Any]]:
    """对 BGE 召回结果进行 Reranker 重排。

    输入：实体提及、候选列表、原始类型和上下文节点。输出：增加重排分数的候选列表。失败时保持原顺序，不直接合并实体。
    """
    if not candidates or str(os.getenv("G3_RERANKER_ENABLED", "1")).lower() in {"0", "false", "no"}:
        return candidates
    try:
        from src.semantic_growth.multimodal_reranker_client import score_documents
        documents = [
            {"text": item.get("text") or item.get("name") or "", "metadata": {"node_id": item.get("node_id"), "node_type": item.get("node_type")}}
            for item in candidates
        ]
        scores = score_documents(
            query={"text": query, "node_type": raw_type, "context_node_id": context_node_id or ""},
            documents=documents,
            instruction="仅对候选实体与证据中的实体提及进行相关性重排，不判断事实真伪，不直接合并实体。",
            timeout=float(os.getenv("G3_RERANKER_TIMEOUT", "4")),
        )
        if not scores:
            return candidates
        enriched = []
        for item, rerank_score in zip(candidates, scores):
            vector_score = float(item.get("score") or 0.0)
            context_bonus = float(item.get("context_bonus") or 0.0)
            enriched.append({
                **item,
                "reranker_score": round(float(rerank_score), 6),
                "rank_score": round(0.65 * vector_score + 0.25 * float(rerank_score) + context_bonus, 6),
            })
        enriched.sort(key=lambda item: item.get("rank_score", 0.0), reverse=True)
        return enriched
    except Exception as exc:
        logger.info("G3 entity reranker skipped: %s", exc)
        return candidates


def _merge_entity_recall_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    '''合并文本和图片召回候选并按节点去重。

    输入：来自不同召回通道的节点候选；输出：按节点 ID 去重后的候选。
    同一节点保留更高分，并记录所有召回方式，供审计页面解释候选来源。
    '''
    merged: dict[str, dict[str, Any]] = {}
    for item in candidates:
        node_id = str(item.get("node_id") or "").strip()
        if not node_id:
            continue
        current = merged.get(node_id)
        if current is None:
            merged[node_id] = dict(item)
            continue
        current_score = float(current.get("final_score") or current.get("score") or 0.0)
        item_score = float(item.get("final_score") or item.get("score") or 0.0)
        methods = list(dict.fromkeys([
            *(current.get("recall_methods") or [current.get("recall_method") or "TEXT_VECTOR_RECALL"]),
            *(item.get("recall_methods") or [item.get("recall_method") or "TEXT_VECTOR_RECALL"]),
        ]))
        if item_score > current_score:
            replacement = dict(item)
            replacement["recall_methods"] = methods
            merged[node_id] = replacement
        else:
            current["recall_methods"] = methods
            if item.get("image_vector_score") is not None:
                current["image_vector_score"] = item["image_vector_score"]
                current["image_vector_distance"] = item.get("image_vector_distance")
                current["image_source_asset_id"] = item.get("image_source_asset_id")
    result = list(merged.values())
    result.sort(key=lambda item: float(item.get("final_score") or item.get("score") or 0.0), reverse=True)
    return result[: max(1, int(limit))]


def _vector_entity_recall(
    source_scenic_id: str,
    query: str,
    rows: list[dict[str, Any]],
    *,
    limit: int = 5,
    context_node_id: str | None = None,
    raw_type: str = "",
    image_asset_id: int | None = None,
) -> list[dict[str, Any]]:
    """用文本向量和可用的图片向量召回实体候选，再调用已有 Reranker。

    输入：领域 ID、实体提及、正式节点和 Top-K/上下文参数。输出：节点候选及向量、重排、上下文分数。由 ``_resolve_name`` 调用，结果只参与决策。
    """
    if not query:
        return []
    image_candidates: list[dict[str, Any]] = []
    if image_asset_id is not None:
        try:
            from src.rag.service.image_embedding_service import recall_image_asset_nodes

            image_candidates = recall_image_asset_nodes(
                source_scenic_id,
                int(image_asset_id),
                limit=max(1, int(limit)),
            )
        except Exception as exc:
            logger.info("image vector entity recall skipped: %s", exc)
    if not rows:
        return image_candidates[: max(1, int(limit))]
    try:
        from src.rag.service.embedding_service import embed_texts
        import numpy as np

        index_signature = hashlib.sha256(
            json.dumps(
                [(str(row.get("source_node_id") or ""), _node_index_text(row)) for row in rows],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        with _VECTOR_INDEX_LOCK:
            index = _VECTOR_INDEX_CACHE.get(str(source_scenic_id))
            if index is None or _VECTOR_INDEX_SIGNATURES.get(str(source_scenic_id)) != index_signature:
                index = _load_or_build_persisted_index(source_scenic_id, rows, embed_texts=embed_texts, np=np)
                _VECTOR_INDEX_CACHE[str(source_scenic_id)] = index
                _VECTOR_INDEX_SIGNATURES[str(source_scenic_id)] = index_signature
        query_vector = np.asarray(embed_texts([query], batch_size=1)[0], dtype=np.float32)
        scored = []
        for item in index:
            vector = np.asarray(item["vector"], dtype=np.float32)
            score = float(np.dot(query_vector, vector))
            context_bonus = 0.04 if context_node_id and item.get("parent_node_id") == str(context_node_id) else 0.0
            scored.append({
                "node_id": item["node_id"],
                "name": item["name"],
                "node_type": item["node_type"],
                "text": item.get("text") or item.get("name") or "",
                "score": round(score, 6),
                "context_bonus": context_bonus,
                "final_score": round(score + context_bonus, 6),
            })
        scored.sort(key=lambda item: item["final_score"], reverse=True)
        for item in scored:
            item["recall_method"] = "TEXT_VECTOR_RECALL"
            item["recall_methods"] = ["TEXT_VECTOR_RECALL"]
        combined = _merge_entity_recall_candidates(
            [*scored, *image_candidates],
            limit=max(1, int(limit) * 2),
        )
        reranked = _rerank_entity_candidates(
            query,
            combined,
            raw_type=raw_type,
            context_node_id=context_node_id,
        )
        return reranked[: max(1, int(limit))]
    except Exception as exc:
        logger.warning("G3 entity vector recall skipped for %s: %s", source_scenic_id, exc)
        if image_candidates:
            return _rerank_entity_candidates(
                query,
                image_candidates[: max(1, int(limit))],
                raw_type=raw_type,
                context_node_id=context_node_id,
            )[: max(1, int(limit))]
        return []


def _hash(value: Any) -> str:
    """计算稳定 SHA-256。输入任意可 JSON 序列化值；输出用于索引内容校验和候选身份的十六进制哈希。"""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _looks_like_non_entity_mention(mention: Any, raw_type: Any = "", predicate: Any = "") -> bool:
    """识别句子或命题而非实体名称。输入实体提及及上下文；输出 True 表示应分流，避免把完整句子建成节点。"""
    text = unicodedata.normalize("NFKC", str(mention or "")).strip()
    if not text:
        return True
    if len(text) > 40 and re.search(r"[，。；：:,.!?！？]", text):
        return True
    clause_cues = ("是", "为", "之一", "储库", "表明", "说明", "导致", "形成", "演化", "过程", "阶段")
    return len(text) >= 8 and any(cue in text for cue in clause_cues)


def normalize_entity_name(value: Any) -> str:
    """生成实体名称比较键。输入原始名称；输出去空白、大小写和标点差异后的稳定键。"""
    value = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return "".join(char for char in value if char.isalnum())


def normalize_discovered_predicate(value: Any) -> str:
    """规范化开放发现谓词。输入证据中的原始谓词；输出统一谓词编码，供事实身份和冲突策略使用。"""
    raw = unicodedata.normalize("NFKC", str(value or "")).strip()
    raw = re.sub(r"^(?:于)?\d{2,4}年(?:同年)?", "", raw).strip()
    raw = re.sub(r"^(?:同年|随后|之后|当时)", "", raw).strip()
    aliases = {
        "坐落于": "位于",
        "地处": "位于",
        "地理位置": "located_in",
        "所在地": "located_in",
        "前身": "predecessor",
        "建校前身": "predecessor",
        "曾用名": "former_name",
        "隶属机构": "parent_organization",
        "直属": "parent_organization",
        "隶属": "parent_organization",
        "合并组成单位": "merged_with",
        "由...合并组建": "merged_with",
        "入选项目": "selected_program",
        "涌现出": "notable_person",
        "更名时间为": "更名时间",
        "入选时间为": "入选时间",
        "创立于": "始建时间",
        "始建于": "始建时间",
        "秉承校训": "校训",
        "彰显文化特质": "文化特质",
    }
    return canonical_predicate(aliases.get(raw, raw))


def _values(value: Any) -> list[str]:
    """把属性值拆成可比较的字符串列表。输入支持标量、列表和 value/values 字典；输出去空白后的值列表。"""
    if isinstance(value, dict):
        value = value.get("value") or value.get("values") or value.get("display_value")
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_values(item))
        return result
    return [item.strip() for item in re.split(r"[,，、;/；|]", str(value or "")) if item.strip()]


def _node_indexes(source_scenic_id: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """加载领域节点并建立精确名、别名和正式节点索引。输入领域 ID；输出三个索引，供实体解析复用。"""
    with ai_session_scope() as db:
        rows = [
            dict(row)
            for row in db.execute(
                text(
                    """
                    select source_node_id, parent_source_node_id, node_name, node_type, description, properties
                    from semantic_nodes where source_scenic_id=:sid
                    """
                ),
                {"sid": str(source_scenic_id)},
            ).mappings().all()
        ]
        # 类型标签来自独立注册表，不从节点名称猜测，注册表不存在时保持兼容。
        try:
            labels = {
                str(item["type_code"]): str(item["type_label"])
                for item in db.execute(
                    text("select type_code, type_label from semantic_domain_type_registry where source_scenic_id=:sid"),
                    {"sid": str(source_scenic_id)},
                ).mappings().all()
            }
        except Exception:
            labels = {}
        for row in rows:
            row["node_type_label"] = labels.get(str(row.get("node_type") or ""))
    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aliases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        name_key = normalize_entity_name(row.get("node_name"))
        if name_key:
            exact[name_key].append(row)
        properties = row.get("properties") if isinstance(row.get("properties"), dict) else {}
        for key, value in properties.items():
            if str(key or "").strip().casefold() not in ALIAS_KEYS:
                continue
            for alias in _values(value):
                alias_key = normalize_entity_name(alias)
                if alias_key:
                    aliases[alias_key].append(row)
    return exact, aliases, rows


def _persist_node_candidate(
    db: Any,
    *,
    source_scenic_id: str,
    growth_run_id: str,
    name: str,
    raw_type: str,
    evidence_unit_ids: list[int],
    confidence: float,
    predicate: str = "",
) -> int:
    inferred_type, inferred_confidence, inference_method = infer_entity_type(raw_type, predicate=predicate, mention=name)
    raw_type = inferred_type or raw_type
    normalized = normalize_entity_name(name)
    # Keep the same stable identity contract as the existing entity resolver
    # so open discovery cannot create a parallel duplicate candidate.
    uid = _hash(
        {
            "domain": str(source_scenic_id),
            "name": normalized,
            "suggested_type": raw_type or "",
        }
    )
    row = db.execute(
        text(
            """
            insert into semantic_node_candidates (
                candidate_uid, trace_id, source_scenic_id, name, normalized_name,
                node_type, raw_type, suggested_type, type_confidence, entity_group_key,
                evidence_ids, confidence, source_count, entity_resolution_status,
                status, metadata, updated_at
            ) values (
                :uid, :trace_id, :sid, :name, :normalized,
                :node_type, :raw_type, :suggested_type, :type_confidence, :group_key,
                '[]'::jsonb, :confidence, :source_count, 'NEW_ENTITY',
                'PENDING', cast(:metadata as jsonb), now()
            ) on conflict (candidate_uid) do update set
                trace_id=excluded.trace_id,
                confidence=greatest(semantic_node_candidates.confidence, excluded.confidence),
                source_count=greatest(semantic_node_candidates.source_count, excluded.source_count),
                metadata=semantic_node_candidates.metadata || excluded.metadata,
                updated_at=now()
            returning id
            """
        ),
        {
            "uid": uid,
            "trace_id": f"{growth_run_id}:open:entity",
            "sid": str(source_scenic_id),
            "name": str(name),
            "normalized": normalized,
            "node_type": raw_type or None,
            "raw_type": raw_type or None,
            "suggested_type": raw_type or None,
            "type_confidence": float(confidence or 0.0),
            "group_key": uid[:32],
            "confidence": float(confidence or 0.0),
            "source_count": max(1, len(set(evidence_unit_ids))),
            "metadata": json.dumps(
                {
                    "growth_run_id": growth_run_id,
                    "evidence_unit_ids": list(dict.fromkeys(evidence_unit_ids)),
                    "discovery_track": "OPEN_DISCOVERY",
                    "type_inference_method": inference_method,
                    "type_inference_confidence": inferred_confidence,
                },
                ensure_ascii=False,
            ),
        },
    ).mappings().one()
    return int(row["id"])


def _resolve_name(
    db: Any,
    *,
    name: str,
    raw_type: str,
    source_scenic_id: str,
    growth_run_id: str,
    evidence_unit_ids: list[int],
    confidence: float,
    exact: dict[str, list[dict[str, Any]]],
    aliases: dict[str, list[dict[str, Any]]],
    published_nodes: list[dict[str, Any]],
    context_node_id: str | None = None,
    predicate: str = "",
    image_asset_id: int | None = None,
) -> dict[str, Any]:
    """解析一个实体提及并返回可解释决策。输入为实体名称、类型、证据和节点索引；输出为 EXACT、ALIAS_MATCH、SEMANTIC_MATCH、AMBIGUOUS 或 NEW_ENTITY 及其分数。由聚合函数调用。
    """
    if _looks_like_non_entity_mention(name, raw_type, predicate):
        return {
            "status": "NON_ENTITY_MENTION",
            "node_id": None,
            "node_candidate_id": None,
            "node_type": "",
            "possible_nodes": [],
            "resolution_method": "CLAUSE_FILTER",
            "vector_top1_score": None,
            "vector_top2_score": None,
            "vector_margin": None,
        }
    key = normalize_entity_name(name)
    exact_rows = exact.get(key, [])
    alias_rows = aliases.get(key, []) if not exact_rows else []
    rows = exact_rows or alias_rows
    if len(rows) == 1:
        return {
            "status": "EXACT" if exact_rows else "ALIAS_MATCH",
            "node_id": str(rows[0].get("source_node_id") or ""),
            "node_candidate_id": None,
            "node_type": str(rows[0].get("node_type") or raw_type or ""),
            "possible_nodes": [],
            "resolution_method": "EXACT" if exact_rows else "ALIAS",
            "vector_top1_score": None,
            "vector_top2_score": None,
            "vector_margin": None,
        }
    if len(rows) > 1:
        return {
            "status": "AMBIGUOUS",
            "node_id": None,
            "node_candidate_id": None,
            "node_type": raw_type or "",
            "possible_nodes": [
                {"node_id": str(row.get("source_node_id") or ""), "name": row.get("node_name"), "node_type": row.get("node_type"), "node_type_label": row.get("node_type_label")}
                for row in rows
            ],
            "resolution_method": "DETERMINISTIC_AMBIGUOUS",
            "vector_top1_score": None,
            "vector_top2_score": None,
            "vector_margin": None,
        }
    vector_candidates = [
        item for item in _vector_entity_recall(
            source_scenic_id,
            name,
            published_nodes,
            context_node_id=context_node_id,
            raw_type=str(raw_type or ""),
            image_asset_id=image_asset_id,
        )
        if _type_compatible(raw_type, item.get("node_type"), item.get("node_type_label"))
    ]
    top1 = vector_candidates[0] if vector_candidates else None
    top2 = vector_candidates[1] if len(vector_candidates) > 1 else None
    top1_score = float((top1 or {}).get("rank_score") or (top1 or {}).get("final_score") or 0.0)
    top2_score = float((top2 or {}).get("rank_score") or (top2 or {}).get("final_score") or 0.0)
    margin = top1_score - top2_score
    vector_projection = [
        {
            "node_id": item.get("node_id"),
            "name": item.get("name"),
            "node_type": item.get("node_type"),
            "node_type_label": item.get("node_type_label"),
            "score": item.get("rank_score") or item.get("final_score"),
            "vector_score": item.get("score"),
            "reranker_score": item.get("reranker_score"),
            "recall_methods": item.get("recall_methods") or [item.get("recall_method")],
            "image_vector_score": item.get("image_vector_score"),
            "image_vector_distance": item.get("image_vector_distance"),
            "image_source_asset_id": item.get("image_source_asset_id"),
        }
        for item in vector_candidates
    ]
    image_recall_count = sum(
        1 for item in vector_candidates
        if "IMAGE_VECTOR_RECALL" in (item.get("recall_methods") or [item.get("recall_method")])
    )
    resolution_metadata = {
        "resolution_method": "TEXT_AND_IMAGE_VECTOR_RECALL" if image_recall_count else "VECTOR_RECALL",
        "vector_top1_score": round(top1_score, 6),
        "vector_top2_score": round(top2_score, 6),
        "vector_margin": round(margin, 6),
        "image_vector_recall_count": image_recall_count,
        "vector_candidates": vector_projection,
    }
    if top1 and top1_score >= _VECTOR_MIN_SCORE and margin >= _VECTOR_MIN_MARGIN:
        return {
            "status": "SEMANTIC_MATCH",
            "node_id": str(top1.get("node_id") or ""),
            "node_candidate_id": None,
            "node_type": str(top1.get("node_type") or raw_type or ""),
            "possible_nodes": vector_projection,
            **resolution_metadata,
        }
    if top1 and top1_score >= (_VECTOR_MIN_SCORE - 0.06):
        return {
            "status": "AMBIGUOUS",
            "node_id": None,
            "node_candidate_id": None,
            "node_type": raw_type or "",
            "possible_nodes": vector_projection,
            **resolution_metadata,
        }
    candidate_id = _persist_node_candidate(
        db,
        source_scenic_id=source_scenic_id,
        growth_run_id=growth_run_id,
        name=name,
        raw_type=raw_type,
        evidence_unit_ids=evidence_unit_ids,
        confidence=confidence,
        predicate=predicate,
    )
    return {
        "status": "NEW_ENTITY",
        "node_id": None,
        "node_candidate_id": candidate_id,
        "node_type": raw_type or "",
        "possible_nodes": [],
        **resolution_metadata,
    }


def resolve_canonicalize_and_aggregate(
    *,
    growth_run_id: str,
    source_scenic_id: str,
    raw_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """执行 G3 实体、谓词和值归一化并按 canonical claim key 聚合。

    输入为 GrowthRun、领域和原始声明；输出为解析/聚合声明及统计。由开放发现编排器调用。
    """
    exact, aliases, published_nodes = _node_indexes(source_scenic_id)
    existing_relation_labels: set[str] = set()
    existing_property_labels: set[str] = set()
    for row in published_nodes:
        properties = row.get("properties") if isinstance(row.get("properties"), dict) else {}
        existing_property_labels.update(canonical_predicate(str(key)) for key in properties)
    with ai_session_scope() as db:
        existing_relation_labels.update(
            canonical_predicate(str(value or ""))
            for value in db.execute(
                text("select distinct coalesce(relation_label, relation_type) from semantic_edges where source_scenic_id=:sid"),
                {"sid": str(source_scenic_id)},
            ).scalars().all()
            if value
        )
        resolved: list[dict[str, Any]] = []
        resolution_counts: dict[str, int] = {}
        for claim in raw_claims:
            if str(claim.get("claim_type") or "").lower() == "background":
                resolution_counts["BACKGROUND"] = resolution_counts.get("BACKGROUND", 0) + 1
                db.execute(
                    text(
                        "update semantic_growth_raw_claims set status='BACKGROUND_RETAINED', "
                        "metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb), updated_at=now() "
                        "where id=:id"
                    ),
                    {
                        "id": int(claim["id"]),
                        "patch": json.dumps(
                            {
                                "semantic_role": claim.get("semantic_role") or "OTHER",
                                "skip_reason": "NON_GRAPH_BACKGROUND_CLAIM",
                            },
                            ensure_ascii=False,
                        ),
                    },
                )
                continue
            unit = claim["evidence_unit"]
            evidence_unit_id = int(unit["id"])
            unit_metadata = unit.get("metadata") if isinstance(unit.get("metadata"), dict) else {}
            image_asset_id = unit_metadata.get("asset_id") if str(unit.get("source_type") or "").lower() == "image" else None
            try:
                image_asset_id = int(image_asset_id) if image_asset_id is not None else None
            except (TypeError, ValueError):
                image_asset_id = None
            subject = _resolve_name(
                db,
                name=claim["subject_text"],
                raw_type=claim.get("subject_type") or "",
                source_scenic_id=source_scenic_id,
                growth_run_id=growth_run_id,
                evidence_unit_ids=[evidence_unit_id],
                confidence=float(claim.get("confidence") or 0.0),
                exact=exact,
                aliases=aliases,
                published_nodes=published_nodes,
                predicate=claim.get("raw_predicate") or "",
                image_asset_id=image_asset_id,
            )
            predicate = normalize_discovered_predicate(claim["raw_predicate"])
            if str(subject.get("status") or "") == "NON_ENTITY_MENTION":
                resolution_counts["NON_ENTITY_MENTION"] = resolution_counts.get("NON_ENTITY_MENTION", 0) + 1
                db.execute(text("update semantic_growth_raw_claims set status='SKIPPED_NON_ENTITY', metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb), updated_at=now() where id=:id"), {"id": int(claim["id"]), "patch": json.dumps({"skip_reason": "subject_is_clause_like_mention"}, ensure_ascii=False)})
                continue
            effective_claim_type = claim["claim_type"]
            if (
                effective_claim_type == "property"
                and predicate in {"located_in", "predecessor", "former_name", "parent_organization", "merged_with", "selected_program", "notable_person"}
                and str(claim.get("object_type") or "") in {"organization", "person", "region", "building", "facility", "poi", "event", "route", "program"}
            ):
                effective_claim_type = "relation"
            target = None
            if effective_claim_type == "relation":
                target = _resolve_name(
                    db,
                    name=claim["object_text"],
                    raw_type=claim.get("object_type") or "",
                    source_scenic_id=source_scenic_id,
                    growth_run_id=growth_run_id,
                    evidence_unit_ids=[evidence_unit_id],
                    confidence=float(claim.get("confidence") or 0.0),
                    exact=exact,
                    aliases=aliases,
                    published_nodes=published_nodes,
                    context_node_id=subject.get("node_id"),
                    predicate=claim.get("raw_predicate") or "",
                    image_asset_id=image_asset_id,
                )
            if target and str(target.get("status") or "") == "NON_ENTITY_MENTION":
                resolution_counts["NON_ENTITY_MENTION"] = resolution_counts.get("NON_ENTITY_MENTION", 0) + 1
                db.execute(text("update semantic_growth_raw_claims set status='SKIPPED_NON_ENTITY', metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb), updated_at=now() where id=:id"), {"id": int(claim["id"]), "patch": json.dumps({"skip_reason": "object_is_clause_like_mention"}, ensure_ascii=False)})
                continue
            for resolution in (subject, target):
                status = str((resolution or {}).get("status") or "")
                if status:
                    resolution_counts[status] = resolution_counts.get(status, 0) + 1
            normalized_value = (
                normalize_entity_name(claim["object_text"])
                if effective_claim_type == "relation"
                else normalize_text_value(str(claim["object_text"] or "")).casefold()
            )
            predicate_labels = existing_relation_labels if effective_claim_type == "relation" else existing_property_labels
            if predicate in predicate_labels:
                predicate_status = "MATCH" if predicate == claim["raw_predicate"] else "SEMANTIC_MATCH"
            else:
                predicate_status = "NEW_RELATION_TYPE" if effective_claim_type == "relation" else "NEW_PROPERTY_TYPE"
            subject_key = subject.get("node_id")
            if not subject_key and subject.get("node_candidate_id"):
                subject_key = f"new:{subject['node_candidate_id']}"
            if not subject_key:
                subject_key = f"ambiguous:{normalize_entity_name(claim['subject_text'])}"
            target_key = ""
            if target:
                target_key = target.get("node_id")
                if not target_key and target.get("node_candidate_id"):
                    target_key = f"new:{target['node_candidate_id']}"
                if not target_key:
                    target_key = f"ambiguous:{normalized_value}"
            claim_type_upper = "RELATION" if effective_claim_type == "relation" else "PROPERTY"
            subject_ref = (
                f"node:{subject_key}" if subject.get("node_id")
                else f"candidate_entity:{subject_key}" if subject.get("node_candidate_id")
                else f"mention:{normalize_entity_name(claim['subject_text'])}"
            )
            object_ref = None
            if claim_type_upper == "RELATION":
                object_ref = (
                    f"node:{target_key}" if target and target.get("node_id")
                    else f"candidate_entity:{target_key}" if target and target.get("node_candidate_id")
                    else f"mention:{normalize_entity_name(claim['object_text'])}"
                )
            canonical_claim = CanonicalClaim(
                domain_id=str(source_scenic_id),
                subject_ref=subject_ref,
                claim_type=claim_type_upper,
                canonical_predicate=predicate,
                normalized_value=normalized_value if claim_type_upper == "PROPERTY" else None,
                object_ref=object_ref,
                temporal_role=str(claim.get("temporal_role") or "") or None,
            )
            identity = claim_keys(canonical_claim)
            # Keep aggregation_key as the compatibility alias for existing
            # growth persistence code, but make it provenance-free.
            aggregation_key = identity["canonical_claim_key"]
            source_key = source_independence_key({
                "source_type": unit.get("source_type"),
                "source_id": unit.get("source_id"),
                "source_doc_id": unit.get("source_doc_id"),
                "source_url": unit.get("source_url"),
                "chunk_id": unit.get("chunk_id"),
                "evidence_unit_uid": unit.get("evidence_unit_uid"),
                "metadata": unit.get("metadata"),
            })
            item = {
                **claim,
                "claim_type": effective_claim_type,
                "subject_resolution": subject,
                "target_resolution": target,
                "canonical_predicate": predicate,
                "normalized_value": normalized_value,
                "predicate_resolution_status": predicate_status,
                "canonical_claim_key": identity["canonical_claim_key"],
                "conflict_scope_key": identity["conflict_scope_key"],
                "aggregation_key": aggregation_key,
                "source_independence_key": source_key,
            }
            resolved.append(item)
            db.execute(
                text(
                    """
                    update semantic_growth_raw_claims
                    set status='RESOLVED', metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb), updated_at=now()
                    where id=:id
                    """
                ),
                {
                    "id": int(claim["id"]),
                    "patch": json.dumps(
                        {
                            "subject_resolution": subject,
                            "target_resolution": target,
                            "canonical_predicate": predicate,
                            "normalized_value": normalized_value,
                            "predicate_resolution_status": predicate_status,
                            "aggregation_key": aggregation_key,
                        },
                        ensure_ascii=False,
                    ),
                },
            )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in resolved:
        groups[item["aggregation_key"]].append(item)
    aggregated: list[dict[str, Any]] = []
    for key, claims in groups.items():
        claims.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
        best = claims[0]
        independent_sources = {item["source_independence_key"] for item in claims if item.get("source_independence_key")}
        evidence_unit_ids = list(dict.fromkeys(int(item["evidence_unit"]["id"]) for item in claims))
        aggregated.append(
            {
                **best,
                "supporting_claims": claims,
                "raw_claim_ids": [int(item["id"]) for item in claims],
                "evidence_unit_ids": evidence_unit_ids,
                "independent_source_count": len(independent_sources),
                "confidence": max(float(item.get("confidence") or 0.0) for item in claims),
            }
        )
    return {
        "raw_claim_count": len(raw_claims),
        "resolved_claim_count": len(resolved),
        "aggregated_count": len(aggregated),
        "aggregated_claims": aggregated,
        "resolution_counts": resolution_counts,
        "semantic_match_count": int(resolution_counts.get("SEMANTIC_MATCH") or 0),
        "ambiguous_resolution_count": int(resolution_counts.get("AMBIGUOUS") or 0),
        "exact_match_count": int(resolution_counts.get("EXACT") or 0),
        "alias_match_count": int(resolution_counts.get("ALIAS_MATCH") or 0),
    }
