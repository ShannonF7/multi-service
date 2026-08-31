from __future__ import annotations

import asyncio
import concurrent.futures
from collections import defaultdict
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.schemas import (
    SemanticCompleteRequest,
    SemanticEvidenceInput,
    SemanticNodeContext,
)
from src.rag.service.semantic_completion_service import complete_semantic_service
from src.rag.service.value_normalization_service import canonical_predicate, normalize_text_value


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result(timeout=600)


def _published_context(source_scenic_id: str, node_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    properties: list[dict[str, Any]] = []
    with ai_session_scope() as db:
        row = db.execute(
            text(
                """
                select properties
                from semantic_nodes
                where source_scenic_id=:source_scenic_id and source_node_id=:node_id
                order by id desc
                limit 1
                """
            ),
            {"source_scenic_id": str(source_scenic_id), "node_id": str(node_id)},
        ).mappings().first()
    raw = (row or {}).get("properties") if row else {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            properties.append({"key": str(key), "value": str(value or "")})
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("key"):
                properties.append({"key": str(item["key"]), "value": str(item.get("value") or "")})
    return properties, []



def _adopted_context(
    source_scenic_id: str,
    node_id: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Load reviewed candidates as strong context, never as published facts."""
    with ai_session_scope() as db:
        rows = db.execute(
            text(
                """
                select id, subject_name, claim_type, predicate, object_value,
                       object_name, source_title, source_url, quote, evidence_ids
                from semantic_claim_candidates
                where source_scenic_id=:source_scenic_id
                  and source_node_id=:node_id
                  and upper(status)='ADOPTED'
                order by reviewed_at desc nulls last, updated_at desc, id desc
                limit :limit
                """
            ),
            {
                "source_scenic_id": str(source_scenic_id),
                "node_id": str(node_id),
                "limit": max(1, min(int(limit), 50)),
            },
        ).mappings().all()
    return [
        {
            "candidate_id": int(row["id"]),
            "subject_name": str(row.get("subject_name") or ""),
            "claim_type": str(row.get("claim_type") or ""),
            "predicate": str(row.get("predicate") or ""),
            "value": str(row.get("object_value") or row.get("object_name") or ""),
            "source_title": str(row.get("source_title") or ""),
            "source_url": str(row.get("source_url") or ""),
            "quote": str(row.get("quote") or "")[:300],
            "evidence_ids": row.get("evidence_ids") if isinstance(row.get("evidence_ids"), list) else [],
            "context_level": "ADOPTED_NOT_PUBLISHED",
        }
        for row in rows
    ]

def build_growth_payload(
    *,
    source_scenic_id: str,
    growth_run_id: str,
    node: dict[str, Any],
    chunks: list[dict[str, Any]],
    open_discovery: bool = False,
) -> SemanticCompleteRequest:
    evidence: list[SemanticEvidenceInput] = []
    source_types: set[str] = set()
    for chunk in chunks:
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue
        is_image_asset = str(chunk.get("asset_type") or "") == "image"
        source_type = "image_asset" if is_image_asset else "domain_kb"
        source_types.add(source_type)
        metadata = dict(chunk.get("metadata") or {}) if isinstance(chunk.get("metadata"), dict) else {}
        if is_image_asset:
            # 图片证据的定位和 OCR 结果只从 node_assets 元数据透传，不能由抽取模型补造。
            metadata.setdefault("asset_id", chunk.get("asset_id"))
            metadata.setdefault("source_doc_id", chunk.get("source_id"))
            metadata.setdefault("page_no", metadata.get("page_no") or metadata.get("page_number"))
            metadata.setdefault("caption", chunk.get("caption"))
            metadata.setdefault("nearby_text", metadata.get("nearby_text") or metadata.get("near_text") or "")
            metadata.setdefault("bbox", metadata.get("bbox"))
        if is_image_asset:
            raw_score = (
                chunk.get("ocr_mean_score")
                or metadata.get("ocr_mean_score")
                or chunk.get("ocr_max_score")
                or metadata.get("ocr_max_score")
                or 0.55
            )
            try:
                evidence_score = min(0.75, max(0.0, float(raw_score)))
            except (TypeError, ValueError):
                evidence_score = 0.55
        else:
            evidence_score = 0.9
        evidence.append(
            SemanticEvidenceInput(
                title=str(chunk.get("title") or chunk.get("source_title") or ("image evidence" if is_image_asset else "domain evidence")),
                content=content[:6000],
                source=str(chunk.get("source_title") or chunk.get("title") or "domain_kb"),
                source_type=source_type,
                source_url=str(chunk.get("source_url") or "") or None,
                quote=str(chunk.get("evidence_text") or content[:500]).strip(),
                score=evidence_score,
                source_doc_id=str(chunk.get("source_id") or "") or None,
                chunk_id=int(chunk["id"]) if chunk.get("id") is not None else None,
                page_no=int(metadata["page_no"]) if str(metadata.get("page_no") or "").isdigit() else None,
                image=str(chunk.get("image_url") or "") or None,
                asset_id=int(chunk["asset_id"]) if chunk.get("asset_id") is not None else None,
                caption=str(chunk.get("caption") or metadata.get("caption") or "") or None,
                nearby_text=str(metadata.get("nearby_text") or "") or None,
                bbox=metadata.get("bbox"),
                ocr_blocks=metadata.get("ocr_blocks") if isinstance(metadata.get("ocr_blocks"), list) else [],
                metadata=metadata,
            )
        )
    existing_properties, existing_relations = _published_context(
        str(source_scenic_id),
        str(node.get("node_id") or ""),
    )
    adopted_context = _adopted_context(
        str(source_scenic_id),
        str(node.get("node_id") or ""),
    )
    if source_types == {"image_asset"}:
        provenance_type = "image_asset"
    elif "image_asset" in source_types:
        provenance_type = "mixed_evidence"
    else:
        provenance_type = "domain_kb"
    return SemanticCompleteRequest(
        scenic_id=str(source_scenic_id),
        node=SemanticNodeContext(
            source_node_id=str(node.get("node_id") or ""),
            name=str(node.get("name") or ""),
            node_type=str(node.get("node_type") or ""),
            parent_name=str(node.get("parent_name") or ""),
            scenic_name=str(node.get("scenic_name") or source_scenic_id),
            description=str(node.get("description") or ""),
        ),
        message=(
            "Discover only verifiable properties, relations, and entities from the supplied evidence. Do not invent facts or use a fixed template; mark unsupported entities as NEW_ENTITY."
            if open_discovery else
            "Extract only verifiable properties and relations for the bound node from the supplied evidence. Do not invent facts or fill missing template fields."
        ),
        target_fields=[],
        relation_intents=[],
        subgraph_depth=0,
        existing_properties=existing_properties,
        existing_relations=existing_relations,
        evidence=evidence,
        max_web_results=0,
        use_web_search=False,
        use_web_extractor=False,
        metadata={
            "completion_mode": "growth_g2",
            "growth_run_id": str(growth_run_id),
            "source_scope": ["provided_evidence"],
            "retrieval_scope": "evidence_open" if open_discovery else "node",
            "open_discovery": bool(open_discovery),
            "question_batch_size": 20,
            "extractor_chunks_per_question": 8,
            "evidence_limit_per_question": 20,
            "target_node_id": str(node.get("node_id") or ""),
            "provenance_type": provenance_type,
            "evidence_source_types": sorted(source_types),
            "adopted_candidate_context": adopted_context,
            "adopted_context_policy": "STRONG_CONTEXT_NOT_PUBLISHED_FACT",
            "adopted_context_is_evidence": False,
        },
    )


def apply_image_candidate_policy(candidate_ids: list[int], *, image_only: bool) -> dict[str, Any]:
    """Keep OCR-only discoveries review-only unless textual evidence is supported."""
    if not image_only or not candidate_ids:
        return {"updated_ids": [], "low_evidence_ids": []}
    with ai_session_scope() as db:
        rows = db.execute(
            text(
                """
                update semantic_claim_candidates
                set status=case
                        when lower(coalesce(evidence_status, ''))='supported' then status
                        else 'LOW_EVIDENCE'
                    end,
                    risk_level=case
                        when lower(coalesce(evidence_status, ''))='supported'
                            then case when upper(coalesce(risk_level, ''))='HIGH' then 'HIGH' else 'MEDIUM' end
                        else 'HIGH'
                    end,
                    publication_policy='MANUAL_REVIEW',
                    metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                    updated_at=now()
                where id=any(:candidate_ids)
                  and upper(coalesce(status, 'PENDING')) not in ('DUPLICATE','ADOPTED','PUBLISHED','REJECTED','INVALIDATED')
                returning id, status
                """
            ),
            {
                "candidate_ids": [int(item) for item in candidate_ids],
                "patch": '{"image_evidence_policy":"OCR_RAW_REVIEW_ONLY","ocr_requires_visual_confirmation":true}',
            },
        ).mappings().all()
    return {
        "updated_ids": [int(row["id"]) for row in rows],
        "low_evidence_ids": [int(row["id"]) for row in rows if str(row.get("status") or "").upper() == "LOW_EVIDENCE"],
    }


def _published_values(source_scenic_id: str, node_id: str) -> dict[str, set[str]]:
    with ai_session_scope() as db:
        row = db.execute(
            text(
                """
                select properties
                from semantic_nodes
                where source_scenic_id=:source_scenic_id and source_node_id=:node_id
                order by id desc
                limit 1
                """
            ),
            {"source_scenic_id": str(source_scenic_id), "node_id": str(node_id)},
        ).mappings().first()
    raw = (row or {}).get("properties") if row else {}
    result: dict[str, set[str]] = defaultdict(set)
    items = raw.items() if isinstance(raw, dict) else []
    for key, value in items:
        result[canonical_predicate(str(key))].add(normalize_text_value(str(value or "")).casefold())
    return result


def filter_published_candidate_ids(source_scenic_id: str, candidate_ids: list[int]) -> dict[str, Any]:
    if not candidate_ids:
        return {"duplicate_ids": [], "kept_ids": []}
    duplicate_ids: list[int] = []
    with ai_session_scope() as db:
        rows = db.execute(
            text(
                """
                select id, source_node_id, claim_type, predicate, object_value, object_name
                from semantic_claim_candidates
                where id = any(:ids)
                """
            ),
            {"ids": [int(item) for item in candidate_ids]},
        ).mappings().all()
        for row in rows:
            if row["claim_type"] != "property":
                continue
            predicate = canonical_predicate(str(row["predicate"] or ""))
            value = normalize_text_value(str(row["object_value"] or row["object_name"] or "")).casefold()
            published = _published_values(str(source_scenic_id), str(row["source_node_id"] or ""))
            if value and value in published.get(predicate, set()):
                duplicate_ids.append(int(row["id"]))
        if duplicate_ids:
            db.execute(
                text(
                    """
                    update semantic_claim_candidates
                    set status='DUPLICATE',
                        metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                        updated_at=now()
                    where id = any(:ids)
                    """
                ),
                {
                    "ids": duplicate_ids,
                    "patch": '{"filtered_reason":"PUBLISHED_FACT_EXISTS","filter_stage":"G2"}',
                },
            )
    return {
        "duplicate_ids": duplicate_ids,
        "kept_ids": [item for item in candidate_ids if item not in set(duplicate_ids)],
    }


def extract_growth_candidates(
    *,
    source_scenic_id: str,
    growth_run_id: str,
    batch: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    published_nodes: list[dict[str, Any]],
    max_nodes: int = 20,
    allow_open_discovery: bool = True,
) -> list[dict[str, Any]]:
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunks_by_consumption = {int(item["consumption_id"]): item for item in batch if item.get("consumption_id") is not None}
    for mention in mentions:
        by_node[str(mention["node_id"])].append(mention)
    node_map = {
        str(node.get("node_id") or node.get("source_node_id") or ""): node
        for node in published_nodes
    }
    results: list[dict[str, Any]] = []
    groups: list[tuple[str, list[dict[str, Any]], bool]] = [
        (node_id, node_mentions, False) for node_id, node_mentions in by_node.items()
    ]
    if allow_open_discovery:
        mentioned_consumptions = {
            int(mention["consumption_id"])
            for mention in mentions
            if mention.get("consumption_id") is not None
        }
        for chunk in batch:
            consumption_id = chunk.get("consumption_id")
            if consumption_id is None or int(consumption_id) in mentioned_consumptions:
                continue
            if not str(chunk.get("content") or "").strip():
                continue
            groups.append((
                f"__evidence__:{int(consumption_id)}",
                [{"consumption_id": int(consumption_id), "node_name": "璇佹嵁鐗囨", "node_type": "evidence_scope"}],
                True,
            ))
    for node_id, node_mentions, open_discovery in groups[: max(1, int(max_nodes))]:
        node = node_map.get(node_id) or {
            "node_id": node_id,
            "name": node_mentions[0].get("node_name") or node_id,
            "node_type": node_mentions[0].get("node_type") or "",
        }
        chunks = [
            chunks_by_consumption[int(mention["consumption_id"])]
            for mention in node_mentions
            if int(mention["consumption_id"]) in chunks_by_consumption
        ]
        payload = build_growth_payload(
            source_scenic_id=str(source_scenic_id),
            growth_run_id=str(growth_run_id),
            node=node,
            chunks=chunks,
            open_discovery=open_discovery,
        )
        trace_id = f"{growth_run_id}:g2:{node_id}"
        try:
            response = _run_async(
                complete_semantic_service(
                    payload,
                    trace_id_override=trace_id,
                )
            )
            candidate_ids = [
                int(getattr(claim, "candidate_id"))
                for claim in (response.candidate_claims or [])
                if getattr(claim, "candidate_id", None)
            ]
            filtered = filter_published_candidate_ids(str(source_scenic_id), candidate_ids)
            image_policy = apply_image_candidate_policy(
                filtered["kept_ids"],
                image_only=bool(chunks) and all(str(chunk.get("asset_type") or "") == "image" for chunk in chunks),
            )
            results.append({
                "node_id": node_id,
                "mention_count": len(node_mentions),
                "candidate_ids": filtered["kept_ids"],
                "duplicate_candidate_ids": filtered["duplicate_ids"],
                "raw_candidate_ids": candidate_ids,
                "image_policy": image_policy,
                "trace_id": trace_id,
                "open_discovery": open_discovery,
                "error": None,
            })
        except Exception as exc:
            results.append({
                "node_id": node_id,
                "mention_count": len(node_mentions),
                "candidate_ids": [],
                "duplicate_candidate_ids": [],
                "raw_candidate_ids": [],
                "trace_id": trace_id,
                "open_discovery": open_discovery,
                "error": str(exc),
            })
    return results

