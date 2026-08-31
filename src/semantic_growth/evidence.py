from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
import re
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from .ocr_client import extract_ocr_batch
from src.rag.service.image_ocr_service import process_image_ocr_batch
from src.rag.service.source_independence_service import source_independence_key

CONSUMER_VERSION = "growth-open-v2"
CHUNK_SCOPE = "__chunk__"
LEASE_SECONDS = 900
logger = logging.getLogger(__name__)


def source_cursor_progress(scope_states: list[str]) -> dict[str, Any]:
    """计算一个证据单元的消费进度。

    输入：同一 source/chunk 下所有 target scope 的消费状态。
    逻辑：只有 PROCESSED 才算成功；缺失、FAILED、RETRYABLE、CLAIMED
    都会让游标保持 OPEN，避免 fan-out 分支未完成却被错误跳过。
    输出：期望分支数、已完成分支数和游标状态。
    """
    states = [str(state or "").upper() for state in scope_states]
    expected = len(states)
    processed = sum(1 for state in states if state == "PROCESSED")
    return {
        "expected_scope_count": expected,
        "processed_scope_count": processed,
        "cursor_state": "ADVANCED" if expected > 0 and expected == processed else "OPEN",
    }


def _ensure_source_cursor(db: Any, identity: dict[str, Any]) -> None:
    """确保一个 source/chunk 有对应游标记录。

    输入：包含 source_scenic_id、source_id、chunk_id、chunk_hash、
    consumer_version 的证据身份；输出：无。
    逻辑：开放式发现和节点对齐共用同一游标表，缺失时幂等创建，
    从而不会出现“消费记录存在但没有游标可审计”的断链。
    """
    db.execute(
        text(
            """
            insert into semantic_growth_source_cursors (
                source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version
            ) values (
                :source_scenic_id, :source_id, :chunk_id, :chunk_hash, :consumer_version
            ) on conflict (source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version)
            do nothing
            """
        ),
        {
            "source_scenic_id": str(identity["source_scenic_id"]),
            "source_id": str(identity["source_id"]),
            "chunk_id": int(identity["chunk_id"]),
            "chunk_hash": str(identity["chunk_hash"]),
            "consumer_version": str(identity["consumer_version"]),
        },
    )


def _source_family_id(item: dict[str, Any]) -> str:
    """计算证据的来源族标识，供 EvidenceUnit 和后续可信度融合复用。

    输入：claim_evidence_batch 返回的证据行；输出：稳定的 document:/image:/web: 标识。
    同一文档的多个 chunk 共用 document 标识；图片按 asset_id 独立归族。
    """
    asset_type = str(item.get("asset_type") or "text").strip().lower()
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return source_independence_key({
        "source_type": asset_type,
        "asset_type": asset_type,
        "asset_id": item.get("asset_id"),
        "source_id": item.get("source_id"),
        "source_doc_id": item.get("source_id") if asset_type != "image" else None,
        "source_url": item.get("source_url"),
        "chunk_id": item.get("id") or item.get("chunk_id"),
        "metadata": metadata,
    })


def _attach_image_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把同文档图片附近的文本上下文写入内存证据行。

    输入：node_assets 查询结果；输出：带 nearby_text/section 元数据的图片行。
    上下文仅来自数据库已存在的文档 chunk，不调用模型、不创建新事实。
    """
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        metadata = dict(item.get("metadata") or {}) if isinstance(item.get("metadata"), dict) else {}
        if str(item.get("asset_type") or "").lower() == "image":
            nearby_text = str(item.get("nearby_text") or "").strip()
            nearby_section = str(item.get("nearby_section") or "").strip()
            if nearby_text:
                metadata.setdefault("nearby_text", nearby_text[:4000])
            if nearby_section:
                metadata.setdefault("section", nearby_section[:500])
        item["metadata"] = metadata
        enriched.append(item)
    return enriched


def _prepare_image_ocr(source_scenic_id: str, image_limit: int) -> None:
    """在领取增长证据前持久化一批缺失 OCR 的图片。

    输入：领域标识和图片批量上限；输出：无。调用已有 image_ocr_service，
    让 OCR 外部调用发生在证据查询事务之前；失败只记录日志，文本证据仍可继续消费。
    """
    if int(image_limit) <= 0:
        return
    try:
        process_image_ocr_batch(
            source_scenic_id=str(source_scenic_id),
            limit=min(int(image_limit), 16),
        )
    except Exception as exc:  # pragma: no cover - 依赖数据库和独立 OCR 服务
        logger.warning("growth image OCR preparation skipped: %s", exc)


def claim_evidence_batch(
    *,
    growth_run_id: str,
    source_scenic_id: str,
    worker_id: str,
    limit: int = 100,
    image_limit: int | None = None,
    consumer_version: str = CONSUMER_VERSION,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    lease_expires = now + timedelta(seconds=LEASE_SECONDS)
    requested_image_limit = (
        int(os.getenv("GROWTH_IMAGE_BATCH_LIMIT", "16"))
        if image_limit is None
        else int(image_limit)
    )
    _prepare_image_ocr(str(source_scenic_id), requested_image_limit)
    with ai_session_scope() as db:
        text_rows = db.execute(
            text(
                """
                select k.id, k.source_scenic_id, k.source_id, k.source_node_id,
                       k.title, k.content, k.evidence_text, k.source_title,
                       k.source_url, k.content_hash, k.created_at,
                       'text' as asset_type, null as asset_id, null as image_url,
                       c.id as consumption_id, c.state as consumption_state,
                       c.lease_owner, c.lease_expires_at
                from knowledge_chunks k
                left join semantic_growth_evidence_consumptions c
                  on c.source_scenic_id = k.source_scenic_id
                 and c.source_id = k.source_id
                 and c.chunk_id = k.id
                 and c.chunk_hash = k.content_hash
                 and c.consumer_version = :consumer_version
                 and c.target_scope = :chunk_scope
                where k.source_scenic_id = :source_scenic_id
                  and k.source_type = 'domain_kb'
                  and (
                    c.id is null
                    or c.state in ('FAILED', 'RETRYABLE')
                    or (c.state = 'CLAIMED' and c.lease_expires_at < now())
                  )
                order by k.created_at asc, k.id asc
                limit :limit
                for update of k skip locked
                """
            ),
            {
                "source_scenic_id": str(source_scenic_id),
                "consumer_version": consumer_version,
                "chunk_scope": CHUNK_SCOPE,
                "limit": max(1, min(int(limit), 200)),
            },
        ).mappings().all()
        # Uploaded/accepted text is persisted as semantic_evidence_items by
        # the A端 sync path.  knowledge_chunks is only the small local-KB
        # index, so reading it alone silently skips most uploaded evidence.
        evidence_rows = db.execute(
            text(
                """
                select coalesce(e.chunk_id, e.id) as id, e.id as evidence_item_id, e.source_scenic_id,
                       coalesce(nullif(e.source_doc_id, ''), 'evidence:' || e.id::text) as source_id,
                       coalesce(e.source_node_id, '__domain__') as source_node_id,
                       coalesce(e.source_title, e.source_doc_id, '上传文本证据') as title,
                       e.content, e.quote as evidence_text,
                       e.source_title, e.source_url,
                       md5(concat_ws('|', e.source_doc_id, e.chunk_id, e.content, e.quote)) as content_hash,
                       e.created_at, 'text' as asset_type, null as asset_id, null as image_url,
                       coalesce(e.source_authority_score, e.source_weight, 0.8) as source_authority,
                       c.id as consumption_id, c.state as consumption_state,
                       c.lease_owner, c.lease_expires_at
                from semantic_evidence_items e
                left join semantic_growth_evidence_consumptions c
                  on c.source_scenic_id = e.source_scenic_id
                 and c.source_id = coalesce(nullif(e.source_doc_id, ''), 'evidence:' || e.id::text)
                 and c.chunk_id = coalesce(e.chunk_id, e.id)
                 and c.chunk_hash = md5(concat_ws('|', e.source_doc_id, e.chunk_id, e.content, e.quote))
                 and c.consumer_version = :consumer_version
                 and c.target_scope = :chunk_scope
                where e.source_scenic_id = :source_scenic_id
                  and lower(coalesce(e.source_type, '')) = 'domain_kb'
                  and e.job_id is null
                  and e.question_id is null
                  and coalesce(nullif(e.content, ''), nullif(e.quote, '')) is not null
                  and (
                    c.id is null
                    or c.state in ('FAILED', 'RETRYABLE')
                    or (c.state = 'CLAIMED' and c.lease_expires_at < now())
                  )
                order by e.created_at asc, e.id asc
                limit :limit
                for update of e skip locked
                """
            ),
            {
                "source_scenic_id": str(source_scenic_id),
                "consumer_version": consumer_version,
                "chunk_scope": CHUNK_SCOPE,
                "limit": max(1, min(int(limit), 200)),
            },
        ).mappings().all()
        image_rows = db.execute(
            text(
                """
                select a.id, a.source_scenic_id,
                       ('asset:' || coalesce(nullif(a.source_asset_id, ''), a.id::text)) as source_id,
                       a.source_node_id, a.title,
                       sn.node_name as source_node_name, sn.node_type as source_node_type,
                       concat_ws(E'\\n', nullif(a.caption, ''), nullif(a.ocr_text, '')) as content,
                       coalesce(a.caption, a.ocr_text, a.title) as evidence_text,
                       a.title as source_title,
                       coalesce(a.url, a.source_url) as source_url,
                       a.caption, a.ocr_text, a.metadata,
                       nearby.nearby_text, nearby.nearby_section,
                       coalesce(a.content_hash, a.file_hash,
                                md5(concat_ws('|', a.url, a.caption, a.ocr_text))) as content_hash,
                       a.created_at, 'image' as asset_type, a.id as asset_id,
                       coalesce(a.url, a.source_url) as image_url,
                       c.id as consumption_id, c.state as consumption_state,
                       c.lease_owner, c.lease_expires_at
                from node_assets a
                left join semantic_nodes sn
                  on sn.source_scenic_id = a.source_scenic_id
                 and sn.source_node_id = a.source_node_id
                left join lateral (
                    select kc.content as nearby_text,
                           coalesce(kc.metadata->>'section', kc.metadata->>'heading', kc.title) as nearby_section
                    from knowledge_chunks kc
                    where kc.source_scenic_id = a.source_scenic_id
                      and kc.source_type = 'domain_kb'
                      and kc.source_id = a.metadata->>'doc_id'
                      and a.metadata->>'page_no' is not null
                      and (
                          kc.metadata->>'page_no' = a.metadata->>'page_no'
                          or kc.metadata->>'chunk_index' = a.metadata->>'page_no'
                      )
                    order by kc.id asc
                    limit 1
                ) nearby on true
                left join semantic_growth_evidence_consumptions c
                  on c.source_scenic_id = a.source_scenic_id
                 and c.source_id = ('asset:' || coalesce(nullif(a.source_asset_id, ''), a.id::text))
                 and c.chunk_id = a.id
                 and c.chunk_hash = coalesce(a.content_hash, a.file_hash,
                                             md5(concat_ws('|', a.url, a.caption, a.ocr_text)))
                 and c.consumer_version = :consumer_version
                 and c.target_scope = :chunk_scope
                where a.source_scenic_id = :source_scenic_id
                  and a.asset_type = 'image'
                  and coalesce(a.url, a.source_url, '') <> ''
                  and (
                    c.id is null
                    or c.state in ('FAILED', 'RETRYABLE')
                    or (c.state = 'CLAIMED' and c.lease_expires_at < now())
                  )
                order by a.created_at asc, a.id asc
                limit :image_limit
                for update of a skip locked
                """
            ),
            {
                "source_scenic_id": str(source_scenic_id),
                "consumer_version": consumer_version,
                "chunk_scope": CHUNK_SCOPE,
                "limit": max(1, min(int(limit), 200)),
                "image_limit": max(0, min(requested_image_limit, int(limit), 16)),
            },
        ).mappings().all()
        # Existing OCR is hosted in its own paddle_ocr environment. Enrich
        # only assets missing caption/OCR; failed or empty OCR assets are not
        # claimed, so a later GrowthRun can retry them.
        pending_ocr = [
            {"asset_id": row.get("asset_id"), "image": row.get("image_url")}
            for row in image_rows
            if row.get("asset_id") is not None
            and row.get("image_url")
            and not str(row.get("caption") or row.get("ocr_text") or "").strip()
            and str((row.get("metadata") or {}).get("ocr_status") or "").upper()
            not in {"SUCCEEDED", "NO_TEXT"}
        ]
        if pending_ocr:
            ocr_rows = extract_ocr_batch(pending_ocr)
            enriched_image_rows = []
            for row in image_rows:
                ocr = ocr_rows.get(int(row["asset_id"])) if row.get("asset_id") is not None else None
                if ocr and str(ocr.get("ocr_text") or "").strip():
                    enriched = dict(row)
                    enriched["content"] = str(ocr.get("ocr_text") or "").strip()
                    enriched["evidence_text"] = enriched["content"]
                    enriched["ocr_text"] = enriched["content"]
                    enriched["ocr_max_score"] = float(ocr.get("max_score") or 0.0)
                    enriched["ocr_mean_score"] = float(ocr.get("mean_score") or 0.0)
                    enriched["ocr_min_score"] = float(ocr.get("min_score") or 0.0)
                    enriched["ocr_line_count"] = int(ocr.get("line_count") or 0)
                    enriched_metadata = dict(enriched.get("metadata") or {}) if isinstance(enriched.get("metadata"), dict) else {}
                    enriched_metadata.update({
                        "ocr_status": "SUCCEEDED",
                        "ocr_model": str(ocr.get("model") or "paddleocr")[:128],
                        "ocr_raw_text": str(ocr.get("ocr_raw_text") or ""),
                        "ocr_blocks": ocr.get("ocr_blocks") if isinstance(ocr.get("ocr_blocks"), list) else [],
                        "ocr_max_score": enriched["ocr_max_score"],
                        "ocr_mean_score": enriched["ocr_mean_score"],
                        "ocr_min_score": enriched["ocr_min_score"],
                        "ocr_line_count": enriched["ocr_line_count"],
                    })
                    enriched["metadata"] = enriched_metadata
                    enriched_image_rows.append(enriched)
                elif str(row.get("content") or "").strip():
                    enriched_image_rows.append(dict(row))
            image_rows = enriched_image_rows
        image_rows = _attach_image_context(image_rows)
        # Text is the primary open-discovery channel. Do not let a large old
        # image library crowd uploaded text out of every batch. Duplicate text
        # mirrored in both stores is consumed once per normalized content.
        text_pool = sorted(
            [*text_rows, *evidence_rows],
            key=lambda item: (item.get("created_at") or now, int(item.get("id") or 0)),
        )
        unique_text: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        for row in text_pool:
            content_key = re.sub(r"\s+", "", str(row.get("content") or row.get("evidence_text") or ""))
            if not content_key or content_key in seen_text:
                continue
            seen_text.add(content_key)
            unique_text.append(dict(row))
        row_limit = max(1, min(int(limit), 200))
        rows = unique_text[:row_limit]
        if len(rows) < row_limit and requested_image_limit > 0:
            rows.extend(
                sorted(
                    image_rows,
                    key=lambda item: (item.get("created_at") or now, int(item.get("id") or 0)),
                )[: min(requested_image_limit, row_limit - len(rows))]
            )
        claimed: list[dict[str, Any]] = []
        for row in rows:
            # 来源族只用于留痕，不改变消费幂等键。
            source_family_id = _source_family_id(row)
            params = {
                "growth_run_id": str(growth_run_id),
                "source_scenic_id": str(row["source_scenic_id"]),
                "source_id": str(row["source_id"]),
                "chunk_id": int(row["id"]),
                "chunk_hash": str(row["content_hash"] or ""),
                "consumer_version": consumer_version,
                "target_scope": CHUNK_SCOPE,
                "worker_id": str(worker_id),
                "lease_expires_at": lease_expires,
            }
            db.execute(
                text(
                    """
                    insert into semantic_growth_evidence_consumptions (
                        growth_run_id, source_scenic_id, source_id, chunk_id, chunk_hash,
                        consumer_version, target_scope, state, lease_owner,
                        lease_expires_at, attempt_count, claimed_at, updated_at
                    ) values (
                        :growth_run_id, :source_scenic_id, :source_id, :chunk_id, :chunk_hash,
                        :consumer_version, :target_scope, 'CLAIMED', :worker_id,
                        :lease_expires_at, 1, now(), now()
                    )
                    on conflict (source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version, target_scope)
                    do update set
                        growth_run_id = excluded.growth_run_id,
                        state = 'CLAIMED',
                        lease_owner = excluded.lease_owner,
                        lease_expires_at = excluded.lease_expires_at,
                        attempt_count = semantic_growth_evidence_consumptions.attempt_count + 1,
                        claimed_at = now(),
                        updated_at = now(),
                        error = null
                    """
                ),
                params,
            )
            _ensure_source_cursor(db, params)
            consumption = db.execute(
                text(
                    """
                    select id from semantic_growth_evidence_consumptions
                    where source_scenic_id=:source_scenic_id and source_id=:source_id
                      and chunk_id=:chunk_id and chunk_hash=:chunk_hash
                      and consumer_version=:consumer_version and target_scope=:target_scope
                    """
                ),
                params,
            ).scalar_one()
            item = dict(row)
            item["source_family_id"] = source_family_id
            item["consumption_id"] = int(consumption)
            item["lease_owner"] = str(worker_id)
            claimed.append(item)
        return claimed


def _name_occurs_in_content(content: str, name: str) -> bool:
    # Numeric node names must not match a substring of a larger number,
    # e.g. node 306 must not match the postal code 030600.
    if not name:
        return False
    if name.isdigit():
        pattern = rf"(?<![0-9A-Za-z]){re.escape(name)}(?![0-9A-Za-z])"
        return re.search(pattern, content) is not None
    return name in content


def extract_mentions_from_batch(
    batch: list[dict[str, Any]],
    published_nodes: list[dict[str, Any]],
    *,
    max_mentions_per_chunk: int = 30,
) -> list[dict[str, Any]]:
    by_name: dict[str, list[tuple[str, str]]] = {}
    for node in published_nodes:
        node_id = str(node.get("node_id") or node.get("source_node_id") or node.get("id") or "").strip()
        name = str(node.get("name") or node.get("node_name") or "").strip()
        if node_id and len(name) >= 2:
            by_name.setdefault(name, []).append((node_id, str(node.get("node_type") or "")))

    # Exact text is safe only when a name maps to one published node. Duplicate
    # names remain unresolved for the graph-context/vector alignment stage.
    candidates = [
        (name, values[0][0], values[0][1])
        for name, values in by_name.items()
        if len(values) == 1
    ]
    candidates.sort(key=lambda item: len(item[0]), reverse=True)

    by_node_id = {
        str(node.get("node_id") or node.get("source_node_id") or node.get("id") or "").strip(): (
            str(node.get("name") or node.get("node_name") or "").strip(),
            str(node.get("node_type") or ""),
        )
        for node in published_nodes
        if str(node.get("node_id") or node.get("source_node_id") or node.get("id") or "").strip()
    }

    result: list[dict[str, Any]] = []
    for chunk in batch:
        chunk_id = chunk.get("id") or chunk.get("chunk_id")
        consumption_id = chunk.get("consumption_id")
        if chunk_id is None or consumption_id is None:
            continue
        content = str(chunk.get("content") or "")
        if str(chunk.get("asset_type") or "") == "image":
            # Only OCR/caption-bearing assets enter extraction. The asset's
            # existing node binding is safe alignment; claims remain pending.
            node_id = str(chunk.get("source_node_id") or "").strip()
            node_name, node_type = by_node_id.get(
                node_id,
                (
                    str(chunk.get("source_node_name") or "").strip(),
                    str(chunk.get("source_node_type") or "").strip(),
                ),
            )
            if node_id and content.strip():
                result.append({
                    "consumption_id": int(consumption_id),
                    "chunk_id": int(chunk_id),
                    "source_scenic_id": str(chunk["source_scenic_id"]),
                    "node_id": node_id,
                    # The binding, not OCR, determines the subject.  A
                    # missing display name is still safe to process; the
                    # source node id remains authoritative.
                    "node_name": node_name or node_id,
                    "node_type": node_type,
                    "mention_text": node_name,
                    "match_method": "ASSET_NODE_BINDING",
                    "match_score": 0.95,
                    "asset_id": chunk.get("asset_id"),
                })
            continue
        seen: set[str] = set()
        chunk_count = 0
        for name, node_id, node_type in candidates:
            if _name_occurs_in_content(content, name) and node_id not in seen:
                result.append({
                    "consumption_id": int(consumption_id),
                    "chunk_id": int(chunk_id),
                    "source_scenic_id": str(chunk["source_scenic_id"]),
                    "node_id": node_id,
                    "node_name": name,
                    "node_type": node_type,
                    "mention_text": name,
                    "match_method": "EXACT_NAME",
                    "match_score": 1.0,
                })
                seen.add(node_id)
                chunk_count += 1
                if chunk_count >= max_mentions_per_chunk:
                    break
    return result


def persist_alignment_results(
    *,
    growth_run_id: str,
    batch: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    worker_id: str,
    consumer_version: str = CONSUMER_VERSION,
) -> list[dict[str, Any]]:
    by_consumption: dict[int, list[dict[str, Any]]] = {}
    for mention in mentions:
        by_consumption.setdefault(int(mention["consumption_id"]), []).append(mention)
    alignments: list[dict[str, Any]] = []
    with ai_session_scope() as db:
        for chunk in batch:
            consumption_id = int(chunk["consumption_id"])
            chunk_mentions = by_consumption.get(consumption_id, [])
            cursor = db.execute(
                text(
                    """
                    insert into semantic_growth_source_cursors (
                        source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version
                    ) values (:source_scenic_id, :source_id, :chunk_id, :chunk_hash, :consumer_version)
                    on conflict (source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version)
                    do update set updated_at=now()
                    returning id
                    """
                ),
                {
                    "source_scenic_id": str(chunk["source_scenic_id"]),
                    "source_id": str(chunk["source_id"]),
                    "chunk_id": int(chunk["id"]),
                    "chunk_hash": str(chunk["content_hash"] or ""),
                    "consumer_version": consumer_version,
                },
            ).scalar_one()
            for mention in chunk_mentions:
                db.execute(
                    text(
                        """
                        insert into semantic_growth_evidence_consumptions (
                            growth_run_id, source_scenic_id, source_id, chunk_id, chunk_hash,
                            consumer_version, target_scope, state, lease_owner,
                            attempt_count, result, processed_at, updated_at
                        ) values (
                            :growth_run_id, :source_scenic_id, :source_id, :chunk_id, :chunk_hash,
                            :consumer_version, :target_scope, 'CLAIMED', :worker_id,
                            0, null, null, now()
                        )
                        on conflict (source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version, target_scope)
                        do update set state='CLAIMED', result=null, error=null,
                                      lease_owner=:worker_id, processed_at=null, updated_at=now()
                        returning id
                        """
                    ),
                    {
                        "growth_run_id": str(growth_run_id),
                        "source_scenic_id": str(chunk["source_scenic_id"]),
                        "source_id": str(chunk["source_id"]),
                        "chunk_id": int(chunk["id"]),
                        "chunk_hash": str(chunk["content_hash"] or ""),
                        "consumer_version": consumer_version,
                        "target_scope": str(mention["node_id"]),
                        "worker_id": str(worker_id),
                    },
                )
                db.execute(
                    text(
                        """
                        insert into semantic_growth_evidence_mentions (
                            consumption_id, source_scenic_id, chunk_id, node_id,
                            node_name, node_type, mention_text, match_method, match_score
                        ) values (
                            :consumption_id, :source_scenic_id, :chunk_id, :node_id,
                            :node_name, :node_type, :mention_text, :match_method, :match_score
                        )
                        on conflict (consumption_id, node_id, mention_text) do nothing
                        """
                    ),
                    mention,
                )
                alignments.append(dict(mention))
            scope_counts = db.execute(
                text(
                    """
                    select count(*) as expected_scope_count,
                           count(*) filter (where state='PROCESSED') as processed_scope_count
                    from semantic_growth_evidence_consumptions
                    where source_scenic_id=:source_scenic_id
                      and source_id=:source_id
                      and chunk_id=:chunk_id
                      and chunk_hash=:chunk_hash
                      and consumer_version=:consumer_version
                    """
                ),
                {
                    "source_scenic_id": str(chunk["source_scenic_id"]),
                    "source_id": str(chunk["source_id"]),
                    "chunk_id": int(chunk["id"]),
                    "chunk_hash": str(chunk["content_hash"] or ""),
                    "consumer_version": consumer_version,
                },
            ).mappings().one()
            expected_scope_count = int(scope_counts["expected_scope_count"] or 0)
            processed_scope_count = int(scope_counts["processed_scope_count"] or 0)
            db.execute(
                text(
                    """
                    update semantic_growth_source_cursors
                    set expected_scope_count=:expected, processed_scope_count=:processed,
                        cursor_state='OPEN', advanced_at=null,
                        updated_at=now()
                    where id=:id
                    """
                ),
                {
                    "id": int(cursor),
                    "expected": expected_scope_count,
                    "processed": processed_scope_count,
                },
            )
    return alignments


def finalize_evidence_batch(
    *,
    batch: list[dict[str, Any]],
    results: list[dict[str, Any]],
    worker_id: str,
    consumer_version: str = CONSUMER_VERSION,
) -> list[dict[str, Any]]:
    """Finalize target scopes first, then advance a chunk cursor only on success.

    A missing, FAILED, or RETRYABLE target scope keeps the source cursor OPEN.
    This deliberately provides at-least-once processing rather than silently
    skipping a fan-out branch that did not finish.
    """
    result_by_scope = {str(item.get("node_id") or ""): item for item in results}
    summaries: list[dict[str, Any]] = []
    with ai_session_scope() as db:
        for chunk in batch:
            identity = {
                "source_scenic_id": str(chunk["source_scenic_id"]),
                "source_id": str(chunk["source_id"]),
                "chunk_id": int(chunk["id"]),
                "chunk_hash": str(chunk["content_hash"] or ""),
                "consumer_version": consumer_version,
            }
            _ensure_source_cursor(db, identity)
            target_rows = db.execute(
                text(
                    """
                    select id, target_scope
                    from semantic_growth_evidence_consumptions
                    where source_scenic_id=:source_scenic_id
                      and source_id=:source_id and chunk_id=:chunk_id
                      and chunk_hash=:chunk_hash and consumer_version=:consumer_version
                      and target_scope<>:chunk_scope
                    """
                ),
                {**identity, "chunk_scope": CHUNK_SCOPE},
            ).mappings().all()
            for target in target_rows:
                scope = str(target["target_scope"])
                result = result_by_scope.get(scope)
                succeeded = result is not None and not result.get("error")
                candidate_count = len(result.get("candidate_ids") or []) if result else 0
                db.execute(
                    text(
                        """
                        update semantic_growth_evidence_consumptions
                        set state=:state, result=:result, error=:error,
                            processed_at=case when :state='PROCESSED' then now() else null end,
                            lease_expires_at=null, updated_at=now()
                        where id=:id
                        """
                    ),
                    {
                        "id": int(target["id"]),
                        "state": "PROCESSED" if succeeded else "RETRYABLE",
                        "result": ("CANDIDATE" if candidate_count else "NO_CHANGE") if succeeded else "FAILED",
                        "error": None if succeeded else str((result or {}).get("error") or "target scope was not processed")[:2000],
                    },
                )
            target_counts = db.execute(
                text(
                    """
                    select count(*) as expected_target_count,
                           count(*) filter (where state='PROCESSED') as processed_target_count,
                           count(*) filter (where result='CANDIDATE') as candidate_target_count
                    from semantic_growth_evidence_consumptions
                    where source_scenic_id=:source_scenic_id
                      and source_id=:source_id and chunk_id=:chunk_id
                      and chunk_hash=:chunk_hash and consumer_version=:consumer_version
                      and target_scope<>:chunk_scope
                    """
                ),
                {**identity, "chunk_scope": CHUNK_SCOPE},
            ).mappings().one()
            expected_targets = int(target_counts["expected_target_count"] or 0)
            processed_targets = int(target_counts["processed_target_count"] or 0)
            all_targets_processed = expected_targets == processed_targets
            chunk_result = "CANDIDATE" if int(target_counts["candidate_target_count"] or 0) else "NO_CHANGE"
            db.execute(
                text(
                    """
                    update semantic_growth_evidence_consumptions
                    set state=:state, result=:result, error=:error,
                        processed_at=case when :state='PROCESSED' then now() else null end,
                        lease_expires_at=null, updated_at=now()
                    where id=:id and target_scope=:chunk_scope
                    """
                ),
                {
                    "id": int(chunk["consumption_id"]),
                    "chunk_scope": CHUNK_SCOPE,
                    "state": "PROCESSED" if all_targets_processed else "RETRYABLE",
                    "result": chunk_result if all_targets_processed else "FAILED",
                    "error": None if all_targets_processed else "one or more target scopes did not finish",
                },
            )
            scope_counts = db.execute(
                text(
                    """
                    select array_agg(state order by target_scope) as scope_states
                    from semantic_growth_evidence_consumptions
                    where source_scenic_id=:source_scenic_id
                      and source_id=:source_id and chunk_id=:chunk_id
                      and chunk_hash=:chunk_hash and consumer_version=:consumer_version
                    """
                ),
                identity,
            ).mappings().one()
            progress = source_cursor_progress(list(scope_counts["scope_states"] or []))
            expected_scopes = int(progress["expected_scope_count"])
            processed_scopes = int(progress["processed_scope_count"])
            cursor_advanced = progress["cursor_state"] == "ADVANCED"
            db.execute(
                text(
                    """
                    update semantic_growth_source_cursors
                    set expected_scope_count=:expected, processed_scope_count=:processed,
                        cursor_state=:cursor_state,
                        advanced_at=case when :cursor_state='ADVANCED' then now() else null end,
                        updated_at=now()
                    where source_scenic_id=:source_scenic_id
                      and source_id=:source_id and chunk_id=:chunk_id
                      and chunk_hash=:chunk_hash and consumer_version=:consumer_version
                    """
                ),
                {
                    **identity,
                    "expected": expected_scopes,
                    "processed": processed_scopes,
                    "cursor_state": "ADVANCED" if cursor_advanced else "OPEN",
                },
            )
            summaries.append({
                "consumption_id": int(chunk["consumption_id"]),
                "expected_scope_count": expected_scopes,
                "processed_scope_count": processed_scopes,
                "cursor_state": "ADVANCED" if cursor_advanced else "OPEN",
            })
    return summaries


def finalize_open_discovery_batch(
    *,
    batch: list[dict[str, Any]],
    results: list[dict[str, Any]],
    worker_id: str,
    consumer_version: str = CONSUMER_VERSION,
) -> list[dict[str, Any]]:
    """Finalize evidence-first discovery without inventing node target scopes.

    Open discovery owns one consumption scope per EvidenceUnit. A failed unit
    remains RETRYABLE and therefore keeps SourceCursor OPEN; successful units
    are PROCESSED even when their result is EXISTS/NO_CHANGE.
    """
    result_by_consumption = {
        int(item["consumption_id"]): item
        for item in results
        if item.get("consumption_id") is not None
    }
    summaries: list[dict[str, Any]] = []
    with ai_session_scope() as db:
        for chunk in batch:
            consumption_id = int(chunk["consumption_id"])
            result = result_by_consumption.get(consumption_id)
            succeeded = result is not None and not result.get("error")
            result_code = str((result or {}).get("result") or "NO_CHANGE")
            identity = {
                "source_scenic_id": str(chunk["source_scenic_id"]),
                "source_id": str(chunk["source_id"]),
                "chunk_id": int(chunk["id"]),
                "chunk_hash": str(chunk["content_hash"] or ""),
                "consumer_version": consumer_version,
            }
            _ensure_source_cursor(db, identity)
            db.execute(
                text(
                    """
                    update semantic_growth_evidence_consumptions
                    set state=:state, result=:result, error=:error,
                        processed_at=case when :state='PROCESSED' then now() else null end,
                        lease_expires_at=null, updated_at=now()
                    where id=:id and lease_owner=:worker_id
                      and state in ('CLAIMED', 'PROCESSED', 'RETRYABLE')
                    """
                ),
                {
                    "id": consumption_id,
                    "worker_id": str(worker_id),
                    "state": "PROCESSED" if succeeded else "RETRYABLE",
                    "result": result_code if succeeded else "FAILED",
                    "error": None if succeeded else str((result or {}).get("error") or "evidence unit was not processed")[:2000],
                },
            )
            scope_states = list(
                db.execute(
                    text(
                        """
                        select state from semantic_growth_evidence_consumptions
                        where source_scenic_id=:source_scenic_id
                          and source_id=:source_id and chunk_id=:chunk_id
                          and chunk_hash=:chunk_hash and consumer_version=:consumer_version
                        order by target_scope
                        """
                    ),
                    identity,
                ).scalars().all()
            )
            progress = source_cursor_progress(scope_states)
            db.execute(
                text(
                    """
                    update semantic_growth_source_cursors
                    set expected_scope_count=:expected, processed_scope_count=:processed,
                        cursor_state=:cursor_state,
                        advanced_at=case when :cursor_state='ADVANCED' then now() else null end,
                        updated_at=now()
                    where source_scenic_id=:source_scenic_id
                      and source_id=:source_id and chunk_id=:chunk_id
                      and chunk_hash=:chunk_hash and consumer_version=:consumer_version
                    """
                ),
                {
                    **identity,
                    "expected": int(progress["expected_scope_count"]),
                    "processed": int(progress["processed_scope_count"]),
                    "cursor_state": str(progress["cursor_state"]),
                },
            )
            summaries.append(
                {
                    "consumption_id": consumption_id,
                    "result": result_code if succeeded else "FAILED",
                    "cursor_state": str(progress["cursor_state"]),
                }
            )
    return summaries


def mark_evidence_failed(*, consumption_id: int, worker_id: str, error: str) -> None:
    with ai_session_scope() as db:
        row = db.execute(
            text(
                """
                update semantic_growth_evidence_consumptions
                set state='FAILED', result='FAILED', error=:error,
                    lease_expires_at=null, updated_at=now()
                -- Alignment may have completed before G2 extraction fails. In
                -- that case the chunk is already PROCESSED but must still be
                -- eligible for a later claim/retry.
                where id=:id and lease_owner=:worker_id and state in ('CLAIMED', 'PROCESSED')
                returning source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version
                """
            ),
            {"id": int(consumption_id), "worker_id": str(worker_id), "error": str(error)[:2000]},
        ).mappings().first()
        if row:
            _reopen_source_cursor(db, dict(row))


def mark_evidence_retryable(*, consumption_id: int, worker_id: str, error: str) -> None:
    with ai_session_scope() as db:
        row = db.execute(
            text(
                """
                update semantic_growth_evidence_consumptions
                set state='RETRYABLE', result='FAILED', error=:error,
                    lease_expires_at=null, updated_at=now()
                where id=:id and lease_owner=:worker_id and state in ('CLAIMED', 'PROCESSED')
                returning source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version
                """
            ),
            {"id": int(consumption_id), "worker_id": str(worker_id), "error": str(error)[:2000]},
        ).mappings().first()
        if row:
            _reopen_source_cursor(db, dict(row))


def _reopen_source_cursor(db: Any, identity: dict[str, Any]) -> None:
    """Synchronize cursor progress after any target becomes non-successful."""
    db.execute(
        text(
            """
            update semantic_growth_source_cursors c
            set expected_scope_count=counts.expected_scope_count,
                processed_scope_count=counts.processed_scope_count,
                cursor_state='OPEN', advanced_at=null, updated_at=now()
            from (
                select count(*) as expected_scope_count,
                       count(*) filter (where state='PROCESSED') as processed_scope_count
                from semantic_growth_evidence_consumptions
                where source_scenic_id=:source_scenic_id
                  and source_id=:source_id and chunk_id=:chunk_id
                  and chunk_hash=:chunk_hash and consumer_version=:consumer_version
            ) counts
            where c.source_scenic_id=:source_scenic_id
              and c.source_id=:source_id and c.chunk_id=:chunk_id
              and c.chunk_hash=:chunk_hash and c.consumer_version=:consumer_version
            """
        ),
        identity,
    )
