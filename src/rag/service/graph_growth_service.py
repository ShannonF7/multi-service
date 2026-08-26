"""Durable graph discovery and evidence-validation queue.

Neo4j/GDS output is stored as a hypothesis. It never writes formal graph data
or semantic candidates directly; validation is delegated to the existing
evidence-first semantic completion jobs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import get_ai_engine
from src.rag.schemas import SemanticCompleteRequest
from src.rag.service.completion_job_service import create_or_get_semantic_completion_job
from src.rag.service.graph_discovery_service import (
    get_published_neighborhood,
    resolve_published_domain_id,
)
from src.rag.service.graph_sync_service import _neo4j_driver

logger = logging.getLogger(__name__)

_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()
ALGORITHM = "gds_node_similarity_v1"


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


def _database() -> str:
    return os.getenv("TRAVEL_NEO4J_DATABASE", "neo4j")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return _clean(value).lower() in {"1", "true", "yes", "on"}


def ensure_graph_growth_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        with get_ai_engine().begin() as connection:
            connection.execute(text("select pg_advisory_xact_lock(hashtext('rag_graph_growth_schema_v1'))"))
            connection.execute(
                text(
                    """
                    create table if not exists semantic_graph_discovery_jobs (
                        id bigserial primary key,
                        event_key varchar(160) not null unique,
                        domain_identifier varchar(128) not null,
                        domain_id varchar(64) null,
                        source_graph_sync_job_id bigint null,
                        algorithm varchar(64) not null default 'gds_node_similarity_v1',
                        payload jsonb not null default '{}'::jsonb,
                        status varchar(32) not null default 'PENDING',
                        attempt_count integer not null default 0,
                        max_attempts integer not null default 3,
                        next_retry_at timestamptz null,
                        locked_by varchar(128) null,
                        locked_at timestamptz null,
                        lease_expires_at timestamptz null,
                        discovery_count integer not null default 0,
                        validation_job_count integer not null default 0,
                        error_message text null,
                        result jsonb not null default '{}'::jsonb,
                        created_at timestamptz not null default now(),
                        updated_at timestamptz not null default now(),
                        started_at timestamptz null,
                        finished_at timestamptz null
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    create table if not exists semantic_graph_discoveries (
                        id bigserial primary key,
                        discovery_key varchar(160) not null unique,
                        first_job_id bigint not null,
                        last_job_id bigint not null,
                        domain_id varchar(64) not null,
                        domain_code varchar(128) null,
                        discovery_type varchar(48) not null default 'potential_relation',
                        algorithm varchar(64) not null,
                        source_node_id varchar(128) not null,
                        source_name text null,
                        source_type varchar(128) null,
                        target_node_id varchar(128) not null,
                        target_name text null,
                        target_type varchar(128) null,
                        relation_hint varchar(128) null,
                        score double precision not null default 0,
                        common_neighbor_count integer not null default 0,
                        support jsonb not null default '{}'::jsonb,
                        validation_question text not null,
                        status varchar(32) not null default 'PENDING_EVIDENCE',
                        evidence_job_id bigint null,
                        last_error text null,
                        created_at timestamptz not null default now(),
                        updated_at timestamptz not null default now(),
                        last_seen_at timestamptz not null default now()
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    create index if not exists semantic_graph_discovery_jobs_pick_idx
                    on semantic_graph_discovery_jobs(status, next_retry_at, created_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    create index if not exists semantic_graph_discoveries_node_idx
                    on semantic_graph_discoveries(domain_id, source_node_id, status, updated_at desc)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    create index if not exists semantic_graph_discoveries_validation_idx
                    on semantic_graph_discoveries(evidence_job_id) where evidence_job_id is not null
                    """
                )
            )
        _SCHEMA_READY = True


def create_or_get_graph_discovery_job(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_graph_growth_schema()
    domain_identifier = _clean(payload.get("domain_identifier") or payload.get("domain_id"))
    if not domain_identifier:
        raise ValueError("domain_identifier or domain_id is required")
    event_key = _clean(payload.get("event_key")) or f"manual:{uuid.uuid4().hex}"
    max_attempts = max(1, min(int(payload.get("max_attempts") or 3), 10))
    normalized = dict(payload)
    normalized["domain_identifier"] = domain_identifier
    normalized["algorithm"] = _clean(payload.get("algorithm") or ALGORITHM)
    normalized["node_ids"] = [
        _clean(value)
        for value in (payload.get("node_ids") or payload.get("snapshot_node_ids") or [])
        if _clean(value)
    ]
    normalized["max_results"] = max(1, min(int(payload.get("max_results") or 20), 100))
    normalized["top_k"] = max(1, min(int(payload.get("top_k") or 10), 50))
    normalized["similarity_cutoff"] = max(0.0, min(float(payload.get("similarity_cutoff") or 0.75), 1.0))
    normalized["enqueue_validation"] = _bool(payload.get("enqueue_validation"), True)

    with get_ai_engine().begin() as connection:
        existed = connection.execute(
            text("select id, status from semantic_graph_discovery_jobs where event_key = :event_key"),
            {"event_key": event_key[:160]},
        ).mappings().first()
        row = connection.execute(
            text(
                """
                insert into semantic_graph_discovery_jobs (
                    event_key, domain_identifier, source_graph_sync_job_id,
                    algorithm, payload, status, max_attempts
                ) values (
                    :event_key, :domain_identifier, :source_graph_sync_job_id,
                    :algorithm, cast(:payload as jsonb), 'PENDING', :max_attempts
                )
                on conflict (event_key) do update set
                    payload = excluded.payload,
                    max_attempts = excluded.max_attempts,
                    updated_at = now()
                returning *
                """
            ),
            {
                "event_key": event_key[:160],
                "domain_identifier": domain_identifier[:128],
                "source_graph_sync_job_id": payload.get("source_graph_sync_job_id"),
                "algorithm": normalized["algorithm"][:64],
                "payload": _json(normalized),
                "max_attempts": max_attempts,
            },
        ).mappings().first()
    result = dict(row or {})
    result["reused"] = bool(existed)
    return result


def get_graph_discovery_job(job_id: int) -> dict[str, Any] | None:
    ensure_graph_growth_schema()
    refresh_graph_discovery_validation_statuses()
    with get_ai_engine().connect() as connection:
        row = connection.execute(
            text("select * from semantic_graph_discovery_jobs where id = :id"),
            {"id": int(job_id)},
        ).mappings().first()
    return dict(row) if row else None


def list_graph_discovery_jobs(
    *,
    domain_identifier: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    ensure_graph_growth_schema()
    filters: list[str] = []
    params: dict[str, Any] = {"limit": max(1, min(int(limit), 200)), "offset": max(0, int(offset))}
    if domain_identifier:
        filters.append("domain_identifier = :domain_identifier")
        params["domain_identifier"] = _clean(domain_identifier)
    if status:
        filters.append("status = :status")
        params["status"] = _clean(status).upper()
    where = " where " + " and ".join(filters) if filters else ""
    with get_ai_engine().connect() as connection:
        total = connection.execute(
            text("select count(*) from semantic_graph_discovery_jobs" + where), params
        ).scalar_one()
        rows = connection.execute(
            text(
                "select * from semantic_graph_discovery_jobs"
                + where
                + " order by created_at desc limit :limit offset :offset"
            ),
            params,
        ).mappings().all()
    return {"items": [dict(row) for row in rows], "total": int(total)}


def claim_next_graph_discovery_job(*, worker_id: str, lease_seconds: int = 600) -> dict[str, Any] | None:
    ensure_graph_growth_schema()
    with get_ai_engine().begin() as connection:
        row = connection.execute(
            text(
                """
                select id
                from semantic_graph_discovery_jobs
                where status = 'PENDING'
                  and attempt_count < max_attempts
                  and (next_retry_at is null or next_retry_at <= now())
                order by created_at asc
                for update skip locked
                limit 1
                """
            )
        ).mappings().first()
        if not row:
            return None
        claimed = connection.execute(
            text(
                """
                update semantic_graph_discovery_jobs
                set status = 'RUNNING', attempt_count = attempt_count + 1,
                    locked_by = :worker_id, locked_at = now(),
                    lease_expires_at = now() + (:lease_seconds || ' seconds')::interval,
                    started_at = coalesce(started_at, now()),
                    error_message = null, updated_at = now()
                where id = :id
                returning *
                """
            ),
            {"id": int(row["id"]), "worker_id": worker_id[:128], "lease_seconds": int(lease_seconds)},
        ).mappings().first()
    return dict(claimed) if claimed else None


def recover_stale_graph_discovery_jobs() -> int:
    ensure_graph_growth_schema()
    with get_ai_engine().begin() as connection:
        retried = connection.execute(
            text(
                """
                update semantic_graph_discovery_jobs
                set status = 'PENDING', locked_by = null, locked_at = null,
                    lease_expires_at = null, next_retry_at = now(),
                    error_message = 'Worker lease expired', updated_at = now()
                where status = 'RUNNING' and lease_expires_at < now()
                  and attempt_count < max_attempts
                """
            )
        ).rowcount or 0
        failed = connection.execute(
            text(
                """
                update semantic_graph_discovery_jobs
                set status = 'FAILED', locked_by = null, locked_at = null,
                    lease_expires_at = null,
                    error_message = 'Worker lease expired and max attempts reached',
                    finished_at = now(), updated_at = now()
                where status = 'RUNNING' and lease_expires_at < now()
                  and attempt_count >= max_attempts
                """
            )
        ).rowcount or 0
    return int(retried) + int(failed)


def _domain_identity(domain_id: str) -> dict[str, str]:
    with _neo4j_driver().session(database=_database()) as session:
        record = session.run(
            "MATCH (domain:KnowledgeDomain {domain_id: $domain_id}) "
            "RETURN domain.domain_id AS domain_id, domain.code AS code, domain.name AS name",
            domain_id=domain_id,
        ).single()
    return {
        "domain_id": domain_id,
        "code": _clean(record.get("code")) if record else "",
        "name": _clean(record.get("name")) if record else "",
    }


def run_gds_node_similarity(
    domain_identifier: str,
    *,
    node_ids: list[str] | None = None,
    similarity_cutoff: float = 0.75,
    top_k: int = 10,
    limit: int = 20,
) -> dict[str, Any]:
    domain_id = resolve_published_domain_id(domain_identifier)
    graph_name = f"travel-growth-{uuid.uuid4().hex[:12]}"
    selected = {_clean(value) for value in (node_ids or []) if _clean(value)}
    rows: list[dict[str, Any]] = []
    projection: dict[str, Any] = {"nodeCount": 0, "relationshipCount": 0}

    with _neo4j_driver().session(database=_database()) as session:
        try:
            record = session.run(
                "MATCH (source:PublishedEntity {domain_id: $domain_id}) "
                "OPTIONAL MATCH (source)-[:PUBLISHED_RELATION]-(target:PublishedEntity {domain_id: $domain_id}) "
                "WITH gds.graph.project($graph_name, source, target) AS graph "
                "RETURN graph.nodeCount AS nodeCount, graph.relationshipCount AS relationshipCount",
                graph_name=graph_name,
                domain_id=domain_id,
            ).single()
            if record:
                projection = dict(record)
            if int(projection.get("nodeCount") or 0) < 2 or int(projection.get("relationshipCount") or 0) < 1:
                return {"domain_id": domain_id, "projection": projection, "items": []}
            result = session.run(
                """
                CALL gds.nodeSimilarity.stream($graph_name, {
                    topK: $top_k,
                    similarityCutoff: $similarity_cutoff,
                    degreeCutoff: 1
                })
                YIELD node1, node2, similarity
                WITH gds.util.asNode(node1) AS source,
                     gds.util.asNode(node2) AS target,
                     similarity
                WHERE source.node_id < target.node_id
                  AND source.node_type = target.node_type
                  AND NOT (source)-[:PUBLISHED_RELATION]-(target)
                  AND (size($node_ids) = 0 OR source.node_id IN $node_ids OR target.node_id IN $node_ids)
                OPTIONAL MATCH (source)-[:PUBLISHED_RELATION]-(common:PublishedEntity {domain_id: $domain_id})
                              -[:PUBLISHED_RELATION]-(target)
                WITH source, target, similarity,
                     count(DISTINCT common) AS common_neighbor_count,
                     collect(DISTINCT common.name)[..8] AS common_neighbor_names
                RETURN source.node_id AS source_node_id,
                       source.name AS source_name,
                       source.node_type AS source_type,
                       source.description AS source_description,
                       target.node_id AS target_node_id,
                       target.name AS target_name,
                       target.node_type AS target_type,
                       target.description AS target_description,
                       similarity,
                       common_neighbor_count,
                       common_neighbor_names
                ORDER BY similarity DESC, common_neighbor_count DESC,
                         source.name ASC, target.name ASC
                LIMIT $limit
                """,
                graph_name=graph_name,
                top_k=max(1, min(int(top_k), 50)),
                similarity_cutoff=max(0.0, min(float(similarity_cutoff), 1.0)),
                node_ids=sorted(selected),
                domain_id=domain_id,
                limit=max(1, min(int(limit), 100)),
            )
            rows = [dict(item) for item in result]
        finally:
            exists = session.run("RETURN gds.graph.exists($graph_name) AS present", graph_name=graph_name).single()
            if exists and exists["present"]:
                session.run(
                    "CALL gds.graph.drop($graph_name) YIELD graphName RETURN graphName",
                    graph_name=graph_name,
                ).consume()

    if selected:
        for item in rows:
            if item.get("target_node_id") in selected and item.get("source_node_id") not in selected:
                for left, right in [
                    ("source_node_id", "target_node_id"),
                    ("source_name", "target_name"),
                    ("source_type", "target_type"),
                    ("source_description", "target_description"),
                ]:
                    item[left], item[right] = item.get(right), item.get(left)
    return {"domain_id": domain_id, "projection": projection, "items": rows}


def _discovery_key(domain_id: str, source_node_id: str, target_node_id: str) -> str:
    pair = sorted([_clean(source_node_id), _clean(target_node_id)])
    raw = f"{domain_id}|{ALGORITHM}|{pair[0]}|{pair[1]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]


def _validation_question(item: dict[str, Any]) -> str:
    return (
        f"请从可信公开资料中验证“{_clean(item.get('source_name'))}”与"
        f"“{_clean(item.get('target_name'))}”是否存在直接、明确的关系。"
        "如果存在，请给出具体关系类型、方向和原文证据；"
        "不得把图结构相似、共同邻居或模型推测本身当作证据。"
    )


def _upsert_discovery(
    *,
    job_id: int,
    domain: dict[str, str],
    item: dict[str, Any],
) -> dict[str, Any]:
    support = {
        "common_neighbor_count": int(item.get("common_neighbor_count") or 0),
        "common_neighbor_names": list(item.get("common_neighbor_names") or []),
        "graph_signal_only": True,
        "not_evidence": True,
    }
    key = _discovery_key(domain["domain_id"], item["source_node_id"], item["target_node_id"])
    with get_ai_engine().begin() as connection:
        row = connection.execute(
            text(
                """
                insert into semantic_graph_discoveries (
                    discovery_key, first_job_id, last_job_id,
                    domain_id, domain_code, discovery_type, algorithm,
                    source_node_id, source_name, source_type,
                    target_node_id, target_name, target_type,
                    relation_hint, score, common_neighbor_count, support,
                    validation_question, status
                ) values (
                    :discovery_key, :job_id, :job_id,
                    :domain_id, :domain_code, 'potential_relation', :algorithm,
                    :source_node_id, :source_name, :source_type,
                    :target_node_id, :target_name, :target_type,
                    '待证据判定', :score, :common_neighbor_count, cast(:support as jsonb),
                    :validation_question, 'PENDING_EVIDENCE'
                )
                on conflict (discovery_key) do update set
                    last_job_id = excluded.last_job_id,
                    score = greatest(semantic_graph_discoveries.score, excluded.score),
                    common_neighbor_count = greatest(
                        semantic_graph_discoveries.common_neighbor_count,
                        excluded.common_neighbor_count
                    ),
                    support = excluded.support,
                    validation_question = excluded.validation_question,
                    last_seen_at = now(), updated_at = now()
                returning *
                """
            ),
            {
                "discovery_key": key,
                "job_id": int(job_id),
                "domain_id": domain["domain_id"],
                "domain_code": domain.get("code") or None,
                "algorithm": ALGORITHM,
                "source_node_id": _clean(item.get("source_node_id")),
                "source_name": _clean(item.get("source_name")),
                "source_type": _clean(item.get("source_type")),
                "target_node_id": _clean(item.get("target_node_id")),
                "target_name": _clean(item.get("target_name")),
                "target_type": _clean(item.get("target_type")),
                "score": float(item.get("similarity") or 0.0),
                "common_neighbor_count": int(item.get("common_neighbor_count") or 0),
                "support": _json(support),
                "validation_question": _validation_question(item),
            },
        ).mappings().first()
    return dict(row or {})


def build_graph_validation_payload(discovery: dict[str, Any]) -> SemanticCompleteRequest:
    domain_id = _clean(discovery.get("domain_id"))
    source_node_id = _clean(discovery.get("source_node_id"))
    neighborhood = get_published_neighborhood(domain_id, source_node_id, depth=1, limit=30)
    if not neighborhood.get("found"):
        raise LookupError("source node is not present in published graph")
    domain = _domain_identity(domain_id)
    root = neighborhood.get("root") or {}
    existing_properties = [
        {"key": item.get("key"), "value": item.get("value"), "value_type": item.get("value_type")}
        for item in (neighborhood.get("facts") or [])
        if item.get("key")
    ]
    existing_relations = [
        {
            "relation_type": item.get("relation_label") or item.get("relation_code"),
            "target_name": item.get("target_name"),
            "confidence": item.get("confidence"),
        }
        for item in (neighborhood.get("relations") or [])
        if _clean(item.get("source_node_id")) == source_node_id
    ]
    graph_nodes = [root] + list(neighborhood.get("nodes") or [])
    graph_relations = [
        {
            "source_node_id": item.get("source_node_id"),
            "source_name": item.get("source_name"),
            "relation_type": item.get("relation_label") or item.get("relation_code"),
            "relation_category": item.get("graph_relation_type"),
            "target_node_id": item.get("target_node_id"),
            "target_name": item.get("target_name"),
        }
        for item in (neighborhood.get("relations") or [])
    ]
    scenic_identifier = domain.get("code") or domain_id
    return SemanticCompleteRequest.parse_obj(
        {
            "scenic_id": scenic_identifier,
            "node": {
                "source_node_id": source_node_id,
                "name": root.get("name") or discovery.get("source_name"),
                "node_type": root.get("node_type") or discovery.get("source_type"),
                "description": root.get("description") or "",
                "scenic_name": domain.get("name") or scenic_identifier,
                "metadata": {
                    "graph_discovery_id": discovery.get("id"),
                    "graph_discovery_key": discovery.get("discovery_key"),
                },
            },
            "message": discovery.get("validation_question") or "",
            "target_fields": [],
            "relation_intents": [],
            "subgraph_depth": 1,
            "existing_properties": existing_properties,
            "existing_relations": existing_relations,
            "graph_context": {
                "scope": "subgraph",
                "depth": 1,
                "nodes": graph_nodes,
                "relations": graph_relations,
                "search_terms": [
                    discovery.get("source_name"),
                    discovery.get("target_name"),
                ],
            },
            "max_web_results": 5,
            "use_web_search": True,
            "use_web_extractor": True,
            "metadata": {
                "completion_mode": "deep",
                "open_discovery": True,
                "graph_discovery": False,
                "graph_growth_validation": True,
                "graph_discovery_id": discovery.get("id"),
                "graph_discovery_key": discovery.get("discovery_key"),
                "retrieval_scope": "subgraph",
                "source_scope": ["domain_kb", "web_search", "web_extractor"],
                "web_search_policy": "local_first_backfill",
                "question_batch_size": 1,
                "web_limit_per_question": 5,
                "candidate_value_policy": "group_recommend_alternatives",
            },
        }
    )


def _enqueue_validation(discovery: dict[str, Any]) -> int | None:
    if discovery.get("evidence_job_id"):
        return int(discovery["evidence_job_id"])
    payload = build_graph_validation_payload(discovery)
    job = create_or_get_semantic_completion_job(payload, created_by="graph-growth-worker")
    evidence_job_id = int(job["id"])
    with get_ai_engine().begin() as connection:
        connection.execute(
            text(
                """
                update semantic_graph_discoveries
                set evidence_job_id = :evidence_job_id,
                    status = 'VALIDATION_QUEUED', last_error = null, updated_at = now()
                where id = :id and evidence_job_id is null
                """
            ),
            {"id": int(discovery["id"]), "evidence_job_id": evidence_job_id},
        )
    return evidence_job_id


def refresh_graph_discovery_validation_statuses() -> int:
    ensure_graph_growth_schema()
    with get_ai_engine().begin() as connection:
        updated = connection.execute(
            text(
                """
                update semantic_graph_discoveries discovery
                set status = case
                        when completion.status = 'DONE' and completion.candidate_count > 0
                            then 'CANDIDATES_READY'
                        when completion.status = 'DONE'
                            then 'NO_SUPPORTED_CANDIDATE'
                        when completion.status = 'FAILED'
                            then 'VALIDATION_FAILED'
                        else discovery.status
                    end,
                    last_error = case
                        when completion.status = 'FAILED' then completion.error_message
                        else discovery.last_error
                    end,
                    updated_at = now()
                from semantic_completion_jobs completion
                where discovery.evidence_job_id = completion.id
                  and discovery.status in ('VALIDATION_QUEUED', 'VALIDATION_RUNNING')
                  and completion.status in ('DONE', 'FAILED')
                """
            )
        ).rowcount or 0
    return int(updated)


def list_graph_discoveries(
    *,
    domain_id: str | None = None,
    source_node_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    ensure_graph_growth_schema()
    refresh_graph_discovery_validation_statuses()
    filters: list[str] = []
    params: dict[str, Any] = {"limit": max(1, min(int(limit), 500)), "offset": max(0, int(offset))}
    if domain_id:
        resolved = resolve_published_domain_id(domain_id)
        filters.append("discovery.domain_id = :domain_id")
        params["domain_id"] = resolved
    if source_node_id:
        filters.append("discovery.source_node_id = :source_node_id")
        params["source_node_id"] = _clean(source_node_id)
    if status:
        filters.append("discovery.status = :status")
        params["status"] = _clean(status).upper()
    where = " where " + " and ".join(filters) if filters else ""
    base = """
        from semantic_graph_discoveries discovery
        left join semantic_completion_jobs completion on completion.id = discovery.evidence_job_id
    """
    with get_ai_engine().connect() as connection:
        total = connection.execute(text("select count(*) " + base + where), params).scalar_one()
        rows = connection.execute(
            text(
                """
                select discovery.*,
                       completion.status as validation_job_status,
                       completion.evidence_count as validation_evidence_count,
                       completion.candidate_count as validation_candidate_count,
                       completion.conflict_count as validation_conflict_count
                """
                + base
                + where
                + " order by discovery.score desc, discovery.updated_at desc limit :limit offset :offset"
            ),
            params,
        ).mappings().all()
    return {"items": [dict(row) for row in rows], "total": int(total)}


def process_graph_discovery_job(job_id: int, *, worker_id: str) -> dict[str, Any]:
    ensure_graph_growth_schema()
    with get_ai_engine().connect() as connection:
        job = connection.execute(
            text("select * from semantic_graph_discovery_jobs where id = :id"),
            {"id": int(job_id)},
        ).mappings().first()
    if not job:
        raise LookupError("graph discovery job not found")
    payload = dict(job.get("payload") or {})
    try:
        gds_result = run_gds_node_similarity(
            payload.get("domain_identifier") or job.get("domain_identifier"),
            node_ids=list(payload.get("node_ids") or []),
            similarity_cutoff=float(payload.get("similarity_cutoff") or 0.75),
            top_k=int(payload.get("top_k") or 10),
            limit=int(payload.get("max_results") or 20),
        )
        domain = _domain_identity(gds_result["domain_id"])
        discoveries: list[dict[str, Any]] = []
        validation_jobs: list[int] = []
        enqueue_validation = _bool(payload.get("enqueue_validation"), True)
        for item in gds_result.get("items") or []:
            discovery = _upsert_discovery(job_id=int(job_id), domain=domain, item=item)
            discoveries.append(discovery)
            if enqueue_validation and not discovery.get("evidence_job_id"):
                try:
                    evidence_job_id = _enqueue_validation(discovery)
                    if evidence_job_id:
                        validation_jobs.append(evidence_job_id)
                except Exception as exc:
                    logger.exception("queue graph validation failed discovery_id=%s", discovery.get("id"))
                    with get_ai_engine().begin() as connection:
                        connection.execute(
                            text(
                                """
                                update semantic_graph_discoveries
                                set status = 'VALIDATION_QUEUE_FAILED', last_error = :error, updated_at = now()
                                where id = :id
                                """
                            ),
                            {"id": int(discovery["id"]), "error": str(exc)[:4000]},
                        )
        result = {
            "domain_id": gds_result["domain_id"],
            "algorithm": ALGORITHM,
            "projection": gds_result.get("projection") or {},
            "discovery_count": len(discoveries),
            "validation_job_count": len(set(validation_jobs)),
            "discovery_ids": [int(item["id"]) for item in discoveries],
            "validation_job_ids": sorted(set(validation_jobs)),
            "policy": "hypothesis_requires_external_evidence",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.exception("graph discovery job failed job_id=%s", job_id)
        with get_ai_engine().begin() as connection:
            row = connection.execute(
                text("select attempt_count, max_attempts from semantic_graph_discovery_jobs where id = :id"),
                {"id": int(job_id)},
            ).mappings().first()
            exhausted = bool(row and int(row["attempt_count"]) >= int(row["max_attempts"]))
            connection.execute(
                text(
                    """
                    update semantic_graph_discovery_jobs
                    set status = :status, error_message = :error,
                        next_retry_at = case when :status = 'PENDING'
                            then now() + (least(300, power(2, greatest(attempt_count, 1))) || ' seconds')::interval
                            else null end,
                        locked_by = null, locked_at = null, lease_expires_at = null,
                        finished_at = case when :status = 'FAILED' then now() else finished_at end,
                        updated_at = now()
                    where id = :id
                    """
                ),
                {
                    "id": int(job_id),
                    "status": "FAILED" if exhausted else "PENDING",
                    "error": str(exc)[:4000],
                },
            )
        raise
    with get_ai_engine().begin() as connection:
        connection.execute(
            text(
                """
                update semantic_graph_discovery_jobs
                set status = 'SUCCEEDED', domain_id = :domain_id,
                    discovery_count = :discovery_count,
                    validation_job_count = :validation_job_count,
                    result = cast(:result as jsonb), error_message = null,
                    locked_by = null, locked_at = null, lease_expires_at = null,
                    finished_at = now(), updated_at = now()
                where id = :id and locked_by = :worker_id
                """
            ),
            {
                "id": int(job_id),
                "worker_id": worker_id,
                "domain_id": result["domain_id"],
                "discovery_count": result["discovery_count"],
                "validation_job_count": result["validation_job_count"],
                "result": _json(result),
            },
        )
    return result


def enqueue_graph_discovery_for_sync(graph_job: dict[str, Any]) -> dict[str, Any] | None:
    if not _bool(os.getenv("GRAPH_DISCOVERY_AUTO_ENQUEUE"), True):
        return None
    payload = dict(graph_job.get("payload") or {})
    domain_identifier = _clean(payload.get("domain_id") or (payload.get("domain") or {}).get("id"))
    node_ids = [
        _clean(value)
        for value in (payload.get("snapshot_node_ids") or [])
        if _clean(value)
    ]
    if not node_ids:
        node_ids = [
            _clean(item.get("id"))
            for item in (payload.get("nodes") or [])
            if isinstance(item, dict) and _clean(item.get("id"))
        ]
    if not domain_identifier or not node_ids:
        return None
    source_event = _clean(graph_job.get("event_key") or graph_job.get("id"))
    digest = hashlib.sha256(source_event.encode("utf-8")).hexdigest()[:32]
    return create_or_get_graph_discovery_job(
        {
            "event_key": f"graph-sync-growth:{digest}",
            "domain_identifier": domain_identifier,
            "source_graph_sync_job_id": graph_job.get("id"),
            "node_ids": node_ids,
            "algorithm": ALGORITHM,
            "similarity_cutoff": float(os.getenv("GRAPH_DISCOVERY_SIMILARITY_CUTOFF", "0.75")),
            "top_k": int(os.getenv("GRAPH_DISCOVERY_TOP_K", "10")),
            "max_results": int(os.getenv("GRAPH_DISCOVERY_MAX_RESULTS_PER_SYNC", "5")),
            "enqueue_validation": True,
        }
    )

