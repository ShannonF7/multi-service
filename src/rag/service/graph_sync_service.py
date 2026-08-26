"""Durable jobs that project A-side published graph snapshots into Neo4j."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase
from sqlalchemy import text

from src.rag.dependencies import get_ai_engine

logger = logging.getLogger(__name__)

_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()
_NEO4J_DRIVER = None
_NEO4J_LOCK = threading.RLock()
_NEO4J_SCHEMA_READY = False


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def ensure_graph_sync_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        with get_ai_engine().begin() as connection:
            connection.execute(text("select pg_advisory_xact_lock(hashtext('rag_graph_sync_schema_v1'))"))
            connection.execute(
                text(
                    """
                    create table if not exists semantic_graph_sync_jobs (
                        id bigserial primary key,
                        event_key varchar(160) not null unique,
                        domain_id varchar(64) not null,
                        payload jsonb not null default '{}'::jsonb,
                        status varchar(20) not null default 'PENDING',
                        attempt_count integer not null default 0,
                        max_attempts integer not null default 5,
                        next_retry_at timestamptz null,
                        locked_by varchar(128) null,
                        locked_at timestamptz null,
                        lease_expires_at timestamptz null,
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
                    create index if not exists semantic_graph_sync_jobs_pick_idx
                    on semantic_graph_sync_jobs(status, next_retry_at, created_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    create index if not exists semantic_graph_sync_jobs_domain_idx
                    on semantic_graph_sync_jobs(domain_id, created_at desc)
                    """
                )
            )
        _SCHEMA_READY = True


def create_or_get_graph_sync_job(payload: dict[str, Any]) -> dict[str, Any]:
    ensure_graph_sync_schema()
    event_key = str(payload.get("event_key") or "").strip()
    domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
    domain_id = str(payload.get("domain_id") or domain.get("id") or "").strip()
    if not event_key:
        raise ValueError("event_key is required")
    if not domain_id:
        raise ValueError("domain_id is required")
    if not isinstance(payload.get("nodes"), list):
        raise ValueError("nodes must be a list")
    max_attempts = max(1, min(int(payload.get("max_attempts") or 5), 20))
    with get_ai_engine().begin() as connection:
        existed = connection.execute(
            text("select id, status from semantic_graph_sync_jobs where event_key = :event_key"),
            {"event_key": event_key},
        ).mappings().first()
        row = connection.execute(
            text(
                """
                insert into semantic_graph_sync_jobs (
                    event_key, domain_id, payload, status, max_attempts
                ) values (
                    :event_key, :domain_id, cast(:payload as jsonb), 'PENDING', :max_attempts
                )
                on conflict (event_key) do update set
                    domain_id = excluded.domain_id,
                    payload = excluded.payload,
                    status = case
                        when semantic_graph_sync_jobs.status = 'SUCCEEDED' then 'SUCCEEDED'
                        else 'PENDING'
                    end,
                    attempt_count = case
                        when semantic_graph_sync_jobs.status = 'FAILED' then 0
                        else semantic_graph_sync_jobs.attempt_count
                    end,
                    max_attempts = excluded.max_attempts,
                    next_retry_at = case
                        when semantic_graph_sync_jobs.status = 'SUCCEEDED' then semantic_graph_sync_jobs.next_retry_at
                        else null
                    end,
                    error_message = case
                        when semantic_graph_sync_jobs.status = 'SUCCEEDED' then semantic_graph_sync_jobs.error_message
                        else null
                    end,
                    updated_at = now()
                returning *
                """
            ),
            {
                "event_key": event_key[:160],
                "domain_id": domain_id[:64],
                "payload": _json(payload),
                "max_attempts": max_attempts,
            },
        ).mappings().first()
    result = dict(row or {})
    result["reused"] = bool(existed)
    return result


def get_graph_sync_job(job_id: int) -> dict[str, Any] | None:
    ensure_graph_sync_schema()
    with get_ai_engine().connect() as connection:
        row = connection.execute(
            text("select * from semantic_graph_sync_jobs where id = :id"),
            {"id": int(job_id)},
        ).mappings().first()
    return dict(row) if row else None


def claim_next_graph_sync_job(*, worker_id: str, lease_seconds: int = 300) -> dict[str, Any] | None:
    ensure_graph_sync_schema()
    with get_ai_engine().begin() as connection:
        row = connection.execute(
            text(
                """
                select *
                from semantic_graph_sync_jobs
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
                update semantic_graph_sync_jobs
                set status = 'RUNNING',
                    attempt_count = attempt_count + 1,
                    locked_by = :worker_id,
                    locked_at = now(),
                    lease_expires_at = now() + (:lease_seconds || ' seconds')::interval,
                    started_at = coalesce(started_at, now()),
                    error_message = null,
                    updated_at = now()
                where id = :id
                returning *
                """
            ),
            {"id": int(row["id"]), "worker_id": worker_id[:128], "lease_seconds": int(lease_seconds)},
        ).mappings().first()
    return dict(claimed) if claimed else None


def recover_stale_graph_sync_jobs() -> int:
    ensure_graph_sync_schema()
    with get_ai_engine().begin() as connection:
        retried = connection.execute(
            text(
                """
                update semantic_graph_sync_jobs
                set status = 'PENDING', locked_by = null, locked_at = null,
                    lease_expires_at = null, next_retry_at = now(),
                    error_message = 'Worker lease expired', updated_at = now()
                where status = 'RUNNING'
                  and lease_expires_at is not null
                  and lease_expires_at < now()
                  and attempt_count < max_attempts
                """
            )
        ).rowcount or 0
        failed = connection.execute(
            text(
                """
                update semantic_graph_sync_jobs
                set status = 'FAILED', locked_by = null, locked_at = null,
                    lease_expires_at = null, error_message = 'Worker lease expired and max attempts reached',
                    finished_at = now(), updated_at = now()
                where status = 'RUNNING'
                  and lease_expires_at is not null
                  and lease_expires_at < now()
                  and attempt_count >= max_attempts
                """
            )
        ).rowcount or 0
    return int(retried) + int(failed)


def _neo4j_driver():
    global _NEO4J_DRIVER
    if _NEO4J_DRIVER is not None:
        return _NEO4J_DRIVER
    with _NEO4J_LOCK:
        if _NEO4J_DRIVER is None:
            uri = os.getenv("TRAVEL_NEO4J_URI", "bolt://127.0.0.1:17687")
            user = os.getenv("TRAVEL_NEO4J_USER", "neo4j")
            password = os.getenv("TRAVEL_NEO4J_PASSWORD")
            if not password:
                raise RuntimeError("TRAVEL_NEO4J_PASSWORD is not configured")
            _NEO4J_DRIVER = GraphDatabase.driver(uri, auth=(user, password))
            _NEO4J_DRIVER.verify_connectivity()
    return _NEO4J_DRIVER


def _ensure_neo4j_schema() -> None:
    global _NEO4J_SCHEMA_READY
    if _NEO4J_SCHEMA_READY:
        return
    with _NEO4J_LOCK:
        if _NEO4J_SCHEMA_READY:
            return
        database = os.getenv("TRAVEL_NEO4J_DATABASE", "neo4j")
        statements = [
            "CREATE CONSTRAINT published_entity_key IF NOT EXISTS "
            "FOR (n:PublishedEntity) REQUIRE n.projection_key IS UNIQUE",
            "CREATE CONSTRAINT published_fact_key IF NOT EXISTS "
            "FOR (n:PublishedFact) REQUIRE n.projection_key IS UNIQUE",
            "CREATE CONSTRAINT knowledge_domain_key IF NOT EXISTS "
            "FOR (n:KnowledgeDomain) REQUIRE n.domain_id IS UNIQUE",
        ]
        with _neo4j_driver().session(database=database) as session:
            for statement in statements:
                session.run(statement).consume()
        _NEO4J_SCHEMA_READY = True

def _evidence_ids(provenance: Any) -> list[str]:
    values: list[str] = []
    for item in provenance if isinstance(provenance, list) else []:
        if not isinstance(item, dict):
            continue
        for evidence_id in item.get("evidence_ids") or []:
            value = str(evidence_id)
            if value and value not in values:
                values.append(value)
    return values


def _projection_rows(payload: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    domain_id = str(payload["domain_id"])
    nodes: list[dict] = []
    for item in payload.get("nodes") or []:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        row = dict(item)
        row["node_id"] = str(item["id"])
        row["projection_key"] = f"{domain_id}:{item['id']}"
        row["parent_projection_key"] = f"{domain_id}:{item['parent_id']}" if item.get("parent_id") else ""
        row["tags"] = [str(value) for value in (item.get("tags") or []) if value is not None]
        nodes.append(row)

    facts: list[dict] = []
    for item in payload.get("properties") or []:
        if not isinstance(item, dict) or item.get("id") is None or item.get("node_id") is None:
            continue
        provenance = item.get("provenance") if isinstance(item.get("provenance"), list) else []
        facts.append(
            {
                "projection_key": f"{domain_id}:property:{item['id']}",
                "entity_key": f"{domain_id}:{item['node_id']}",
                "domain_id": domain_id,
                "formal_property_id": str(item["id"]),
                "key": str(item.get("key") or ""),
                "value": str(item.get("value") or ""),
                "value_type": str(item.get("value_type") or "string"),
                "version": int(item.get("version") or 1),
                "evidence_ids": _evidence_ids(provenance),
                "provenance_json": _json(provenance),
                "updated_at": str(item.get("updated_at") or ""),
            }
        )

    relations: list[dict] = []
    for item in payload.get("relations") or []:
        if not isinstance(item, dict) or item.get("id") is None:
            continue
        if item.get("source_node_id") is None or item.get("target_node_id") is None:
            continue
        provenance = item.get("provenance") if isinstance(item.get("provenance"), list) else []
        relations.append(
            {
                "projection_key": f"{domain_id}:relation:{item['id']}",
                "source_key": f"{domain_id}:{item['source_node_id']}",
                "target_key": f"{domain_id}:{item['target_node_id']}",
                "domain_id": domain_id,
                "formal_relation_id": str(item["id"]),
                "relation_code": str(item.get("relation_code") or ""),
                "relation_label": str(item.get("relation_label") or item.get("relation_code") or ""),
                "relation_category": str(item.get("relation_category") or "semantic"),
                "description": str(item.get("description") or ""),
                "confidence": float(item.get("confidence") or 0.0),
                "is_verified": bool(item.get("is_verified")),
                "extraction_method": str(item.get("extraction_method") or ""),
                "version": int(item.get("version") or 1),
                "evidence_ids": _evidence_ids(provenance),
                "provenance_json": _json(provenance),
                "updated_at": str(item.get("updated_at") or ""),
            }
        )
    return nodes, facts, relations


def project_graph_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    domain = payload.get("domain") if isinstance(payload.get("domain"), dict) else {}
    domain_id = str(payload.get("domain_id") or domain.get("id") or "").strip()
    if not domain_id:
        raise ValueError("domain_id is required")
    payload = dict(payload)
    payload["domain_id"] = domain_id
    nodes, facts, relations = _projection_rows(payload)
    snapshot_keys = [f"{domain_id}:{value}" for value in payload.get("snapshot_node_ids") or []]
    deleted_keys = [f"{domain_id}:{value}" for value in payload.get("deleted_node_ids") or []]
    replace_domain = bool(payload.get("replace_domain"))

    def write_projection(tx):
        tx.run(
            "MERGE (d:KnowledgeDomain {domain_id: $domain_id}) "
            "SET d.code = $code, d.name = $name, d.description = $description, d.updated_at = datetime()",
            domain_id=domain_id,
            code=str(domain.get("code") or ""),
            name=str(domain.get("name") or ""),
            description=str(domain.get("description") or ""),
        ).consume()
        if replace_domain:
            tx.run(
                "MATCH (n:PublishedEntity {domain_id: $domain_id})-[:HAS_FACT]->(f:PublishedFact) DETACH DELETE f",
                domain_id=domain_id,
            ).consume()
            tx.run(
                "MATCH (n:PublishedEntity {domain_id: $domain_id}) DETACH DELETE n",
                domain_id=domain_id,
            ).consume()
        else:
            if deleted_keys:
                tx.run(
                    "MATCH (n:PublishedEntity) "
                    "WHERE n.projection_key IN $keys DETACH DELETE n",
                    keys=deleted_keys,
                ).consume()
            if snapshot_keys:
                tx.run(
                    "MATCH (n:PublishedEntity)-[:HAS_FACT]->(f:PublishedFact) "
                    "WHERE n.projection_key IN $keys DETACH DELETE f",
                    keys=snapshot_keys,
                ).consume()
                tx.run(
                    "MATCH (n:PublishedEntity)-[r:PUBLISHED_RELATION]->() "
                    "WHERE n.projection_key IN $keys DELETE r",
                    keys=snapshot_keys,
                ).consume()
                tx.run(
                    "MATCH (:PublishedEntity)-[r:PARENT_OF]->(n:PublishedEntity) "
                    "WHERE n.projection_key IN $keys DELETE r",
                    keys=snapshot_keys,
                ).consume()
        if nodes:
            tx.run(
                "UNWIND $rows AS row "
                "MATCH (d:KnowledgeDomain {domain_id: $domain_id}) "
                "MERGE (n:PublishedEntity {projection_key: row.projection_key}) "
                "SET n.domain_id = $domain_id, n.node_id = row.node_id, n.name = row.name, "
                "n.node_type = row.node_type, n.description = row.description, n.tags = row.tags, "
                "n.parent_node_id = coalesce(toString(row.parent_id), ''), n.is_active = true, "
                "n.lng = row.lng, n.lat = row.lat, n.source_updated_at = row.updated_at, n.projected_at = datetime() "
                "MERGE (d)-[:CONTAINS]->(n)",
                rows=nodes,
                domain_id=domain_id,
            ).consume()
            tx.run(
                "UNWIND $rows AS row "
                "WITH row WHERE row.parent_projection_key <> '' "
                "MATCH (parent:PublishedEntity {projection_key: row.parent_projection_key}) "
                "MATCH (child:PublishedEntity {projection_key: row.projection_key}) "
                "MERGE (parent)-[:PARENT_OF]->(child)",
                rows=nodes,
            ).consume()
        if facts:
            tx.run(
                "UNWIND $rows AS row "
                "MATCH (n:PublishedEntity {projection_key: row.entity_key}) "
                "MERGE (f:PublishedFact {projection_key: row.projection_key}) "
                "SET f.domain_id = row.domain_id, f.formal_property_id = row.formal_property_id, "
                "f.key = row.key, f.value = row.value, f.value_type = row.value_type, "
                "f.version = row.version, f.evidence_ids = row.evidence_ids, "
                "f.provenance_json = row.provenance_json, f.source_updated_at = row.updated_at, "
                "f.projected_at = datetime() "
                "MERGE (n)-[:HAS_FACT]->(f)",
                rows=facts,
            ).consume()
        if relations:
            tx.run(
                "UNWIND $rows AS row "
                "MATCH (source:PublishedEntity {projection_key: row.source_key}) "
                "MATCH (target:PublishedEntity {projection_key: row.target_key}) "
                "MERGE (source)-[r:PUBLISHED_RELATION {projection_key: row.projection_key}]->(target) "
                "SET r.domain_id = row.domain_id, r.formal_relation_id = row.formal_relation_id, "
                "r.relation_code = row.relation_code, r.relation_label = row.relation_label, "
                "r.relation_category = row.relation_category, r.description = row.description, "
                "r.confidence = row.confidence, r.is_verified = row.is_verified, "
                "r.extraction_method = row.extraction_method, r.version = row.version, "
                "r.evidence_ids = row.evidence_ids, r.provenance_json = row.provenance_json, "
                "r.source_updated_at = row.updated_at, r.projected_at = datetime()",
                rows=relations,
            ).consume()

    database = os.getenv("TRAVEL_NEO4J_DATABASE", "neo4j")
    _ensure_neo4j_schema()
    with _neo4j_driver().session(database=database) as session:
        session.execute_write(write_projection)
    return {
        "domain_id": domain_id,
        "node_count": len(nodes),
        "fact_count": len(facts),
        "relation_count": len(relations),
        "deleted_node_count": len(deleted_keys),
        "replace_domain": replace_domain,
        "projected_at": datetime.now(timezone.utc).isoformat(),
    }


def process_graph_sync_job(job_id: int, *, worker_id: str) -> dict[str, Any]:
    job = get_graph_sync_job(job_id)
    if not job:
        raise LookupError("graph sync job not found")
    try:
        result = project_graph_snapshot(job.get("payload") or {})
    except Exception as exc:
        logger.exception("graph projection failed job_id=%s", job_id)
        with get_ai_engine().begin() as connection:
            row = connection.execute(
                text("select attempt_count, max_attempts from semantic_graph_sync_jobs where id = :id"),
                {"id": int(job_id)},
            ).mappings().first()
            exhausted = bool(row and int(row["attempt_count"]) >= int(row["max_attempts"]))
            connection.execute(
                text(
                    """
                    update semantic_graph_sync_jobs
                    set status = :status, error_message = :error_message,
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
                    "error_message": str(exc)[:4000],
                },
            )
        raise
    with get_ai_engine().begin() as connection:
        connection.execute(
            text(
                """
                update semantic_graph_sync_jobs
                set status = 'SUCCEEDED', result = cast(:result as jsonb), error_message = null,
                    locked_by = null, locked_at = null, lease_expires_at = null,
                    finished_at = now(), updated_at = now()
                where id = :id and locked_by = :worker_id
                """
            ),
            {"id": int(job_id), "worker_id": worker_id, "result": _json(result)},
        )
    return result


def neo4j_health() -> dict[str, Any]:
    database = os.getenv("TRAVEL_NEO4J_DATABASE", "neo4j")
    _ensure_neo4j_schema()
    with _neo4j_driver().session(database=database) as session:
        row = session.run("RETURN 1 AS ok").single()
    return {"status": "ok" if row and row["ok"] == 1 else "error", "database": database}


def neo4j_domain_stats(domain_id: str) -> dict[str, Any]:
    database = os.getenv("TRAVEL_NEO4J_DATABASE", "neo4j")
    query = (
        "MATCH (n:PublishedEntity {domain_id: $domain_id}) "
        "OPTIONAL MATCH (n)-[:HAS_FACT]->(f:PublishedFact) "
        "WITH count(DISTINCT n) AS nodes, count(DISTINCT f) AS facts "
        "MATCH (d:KnowledgeDomain {domain_id: $domain_id}) "
        "OPTIONAL MATCH (:PublishedEntity {domain_id: $domain_id})"
        "-[r:PUBLISHED_RELATION|PARENT_OF]->(:PublishedEntity {domain_id: $domain_id}) "
        "RETURN nodes, facts, count(DISTINCT r) AS relations, "
        "count(DISTINCT CASE WHEN type(r) = 'PUBLISHED_RELATION' THEN r END) AS published_relations, "
        "count(DISTINCT CASE WHEN type(r) = 'PUBLISHED_RELATION' "
        "AND coalesce(r.relation_category, 'semantic') = 'semantic' THEN r END) AS semantic_relations, "
        "count(DISTINCT CASE WHEN type(r) = 'PUBLISHED_RELATION' "
        "AND r.relation_category = 'spatial' THEN r END) AS spatial_relations, "
        "count(DISTINCT CASE WHEN type(r) = 'PARENT_OF' THEN r END) AS hierarchy_relations, "
        "d.name AS domain_name, toString(d.updated_at) AS last_sync_time"
    )
    with _neo4j_driver().session(database=database) as session:
        row = session.run(query, domain_id=str(domain_id)).single()
    if not row:
        return {
            "domain_id": str(domain_id),
            "nodes": 0,
            "facts": 0,
            "relations": 0,
            "published_relations": 0,
            "semantic_relations": 0,
            "spatial_relations": 0,
            "hierarchy_relations": 0,
            "last_sync_time": None,
        }
    return {"domain_id": str(domain_id), **dict(row)}
