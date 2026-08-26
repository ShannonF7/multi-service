"""Read-only discovery over the published Neo4j projection.

Graph results are hypotheses and search hints, never evidence. They must pass
through the normal retrieval, evidence verification, and candidate pipeline.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

from src.rag.schemas import SemanticCompleteRequest
from src.rag.service.graph_sync_service import _neo4j_driver
from src.rag.service.planner_service import CompletionQuestion


GRAPH_DISCOVERY_MODES = {"deep", "web", "full", "batch"}


def _database() -> str:
    return os.getenv("TRAVEL_NEO4J_DATABASE", "neo4j")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _dedupe(values: list[Any], *, limit: int = 12) -> list[str]:
    result: list[str] = []
    for value in values:
        text = _clean(value)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _question_id(kind: str, *values: Any) -> str:
    raw = "|".join(_clean(value).lower() for value in values)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"graph:{kind}:{digest}"


def resolve_published_domain_id(domain_identifier: str) -> str:
    domain_identifier = _clean(domain_identifier)
    if not domain_identifier:
        raise ValueError("domain identifier is required")
    with _neo4j_driver().session(database=_database()) as session:
        record = session.run(
            "MATCH (domain:KnowledgeDomain) "
            "WHERE domain.domain_id = $identifier OR domain.code = $identifier "
            "RETURN domain.domain_id AS domain_id "
            "ORDER BY CASE WHEN domain.domain_id = $identifier THEN 0 ELSE 1 END "
            "LIMIT 1",
            identifier=domain_identifier,
        ).single()
    return _clean(record.get("domain_id")) if record else domain_identifier


def graph_discovery_enabled(payload: SemanticCompleteRequest) -> bool:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    configured = metadata.get("graph_discovery")
    if configured is not None and not bool(configured):
        return False
    mode = _clean(metadata.get("completion_mode") or metadata.get("job_mode") or "quick").lower()
    return mode in GRAPH_DISCOVERY_MODES and int(payload.subgraph_depth or 0) != 0


def search_published_entities(
    domain_id: str,
    query: str,
    *,
    node_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    domain_id = _clean(domain_id)
    query = _clean(query)
    node_type = _clean(node_type)
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    if not domain_id:
        raise ValueError("domain_id is required")
    if len(query) > 100:
        raise ValueError("query exceeds 100 characters")
    domain_id = resolve_published_domain_id(domain_id)
    normalized_query = query.casefold()
    match_clause = """
        MATCH (node:PublishedEntity {domain_id: $domain_id})
        WHERE ($node_type = '' OR node.node_type = $node_type)
          AND (
              toLower(coalesce(node.name, '')) CONTAINS $search_query
              OR toLower(coalesce(node.description, '')) CONTAINS $search_query
              OR any(tag IN coalesce(node.tags, []) WHERE toLower(toString(tag)) CONTAINS $search_query)
              OR EXISTS {
                  MATCH (node)-[:HAS_FACT]->(fact:PublishedFact)
                  WHERE toLower(coalesce(toString(fact.value), '')) CONTAINS $search_query
                     OR toLower(coalesce(fact.key, '')) CONTAINS $search_query
              }
          )
    """
    count_query = match_clause + " RETURN count(node) AS total"
    page_query = match_clause + """
        OPTIONAL MATCH (node)-[edge:PUBLISHED_RELATION|PARENT_OF]-(
            neighbor:PublishedEntity {domain_id: $domain_id}
        )
        WITH node, count(DISTINCT edge) AS degree,
             CASE
                 WHEN $search_query = '' AND coalesce(toString(node.parent_node_id), '') = '' THEN 20
                 WHEN $search_query = '' THEN 0
                 WHEN toLower(coalesce(node.name, '')) = $search_query THEN 100
                 WHEN toLower(coalesce(node.name, '')) STARTS WITH $search_query THEN 80
                 WHEN toLower(coalesce(node.name, '')) CONTAINS $search_query THEN 60
                 ELSE 30
             END AS match_score
        RETURN node {
            .node_id, .name, .node_type, .description, .parent_node_id,
            .tags, .lng, .lat, degree: degree, match_score: match_score
        } AS item
        ORDER BY match_score DESC, degree DESC, node.name ASC, node.node_id ASC
        SKIP $offset
        LIMIT $limit
    """
    params = {
        "domain_id": domain_id,
        "search_query": normalized_query,
        "node_type": node_type,
        "offset": offset,
        "limit": limit,
    }
    with _neo4j_driver().session(database=_database()) as session:
        total_record = session.run(count_query, **params).single()
        items = [
            dict(record.get("item") or {})
            for record in session.run(page_query, **params)
        ]
    total = int((total_record or {}).get("total") or 0)
    return {
        "domain_id": domain_id,
        "query": query,
        "node_type": node_type or None,
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(items) < total,
    }


def get_published_domain_overview(
    domain_id: str,
    *,
    node_limit: int = 500,
    relation_limit: int = 3000,
) -> dict[str, Any]:
    domain_id = _clean(domain_id)
    node_limit = max(1, min(int(node_limit), 2000))
    relation_limit = max(1, min(int(relation_limit), 10000))
    if not domain_id:
        raise ValueError("domain_id is required")
    domain_id = resolve_published_domain_id(domain_id)

    node_query = """
        MATCH (node:PublishedEntity {domain_id: $domain_id})
        OPTIONAL MATCH (node)-[:HAS_FACT]->(fact:PublishedFact)
        WITH node, collect(DISTINCT fact {
            .formal_property_id, .key, .value, .value_type,
            .evidence_ids, .provenance_json
        }) AS all_facts
        WITH node, [item IN all_facts WHERE item.key IS NOT NULL] AS facts
        RETURN node {
            .node_id, .name, .node_type, .description, .parent_node_id,
            .tags, .lng, .lat, facts: facts, fact_count: size(facts)
        } AS item
        ORDER BY CASE WHEN coalesce(toString(node.parent_node_id), '') = '' THEN 0 ELSE 1 END,
                 node.name ASC, node.node_id ASC
        LIMIT $fetch_limit
    """
    relation_query = """
        MATCH (source:PublishedEntity)-[relation:PUBLISHED_RELATION|PARENT_OF]->(target:PublishedEntity)
        WHERE source.projection_key IN $projection_keys
          AND target.projection_key IN $projection_keys
        RETURN source.node_id AS source_node_id, source.name AS source_name,
               target.node_id AS target_node_id, target.name AS target_name,
               type(relation) AS graph_relation_type,
               coalesce(relation.relation_code, 'PARENT_OF') AS relation_code,
               coalesce(relation.relation_label, '层级归属') AS relation_label,
               relation.formal_relation_id AS formal_relation_id,
               coalesce(relation.relation_category,
                        CASE WHEN type(relation) = 'PARENT_OF' THEN 'hierarchy' ELSE 'semantic' END) AS relation_category,
               relation.description AS description,
               relation.confidence AS confidence,
               relation.is_verified AS is_verified,
               relation.extraction_method AS extraction_method,
               relation.updated_at AS updated_at,
               relation.evidence_ids AS evidence_ids,
               relation.provenance_json AS provenance_json
        ORDER BY graph_relation_type, relation_label, source_name, target_name
        LIMIT $fetch_limit
    """
    with _neo4j_driver().session(database=_database()) as session:
        node_total_record = session.run(
            "MATCH (node:PublishedEntity {domain_id: $domain_id}) RETURN count(node) AS total",
            domain_id=domain_id,
        ).single()
        relation_total_record = session.run(
            "MATCH (:PublishedEntity {domain_id: $domain_id})"
            "-[relation:PUBLISHED_RELATION|PARENT_OF]->"
            "(:PublishedEntity {domain_id: $domain_id}) RETURN count(relation) AS total",
            domain_id=domain_id,
        ).single()
        fact_total_record = session.run(
            "MATCH (:PublishedEntity {domain_id: $domain_id})-[:HAS_FACT]->(fact:PublishedFact) "
            "RETURN count(fact) AS total",
            domain_id=domain_id,
        ).single()
        node_rows = [
            dict(record.get("item") or {})
            for record in session.run(
                node_query,
                domain_id=domain_id,
                fetch_limit=node_limit + 1,
            )
        ]
        node_truncated = len(node_rows) > node_limit
        nodes = node_rows[:node_limit]
        projection_keys = [
            f"{domain_id}:{item.get('node_id')}"
            for item in nodes
            if item.get("node_id") is not None
        ]
        relation_rows = [
            dict(record)
            for record in session.run(
                relation_query,
                projection_keys=projection_keys,
                fetch_limit=relation_limit + 1,
            )
        ] if projection_keys else []

    relation_truncated = len(relation_rows) > relation_limit
    return {
        "domain_id": domain_id,
        "nodes": nodes,
        "relations": relation_rows[:relation_limit],
        "total_nodes": int((node_total_record or {}).get("total") or 0),
        "total_relations": int((relation_total_record or {}).get("total") or 0),
        "total_facts": int((fact_total_record or {}).get("total") or 0),
        "node_limit": node_limit,
        "relation_limit": relation_limit,
        "node_truncated": node_truncated,
        "relation_truncated": relation_truncated,
        "truncated": node_truncated or relation_truncated,
    }

def get_published_node_detail(
    domain_id: str,
    node_id: str,
    *,
    relation_limit: int = 2000,
) -> dict[str, Any]:
    domain_id = _clean(domain_id)
    node_id = _clean(node_id)
    relation_limit = max(1, min(int(relation_limit), 5000))
    if not domain_id or not node_id:
        raise ValueError("domain_id and node_id are required")
    domain_id = resolve_published_domain_id(domain_id)

    root_query = """
        MATCH (root:PublishedEntity {domain_id: $domain_id, node_id: $node_id})
        OPTIONAL MATCH (root)-[:HAS_FACT]->(fact:PublishedFact)
        RETURN root {
            .node_id, .name, .node_type, .description, .parent_node_id,
            .tags, .lng, .lat
        } AS root,
        collect(DISTINCT fact {
            .formal_property_id, .key, .value, .value_type,
            .evidence_ids, .provenance_json,
            updated_at: coalesce(fact.source_updated_at, toString(fact.projected_at))
        }) AS facts
    """
    relation_count_query = """
        MATCH (root:PublishedEntity {domain_id: $domain_id, node_id: $node_id})
              -[relation:PUBLISHED_RELATION|PARENT_OF]-
              (neighbor:PublishedEntity {domain_id: $domain_id})
        RETURN count(DISTINCT relation) AS total
    """
    relation_query = """
        MATCH (root:PublishedEntity {domain_id: $domain_id, node_id: $node_id})
              -[relation:PUBLISHED_RELATION|PARENT_OF]-
              (neighbor:PublishedEntity {domain_id: $domain_id})
        WITH root, neighbor, relation,
             startNode(relation) AS source, endNode(relation) AS target
        RETURN source.node_id AS source_node_id, source.name AS source_name,
               target.node_id AS target_node_id, target.name AS target_name,
               type(relation) AS graph_relation_type,
               coalesce(relation.relation_code, 'PARENT_OF') AS relation_code,
               coalesce(relation.relation_label, '层级归属') AS relation_label,
               relation.formal_relation_id AS formal_relation_id,
               coalesce(relation.relation_category,
                        CASE WHEN type(relation) = 'PARENT_OF' THEN 'hierarchy' ELSE 'semantic' END) AS relation_category,
               relation.description AS description,
               relation.confidence AS confidence,
               relation.is_verified AS is_verified,
               relation.extraction_method AS extraction_method,
               relation.updated_at AS updated_at,
               relation.evidence_ids AS evidence_ids,
               relation.provenance_json AS provenance_json,
               neighbor {
                   .node_id, .name, .node_type, .description, .parent_node_id,
                   .tags, .lng, .lat
               } AS neighbor
        ORDER BY graph_relation_type, relation_label, source_name, target_name
        LIMIT $fetch_limit
    """
    with _neo4j_driver().session(database=_database()) as session:
        root_record = session.run(
            root_query,
            domain_id=domain_id,
            node_id=node_id,
        ).single()
        if not root_record:
            return {
                "domain_id": domain_id,
                "node_id": node_id,
                "found": False,
                "root": None,
                "facts": [],
                "relations": [],
                "neighbors": [],
                "total_relations": 0,
                "truncated": False,
            }
        count_record = session.run(
            relation_count_query,
            domain_id=domain_id,
            node_id=node_id,
        ).single()
        rows = list(session.run(
            relation_query,
            domain_id=domain_id,
            node_id=node_id,
            fetch_limit=relation_limit + 1,
        ))

    relation_rows = rows[:relation_limit]
    neighbors: dict[str, dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []
    for record in relation_rows:
        item = dict(record)
        neighbor = dict(item.pop("neighbor") or {})
        if neighbor.get("node_id") is not None:
            neighbors[str(neighbor["node_id"])] = neighbor
        relations.append(item)
    total_relations = int((count_record or {}).get("total") or 0)
    return {
        "domain_id": domain_id,
        "node_id": node_id,
        "found": True,
        "root": dict(root_record.get("root") or {}),
        "facts": [dict(item) for item in (root_record.get("facts") or []) if item.get("key")],
        "relations": relations,
        "neighbors": list(neighbors.values()),
        "total_relations": total_relations,
        "relation_limit": relation_limit,
        "truncated": total_relations > len(relations),
    }

def get_published_neighborhood(
    domain_id: str,
    node_id: str,
    *,
    depth: int = 1,
    limit: int = 80,
    offset: int = 0,
    node_types: list[str] | None = None,
    relation_types: list[str] | None = None,
) -> dict[str, Any]:
    domain_id = _clean(domain_id)
    node_id = _clean(node_id)
    depth = max(1, min(int(depth), 3))
    limit = max(1, min(int(limit), 300))
    offset = max(0, int(offset))
    node_types = list(dict.fromkeys(_clean(value) for value in (node_types or []) if _clean(value)))
    relation_types = list(dict.fromkeys(_clean(value) for value in (relation_types or []) if _clean(value)))
    if not domain_id or not node_id:
        raise ValueError("domain_id and node_id are required")
    domain_id = resolve_published_domain_id(domain_id)

    root_query = """
        MATCH (root:PublishedEntity {domain_id: $domain_id, node_id: $node_id})
        OPTIONAL MATCH (root)-[:HAS_FACT]->(fact:PublishedFact)
        RETURN root {
            .node_id, .name, .node_type, .description, .parent_node_id,
            .tags, .lng, .lat
        } AS root,
        collect(DISTINCT fact {
            .formal_property_id, .key, .value, .value_type,
            .evidence_ids, .provenance_json,
            updated_at: coalesce(fact.source_updated_at, toString(fact.projected_at))
        }) AS facts
    """
    neighbor_count_query = f"""
        MATCH (root:PublishedEntity {{domain_id: $domain_id, node_id: $node_id}})
        MATCH path=(root)-[:PUBLISHED_RELATION|PARENT_OF*1..{depth}]-(neighbor:PublishedEntity)
        WHERE neighbor.domain_id = $domain_id
          AND (size($node_types) = 0 OR neighbor.node_type IN $node_types)
          AND (size($relation_types) = 0 OR all(edge IN relationships(path)
              WHERE coalesce(edge.relation_code, type(edge)) IN $relation_types))
        WITH DISTINCT neighbor
        RETURN count(neighbor) AS total
    """
    neighbor_query = f"""
        MATCH (root:PublishedEntity {{domain_id: $domain_id, node_id: $node_id}})
        MATCH path=(root)-[:PUBLISHED_RELATION|PARENT_OF*1..{depth}]-(neighbor:PublishedEntity)
        WHERE neighbor.domain_id = $domain_id
          AND (size($node_types) = 0 OR neighbor.node_type IN $node_types)
          AND (size($relation_types) = 0 OR all(edge IN relationships(path)
              WHERE coalesce(edge.relation_code, type(edge)) IN $relation_types))
        WITH neighbor, min(length(path)) AS graph_depth
        ORDER BY graph_depth ASC, neighbor.name ASC
        SKIP $offset
        LIMIT $limit
        OPTIONAL MATCH (neighbor)-[:HAS_FACT]->(fact:PublishedFact)
        WITH neighbor, graph_depth,
             [item IN collect(DISTINCT fact {{
                 .formal_property_id, .key, .value, .value_type,
                 .evidence_ids, .provenance_json,
                 updated_at: coalesce(fact.source_updated_at, toString(fact.projected_at))
             }}) WHERE item.key IS NOT NULL] AS facts
        RETURN collect(neighbor {{
            .node_id, .name, .node_type, .description, .parent_node_id,
            .tags, graph_depth: graph_depth, facts: facts, fact_count: size(facts)
        }}) AS neighbors
    """

    with _neo4j_driver().session(database=_database()) as session:
        root_record = session.run(root_query, domain_id=domain_id, node_id=node_id).single()
        if not root_record:
            return {
                "domain_id": domain_id,
                "node_id": node_id,
                "found": False,
                "root": None,
                "facts": [],
                "nodes": [],
                "relations": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "has_more": False,
            }
        total_record = session.run(
            neighbor_count_query,
            domain_id=domain_id,
            node_id=node_id,
            node_types=node_types,
            relation_types=relation_types,
        ).single()
        neighbor_record = session.run(
            neighbor_query,
            domain_id=domain_id,
            node_id=node_id,
            node_types=node_types,
            relation_types=relation_types,
            limit=limit,
            offset=offset,
        ).single()
        neighbors = list((neighbor_record or {}).get("neighbors") or [])
        projection_keys = [f"{domain_id}:{node_id}"] + [
            f"{domain_id}:{item.get('node_id')}"
            for item in neighbors
            if item.get("node_id") is not None
        ]
        relation_records = session.run(
            """
            MATCH (source:PublishedEntity)-[relation:PUBLISHED_RELATION|PARENT_OF]->(target:PublishedEntity)
            WHERE source.projection_key IN $projection_keys
              AND target.projection_key IN $projection_keys
              AND (size($relation_types) = 0
                   OR coalesce(relation.relation_code, type(relation)) IN $relation_types)
            RETURN source.node_id AS source_node_id, source.name AS source_name,
                   target.node_id AS target_node_id, target.name AS target_name,
                   type(relation) AS graph_relation_type,
                   coalesce(relation.relation_code, 'PARENT_OF') AS relation_code,
                   coalesce(relation.relation_label, '层级归属') AS relation_label,
                   relation.formal_relation_id AS formal_relation_id,
                   coalesce(relation.relation_category,
                            CASE WHEN type(relation) = 'PARENT_OF' THEN 'hierarchy' ELSE 'semantic' END) AS relation_category,
                   relation.description AS description,
                   relation.confidence AS confidence,
                   relation.is_verified AS is_verified,
                   relation.extraction_method AS extraction_method,
                   relation.updated_at AS updated_at,
                   relation.evidence_ids AS evidence_ids,
                   relation.provenance_json AS provenance_json
            ORDER BY graph_relation_type, relation_label, source_name, target_name
            LIMIT $relation_limit
            """,
            projection_keys=projection_keys,
            relation_types=relation_types,
            relation_limit=min(limit * 4, 1000),
        )
        relations = [dict(record) for record in relation_records]

    total = int((total_record or {}).get("total") or 0)
    return {
        "domain_id": domain_id,
        "node_id": node_id,
        "found": True,
        "root": dict(root_record.get("root") or {}),
        "facts": [dict(item) for item in (root_record.get("facts") or []) if item.get("key")],
        "nodes": [dict(item) for item in neighbors],
        "relations": relations,
        "total": total,
        "limit": limit,
        "offset": offset,
        "node_types": node_types,
        "relation_types": relation_types,
        "supernode": total > limit,
        "has_more": offset + len(neighbors) < total,
    }


def find_published_shortest_path(
    domain_id: str,
    source_node_id: str,
    target_node_id: str,
    *,
    max_depth: int = 6,
) -> dict[str, Any]:
    domain_id = _clean(domain_id)
    source_node_id = _clean(source_node_id)
    target_node_id = _clean(target_node_id)
    max_depth = max(1, min(int(max_depth), 10))
    if not domain_id or not source_node_id or not target_node_id:
        raise ValueError("domain_id, source_node_id and target_node_id are required")
    domain_id = resolve_published_domain_id(domain_id)
    query = f"""
        MATCH (source:PublishedEntity {{domain_id: $domain_id, node_id: $source_node_id}})
        MATCH (target:PublishedEntity {{domain_id: $domain_id, node_id: $target_node_id}})
        MATCH path=shortestPath((source)-[:PUBLISHED_RELATION|PARENT_OF*..{max_depth}]-(target))
        RETURN [node IN nodes(path) | node {{.node_id, .name, .node_type}}] AS nodes,
               [relation IN relationships(path) | {{
                   graph_relation_type: type(relation),
                   relation_code: coalesce(relation.relation_code, 'PARENT_OF'),
                   relation_label: coalesce(relation.relation_label, '层级归属'),
                   formal_relation_id: relation.formal_relation_id
               }}] AS relations,
               length(path) AS depth
    """
    with _neo4j_driver().session(database=_database()) as session:
        record = session.run(
            query,
            domain_id=domain_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
        ).single()
    if not record:
        return {
            "domain_id": domain_id,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "found": False,
            "nodes": [],
            "relations": [],
        }
    return {
        "domain_id": domain_id,
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "found": True,
        "depth": int(record.get("depth") or 0),
        "nodes": [dict(item) for item in (record.get("nodes") or [])],
        "relations": [dict(item) for item in (record.get("relations") or [])],
    }


def discover_published_graph_hypotheses(
    domain_id: str,
    node_id: str,
    *,
    peer_limit: int = 5,
    related_limit: int = 5,
) -> dict[str, Any]:
    domain_id = _clean(domain_id)
    node_id = _clean(node_id)
    peer_limit = max(0, min(int(peer_limit), 12))
    related_limit = max(0, min(int(related_limit), 12))
    if not domain_id or not node_id:
        raise ValueError("domain_id and node_id are required")
    domain_id = resolve_published_domain_id(domain_id)

    peer_query = """
        MATCH (root:PublishedEntity {domain_id: $domain_id, node_id: $node_id})
        MATCH (peer:PublishedEntity {domain_id: $domain_id})-[relation:PUBLISHED_RELATION]->(target:PublishedEntity {domain_id: $domain_id})
        WHERE peer <> root
          AND peer.node_type = root.node_type
          AND coalesce(relation.relation_code, relation.relation_label, '') <> ''
          AND NOT EXISTS {
              MATCH (root)-[existing:PUBLISHED_RELATION]->()
              WHERE existing.relation_code = relation.relation_code
          }
        WITH root, relation.relation_code AS relation_code,
             coalesce(relation.relation_label, relation.relation_code) AS relation_label,
             target.node_type AS target_type,
             count(DISTINCT peer) AS peer_support,
             collect(DISTINCT peer.name)[..5] AS peer_examples,
             collect(DISTINCT target.name)[..5] AS target_examples
        WHERE peer_support >= 2
        RETURN root.name AS source_name, relation_code, relation_label,
               target_type, peer_support, peer_examples, target_examples
        ORDER BY peer_support DESC, relation_label ASC
        LIMIT $limit
    """
    related_query = """
        MATCH (root:PublishedEntity {domain_id: $domain_id, node_id: $node_id})
        MATCH (root)-[:PUBLISHED_RELATION]-(bridge:PublishedEntity {domain_id: $domain_id})
                    -[:PUBLISHED_RELATION]-(candidate:PublishedEntity {domain_id: $domain_id})
        WHERE candidate <> root
          AND NOT (root)-[:PUBLISHED_RELATION]-(candidate)
        WITH root, candidate, count(DISTINCT bridge) AS common_neighbors,
             collect(DISTINCT bridge.name)[..5] AS bridge_examples
        WHERE common_neighbors >= 1
        RETURN root.name AS source_name,
               candidate.node_id AS target_node_id,
               candidate.name AS target_name,
               candidate.node_type AS target_type,
               common_neighbors, bridge_examples
        ORDER BY common_neighbors DESC, target_name ASC
        LIMIT $limit
    """
    with _neo4j_driver().session(database=_database()) as session:
        root = session.run(
            "MATCH (root:PublishedEntity {domain_id: $domain_id, node_id: $node_id}) "
            "RETURN root.name AS name, root.node_type AS node_type",
            domain_id=domain_id,
            node_id=node_id,
        ).single()
        if not root:
            return {
                "domain_id": domain_id,
                "node_id": node_id,
                "found": False,
                "relation_patterns": [],
                "related_entities": [],
            }
        relation_patterns = [dict(record) for record in session.run(
            peer_query,
            domain_id=domain_id,
            node_id=node_id,
            limit=peer_limit,
        )] if peer_limit else []
        related_entities = [dict(record) for record in session.run(
            related_query,
            domain_id=domain_id,
            node_id=node_id,
            limit=related_limit,
        )] if related_limit else []

    return {
        "domain_id": domain_id,
        "node_id": node_id,
        "found": True,
        "source_name": root.get("name"),
        "source_type": root.get("node_type"),
        "relation_patterns": relation_patterns,
        "related_entities": related_entities,
        "policy": "hypothesis_only_not_evidence",
    }


def _build_discovery_questions(
    payload: SemanticCompleteRequest,
    discovery: dict[str, Any],
    *,
    max_questions: int,
) -> list[CompletionQuestion]:
    source_name = _clean(discovery.get("source_name") or payload.node.name)
    questions: list[CompletionQuestion] = []
    seen: set[str] = set()

    def add(question: CompletionQuestion) -> None:
        if question.question_id in seen or len(questions) >= max_questions:
            return
        seen.add(question.question_id)
        questions.append(question)

    for item in discovery.get("relation_patterns") or []:
        relation = _clean(item.get("relation_label") or item.get("relation_code"))
        target_type = _clean(item.get("target_type") or "实体")
        if not relation:
            continue
        add(
            CompletionQuestion(
                question_id=_question_id("relation-pattern", source_name, relation, target_type),
                target_kind="relation",
                target_field=None,
                relation_intent=relation,
                temporal_role=None,
                query_text=f"可信资料是否记载{source_name}与某个{target_type}存在“{relation}”关系？具体对象是谁？",
                search_terms=_dedupe([
                    source_name,
                    relation,
                    target_type,
                    *(item.get("target_examples") or []),
                ]),
                priority=58,
                metadata={
                    "planner": "published_graph_discovery_v1",
                    "graph_discovery_kind": "relation_pattern",
                    "graph_support": int(item.get("peer_support") or 0),
                    "peer_examples": list(item.get("peer_examples") or []),
                    "target_examples": list(item.get("target_examples") or []),
                    "not_evidence": True,
                },
            )
        )

    for item in discovery.get("related_entities") or []:
        target_name = _clean(item.get("target_name"))
        if not target_name:
            continue
        add(
            CompletionQuestion(
                question_id=_question_id("related-entity", source_name, target_name),
                target_kind="relation",
                target_field=None,
                relation_intent="关联",
                temporal_role=None,
                query_text=f"可信资料是否明确记载{source_name}与{target_name}存在关系？如果存在，具体是什么关系？",
                search_terms=_dedupe([source_name, target_name, *(item.get("bridge_examples") or [])]),
                priority=54,
                metadata={
                    "planner": "published_graph_discovery_v1",
                    "graph_discovery_kind": "related_entity",
                    "candidate_target_node_id": item.get("target_node_id"),
                    "candidate_target_name": target_name,
                    "graph_support": int(item.get("common_neighbors") or 0),
                    "bridge_examples": list(item.get("bridge_examples") or []),
                    "not_evidence": True,
                },
            )
        )
    return questions


def augment_questions_with_graph_discovery(
    payload: SemanticCompleteRequest,
    questions: list[CompletionQuestion],
) -> tuple[list[CompletionQuestion], dict[str, Any]]:
    if not graph_discovery_enabled(payload):
        return questions, {
            "enabled": False,
            "reason": "disabled_for_mode_or_self_scope",
            "question_count": 0,
        }
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    max_questions = max(0, min(int(metadata.get("graph_discovery_max_questions") or 3), 8))
    if max_questions == 0:
        return questions, {"enabled": False, "reason": "question_limit_zero", "question_count": 0}
    discovery = discover_published_graph_hypotheses(
        str(payload.scenic_id),
        str(payload.node.source_node_id),
        peer_limit=max_questions,
        related_limit=max_questions,
    )
    graph_questions = _build_discovery_questions(payload, discovery, max_questions=max_questions)
    existing_ids = {item.question_id for item in questions}
    merged = list(questions)
    merged.extend(item for item in graph_questions if item.question_id not in existing_ids)
    diagnostics = {
        "enabled": True,
        "found": bool(discovery.get("found")),
        "policy": discovery.get("policy") or "hypothesis_only_not_evidence",
        "relation_pattern_count": len(discovery.get("relation_patterns") or []),
        "related_entity_count": len(discovery.get("related_entities") or []),
        "question_count": len(graph_questions),
        "questions": [
            {
                "question_id": item.question_id,
                "query_text": item.query_text,
                "relation_intent": item.relation_intent,
                "metadata": item.metadata,
            }
            for item in graph_questions
        ],
    }
    return merged, diagnostics

