"""Resolve relation targets against the published domain graph."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.schemas import CandidateClaim, SemanticCompleteRequest
from src.rag.service.entity_type_service import infer_entity_type


ALIAS_KEYS = {"alias", "aliases", "别名", "曾用名", "简称", "又名", "英文名"}

# Preserve explicit model types; unknown types remain reviewable custom types.
ENTITY_TYPE_ALIASES = {
    "poi": "poi", "place": "poi", "地点": "poi", "场所": "poi",
    "building": "building", "建筑": "building",
    "region": "region", "location": "region", "区域": "region",
    "person": "person", "人物": "person",
    "object": "object", "物品": "object", "文物": "object", "artifact": "object",
    "facility": "facility", "设施": "facility",
    "organization": "organization", "organisation": "organization", "机构": "organization",
    "school": "organization", "大学": "organization",
    "event": "event", "事件": "event", "route": "route", "路线": "route",
    "concept": "concept", "概念": "concept", "program": "program", "项目": "program",
    "scenicarea": "scenicarea", "景区": "scenicarea",
}

def normalize_entity_type(value: Any) -> str:
    raw = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    if "=" in raw:
        code, _label = raw.split("=", 1)
        if code.strip().startswith(("type_", "domain_")) or code.strip() == "ce5":
            raw = code.strip()
    return ENTITY_TYPE_ALIASES.get(raw, raw[:64]) if raw else ""


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
        if str(key or "").strip().casefold() not in ALIAS_KEYS:
            continue
        aliases.update(normalize_entity_name(item) for item in _text_values(value))
    return {item for item in aliases if item}


def _schema_relation_types(payload: SemanticCompleteRequest, predicate: str) -> list[str]:
    schema = (payload.metadata or {}).get("domain_schema") or {}
    found: list[str] = []
    for collection in (schema.get("relations") or {}, schema.get("relation_intents") or {}):
        items = collection.values() if isinstance(collection, dict) else collection
        for item_group in items or []:
            group = item_group if isinstance(item_group, list) else [item_group]
            for item in group:
                if not isinstance(item, dict):
                    continue
                labels = {str(item.get("label") or "").strip(), str(item.get("code") or "").strip()}
                if predicate and predicate not in labels:
                    continue
                for node_type in item.get("allowed_target_types") or []:
                    value = str(node_type or "").strip()
                    if value and value not in found:
                        found.append(value)
    return found


def _type_suggestion(payload: SemanticCompleteRequest, claim: CandidateClaim) -> tuple[str | None, str | None, float]:
    raw_type = normalize_entity_type(claim.object_type)
    inferred_type, inferred_confidence, _ = infer_entity_type(raw_type, predicate=claim.predicate, mention=claim.object_name or claim.object_value, quote=claim.quote)
    if inferred_type:
        return raw_type or None, inferred_type, round(max(inferred_confidence, min(float(claim.confidence or 0.0), 0.9)), 3)
    schema_types = _schema_relation_types(payload, str(claim.predicate or "").strip())
    if len(schema_types) == 1:
        return None, schema_types[0], 0.65
    return None, None, 0.0


def _candidate_uid(source_scenic_id: str, normalized_name: str, suggested_type: str | None) -> str:
    raw = json.dumps(
        {"domain": source_scenic_id, "name": normalized_name, "suggested_type": suggested_type or ""},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolve_candidate_entities(
    payload: SemanticCompleteRequest,
    claims: list[CandidateClaim],
    *,
    trace_id: str,
    job_id: int | None = None,
    persist_new_candidates: bool = True,
) -> dict[str, int]:
    """Annotate relation claims and persist unresolved targets as node candidates."""
    relation_claims = [claim for claim in claims if claim.claim_type == "relation" and (claim.object_name or claim.object_value)]
    stats = defaultdict(int)
    if not relation_claims:
        return dict(stats)

    # Import lazily to avoid a module cycle during service initialization.
    from src.rag.service.semantic_candidate_store import apply_semantic_candidate_schema

    source_scenic_id = str(payload.scenic_id or "").strip()
    with ai_session_scope() as db:
        apply_semantic_candidate_schema(db)
        rows = db.execute(
            text(
                """
                select source_node_id, node_name, node_type, properties
                from semantic_nodes
                where source_scenic_id = :source_scenic_id
                """
            ),
            {"source_scenic_id": source_scenic_id},
        ).mappings().all()
        exact_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        alias_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            record = dict(row)
            name_key = normalize_entity_name(record.get("node_name"))
            if name_key:
                exact_index[name_key].append(record)
            for alias in _aliases(record.get("properties")):
                alias_index[alias].append(record)

        for claim in relation_claims:
            raw_name = str(claim.object_name or claim.object_value or "").strip()
            normalized_name = normalize_entity_name(raw_name)
            raw_type, suggested_type, type_confidence = _type_suggestion(payload, claim)
            exact = exact_index.get(normalized_name, [])
            aliases = alias_index.get(normalized_name, []) if not exact else []
            matches = exact or aliases
            status = "NEW_ENTITY"
            target_node_id = None
            target_node_candidate_id = None
            possible_nodes = [
                {
                    "target_node_id": str(item.get("source_node_id") or ""),
                    "name": item.get("node_name"),
                    "node_type": item.get("node_type"),
                    "match_type": "EXACT_MATCH" if exact else "ALIAS_MATCH",
                }
                for item in matches
            ]
            if len(matches) == 1:
                status = "EXACT_MATCH" if exact else "ALIAS_MATCH"
                target_node_id = str(matches[0].get("source_node_id") or "") or None
                stats[status.lower()] += 1
            elif len(matches) > 1:
                status = "AMBIGUOUS"
                stats["ambiguous"] += 1
            else:
                if not persist_new_candidates:
                    stats["new_entity"] += 1
                    claim.entity_resolution_status = status
                    claim.raw_type = raw_type
                    claim.suggested_type = suggested_type
                    claim.type_confidence = type_confidence
                    claim.metadata = dict(claim.metadata or {})
                    claim.metadata.update(
                        {
                            "target_node_id": None,
                            "target_node_candidate_id": None,
                            "entity_resolution_status": status,
                            "possible_nodes": [],
                            "raw_type": raw_type,
                            "suggested_type": suggested_type,
                            "type_confidence": type_confidence,
                        }
                    )
                    continue
                uid = _candidate_uid(source_scenic_id, normalized_name, suggested_type)
                entity_group_key = uid[:32]
                row = db.execute(
                    text(
                        """
                        insert into semantic_node_candidates (
                            candidate_uid, trace_id, job_id, source_scenic_id, name,
                            normalized_name, node_type, raw_type, suggested_type,
                            type_confidence, entity_group_key, parent_node_id,
                            evidence_ids, confidence, source_count,
                            entity_resolution_status, status, metadata, updated_at
                        ) values (
                            :candidate_uid, :trace_id, :job_id, :source_scenic_id, :name,
                            :normalized_name, :node_type, :raw_type, :suggested_type,
                            :type_confidence, :entity_group_key, :parent_node_id,
                            cast(:evidence_ids as jsonb), :confidence, :source_count,
                            'NEW_ENTITY', 'PENDING', cast(:metadata as jsonb), now()
                        )
                        on conflict (candidate_uid) do update set
                            trace_id = excluded.trace_id,
                            job_id = excluded.job_id,
                            raw_type = coalesce(excluded.raw_type, semantic_node_candidates.raw_type),
                            suggested_type = coalesce(excluded.suggested_type, semantic_node_candidates.suggested_type),
                            type_confidence = greatest(semantic_node_candidates.type_confidence, excluded.type_confidence),
                            evidence_ids = excluded.evidence_ids,
                            confidence = greatest(semantic_node_candidates.confidence, excluded.confidence),
                            source_count = greatest(semantic_node_candidates.source_count, excluded.source_count),
                            metadata = semantic_node_candidates.metadata || excluded.metadata,
                            updated_at = now()
                        returning id
                        """
                    ),
                    {
                        "candidate_uid": uid,
                        "trace_id": trace_id,
                        "job_id": int(job_id) if job_id is not None else None,
                        "source_scenic_id": source_scenic_id,
                        "name": raw_name,
                        "normalized_name": normalized_name,
                        "node_type": suggested_type,
                        "raw_type": raw_type,
                        "suggested_type": suggested_type,
                        "type_confidence": type_confidence,
                        "entity_group_key": entity_group_key,
                        "parent_node_id": str(claim.subject_node_id or payload.node.source_node_id or ""),
                        "evidence_ids": json.dumps(claim.evidence_ids or [], ensure_ascii=False),
                        "confidence": float(claim.confidence or 0.0),
                        "source_count": 1 if (claim.source_id or claim.source_url or claim.evidence_ids) else 0,
                        "metadata": json.dumps(
                            {
                                "relation_type": claim.predicate,
                                "source_node_id": str(claim.subject_node_id or payload.node.source_node_id or ""),
                                "raw_type": raw_type,
                                "suggested_type": suggested_type,
                                "type_confidence": type_confidence,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ).mappings().first()
                target_node_candidate_id = int(row["id"]) if row else None
                stats["new_entity"] += 1

            claim.target_node_id = target_node_id
            claim.target_node_candidate_id = target_node_candidate_id
            claim.entity_resolution_status = status
            claim.possible_nodes = possible_nodes
            claim.raw_type = raw_type
            claim.suggested_type = suggested_type
            claim.type_confidence = type_confidence
            claim.metadata = dict(claim.metadata or {})
            claim.metadata.update(
                {
                    "target_node_id": target_node_id,
                    "target_node_candidate_id": target_node_candidate_id,
                    "entity_resolution_status": status,
                    "possible_nodes": possible_nodes,
                    "raw_type": raw_type,
                    "suggested_type": suggested_type,
                    "type_confidence": type_confidence,
                }
            )
    return dict(stats)
