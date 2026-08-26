"""Scenic sync service for A端一键入库.

First implementation lives fully inside src/rag and targets AI_DB. It upserts the
structured graph layer and leaves chunk/embedding/conflict workflows for the next
pipeline step.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.rag.dependencies import ai_session_scope

RAG_DIR = Path(__file__).resolve().parents[1]
MIGRATION_FILE = RAG_DIR / "migrations" / "20260626_sync_claims.sql"


def _to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"Unsupported payload type: {type(value)!r}")


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _sid(value: Any) -> Optional[str]:
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None


def _parse_property_value(prop: MutableMapping[str, Any]) -> None:
    raw = prop.get("raw_value")
    if raw is None:
        raw = prop.get("value")
    if not isinstance(raw, str):
        return
    raw = raw.strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        return
    try:
        data = json.loads(raw)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    prop.setdefault("raw_value", raw)
    if prop.get("value") in (None, "") and data.get("value") is not None:
        prop["value"] = str(data.get("value"))
    for key in ("source_text", "source_url", "confidence", "status"):
        if prop.get(key) in (None, "") and data.get(key) not in (None, ""):
            prop[key] = data.get(key)


def _execute_statements(db: Session, statements: Iterable[str]) -> None:
    for stmt in statements:
        sql = stmt.strip()
        if sql:
            db.execute(text(sql))


def apply_sync_schema(db: Session) -> None:
    if not MIGRATION_FILE.exists():
        raise RuntimeError(f"Missing migration file: {MIGRATION_FILE}")
    raw = MIGRATION_FILE.read_text(encoding="utf-8")
    statements = []
    current: List[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current).rstrip(";"))
            current = []
    if current:
        statements.append("\n".join(current))
    _execute_statements(db, statements)


def _record_event(db: Session, *, job_id: str, event_type: str, step: str, message: str, level: str = "info", payload: Optional[Dict[str, Any]] = None) -> None:
    db.execute(text("""
        insert into sync_job_events (job_id, event_type, step, level, message, payload)
        values (:job_id, :event_type, :step, :level, :message, cast(:payload as jsonb))
    """), {"job_id": job_id, "event_type": event_type, "step": step, "level": level, "message": message, "payload": _json(payload or {})})


def _set_job_status(db: Session, *, job_id: str, status: str, current_step: Optional[str] = None, counts: Optional[Dict[str, Any]] = None, diagnostics: Optional[List[Dict[str, Any]]] = None, error_message: Optional[str] = None, finish: bool = False) -> None:
    db.execute(text("""
        update sync_jobs
        set status = :status,
            current_step = coalesce(:current_step, current_step),
            counts = coalesce(cast(:counts as jsonb), counts),
            diagnostics = coalesce(cast(:diagnostics as jsonb), diagnostics),
            error_message = :error_message,
            started_at = case when started_at is null and :status = 'PROCESSING' then now() else started_at end,
            finished_at = case when :finish then now() else finished_at end
        where job_id = :job_id
    """), {"job_id": job_id, "status": status, "current_step": current_step, "counts": _json(counts) if counts is not None else None, "diagnostics": _json(diagnostics) if diagnostics is not None else None, "error_message": error_message, "finish": finish})


def _upsert_job(db: Session, payload: Dict[str, Any], payload_hash: str) -> str:
    scenic = payload["scenic"]
    source_system = payload.get("source_system") or "A"
    source_scenic_id = _sid(scenic.get("source_scenic_id")) or _sid(scenic.get("code"))
    if not source_scenic_id:
        raise ValueError("scenic.source_scenic_id is required")
    idempotency_key = payload.get("idempotency_key") or f"{source_system}:scenic:{source_scenic_id}:{payload_hash}"
    row = db.execute(text("select job_id from sync_jobs where idempotency_key = :idempotency_key limit 1"), {"idempotency_key": idempotency_key}).fetchone()
    if row:
        job_id = row[0]
        db.execute(text("""
            update sync_jobs
            set status='PROCESSING', current_step='received', payload_hash=:payload_hash,
                source_job_id=:source_job_id, source_system=:source_system,
                submitted_by=:submitted_by, error_message=null, started_at=now(), finished_at=null
            where job_id=:job_id
        """), {"job_id": job_id, "payload_hash": payload_hash, "source_job_id": payload.get("source_job_id"), "source_system": source_system, "submitted_by": payload.get("submitted_by")})
        return job_id
    job_id = str(uuid.uuid4())
    db.execute(text("""
        insert into sync_jobs (
            job_id, source_scenic_id, sync_version, job_type, status,
            source_system, source_job_id, idempotency_key, payload_hash,
            current_step, counts, diagnostics, metadata, submitted_by, started_at
        ) values (
            :job_id, :source_scenic_id, :sync_version, 'scenic_sync', 'PROCESSING',
            :source_system, :source_job_id, :idempotency_key, :payload_hash,
            'received', cast(:counts as jsonb), cast(:diagnostics as jsonb),
            cast(:metadata as jsonb), :submitted_by, now()
        )
    """), {"job_id": job_id, "source_scenic_id": source_scenic_id, "sync_version": payload.get("schema_version"), "source_system": source_system, "source_job_id": payload.get("source_job_id"), "idempotency_key": idempotency_key, "payload_hash": payload_hash, "counts": _json({}), "diagnostics": _json(payload.get("diagnostics") or []), "metadata": _json(payload.get("metadata") or {}), "submitted_by": payload.get("submitted_by")})
    return job_id


def _upsert_scenic(db: Session, scenic: Dict[str, Any]) -> int:
    source_scenic_id = _sid(scenic.get("source_scenic_id")) or _sid(scenic.get("code"))
    if not source_scenic_id:
        raise ValueError("scenic.source_scenic_id is required")
    row = db.execute(text("""
        insert into scenic_areas (source_scenic_id, source_scenic_pk, name, description, location, metadata, updated_at)
        values (:source_scenic_id, :source_scenic_pk, :name, :description, :location, cast(:metadata as jsonb), now())
        on conflict (source_scenic_id) do update set
            source_scenic_pk = excluded.source_scenic_pk,
            name = excluded.name,
            description = excluded.description,
            location = excluded.location,
            metadata = excluded.metadata,
            updated_at = now()
        returning id
    """), {"source_scenic_id": source_scenic_id, "source_scenic_pk": scenic.get("source_scenic_pk"), "name": scenic.get("name") or scenic.get("code") or source_scenic_id, "description": scenic.get("description"), "location": scenic.get("location"), "metadata": _json(scenic.get("metadata") or {})}).fetchone()
    return int(row[0])


def _upsert_nodes(db: Session, scenic_id: int, source_scenic_id: str, nodes: List[Dict[str, Any]], sync_version: str) -> int:
    count = 0
    for node in nodes:
        source_node_id = _sid(node.get("source_node_id"))
        if not source_node_id:
            continue
        db.execute(text("""
            insert into semantic_nodes (
                scenic_id, source_scenic_id, source_node_id, parent_source_node_id,
                node_name, node_type, description, lng, lat, properties, tags,
                content_hash, sync_version, source_updated_at, source_table,
                source_pk, source_url, source_title, updated_at
            ) values (
                :scenic_id, :source_scenic_id, :source_node_id, :parent_source_node_id,
                :node_name, :node_type, :description, :lng, :lat,
                cast(:properties as jsonb), cast(:tags as jsonb), :content_hash,
                :sync_version, :source_updated_at, 'wiki_custom_node',
                :source_pk, :source_url, :source_title, now()
            )
            on conflict (scenic_id, source_node_id) do update set
                parent_source_node_id = excluded.parent_source_node_id,
                node_name = excluded.node_name,
                node_type = excluded.node_type,
                description = excluded.description,
                lng = excluded.lng,
                lat = excluded.lat,
                properties = excluded.properties,
                tags = excluded.tags,
                content_hash = excluded.content_hash,
                sync_version = excluded.sync_version,
                source_updated_at = excluded.source_updated_at,
                source_pk = excluded.source_pk,
                source_url = excluded.source_url,
                source_title = excluded.source_title,
                updated_at = now()
        """), {"scenic_id": scenic_id, "source_scenic_id": source_scenic_id, "source_node_id": source_node_id, "parent_source_node_id": _sid(node.get("parent_source_node_id")), "node_name": node.get("name") or node.get("node_name"), "node_type": node.get("node_type"), "description": node.get("description"), "lng": node.get("lng"), "lat": node.get("lat"), "properties": _json(node.get("properties") or {}), "tags": _json(node.get("tags") or []), "content_hash": _hash_payload(node), "sync_version": sync_version, "source_updated_at": node.get("source_updated_at"), "source_pk": source_node_id, "source_url": node.get("source_url"), "source_title": node.get("name") or node.get("node_name")})
        count += 1
    return count


def _upsert_properties(db: Session, scenic_id: int, source_scenic_id: str, props: List[Dict[str, Any]], sync_version: str) -> int:
    count = 0
    aggregated: Dict[str, Dict[str, Any]] = {}
    status_rank = {"accepted": 3, "disputed": 2, "proposed": 1, "rejected": 0, "replaced": 0}
    for prop in props:
        _parse_property_value(prop)
        source_property_id = _sid(prop.get("source_property_id") or prop.get("id"))
        source_node_id = _sid(prop.get("source_node_id") or prop.get("node_id"))
        key = _sid(prop.get("key") or prop.get("property_key"))
        if not (source_property_id and source_node_id and key):
            continue
        claim_status = prop.get("claim_status") or prop.get("status") or prop.get("outer_status")
        metadata = dict(prop.get("metadata") or {})
        metadata.update({"outer_status": prop.get("outer_status"), "reviewed_at": prop.get("reviewed_at")})
        db.execute(text("""
            insert into node_property_claims (
                scenic_id, source_scenic_id, source_property_id, source_node_id,
                property_key, raw_value, value, value_type, outer_status,
                claim_status, confidence, source_text, source_url, evidence_source_id,
                is_locked, version, sync_version, source_table, source_pk, metadata, updated_at
            ) values (
                :scenic_id, :source_scenic_id, :source_property_id, :source_node_id,
                :property_key, :raw_value, :value, :value_type, :outer_status,
                :claim_status, :confidence, :source_text, :source_url, :evidence_source_id,
                :is_locked, :version, :sync_version, 'wiki_custom_nodeproperty',
                :source_pk, cast(:metadata as jsonb), now()
            )
            on conflict (scenic_id, source_property_id) do update set
                source_node_id = excluded.source_node_id,
                property_key = excluded.property_key,
                raw_value = excluded.raw_value,
                value = excluded.value,
                value_type = excluded.value_type,
                outer_status = excluded.outer_status,
                claim_status = excluded.claim_status,
                confidence = excluded.confidence,
                source_text = excluded.source_text,
                source_url = excluded.source_url,
                evidence_source_id = excluded.evidence_source_id,
                is_locked = excluded.is_locked,
                version = excluded.version,
                sync_version = excluded.sync_version,
                source_pk = excluded.source_pk,
                metadata = excluded.metadata,
                updated_at = now()
        """), {"scenic_id": scenic_id, "source_scenic_id": source_scenic_id, "source_property_id": source_property_id, "source_node_id": source_node_id, "property_key": key, "raw_value": prop.get("raw_value"), "value": None if prop.get("value") is None else str(prop.get("value")), "value_type": prop.get("value_type") or "string", "outer_status": prop.get("outer_status"), "claim_status": claim_status, "confidence": prop.get("confidence"), "source_text": prop.get("source_text"), "source_url": prop.get("source_url"), "evidence_source_id": prop.get("evidence_source_id"), "is_locked": bool(prop.get("is_locked")), "version": prop.get("version") or 1, "sync_version": sync_version, "source_pk": source_property_id, "metadata": _json(metadata)})
        current = aggregated.setdefault(source_node_id, {})
        prior = current.get(key)
        new_rank = status_rank.get(str(claim_status or "").lower(), 1)
        old_rank = status_rank.get(str((prior or {}).get("status") or "").lower(), -1) if prior else -1
        if prior is None or new_rank >= old_rank:
            current[key] = {"value": prop.get("value"), "status": claim_status, "source_property_id": source_property_id}
        count += 1
    for source_node_id, values in aggregated.items():
        db.execute(text("""
            update semantic_nodes
            set properties = cast(:properties as jsonb), updated_at = now()
            where scenic_id = :scenic_id and source_node_id = :source_node_id
        """), {"scenic_id": scenic_id, "source_node_id": source_node_id, "properties": _json(values)})
    return count


def _upsert_edges(db: Session, scenic_id: int, source_scenic_id: str, relations: List[Dict[str, Any]], sync_version: str) -> int:
    count = 0
    for rel in relations:
        source_node_id = _sid(rel.get("source_node_id"))
        target_node_id = _sid(rel.get("target_node_id"))
        relation_type = _sid(rel.get("relation_type"))
        if not (source_node_id and target_node_id and relation_type):
            continue
        source_relation_id = _sid(rel.get("source_relation_id") or rel.get("id"))
        layer = rel.get("relation_layer") or rel.get("relation_category") or "semantic"
        layer = layer if layer in {"spatial", "semantic"} else "semantic"
        properties = dict(rel.get("metadata") or {})
        for key in ("evidence_source_id", "extraction_method", "version", "sort_order"):
            if rel.get(key) is not None:
                properties[key] = rel.get(key)
        if source_relation_id:
            existing = db.execute(text("select id from semantic_edges where scenic_id=:scenic_id and source_relation_id=:source_relation_id limit 1"), {"scenic_id": scenic_id, "source_relation_id": source_relation_id}).fetchone()
            if existing:
                db.execute(text("""
                    update semantic_edges set
                        source_node_id=:source_node_id, target_node_id=:target_node_id,
                        relation_type=:relation_type, relation_label=:relation_label,
                        relation_layer=:relation_layer, relation_category=:relation_category,
                        description=:description, evidence_text=:evidence_text,
                        confidence=:confidence, is_verified=:is_verified,
                        properties=cast(:properties as jsonb), sync_version=:sync_version,
                        source_pk=:source_pk, source_url=:source_url, updated_at=now()
                    where id=:id
                """), {"id": existing[0], "source_node_id": source_node_id, "target_node_id": target_node_id, "relation_type": relation_type, "relation_label": rel.get("relation_type_label") or rel.get("relation_label"), "relation_layer": layer, "relation_category": rel.get("relation_category"), "description": rel.get("description"), "evidence_text": rel.get("evidence_text"), "confidence": rel.get("confidence"), "is_verified": bool(rel.get("is_verified")), "properties": _json(properties), "sync_version": sync_version, "source_pk": source_relation_id, "source_url": rel.get("source_url")})
                count += 1
                continue
        db.execute(text("""
            insert into semantic_edges (
                scenic_id, source_scenic_id, source_relation_id, source_node_id,
                target_node_id, relation_type, relation_label, relation_layer,
                relation_category, description, evidence_text, confidence,
                is_verified, properties, sync_version, source_table, source_pk,
                source_url, updated_at
            ) values (
                :scenic_id, :source_scenic_id, :source_relation_id, :source_node_id,
                :target_node_id, :relation_type, :relation_label, :relation_layer,
                :relation_category, :description, :evidence_text, :confidence,
                :is_verified, cast(:properties as jsonb), :sync_version,
                'wiki_custom_noderelation', :source_pk, :source_url, now()
            )
            on conflict (scenic_id, source_node_id, target_node_id, relation_type) do update set
                source_relation_id = excluded.source_relation_id,
                relation_label = excluded.relation_label,
                relation_layer = excluded.relation_layer,
                relation_category = excluded.relation_category,
                description = excluded.description,
                evidence_text = excluded.evidence_text,
                confidence = excluded.confidence,
                is_verified = excluded.is_verified,
                properties = excluded.properties,
                sync_version = excluded.sync_version,
                source_pk = excluded.source_pk,
                source_url = excluded.source_url,
                updated_at = now()
        """), {"scenic_id": scenic_id, "source_scenic_id": source_scenic_id, "source_relation_id": source_relation_id, "source_node_id": source_node_id, "target_node_id": target_node_id, "relation_type": relation_type, "relation_label": rel.get("relation_type_label") or rel.get("relation_label"), "relation_layer": layer, "relation_category": rel.get("relation_category"), "description": rel.get("description"), "evidence_text": rel.get("evidence_text"), "confidence": rel.get("confidence"), "is_verified": bool(rel.get("is_verified")), "properties": _json(properties), "sync_version": sync_version, "source_pk": source_relation_id, "source_url": rel.get("source_url")})
        count += 1
    return count


def _upsert_assets(db: Session, scenic_id: int, source_scenic_id: str, payload: Dict[str, Any], sync_version: str) -> int:
    assets = {str(a.get("source_asset_id")): a for a in payload.get("image_assets") or [] if a.get("source_asset_id") is not None}
    count = 0
    for binding in payload.get("image_bindings") or []:
        if binding.get("object_type") not in (None, "node"):
            continue
        source_asset_id = _sid(binding.get("source_asset_id"))
        source_node_id = _sid(binding.get("source_node_id"))
        if not (source_asset_id and source_node_id):
            continue
        source_binding_id = _sid(binding.get("source_binding_id"))
        asset = assets.get(source_asset_id, {})
        metadata = dict(asset.get("metadata") or {})
        metadata.update(binding.get("metadata") or {})
        for key in ("original_filename", "source", "exif_lat", "exif_lng", "sort_order"):
            if asset.get(key) is not None:
                metadata[key] = asset.get(key)
            if binding.get(key) is not None:
                metadata[key] = binding.get(key)
        existing = db.execute(text("""
            select id from node_assets
            where scenic_id=:scenic_id and source_asset_id=:source_asset_id
              and coalesce(source_binding_id, '__NULL__') = coalesce(:source_binding_id, '__NULL__')
            limit 1
        """), {"scenic_id": scenic_id, "source_asset_id": source_asset_id, "source_binding_id": source_binding_id}).fetchone()
        params = {"scenic_id": scenic_id, "source_scenic_id": source_scenic_id, "source_asset_id": source_asset_id, "source_binding_id": source_binding_id, "source_node_id": source_node_id, "asset_type": asset.get("asset_type") or "image", "url": asset.get("url") or asset.get("file_url"), "title": asset.get("title") or asset.get("original_filename"), "caption": asset.get("caption"), "ocr_text": asset.get("ocr_text"), "role": binding.get("role"), "is_cover": bool(binding.get("is_cover")), "file_hash": asset.get("file_hash"), "metadata": _json(metadata), "content_hash": _hash_payload({"asset": asset, "binding": binding}), "sync_version": sync_version, "source_pk": source_asset_id, "source_url": asset.get("url") or asset.get("file_url")}
        if existing:
            params["id"] = existing[0]
            db.execute(text("""
                update node_assets set
                    source_node_id=:source_node_id, asset_type=:asset_type, url=:url,
                    title=:title, caption=:caption, ocr_text=:ocr_text, role=:role,
                    is_cover=:is_cover, file_hash=:file_hash, metadata=cast(:metadata as jsonb),
                    content_hash=:content_hash, sync_version=:sync_version, source_pk=:source_pk,
                    source_url=:source_url, updated_at=now()
                where id=:id
            """), params)
        else:
            db.execute(text("""
                insert into node_assets (
                    scenic_id, source_scenic_id, source_asset_id, source_binding_id,
                    source_node_id, asset_type, url, title, caption, ocr_text, role,
                    is_cover, file_hash, metadata, content_hash, sync_version,
                    source_table, source_pk, source_url, updated_at
                ) values (
                    :scenic_id, :source_scenic_id, :source_asset_id, :source_binding_id,
                    :source_node_id, :asset_type, :url, :title, :caption, :ocr_text, :role,
                    :is_cover, :file_hash, cast(:metadata as jsonb), :content_hash,
                    :sync_version, 'wiki_custom_imageasset', :source_pk, :source_url, now()
                )
            """), params)
        count += 1
    return count


async def sync_scenic_service(request: Any) -> Dict[str, Any]:
    payload = _to_dict(request)
    payload_hash = _hash_payload(payload)
    diagnostics = list(payload.get("diagnostics") or [])
    counts = {"nodes": len(payload.get("nodes") or []), "relations": len(payload.get("relations") or []), "properties": len(payload.get("properties") or []), "image_assets": len(payload.get("image_assets") or []), "image_bindings": len(payload.get("image_bindings") or [])}
    with ai_session_scope() as db:
        apply_sync_schema(db)
        job_id = _upsert_job(db, payload, payload_hash)
        _record_event(db, job_id=job_id, event_type="received", step="received", message="sync payload received", payload=counts)
        try:
            _set_job_status(db, job_id=job_id, status="PROCESSING", current_step="upsert_scenic", counts=counts, diagnostics=diagnostics)
            scenic_id = _upsert_scenic(db, payload["scenic"])
            source_scenic_id = _sid(payload["scenic"].get("source_scenic_id")) or _sid(payload["scenic"].get("code"))
            sync_version = payload.get("schema_version") or "unknown"
            _record_event(db, job_id=job_id, event_type="progress", step="upsert_scenic", message="scenic upserted", payload={"scenic_id": scenic_id})
            _set_job_status(db, job_id=job_id, status="PROCESSING", current_step="upsert_nodes")
            upserted_nodes = _upsert_nodes(db, scenic_id, source_scenic_id, payload.get("nodes") or [], sync_version)
            _set_job_status(db, job_id=job_id, status="PROCESSING", current_step="upsert_properties")
            upserted_props = _upsert_properties(db, scenic_id, source_scenic_id, payload.get("properties") or [], sync_version)
            _set_job_status(db, job_id=job_id, status="PROCESSING", current_step="upsert_relations")
            upserted_edges = _upsert_edges(db, scenic_id, source_scenic_id, payload.get("relations") or [], sync_version)
            _set_job_status(db, job_id=job_id, status="PROCESSING", current_step="upsert_assets")
            upserted_assets = _upsert_assets(db, scenic_id, source_scenic_id, payload, sync_version)
            counts.update({"upserted_nodes": upserted_nodes, "upserted_properties": upserted_props, "upserted_relations": upserted_edges, "upserted_assets": upserted_assets})
            _set_job_status(db, job_id=job_id, status="SUCCESS", current_step="structured_sync_completed", counts=counts, diagnostics=diagnostics, finish=True)
            _record_event(db, job_id=job_id, event_type="completed", step="structured_sync_completed", message="structured sync completed", payload=counts)
            return {"job_id": job_id, "status": "SUCCESS", "source_scenic_id": source_scenic_id, "counts": counts, "diagnostics": diagnostics}
        except Exception as exc:
            _set_job_status(db, job_id=job_id, status="FAILED", current_step="failed", counts=counts, diagnostics=diagnostics, error_message=str(exc), finish=True)
            _record_event(db, job_id=job_id, event_type="failed", step="failed", level="error", message=str(exc))
            raise


async def get_job_status(job_id: str) -> Dict[str, Any]:
    with ai_session_scope() as db:
        apply_sync_schema(db)
        row = db.execute(text("""
            select job_id, status, current_step, counts, diagnostics, error_message
            from sync_jobs where job_id=:job_id
        """), {"job_id": job_id}).mappings().fetchone()
        if not row:
            return {"job_id": job_id, "status": "NOT_FOUND", "events": []}
        events = db.execute(text("""
            select event_type, step, level, message, payload, created_at
            from sync_job_events
            where job_id=:job_id
            order by created_at desc, id desc
            limit 50
        """), {"job_id": job_id}).mappings().all()
        return {"job_id": row["job_id"], "status": row["status"], "current_step": row["current_step"], "counts": row["counts"] or {}, "diagnostics": row["diagnostics"] or [], "error_message": row["error_message"], "events": [dict(item) for item in events]}


async def build_indexes() -> Dict[str, Any]:
    return {"status": "ok", "msg": "index build is handled by dedicated embedding/index tasks"}
