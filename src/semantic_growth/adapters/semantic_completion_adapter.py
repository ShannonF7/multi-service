"""
Adapter: LangGraph growth nodes → semantic completion services.

Design:
  1. build_node_snapshot_from_opportunity  — Opportunity snapshot is authoritative
  2. enrich_node_snapshot_from_detail      — Detail only fills missing fields
  3. validate_node_snapshot                — Reject invalid context, never fallback to "node_xxx"
  4. run_semantic_completion_for_node      — Build request → call pipeline → extract results
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.rag.service.graph_discovery_service import (
    get_published_domain_overview,
    get_published_node_detail,
    resolve_published_domain_id,
)
from src.rag.service.semantic_completion_service import complete_semantic_service
from src.rag.service.gap_status_service import list_semantic_gap_status
from src.rag.schemas import SemanticCompleteRequest, SemanticNodeContext

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════

def discover_domain_nodes(domain_identifier: str) -> list[dict[str, Any]]:
    """Get all published nodes for a domain from Neo4j.

    Returns node dicts carrying the snapshot fields that downstream
    workers need: node_id, name, node_type, parent_name, scenic_name,
    and existing property/relation keys for gap computation.
    """
    try:
        domain_id = resolve_published_domain_id(domain_identifier)
    except Exception:
        domain_id = domain_identifier

    try:
        overview = get_published_domain_overview(domain_identifier)
    except Exception as exc:
        logger.warning("discover_domain_nodes overview failed: %s", exc)
        return []

    nodes = overview.get("nodes") or []
    enriched = []

    for node in nodes:
        node_id = str(node.get("node_id") or node.get("id") or "")
        if not node_id:
            continue

        # ── Build a self-contained node snapshot ──
        snapshot: dict[str, Any] = {
            "node_id": node_id,
            "name": str(node.get("name") or ""),
            "node_type": str(node.get("node_type") or node.get("type") or ""),
            "parent_name": str(node.get("parent_name") or ""),
            "parent_node_id": str(node.get("parent_node_id") or node.get("parent_id") or ""),
            "scenic_name": str(node.get("scenic_name") or node.get("domain_name") or ""),
            "existing_property_keys": _extract_property_keys(node),
            "existing_relation_types": _extract_relation_types(node),
            "discovered_from": "domain_overview",
            "_domain_id": domain_id,
        }

        # Enrich with gap status from DB
        try:
            gaps = list_semantic_gap_status(
                source_scenic_id=domain_identifier,
                source_node_id=node_id,
                limit=5,
            )
            snapshot["_has_gaps"] = bool(gaps.get("items"))
            snapshot["_open_gaps"] = [
                g for g in (gaps.get("items") or [])
                if g.get("status") not in ("completed", "resolved")
            ]
            snapshot["_has_completion"] = bool(gaps.get("items"))
        except Exception:
            snapshot["_has_gaps"] = False
            snapshot["_open_gaps"] = []
            snapshot["_has_completion"] = False

        enriched.append(snapshot)

    return enriched


def _extract_property_keys(node: dict[str, Any]) -> list[str]:
    """Extract existing property keys from a node dict or object."""
    props = node.get("properties") or {}
    if isinstance(props, dict):
        return [str(k) for k in props.keys()]
    if isinstance(props, list):
        return [
            str(getattr(p, "key", p.get("key", "")))
            for p in props
            if str(getattr(p, "key", p.get("key", "")))
        ]
    return []


def _extract_relation_types(node: dict[str, Any]) -> list[str]:
    """Extract existing relation types from a node dict."""
    rels = node.get("relations") or []
    types: list[str] = []
    for rel in (rels if isinstance(rels, list) else []):
        types.append(_normalize_relation(rel)["relation_type"])
    return [t for t in types if t]


# ═══════════════════════════════════════════════════════════════════
# Relation normalization (fixes getattr(rel, …, rel.get(…)) bug)
# ═══════════════════════════════════════════════════════════════════

def _normalize_relation(rel: Any) -> dict[str, str]:
    """Safely extract relation_type and target_name from dict or object."""
    if isinstance(rel, dict):
        return {
            "relation_type": str(
                rel.get("relation_type") or rel.get("type") or ""
            ),
            "target_name": str(
                rel.get("target_name") or rel.get("target") or ""
            ),
        }
    # Attribute-based object
    return {
        "relation_type": str(getattr(rel, "relation_type", "") or ""),
        "target_name": str(getattr(rel, "target_name", "") or ""),
    }


def _normalize_properties(props: Any) -> list[dict[str, str]]:
    """Safely extract key-value pairs from properties dict or list."""
    result: list[dict[str, str]] = []
    if isinstance(props, dict):
        for k, v in props.items():
            result.append({"key": str(k), "value": str(v)})
    elif isinstance(props, list):
        for item in props:
            if isinstance(item, dict):
                result.append({
                    "key": str(item.get("key", "")),
                    "value": str(item.get("value", "")),
                })
            else:
                key = str(getattr(item, "key", ""))
                val = str(getattr(item, "value", ""))
                if key:
                    result.append({"key": key, "value": val})
    return result


# ═══════════════════════════════════════════════════════════════════
# Gap detection
# ═══════════════════════════════════════════════════════════════════

def build_completion_payload_for_node(domain_identifier: str, node: dict[str, Any]) -> dict[str, Any]:
    """Return unconstrained growth context; no fixed property/relation template."""
    return {
        "node_id": str(node.get("node_id") or node.get("id") or ""),
        "node_name": str(node.get("name") or ""),
        "node_type": str(node.get("node_type") or node.get("type") or ""),
        "target_fields": [],
        "relation_intents": [],
        "published_properties": list(node.get("existing_property_keys") or []),
        "published_relations": list(node.get("existing_relation_types") or []),
    }

def _expected_properties(node_type: str, template_config: dict) -> list[str]:
    defaults: dict[str, list[str]] = {
        "building": ["名称", "别名", "始建年代", "修缮年代", "地址", "简介", "建筑风格", "保护级别"],
        "poi": ["名称", "别名", "简介", "位置", "年代", "类型"],
        "person": ["姓名", "别称", "生卒年", "简介", "主要成就"],
        "object": ["名称", "年代", "材质", "简介", "来源"],
        "region": ["名称", "简介", "历史沿革", "面积"],
        "scenicarea": ["名称", "简介", "特色", "主要景点"],
        "event": ["名称", "时间", "简介", "相关人物", "影响"],
        "literature": ["名称", "作者", "年代", "简介", "类型"],
    }
    expected = list(defaults.get(node_type.lower(), ["名称", "简介"]))
    template_props = template_config.get("properties") or template_config.get("node_properties") or []
    for tp in template_props:
        name = str(tp.get("name") or tp.get("key") or "")
        if name and name not in expected:
            expected.append(name)
    return expected


def _expected_relations(node_type: str, template_config: dict) -> list[str]:
    defaults: dict[str, list[str]] = {
        "building": ["位于", "包含", "属于"],
        "poi": ["位于", "属于", "关联"],
        "person": ["关联", "创作"],
        "object": ["位于", "关联"],
        "region": ["包含", "位于"],
        "scenicarea": ["包含", "位于"],
    }
    expected = list(defaults.get(node_type.lower(), ["关联"]))
    template_rels = template_config.get("relations") or template_config.get("node_relations") or []
    for tr in template_rels:
        name = str(tr.get("name") or tr.get("relation_type") or "")
        if name and name not in expected:
            expected.append(name)
    return expected


# ═══════════════════════════════════════════════════════════════════
# Three-step node context builder
# ═══════════════════════════════════════════════════════════════════

def build_node_context_from_opportunity(opp: dict[str, Any]) -> dict[str, Any]:
    """Step 1: extract authoratative fields from the opportunity snapshot."""
    return {
        "node_id": str(opp.get("node_id") or ""),
        "name": str(
            opp.get("_node_name")
            or opp.get("node_name")
            or opp.get("_snapshot_name")
            or ""
        ),
        "node_type": str(
            opp.get("_node_type")
            or opp.get("node_type")
            or opp.get("_snapshot_type")
            or ""
        ),
        "parent_name": str(
            opp.get("_parent_name")
            or opp.get("parent_name")
            or ""
        ),
        "scenic_name": str(
            opp.get("_scenic_name")
            or opp.get("scenic_name")
            or ""
        ),
        "graph_context": dict(
            opp.get("_graph_context")
            or opp.get("graph_context")
            or {}
        ),
        "existing_properties": list(
            opp.get("_existing_properties")
            or opp.get("existing_properties")
            or []
        ),
        "existing_relations": list(
            opp.get("_existing_relations")
            or opp.get("existing_relations")
            or []
        ),
    }


def enrich_node_context_from_detail(
    ctx: dict[str, Any],
    *,
    domain_id: str,
    node_id: str,
) -> dict[str, Any]:
    """Step 2: fill missing fields from Neo4j detail. Never overwrite non-empty fields."""
    try:
        detail = get_published_node_detail(domain_id, node_id) or {}
    except Exception:
        detail = {}

    # Identity fields: snapshot wins if non-empty
    ctx["name"] = ctx["name"] or str(detail.get("name") or "")
    ctx["node_type"] = ctx["node_type"] or str(detail.get("node_type") or "")
    ctx["parent_name"] = ctx["parent_name"] or str(detail.get("parent_name") or "")
    ctx["scenic_name"] = ctx["scenic_name"] or str(detail.get("scenic_name") or "")

    # Factual fields: detail supplements
    if not ctx["graph_context"]:
        ctx["graph_context"] = detail.get("graph_context") or {}

    if not ctx["existing_properties"]:
        ctx["existing_properties"] = _normalize_properties(
            detail.get("properties")
        )

    if not ctx["existing_relations"]:
        published_rels = detail.get("relations") or []
        ctx["existing_relations"] = [
            _normalize_relation(rel)
            for rel in (published_rels if isinstance(published_rels, list) else [])
        ]

    return ctx


def validate_node_context(ctx: dict[str, Any]) -> None:
    """Step 3: reject context that would produce meaningless queries."""
    if not ctx.get("name"):
        raise InvalidNodeContextError(
            f"node name is required: node_id={ctx.get('node_id', '?')}"
        )

    # Defensive check: ensure we never query with a numeric-only fallback
    name = str(ctx["name"])
    if name.startswith("node_") and name.split("_", 1)[-1].isdigit():
        raise InvalidNodeContextError(
            f"node name is a numeric fallback: {name}"
        )


class InvalidNodeContextError(ValueError):
    """Raised when node context is insufficient for semantic completion."""
    pass


class InvalidOpportunityError(ValueError):
    """Raised when an opportunity has no actionable gaps."""
    pass


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

def run_semantic_completion_for_node(
    domain_id: str,
    node_id: str,
    opportunity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the semantic completion pipeline for one opportunity.

    1. Build node context from opportunity snapshot
    2. Enrich with detail (supplement only)
    3. Validate context
    4. Build SemanticCompleteRequest
    5. Call complete_semantic_service
    """
    opp = opportunity or {}
    gap_fields = opp.get("_gap_fields") or []
    relation_intents = opp.get("_relation_intents") or []

    # ── Steps 1-3: build, enrich, validate ──
    ctx = build_node_context_from_opportunity(opp)

    if ctx["name"] or ctx["node_type"]:
        ctx = enrich_node_context_from_detail(
            ctx,
            domain_id=domain_id,
            node_id=node_id,
        )

    try:
        validate_node_context(ctx)
    except InvalidNodeContextError as exc:
        logger.warning("Skipping node %s: %s", node_id, exc)
        return {
            "candidate_ids": [],
            "evidence_count": 0,
            "trace_id": "",
            "claim_count": 0,
            "status": "invalid_context",
            "error_code": "NODE_NAME_MISSING",
            "node_id": node_id,
        }

    # ── Step 4: build request ──
    # Self-growth: gap_fields as search hints, not extraction constraints.
    # When empty, use growth discovery keywords tuned for the node type.
    if not gap_fields:
        gap_fields = []
    if not relation_intents:
        relation_intents = []

    growth_message = (
        f"请从证据中提取关于「{ctx['name']}」的所有可验证事实，"
        f"不限于指定字段。包括但不限于：名称、简介、历史、位置、特色、"
        f"关联人物、关联地点、年代、类别、重要事件。"
    )

    node_ctx = SemanticNodeContext(
        source_node_id=node_id,
        name=ctx["name"],
        node_type=ctx["node_type"],
        parent_name=ctx["parent_name"],
        scenic_name=ctx["scenic_name"],
    )

    request = SemanticCompleteRequest(
        scenic_id=str(domain_id),
        node=node_ctx,
        message=growth_message,
        target_fields=gap_fields,
        relation_intents=relation_intents,
        subgraph_depth=1,
        existing_properties=ctx["existing_properties"],
        existing_relations=ctx["existing_relations"],
        graph_context=ctx["graph_context"],
        evidence=[],
        max_web_results=5,
        use_web_search=False,
        use_web_extractor=False,
        metadata={
            "completion_mode": "growth",
            "growth_opportunity_id": opp.get("opportunity_id", ""),
            "growth_run_id": opp.get("growth_run_id", ""),
            "source_scope": ["domain_kb", "provided_evidence"],
        },
    )

    # ── Step 5: run pipeline ──
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(complete_semantic_service(request))
                )
                result = future.result(timeout=300)
        else:
            result = loop.run_until_complete(complete_semantic_service(request))
    except RuntimeError:
        result = asyncio.run(complete_semantic_service(request))

    claims = getattr(result, "candidate_claims", []) or []
    candidate_ids = [
        int(getattr(c, "candidate_id", 0))
        for c in claims
        if getattr(c, "candidate_id", None)
    ]

    return {
        "candidate_ids": candidate_ids,
        "evidence_count": len(getattr(result, "evidence_chunks", []) or []),
        "trace_id": getattr(result, "trace_id", ""),
        "claim_count": len(claims),
    }
