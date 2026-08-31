"""
Phase 2 LangGraph — Send pattern with Annotated reducers.

Flow:
  START → load_scope → discover_gaps
    → Send(process_node × N)  [parallel, max_concurrency=3]
    → aggregate_results
    → conditional: human_review(interrupt) or END

Reducer fields (operator.add) accumulate partial results from workers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from .adapters.semantic_completion_adapter import (
    build_completion_payload_for_node,
    discover_domain_nodes,
    run_semantic_completion_for_node,
)
from .repository import (
    create_candidate_link,
    create_opportunity,
    finish_review,
    record_step,
    set_run_status,
    finish_opportunity,
)
from .evidence import (
    claim_evidence_batch,
    extract_mentions_from_batch,
    finalize_evidence_batch,
    mark_evidence_failed,
    persist_alignment_results,
)
from .discovery_orchestrator import run_open_discovery_batch
from .state import GrowthState
from src.rag.dependencies import ai_session_scope
from .candidate_extraction import extract_growth_candidates
from .normalization import normalize_candidate_batch
from .conflict_quality import sanitize_same_value_conflicts
from .dependencies import persist_candidate_dependencies, refresh_dependency_states
from sqlalchemy import text

logger = logging.getLogger(__name__)


def _restrict_scope_to_seed_nodes(
    nodes: list[dict[str, Any]],
    seed_node_ids: list[Any] | None,
) -> list[dict[str, Any]]:
    """按发布后的受影响节点缩小下一轮工作范围。

    输入：已加载的正式节点列表，以及上一轮发布产生的 seed_node_ids。
    输出：只保留 source/node_id 命中的节点；没有种子时原样返回。
    该函数不修改数据库，也不影响证据游标和旧补全链。
    """
    seeds = {
        str(node_id).strip()
        for node_id in (seed_node_ids or [])
        if node_id is not None and str(node_id).strip()
    }
    if not seeds:
        return nodes
    return [
        node for node in nodes
        if str(node.get("node_id") or node.get("source_node_id") or "").strip() in seeds
    ]


# ═══════════════════════════════════════════════════════════
# Node: load_scope
# ═══════════════════════════════════════════════════════════

def load_scope(state: GrowthState) -> dict[str, Any]:
    """Load all published domain nodes from Neo4j."""
    growth_run_id = state["growth_run_id"]
    domain_id = state["domain_id"]

    record_step(growth_run_id, "load_scope", {"domain_id": domain_id})
    set_run_status(growth_run_id, "RUNNING")

    try:
        nodes = discover_domain_nodes(domain_id)
    except Exception as exc:
        logger.warning("load_scope failed, using database nodes: %s", exc)
        nodes = []

    if not nodes:
        with ai_session_scope() as db:
            rows = db.execute(
                text(
                    """
                    select source_node_id, node_name, node_type,
                           parent_source_node_id, description
                    from semantic_nodes
                    where source_scenic_id = :source_scenic_id
                    order by id
                    limit 2000
                    """
                ),
                {"source_scenic_id": str(domain_id)},
            ).mappings().all()
        nodes = [
            {
                "node_id": str(row["source_node_id"]),
                "name": str(row["node_name"] or row["source_node_id"]),
                "node_type": str(row["node_type"] or ""),
                "parent_node_id": str(row["parent_source_node_id"] or ""),
                "description": str(row["description"] or ""),
            }
            for row in rows
            if row.get("source_node_id")
        ]

    scoped_nodes = _restrict_scope_to_seed_nodes(nodes, state.get("seed_node_ids"))
    record_step(
        growth_run_id,
        "load_scope_done",
        {
            "node_count": len(scoped_nodes),
            "unscoped_node_count": len(nodes),
            "seed_node_count": len(state.get("seed_node_ids") or []),
            "source": "neo4j_or_semantic_nodes",
        },
    )

    return {
        "published_nodes": scoped_nodes,
        "iteration": state.get("iteration", 0) + 1,
        "review_status": "not_required",
    }


# ═══════════════════════════════════════════════════════════
# G1 nodes: evidence batch -> mentions -> alignment
# ═══════════════════════════════════════════════════════════

def load_evidence_batch(state: GrowthState) -> dict[str, Any]:
    growth_run_id = state["growth_run_id"]
    set_run_status(growth_run_id, "RUNNING")
    worker_id = f"growth:{growth_run_id}"
    processed_count = int(state.get("evidence_processed_count") or 0)
    image_processed_count = int(state.get("image_evidence_processed_count") or 0)
    max_image_per_run = int(state.get("max_image_evidence_per_run") or 32)
    max_per_run = int(
        state.get("max_evidence_per_run")
        or state.get("max_opportunities_per_iteration")
        or 10
    )
    batch_size = int(state.get("batch_size") or min(max_per_run, 10))
    remaining = max(0, max_per_run - processed_count)
    batch = [] if remaining <= 0 else claim_evidence_batch(
        growth_run_id=growth_run_id,
        source_scenic_id=state["domain_id"],
        worker_id=worker_id,
        limit=min(batch_size, remaining),
        image_limit=max(0, min(batch_size, max_image_per_run - image_processed_count)),
    )
    batch_count = len(batch)
    image_count = sum(1 for item in batch if str(item.get("asset_type") or "") == "image")
    record_step(
        growth_run_id,
        "load_evidence_batch",
        {
            "batch_count": batch_count,
            "image_count": image_count,
            "batch_iteration": int(state.get("batch_iteration") or 0) + 1,
            "processed_before": processed_count,
            "processed_after": processed_count + batch_count,
            "max_evidence_per_run": max_per_run,
            "consumer_version": "growth-open-v2",
        },
    )
    return {
        "evidence_batch": batch,
        "evidence_consumption_ids": [int(item["consumption_id"]) for item in batch],
        "batch_iteration": int(state.get("batch_iteration") or 0) + 1,
        "evidence_processed_count": processed_count + batch_count,
        "image_evidence_processed_count": image_processed_count + image_count,
        "last_batch_count": batch_count,
        "batch_candidate_ids": [],
    }


def open_discovery_batch(state: GrowthState) -> dict[str, Any]:
    """EvidenceUnit-first discovery; entity alignment happens after extraction."""
    growth_run_id = state["growth_run_id"]
    batch = state.get("evidence_batch") or []
    if not batch:
        record_step(growth_run_id, "open_discovery", {"evidence_unit_count": 0})
        return {
            "candidate_ids": [],
            "batch_candidate_ids": [],
            "candidate_results": [],
            "discovery_summary": {},
        }
    try:
        result = run_open_discovery_batch(
            growth_run_id=growth_run_id,
            source_scenic_id=str(state["domain_id"]),
            batch=batch,
            iteration=int(state.get("batch_iteration") or 1),
            max_concurrency=int(state.get("extraction_concurrency") or 4),
            domain_schema=state.get("domain_schema") or {},
        )
    except Exception as exc:
        logger.exception("open discovery failed for growth run %s", growth_run_id)
        worker_id = f"growth:{growth_run_id}"
        for item in batch:
            try:
                mark_evidence_failed(
                    consumption_id=int(item["consumption_id"]),
                    worker_id=worker_id,
                    error=str(exc),
                )
            except Exception:
                logger.exception("failed to mark open-discovery evidence consumption")
        result = {
            "candidate_ids": [],
            "unit_results": [
                {"consumption_id": item.get("consumption_id"), "error": str(exc), "candidate_ids": []}
                for item in batch
            ],
            "evidence_unit_count": 0,
            "raw_entity_count": 0,
            "raw_claim_count": 0,
            "aggregated_count": 0,
            "exists_count": 0,
            "operation_counts": {},
            "error_count": len(batch),
            "errors": [str(exc)],
        }
    candidate_ids = [str(value) for value in result.get("candidate_ids") or []]
    record_step(
        growth_run_id,
        "open_discovery",
        {
            "evidence_unit_count": int(result.get("evidence_unit_count") or 0),
            "raw_entity_count": int(result.get("raw_entity_count") or 0),
            "raw_claim_count": int(result.get("raw_claim_count") or 0),
            "aggregated_count": int(result.get("aggregated_count") or 0),
            "candidate_count": len(candidate_ids),
            "existing_fact_binding_count": int(result.get("exists_count") or 0),
            "operation_counts": result.get("operation_counts") or {},
            "error_count": int(result.get("error_count") or 0),
            "open_cursor_count": sum(
                1 for item in result.get("cursor_results") or []
                if item.get("cursor_state") != "ADVANCED"
            ),
            "discovery_track": "OPEN_DISCOVERY",
        },
    )
    return {
        "candidate_ids": candidate_ids,
        "batch_candidate_ids": candidate_ids,
        "candidate_results": result.get("unit_results") or [],
        "discovery_summary": result,
    }


def extract_mentions(state: GrowthState) -> dict[str, Any]:
    batch = state.get("evidence_batch") or []
    mentions = extract_mentions_from_batch(batch, state.get("published_nodes") or [])
    record_step(
        state["growth_run_id"],
        "extract_mentions",
        {"batch_count": len(batch), "mention_count": len(mentions)},
    )
    return {"mention_batch": mentions}


def align_nodes(state: GrowthState) -> dict[str, Any]:
    growth_run_id = state["growth_run_id"]
    batch = state.get("evidence_batch") or []
    mentions = state.get("mention_batch") or []
    worker_id = f"growth:{growth_run_id}"
    try:
        alignments = persist_alignment_results(
            growth_run_id=growth_run_id,
            batch=batch,
            mentions=mentions,
            worker_id=worker_id,
        )
    except Exception as exc:
        logger.exception("align_nodes failed for growth run %s", growth_run_id)
        for item in batch:
            try:
                mark_evidence_failed(
                    consumption_id=int(item["consumption_id"]),
                    worker_id=worker_id,
                    error=str(exc),
                )
            except Exception:
                logger.exception("failed to mark evidence consumption")
        set_run_status(
            growth_run_id,
            "FAILED",
            "evidence_alignment_failed",
            status_reason_code="EVIDENCE_ALIGNMENT_FAILED",
            warning_codes=["EVIDENCE_FAILURE"],
        )
        return {"alignment_batch": [], "stop_reason": "evidence_batch_failed"}
    record_step(
        growth_run_id,
        "align_nodes",
        {"aligned_count": len(alignments), "chunk_count": len(batch)},
    )
    return {"alignment_batch": alignments}


def extract_candidates(state: GrowthState) -> dict[str, Any]:
    growth_run_id = state["growth_run_id"]
    batch = state.get("evidence_batch") or []
    mentions = state.get("mention_batch") or []
    if not batch:
        finalize_evidence_batch(
            batch=batch,
            results=[],
            worker_id=f"growth:{growth_run_id}",
        )
        record_step(growth_run_id, "extract_candidates", {"node_count": 0, "candidate_count": 0})
        return {"candidate_ids": [], "batch_candidate_ids": [], "candidate_results": []}

    results = extract_growth_candidates(
        source_scenic_id=state["domain_id"],
        growth_run_id=growth_run_id,
        batch=batch,
        mentions=mentions,
        published_nodes=state.get("published_nodes") or [],
        # Candidate discovery throughput is independent from the review
        # opportunity budget.  Every evidence chunk in this batch may produce
        # an open-discovery group; review limits are applied after extraction.
        max_nodes=min(len(batch), 100),
        allow_open_discovery=True,
    )
    candidate_ids: list[str] = []
    for result in results:
        node_id = str(result.get("node_id") or "")
        opportunity = create_opportunity(
            growth_run_id=growth_run_id,
            node_id=node_id,
            opportunity_type="evidence_candidate_extraction",
            reason="G1证据提及进入G2开放式候选抽取",
            metadata={
                "mention_count": int(result.get("mention_count") or 0),
                "trace_id": result.get("trace_id"),
            },
        )
        ids = [int(item) for item in (result.get("candidate_ids") or [])]
        for candidate_id in ids:
            create_candidate_link(
                growth_run_id=growth_run_id,
                opportunity_id=opportunity["opportunity_id"],
                candidate_id=candidate_id,
                iteration=int(state.get("batch_iteration") or state.get("iteration") or 1),
            )
            candidate_ids.append(str(candidate_id))
        if result.get("error"):
            finish_opportunity(opportunity["opportunity_id"], status="FAILED")
        else:
            finish_opportunity(
                opportunity["opportunity_id"],
                status="COMPLETED" if ids else "NO_CHANGE",
            )
    cursor_results = finalize_evidence_batch(
        batch=batch,
        results=results,
        worker_id=f"growth:{growth_run_id}",
    )
    record_step(
        growth_run_id,
        "extract_candidates",
        {
            "node_count": len(results),
            "candidate_count": len(set(candidate_ids)),
            "duplicate_count": sum(len(item.get("duplicate_candidate_ids") or []) for item in results),
            "error_count": sum(1 for item in results if item.get("error")),
            "open_cursor_count": sum(1 for item in cursor_results if item.get("cursor_state") != "ADVANCED"),
        },
    )
    return {
        "candidate_ids": list(dict.fromkeys(candidate_ids)),
        "batch_candidate_ids": list(dict.fromkeys(candidate_ids)),
        "candidate_results": results,
    }


def normalize_candidates(state: GrowthState) -> dict[str, Any]:
    """G3 mixed normalization: deterministic first, vector recall as evidence only."""
    growth_run_id = state["growth_run_id"]
    candidate_ids = state.get("batch_candidate_ids") or []
    if not candidate_ids:
        record_step(growth_run_id, "normalize_candidates", {"candidate_count": 0})
        return {"normalization_results": []}
    result = normalize_candidate_batch(
        source_scenic_id=str(state["domain_id"]),
        candidate_ids=candidate_ids,
        vector_limit=3,
        max_vector_queries=6,
    )
    record_step(
        growth_run_id,
        "normalize_candidates",
        {
            "candidate_count": int(result.get("candidate_count") or 0),
            "updated_count": int(result.get("updated_count") or 0),
            "vector_recall_count": int(result.get("vector_recall_count") or 0),
            "vector_query_count": int(result.get("vector_query_count") or 0),
            "multimodal_query_count": int(result.get("multimodal_query_count") or 0),
            "multimodal_scored_count": int(result.get("multimodal_scored_count") or 0),
            "error_count": len(result.get("errors") or []),
            "vector_role": "evidence_recall_only",
            "multimodal_role": "evidence_augmentation_review_only",
        },
    )
    return {"normalization_results": [result]}


def validate_conflicts(state: GrowthState) -> dict[str, Any]:
    """G4 removes same-value pseudo-conflicts without changing reviewed facts."""
    growth_run_id = state["growth_run_id"]
    candidate_ids = state.get("batch_candidate_ids") or []
    if not candidate_ids:
        record_step(growth_run_id, "validate_conflicts", {"candidate_count": 0})
        return {"conflict_results": []}
    result = sanitize_same_value_conflicts(
        source_scenic_id=str(state["domain_id"]),
        candidate_ids=candidate_ids,
    )
    record_step(
        growth_run_id,
        "validate_conflicts",
        {
            "candidate_count": int(result.get("candidate_count") or 0),
            "groups_checked": int(result.get("groups_checked") or 0),
            "resolved_group_count": int(result.get("resolved_group_count") or 0),
            "resolved_candidate_count": int(result.get("resolved_candidate_count") or 0),
            "error_count": len(result.get("errors") or []),
        },
    )
    return {"conflict_results": [result]}


def persist_dependencies(state: GrowthState) -> dict[str, Any]:
    """G5 records candidate dependencies and the affected graph scope."""
    growth_run_id = state["growth_run_id"]
    candidate_ids = state.get("batch_candidate_ids") or []
    result = {}
    if candidate_ids:
        result = persist_candidate_dependencies(
            growth_run_id=growth_run_id,
            source_scenic_id=str(state["domain_id"]),
            candidate_ids=candidate_ids,
        )
        refresh_dependency_states(growth_run_id=growth_run_id)
    record_step(
        growth_run_id,
        "persist_dependencies",
        {
            "candidate_count": int(result.get("candidate_count") or 0),
            "dependency_count": int(result.get("dependency_count") or 0),
            "affected_scope_count": len(result.get("affected_scope") or []),
            "error_count": len(result.get("errors") or []),
        },
    )
    batch_results = state.get("candidate_results") or []
    discovery = state.get("discovery_summary") or {}
    normalization = (state.get("normalization_results") or [{}])[0]
    conflict = (state.get("conflict_results") or [{}])[0]
    batch_summary = {
        "evidence_count": len(state.get("evidence_batch") or []),
        "image_count": sum(1 for item in (state.get("evidence_batch") or []) if str(item.get("asset_type") or "") == "image"),
        "mention_count": len(state.get("mention_batch") or []),
        "aligned_count": len(state.get("alignment_batch") or []),
        "evidence_unit_count": int(discovery.get("evidence_unit_count") or 0),
        "raw_entity_count": int(discovery.get("raw_entity_count") or 0),
            "raw_claim_count": int(discovery.get("raw_claim_count") or 0),
            "canonical_claim_count": int(discovery.get("canonical_claim_count") or 0),
            "aggregated_count": int(discovery.get("aggregated_count") or 0),
            "existing_fact_binding_count": int(discovery.get("exists_count") or 0),
            "new_candidate_count": int(discovery.get("new_candidate_count") or len(candidate_ids)),
            "new_entity_count": int(discovery.get("new_entity_count") or 0),
            "conflict_count": int(discovery.get("conflict_count") or 0),
            "low_evidence_count": int(discovery.get("low_evidence_count") or 0),
            "semantic_match_count": int(discovery.get("semantic_match_count") or 0),
            "ambiguous_resolution_count": int(discovery.get("ambiguous_resolution_count") or 0),
            "exact_match_count": int(discovery.get("exact_match_count") or 0),
            "alias_match_count": int(discovery.get("alias_match_count") or 0),
            "trust_score_sum": float(discovery.get("trust_score_sum") or 0.0),
            "trust_scored_count": int(discovery.get("trust_scored_count") or 0),
            "trust_high_count": int((discovery.get("trust_risk_counts") or {}).get("HIGH") or 0),
            "trust_medium_count": int((discovery.get("trust_risk_counts") or {}).get("MEDIUM") or 0),
            "trust_low_count": int((discovery.get("trust_risk_counts") or {}).get("LOW") or 0),
            "operation_add_count": int((discovery.get("operation_counts") or {}).get("ADD") or 0),
            "operation_mint_add_count": int((discovery.get("operation_counts") or {}).get("MINT_ADD") or 0),
            "operation_update_count": int((discovery.get("operation_counts") or {}).get("UPDATE") or 0),
            "operation_deprecate_count": int((discovery.get("operation_counts") or {}).get("DEPRECATE") or 0),
            "operation_exists_count": int((discovery.get("operation_counts") or {}).get("EXISTS") or 0),
        "candidate_count": len(candidate_ids),
        "extraction_error_count": sum(1 for item in batch_results if item.get("error")),
        "normalization_updated_count": int(normalization.get("updated_count") or 0),
        "vector_recall_count": int(normalization.get("vector_recall_count") or 0),
        "vector_query_count": int(normalization.get("vector_query_count") or 0),
        "normalization_error_count": len(normalization.get("errors") or []),
        "same_value_conflict_resolved_count": int(conflict.get("resolved_candidate_count") or 0),
        "conflict_validation_error_count": len(conflict.get("errors") or []),
        "dependency_count": int(result.get("dependency_count") or 0),
        "affected_scope_count": len(result.get("affected_scope") or []),
        "dependency_error_count": len(result.get("errors") or []),
    }
    return {
        "dependency_results": [result] if result else [],
        "batch_summaries": [batch_summary],
    }


def route_after_evidence_batch(state: GrowthState) -> str:
    """Continue consuming evidence until the run budget is exhausted."""
    if state.get("stop_reason"):
        return "aggregate_results"
    if int(state.get("last_batch_count") or 0) <= 0:
        return "aggregate_results"
    processed = int(state.get("evidence_processed_count") or 0)
    maximum = int(state.get("max_evidence_per_run") or 0)
    if maximum and processed >= maximum:
        return "aggregate_results"
    return "load_evidence_batch"


# ═══════════════════════════════════════════════════════════
# Node: discover_gaps
# ═══════════════════════════════════════════════════════════

def discover_gaps(state: GrowthState) -> dict[str, Any]:
    """For each published node, detect template gaps and create opportunities."""
    growth_run_id = state["growth_run_id"]
    domain_id = state["domain_id"]
    nodes = state.get("published_nodes") or []

    # G0 freezes the old template-gap broad scan. Only explicitly node-bound
    # evidence can enter the graph until the G1 mention-extraction pipeline exists.
    with ai_session_scope() as db:
        rows = db.execute(text("""
            select distinct source_node_id from semantic_claim_candidates
            where source_scenic_id = :sid and upper(status) = 'ADOPTED'
            union
            select distinct source_node_id from knowledge_chunks
            where source_scenic_id = :sid and source_type = 'domain_kb' and source_node_id is not null
        """), {"sid": str(state["domain_id"])}).all()
    evidence_nodes = {str(r[0]) for r in rows if r[0] and str(r[0]) != "__domain__"}
    nodes = [n for n in nodes if str(n.get("node_id") or n.get("id") or "") in evidence_nodes]

    # G0 deliberately freezes the legacy completion adapter. G1 will replace
    # this return with load_evidence_batch/extract_mentions/align_nodes.
    # Keeping the run explicit and successful-as-NO_CHANGE prevents the old
    # template-gap path from creating misleading candidates in the meantime.
    if nodes:
        record_step(
            growth_run_id,
            "g1_pending",
            {"evidence_node_count": len(nodes), "status": "legacy_adapter_disabled"},
        )
        set_run_status(
            growth_run_id,
            "NO_CHANGE",
            "g1_not_implemented",
            status_reason_code="G1_PENDING",
        )
        return {"opportunities": [], "stop_reason": "g1_not_implemented"}

    opportunities: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("node_id") or node.get("id") or "")
        if not node_id:
            continue
        try:
            payload = build_completion_payload_for_node(domain_id, node)
            gap_fields = payload.get("target_fields") or []
            relation_intents = payload.get("relation_intents") or []
            has_gaps = bool(gap_fields or relation_intents)

            opp = create_opportunity(
                growth_run_id=growth_run_id,
                node_id=node_id,
                opportunity_type="template_gap" if has_gaps else "initial_scan",
                target_property=", ".join(gap_fields[:5]) if gap_fields else None,
                target_relation=", ".join(relation_intents[:3]) if relation_intents else None,
                reason=(
                    f"模板缺口：{len(gap_fields)}属性, {len(relation_intents)}关系"
                    if has_gaps
                    else "首次语义扫描"
                ),
                metadata={
                    "node_name": node.get("name", ""),
                    "node_type": node.get("node_type", ""),
                    "gap_fields": gap_fields,
                    "relation_intents": relation_intents,
                },
            )
            # ── Runtime fields (NOT persisted) — self-contained node snapshot ──
            opp["_gap_fields"] = gap_fields
            opp["_relation_intents"] = relation_intents
            opp["_node_name"] = node.get("name", "")
            opp["_node_type"] = node.get("node_type", "")
            opp["_parent_name"] = node.get("parent_name", "")
            opp["_scenic_name"] = node.get("scenic_name", "")
            opp["_existing_properties"] = [
                {"key": k, "value": ""}
                for k in (node.get("existing_property_keys") or [])
            ]
            opp["_existing_relations"] = [
                {"relation_type": t, "target_name": ""}
                for t in (node.get("existing_relation_types") or [])
            ]
            opp["_graph_context"] = node.get("graph_context") or {}
            opportunities.append(opp)
        except Exception as exc:
            logger.warning("discover_gaps: node %s failed: %s", node_id, exc)

    record_step(
        growth_run_id, "discover_gaps",
        {"opportunity_count": len(opportunities), "total_nodes": len(nodes)},
    )

    if not opportunities:
        set_run_status(growth_run_id, "NO_CHANGE", "no_evidence_nodes_found", status_reason_code="NO_EVIDENCE")
        return {"opportunities": [], "stop_reason": "no_gaps_found"}

    return {"opportunities": opportunities}


# ═══════════════════════════════════════════════════════════
# Dispatch: Send one worker per opportunity
# ═══════════════════════════════════════════════════════════

def dispatch_opportunities(state: GrowthState) -> list[Send] | str:
    """Route: END if stopped; aggregate if no ops; otherwise Send workers."""
    if state.get("stop_reason"):
        return END

    opportunities = state.get("opportunities") or []
    limit = int(state.get("max_opportunities_per_iteration") or state.get("max_iterations", 5))
    selected = opportunities[:limit]

    if not selected:
        return "aggregate_results"

    return [
        Send(
            "process_node",
            {
                "growth_run_id": state["growth_run_id"],
                "domain_id": state["domain_id"],
                "thread_id": state["thread_id"],
                "active_opportunity": opp,
                "iteration": state.get("iteration", 1),
            },
        )
        for opp in selected
    ]


# ═══════════════════════════════════════════════════════════
# Worker Node: process_node
#   Returns ONLY incrementals — no run-identity fields.
# ═══════════════════════════════════════════════════════════

def process_node(state: GrowthState) -> dict[str, Any]:
    """Run semantic completion for one node opportunity."""
    growth_run_id = state["growth_run_id"]
    domain_id = state["domain_id"]
    opp = state.get("active_opportunity") or {}
    node_id = str(opp.get("node_id") or "")
    opportunity_id = str(opp.get("opportunity_id") or opp.get("id") or "")
    iteration = int(state.get("iteration") or 1)

    if not node_id:
        return {
            "failed_opportunity_ids": [opportunity_id],
            "errors": [{"opportunity_id": opportunity_id, "error": "missing node_id"}],
            "opportunity_results": [{"opportunity_id": opportunity_id, "status": "failed", "candidate_ids": []}],
        }

    step_start = time.time()
    record_step(growth_run_id, f"process:{node_id}", {"node_id": node_id}, opportunity_id)

    try:
        result = run_semantic_completion_for_node(domain_id, node_id, opp)
        candidate_ids = [str(cid) for cid in (result.get("candidate_ids") or [])]
        elapsed = round(time.time() - step_start, 2)

        # Link candidates to this growth run
        for cid in candidate_ids:
            try:
                create_candidate_link(
                    growth_run_id=growth_run_id,
                    opportunity_id=opportunity_id,
                    candidate_id=int(cid),
                    iteration=iteration,
                )
            except Exception as exc:
                logger.warning("link candidate %s: %s", cid, exc)

        record_step(
            growth_run_id, f"done:{node_id}",
            {"node_id": node_id, "candidate_count": len(candidate_ids),
             "evidence_count": result.get("evidence_count", 0), "elapsed_ms": elapsed},
            opportunity_id,
        )

        status = "candidate_generated" if candidate_ids else "no_evidence"

        return {
            "candidate_ids": candidate_ids,
            "processed_opportunity_ids": [opportunity_id],
            "opportunity_results": [{"opportunity_id": opportunity_id, "status": status, "candidate_ids": candidate_ids}],
            "no_evidence_opportunity_ids": [] if candidate_ids else [opportunity_id],
        }

    except Exception as exc:
        logger.error("process_node %s failed: %s", node_id, exc, exc_info=True)
        record_step(
            growth_run_id, f"error:{node_id}",
            {"node_id": node_id, "error": str(exc)}, opportunity_id,
        )
        return {
            "failed_opportunity_ids": [opportunity_id],
            "errors": [{"opportunity_id": opportunity_id, "node_id": node_id, "error": str(exc)}],
            "opportunity_results": [{"opportunity_id": opportunity_id, "status": "failed", "candidate_ids": []}],
        }


# ═══════════════════════════════════════════════════════════
# Node: aggregate_results
# ═══════════════════════════════════════════════════════════

def aggregate_results(state: GrowthState) -> dict[str, Any]:
    """Aggregate G1 evidence alignment or legacy candidate workers."""
    growth_run_id = state["growth_run_id"]

    if "evidence_batch" in state:
        if state.get("stop_reason") == "evidence_batch_failed":
            return {
                "final_candidate_ids": [],
                "result_summary": {"evidence_count": len(state.get("evidence_batch") or [])},
                "aggregation_done": True,
                "review_status": "not_required",
            }
        batch = state.get("evidence_batch") or []
        mentions = state.get("mention_batch") or []
        alignments = state.get("alignment_batch") or []
        candidate_ids = list(dict.fromkeys(state.get("candidate_ids") or []))
        candidate_results = state.get("candidate_results") or []
        extraction_errors = [item for item in candidate_results if item.get("error")]
        normalization_results = state.get("normalization_results") or []
        normalization_summary = normalization_results[0] if normalization_results else {}
        conflict_results = state.get("conflict_results") or []
        conflict_summary = conflict_results[0] if conflict_results else {}
        dependency_results = state.get("dependency_results") or []
        dependency_summary = dependency_results[0] if dependency_results else {}
        batch_summaries = state.get("batch_summaries") or []
        if batch_summaries:
            totals = {
                key: sum(int(item.get(key) or 0) for item in batch_summaries)
                for key in (
                    "evidence_count", "image_count", "mention_count", "aligned_count", "candidate_count",
                    "evidence_unit_count", "raw_entity_count", "raw_claim_count", "aggregated_count",
                    "canonical_claim_count", "existing_fact_binding_count", "new_candidate_count",
                    "new_entity_count", "conflict_count", "low_evidence_count",
                    "operation_add_count", "operation_mint_add_count", "operation_update_count",
                    "operation_deprecate_count", "operation_exists_count",
                    "semantic_match_count", "ambiguous_resolution_count", "exact_match_count", "alias_match_count",
                    "trust_scored_count", "trust_high_count", "trust_medium_count", "trust_low_count",
                    "extraction_error_count", "normalization_updated_count", "vector_recall_count",
                    "vector_query_count", "normalization_error_count", "same_value_conflict_resolved_count",
                    "conflict_validation_error_count", "dependency_count", "affected_scope_count",
                    "dependency_error_count",
                )
            }
            totals["trust_score_sum"] = sum(float(item.get("trust_score_sum") or 0.0) for item in batch_summaries)
        else:
            totals = {
                "evidence_count": len(batch),
                "image_count": sum(1 for item in batch if str(item.get("asset_type") or "") == "image"),
                "mention_count": len(mentions),
                "aligned_count": len(alignments), "candidate_count": len(candidate_ids),
                "extraction_error_count": len(extraction_errors),
                "normalization_updated_count": int(normalization_summary.get("updated_count") or 0),
                "vector_recall_count": int(normalization_summary.get("vector_recall_count") or 0),
                "vector_query_count": int(normalization_summary.get("vector_query_count") or 0),
                "normalization_error_count": len(normalization_summary.get("errors") or []),
                "same_value_conflict_resolved_count": int(conflict_summary.get("resolved_candidate_count") or 0),
                "conflict_validation_error_count": len(conflict_summary.get("errors") or []),
                "dependency_count": int(dependency_summary.get("dependency_count") or 0),
                "affected_scope_count": len(dependency_summary.get("affected_scope") or []),
                "dependency_error_count": len(dependency_summary.get("errors") or []),
                "evidence_unit_count": int((state.get("discovery_summary") or {}).get("evidence_unit_count") or 0),
                "raw_entity_count": int((state.get("discovery_summary") or {}).get("raw_entity_count") or 0),
                "raw_claim_count": int((state.get("discovery_summary") or {}).get("raw_claim_count") or 0),
                "canonical_claim_count": int((state.get("discovery_summary") or {}).get("canonical_claim_count") or 0),
                "aggregated_count": int((state.get("discovery_summary") or {}).get("aggregated_count") or 0),
                "existing_fact_binding_count": int((state.get("discovery_summary") or {}).get("exists_count") or 0),
                "new_candidate_count": len(candidate_ids),
                "new_entity_count": int((state.get("discovery_summary") or {}).get("new_entity_count") or 0),
                "conflict_count": int((state.get("discovery_summary") or {}).get("conflict_count") or 0),
                "low_evidence_count": int((state.get("discovery_summary") or {}).get("low_evidence_count") or 0),
                "semantic_match_count": int((state.get("discovery_summary") or {}).get("semantic_match_count") or 0),
                "ambiguous_resolution_count": int((state.get("discovery_summary") or {}).get("ambiguous_resolution_count") or 0),
                "exact_match_count": int((state.get("discovery_summary") or {}).get("exact_match_count") or 0),
                "alias_match_count": int((state.get("discovery_summary") or {}).get("alias_match_count") or 0),
                "trust_score_sum": float((state.get("discovery_summary") or {}).get("trust_score_sum") or 0.0),
                "trust_scored_count": int((state.get("discovery_summary") or {}).get("trust_scored_count") or 0),
                "trust_high_count": int(((state.get("discovery_summary") or {}).get("trust_risk_counts") or {}).get("HIGH") or 0),
                "trust_medium_count": int(((state.get("discovery_summary") or {}).get("trust_risk_counts") or {}).get("MEDIUM") or 0),
                "trust_low_count": int(((state.get("discovery_summary") or {}).get("trust_risk_counts") or {}).get("LOW") or 0),
                "operation_add_count": int(((state.get("discovery_summary") or {}).get("operation_counts") or {}).get("ADD") or 0),
                "operation_mint_add_count": int(((state.get("discovery_summary") or {}).get("operation_counts") or {}).get("MINT_ADD") or 0),
                "operation_update_count": int(((state.get("discovery_summary") or {}).get("operation_counts") or {}).get("UPDATE") or 0),
                "operation_deprecate_count": int(((state.get("discovery_summary") or {}).get("operation_counts") or {}).get("DEPRECATE") or 0),
                "operation_exists_count": int(((state.get("discovery_summary") or {}).get("operation_counts") or {}).get("EXISTS") or 0),
            }
        raw_claim_count = int(totals.get("raw_claim_count") or 0)
        canonical_claim_count = int(totals.get("canonical_claim_count") or 0)
        aggregated_count = int(totals.get("aggregated_count") or 0)
        totals["candidate_count"] = len(candidate_ids)
        totals["persisted_candidate_count"] = len(candidate_ids)
        totals["new_candidate_count"] = len(candidate_ids)
        totals["canonicalization_ratio"] = round(
            canonical_claim_count / raw_claim_count, 4
        ) if raw_claim_count else 0.0
        totals["aggregation_ratio"] = round(
            1 - (aggregated_count / canonical_claim_count), 4
        ) if canonical_claim_count else 0.0
        totals["existing_enrichment_ratio"] = round(
            int(totals.get("existing_fact_binding_count") or 0) / canonical_claim_count, 4
        ) if canonical_claim_count else 0.0
        totals["novelty_ratio"] = round(
            (int(totals.get("operation_add_count") or 0) + int(totals.get("operation_mint_add_count") or 0))
            / canonical_claim_count,
            4,
        ) if canonical_claim_count else 0.0
        totals["evidence_consumed"] = int(totals.get("evidence_unit_count") or 0)
        totals["existing_binding_count"] = int(totals.get("existing_fact_binding_count") or 0)
        totals["average_trust_score"] = round(
            float(totals.get("trust_score_sum") or 0.0) / int(totals.get("trust_scored_count") or 1), 4
        ) if int(totals.get("trust_scored_count") or 0) else 0.0
        extraction_error_count = int(totals["extraction_error_count"])
        if candidate_ids:
            reason = "g2_candidates_ready"
            set_run_status(
                growth_run_id,
                "WAITING_REVIEW",
                reason,
                status_reason_code="WAITING_REVIEW_WITH_WARNINGS" if extraction_error_count else "WAITING_REVIEW",
                failed_opportunity_count=extraction_error_count,
                warning_codes=["CANDIDATE_EXTRACTION_FAILURE"] if extraction_error_count else [],
            )
            review_status = "waiting"
            stop_reason = None
        else:
            reason = "g1_evidence_aligned" if totals["evidence_count"] else "no_new_evidence"
            status_reason = "G1_EVIDENCE_ALIGNED" if totals["evidence_count"] else "NO_NEW_EVIDENCE"
            set_run_status(
                growth_run_id,
                "NO_CHANGE",
                reason,
                status_reason_code=status_reason,
                failed_opportunity_count=extraction_error_count,
                warning_codes=["CANDIDATE_EXTRACTION_FAILURE"] if extraction_error_count else [],
            )
            review_status = "not_required"
            stop_reason = reason
        record_step(
            growth_run_id,
            "g2_aggregate",
            {
                **totals,
                "batch_count": len(batch_summaries),
                "evidence_processed_count": int(state.get("evidence_processed_count") or totals["evidence_count"]),
                "max_evidence_per_run": int(state.get("max_evidence_per_run") or 0),
            },
        )
        return {
            "final_candidate_ids": candidate_ids,
            "result_summary": {
                **totals,
                "batch_count": len(batch_summaries),
                "evidence_processed_count": int(state.get("evidence_processed_count") or totals["evidence_count"]),
                "max_evidence_per_run": int(state.get("max_evidence_per_run") or 0),
            },
            "aggregation_done": True,
            "review_status": review_status,
            "stop_reason": stop_reason,
        }

    # Dedup candidate_ids (reducer already appended, just ensure uniqueness)
    all_candidate_ids = state.get("candidate_ids") or []
    unique_ids = list(dict.fromkeys(all_candidate_ids))

    # Sort results
    results = sorted(
        state.get("opportunity_results") or [],
        key=lambda item: str(item.get("opportunity_id", "")),
    )

    summary = {
        "total_opportunities": len(state.get("opportunities") or []),
        "processed": len(state.get("processed_opportunity_ids") or []),
        "candidate_generated": sum(1 for r in results if r.get("status") == "candidate_generated"),
        "no_evidence": len(state.get("no_evidence_opportunity_ids") or []),
        "failed": len(state.get("failed_opportunity_ids") or []),
    }

    record_step(growth_run_id, "aggregate_results", summary)

    failed_count = len(set(state.get("failed_opportunity_ids") or []))
    if unique_ids:
        set_run_status(
            growth_run_id,
            "WAITING_REVIEW",
            "partial_worker_failure" if failed_count else None,
            status_reason_code="WAITING_REVIEW_WITH_WARNINGS" if failed_count else "WAITING_REVIEW",
            failed_opportunity_count=failed_count,
            warning_codes=["WORKER_FAILURE"] if failed_count else [],
        )
        return {
            "final_candidate_ids": unique_ids,
            "result_summary": summary,
            "aggregation_done": True,
            "review_status": "waiting",
        }
    elif failed_count and not state.get("processed_opportunity_ids"):
        set_run_status(
            growth_run_id,
            "FAILED",
            "all_workers_failed",
            status_reason_code="ALL_WORKERS_FAILED",
            failed_opportunity_count=failed_count,
            warning_codes=["WORKER_FAILURE"],
        )
        return {
            "final_candidate_ids": [],
            "result_summary": summary,
            "aggregation_done": True,
            "review_status": "not_required",
            "stop_reason": "all_workers_failed",
        }
    else:
        set_run_status(
            growth_run_id,
            "NO_CHANGE",
            "no_candidates_generated",
            status_reason_code="NO_CHANGE",
            failed_opportunity_count=failed_count,
            warning_codes=["WORKER_FAILURE"] if failed_count else [],
        )
        return {
            "final_candidate_ids": [],
            "result_summary": summary,
            "aggregation_done": True,
            "review_status": "not_required",
            "stop_reason": "no_candidates_generated",
        }


# ═══════════════════════════════════════════════════════════
# Node: human_review (interrupt)
# ═══════════════════════════════════════════════════════════

def human_review(state: GrowthState) -> dict[str, Any]:
    """LangGraph interrupt — pause for human review of candidates."""
    growth_run_id = state["growth_run_id"]
    candidate_ids = state.get("final_candidate_ids") or state.get("candidate_ids") or []

    decision = interrupt({
        "growth_run_id": growth_run_id,
        "thread_id": state.get("thread_id", ""),
        "candidate_ids": candidate_ids,
        "review_status": "waiting",
    })

    action = str((decision or {}).get("action") or "").lower()
    if action not in {"accept", "modify", "reject", "round_complete"}:
        raise ValueError(f"action must be accept/modify/reject/round_complete, got: {action}")
    if action == "round_complete":
        set_run_status(growth_run_id, "COMPLETED", "review_round_complete")
    else:
        finish_review(growth_run_id, action, candidate_ids, decision)
    record_step(growth_run_id, "human_review", {"action": action})

    return {
        "review_status": {"accept": "accepted", "modify": "modified", "reject": "rejected", "round_complete": "accepted"}[action],
        "review_payload": decision,
        "stop_reason": f"review_{action}",
    }


# ═══════════════════════════════════════════════════════════
# Build graph
# ═══════════════════════════════════════════════════════════

def build_graph(checkpointer: Any) -> Any:
    """Build the evidence-driven Growth graph."""
    builder = StateGraph(GrowthState)

    builder.add_node("load_scope", load_scope)
    builder.add_node("load_evidence_batch", load_evidence_batch)
    builder.add_node("open_discovery_batch", open_discovery_batch)
    builder.add_node("normalize_candidates", normalize_candidates)
    builder.add_node("validate_conflicts", validate_conflicts)
    builder.add_node("persist_dependencies", persist_dependencies)
    builder.add_node("aggregate_results", aggregate_results)
    builder.add_node("human_review", human_review)

    # G1 evidence-driven path. Legacy discover_gaps/process_node remain
    # isolated compatibility code and are not reachable from the graph.
    builder.add_edge(START, "load_evidence_batch")
    builder.add_edge("load_evidence_batch", "open_discovery_batch")
    builder.add_edge("open_discovery_batch", "normalize_candidates")
    builder.add_edge("normalize_candidates", "validate_conflicts")
    builder.add_edge("validate_conflicts", "persist_dependencies")
    builder.add_conditional_edges(
        "persist_dependencies",
        route_after_evidence_batch,
        path_map={
            "load_evidence_batch": "load_evidence_batch",
            "aggregate_results": "aggregate_results",
        },
    )

    builder.add_conditional_edges(
        "aggregate_results",
        lambda s: "human_review" if s.get("review_status") == "waiting" else END,
        path_map={"human_review": "human_review", "__end__": END},
    )
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=checkpointer)
