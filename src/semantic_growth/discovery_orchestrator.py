from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope

from .candidate_aggregation_service import resolve_canonicalize_and_aggregate
from .evidence import finalize_open_discovery_batch
from .kg_delta_service import classify_kg_deltas, persist_kg_deltas
from .open_discovery_service import (
    discover_evidence_units,
    materialize_evidence_units,
    persist_raw_discovery,
)
from .repository import (
    create_candidate_link,
    create_opportunity,
    finish_opportunity,
)


def _unit_outcomes(
    *,
    units: list[dict[str, Any]],
    discoveries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors = {
        int(item["unit"]["id"]): str(item.get("error") or "")
        for item in discoveries
        if item.get("error")
    }
    with ai_session_scope() as db:
        candidate_rows = db.execute(
            text(
                """
                select evidence_unit_id, array_agg(distinct candidate_id) as candidate_ids
                from semantic_growth_candidate_evidence_bindings
                where evidence_unit_id=any(:ids)
                group by evidence_unit_id
                """
            ),
            {"ids": [int(item["id"]) for item in units] or [0]},
        ).mappings().all()
        fact_rows = db.execute(
            text(
                """
                select evidence_unit_id, count(*) as binding_count
                from semantic_growth_fact_evidence_bindings
                where evidence_unit_id=any(:ids)
                group by evidence_unit_id
                """
            ),
            {"ids": [int(item["id"]) for item in units] or [0]},
        ).mappings().all()
    candidates_by_unit = {
        int(row["evidence_unit_id"]): [int(value) for value in (row["candidate_ids"] or [])]
        for row in candidate_rows
    }
    facts_by_unit = {int(row["evidence_unit_id"]): int(row["binding_count"] or 0) for row in fact_rows}
    outcomes: list[dict[str, Any]] = []
    for unit in units:
        unit_id = int(unit["id"])
        candidate_ids = candidates_by_unit.get(unit_id, [])
        fact_count = facts_by_unit.get(unit_id, 0)
        outcomes.append(
            {
                "consumption_id": unit.get("consumption_id"),
                "evidence_unit_id": unit_id,
                "candidate_ids": candidate_ids,
                "fact_binding_count": fact_count,
                "result": "CANDIDATE" if candidate_ids else ("EVIDENCE_ONLY" if fact_count else "NO_CHANGE"),
                "error": errors.get(unit_id) or None,
            }
        )
    return outcomes


def _link_candidates(
    *,
    growth_run_id: str,
    candidate_ids: list[int],
    iteration: int,
) -> None:
    if not candidate_ids:
        return
    with ai_session_scope() as db:
        rows = [
            dict(row)
            for row in db.execute(
                text(
                    """
                    select id, source_node_id, claim_type, predicate, update_operation
                    from semantic_claim_candidates where id=any(:ids)
                    """
                ),
                {"ids": candidate_ids},
            ).mappings().all()
        ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("source_node_id") or "__open_discovery__")].append(row)
    for node_id, items in grouped.items():
        opportunity = create_opportunity(
            growth_run_id=growth_run_id,
            node_id=node_id,
            opportunity_type="open_discovery",
            reason="证据优先开放发现产生知识差量",
            metadata={
                "candidate_count": len(items),
                "operations": sorted({str(item.get("update_operation") or "ADD") for item in items}),
                "discovery_track": "OPEN_DISCOVERY",
            },
        )
        for item in items:
            create_candidate_link(
                growth_run_id=growth_run_id,
                opportunity_id=opportunity["opportunity_id"],
                candidate_id=int(item["id"]),
                iteration=int(iteration),
            )
        finish_opportunity(opportunity["opportunity_id"], status="COMPLETED")


def run_open_discovery_batch(
    *,
    growth_run_id: str,
    source_scenic_id: str,
    batch: list[dict[str, Any]],
    iteration: int,
    max_concurrency: int = 4,
    domain_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    units = materialize_evidence_units(
        growth_run_id=growth_run_id,
        source_scenic_id=source_scenic_id,
        batch=batch,
    )
    discoveries = discover_evidence_units(units, max_concurrency=max_concurrency)
    raw = persist_raw_discovery(growth_run_id=growth_run_id, discoveries=discoveries)
    aggregation = resolve_canonicalize_and_aggregate(
        growth_run_id=growth_run_id,
        source_scenic_id=source_scenic_id,
        raw_claims=raw["claims"],
    )
    classified = classify_kg_deltas(
        source_scenic_id=source_scenic_id,
        aggregated_claims=aggregation["aggregated_claims"],
        domain_schema=domain_schema,
    )
    delta = persist_kg_deltas(
        growth_run_id=growth_run_id,
        source_scenic_id=source_scenic_id,
        classified_claims=classified,
    )
    candidate_ids = [int(value) for value in delta["candidate_ids"]]
    _link_candidates(
        growth_run_id=growth_run_id,
        candidate_ids=candidate_ids,
        iteration=iteration,
    )
    outcomes = _unit_outcomes(units=units, discoveries=discoveries)
    cursors = finalize_open_discovery_batch(
        batch=batch,
        results=outcomes,
        worker_id=f"growth:{growth_run_id}",
    )
    return {
        "candidate_ids": candidate_ids,
        "unit_results": outcomes,
        "cursor_results": cursors,
        "evidence_unit_count": len(units),
        "raw_entity_count": int(raw["entity_count"]),
        "raw_claim_count": int(raw["claim_count"]),
        "canonical_claim_count": int(aggregation["resolved_claim_count"]),
        "aggregated_count": int(aggregation["aggregated_count"]),
        "exists_count": int(delta["exists_count"]),
        "new_candidate_count": len(candidate_ids),
        "new_entity_count": int(delta["operation_counts"].get("MINT_ADD") or 0),
        "conflict_count": int(delta["operation_counts"].get("CONFLICT") or 0),
        "low_evidence_count": int(delta.get("low_evidence_count") or 0),
        "average_trust_score": float(delta.get("average_trust_score") or 0.0),
        "trust_score_sum": float(delta.get("trust_score_sum") or 0.0),
        "trust_scored_count": int(delta.get("trust_scored_count") or 0),
        "trust_risk_counts": delta.get("trust_risk_counts") or {},
        "semantic_match_count": int(aggregation.get("semantic_match_count") or 0),
        "ambiguous_resolution_count": int(aggregation.get("ambiguous_resolution_count") or 0),
        "exact_match_count": int(aggregation.get("exact_match_count") or 0),
        "alias_match_count": int(aggregation.get("alias_match_count") or 0),
        "operation_counts": delta["operation_counts"],
        "add_count": int(delta["operation_counts"].get("ADD") or 0),
        "mint_add_count": int(delta["operation_counts"].get("MINT_ADD") or 0),
        "update_count": int(delta["operation_counts"].get("UPDATE") or 0),
        "deprecate_count": int(delta["operation_counts"].get("DEPRECATE") or 0),
        "exists_operation_count": int(delta["operation_counts"].get("EXISTS") or 0),
        "error_count": int(raw["error_count"]),
        "errors": raw["errors"],
    }
