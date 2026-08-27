"""Phase 2 GrowthState — TypedDict with Annotated reducers for Send pattern."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from typing_extensions import TypedDict


class GrowthState(TypedDict, total=False):
    # ── Run identity (set once at start, never returned by workers) ──
    growth_run_id: str
    thread_id: str
    domain_id: str
    scenic_id: str
    domain_schema: dict[str, Any]

    # ── Control ──
    iteration: int
    max_iterations: int
    max_opportunities_per_iteration: int
    batch_size: int
    max_evidence_per_run: int
    max_image_evidence_per_run: int
    image_evidence_processed_count: int
    extraction_concurrency: int
    growth_track: str
    batch_iteration: int
    evidence_processed_count: int
    last_batch_count: int

    # ── Discovery (set by load_scope / discover_gaps) ──
    published_nodes: list[dict[str, Any]]
    opportunities: list[dict[str, Any]]
    active_opportunity: dict[str, Any] | None
    evidence_batch: list[dict[str, Any]]
    mention_batch: list[dict[str, Any]]
    alignment_batch: list[dict[str, Any]]
    evidence_consumption_ids: list[int]
    candidate_results: list[dict[str, Any]]
    discovery_summary: dict[str, Any]
    normalization_results: list[dict[str, Any]]
    conflict_results: list[dict[str, Any]]
    dependency_results: list[dict[str, Any]]
    # Current-batch candidates are replaced on each loop; candidate_ids below
    # remains the all-batches accumulated result via its reducer.
    batch_candidate_ids: list[str]

    # ── Stop control ──
    stop_reason: str | None

    # ── Reducer fields (accumulated across parallel workers) ──
    candidate_ids: Annotated[list[str], operator.add]
    processed_opportunity_ids: Annotated[list[str], operator.add]
    no_evidence_opportunity_ids: Annotated[list[str], operator.add]
    failed_opportunity_ids: Annotated[list[str], operator.add]
    opportunity_results: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, Any]], operator.add]
    batch_summaries: Annotated[list[dict[str, Any]], operator.add]

    # ── Aggregated final output (non-reducer, set once by aggregate) ──
    final_candidate_ids: list[str]
    result_summary: dict[str, Any]
    aggregation_done: bool

    # ── Review ──
    review_status: Literal[
        "not_required", "waiting", "accepted", "modified", "rejected"
    ]
    review_payload: dict[str, Any] | None
