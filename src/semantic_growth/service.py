"""Phase 2 service — TypedDict state, max_concurrency config."""

from __future__ import annotations

import logging

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from src.rag.dependencies import build_ai_db_url

from .graph import build_graph
from .repository import (
    create_run,
    get_run,
    get_run_detail,
    list_runs,
    set_run_paused,
    set_run_cancelled,
    set_run_status,
)

logger = logging.getLogger(__name__)

_CHECKPOINT_SETUP_LOCK = threading.Lock()
_CHECKPOINT_SETUP_DONE = False


@contextmanager
def growth_graph() -> Iterator[Any]:
    global _CHECKPOINT_SETUP_DONE
    with PostgresSaver.from_conn_string(build_ai_db_url()) as checkpointer:
        if not _CHECKPOINT_SETUP_DONE:
            with _CHECKPOINT_SETUP_LOCK:
                if not _CHECKPOINT_SETUP_DONE:
                    checkpointer.setup()
                    _CHECKPOINT_SETUP_DONE = True
        yield build_graph(checkpointer)


def _build_initial_state(run: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Build the initial GrowthState for graph invocation."""
    # budget limits candidate/opportunity fan-out; evidence consumption has
    # its own bounded run budget so a default budget of 10 does not silently
    # skip the rest of the uploaded evidence.
    requested_evidence = payload.get("max_evidence_per_run")
    if requested_evidence is None:
        requested_evidence = 500
    max_evidence_per_run = max(1, min(int(requested_evidence), 2000))
    batch_size = max(
        1,
        min(
            int(payload.get("batch_size") or min(max_evidence_per_run, 16)),
            min(max_evidence_per_run, 32),
        ),
    )
    return {
        "growth_run_id": run["growth_run_id"],
        "thread_id": run["thread_id"],
        "domain_id": run["domain_id"],
        "scenic_id": run.get("scenic_id") or "",
        "seed_node_ids": run.get("seed_node_ids") or [],
        "iteration": 0,
        "max_iterations": int(run.get("max_iterations") or 5),
        "max_opportunities_per_iteration": max(
            1,
            min(
                int(
                    payload.get("budget")
                    or payload.get("max_opportunities_per_iteration")
                    or 10
                ),
                200,
            ),
        ),
        "batch_size": batch_size,
        "max_evidence_per_run": max_evidence_per_run,
        "max_image_evidence_per_run": max(0, min(int(payload.get("max_image_evidence_per_run") or 32), 200)),
        "image_evidence_processed_count": 0,
        "extraction_concurrency": max(1, min(int(payload.get("extraction_concurrency") or 4), 8)),
        "growth_track": "OPEN_DISCOVERY",
        "batch_iteration": 0,
        "evidence_processed_count": 0,
        "last_batch_count": 0,
        "review_status": "not_required",
    }


def start_growth_run(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a growth run and start the LangGraph."""
    run = create_run(payload)
    state = _build_initial_state(run, payload)
    config = {
        "configurable": {"thread_id": run["thread_id"]},
        "max_concurrency": int(payload.get("max_concurrency") or 3),
    }

    with growth_graph() as graph:
        result = graph.invoke(state, config=config)
        snapshot = graph.get_state(config)

    return {
        "growth_run_id": run["growth_run_id"],
        "thread_id": run["thread_id"],
        "status": (get_run(run["growth_run_id"]) or {}).get("status", "WAITING_REVIEW"),
        "interrupts": [item.value for item in result.get("__interrupt__", [])],
        "next": list(snapshot.next) if snapshot else [],
    }


def start_growth_run_background(payload: dict[str, Any]) -> None:
    """Run the graph outside the HTTP request so the A端 can track progress."""
    growth_run_id = str(payload.get("growth_run_id") or "")
    try:
        start_growth_run(payload)
    except Exception as exc:
        logger.exception("background growth run failed: %s", growth_run_id)
        if growth_run_id:
            set_run_status(
                growth_run_id,
                "FAILED",
                "background_start_failed",
                status_reason_code="BACKGROUND_START_FAILED",
                warning_codes=["START_FAILURE"],
            )


def resume_growth_run(growth_run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Resume a growth run after human review."""
    run = get_run(growth_run_id)
    if not run:
        raise LookupError("growth run not found")
    thread_id = str(payload.get("thread_id") or "")
    if thread_id != run["thread_id"]:
        raise ValueError("thread_id does not match growth run")
    if str(run.get("status") or "").upper() != "WAITING_REVIEW":
        if str(run.get("status") or "").upper() == "PAUSED":
            raise ValueError("growth run is PAUSED; use /resume-paused")
        raise ValueError("growth run is not waiting for review")

    config = {"configurable": {"thread_id": thread_id}}

    with growth_graph() as graph:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            raise ValueError("growth run is not waiting for review")
        result = graph.invoke(Command(resume=payload), config=config)
        final_snapshot = graph.get_state(config)

    return {
        "growth_run_id": growth_run_id,
        "thread_id": thread_id,
        "status": (get_run(growth_run_id) or {}).get("status"),
        "review_status": result.get("review_status"),
        "next": list(final_snapshot.next) if final_snapshot else [],
    }


def resume_paused_growth_run(growth_run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    run = get_run(growth_run_id)
    if not run:
        raise LookupError("growth run not found")
    if str(run.get("status") or "").upper() != "PAUSED":
        raise ValueError("growth run is not paused")
    thread_id = str(payload.get("thread_id") or "")
    if thread_id != run["thread_id"]:
        raise ValueError("thread_id does not match growth run")
    config = {"configurable": {"thread_id": thread_id}}
    with growth_graph() as graph:
        snapshot = graph.get_state(config)
    if not snapshot.next:
        raise ValueError("paused growth run has no resumable checkpoint")
    set_run_status(growth_run_id, "WAITING_REVIEW", "manual_resume_paused", status_reason_code="WAITING_REVIEW")
    return {
        "growth_run_id": growth_run_id,
        "thread_id": thread_id,
        "status": "WAITING_REVIEW",
        "next": list(snapshot.next),
    }


def growth_run_state(growth_run_id: str) -> dict[str, Any]:
    """Get full state detail for a growth run."""
    detail = get_run_detail(growth_run_id)
    if not detail:
        raise LookupError("growth run not found")
    config = {"configurable": {"thread_id": detail["run"]["thread_id"]}}
    with growth_graph() as graph:
        snapshot = graph.get_state(config)
    detail["graph"] = {
        "next": list(snapshot.next) if snapshot else [],
        "checkpoint_id": (snapshot.config.get("configurable") or {}).get("checkpoint_id"),
        "values": snapshot.values if snapshot else {},
    }
    return detail


def list_growth_runs_svc(domain_id=None, status=None, limit=50):
    """Service wrapper: list growth runs."""
    return list_runs(domain_id, status, limit)


def pause_growth_run_svc(growth_run_id):
    """Service wrapper: pause a growth run."""
    return set_run_paused(growth_run_id)


def cancel_growth_run_svc(growth_run_id):
    """Service wrapper: cancel a growth run."""
    return set_run_cancelled(growth_run_id)
