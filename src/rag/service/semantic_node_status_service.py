"""Node-level semantic completion status aggregation."""

from __future__ import annotations

from typing import Any

from src.rag.service.completion_job_service import list_semantic_completion_jobs
from src.rag.service.gap_status_service import list_semantic_gap_status
from src.rag.service.semantic_candidate_store import list_semantic_candidate_groups, list_semantic_candidates

RUNNING_JOB_STATUSES = {"PENDING", "RUNNING"}
REVIEW_STATUSES = {"PENDING", "CONFLICT", "LOW_EVIDENCE"}
CONFLICT_CLASSES = {"conflicting", "scope_mismatch", "entity_ambiguity", "value_conflict", "temporal_conflict", "exclusive_relation_conflict"}


def node_result_scope(source_scenic_id: str, source_node_id: str) -> dict[str, str]:
    return {"source_scenic_id": str(source_scenic_id), "source_node_id": str(source_node_id)}


def list_node_jobs(*, source_scenic_id: str, source_node_id: str, limit: int = 50) -> dict[str, Any]:
    return list_semantic_completion_jobs(**node_result_scope(source_scenic_id, source_node_id), limit=limit)


def list_node_candidates(*, source_scenic_id: str, source_node_id: str, status: str | None = None, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    return list_semantic_candidates(**node_result_scope(source_scenic_id, source_node_id), status=status, limit=limit, offset=offset)


def list_node_conflicts(*, source_scenic_id: str, source_node_id: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    candidates = list_semantic_candidates(**node_result_scope(source_scenic_id, source_node_id), limit=limit, offset=offset)
    conflict_items = []
    for item in candidates.get("items", []):
        if str(item.get("status") or "").upper() == "CONFLICT" or str(item.get("conflict_class") or "") in CONFLICT_CLASSES:
            conflict_items.append(item)
    groups = list_semantic_candidate_groups(**node_result_scope(source_scenic_id, source_node_id), limit=limit, offset=0)
    conflict_groups = [g for g in groups.get("items", []) if str(g.get("conflict_class") or "") in CONFLICT_CLASSES]
    return {"items": conflict_items, "groups": conflict_groups, "total": len(conflict_items), "group_total": len(conflict_groups)}


def list_node_gaps(*, source_scenic_id: str, source_node_id: str, status: str | None = None, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    return list_semantic_gap_status(**node_result_scope(source_scenic_id, source_node_id), status=status, limit=limit, offset=offset)


def get_node_semantic_status(*, source_scenic_id: str, source_node_id: str) -> dict[str, Any]:
    jobs = list_node_jobs(source_scenic_id=source_scenic_id, source_node_id=source_node_id, limit=20)
    candidates = list_node_candidates(source_scenic_id=source_scenic_id, source_node_id=source_node_id, limit=200)
    groups = list_semantic_candidate_groups(**node_result_scope(source_scenic_id, source_node_id), limit=200)
    gaps = list_node_gaps(source_scenic_id=source_scenic_id, source_node_id=source_node_id, limit=200)

    job_items = jobs.get("items", [])
    candidate_items = candidates.get("items", [])
    group_items = groups.get("items", [])
    gap_items = gaps.get("items", [])

    running_jobs = [j for j in job_items if str(j.get("status") or "").upper() in RUNNING_JOB_STATUSES]
    failed_jobs = [j for j in job_items if str(j.get("status") or "").upper() == "FAILED"]
    review_candidates = [c for c in candidate_items if str(c.get("status") or "").upper() in REVIEW_STATUSES]
    adopted_candidates = [c for c in candidate_items if str(c.get("status") or "").upper() in {"ADOPTED", "PUBLISHED"}]
    conflict_candidates = [c for c in candidate_items if str(c.get("status") or "").upper() == "CONFLICT" or str(c.get("conflict_class") or "") in CONFLICT_CLASSES]
    conflict_groups = [g for g in group_items if str(g.get("conflict_class") or "") in CONFLICT_CLASSES]
    open_gaps = [g for g in gap_items if str(g.get("status") or "") not in {"completed", "resolved"}]

    if running_jobs:
        status = "RUNNING"
    elif conflict_candidates or conflict_groups:
        status = "CONFLICTED"
    elif review_candidates:
        status = "PENDING_REVIEW"
    elif adopted_candidates and not open_gaps:
        status = "COMPLETED"
    elif candidate_items or gap_items or job_items:
        status = "PENDING_REVIEW" if review_candidates else "EMPTY"
    else:
        status = "EMPTY"

    latest_job = job_items[0] if job_items else None
    return {
        "source_scenic_id": str(source_scenic_id),
        "source_node_id": str(source_node_id),
        "status": status,
        "latest_job": latest_job,
        "counts": {
            "jobs": int(jobs.get("total") or len(job_items)),
            "running_jobs": len(running_jobs),
            "failed_jobs": len(failed_jobs),
            "candidates": int(candidates.get("total") or len(candidate_items)),
            "pending_review_candidates": len(review_candidates),
            "adopted_candidates": len(adopted_candidates),
            "conflict_candidates": len(conflict_candidates),
            "candidate_groups": int(groups.get("total") or len(group_items)),
            "conflict_groups": len(conflict_groups),
            "gaps": int(gaps.get("total") or len(gap_items)),
            "open_gaps": len(open_gaps),
        },
        "jobs": job_items[:20],
        "candidates": candidate_items[:50],
        "candidate_groups": group_items[:50],
        "gaps": gap_items[:50],
    }
