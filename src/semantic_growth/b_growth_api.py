from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from src.rag.service.semantic_candidate_store import (
    update_semantic_candidate_status,
    update_semantic_candidate_status_batch,
)

from .dependencies import refresh_dependency_states, review_node_candidate
from .audit_service import audit_growth_run
from .repository import create_run, get_run, record_publication_result, update_publication_sync_status
from .service import (
    growth_run_state,
    resume_growth_run,
    list_growth_runs_svc,
    pause_growth_run_svc,
    cancel_growth_run_svc,
    resume_paused_growth_run,
    start_growth_run_background,
)

router = APIRouter(prefix="/rag/growth-runs", tags=["Semantic Growth"])


@router.post("")
def create_growth_run(payload: dict = Body(default={}), background_tasks: BackgroundTasks = None):
    if not str(payload.get("domain_id") or "").strip():
        raise HTTPException(status_code=400, detail="domain_id is required")
    try:
        import uuid
        queued = dict(payload)
        queued.setdefault("growth_run_id", f"growth-{uuid.uuid4().hex[:20]}")
        run = create_run(queued)
        if background_tasks is None:
            raise HTTPException(status_code=500, detail="background task runner unavailable")
        background_tasks.add_task(start_growth_run_background, queued)
        return {
            "growth_run_id": run["growth_run_id"],
            "thread_id": run["thread_id"],
            "status": "STARTING",
            "next": ["load_scope"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{growth_run_id}")
def get_growth_run(growth_run_id: str):
    try:
        return growth_run_state(growth_run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{growth_run_id}/audit")
def get_growth_run_audit(growth_run_id: str, sample_limit: int = 50):
    """返回 G2.5 只读谱系与一致性审计结果。

    接口只读取自增长数据表，不会修改补全或自增长状态。
    """
    try:
        return audit_growth_run(growth_run_id, sample_limit=sample_limit)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{growth_run_id}/status")
def get_growth_run_status(growth_run_id: str):
    """Lightweight polling endpoint; never loads candidates, evidence or lineage."""
    try:
        run = get_run(growth_run_id)
        if not run:
            raise HTTPException(status_code=404, detail="growth run not found")
        return {
            "growth_run_id": run.get("growth_run_id"),
            "status": run.get("status"),
            "stop_reason": run.get("stop_reason"),
            "status_reason_code": run.get("status_reason_code"),
            "updated_at": run.get("updated_at"),
            "finished_at": run.get("finished_at"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{growth_run_id}/resume")
def resume_growth(growth_run_id: str, payload: dict = Body(default={})):
    try:
        return resume_growth_run(growth_run_id, payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{growth_run_id}/resume-paused")
def resume_paused_growth(growth_run_id: str, payload: dict = Body(default={})):
    try:
        return resume_paused_growth_run(growth_run_id, payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("")
def list_growth_runs(domain_id: str | None = None, status: str | None = None, limit: int = 50):
    """List all growth runs, optionally filtered by domain and status."""
    try:
        return {"runs": list_growth_runs_svc(domain_id, status, limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{growth_run_id}/pause")
def pause_growth(growth_run_id: str):
    """Pause a running growth run."""
    try:
        result = pause_growth_run_svc(growth_run_id)
        if not result:
            raise HTTPException(status_code=409, detail="run cannot be paused in current state")
        return result
    except HTTPException as exc:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{growth_run_id}/cancel")
def cancel_growth(growth_run_id: str):
    """Cancel a growth run."""
    try:
        result = cancel_growth_run_svc(growth_run_id)
        if not result:
            raise HTTPException(status_code=409, detail="run cannot be cancelled in current state")
        return result
    except HTTPException as exc:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/publication-sync")
def update_growth_publication_sync(payload: dict = Body(default={})):
    batch_id = str(payload.get("publication_batch_id") or "").strip()
    if not batch_id:
        raise HTTPException(status_code=400, detail="publication_batch_id is required")
    try:
        result = update_publication_sync_status(
            batch_id,
            status=str(payload.get("status") or ""),
            error=str(payload.get("error") or ""),
            affected_scope=payload.get("affected_scope") if isinstance(payload.get("affected_scope"), list) else None,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="publication record not found")
        return {"publication": result}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{growth_run_id}/complete-round")
def complete_growth_round(growth_run_id: str, payload: dict = Body(default={})): 
    try:
        detail = growth_run_state(growth_run_id)
        candidates = detail.get("candidates") or []
        node_candidates = detail.get("node_candidates") or []
        if (not candidates and not node_candidates) or any(
            str(c.get("status") or "").upper() not in {"ADOPTED", "PUBLISHED", "REJECTED", "INVALIDATED"}
            for c in [*candidates, *node_candidates]
        ):
            raise HTTPException(status_code=409, detail="all growth candidates and new-entity candidates must be reviewed first")
        publication = payload.get("publication")
        publication_record = None
        if isinstance(publication, dict):
            publication_payload = dict(publication)
            publication_payload.setdefault("affected_scope", detail.get("affected_scope") or [])
            publication_record = record_publication_result(growth_run_id, publication_payload)
        result = resume_growth_run(
            growth_run_id,
            {**payload, "action": "round_complete", "thread_id": payload.get("thread_id") or detail["run"]["thread_id"]},
        )
        if publication_record is not None:
            result["publication"] = publication_record
        return result
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{growth_run_id}/node-candidates/{candidate_id}/review")
def review_growth_node_candidate(growth_run_id: str, candidate_id: int, payload: dict = Body(default={})):
    try:
        detail = growth_run_state(growth_run_id)
        linked = {int(c.get("id")) for c in (detail.get("node_candidates") or []) if c.get("id") is not None}
        if candidate_id not in linked:
            raise HTTPException(status_code=404, detail="node candidate is not linked to growth run")
        action = str(payload.get("action") or "").lower()
        status = {"accept": "ADOPTED", "reject": "REJECTED", "invalidate": "INVALIDATED"}.get(action)
        if not status:
            raise HTTPException(status_code=400, detail="action must be accept, reject, or invalidate")
        result = review_node_candidate(
            candidate_id,
            status=status,
            reviewed_by=str(payload.get("reviewer_id") or "growth-review"),
            review_note=str(payload.get("review_note") or ""),
        )
        return {
            "growth_run_id": growth_run_id,
            "node_candidate": result,
            "action": action,
            "dependency_refresh": refresh_dependency_states(
                growth_run_id=growth_run_id,
                reviewed_candidate_id=candidate_id,
            ),
        }
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{growth_run_id}/candidates/{candidate_id}/review")
def review_growth_candidate(growth_run_id: str, candidate_id: int, payload: dict = Body(default={})): 
    try:
        detail = growth_run_state(growth_run_id)
        linked = {int(c.get("id")) for c in (detail.get("candidates") or []) if c.get("id") is not None}
        if candidate_id not in linked:
            raise HTTPException(status_code=404, detail="candidate is not linked to growth run")
        action = str(payload.get("action") or "").lower()
        status = {"accept": "ADOPTED", "modify": "ADOPTED", "reject": "REJECTED", "invalidate": "INVALIDATED"}.get(action)
        if not status:
            raise HTTPException(status_code=400, detail="action must be accept, modify, reject, or invalidate")
        note = str(payload.get("review_note") or payload.get("modified_value") or "")
        result = update_semantic_candidate_status(candidate_id, status=status, reviewed_by=str(payload.get("reviewer_id") or "growth-review"), review_note=note, object_value=(str(payload.get("modified_value")) if action == "modify" and payload.get("modified_value") is not None else None))
        dependency_refresh = refresh_dependency_states(
            growth_run_id=growth_run_id,
            reviewed_candidate_id=candidate_id,
        )
        return {
            "growth_run_id": growth_run_id,
            "candidate": result,
            "action": action,
            "dependency_refresh": dependency_refresh,
        }
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{growth_run_id}/candidates/review-batch")
def review_growth_candidates_batch(growth_run_id: str, payload: dict = Body(default={})):
    try:
        detail = growth_run_state(growth_run_id)
        linked = {
            int(candidate["id"]): str(candidate.get("status") or "").upper()
            for candidate in (detail.get("candidates") or [])
            if candidate.get("id") is not None
        }
        try:
            candidate_ids = list(dict.fromkeys(int(value) for value in (payload.get("candidate_ids") or [])))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="candidate_ids must contain integers") from exc
        if not candidate_ids or len(candidate_ids) > 1000:
            raise HTTPException(status_code=400, detail="candidate_ids must contain 1-1000 items")
        missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in linked]
        if missing:
            raise HTTPException(status_code=404, detail=f"candidates are not linked to growth run: {missing[:10]}")
        terminal = {"ADOPTED", "PUBLISHED", "REJECTED", "INVALIDATED", "DUPLICATE"}
        blocked = [candidate_id for candidate_id in candidate_ids if linked[candidate_id] == "BLOCKED_BY_DEPENDENCY"]
        if blocked:
            raise HTTPException(status_code=409, detail=f"blocked candidates cannot be reviewed yet: {blocked[:10]}")
        reviewable_ids = [candidate_id for candidate_id in candidate_ids if linked[candidate_id] not in terminal]
        if not reviewable_ids:
            return {
                "growth_run_id": growth_run_id,
                "requested_count": len(candidate_ids),
                "updated_count": 0,
                "candidate_ids": [],
                "dependency_refresh": refresh_dependency_states(growth_run_id=growth_run_id),
            }
        action = str(payload.get("action") or "").lower()
        status = {"accept": "ADOPTED", "reject": "REJECTED", "invalidate": "INVALIDATED"}.get(action)
        if not status:
            raise HTTPException(status_code=400, detail="action must be accept, reject, or invalidate")
        result = update_semantic_candidate_status_batch(
            reviewable_ids,
            status=status,
            reviewed_by=str(payload.get("reviewer_id") or "growth-review"),
            review_note=str(payload.get("review_note") or ""),
        )
        return {
            "growth_run_id": growth_run_id,
            **result,
            "dependency_refresh": refresh_dependency_states(growth_run_id=growth_run_id),
        }
    except HTTPException:
        raise
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
