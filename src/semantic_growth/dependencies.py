from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope

logger = logging.getLogger(__name__)

TERMINAL_UPSTREAM = {"ADOPTED", "PUBLISHED"}
REJECTED_UPSTREAM = {"REJECTED", "INVALIDATED"}


def _dependency_state(status: str, *, node_candidate: bool = False) -> str:
    normalized = str(status or "").upper()
    if normalized in REJECTED_UPSTREAM:
        return "INVALIDATED"
    if normalized in TERMINAL_UPSTREAM:
        return "PENDING"
    return "BLOCKED_BY_DEPENDENCY"


def review_node_candidate(candidate_id: int, *, status: str, reviewed_by: str = "", review_note: str = "") -> dict[str, Any] | None:
    normalized = str(status or "").strip().upper()
    if normalized not in {"ADOPTED", "REJECTED", "INVALIDATED"}:
        raise ValueError("invalid node candidate status")
    with ai_session_scope() as db:
        row = db.execute(
            text(
                """
                update semantic_node_candidates
                set status=:status,
                    metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                    updated_at=now()
                where id=:id
                returning *
                """
            ),
            {
                "id": int(candidate_id),
                "status": normalized,
                "patch": json.dumps(
                    {"reviewed_by": str(reviewed_by or ""), "review_note": str(review_note or "")},
                    ensure_ascii=False,
                ),
            },
        ).mappings().first()
    return dict(row) if row else None


def _candidate_dependency_ids(metadata: Any) -> list[int]:
    if not isinstance(metadata, dict):
        return []
    values = metadata.get("dependency_candidate_ids") or metadata.get("upstream_candidate_ids") or []
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    result: list[int] = []
    for value in values:
        try:
            candidate_id = int(value)
        except (TypeError, ValueError):
            continue
        if candidate_id > 0:
            result.append(candidate_id)
    return list(dict.fromkeys(result))


def _scope_for_nodes(db: Any, source_scenic_id: str, node_ids: set[str]) -> list[dict[str, Any]]:
    if not node_ids:
        return []
    rows = db.execute(
        text(
            """
            select source_node_id, parent_source_node_id, node_name, node_type
            from semantic_nodes
            where source_scenic_id=:source_scenic_id
              and source_node_id=any(:node_ids)
            """
        ),
        {"source_scenic_id": str(source_scenic_id), "node_ids": list(node_ids)},
    ).mappings().all()
    scope: dict[str, dict[str, Any]] = {}
    for row in rows:
        node_id = str(row.get("source_node_id") or "")
        if node_id:
            scope[node_id] = {
                "node_id": node_id,
                "name": row.get("node_name"),
                "node_type": row.get("node_type"),
                "role": "candidate_source",
            }
        parent_id = str(row.get("parent_source_node_id") or "")
        if parent_id:
            scope.setdefault(parent_id, {"node_id": parent_id, "role": "parent_context"})
    return list(scope.values())


def persist_candidate_dependencies(
    *,
    growth_run_id: str,
    source_scenic_id: str,
    candidate_ids: list[int | str],
) -> dict[str, Any]:
    ids = list(dict.fromkeys(int(item) for item in candidate_ids if str(item).isdigit()))
    if not ids:
        return {"candidate_count": 0, "dependency_count": 0, "affected_scope": [], "errors": []}

    try:
        with ai_session_scope() as db:
            candidates = [
                dict(row)
                for row in db.execute(
                    text(
                        """
                        select id, source_node_id, target_source_node_id,
                               target_node_candidate_id, claim_type, object_name,
                               metadata, status
                        from semantic_claim_candidates
                        where id=any(:ids) and source_scenic_id=:source_scenic_id
                        """
                    ),
                    {"ids": ids, "source_scenic_id": str(source_scenic_id)},
                ).mappings().all()
            ]
            dependencies: list[dict[str, Any]] = []
            scope_node_ids = {str(row.get("source_node_id") or "") for row in candidates if row.get("source_node_id")}
            for row in candidates:
                downstream_id = int(row["id"])
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                for upstream_id in _candidate_dependency_ids(metadata):
                    dependencies.append(
                        {
                            "downstream_candidate_id": downstream_id,
                            "upstream_candidate_id": upstream_id,
                            "upstream_node_candidate_id": None,
                            "dependency_type": "CANDIDATE_CHAIN",
                            "reason": "候选依赖上游候选",
                            "node_candidate": False,
                        }
                    )
                subject_node_candidate_id = metadata.get("subject_node_candidate_id")
                if subject_node_candidate_id:
                    dependencies.append(
                        {
                            "downstream_candidate_id": downstream_id,
                            "upstream_candidate_id": None,
                            "upstream_node_candidate_id": int(subject_node_candidate_id),
                            "dependency_type": "CLAIM_FROM_NEW_ENTITY",
                            "reason": "知识候选的主体是待审核新实体",
                            "node_candidate": True,
                        }
                    )
                node_candidate_id = row.get("target_node_candidate_id")
                if node_candidate_id and int(node_candidate_id) != int(subject_node_candidate_id or 0):
                    dependencies.append(
                        {
                            "downstream_candidate_id": downstream_id,
                            "upstream_candidate_id": None,
                            "upstream_node_candidate_id": int(node_candidate_id),
                            "dependency_type": "RELATION_TO_NEW_ENTITY",
                            "reason": "关系候选指向待统一的新实体",
                            "node_candidate": True,
                        }
                    )
                target_id = str(row.get("target_source_node_id") or "")
                if target_id:
                    scope_node_ids.add(target_id)

            for dependency in dependencies:
                upstream_status = "PENDING"
                if dependency["upstream_candidate_id"] is not None:
                    upstream_status = str(
                        db.execute(
                            text("select status from semantic_claim_candidates where id=:id"),
                            {"id": dependency["upstream_candidate_id"]},
                        ).scalar()
                        or "PENDING"
                    )
                else:
                    upstream_status = str(
                        db.execute(
                            text("select status from semantic_node_candidates where id=:id"),
                            {"id": dependency["upstream_node_candidate_id"]},
                        ).scalar()
                        or "PENDING"
                    )
                state = _dependency_state(upstream_status, node_candidate=dependency["node_candidate"])
                db.execute(
                    text(
                        """
                        insert into semantic_growth_candidate_dependencies (
                            growth_run_id, downstream_candidate_id, upstream_candidate_id,
                            upstream_node_candidate_id, dependency_type, state, reason, metadata, updated_at
                        ) values (
                            :growth_run_id, :downstream_candidate_id, :upstream_candidate_id,
                            :upstream_node_candidate_id, :dependency_type, :state, :reason,
                            cast(:metadata as jsonb), now()
                        )
                        on conflict (
                            growth_run_id, downstream_candidate_id, upstream_candidate_id,
                            upstream_node_candidate_id, dependency_type
                        ) do update set
                            state=excluded.state,
                            reason=excluded.reason,
                            metadata=excluded.metadata,
                            updated_at=now()
                        """
                    ),
                    {
                        **dependency,
                        "growth_run_id": str(growth_run_id),
                        "state": state,
                        "metadata": json.dumps(
                            {
                                "upstream_status": upstream_status,
                                "blocking_mode": "external_entity_review_pending" if dependency["node_candidate"] else "candidate_chain",
                            },
                            ensure_ascii=False,
                        ),
                    },
                )
                db.execute(
                    text(
                        """
                        update semantic_claim_candidates
                        set metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                            updated_at=now()
                        where id=:candidate_id
                        """
                    ),
                    {
                        "candidate_id": dependency["downstream_candidate_id"],
                        "patch": json.dumps(
                            {
                                "g5_dependency_state": state,
                                "g5_dependency_type": dependency["dependency_type"],
                                "g5_dependency_reason": dependency["reason"],
                            },
                            ensure_ascii=False,
                        ),
                    },
                )

            affected_scope = _scope_for_nodes(db, str(source_scenic_id), scope_node_ids)
            db.execute(
                text(
                    """
                    update semantic_growth_runs
                    set metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                        updated_at=now()
                    where growth_run_id=:growth_run_id
                    """
                ),
                {
                    "growth_run_id": str(growth_run_id),
                    "patch": json.dumps(
                        {
                            "g5_affected_scope": affected_scope,
                            "g5_dependency_count": len(dependencies),
                            "g5_dependency_version": "growth-g5-v1",
                        },
                        ensure_ascii=False,
                    ),
                },
            )
        return {
            "candidate_count": len(candidates),
            "dependency_count": len(dependencies),
            "affected_scope": affected_scope,
            "errors": [],
        }
    except Exception as exc:
        logger.warning("G5 dependency persistence failed: %s", exc)
        return {"candidate_count": len(ids), "dependency_count": 0, "affected_scope": [], "errors": [str(exc)]}


def refresh_dependency_states(*, growth_run_id: str, reviewed_candidate_id: int | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"growth_run_id": str(growth_run_id)}
    where = "where d.growth_run_id=:growth_run_id"
    if reviewed_candidate_id is not None:
        where += " and (d.upstream_candidate_id=:reviewed_candidate_id or d.upstream_node_candidate_id=:reviewed_candidate_id or d.downstream_candidate_id=:reviewed_candidate_id)"
        params["reviewed_candidate_id"] = int(reviewed_candidate_id)
    try:
        with ai_session_scope() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    text(
                        """
                        select d.id, d.downstream_candidate_id, d.upstream_candidate_id,
                               d.upstream_node_candidate_id, d.dependency_type,
                               c.status as upstream_candidate_status,
                               n.status as upstream_node_status,
                               downstream.status as downstream_status
                        from semantic_growth_candidate_dependencies d
                        left join semantic_claim_candidates c on c.id=d.upstream_candidate_id
                        left join semantic_node_candidates n on n.id=d.upstream_node_candidate_id
                        join semantic_claim_candidates downstream on downstream.id=d.downstream_candidate_id
                        """
                        + where
                    ),
                    params,
                ).mappings().all()
            ]
            counts = defaultdict(int)
            for row in rows:
                node_candidate = row.get("upstream_node_candidate_id") is not None
                upstream_status = row.get("upstream_node_status") if node_candidate else row.get("upstream_candidate_status")
                state = _dependency_state(str(upstream_status or "PENDING"), node_candidate=node_candidate)
                counts[state] += 1
                db.execute(
                    text(
                        """
                        update semantic_growth_candidate_dependencies
                        set state=:state,
                            metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                            updated_at=now()
                        where id=:id
                        """
                    ),
                    {
                        "id": int(row["id"]),
                        "state": state,
                        "patch": json.dumps({"upstream_status": upstream_status}, ensure_ascii=False),
                    },
                )
                downstream_status = str(row.get("downstream_status") or "").upper()
                next_status = None
                if state == "BLOCKED_BY_DEPENDENCY" and downstream_status in {"PENDING", "CONFLICT", "LOW_EVIDENCE"}:
                    next_status = "BLOCKED_BY_DEPENDENCY"
                elif state == "INVALIDATED" and downstream_status not in {"ADOPTED", "REJECTED", "INVALIDATED"}:
                    next_status = "INVALIDATED"
                elif state == "PENDING" and downstream_status == "BLOCKED_BY_DEPENDENCY":
                    next_status = "PENDING"
                if next_status:
                    db.execute(
                        text(
                            """
                            update semantic_claim_candidates
                            set status=:status,
                                metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                                updated_at=now()
                            where id=:id
                            """
                        ),
                        {
                            "id": int(row["downstream_candidate_id"]),
                            "status": next_status,
                            "patch": json.dumps({"g5_dependency_state": state}, ensure_ascii=False),
                        },
                    )
            return {"dependency_count": len(rows), "state_counts": dict(counts), "errors": []}
    except Exception as exc:
        logger.warning("G5 dependency refresh failed: %s", exc)
        return {"dependency_count": 0, "state_counts": {}, "errors": [str(exc)]}
