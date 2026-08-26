from __future__ import annotations

import json
import logging
import unicodedata
from collections import defaultdict
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from src.rag.service.value_normalization_service import normalize_text_value

logger = logging.getLogger(__name__)


def value_key(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return "".join(char for char in normalized if char.isalnum())


def sanitize_same_value_conflicts(
    *,
    source_scenic_id: str,
    candidate_ids: list[int | str],
) -> dict[str, Any]:
    """Remove meaningless same-value conflict presentation for this batch.

    Only CONFLICT candidates are downgraded to PENDING. ADOPTED/REJECTED/
    INVALIDATED candidates are never changed.
    """
    ids = list(dict.fromkeys(int(item) for item in candidate_ids if str(item).isdigit()))
    if not ids:
        return {"candidate_count": 0, "groups_checked": 0, "resolved_group_count": 0, "resolved_candidate_count": 0, "errors": []}

    try:
        with ai_session_scope() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    text(
                        """
                        select id, conflict_group, candidate_group_key, status,
                               claim_type, predicate, object_value, object_name,
                               source_id, source_url, evidence_ids
                        from semantic_claim_candidates
                        where id=any(:ids) and source_scenic_id=:source_scenic_id
                        """
                    ),
                    {"ids": ids, "source_scenic_id": str(source_scenic_id)},
                ).mappings().all()
            ]
    except Exception as exc:
        logger.warning("G4 conflict quality load failed: %s", exc)
        return {"candidate_count": len(ids), "groups_checked": 0, "resolved_group_count": 0, "resolved_candidate_count": 0, "errors": [str(exc)]}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = str(row.get("conflict_group") or "").strip()
        if group:
            grouped[group].append(row)

    resolved_groups: list[tuple[str, list[dict[str, Any]], set[str]]] = []
    for group, items in grouped.items():
        values = {
            value_key(normalize_text_value(str(item.get("object_value") or item.get("object_name") or "")))
            for item in items
        }
        values.discard("")
        if len(values) <= 1:
            resolved_groups.append((group, items, values))

    if not resolved_groups:
        return {"candidate_count": len(rows), "groups_checked": len(grouped), "resolved_group_count": 0, "resolved_candidate_count": 0, "errors": []}

    resolved_ids = [int(item["id"]) for _, items, _ in resolved_groups for item in items]
    group_keys = list({
        str(item.get("candidate_group_key") or "")
        for _, items, _ in resolved_groups
        for item in items
        if item.get("candidate_group_key")
    })
    with ai_session_scope() as db:
        db.execute(
            text(
                """
                update semantic_claim_candidates
                set conflict_group=null,
                    conflict_class='same_value',
                    gap_status='pending_review',
                    status=case when upper(status)='CONFLICT' then 'PENDING' else status end,
                    metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                    updated_at=now()
                where id=any(:ids) and source_scenic_id=:source_scenic_id
                """
            ),
            {
                "ids": resolved_ids,
                "source_scenic_id": str(source_scenic_id),
                "patch": json.dumps(
                    {
                        "g4_conflict_quality": "AUTO_RESOLVED_SAME_VALUE",
                        "g4_conflict_reason": "same normalized value; no meaningful disagreement",
                    },
                    ensure_ascii=False,
                ),
            },
        )
        if group_keys:
            db.execute(
                text(
                    """
                    update semantic_candidate_groups
                    set conflict_class='same_value',
                        gap_status='pending_review',
                        distinct_value_count=1,
                        metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                        updated_at=now()
                    where candidate_group_key=any(:group_keys)
                    """
                ),
                {
                    "group_keys": group_keys,
                    "patch": json.dumps({"g4_conflict_quality": "AUTO_RESOLVED_SAME_VALUE"}, ensure_ascii=False),
                },
            )
        db.execute(
            text(
                """
                update semantic_conflict_groups
                set status='AUTO_RESOLVED',
                    distinct_value_count=1,
                    metadata=coalesce(metadata, '{}'::jsonb) || cast(:patch as jsonb),
                    updated_at=now()
                where conflict_group=any(:conflict_groups)
                  and source_scenic_id=:source_scenic_id
                """
            ),
            {
                "conflict_groups": [group for group, _, _ in resolved_groups],
                "source_scenic_id": str(source_scenic_id),
                "patch": json.dumps({"g4_conflict_quality": "AUTO_RESOLVED_SAME_VALUE"}, ensure_ascii=False),
            },
        )

    return {
        "candidate_count": len(rows),
        "groups_checked": len(grouped),
        "resolved_group_count": len(resolved_groups),
        "resolved_candidate_count": len(resolved_ids),
        "errors": [],
    }
