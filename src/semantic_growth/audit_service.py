"""GrowthRun G2.5 只读审计服务。

本模块只读取自增长相关数据表，不参与补全链路，也不会修改任务、候选或证据状态。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from src.rag.dependencies import ai_session_scope
from .repository import ensure_schema


def _count(db: Any, table: str, where: str, params: dict[str, Any]) -> int:
    """统计一个 GrowthRun 在指定数据表中的行数。

    输入：数据库会话、受信任的表名、SQL 条件和参数。输出：整数行数。
    仅由 ``audit_growth_run`` 调用；表名均来自本模块的固定常量。
    """
    row = db.execute(text(f"select count(*) as n from {table} where {where}"), params).mappings().first()
    return int(row["n"] or 0) if row else 0


def audit_growth_run(growth_run_id: str, *, sample_limit: int = 50) -> dict[str, Any]:
    """生成一个 GrowthRun 的只读谱系与一致性报告。

    输入：``growth_run_id`` 和可选的问题样本上限。输出：任务信息、阶段计数、
    派生比率和可处理的问题列表。由 B 端两个路由中的
    ``/rag/growth-runs/{id}/audit`` 接口调用。
    """
    run_id = str(growth_run_id or "").strip()
    if not run_id:
        raise ValueError("growth_run_id is required")
    sample_limit = max(1, min(int(sample_limit or 50), 200))
    ensure_schema()
    params = {"growth_run_id": run_id}
    with ai_session_scope() as db:
        run = db.execute(
            text("select growth_run_id, domain_id, status, status_reason_code, metadata, created_at, updated_at from semantic_growth_runs where growth_run_id=:growth_run_id"),
            params,
        ).mappings().first()
        if not run:
            raise LookupError("growth run not found")

        counts = {
            "evidence_unit_count": _count(db, "semantic_growth_evidence_units", "growth_run_id=:growth_run_id", params),
            "raw_entity_count": _count(db, "semantic_growth_raw_entities", "growth_run_id=:growth_run_id", params),
            "raw_claim_count": _count(db, "semantic_growth_raw_claims", "growth_run_id=:growth_run_id", params),
            "candidate_link_count": _count(db, "semantic_growth_candidate_links", "growth_run_id=:growth_run_id", params),
            "candidate_evidence_binding_count": _count(db, "semantic_growth_candidate_evidence_bindings", "growth_run_id=:growth_run_id", params),
            "fact_evidence_binding_count": _count(db, "semantic_growth_fact_evidence_bindings", "growth_run_id=:growth_run_id", params),
            "step_count": _count(db, "semantic_growth_step_records", "growth_run_id=:growth_run_id", params),
        }
        candidate_rows = db.execute(
            text(
                """
                select c.id, c.canonical_claim_key, c.update_operation, c.status,
                       c.source_node_id, c.predicate, c.object_value, c.object_name,
                       count(distinct b.id) as evidence_binding_count
                from semantic_growth_candidate_links l
                join semantic_claim_candidates c on c.id=l.candidate_id
                left join semantic_growth_candidate_evidence_bindings b
                  on b.growth_run_id=l.growth_run_id and b.candidate_id=c.id
                where l.growth_run_id=:growth_run_id
                group by c.id, c.canonical_claim_key, c.update_operation, c.status,
                         c.source_node_id, c.predicate, c.object_value, c.object_name
                order by c.id
                limit :sample_limit
                """
            ),
            {**params, "sample_limit": sample_limit},
        ).mappings().all()
        evidence_rows = db.execute(
            text(
                """
                select e.id, e.source_id, e.chunk_id, e.chunk_hash, e.source_type,
                       coalesce(c.state, 'UNKNOWN') as consumption_state,
                       c.result as consumption_result,
                       count(distinct rc.id) as raw_claim_count,
                       count(distinct cb.id) as candidate_binding_count,
                       count(distinct fb.id) as fact_binding_count
                from semantic_growth_evidence_units e
                left join semantic_growth_evidence_consumptions c on c.id=e.consumption_id
                left join semantic_growth_raw_claims rc on rc.evidence_unit_id=e.id
                left join semantic_growth_candidate_evidence_bindings cb on cb.evidence_unit_id=e.id and cb.growth_run_id=:growth_run_id
                left join semantic_growth_fact_evidence_bindings fb on fb.evidence_unit_id=e.id and fb.growth_run_id=:growth_run_id
                where e.growth_run_id=:growth_run_id
                group by e.id, e.source_id, e.chunk_id, e.chunk_hash, e.source_type, c.state, c.result
                order by e.id
                limit :sample_limit
                """
            ),
            {**params, "sample_limit": sample_limit},
        ).mappings().all()
        operation_rows = db.execute(
            text(
                """
                select coalesce(c.update_operation, 'UNCLASSIFIED') as operation, count(distinct c.id) as n
                from semantic_growth_candidate_links l
                join semantic_claim_candidates c on c.id=l.candidate_id
                where l.growth_run_id=:growth_run_id
                group by coalesce(c.update_operation, 'UNCLASSIFIED')
                order by operation
                """
            ),
            params,
        ).mappings().all()

    issues: list[dict[str, Any]] = []
    for row in candidate_rows:
        if not str(row.get("canonical_claim_key") or "").strip():
            issues.append({"code": "CANDIDATE_MISSING_CANONICAL_KEY", "candidate_id": int(row["id"])})
        if int(row.get("evidence_binding_count") or 0) == 0:
            issues.append({"code": "CANDIDATE_MISSING_LINEAGE", "candidate_id": int(row["id"])})
    for row in evidence_rows:
        state = str(row.get("consumption_state") or "").upper()
        if state == "FAILED":
            issues.append({"code": "EVIDENCE_CONSUMPTION_FAILED", "evidence_unit_id": int(row["id"])})
        if state == "PROCESSED" and row.get("consumption_result") is None:
            issues.append({"code": "PROCESSED_WITHOUT_RESULT", "evidence_unit_id": int(row["id"])})

    operation_counts = {str(row["operation"]): int(row["n"] or 0) for row in operation_rows}
    canonical_count = sum(1 for row in candidate_rows if str(row.get("canonical_claim_key") or "").strip())
    lineage_complete = not any(issue["code"] in {"CANDIDATE_MISSING_CANONICAL_KEY", "CANDIDATE_MISSING_LINEAGE"} for issue in issues)
    raw_claim_count = counts["raw_claim_count"]
    return {
        "run": dict(run),
        "counts": counts,
        "operations": operation_counts,
        "ratios": {
            "candidate_lineage_coverage": round(counts["candidate_evidence_binding_count"] / max(1, counts["candidate_link_count"]), 4),
            "raw_claim_to_candidate_ratio": round(raw_claim_count / max(1, canonical_count), 4),
            "fact_binding_ratio": round(counts["fact_evidence_binding_count"] / max(1, counts["evidence_unit_count"]), 4),
        },
        "lineage_complete": lineage_complete,
        "issues": issues[:sample_limit],
        "issue_count": len(issues),
        "candidate_samples": [dict(row) for row in candidate_rows],
        "evidence_samples": [dict(row) for row in evidence_rows],
    }
