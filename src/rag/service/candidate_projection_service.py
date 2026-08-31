"""候选审核投影服务。

本文件用途：把统一候选表中的历史记录投影成用户可理解的业务分类。
输入：semantic_claim_candidates 查询行，允许字段缺失。
输出：包含 discovery_track、candidate_kind、review_surface、origin_ref 的新字典。
本模块只读、不写数据库；分类规则必须保持确定性，便于单元测试和回溯。
"""
from __future__ import annotations
from typing import Any

TRACKS = {"TARGETED_COMPLETION", "OPEN_DISCOVERY", "ASSET_BINDING", "UNKNOWN"}
KINDS = {"PROPERTY", "RELATION", "NODE", "ASSET_BINDING", "CONFLICT", "FACT"}
SURFACES = {"NODE_WORKBENCH", "GROWTH_RUN", "AUDIT_ONLY", "UNKNOWN"}

def _text(value: Any) -> str:
    """将数据库可空字段转为去空格字符串，避免 None 参与规则判断。"""
    return str(value or "").strip()

def classify_candidate(row: dict[str, Any]) -> dict[str, str]:
    """根据已有来源、任务和候选字段计算稳定的审核分类。

    输入：候选行字典；metadata 可为空或不是字典。
    输出：discovery_track、candidate_kind、review_surface 三个分类字段。
    异常：不会因单个字段格式异常抛出，未知值统一落到 UNKNOWN/FACT。
    """
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    explicit_track = _text(metadata.get("discovery_track")).upper()
    explicit_kind = _text(metadata.get("candidate_kind")).upper()
    explicit_surface = _text(metadata.get("review_surface")).upper()
    run_id = _text(row.get("run_id"))
    provenance = _text(row.get("provenance_type")).lower()
    retrieval = _text(row.get("retrieval_source")).lower()
    candidate_type = _text(row.get("candidate_type")).lower()
    claim_type = _text(row.get("claim_type")).lower()
    status = _text(row.get("status")).upper()
    operation = _text(row.get("update_operation")).upper()

    if explicit_track in TRACKS:
        track = explicit_track
    elif explicit_kind == "ASSET_BINDING" or "asset" in candidate_type and "binding" in candidate_type:
        track = "ASSET_BINDING"
    elif run_id.startswith("growth-") or provenance == "growth_evidence_unit" or retrieval == "provided_evidence":
        track = "OPEN_DISCOVERY"
    elif row.get("question_id") is not None or row.get("job_id") is not None or provenance in {"web", "local_kb"}:
        track = "TARGETED_COMPLETION"
    else:
        track = "UNKNOWN"

    if explicit_kind in KINDS:
        kind = explicit_kind
    elif status == "CONFLICT" or operation == "CONFLICT":
        kind = "CONFLICT"
    elif "entity" in candidate_type:
        kind = "NODE"
    elif claim_type == "property" or "property" in candidate_type:
        kind = "PROPERTY"
    elif claim_type == "relation" or "relation" in candidate_type:
        kind = "RELATION"
    else:
        kind = "FACT"

    if explicit_surface in SURFACES:
        surface = explicit_surface
    elif operation in {"EXISTS", "ENRICH"}:
        surface = "AUDIT_ONLY"
    elif track in {"OPEN_DISCOVERY", "ASSET_BINDING"}:
        surface = "GROWTH_RUN"
    elif track == "TARGETED_COMPLETION":
        surface = "NODE_WORKBENCH"
    else:
        surface = "UNKNOWN"

    return {"discovery_track": track, "candidate_kind": kind, "review_surface": surface}



def classification_sql() -> dict[str, str]:
    """返回 PostgreSQL 分类表达式，供候选列表在数据库侧过滤。

    输入：无，表达式只依赖 semantic_claim_candidates 的现有列。
    输出：discovery_track、candidate_kind、review_surface 三个 SQL CASE 表达式。
    """
    track = """case
        when upper(coalesce(metadata->>'discovery_track', '')) in ('TARGETED_COMPLETION','OPEN_DISCOVERY','ASSET_BINDING','UNKNOWN')
            then upper(metadata->>'discovery_track')
        when upper(coalesce(metadata->>'candidate_kind', '')) = 'ASSET_BINDING'
            or lower(coalesce(candidate_type, '')) like '%asset%binding%'
            then 'ASSET_BINDING'
        when coalesce(run_id, '') like 'growth-%'
            or lower(coalesce(provenance_type, '')) = 'growth_evidence_unit'
            or lower(coalesce(retrieval_source, '')) = 'provided_evidence'
            then 'OPEN_DISCOVERY'
        when question_id is not null or job_id is not null
            or lower(coalesce(provenance_type, '')) in ('web', 'local_kb')
            then 'TARGETED_COMPLETION'
        else 'UNKNOWN'
    end"""
    kind = """case
        when upper(coalesce(metadata->>'candidate_kind', '')) in ('PROPERTY','RELATION','NODE','ASSET_BINDING','CONFLICT','FACT')
            then upper(metadata->>'candidate_kind')
        when upper(coalesce(status, '')) = 'CONFLICT'
            or upper(coalesce(update_operation, '')) = 'CONFLICT'
            then 'CONFLICT'
        when lower(coalesce(candidate_type, '')) like '%entity%' then 'NODE'
        when lower(coalesce(claim_type, '')) = 'property'
            or lower(coalesce(candidate_type, '')) like '%property%' then 'PROPERTY'
        when lower(coalesce(claim_type, '')) = 'relation'
            or lower(coalesce(candidate_type, '')) like '%relation%' then 'RELATION'
        else 'FACT'
    end"""
    surface = f"""case
        when upper(coalesce(update_operation, '')) in ('EXISTS','ENRICH') then 'AUDIT_ONLY'
        when ({track}) in ('OPEN_DISCOVERY','ASSET_BINDING') then 'GROWTH_RUN'
        when ({track}) = 'TARGETED_COMPLETION' then 'NODE_WORKBENCH'
        else 'UNKNOWN'
    end"""
    return {"discovery_track": track, "candidate_kind": kind, "review_surface": surface}

def project_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """返回保留原字段并附加分类和来源引用的审核投影。

    origin_ref 优先使用 GrowthRun、问题或任务、图片资产中的已有标识。
    """
    projected = dict(row)
    classification = classify_candidate(projected)
    projected.update(classification)
    if classification["discovery_track"] in {"OPEN_DISCOVERY", "ASSET_BINDING"} and projected.get("run_id"):
        projected["origin_ref"] = {"run_id": projected.get("run_id")}
    elif projected.get("question_id") is not None:
        projected["origin_ref"] = {"question_id": projected.get("question_id"), "job_id": projected.get("job_id")}
    elif projected.get("job_id") is not None:
        projected["origin_ref"] = {"job_id": projected.get("job_id")}
    else:
        projected["origin_ref"] = {}
    return projected
