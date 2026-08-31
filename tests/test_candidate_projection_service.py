"""候选审核投影服务的单元测试。

测试目的：锁定补全、自增长、图片关联和事实增强的分类边界。
测试不连接数据库，不调用外部模型。
"""
from src.rag.service.candidate_projection_service import classify_candidate, project_candidate

def test_growth_entity_projects_to_growth_node():
    got = classify_candidate({"run_id": "growth-test", "provenance_type": "growth_evidence_unit", "candidate_type": "discovered_entity"})
    assert got == {"discovery_track": "OPEN_DISCOVERY", "candidate_kind": "NODE", "review_surface": "GROWTH_RUN"}

def test_completion_template_projects_to_node_workbench():
    got = classify_candidate({"question_id": 7, "job_id": 9, "provenance_type": "web", "candidate_type": "template_property", "claim_type": "property"})
    assert got == {"discovery_track": "TARGETED_COMPLETION", "candidate_kind": "PROPERTY", "review_surface": "NODE_WORKBENCH"}

def test_asset_binding_is_growth_surface():
    got = classify_candidate({"run_id": "growth-test", "candidate_type": "asset_binding", "metadata": {"candidate_kind": "ASSET_BINDING"}})
    assert got == {"discovery_track": "ASSET_BINDING", "candidate_kind": "ASSET_BINDING", "review_surface": "GROWTH_RUN"}

def test_exists_is_audit_only_and_preserves_origin():
    row = {"run_id": "growth-test", "candidate_type": "discovered_fact", "claim_type": "property", "update_operation": "EXISTS"}
    got = project_candidate(row)
    assert got["review_surface"] == "AUDIT_ONLY"
    assert got["origin_ref"] == {"run_id": "growth-test"}
