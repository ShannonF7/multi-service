"""GrowthRun 证据消费审计的纯函数回归测试。"""

from src.semantic_growth.audit_service import _collect_evidence_issues


def test_claimed_without_any_output_is_reported():
    """CLAIMED 且没有抽取结果必须报告悬挂消费。"""
    issues = _collect_evidence_issues([
        {"id": 1, "consumption_state": "CLAIMED", "consumption_result": None,
         "raw_claim_count": 0, "candidate_binding_count": 0, "fact_binding_count": 0},
    ])
    assert issues == [{"code": "CLAIMED_WITHOUT_RESULT", "evidence_unit_id": 1}]


def test_claimed_with_fact_binding_is_not_reported():
    """已有事实绑定说明消费已产生结果，不应误报悬挂。"""
    issues = _collect_evidence_issues([
        {"id": 2, "consumption_state": "CLAIMED", "consumption_result": None,
         "raw_claim_count": 0, "candidate_binding_count": 0, "fact_binding_count": 1},
    ])
    assert issues == []


def test_processed_without_result_remains_a_separate_issue():
    """PROCESSED 缺少结果仍沿用原有一致性检查。"""
    issues = _collect_evidence_issues([
        {"id": 3, "consumption_state": "PROCESSED", "consumption_result": None,
         "raw_claim_count": 0, "candidate_binding_count": 0, "fact_binding_count": 0},
    ])
    assert issues == [{"code": "PROCESSED_WITHOUT_RESULT", "evidence_unit_id": 3}]
