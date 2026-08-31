from src.rag.service.claim_evidence_fusion_service import (
    fuse_evidence,
    lexical_entailment_baseline,
)


def test_lexical_entailment_requires_claim_value_support():
    supported = lexical_entailment_baseline(
        "University founded in 1902",
        subject="University",
        predicate="founded",
        value="1902",
    )
    unsupported = lexical_entailment_baseline(
        "University is located in Taiyuan",
        subject="University",
        predicate="founded",
        value="1902",
    )
    assert supported > unsupported
    assert unsupported <= 0.2


def test_fusion_collapses_chunks_in_one_source_group():
    result = fuse_evidence([
        {"source_independence_key": "doc:1", "source_type": "document", "evidence_text": "A=B", "claim_subject": "A", "claim_value": "B"},
        {"source_independence_key": "doc:1", "source_type": "document", "evidence_text": "A=B", "claim_subject": "A", "claim_value": "B"},
        {"source_independence_key": "doc:2", "source_type": "web", "evidence_text": "A=B", "claim_subject": "A", "claim_value": "B"},
    ])
    assert result["evidence_count"] == 3
    assert result["independent_source_count"] == 2
    assert result["cross_source_support"] > 0.3333


def test_independent_source_groups_use_noisy_or():
    """独立来源应增强支持度，同来源重复 chunk 不应增强。"""
    one = fuse_evidence(
        [{"source_independence_key": "document:1", "source_quality": 0.5}],
        weights={"source_quality": 1.0},
    )
    duplicate = fuse_evidence(
        [
            {"source_independence_key": "document:1", "source_quality": 0.5},
            {"source_independence_key": "document:1", "source_quality": 0.5},
        ],
        weights={"source_quality": 1.0},
    )
    independent = fuse_evidence(
        [
            {"source_independence_key": "document:1", "source_quality": 0.5},
            {"source_independence_key": "document:2", "source_quality": 0.5},
        ],
        weights={"source_quality": 1.0},
    )
    assert duplicate["evidence_support_score"] == one["evidence_support_score"] == 0.5
    assert independent["evidence_support_score"] == 0.75


def test_missing_source_keys_do_not_merge_unrelated_bindings():
    """缺少来源键时按绑定序号隔离，避免错误合并为 unknown。"""
    result = fuse_evidence(
        [{"source_quality": 0.4}, {"source_quality": 0.6}],
        weights={"source_quality": 1.0},
    )
    assert result["independent_source_count"] == 2
