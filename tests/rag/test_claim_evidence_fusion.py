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
