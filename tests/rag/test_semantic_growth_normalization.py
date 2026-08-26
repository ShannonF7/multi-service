from src.semantic_growth.normalization import (
    _entity_recall,
    _joint_resolution_score,
    normalize_entity_name,
)


def test_g3_entity_recall_prefers_exact_and_requires_review_for_ambiguous():
    exact = {"ta": [{"source_node_id": "1", "node_name": "TA", "node_type": "poi"}]}
    aliases = {"ta": [{"source_node_id": "2", "node_name": "台安", "node_type": "poi"}]}
    result = _entity_recall("ＴＡ", exact, aliases)
    assert result["match_type"] == "EXACT_MATCH"
    assert result["candidate_count"] == 1
    assert result["decision"] == "AUTO_EXACT_RECALL"


def test_g3_entity_recall_does_not_auto_merge_ambiguous_matches():
    result = _entity_recall(
        "甲",
        {"甲": [
            {"source_node_id": "1", "node_name": "甲", "node_type": "poi"},
            {"source_node_id": "2", "node_name": "甲", "node_type": "poi"},
        ]},
        {},
    )
    assert result["candidate_count"] == 2
    assert result["decision"] == "REVIEW_REQUIRED"
    assert normalize_entity_name(" 二郎庙 ") == "二郎庙"


def test_g3_joint_score_keeps_exact_match_review_only():
    entity = {
        "decision": "AUTO_EXACT_RECALL",
        "candidates": [{"target_node_id": "2"}],
    }
    result = _joint_resolution_score(
        entity,
        [],
        evidence_score=0.9,
        graph_score=1.0,
        subject_node_id="1",
        nodes_by_id={"1": {"parent_source_node_id": "2"}, "2": {}},
    )
    assert result["rerank_decision"] == "REVIEW_READY_DETERMINISTIC"
    assert result["policy"] == "RECALL_ONLY_HUMAN_REVIEW"


def test_g3_joint_score_does_not_promote_vector_only_match():
    result = _joint_resolution_score(
        {"decision": "NO_GRAPH_MATCH", "candidates": []},
        [{"distance": 0.1}],
        evidence_score=0.9,
        graph_score=0.0,
        subject_node_id="1",
        nodes_by_id={"1": {}},
    )
    assert result["rerank_decision"] != "REVIEW_READY_DETERMINISTIC"
    assert result["joint_score"] < 0.45
