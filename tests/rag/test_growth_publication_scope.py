"""P5 发布受影响节点提取的回归测试。"""

from src.semantic_growth.repository import _normalize_affected_node_ids


def test_explicit_affected_node_ids_are_deduplicated_in_order():
    assert _normalize_affected_node_ids(
        [{"node_id": "scope-1"}],
        ["n-2", "n-1", "n-2", "", None],
    ) == ["n-2", "n-1"]


def test_affected_node_ids_fallback_to_scope_node_ids():
    assert _normalize_affected_node_ids(
        [
            {"node_id": "n-1", "role": "candidate_source"},
            {"node_id": "n-1", "role": "parent_context"},
            {"source_node_id": "n-2"},
            {"name": "ignored"},
        ]
    ) == ["n-1", "n-2"]


def test_empty_scope_returns_empty_ids():
    assert _normalize_affected_node_ids([]) == []
