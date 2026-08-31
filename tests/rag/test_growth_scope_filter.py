"""G6 下一轮受影响节点范围过滤测试。"""

from src.semantic_growth.graph import _restrict_scope_to_seed_nodes


def test_seed_nodes_limit_next_round_scope():
    nodes = [
        {"node_id": "n-1", "name": "节点一"},
        {"node_id": "n-2", "name": "节点二"},
        {"node_id": "n-3", "name": "节点三"},
    ]
    assert _restrict_scope_to_seed_nodes(nodes, ["n-3", "n-1"]) == [
        {"node_id": "n-1", "name": "节点一"},
        {"node_id": "n-3", "name": "节点三"},
    ]


def test_empty_seed_keeps_full_scope():
    nodes = [{"node_id": "n-1"}, {"node_id": "n-2"}]
    assert _restrict_scope_to_seed_nodes(nodes, []) == nodes


def test_source_node_id_is_supported_for_database_projection():
    nodes = [{"source_node_id": "n-1"}, {"source_node_id": "n-2"}]
    assert _restrict_scope_to_seed_nodes(nodes, ["n-2"]) == [{"source_node_id": "n-2"}]
