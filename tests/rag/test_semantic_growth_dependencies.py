from src.semantic_growth.dependencies import _dependency_state


def test_g5_dependency_state_flow():
    assert _dependency_state("PENDING") == "BLOCKED_BY_DEPENDENCY"
    assert _dependency_state("ADOPTED") == "PENDING"
    assert _dependency_state("REJECTED") == "INVALIDATED"


def test_g5_new_entity_dependency_blocks_until_entity_review():
    assert _dependency_state("PENDING", node_candidate=True) == "BLOCKED_BY_DEPENDENCY"
    assert _dependency_state("ADOPTED", node_candidate=True) == "PENDING"
    assert _dependency_state("REJECTED", node_candidate=True) == "INVALIDATED"
