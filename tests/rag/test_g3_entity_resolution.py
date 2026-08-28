"""G3 实体统一决策回归测试：验证精确、向量、歧义和新实体分支。"""

from src.semantic_growth import candidate_aggregation_service as service


def _node(node_id="n1", name="张壁古堡", node_type="poi"):
    return {"source_node_id": node_id, "node_name": name, "node_type": node_type, "parent_source_node_id": "root"}


def test_exact_name_resolution_does_not_call_vector_model():
    """精确名称命中时直接复用正式节点，不触发向量召回。"""
    result = service._resolve_name(
        None,
        name="张壁古堡",
        raw_type="poi",
        source_scenic_id="4",
        growth_run_id="g-test",
        evidence_unit_ids=[1],
        confidence=0.9,
        exact={"张壁古堡": [_node()]},
        aliases={},
        published_nodes=[_node()],
    )
    assert result["status"] == "EXACT"
    assert result["node_id"] == "n1"


def test_vector_match_requires_score_and_margin(monkeypatch):
    """向量结果同时满足分数和 Top1/Top2 间隔才允许语义合并。"""
    monkeypatch.setattr(
        service,
        "_vector_entity_recall",
        lambda *args, **kwargs: [
            {"node_id": "n1", "name": "张壁古堡", "node_type": "poi", "rank_score": 0.92, "final_score": 0.92},
            {"node_id": "n2", "name": "王家大院", "node_type": "poi", "rank_score": 0.70, "final_score": 0.70},
        ],
    )
    result = service._resolve_name(
        None, name="张壁古堡景区", raw_type="poi", source_scenic_id="4", growth_run_id="g-test",
        evidence_unit_ids=[1], confidence=0.8, exact={}, aliases={}, published_nodes=[_node(), _node("n2", "王家大院")],
    )
    assert result["status"] == "SEMANTIC_MATCH"
    assert result["node_id"] == "n1"
    assert result["vector_margin"] == 0.22


def test_vector_near_tie_is_ambiguous(monkeypatch):
    """近似并列候选必须进入歧义审核，不能自动合并。"""
    monkeypatch.setattr(
        service,
        "_vector_entity_recall",
        lambda *args, **kwargs: [
            {"node_id": "n1", "name": "南堡门", "node_type": "poi", "rank_score": 0.82, "final_score": 0.82},
            {"node_id": "n2", "name": "北堡门", "node_type": "poi", "rank_score": 0.79, "final_score": 0.79},
        ],
    )
    result = service._resolve_name(
        None, name="堡门", raw_type="poi", source_scenic_id="4", growth_run_id="g-test",
        evidence_unit_ids=[1], confidence=0.8, exact={}, aliases={}, published_nodes=[_node("n1", "南堡门"), _node("n2", "北堡门")],
    )
    assert result["status"] == "AMBIGUOUS"
    assert result["node_id"] is None


def test_low_vector_score_mints_reviewable_entity_candidate(monkeypatch):
    """低向量分数不能强行匹配，应生成可审核的新实体候选。"""
    monkeypatch.setattr(
        service,
        "_vector_entity_recall",
        lambda *args, **kwargs: [{"node_id": "n1", "name": "旧址", "node_type": "poi", "rank_score": 0.55, "final_score": 0.55}],
    )
    monkeypatch.setattr(service, "_persist_node_candidate", lambda *args, **kwargs: 99)
    result = service._resolve_name(
        None, name="新发现遗址", raw_type="poi", source_scenic_id="4", growth_run_id="g-test",
        evidence_unit_ids=[1], confidence=0.8, exact={}, aliases={}, published_nodes=[_node("n1", "旧址")],
    )
    assert result["status"] == "NEW_ENTITY"
    assert result["node_candidate_id"] == 99
