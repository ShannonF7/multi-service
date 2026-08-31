"""P0 页面隔离回归测试：节点分组接口不得默认读取 GrowthRun 候选。"""

import asyncio

from src.rag import semantic_router


def test_node_group_defaults_to_targeted_completion(monkeypatch):
    """带节点范围且未指定来源时，路由必须补上定向补全过滤。"""
    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return {"items": [], "total": 0}

    monkeypatch.setattr(semantic_router, "list_semantic_candidate_groups", fake_list)
    result = asyncio.run(
        semantic_router.semantic_candidate_groups(
            source_node_id="2144", discovery_track=None
        )
    )
    assert result == {"items": [], "total": 0}
    assert captured["discovery_track"] == "TARGETED_COMPLETION"


def test_explicit_growth_group_filter_is_preserved(monkeypatch):
    """管理查询显式要求开放发现时，路由不得覆盖调用方选择。"""
    captured = {}

    def fake_list(**kwargs):
        captured.update(kwargs)
        return {"items": [], "total": 0}

    monkeypatch.setattr(semantic_router, "list_semantic_candidate_groups", fake_list)
    asyncio.run(
        semantic_router.semantic_candidate_groups(
            source_node_id="2144", discovery_track="OPEN_DISCOVERY"
        )
    )
    assert captured["discovery_track"] == "OPEN_DISCOVERY"
