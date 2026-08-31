"""图片资产绑定目标校验的单元测试。"""

from src.rag.service.asset_binding_service import validate_asset_binding_target


def test_existing_node_target_is_allowed():
    result = validate_asset_binding_target(
        {"source_asset_id": "img-1", "source_node_id": "node-7"},
        {"node-7"},
    )
    assert result == {
        "allowed": True,
        "reason": "OK",
        "asset_id": "img-1",
        "target_node_id": "node-7",
    }


def test_missing_target_node_is_rejected():
    result = validate_asset_binding_target(
        {"source_asset_id": "img-1", "source_node_id": "node-new"},
        {"node-7"},
    )
    assert result["allowed"] is False
    assert result["reason"] == "TARGET_NODE_NOT_FOUND"


def test_invalid_binding_fields_have_stable_reasons():
    assert validate_asset_binding_target({}, set())["reason"] == "ASSET_ID_REQUIRED"
    assert validate_asset_binding_target(
        {"source_asset_id": "img-1"},
        set(),
    )["reason"] == "TARGET_NODE_ID_REQUIRED"
    assert validate_asset_binding_target(
        {"source_asset_id": "img-1", "source_node_id": "node-7", "object_type": "edge"},
        {"node-7"},
    )["reason"] == "UNSUPPORTED_OBJECT_TYPE"
