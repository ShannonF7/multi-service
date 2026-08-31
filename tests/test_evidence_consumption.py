"""证据消费游标的纯逻辑回归测试。

这些测试只调用无数据库副作用的进度计算函数，不写入生产表。
"""

from src.semantic_growth.evidence import source_cursor_progress


def test_all_scopes_processed_advances_cursor():
    """全部分支成功后才允许推进游标。"""
    result = source_cursor_progress(["PROCESSED", "processed"])
    assert result == {
        "expected_scope_count": 2,
        "processed_scope_count": 2,
        "cursor_state": "ADVANCED",
    }


def test_retryable_scope_keeps_cursor_open():
    """可重试分支未完成时必须保持 OPEN。"""
    result = source_cursor_progress(["PROCESSED", "RETRYABLE"])
    assert result["processed_scope_count"] == 1
    assert result["cursor_state"] == "OPEN"


def test_failed_scope_keeps_cursor_open():
    """失败分支不能被当成已消费。"""
    result = source_cursor_progress(["PROCESSED", "FAILED"])
    assert result["processed_scope_count"] == 1
    assert result["cursor_state"] == "OPEN"


def test_claimed_scope_keeps_cursor_open():
    """仍在租约中的分支尚未完成，游标不能推进。"""
    result = source_cursor_progress(["CLAIMED"])
    assert result["processed_scope_count"] == 0
    assert result["cursor_state"] == "OPEN"


def test_empty_scope_list_is_not_complete():
    """没有任何分支不代表证据消费完成。"""
    result = source_cursor_progress([])
    assert result == {
        "expected_scope_count": 0,
        "processed_scope_count": 0,
        "cursor_state": "OPEN",
    }


def test_unknown_scope_state_is_not_success():
    """未知状态按未完成处理，防止新状态静默推进游标。"""
    result = source_cursor_progress(["PROCESSED", "UNKNOWN"])
    assert result["processed_scope_count"] == 1
    assert result["cursor_state"] == "OPEN"


def test_text_chunks_share_document_source_family():
    """同一文档的不同 chunk 必须归入同一来源族。"""
    from src.semantic_growth.evidence import _source_family_id

    first = _source_family_id({"asset_type": "text", "source_id": "doc-1", "id": 1})
    second = _source_family_id({"asset_type": "text", "source_id": "doc-1", "id": 2})
    assert first == second == "document:doc-1"


def test_image_assets_have_distinct_source_families():
    """不同图片资产不能被错误合并成同一文本来源。"""
    from src.semantic_growth.evidence import _source_family_id

    first = _source_family_id({"asset_type": "image", "asset_id": 10, "source_id": "asset:10", "id": 10})
    second = _source_family_id({"asset_type": "image", "asset_id": 11, "source_id": "asset:11", "id": 11})
    assert first == "image:10"
    assert second == "image:11"
