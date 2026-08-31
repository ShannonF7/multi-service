"""GrowthRun 详情投影的回归测试。

验证候选统一事实键和谱系摘要会保留，避免页面无法解释候选来源。
"""
from src.semantic_growth.api import _compact_growth_detail


def test_compact_detail_preserves_claim_identity_and_lineage_summary():
    """轻量详情仍应携带 canonical key、谱系数量和证据绑定 ID。"""
    detail = {
        "candidates": [{
            "id": 7,
            "candidate_uid": "candidate-7",
            "run_id": "growth-test",
            "source_node_id": "node-1",
            "subject_name": "主体",
            "subject_type": "POI",
            "claim_type": "property",
            "candidate_type": "discovered_fact",
            "predicate": "开放时间",
            "object_value": "08:00",
            "status": "PENDING",
            "canonical_claim_key": "claim-key-7",
            "conflict_scope_key": "scope-key-7",
            "aggregation_key": "claim-key-7",
            "growth_lineage": [{
                "raw_claim_id": 11,
                "evidence_unit_id": 22,
                "source_independence_key": "document:doc-1",
                "support_role": "SUPPORTS",
                "evidence_score": 0.88,
            }],
        }],
        "evidence_bindings": [],
        "graph": {},
    }
    result = _compact_growth_detail(detail)
    candidate = result["candidates"][0]
    assert candidate["canonical_claim_key"] == "claim-key-7"
    assert candidate["lineage_count"] == 1
    assert candidate["lineage"] == [{
        "raw_claim_id": 11,
        "evidence_unit_id": 22,
        "source_independence_key": "document:doc-1",
        "support_role": "SUPPORTS",
        "evidence_score": 0.88,
    }]


def test_compact_detail_keeps_image_binding_provenance():
    """图片证据投影不得丢失原图地址和 OCR 定位信息。"""
    detail = {
        "candidates": [],
        "evidence_bindings": [{
            "source_type": "image_asset",
            "source_title": "寺庙照片",
            "source_url": "https://example.test/a.jpg",
            "evidence_content": "图片中的牌匾",
            "evidence_metadata": {
                "asset_id": 9,
                "page_no": 2,
                "ocr_raw_text": "魁星楼",
                "ocr_blocks": [{"text": "魁星楼", "bbox": [1, 2, 3, 4]}],
            },
        }],
        "graph": {},
    }
    row = _compact_growth_detail(detail)["evidence_bindings"][0]
    assert row["asset_id"] == 9
    assert row["image_url"] == "https://example.test/a.jpg"
    assert row["page_no"] == 2
    assert row["ocr_blocks"][0]["bbox"] == [1, 2, 3, 4]
