"""图片 OCR 证据留痕和证据字段透传测试。"""

from src.rag.schemas import SemanticEvidenceInput
from src.rag.service import image_ocr_service
from src.semantic_growth.paddle_ocr_http_service import _bbox, _parse
from src.semantic_growth import evidence as growth_evidence


def test_bbox_normalizes_polygon():
    """四点框应归一化为可用于前端高亮的矩形框。"""
    assert _bbox([[1, 2], [11, 2], [11, 12], [1, 12]]) == [1.0, 2.0, 11.0, 12.0]


def test_parse_preserves_raw_text_and_blocks():
    """低分行不进入清洗文本，但原文和高分框必须可追溯。"""
    result = _parse(
        {
            "rec_texts": ["魁星楼", "噪声"],
            "rec_scores": [0.95, 0.20],
            "rec_boxes": [
                [[1, 2], [11, 2], [11, 12], [1, 12]],
                [20, 20, 30, 30],
            ],
        }
    )
    assert result["ocr_text"] == "魁星楼"
    assert result["ocr_raw_text"] == "魁星楼\n噪声"
    assert result["ocr_blocks"] == [
        {"text": "魁星楼", "score": 0.95, "bbox": [1.0, 2.0, 11.0, 12.0], "order": 0}
    ]
    assert result["model"] == "paddleocr"


def test_evidence_input_keeps_image_provenance():
    """图片证据字段应在统一输入模型中保留。"""
    evidence = SemanticEvidenceInput(
        content="魁星楼",
        source_type="image_asset",
        source_doc_id="doc-1",
        chunk_id=3,
        page_no=2,
        asset_id=9,
        caption="图片标题",
        nearby_text="图下注释",
        bbox=[1, 2, 11, 12],
        ocr_blocks=[{"text": "魁星楼", "score": 0.9, "bbox": [1, 2, 11, 12]}],
        metadata={"ocr_model": "paddleocr"},
    )
    data = evidence.model_dump()
    assert data["page_no"] == 2
    assert data["asset_id"] == 9
    assert data["ocr_blocks"][0]["bbox"] == [1, 2, 11, 12]
    assert data["metadata"]["ocr_model"] == "paddleocr"


def test_process_image_ocr_urls_returns_blocks(monkeypatch):
    """未入库图片 URL 的预览 OCR 也应返回框和原始文本。"""
    monkeypatch.setattr(
        image_ocr_service,
        "extract_ocr_batch",
        lambda items: {
            9: {
                "status": "ok",
                "ocr_text": "魁星楼",
                "ocr_raw_text": "魁星楼",
                "ocr_blocks": [{"text": "魁星楼", "score": 0.9, "bbox": [1, 2, 3, 4]}],
                "model": "paddleocr",
            }
        },
    )
    result = image_ocr_service.process_image_ocr_urls(
        [{"asset_id": 9, "image": "/tmp/test.png"}]
    )
    item = result["items"][0]
    assert item["ocr_raw_text"] == "魁星楼"
    assert item["ocr_blocks"][0]["bbox"] == [1, 2, 3, 4]


def test_prepare_image_ocr_respects_zero_limit(monkeypatch):
    """明确关闭图片批量时不得偷偷识别图片。"""
    calls = []
    monkeypatch.setattr(
        growth_evidence,
        "process_image_ocr_batch",
        lambda **kwargs: calls.append(kwargs),
    )
    growth_evidence._prepare_image_ocr("4", 0)
    assert calls == []


def test_parse_accepts_numpy_boxes():
    """NumPy 框数组不得因布尔判断而解析失败。"""
    np = __import__("numpy")
    result = _parse({
        "rec_texts": ["牌匾"],
        "rec_scores": [0.9],
        "rec_boxes": np.array([[1, 2, 3, 4]]),
    })
    assert result["ocr_blocks"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_attach_image_context_uses_existing_chunk_only():
    """图片上下文只透传已有 chunk，缺失时保持空值。"""
    rows = growth_evidence._attach_image_context([
        {
            "asset_type": "image",
            "metadata": {"doc_id": "doc-1"},
            "nearby_text": "同页说明",
            "nearby_section": "建筑介绍",
        },
        {"asset_type": "image", "metadata": {}},
    ])
    assert rows[0]["metadata"]["nearby_text"] == "同页说明"
    assert rows[0]["metadata"]["section"] == "建筑介绍"
    assert "nearby_text" not in rows[1]["metadata"]
