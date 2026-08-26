from src.semantic_growth.candidate_extraction import build_growth_payload
from src.semantic_growth.evidence import extract_mentions_from_batch


def test_image_asset_uses_existing_node_binding_without_name_hallucination():
    mentions = extract_mentions_from_batch(
        [
            {
                "id": 77,
                "consumption_id": 88,
                "source_scenic_id": "4",
                "source_node_id": "node-1",
                "asset_type": "image",
                "asset_id": 77,
                "content": "碑刻正面，落款为某年。",
            }
        ],
        [{"node_id": "node-1", "name": "二郎庙", "node_type": "building"}],
    )
    assert len(mentions) == 1
    assert mentions[0]["match_method"] == "ASSET_NODE_BINDING"
    assert mentions[0]["node_name"] == "二郎庙"


def test_image_asset_payload_keeps_ocr_text_and_image_provenance():
    payload = build_growth_payload(
        source_scenic_id="4",
        growth_run_id="growth-test",
        node={"node_id": "node-1", "name": "二郎庙", "node_type": "building"},
        chunks=[
            {
                "id": 77,
                "source_id": "asset:77",
                "source_node_id": "node-1",
                "asset_type": "image",
                "asset_id": 77,
                "image_url": "https://example.test/77.jpg",
                "title": "二郎庙匾额",
                "content": "二郎庙匾额，落款为某年。",
                "evidence_text": "二郎庙匾额，落款为某年。",
                "source_url": "https://example.test/77.jpg",
                "metadata": {"ocr_max_score": 0.98},
            }
        ],
    )
    evidence = payload.evidence[0]
    assert evidence.source_type == "image_asset"
    assert evidence.image == "https://example.test/77.jpg"
    assert evidence.asset_id == 77
    assert evidence.source_doc_id == "asset:77"
    assert evidence.score == 0.75
    assert payload.metadata["provenance_type"] == "image_asset"
    assert payload.metadata["adopted_context_is_evidence"] is False
