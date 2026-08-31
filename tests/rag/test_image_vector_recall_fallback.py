from src.semantic_growth import candidate_aggregation_service as service


def test_image_recall_is_available_when_formal_text_index_is_empty(monkeypatch):
    monkeypatch.setattr(
        service,
        "_rerank_entity_candidates",
        lambda query, candidates, **kwargs: candidates,
    )
    monkeypatch.setattr(
        "src.rag.service.image_embedding_service.recall_image_asset_nodes",
        lambda source_scenic_id, asset_id, limit=5: [
            {
                "node_id": "node-image",
                "name": "图片对应节点",
                "node_type": "building",
                "final_score": 0.88,
                "recall_method": "IMAGE_VECTOR_RECALL",
            }
        ],
    )
    result = service._vector_entity_recall(
        "4",
        "图片中的实体",
        [],
        image_asset_id=77,
        limit=5,
    )
    assert result[0]["node_id"] == "node-image"
    assert result[0]["recall_method"] == "IMAGE_VECTOR_RECALL"
