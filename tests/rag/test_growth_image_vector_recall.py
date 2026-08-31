from src.semantic_growth import candidate_aggregation_service as aggregation


def test_merge_image_and_text_recall_deduplicates_node_and_keeps_methods():
    result = aggregation._merge_entity_recall_candidates(
        [
            {
                "node_id": "n1",
                "name": "节点一",
                "score": 0.81,
                "final_score": 0.81,
                "recall_method": "TEXT_VECTOR_RECALL",
            },
            {
                "node_id": "n1",
                "name": "节点一",
                "score": 0.88,
                "final_score": 0.88,
                "image_vector_score": 0.88,
                "recall_method": "IMAGE_VECTOR_RECALL",
            },
            {
                "node_id": "n2",
                "name": "节点二",
                "score": 0.70,
                "final_score": 0.70,
                "recall_method": "IMAGE_VECTOR_RECALL",
            },
        ],
        limit=5,
    )
    assert [item["node_id"] for item in result] == ["n1", "n2"]
    assert result[0]["recall_methods"] == ["TEXT_VECTOR_RECALL", "IMAGE_VECTOR_RECALL"]
    assert result[0]["image_vector_score"] == 0.88


def test_resolve_name_passes_image_asset_to_recall(monkeypatch):
    captured = {}

    def fake_recall(*_args, **kwargs):
        captured["image_asset_id"] = kwargs.get("image_asset_id")
        return [
            {
                "node_id": "n1",
                "name": "节点一",
                "node_type": "building",
                "final_score": 0.90,
                "rank_score": 0.90,
            },
        ]

    monkeypatch.setattr(aggregation, "_vector_entity_recall", fake_recall)
    monkeypatch.setattr(aggregation, "_VECTOR_MIN_SCORE", 0.80)
    result = aggregation._resolve_name(
        object(),
        name="节点一别名",
        raw_type="building",
        source_scenic_id="4",
        growth_run_id="growth-test",
        evidence_unit_ids=[1],
        confidence=0.9,
        exact={},
        aliases={},
        published_nodes=[{"source_node_id": "n1", "node_name": "节点一", "node_type": "building"}],
        image_asset_id=77,
    )
    assert captured["image_asset_id"] == 77
    assert result["status"] == "SEMANTIC_MATCH"
    assert result["node_id"] == "n1"
