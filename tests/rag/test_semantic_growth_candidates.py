from src.semantic_growth import candidate_extraction as module


class _Claim:
    def __init__(self, candidate_id):
        self.candidate_id = candidate_id


class _Response:
    candidate_claims = [_Claim(123)]


def test_growth_extraction_groups_mentions_and_keeps_open_payload(monkeypatch):
    captured = {}

    async def fake_complete(payload, *, trace_id_override=None):
        captured["payload"] = payload
        captured["trace"] = trace_id_override
        return _Response()

    monkeypatch.setattr(module, "complete_semantic_service", fake_complete)
    monkeypatch.setattr(
        module,
        "filter_published_candidate_ids",
        lambda scenic_id, ids: {"duplicate_ids": [], "kept_ids": ids},
    )
    results = module.extract_growth_candidates(
        source_scenic_id="4",
        growth_run_id="growth-test",
        batch=[{
            "consumption_id": 1,
            "content": "太原理工大学是一所大学。",
            "title": "证据",
            "source_url": "domain-kb://4/x/1",
            "evidence_text": "太原理工大学是一所大学。",
        }],
        mentions=[{
            "consumption_id": 1,
            "node_id": "2144",
            "node_name": "太原理工大学",
            "node_type": "poi",
        }],
        published_nodes=[{
            "node_id": "2144",
            "name": "太原理工大学",
            "node_type": "poi",
        }],
    )
    assert results[0]["candidate_ids"] == [123]
    assert captured["payload"].target_fields == []
    assert captured["payload"].relation_intents == []
    assert captured["payload"].metadata["source_scope"] == ["provided_evidence"]


def test_numeric_node_name_does_not_match_larger_number():
    from src.semantic_growth.evidence import extract_mentions_from_batch

    mentions = extract_mentions_from_batch(
        [{
            "id": 1,
            "consumption_id": 1,
            "source_scenic_id": "4",
            "content": "postal 030600, room 306.",
        }],
        [{"node_id": "1469", "name": "306", "node_type": "POI"}],
    )
    assert len(mentions) == 1
    assert mentions[0]["mention_text"] == "306"
