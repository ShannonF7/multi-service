from src.semantic_growth.evidence import extract_mentions_from_batch, source_cursor_progress

def test_exact_mentions_prefer_longest_name():
    batch = [{
        "consumption_id": 1,
        "chunk_id": 22276,
        "source_scenic_id": "4",
        "content": "太原理工大学东门位于校园东侧。",
    }]
    nodes = [
        {"node_id": "a", "name": "东门", "node_type": "POI"},
        {"node_id": "b", "name": "太原理工大学", "node_type": "School"},
    ]
    items = extract_mentions_from_batch(batch, nodes)
    assert [item["node_id"] for item in items] == ["b", "a"]


def test_duplicate_exact_name_is_not_auto_aligned():
    batch = [{
        "consumption_id": 2,
        "chunk_id": 22278,
        "source_scenic_id": "4",
        "content": "实验室正在建设。",
    }]
    nodes = [
        {"node_id": "a", "name": "实验室", "node_type": "POI"},
        {"node_id": "b", "name": "实验室", "node_type": "POI"},
    ]
    assert extract_mentions_from_batch(batch, nodes) == []


def test_image_asset_binding_aligns_even_when_node_is_outside_overview_page():
    batch = [{
        "consumption_id": 3,
        "chunk_id": 223,
        "source_scenic_id": "4",
        "source_node_id": "999",
        "source_node_name": "远端绑定节点",
        "source_node_type": "POI",
        "asset_type": "image",
        "content": "图片中的 OCR 文本",
    }]
    items = extract_mentions_from_batch(batch, [])
    assert items[0]["node_id"] == "999"
    assert items[0]["match_method"] == "ASSET_NODE_BINDING"


def test_source_cursor_advances_only_when_every_scope_processed():
    assert source_cursor_progress(["PROCESSED", "PROCESSED"]) == {
        "expected_scope_count": 2,
        "processed_scope_count": 2,
        "cursor_state": "ADVANCED",
    }
    assert source_cursor_progress(["PROCESSED", "FAILED"])["cursor_state"] == "OPEN"
    assert source_cursor_progress(["PROCESSED", "RETRYABLE"])["cursor_state"] == "OPEN"
    assert source_cursor_progress(["PROCESSED", "CLAIMED"])["cursor_state"] == "OPEN"
    assert source_cursor_progress([])["cursor_state"] == "OPEN"


def test_finalize_fanout_keeps_cursor_open_until_failed_scope_succeeds(monkeypatch):
    from contextlib import contextmanager
    from uuid import uuid4

    from sqlalchemy import text
    from sqlalchemy.orm import Session

    from src.rag.dependencies import get_ai_engine
    import src.semantic_growth.evidence as evidence_module

    connection = get_ai_engine().connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    @contextmanager
    def rollback_scope():
        yield session
        session.flush()

    monkeypatch.setattr(evidence_module, "ai_session_scope", rollback_scope)
    source_id = f"codex-p0-{uuid4().hex}"
    identity = {
        "source_scenic_id": "__codex_test__",
        "source_id": source_id,
        "chunk_id": 900000001,
        "chunk_hash": uuid4().hex,
        "consumer_version": "growth-g1-p0-test",
    }
    try:
        cursor_id = session.execute(
            text(
                """
                insert into semantic_growth_source_cursors (
                    source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version
                ) values (
                    :source_scenic_id, :source_id, :chunk_id, :chunk_hash, :consumer_version
                ) returning id
                """
            ),
            identity,
        ).scalar_one()
        consumption_ids = {}
        for scope in (evidence_module.CHUNK_SCOPE, "node-a", "node-b"):
            consumption_ids[scope] = session.execute(
                text(
                    """
                    insert into semantic_growth_evidence_consumptions (
                        growth_run_id, source_scenic_id, source_id, chunk_id, chunk_hash,
                        consumer_version, target_scope, state, lease_owner
                    ) values (
                        'codex-p0-test', :source_scenic_id, :source_id, :chunk_id, :chunk_hash,
                        :consumer_version, :target_scope, 'CLAIMED', 'codex-p0-worker'
                    ) returning id
                    """
                ),
                {**identity, "target_scope": scope},
            ).scalar_one()
        batch = [{
            **identity,
            "id": identity["chunk_id"],
            "content_hash": identity["chunk_hash"],
            "consumption_id": consumption_ids[evidence_module.CHUNK_SCOPE],
        }]
        first = evidence_module.finalize_evidence_batch(
            batch=batch,
            results=[
                {"node_id": "node-a", "candidate_ids": [1], "error": None},
                {"node_id": "node-b", "candidate_ids": [], "error": "worker failed"},
            ],
            worker_id="codex-p0-worker",
            consumer_version=identity["consumer_version"],
        )
        assert first[0]["cursor_state"] == "OPEN"
        assert session.execute(
            text("select cursor_state from semantic_growth_source_cursors where id=:id"),
            {"id": cursor_id},
        ).scalar_one() == "OPEN"

        second = evidence_module.finalize_evidence_batch(
            batch=batch,
            results=[
                {"node_id": "node-a", "candidate_ids": [1], "error": None},
                {"node_id": "node-b", "candidate_ids": [], "error": None},
            ],
            worker_id="codex-p0-worker",
            consumer_version=identity["consumer_version"],
        )
        assert second[0] == {
            "consumption_id": consumption_ids[evidence_module.CHUNK_SCOPE],
            "expected_scope_count": 3,
            "processed_scope_count": 3,
            "cursor_state": "ADVANCED",
        }
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
