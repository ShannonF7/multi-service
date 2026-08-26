from src.rag.service import graph_sync_service


class _Result:
    def consume(self):
        return None


class _Transaction:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return _Result()


class _Session:
    def __init__(self, transaction):
        self.transaction = transaction

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute_write(self, callback):
        callback(self.transaction)


class _Driver:
    def __init__(self, transaction):
        self.transaction = transaction

    def session(self, **kwargs):
        return _Session(self.transaction)


def test_incremental_projection_deletes_explicitly_removed_nodes(monkeypatch):
    transaction = _Transaction()
    monkeypatch.setattr(graph_sync_service, "_ensure_neo4j_schema", lambda: None)
    monkeypatch.setattr(graph_sync_service, "_neo4j_driver", lambda: _Driver(transaction))

    result = graph_sync_service.project_graph_snapshot(
        {
            "event_key": "delete-node-2073",
            "domain_id": "16",
            "domain": {"id": "16", "name": "Moon"},
            "replace_domain": False,
            "snapshot_node_ids": [],
            "deleted_node_ids": [2073],
            "nodes": [],
            "properties": [],
            "relations": [],
        }
    )

    delete_calls = [
        params
        for query, params in transaction.calls
        if "DETACH DELETE n" in query and "projection_key IN $keys" in query
    ]
    assert delete_calls == [{"keys": ["16:2073"]}]
    assert result["deleted_node_count"] == 1