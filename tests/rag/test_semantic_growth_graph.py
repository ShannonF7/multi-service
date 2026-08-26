from langgraph.graph import END
from src.semantic_growth import graph as growth_graph


def test_no_candidate_run_is_no_change(monkeypatch):
    statuses = []
    monkeypatch.setattr(growth_graph, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(growth_graph, "set_run_status", lambda *args, **kwargs: statuses.append(args))
    result = growth_graph.aggregate_results({
        "growth_run_id": "run-no-change",
        "opportunities": [],
        "candidate_ids": [],
        "processed_opportunity_ids": [],
        "failed_opportunity_ids": [],
        "opportunity_results": [],
    })
    assert result["review_status"] == "not_required"
    assert result["stop_reason"] == "no_candidates_generated"
    assert statuses[-1][1:3] == ("NO_CHANGE", "no_candidates_generated")


def test_all_worker_failures_are_failed(monkeypatch):
    statuses = []
    monkeypatch.setattr(growth_graph, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(growth_graph, "set_run_status", lambda *args, **kwargs: statuses.append((args, kwargs)))
    result = growth_graph.aggregate_results({
        "growth_run_id": "run-failed",
        "opportunities": [{"opportunity_id": "a"}, {"opportunity_id": "b"}],
        "candidate_ids": [],
        "processed_opportunity_ids": [],
        "failed_opportunity_ids": ["a", "b"],
        "opportunity_results": [
            {"opportunity_id": "a", "status": "failed"},
            {"opportunity_id": "b", "status": "failed"},
        ],
    })
    assert result["review_status"] == "not_required"
    assert result["stop_reason"] == "all_workers_failed"
    assert statuses[-1][0][1:3] == ("FAILED", "all_workers_failed")
    assert statuses[-1][1]["status_reason_code"] == "ALL_WORKERS_FAILED"


def test_partial_worker_failure_keeps_reviewable_candidates(monkeypatch):
    statuses = []
    monkeypatch.setattr(growth_graph, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(growth_graph, "set_run_status", lambda *args, **kwargs: statuses.append((args, kwargs)))
    result = growth_graph.aggregate_results({
        "growth_run_id": "run-partial",
        "opportunities": [{"opportunity_id": "ok"}, {"opportunity_id": "bad"}],
        "candidate_ids": ["101"],
        "processed_opportunity_ids": ["ok"],
        "failed_opportunity_ids": ["bad"],
        "opportunity_results": [
            {"opportunity_id": "ok", "status": "candidate_generated", "candidate_ids": ["101"]},
            {"opportunity_id": "bad", "status": "failed", "candidate_ids": []},
        ],
    })
    assert result["review_status"] == "waiting"
    assert result["final_candidate_ids"] == ["101"]
    assert statuses[-1][0][1:3] == ("WAITING_REVIEW", "partial_worker_failure")
    assert statuses[-1][1]["status_reason_code"] == "WAITING_REVIEW_WITH_WARNINGS"


def test_dispatch_stops_frozen_legacy_path():
    assert growth_graph.dispatch_opportunities({"stop_reason": "g1_not_implemented"}) is END


def test_g2_candidate_batch_enters_waiting_review(monkeypatch):
    statuses = []
    monkeypatch.setattr(growth_graph, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(growth_graph, "set_run_status", lambda *args, **kwargs: statuses.append((args, kwargs)))
    result = growth_graph.aggregate_results({
        "growth_run_id": "run-g2",
        "evidence_batch": [{"consumption_id": 1}],
        "mention_batch": [{"node_id": "n1"}],
        "alignment_batch": [{"node_id": "n1"}],
        "candidate_ids": ["101", "102"],
        "candidate_results": [{"node_id": "n1", "candidate_ids": [101, 102], "error": None}],
    })
    assert result["review_status"] == "waiting"
    assert result["final_candidate_ids"] == ["101", "102"]
    assert statuses[-1][0][1:3] == ("WAITING_REVIEW", "g2_candidates_ready")


def test_evidence_batch_route_continues_until_budget():
    assert growth_graph.route_after_evidence_batch({
        "last_batch_count": 2,
        "evidence_processed_count": 2,
        "max_evidence_per_run": 6,
    }) == "load_evidence_batch"
    assert growth_graph.route_after_evidence_batch({
        "last_batch_count": 2,
        "evidence_processed_count": 6,
        "max_evidence_per_run": 6,
    }) == "aggregate_results"
    assert growth_graph.route_after_evidence_batch({
        "last_batch_count": 0,
        "evidence_processed_count": 2,
        "max_evidence_per_run": 6,
    }) == "aggregate_results"


def test_evidence_aggregate_sums_all_batch_summaries(monkeypatch):
    statuses = []
    monkeypatch.setattr(growth_graph, "record_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(growth_graph, "set_run_status", lambda *args, **kwargs: statuses.append((args, kwargs)))
    result = growth_graph.aggregate_results({
        "growth_run_id": "run-multi-batch",
        "evidence_batch": [],
        "candidate_ids": ["101", "102"],
        "batch_summaries": [
            {"evidence_count": 2, "mention_count": 3, "aligned_count": 3, "candidate_count": 1},
            {"evidence_count": 2, "mention_count": 4, "aligned_count": 4, "candidate_count": 1},
        ],
    })
    assert result["result_summary"]["evidence_count"] == 4
    assert result["result_summary"]["mention_count"] == 7
    assert result["result_summary"]["batch_count"] == 2
    assert result["final_candidate_ids"] == ["101", "102"]
    assert statuses[-1][0][1] == "WAITING_REVIEW"
