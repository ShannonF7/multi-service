from __future__ import annotations

import json
from collections import Counter
from http import HTTPStatus
from types import SimpleNamespace

from src.rag.schemas import CandidateClaim, EvidenceChunk, SemanticCompleteRequest
from src.rag.service import semantic_completion_service as service
from src.rag.service.completion_job_service import _normalize_mode_payload
from src.rag.service.planner_service import plan_completion_questions


def make_payload(
    *,
    mode: str,
    target_fields: list[str],
    relation_intents: list[str] | None = None,
    node_type: str = "building",
    metadata: dict | None = None,
) -> SemanticCompleteRequest:
    values = {"completion_mode": mode, **(metadata or {})}
    return SemanticCompleteRequest.parse_obj(
        {
            "scenic_id": "4",
            "node": {
                "source_node_id": "946",
                "name": "测试节点",
                "node_type": node_type,
                "scenic_name": "测试领域",
            },
            "target_fields": target_fields,
            "relation_intents": relation_intents or [],
            "subgraph_depth": 1,
            "use_web_search": mode in {"deep", "batch"},
            "use_web_extractor": mode in {"deep", "batch"},
            "metadata": values,
        }
    )


def make_chunk(index: int, question_id: str, field: str) -> EvidenceChunk:
    return EvidenceChunk(
        source_id=f"S{index}",
        title=f"来源 {index}",
        content=("可核验证据内容。" * 500),
        quote=f"{field}的证据 {index}",
        source="test",
        source_type="provided",
        source_url=f"https://example.test/{index}",
        score=0.9,
        question_id=question_id,
        target_kind="property",
        target_field=field,
        final_evidence_score=0.9,
    )


def test_deep_and_batch_plan_all_questions_and_use_eight_as_batch_size():
    fields = [f"属性{i}" for i in range(1, 11)] + ["时期"]
    for mode in ("deep", "batch"):
        payload = make_payload(mode=mode, target_fields=fields, relation_intents=["包含", "位于"])
        normalized = SemanticCompleteRequest.parse_obj(_normalize_mode_payload(payload))
        questions = plan_completion_questions(normalized)

        assert len(questions) == 17
        assert len([item for item in questions if item.temporal_role]) == 5
        assert [len(batch) for batch in service.batch_completion_questions(normalized, questions)] == [8, 8, 1]


def test_quick_three_is_batch_size_not_total_limit():
    payload = make_payload(mode="quick", target_fields=[f"属性{i}" for i in range(1, 6)])
    normalized = SemanticCompleteRequest.parse_obj(_normalize_mode_payload(payload))
    questions = plan_completion_questions(normalized)

    assert len(questions) == 5
    assert [len(batch) for batch in service.batch_completion_questions(normalized, questions)] == [3, 2]

def test_quick_web_enables_page_extraction():
    payload = make_payload(mode="quick_web", target_fields=["职位"])
    normalized = _normalize_mode_payload(payload)

    assert normalized["use_web_search"] is True
    assert normalized["use_web_extractor"] is True
    assert "web_extractor" in normalized["metadata"]["source_scope"]

def test_web_search_uses_max_strategy_and_prefers_primary_sources(monkeypatch):
    captured = {}

    def fake_generation_call(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            status_code=HTTPStatus.OK,
            output=SimpleNamespace(
                search_info={"search_results": []},
                choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
            ),
        )

    monkeypatch.setattr(service.Generation, "call", fake_generation_call)
    results, _ = service.web_search("张思炯的职位是什么？", limit=5)

    assert results == []
    assert captured["search_options"]["search_strategy"] == "max"
    assert "优先官方网站、机构官网" in captured["messages"][0]["content"]


def test_title_only_search_result_is_not_sent_to_extractor():
    payload = make_payload(mode="quick_web", target_fields=["职位"])
    questions = plan_completion_questions(payload)
    question = questions[0]
    title_only = EvidenceChunk(
        source_id="S1",
        title="张思炯",
        content="张思炯",
        quote="张思炯",
        source="web-search",
        source_type="web_search",
        source_url="https://example.test/title-only",
        question_id=question.question_id,
    )
    full_page = EvidenceChunk(
        source_id="S2",
        title="张思炯",
        content="张思炯是研究员、博士生导师。",
        quote="张思炯是研究员、博士生导师。",
        source="web-page",
        source_type="web_extractor",
        source_url="https://example.test/full-page",
        question_id=question.question_id,
    )

    selected = service.select_extractor_chunks(payload, questions, [title_only, full_page])

    assert [chunk.source_id for chunk in selected] == ["S2"]


def test_subject_echo_property_is_rejected():
    chunk = EvidenceChunk(
        source_id="S1",
        title="张思炯简介",
        content="张思炯是研究员、博士生导师。",
        quote="张思炯是研究员、博士生导师。",
        source="web-page",
        source_type="web_extractor",
        source_url="https://example.test/profile",
    )
    claim = CandidateClaim(
        claim_id="c_001",
        claim_type="property",
        subject_node_id="1916",
        subject_name="张思炯",
        predicate="职位",
        object_value="张思炯",
        source_id="S1",
        quote="张思炯",
        confidence=1.0,
    )

    verified = service.verify_claim_evidence(claim, [chunk])

    assert verified.status == "low_evidence"
    assert verified.support_status == "unsupported"
    assert verified.metadata["evidence_rejection_reason"] == "subject_echo"


def test_legacy_max_questions_is_only_a_batch_hint():
    payload = make_payload(
        mode="deep",
        target_fields=[f"属性{i}" for i in range(1, 10)],
        metadata={"max_questions": 4},
    )
    questions = plan_completion_questions(payload)

    assert len(questions) == 9
    assert [len(batch) for batch in service.batch_completion_questions(payload, questions)] == [4, 4, 1]


def test_claim_extraction_batches_cover_every_question_and_deduplicate(monkeypatch):
    payload = make_payload(mode="deep", target_fields=[f"属性{i}" for i in range(1, 11)])
    questions = plan_completion_questions(payload)
    chunks = [
        make_chunk(index, question.question_id, question.target_field or "")
        for index, question in enumerate(questions, start=1)
    ]
    calls_seen: list[list[str]] = []

    def fake_call(_payload, extractor_chunks, trace_id=""):
        calls_seen.append([str(chunk.question_id) for chunk in extractor_chunks])
        calls = []
        for chunk in extractor_chunks:
            args = {
                "predicate": chunk.target_field,
                "object_value": f"{chunk.target_field}值",
                "source_id": chunk.source_id,
                "question_id": chunk.question_id,
                "quote": chunk.quote,
                "confidence": 0.9,
            }
            call = {"name": "extract_property_claim", "arguments": json.dumps(args, ensure_ascii=False)}
            calls.extend([call, call])
        return calls, {"tool_calls": len(calls)}

    monkeypatch.setattr(service, "call_claim_extractor", fake_call)
    claims, diagnostics = service.extract_claims_in_question_batches(payload, questions, chunks)

    assert diagnostics["batch_count"] == 2
    assert diagnostics["planned_question_count"] == 10
    assert set(item for batch in calls_seen for item in batch) == {item.question_id for item in questions}
    assert len(claims) == 10
    assert len({claim.claim_id for claim in claims}) == 10


def test_evidence_limit_is_applied_per_question_not_globally(monkeypatch):
    payload = make_payload(
        mode="deep",
        target_fields=[f"属性{i}" for i in range(1, 10)],
        metadata={"source_scope": ["domain_kb"], "domain_kb_limit_per_question": 8, "evidence_limit_per_question": 8},
    )
    questions = plan_completion_questions(payload)

    def fake_search(_scenic_id, query, limit=5):
        return [
            {
                "title": f"{query} 来源 {index}",
                "content": f"{query} 可核验证据 {index}",
                "source": "domain-kb-test",
                "source_type": "domain_kb",
                "source_doc_id": f"doc-{index}",
                "chunk_id": index,
                "score": 0.9,
            }
            for index in range(1, limit + 1)
        ]

    monkeypatch.setattr(service, "search_domain_kb", fake_search)
    chunks, _ = service.collect_evidence(payload, trace_id="test-evidence", questions=questions)
    counts = Counter(chunk.question_id for chunk in chunks)

    assert len(chunks) == 72
    assert set(counts.values()) == {8}
    assert set(counts) == {question.question_id for question in questions}

def test_extractor_evidence_json_is_complete(monkeypatch):
    payload = make_payload(mode="deep", target_fields=[f"属性{i}" for i in range(1, 13)])
    questions = plan_completion_questions(payload)
    chunks = [
        make_chunk(index, question.question_id, question.target_field or "")
        for index, question in enumerate(questions, start=1)
    ]

    def fake_generation_call(**kwargs):
        user_message = kwargs["messages"][1]["content"]
        evidence_text = user_message.split("evidence_chunks: ", 1)[1]
        parsed = json.loads(evidence_text)
        assert len(parsed) == len(chunks)
        assert len(evidence_text) > 18000
        return SimpleNamespace(
            status_code=HTTPStatus.OK,
            output=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]
            ),
        )

    monkeypatch.setattr(service.Generation, "call", fake_generation_call)
    calls, metadata = service.call_claim_extractor(payload, chunks, trace_id="test-json")

    assert calls == []
    assert metadata["warning"] == "no_tool_calls"


def test_growth_relation_target_must_appear_in_supplied_evidence():
    chunk = EvidenceChunk(
        source_id="S1", title="证据", content="二郎庙内保存有石碑。",
        quote="二郎庙内保存有石碑。", source="domain_kb", source_type="domain_kb",
        source_doc_id="doc-1", chunk_id=1,
    )
    claim = CandidateClaim(
        claim_id="c_001", claim_type="relation", subject_node_id="1469",
        subject_name="二郎庙", predicate="包含", object_name="306", source_id="S1",
        quote="二郎庙内保存有石碑。", confidence=1.0,
        metadata={"completion_mode": "growth_g2"},
    )
    verified = service.verify_claim_evidence(claim, [chunk])
    assert verified.status == "low_evidence"
    assert verified.metadata["evidence_rejection_reason"] == "growth_value_or_quote_not_in_evidence"


def test_growth_property_value_and_quote_are_anchored():
    chunk = EvidenceChunk(
        source_id="S1", title="证据", content="二郎庙始建于明代。",
        quote="二郎庙始建于明代。", source="domain_kb", source_type="domain_kb",
        source_doc_id="doc-1", chunk_id=1,
    )
    claim = CandidateClaim(
        claim_id="c_001", claim_type="property", subject_node_id="1469",
        subject_name="二郎庙", predicate="始建年代", object_value="明代", source_id="S1",
        quote="二郎庙始建于明代。", confidence=1.0,
        metadata={"completion_mode": "growth_g2"},
    )
    verified = service.verify_claim_evidence(claim, [chunk])
    assert verified.status in {"adoptable", "needs_review"}
