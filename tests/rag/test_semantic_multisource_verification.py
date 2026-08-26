from src.rag.schemas import EvidenceChunk, SemanticCompleteRequest
from src.rag.service import semantic_completion_service as service
from src.rag.service.completion_job_service import _normalize_mode_payload
from src.rag.service.planner_service import plan_completion_questions


def make_request(**overrides):
    payload = {
        "scenic_id": "4",
        "node": {
            "source_node_id": "946",
            "name": "测试节点",
            "node_type": "building",
            "scenic_name": "测试领域",
        },
        "target_fields": ["面积"],
        "relation_intents": [],
        "subgraph_depth": 1,
        "existing_properties": [{"key": "高度", "value": "15.3米"}],
        "existing_relations": [{"relation_type": "位于", "target_name": "测试区域"}],
        "metadata": {"completion_mode": "deep"},
    }
    payload.update(overrides)
    return SemanticCompleteRequest.parse_obj(payload)


def test_deep_mode_requires_cross_source_verification():
    normalized = _normalize_mode_payload(make_request())
    metadata = normalized["metadata"]

    assert metadata["web_search_policy"] == "always"
    assert metadata["verify_existing_facts"] is True
    assert metadata["web_query_variants_per_question"] == 2
    assert metadata["evidence_limit_per_question"] == 12
    assert metadata["extractor_chunks_per_question"] == 5


def test_deep_planner_rechecks_existing_properties_and_relations():
    normalized = SemanticCompleteRequest.parse_obj(_normalize_mode_payload(make_request()))
    questions = plan_completion_questions(normalized)
    question_ids = {item.question_id for item in questions}

    assert "prop:面积" in question_ids
    assert "prop:高度" in question_ids
    assert "rel:位于" in question_ids


def test_web_query_variants_add_independent_source_query():
    normalized = SemanticCompleteRequest.parse_obj(_normalize_mode_payload(make_request()))
    question = next(
        item for item in plan_completion_questions(normalized)
        if item.question_id == "prop:面积"
    )

    variants = service.build_web_query_variants(normalized, question)

    assert len(variants) == 2
    assert variants[0] == question.query_text
    assert "官方资料" in variants[1]
    assert "不同来源" in variants[1]


def test_local_coverage_does_not_suppress_web_and_evidence_is_mixed(monkeypatch):
    payload = make_request(
        metadata={
            "completion_mode": "deep",
            "source_scope": ["domain_kb", "web_search"],
            "web_search_policy": "always",
            "web_query_variants_per_question": 1,
            "domain_kb_limit_per_question": 8,
            "web_limit_per_question": 4,
            "evidence_limit_per_question": 6,
        },
        use_web_search=True,
        use_web_extractor=False,
    )
    questions = plan_completion_questions(payload)

    def fake_local(_scenic_id, query, limit=5):
        return [
            {
                "title": f"本地资料 {index}",
                "content": f"测试节点面积本地证据 {index}",
                "source_type": "domain_kb",
                "source_doc_id": f"doc-{index}",
                "chunk_id": index,
                "score": 2.0,
            }
            for index in range(limit)
        ]

    def fake_web(_query, limit=5):
        return [
            {
                "title": "机构来源一",
                "content": "测试节点面积为一百平方米。",
                "source_type": "web_search",
                "source_url": "https://one.example.test/fact",
                "score": 1.5,
            },
            {
                "title": "机构来源二",
                "content": "测试节点面积记载为一百二十平方米。",
                "source_type": "web_search",
                "source_url": "https://two.example.test/fact",
                "score": 1.4,
            },
        ], {}

    monkeypatch.setattr(service, "search_domain_kb", fake_local)
    monkeypatch.setattr(service, "web_search", fake_web)

    chunks, diagnostics = service.collect_evidence(
        payload,
        trace_id="multisource-test",
        questions=questions[:1],
    )

    assert any(item.source_type == "domain_kb" for item in chunks)
    assert len({item.source_url for item in chunks if item.source_type == "web_search"}) == 2
    assert not any(
        item.get("reason") == "local_coverage_sufficient"
        for item in diagnostics
    )


def test_extractor_selection_keeps_distinct_source_identities():
    payload = make_request(metadata={"completion_mode": "deep", "extractor_chunks_per_question": 3})
    question = plan_completion_questions(payload)[0]
    chunks = [
        EvidenceChunk(
            source_id="S1",
            title="本地一",
            content="测试节点面积本地证据一。",
            source_type="domain_kb",
            source_doc_id="doc-1",
            question_id=question.question_id,
            score=0.95,
            final_evidence_score=0.95,
            metadata={"source_identity": "document:doc-1"},
        ),
        EvidenceChunk(
            source_id="S2",
            title="本地二",
            content="测试节点面积本地证据二。",
            source_type="domain_kb",
            source_doc_id="doc-2",
            question_id=question.question_id,
            score=0.94,
            final_evidence_score=0.94,
            metadata={"source_identity": "document:doc-2"},
        ),
        EvidenceChunk(
            source_id="S3",
            title="网站一",
            content="测试节点面积网站证据一。",
            source_type="web_extractor",
            source_url="https://one.example.test/fact",
            question_id=question.question_id,
            score=0.9,
            final_evidence_score=0.9,
            metadata={"source_identity": "domain:one.example.test"},
        ),
    ]

    selected = service.select_extractor_chunks(payload, [question], chunks)

    assert {item.source_id for item in selected} == {"S1", "S2", "S3"}
