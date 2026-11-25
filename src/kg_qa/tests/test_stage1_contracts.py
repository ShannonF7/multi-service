import pytest

from src.kg_qa.domain import DomainAccessError, DomainGuard
from src.kg_qa.input_analyzer import analyze_input
from src.kg_qa.schemas import QARequest


def test_domain_guard_builds_scoped_context():
    guard = DomainGuard(
        resolver=lambda value: f"graph-{value}",
        source_id_resolver=lambda value: f"source-{value}",
        allowed_domains={"zhangbi"},
    )
    context = guard.validate(QARequest(domain_id="zhangbi", message="魁星楼在哪里？"))
    assert context.domain_id == "zhangbi"
    assert context.source_scenic_id == "source-zhangbi"
    assert context.graph_domain_id == "graph-zhangbi"


def test_domain_guard_rejects_unallowed_domain_and_missing_scope_node():
    guard = DomainGuard(resolver=lambda value: value, allowed_domains={"a"})
    with pytest.raises(DomainAccessError):
        guard.validate(QARequest(domain_id="b", message="问题"))
    with pytest.raises(DomainAccessError):
        guard.validate(QARequest(domain_id="a", message="问题", scope="NODE"))


def test_analyzer_distinguishes_text_image_and_multimodal():
    text = analyze_input(QARequest(domain_id="a", message="魁星楼为什么建在这里？"))
    assert text.modality_intent == "TEXT_TO_TEXT"
    assert "EXPLANATION" in text.knowledge_intents
    assert text.mentions == ["魁星楼"]

    image = analyze_input(QARequest(domain_id="a", image_path="/tmp/query.jpg"))
    assert image.modality_intent == "IMAGE_TO_TEXT"

    both = analyze_input(QARequest(domain_id="a", message="这是什么建筑？", image_ids=["asset-1"]))
    assert both.modality_intent == "MULTIMODAL_TO_TEXT"
    assert both.query_modality == "MULTIMODAL"


def test_analyzer_rejects_empty_input():
    with pytest.raises(ValueError):
        analyze_input(QARequest(domain_id="a"))


def test_analyzer_routes_casual_chat_without_entity_mentions():
    result = analyze_input(QARequest(domain_id="a", message="你好"))
    assert result.knowledge_intents == ["CHAT"]
    assert result.mentions == []
