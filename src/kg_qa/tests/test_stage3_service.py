from src.kg_qa.answer import CallableAIAnswerGenerator, FallbackAnswerGenerator
from src.kg_qa.domain import DomainGuard
from src.kg_qa.entity_linker import EntityLinker
from src.kg_qa.retriever import DefaultRetriever
from src.kg_qa.schemas import QARequest
from src.kg_qa.service import QAService
from src.kg_qa.api import router


class EntityGateway:
    def search(self, domain_id, query, limit=10):
        return [{"node_id": "n1", "name": "魁星楼", "node_type": "building"}]


class Graph:
    def node_detail(self, domain_id, node_id):
        return {"node_id": node_id, "root": {"name": "魁星楼"}, "facts": [{"key": "位置", "value": "中部"}]}

    def neighborhood(self, domain_id, node_id, depth, limit):
        return {"node_id": node_id, "facts": [], "relations": []}

    def path(self, domain_id, source_node_id, target_node_id, max_depth):
        return {"node_id": source_node_id, "path": "无"}


def make_service(generator=None):
    return QAService(
        domain_guard=DomainGuard(resolver=lambda value: "graph-a", source_id_resolver=lambda value: "source-a"),
        entity_linker=EntityLinker(EntityGateway()),
        retriever=DefaultRetriever(
            graph=Graph(),
            text_search=lambda domain, query, limit: [{
                "chunk_id": 1, "content": "魁星楼位于中部。", "score": 0.8, "source_type": "domain_kb"
            }],
        ),
        answer_generator=generator or FallbackAnswerGenerator(),
    )


def test_service_runs_complete_grounded_pipeline():
    response = make_service().run(QARequest(domain_id="a", message="魁星楼在哪里？"))
    assert response.status == "COMPLETED"
    assert response.evidence
    assert response.citations
    assert [event.stage for event in response.trace if event.status == "COMPLETED"] == [
        "DOMAIN_GUARD", "INPUT_ANALYSIS", "ENTITY_LINKING", "QUERY_PLANNING",
        "RETRIEVAL", "EVIDENCE_FUSION", "ANSWER_GENERATION", "ANSWER_VALIDATION",
    ]


def test_service_stops_on_ambiguous_entity():
    class AmbiguousGateway:
        def search(self, domain_id, query, limit=10):
            return [{"node_id": "n1", "name": "城门"}, {"node_id": "n2", "name": "城门"}]

    service = make_service()
    service.entity_linker = EntityLinker(AmbiguousGateway())
    response = service.run(QARequest(domain_id="a", message="城门在哪里？"))
    assert response.status == "NEEDS_CLARIFICATION"
    assert "请确认" in response.answer
    assert not response.evidence


def test_new_ai_adapter_explicitly_disables_web_search_and_validates_citations():
    calls = []

    def client(**kwargs):
        calls.append(kwargs)
        return {"answer": "根据资料可确认。", "citations": [{"evidence_id": "text:1"}]}

    response = make_service(CallableAIAnswerGenerator(client)).run(
        QARequest(domain_id="a", message="魁星楼在哪里？")
    )
    assert response.status == "COMPLETED"
    assert calls[0]["web_search"] is False
    assert response.citations[0].evidence_id == "text:1"


def test_invalid_ai_citation_is_removed_and_reported():
    def client(**kwargs):
        return {"answer": "无法确认。", "citations": [{"evidence_id": "not-returned"}]}

    response = make_service(CallableAIAnswerGenerator(client)).run(
        QARequest(domain_id="a", message="魁星楼在哪里？")
    )
    assert response.status == "COMPLETED"
    assert not response.citations
    assert any("未返回的证据引用" in warning for warning in response.warnings)


def test_api_exposes_only_the_new_kg_qa_query_route():
    routes = {(route.path, tuple(sorted(route.methods or []))) for route in router.routes}
    assert ("/kg-qa/query", ("POST",)) in routes


def test_service_handles_casual_chat_without_domain_or_retrieval():
    class ExplodingGuard:
        def validate(self, request):
            raise AssertionError("闲聊不应进入领域校验")

    class ExplodingRetriever:
        def execute(self, *args, **kwargs):
            raise AssertionError("闲聊不应执行检索")

    response = QAService(
        domain_guard=ExplodingGuard(),
        retriever=ExplodingRetriever(),
    ).run(QARequest(domain_id="a", message="你好"))
    assert response.status == "COMPLETED"
    assert response.analysis.knowledge_intents == ["CHAT"]
    assert response.evidence == []
    assert [event.stage for event in response.trace if event.status == "COMPLETED"] == [
        "INPUT_ANALYSIS", "CASUAL_CHAT",
    ]


def test_service_emits_context_compression_stage_after_session_limit():
    from src.kg_qa.session import ConversationSessionStore

    service = make_service()
    service.session_store = ConversationSessionStore(max_turns=2, keep_turns=1)
    for index in range(3):
        response = service.run(QARequest(domain_id="a", message=f"魁星楼在哪里？{index}", session_id="session-test"))
        assert response.status == "COMPLETED"
    stages = [event.stage for event in response.trace if event.status == "COMPLETED"]
    assert "SESSION_CONTEXT" in stages
    assert "CONTEXT_COMPRESSION" in stages
