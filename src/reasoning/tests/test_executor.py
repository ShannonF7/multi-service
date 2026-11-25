from src.kg_qa.entity_linker import EntityLinker
from src.kg_qa.retriever import DefaultRetriever
from src.kg_qa.schemas import DomainContext, QARequest

from src.reasoning.executor import ReasoningExecutor
from src.reasoning.query_understanding import understand_question


class FakeEntityGateway:
    """测试用的领域实体网关，不访问真实数据库。"""

    def __init__(self, items):
        self.items = items

    def search(self, domain_id, query, limit=10):
        return list(self.items.get(query, []))


class FakeGraphGateway:
    """测试用的已发布图谱网关。"""

    def __init__(self):
        self.detail_calls = []

    def node_detail(self, domain_id, node_id):
        self.detail_calls.append((domain_id, node_id))
        return {
            "domain_id": domain_id,
            "node_id": node_id,
            "root": {
                "node_id": node_id,
                "name": "校长办公室",
                "description": "位于行政楼一层。",
            },
            "relations": [
                {
                    "source_name": "校长办公室",
                    "relation_label": "位于",
                    "target_name": "行政楼",
                }
            ],
            "neighbors": [],
        }

    def neighborhood(self, domain_id, node_id, depth, limit):
        return {"domain_id": domain_id, "node_id": node_id, "root": {"node_id": node_id, "name": "校长办公室"}}

    def path(self, domain_id, source_node_id, target_node_id, max_depth):
        return {
            "domain_id": domain_id,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "nodes": [
                {"node_id": source_node_id, "name": "起点"},
                {"node_id": target_node_id, "name": "终点"},
            ],
        }


def make_context():
    return DomainContext(
        domain_id="3",
        source_scenic_id="scenic-3",
        graph_domain_id="graph-3",
    )


def make_request(message):
    return QARequest(domain_id="3", message=message, max_evidence=8)


def test_executor_uses_published_graph_and_text_evidence():
    graph = FakeGraphGateway()
    entity_gateway = FakeEntityGateway(
        {
            "校长办公室": [
                {
                    "node_id": "10",
                    "name": "校长办公室",
                    "node_type": "办公室",
                    "hierarchy_path": ["行政楼", "校长办公室"],
                    "match_type": "EXACT",
                }
            ]
        }
    )
    text_calls = []

    def text_search(source_scenic_id, query, limit):
        text_calls.append((source_scenic_id, query, limit))
        return [{"content": "校长办公室在行政楼一层。", "title": "办公地点说明", "source_id": "doc-1", "score": 0.8}]

    retriever = DefaultRetriever(
        graph=graph,
        text_search=text_search,
        unified_search=lambda **kwargs: {"items": [], "diagnostics": {}},
    )
    executor = ReasoningExecutor(
        entity_linker=EntityLinker(entity_gateway),
        retriever=retriever,
    )
    query = understand_question("校长办公室在哪里？", "3")
    result = executor.run(query, make_request("校长办公室在哪里？"), make_context())

    assert result.warnings == []
    assert result.entities[0].status == "LINKED"
    assert graph.detail_calls == [("graph-3", "10")]
    assert text_calls and text_calls[0][0] == "scenic-3"
    assert any(item.kind == "graph_fact" for item in result.evidence)
    assert any(item.kind == "text" for item in result.evidence)
    assert result.plan.steps[0].action == "GRAPH_NODE_DETAIL"


def test_executor_stops_on_ambiguous_entity():
    graph = FakeGraphGateway()
    entity_gateway = FakeEntityGateway(
        {
            "财务部": [
                {"node_id": "1", "name": "财务部", "hierarchy_path": ["行政楼", "一层"]},
                {"node_id": "2", "name": "财务部", "hierarchy_path": ["行政楼", "二层"]},
            ]
        }
    )
    retriever = DefaultRetriever(
        graph=graph,
        text_search=lambda *args: [],
        unified_search=lambda **kwargs: {"items": [], "diagnostics": {}},
    )
    executor = ReasoningExecutor(
        entity_linker=EntityLinker(entity_gateway),
        retriever=retriever,
    )
    result = executor.run(
        understand_question("财务部在哪里？", "3"),
        make_request("财务部在哪里？"),
        make_context(),
    )

    assert result.clarification is not None
    assert "1. 财务部（行政楼·一层）" in result.clarification
    assert "2. 财务部（行政楼·二层）" in result.clarification
    assert graph.detail_calls == []
    assert result.evidence == []


def test_executor_chat_and_missing_location_do_not_retrieve():
    graph = FakeGraphGateway()
    retriever = DefaultRetriever(
        graph=graph,
        text_search=lambda *args: (_ for _ in ()).throw(AssertionError("chat must not search")),
        unified_search=lambda **kwargs: (_ for _ in ()).throw(AssertionError("chat must not search")),
    )
    executor = ReasoningExecutor(retriever=retriever)

    chat = executor.run(
        understand_question("你好呀", "3"),
        make_request("你好呀"),
        make_context(),
    )
    no_location = executor.run(
        understand_question("我现在在哪里？", "3"),
        make_request("我现在在哪里？"),
        make_context(),
    )
    assert chat.evidence == []
    assert no_location.warnings == ["未收到可用的实时位置，未访问图谱和证据库"]
