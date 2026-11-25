from src.kg_qa.evidence import fuse_evidence
from src.kg_qa.entity_linker import EntityLinker
from src.kg_qa.planner import build_query_plan
from src.kg_qa.retriever import DefaultRetriever
from src.kg_qa.schemas import DomainContext, InputAnalysis, LinkedEntity, QARequest
from src.kg_qa.trace import TraceCollector


class FakeEntityGateway:
    def search(self, domain_id, query, limit=10):
        if query == "魁星楼":
            return [{"node_id": "n1", "name": "魁星楼", "node_type": "building"}]
        if query == "城门":
            return [
                {"node_id": "n2", "name": "东城门"},
                {"node_id": "n3", "name": "西城门"},
            ]
        return []


class FakeGraph:
    def node_detail(self, domain_id, node_id):
        return {"node_id": node_id, "facts": [{"key": "位置", "value": "中部"}], "relations": []}

    def neighborhood(self, domain_id, node_id, depth, limit):
        return {"node_id": node_id, "facts": [], "relations": [{
            "source_name": "魁星楼", "relation_label": "位于", "target_name": "中部"
        }]}

    def path(self, domain_id, source_node_id, target_node_id, max_depth):
        return {"node_id": source_node_id, "path": "东城门 -> 中部 -> 魁星楼"}


def test_entity_linker_separates_linked_ambiguous_and_unresolved():
    context = DomainContext(domain_id="a", source_scenic_id="a", graph_domain_id="a")
    values = EntityLinker(FakeEntityGateway()).link(context, ["魁星楼", "城门", "不存在"])
    assert values[0].status == "LINKED"
    assert values[0].node_id == "n1"
    assert values[1].status == "AMBIGUOUS"
    assert values[2].status == "UNRESOLVED"


def test_planner_creates_bounded_graph_and_text_steps():
    request = QARequest(domain_id="a", message="魁星楼附近有哪些建筑？")
    analysis = InputAnalysis(
        query_modality="TEXT", target_modality="TEXT", modality_intent="TEXT_TO_TEXT",
        knowledge_intents=["LIST", "SPATIAL"], mentions=["魁星楼"], has_text=True,
    )
    entities = [LinkedEntity(mention="魁星楼", status="LINKED", node_id="n1", name="魁星楼")]
    plan = build_query_plan(request, analysis, entities)
    assert [item.type for item in plan.steps] == [
        "GRAPH_NODE_DETAIL", "GRAPH_NEIGHBORHOOD", "TEXT_SEARCH"
    ]
    assert plan.max_hops == 2
    assert plan.max_nodes == 20
    assert plan.max_edges == 40


def test_retriever_uses_domain_context_for_every_backend_call():
    seen = []

    def text_search(domain_id, query, limit):
        seen.append((domain_id, query, limit))
        return [{"chunk_id": 7, "content": "领域原文", "score": 0.7, "source_type": "domain_kb"}]

    context = DomainContext(domain_id="public", source_scenic_id="source-a", graph_domain_id="graph-a")
    request = QARequest(domain_id="public", message="魁星楼在哪里？")
    analysis = InputAnalysis(
        query_modality="TEXT", target_modality="TEXT", modality_intent="TEXT_TO_TEXT",
        knowledge_intents=["SPATIAL"], mentions=["魁星楼"], has_text=True,
    )
    entities = [LinkedEntity(mention="魁星楼", status="LINKED", node_id="n1", name="魁星楼")]
    plan = build_query_plan(request, analysis, entities)
    result = DefaultRetriever(graph=FakeGraph(), text_search=text_search).execute(context, request, plan)
    assert seen == [("source-a", "魁星楼在哪里？", 8)]
    assert result["graph_results"][0]["node_id"] == "n1"
    assert result["retrieval_results"][0]["chunk_id"] == 7


def test_evidence_fusion_normalizes_and_deduplicates_sources():
    values = fuse_evidence(
        "graph-a",
        [{"node_id": "n1", "facts": [{"key": "位置", "value": "中部"}]}],
        [
            {"chunk_id": 7, "content": "领域原文", "score": 0.7, "source_type": "domain_kb"},
            {"chunk_id": 7, "content": "领域原文", "score": 0.6, "source_type": "domain_kb"},
        ],
    )
    assert [item.kind for item in values] == ["graph_fact", "text"]
    assert values[0].evidence_id == "graph:graph-a:n1:fact:位置"


def test_evidence_fusion_turns_published_path_nodes_into_citation():
    values = fuse_evidence(
        "graph-a",
        [{
            "source_node_id": "n1",
            "target_node_id": "n2",
            "nodes": [{"node_id": "n1", "name": "东城门"}, {"node_id": "n2", "name": "魁星楼"}],
            "relations": [],
        }],
        [],
    )
    assert values[0].content == "东城门 -> 魁星楼"
    assert values[0].evidence_id == "graph:graph-a:path:n1:n2"


def test_trace_records_completion_and_failure_without_thought_chain():
    trace = TraceCollector(run_id="qa-test")
    assert trace.run("input", lambda: "ok") == "ok"
    try:
        trace.run("retrieve", lambda: (_ for _ in ()).throw(RuntimeError("backend down")))
    except RuntimeError:
        pass
    assert [item.status for item in trace.events] == ["STARTED", "COMPLETED", "STARTED", "FAILED"]
    assert trace.events[-1].metadata["error_type"] == "RuntimeError"


def test_planner_skips_all_retrieval_steps_for_chat():
    request = QARequest(domain_id="a", message="你好")
    analysis = InputAnalysis(
        query_modality="TEXT", target_modality="TEXT", modality_intent="TEXT_TO_TEXT",
        knowledge_intents=["CHAT"], mentions=[], has_text=True,
    )
    plan = build_query_plan(request, analysis, [])
    assert plan.steps == []
    assert plan.max_hops == 0
