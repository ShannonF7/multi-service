from src.kg_qa.schemas import DomainContext, QARequest

from src.reasoning.integration import build_context_view, build_logical_query


class FakeSession:
    def to_dict(self):
        return {
            "session_id": "s-1",
            "turns": [{"entity_name": "行知楼"}],
            "long_term_memory": {"last_entity": {"name": "行知楼"}},
        }


def test_integration_preserves_string_ids_and_location():
    context = DomainContext(
        domain_id="3",
        source_scenic_id="scenic-3",
        graph_domain_id="graph-3",
        location_node_id="node-9",
        location_node_name="关帝庙戏台",
    )
    request = QARequest(domain_id="3", message="我现在在哪里？")
    view = build_context_view(FakeSession(), context, request)
    query = build_logical_query("我现在在哪里？", context, FakeSession(), request)

    assert view.location_node_id == "node-9"
    assert view.location_node_name == "关帝庙戏台"
    assert query.intent == "CURRENT_LOCATION"
    assert query.targets[0].node_id == "node-9"
    assert query.targets[0].source == "LOCATION"


def test_integration_keeps_previous_entity_for_pronoun():
    context = DomainContext(
        domain_id="3",
        source_scenic_id="scenic-3",
        graph_domain_id="graph-3",
    )
    request = QARequest(domain_id="3", message="它位于哪里？")
    query = build_logical_query("它位于哪里？", context, FakeSession(), request)

    assert query.standalone_question == "关于行知楼，它位于哪里？"
    assert query.targets[0].canonical_name == "行知楼"
