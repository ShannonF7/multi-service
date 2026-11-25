from src.reasoning.logical_form import LogicalQuery, TargetReference
from src.reasoning.planner import build_reasoning_plan


def test_chat_does_not_use_knowledge_actions():
    query = LogicalQuery(
        question="你好呀",
        standalone_question="你好呀",
        domain_id=3,
        intent="CHAT",
        reasoning_mode="NONE",
    )
    plan = build_reasoning_plan(query)
    assert plan.steps == ()
    assert "闲聊不访问知识库" in plan.stop_conditions


def test_current_location_without_browser_location_stops_before_evidence():
    query = LogicalQuery(
        question="我现在在哪里？",
        standalone_question="我现在在哪里？",
        domain_id=3,
        intent="CURRENT_LOCATION",
        relation="located_in",
        reasoning_mode="CURRENT_LOCATION",
    )
    plan = build_reasoning_plan(query)
    assert [step.action for step in plan.steps] == ["CURRENT_LOCATION_RESOLUTION"]
    assert plan.max_hops == 0
    assert "不访问证据库" in plan.stop_conditions[1]


def test_location_plan_is_bounded_and_published():
    query = LogicalQuery(
        question="校长办公室在哪里？",
        standalone_question="校长办公室在哪里？",
        domain_id=3,
        intent="LOCATION",
        relation="located_in",
        targets=(TargetReference("校长办公室", node_id="10"),),
        reasoning_mode="DIRECT_RELATION",
    )
    plan = build_reasoning_plan(query)
    actions = [step.action for step in plan.steps]
    assert actions == ["GRAPH_NODE_DETAIL", "DOCUMENT_EVIDENCE_SEARCH", "BUILD_ANSWER_EVIDENCE"]
    assert plan.steps[0].params["parent_only"] is True
    assert plan.max_hops == 1
    assert all("cypher" not in str(step.params).lower() for step in plan.steps)


def test_path_plan_has_rule_and_hop_limit():
    query = LogicalQuery(
        question="从行知楼怎么走到图书馆？",
        standalone_question="从行知楼怎么走到图书馆？",
        domain_id=3,
        intent="PATH",
        relation="route",
        targets=(
            TargetReference("行知楼", node_id="10"),
            TargetReference("图书馆", node_id="20"),
        ),
        reasoning_mode="PATH_SEARCH",
    )
    plan = build_reasoning_plan(query)
    actions = [step.action for step in plan.steps]
    assert "GRAPH_PATH_SEARCH" in actions
    assert "RULE_EVALUATION" in actions
    assert plan.max_hops == 4
    assert plan.max_nodes <= 40
    assert plan.max_edges <= 80
