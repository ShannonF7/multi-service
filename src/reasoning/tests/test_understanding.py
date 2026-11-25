from src.reasoning.context import ConversationContextView
from src.reasoning.query_understanding import understand_question


def test_location_question_keeps_qualified_mention():
    result = understand_question("太原理工大学明向校区东部教学区有哪些楼座？", 3)
    assert result.intent == "AGGREGATION"
    assert result.targets[0].mention == "太原理工大学明向校区东部教学区"


def test_location_query_has_relation():
    result = understand_question("校长办公室在哪里？", 3)
    assert result.intent == "LOCATION"
    assert result.relation == "located_in"
    assert result.reasoning_mode == "DIRECT_RELATION"
    assert result.targets[0].mention == "校长办公室"


def test_located_in_is_location_intent():
    result = understand_question("行知楼位于哪里？", 3)
    assert result.intent == "LOCATION"
    assert result.targets[0].mention == "行知楼"


def test_context_question_uses_previous_entity():
    context = ConversationContextView(turns=({"entity_name": "行知楼"},))
    result = understand_question("它位于哪里？", 3, context=context)
    assert result.targets[0].canonical_name == "行知楼"
    assert result.used_context is True


def test_current_location_has_no_pseudo_target():
    result = understand_question("我现在在哪里？", 3)
    assert result.intent == "CURRENT_LOCATION"
    assert result.targets == ()


def test_pending_choice_becomes_target():
    context = ConversationContextView(
        pending_clarification={
            "question": "财务部在哪里？",
            "candidates": [
                {"id": 1, "name": "财务部（一层）"},
                {"id": 2, "name": "财务部（二层）"},
            ],
        }
    )
    result = understand_question("二楼的", 3, context=context)
    assert result.targets[0].node_id == "2"
    assert result.targets[0].source == "PENDING_SELECTION"


def test_chat_has_no_target():
    result = understand_question("你好呀", 3)
    assert result.intent == "CHAT"
    assert result.targets == ()
    assert result.requires_knowledge is False
