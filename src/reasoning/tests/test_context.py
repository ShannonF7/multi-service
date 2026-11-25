from src.reasoning.context import (
    ConversationContextView,
    is_current_location_question,
    resolve_context,
)


def test_current_location_does_not_create_fake_entity():
    assert is_current_location_question("我现在在哪里？")
    result = resolve_context("我现在在哪里？")
    assert result.active_entity_name is None
    assert result.active_entity_id is None


def test_pronoun_uses_latest_entity():
    context = ConversationContextView(
        turns=(
            {"entity_name": "旧教学楼"},
            {"entity_name": "行知楼"},
        )
    )
    result = resolve_context("它的图片呢？", context)
    assert result.active_entity_name == "行知楼"
    assert result.standalone_question == "关于行知楼，它的图片呢？"
    assert result.used_context is True


def test_pending_floor_selection_uses_previous_candidates():
    context = ConversationContextView(
        pending_clarification={
            "question": "财务部在哪里？",
            "candidates": [
                {"id": 1, "name": "财务部（行政楼·b座·一层）"},
                {"id": 2, "name": "财务部（行政楼·b座·二层）"},
            ],
        }
    )
    result = resolve_context("二楼的", context)
    assert result.active_entity_id == "2"
    assert result.active_entity_source == "PENDING_SELECTION"
    assert result.pending_selection is not None
    assert result.pending_selection.index == 2
