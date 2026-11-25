from src.reasoning.response import unresolved_position_answer


def test_unresolved_answer_is_natural_and_actionable():
    answer = unresolved_position_answer(["校长办公室"])
    assert answer.startswith("我暂时还没")
    assert "校长办公室" in answer
    assert "校区、楼栋或部门" in answer
    assert "实体" not in answer
    assert "强行" not in answer
    assert "无关资料" not in answer


def test_unresolved_answer_handles_multiple_names():
    answer = unresolved_position_answer(["校长办公室", "财务室"])
    assert "校长办公室、财务室" in answer
