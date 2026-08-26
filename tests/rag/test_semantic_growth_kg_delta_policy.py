from src.semantic_growth.kg_delta_service import _property_policy, _relation_key, _relation_policy


def test_multi_value_relation_targets_are_appendable():
    assert _relation_policy("包含")["conflict_policy"] == "append"
    assert _relation_policy("相邻")["conflict_policy"] == "append"


def test_exclusive_relation_aliases_share_one_policy():
    assert _relation_key("located_in") == _relation_key("位于") == "空间位置"
    assert _relation_policy("located_in")["conflict_policy"] == "exclusive"
    assert _relation_policy("位于")["cardinality"] == "single"


def test_multi_value_properties_do_not_force_conflict():
    assert _property_policy("别名")["conflict_policy"] == "append"
    assert _property_policy("描述")["cardinality"] == "multi"


def test_unknown_property_defaults_to_exclusive():
    assert _property_policy("建筑高度")["conflict_policy"] == "exclusive"


def test_compatible_temporal_roles_are_multi_value():
    assert _property_policy("时间", "renovation_time")["conflict_policy"] == "append"
