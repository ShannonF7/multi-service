from src.semantic_growth.conflict_quality import value_key


def test_g4_same_value_key_collapses_formatting():
    assert value_key("  古建（明代） ") == value_key("古建明代")


def test_g4_same_value_key_keeps_different_values_distinct():
    assert value_key("1902年") != value_key("1903年")
