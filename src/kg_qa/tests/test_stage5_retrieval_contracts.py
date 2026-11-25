from src.kg_qa.evidence import fuse_evidence
from src.kg_qa.input_analyzer import _extract_mentions


def test_chinese_location_question_drops_trailing_particle():
    assert _extract_mentions("北京故宫在哪里？") == ["北京故宫"]


def test_graph_root_description_becomes_evidence():
    evidence = fuse_evidence(
        "2",
        [{"node_id": "1925", "root": {"name": "北京故宫", "description": "明清皇家宫殿。"}}],
        [],
    )
    assert evidence[0].kind == "graph_fact"
    assert evidence[0].content == "明清皇家宫殿。"

