from src.kg_qa.answer import validate_generated_answer
from src.kg_qa.input_analyzer import analyze_input
from src.kg_qa.schemas import EvidenceItem, QARequest, InputAnalysis
from src.kg_qa.service import _select_answer_images

def test_evidence_id_marker_is_normalized():
    evidence = [EvidenceItem(evidence_id="graph:4:964:profile", kind="graph_fact", content="坐标")]
    answer, citations, warnings = validate_generated_answer(
        {"answer": "行知楼在这里[evidence_id: graph:4:964:profile]", "citations": []},
        evidence,
    )
    assert answer == "行知楼在这里[graph:4:964:profile]"
    assert [item.evidence_id for item in citations] == ["graph:4:964:profile"]
    assert not any("evidence_id:" in item for item in warnings)

def test_image_intent_returns_at_most_three_unique_images():
    evidence = [EvidenceItem(
        evidence_id="graph:4:964:profile",
        kind="graph_fact",
        content="图片",
        image_url="/a.webp",
        metadata={"assets": [{"image_url": "/a.webp"}, {"image_url": "/b.webp"}, {"image_url": "/c.webp"}, {"image_url": "/d.webp"}]},
    )]
    analysis = InputAnalysis(query_modality="TEXT", target_modality="TEXT", modality_intent="TEXT_TO_TEXT", knowledge_intents=["IMAGE"])
    images = _select_answer_images(evidence, analysis, "行知楼有哪些图片")
    assert [item["image_url"] for item in images] == ["/a.webp", "/b.webp", "/c.webp"]

def test_image_suffix_is_not_sent_as_entity_name():
    analysis = analyze_input(QARequest(domain_id="4", message="行知楼有哪些图片"))
    assert analysis.mentions == ["行知楼"]


def test_entity_linker_supports_hierarchy_qualifier_and_alias():
    from src.kg_qa.entity_linker import EntityLinker
    from src.kg_qa.schemas import DomainContext

    class Gateway:
        def search(self, domain_id, query, limit=10):
            if query == "\u884c\u77e5\u697c":
                return [{"node_id": "964", "name": "\u884c\u77e5\u697c", "hierarchy_path": ["\u592a\u539f\u7406\u5de5\u5927\u5b66\uff08\u660e\u5411\u6821\u533a\uff09", "\u4e1c\u90e8\u6559\u5b66\u533a", "\u884c\u77e5\u697c"]}]
            if query == "\u8d22\u52a1\u90e8":
                return [
                    {"node_id": "1802", "name": "\u8d22\u52a1\u90e8", "hierarchy_path": ["\u884c\u653f\u697c", "b\u5ea7", "\u4e00\u5c42", "\u8d22\u52a1\u90e8"]},
                    {"node_id": "1122", "name": "\u8d22\u52a1\u90e8", "hierarchy_path": ["\u884c\u653f\u697c", "b\u5ea7", "\u4e8c\u5c42", "\u8d22\u52a1\u90e8"]},
                ]
            return []

    context = DomainContext(domain_id="4", source_scenic_id="4", graph_domain_id="4")
    linker = EntityLinker(Gateway())

    qualified = linker.link(context, ["\u592a\u539f\u7406\u5de5\u5927\u5b66\u660e\u5411\u6821\u533a\u4e1c\u90e8\u6559\u5b66\u533a\u884c\u77e5\u697c"])[0]
    assert qualified.status == "LINKED"
    assert qualified.node_id == "964"

    alias = linker.link(context, ["\u8d22\u52a1\u5ba4"])[0]
    assert alias.status == "AMBIGUOUS"
    assert {item.node_id for item in alias.candidates} == {"1802", "1122"}


def test_entity_analyzer_preserves_multi_level_qualifier():
    analysis = analyze_input(QARequest(
        domain_id="4",
        message="\u592a\u539f\u7406\u5de5\u5927\u5b66\u660e\u5411\u6821\u533a\u884c\u653f\u697cb\u5ea7\u4e8c\u5c42\u8d22\u52a1\u90e8\u5728\u54ea\u91cc",
    ))
    assert analysis.mentions == ["\u592a\u539f\u7406\u5de5\u5927\u5b66\u660e\u5411\u6821\u533a\u884c\u653f\u697cb\u5ea7\u4e8c\u5c42\u8d22\u52a1\u90e8"]


def test_candidate_label_hides_technical_node_id():
    from src.kg_qa.service import _candidate_label
    from src.kg_qa.schemas import EntityCandidate

    label = _candidate_label(EntityCandidate(
        node_id="1122",
        name="财务部",
        hierarchy_path=["行政楼", "b座", "二层", "财务部"],
    ))
    assert label == "财务部（行政楼·b座·二层）"
    assert "节点" not in label
    assert "1122" not in label


def test_entity_linker_prefers_domain_root_over_article_titles():
    from src.kg_qa.entity_linker import EntityLinker
    from src.kg_qa.schemas import DomainContext

    class Gateway:
        def search(self, domain_id, query, limit=10):
            return [
                {"node_id": "3", "name": "山西张壁古堡", "node_type": "", "hierarchy_path": [None, None, None, "山西张壁古堡"]},
                {"node_id": "761", "name": "张壁古堡与电影电视剧", "node_type": "", "hierarchy_path": [None, "山西张壁古堡", "其他", "张壁古堡与电影电视剧"]},
            ]

    linked = EntityLinker(Gateway()).link(DomainContext(domain_id="3", source_scenic_id="3", graph_domain_id="3"), ["张壁古堡"])[0]
    assert linked.status == "LINKED"
    assert linked.node_id == "3"
    assert linked.name == "山西张壁古堡"
    assert all("None" not in part for candidate in linked.candidates for part in candidate.hierarchy_path)


def test_simple_location_question_does_not_expand_neighborhood():
    from src.kg_qa.planner import build_query_plan
    from src.kg_qa.schemas import DomainContext, EntityCandidate, LinkedEntity

    request = QARequest(domain_id="3", message="张壁古堡在哪里")
    analysis = analyze_input(request)
    linked = [LinkedEntity(mention="张壁古堡", status="LINKED", node_id="1159", name="山西张壁古堡")]
    plan = build_query_plan(request, analysis, linked)
    assert not any(step.type == "GRAPH_NEIGHBORHOOD" for step in plan.steps)

def test_nearby_location_question_expands_neighborhood():
    from src.kg_qa.planner import build_query_plan
    from src.kg_qa.schemas import LinkedEntity

    request = QARequest(domain_id="3", message="张壁古堡附近有哪些景点")
    analysis = analyze_input(request)
    linked = [LinkedEntity(mention="张壁古堡", status="LINKED", node_id="1159", name="山西张壁古堡")]
    plan = build_query_plan(request, analysis, linked)
    assert any(step.type == "GRAPH_NEIGHBORHOOD" for step in plan.steps)


def test_nested_domain_block_id_resolves_to_text_evidence():
    evidence = [EvidenceItem(
        evidence_id="text:doc-1",
        kind="text",
        content="张壁古堡位于张壁村",
        metadata={"raw": {"items": [{"content_blocks": [{"id": "zhangbigubao_00001"}]}]}},
    )]
    answer, citations, warnings = validate_generated_answer(
        {"answer": "张壁古堡位于张壁村[ zhangbigubao_00001 ]", "citations": []},
        evidence,
    )
    assert answer == "张壁古堡位于张壁村[text:doc-1]"
    assert [item.evidence_id for item in citations] == ["text:doc-1"]
    assert warnings == []


def test_nested_domain_json_string_block_id_resolves_to_text_evidence():
    evidence = [EvidenceItem(
        evidence_id="text:doc-2",
        kind="text",
        content="张壁古堡位于张壁村",
        metadata={"raw": {"content": '{"content_blocks": [{"id": "zhangbigubao_00002"}]}' }},
    )]
    answer, citations, warnings = validate_generated_answer(
        {"answer": "张壁古堡位于张壁村[zhangbigubao_00002]", "citations": []},
        evidence,
    )
    assert answer == "张壁古堡位于张壁村[text:doc-2]"
    assert [item.evidence_id for item in citations] == ["text:doc-2"]
    assert warnings == []


def test_domain_block_numeric_suffix_resolves_to_text_evidence():
    evidence = [EvidenceItem(
        evidence_id="text:doc-3",
        kind="text",
        content="张壁古堡位于张壁村",
        metadata={"raw": {"content": '{"content_blocks": [{"id": "zhangbigubao_00000"}]}' }},
    )]
    answer, citations, warnings = validate_generated_answer(
        {"answer": "张壁古堡位于张壁村[00000]", "citations": []},
        evidence,
    )
    assert answer == "张壁古堡位于张壁村[text:doc-3]"
    assert [item.evidence_id for item in citations] == ["text:doc-3"]
    assert warnings == []


def test_repeated_domain_block_id_uses_best_text_evidence():
    evidence = [
        EvidenceItem(evidence_id="text:best", kind="text", content="高相关来源", score=0.9, metadata={"raw": {"content": '{"content_blocks": [{"id": "block_00001"}]}'}}),
        EvidenceItem(evidence_id="text:other", kind="text", content="同资料低相关片段", score=0.5, metadata={"raw": {"content": '{"content_blocks": [{"id": "block_00001"}]}'}}),
    ]
    answer, citations, warnings = validate_generated_answer(
        {"answer": "事实[block_00001]", "citations": []},
        evidence,
    )
    assert answer == "事实[text:best]"
    assert [item.evidence_id for item in citations] == ["text:best"]
    assert warnings == []
