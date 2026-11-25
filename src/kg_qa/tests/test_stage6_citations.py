from src.kg_qa.evidence import fuse_evidence
from src.kg_qa.answer import validate_generated_answer


def test_graph_fact_exposes_source_location_and_quote():
    evidence = fuse_evidence(
        "16",
        [
            {
                "node_id": "1949",
                "root": {"name": "月球"},
                "facts": [
                    {
                        "key": "平均密度",
                        "value": "3.34克/立方厘米",
                        "evidence_ids": ["3455"],
                        "provenance_json": (
                            '[{"quote":"月亮的平均密度是3.34克/立方厘米。",'
                            '"source_title":"月球资料","source_url":"https://example.com/moon",'
                            '"page_no":12,"evidence_ids":[3455]}]'
                        ),
                    }
                ],
            }
        ],
        [],
    )
    item = evidence[0]
    assert item.content == "月亮的平均密度是3.34克/立方厘米。"
    assert item.source_url == "https://example.com/moon"
    assert item.page_no == 12
    assert item.location_label == "月球资料 · 第 12 页"
    assert item.metadata["source_evidence_ids"] == ["3455"]


def test_answer_adds_valid_source_marker_when_model_only_returns_citation_list():
    evidence = fuse_evidence(
        "16",
        [{"node_id": "1", "root": {"name": "实体", "description": "事实"}}],
        [],
    )
    answer, citations, warnings = validate_generated_answer(
        {"answer": "事实。", "citations": [{"evidence_id": evidence[0].evidence_id}]},
        evidence,
    )
    assert evidence[0].evidence_id in answer
    assert citations[0].label == evidence[0].location_label
    assert not warnings



def test_answer_merges_inline_marker_and_removes_unknown_internal_id():
    evidence = fuse_evidence(
        "16",
        [{"node_id": "1", "root": {"name": "实体", "description": "事实"}}],
        [],
    )
    evidence_id = evidence[0].evidence_id
    answer, citations, warnings = validate_generated_answer(
        {"answer": f"事实。 [{evidence_id}] [1d28014f0b8649da82b762c7d28d0f35]", "citations": []},
        evidence,
    )
    assert f"[{evidence_id}]" in answer
    assert "1d28014f0b8649da82b762c7d28d0f35" not in answer
    assert [item.evidence_id for item in citations] == [evidence_id]
    assert not any("忽略未返回" in warning for warning in warnings)


def test_numeric_source_marker_is_resolved_to_actual_evidence_id():
    evidence = fuse_evidence(
        "16",
        [],
        [{"chunk_id": 91, "content": "课题组资料", "source_type": "domain_kb", "title": "课题组介绍"}],
    )
    answer, citations, warnings = validate_generated_answer(
        {"answer": "课题组资料[91]", "citations": []}, evidence,
    )
    assert answer == f"课题组资料[{evidence[0].evidence_id}]"
    assert [item.evidence_id for item in citations] == [evidence[0].evidence_id]
    assert not warnings


def test_image_evidence_exposes_displayable_image_url():
    evidence = fuse_evidence(
        "16", [], [{"asset_id": 7, "url": "/media/xingmian.webp", "title": "行勉楼"}],
    )
    assert evidence[0].kind == "image"
    assert evidence[0].image_url == "/media/xingmian.webp"


def test_answer_unwraps_json_decodes_html_entities_and_removes_bold_citation():
    evidence = fuse_evidence(
        "16", [{"node_id": "1", "root": {"name": "place", "description": "fact"}}], [],
    )
    evidence_id = evidence[0].evidence_id
    raw = f'{{"answer":"112.71814&#x32; **[{evidence_id}]**","citations":[{{"evidence_id":"{evidence_id}"}}]}}'
    answer, citations, warnings = validate_generated_answer({"answer": raw, "citations": []}, evidence)
    assert answer == f"112.718142 [{evidence_id}]"
    assert [item.evidence_id for item in citations] == [evidence_id]
    assert "&#x" not in answer
    assert "**" not in answer
    assert not warnings


def test_dashscope_stream_holds_json_wrapper_and_emits_only_answer(monkeypatch):
    import sys
    from types import SimpleNamespace
    from src.kg_qa.answer import DashScopeQAClient

    class FakeGeneration:
        @staticmethod
        def call(**kwargs):
            def item(content):
                return {"output": {"choices": [{"message": {"content": content}}]}}
            return iter([
                item('plain answer\n{"answer":"plain answer","citations":[]}'),
            ])

    monkeypatch.setitem(sys.modules, "dashscope", SimpleNamespace(Generation=FakeGeneration))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    deltas = []
    result = DashScopeQAClient().generate_stream(**{
        "question": "question",
        "context": SimpleNamespace(name="domain", domain_id="domain"),
        "analysis": SimpleNamespace(knowledge_intents=[]),
        "entities": [],
        "plan": SimpleNamespace(steps=[]),
        "evidence": [],
        "on_delta": deltas.append,
    })
    assert result["answer"] == "plain answer"
    assert "{" not in "".join(deltas)
    assert "plain answer" in "".join(deltas)
