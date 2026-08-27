from src.rag.schemas import EvidenceChunk, SemanticCompleteRequest
from src.rag.service.conflict_classification_service import _source_key
from src.rag.service.semantic_completion_service import claims_from_tool_calls
from src.semantic_growth.kg_delta_service import _property_policy, _relation_policy


def _payload(schema=None):
    return SemanticCompleteRequest(
        scenic_id="4",
        node={"source_node_id": "n1", "name": "魁星楼", "node_type": "poi", "scenic_name": "张壁古堡"},
        metadata={"domain_schema": schema or {}},
        use_web_search=False,
    )


def test_completion_router_keeps_action_sentence_in_background_lane():
    chunk = EvidenceChunk(source_id="s1", content="学校落实立德树人根本任务。", quote="落实立德树人根本任务")
    claims = claims_from_tool_calls(
        _payload(), [chunk],
        [{"name": "extract_property_claim", "arguments": '{"source_id":"s1","predicate":"落实立德树人根本任务","object_value":"是","quote":"落实立德树人根本任务","confidence":0.9}'}],
    )
    assert len(claims) == 1
    assert claims[0].claim_type == "fact"
    assert claims[0].metadata["semantic_role"] == "ACTION"


def test_completion_router_accepts_declared_boolean_property():
    chunk = EvidenceChunk(source_id="s1", content="魁星楼是世界遗产。", quote="世界遗产")
    claims = claims_from_tool_calls(
        _payload({"boolean_properties": ["世界遗产"]}), [chunk],
        [{"name": "extract_property_claim", "arguments": '{"source_id":"s1","predicate":"世界遗产","object_value":"是","quote":"世界遗产","confidence":0.9}'}],
    )
    assert claims[0].claim_type == "property"


def test_source_key_groups_chunks_from_same_document():
    first = {"source_type": "domain_kb", "source_doc_id": "doc-1", "chunk_id": 1}
    second = {"source_type": "domain_kb", "source_doc_id": "doc-1", "chunk_id": 2}
    a = type("Claim", (), {"metadata": first, "source_url": "", "source_id": "chunk-1", "evidence_ids": [1]})()
    b = type("Claim", (), {"metadata": second, "source_url": "", "source_id": "chunk-2", "evidence_ids": [2]})()
    assert _source_key(a) == _source_key(b) == "document:doc-1"


def test_growth_schema_overrides_default_cardinality():
    assert _property_policy("建筑高度", domain_schema={"properties": {"建筑高度": {"cardinality": "multi"}}})["conflict_policy"] == "append"
    assert _relation_policy("位于", domain_schema={"relations": {"位于": {"cardinality": "multi"}}})["conflict_policy"] == "append"
