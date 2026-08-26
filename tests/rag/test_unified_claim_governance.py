from src.rag.service.claim_contract import CanonicalClaim
from src.rag.service.claim_identity_service import claim_keys
from src.rag.service.claim_type_router import route_claim
from src.rag.service.source_weighting_service import infer_authority_class


def test_fact_identity_excludes_provenance_and_conflict_scope_keeps_values_separate():
    first = CanonicalClaim(
        domain_id="4",
        subject_ref="node:1023",
        claim_type="property",
        canonical_predicate="construction_time",
        normalized_value="明代",
        temporal_role="construction_time",
    )
    second = CanonicalClaim(
        domain_id="4",
        subject_ref="node:1023",
        claim_type="PROPERTY",
        canonical_predicate="construction_time",
        normalized_value="清代",
        temporal_role="construction_time",
    )
    assert claim_keys(first)["canonical_claim_key"] != claim_keys(second)["canonical_claim_key"]
    assert claim_keys(first)["conflict_scope_key"] == claim_keys(second)["conflict_scope_key"]


def test_relation_identity_uses_object_ref():
    first = CanonicalClaim("4", "node:1", "RELATION", "located_in", object_ref="node:2")
    second = CanonicalClaim("4", "node:1", "RELATION", "located_in", object_ref="node:3")
    assert claim_keys(first)["canonical_claim_key"] != claim_keys(second)["canonical_claim_key"]
    assert claim_keys(first)["conflict_scope_key"] == claim_keys(second)["conflict_scope_key"]


def test_action_clause_is_background_but_declared_boolean_property_is_graph_claim():
    action = route_claim(claim_type="property", predicate="落实立德树人根本任务", value="是")
    boolean = route_claim(
        claim_type="property",
        predicate="世界遗产",
        value="是",
        schema={"boolean_properties": ["世界遗产"]},
    )
    assert action == {"claim_type": "BACKGROUND", "semantic_role": "ACTION"}
    assert boolean == {"claim_type": "PROPERTY", "semantic_role": ""}


def test_source_authority_keywords_are_utf8_and_conservative():
    assert infer_authority_class({"title": "中华人民共和国教育部"}) == "government"
    assert infer_authority_class({"title": "太原理工大学官网"}) == "official"
    assert infer_authority_class({"title": "新华社报道"}) == "news"
