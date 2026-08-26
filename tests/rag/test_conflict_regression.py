from src.rag.schemas import CandidateClaim, ClaimConflict, SemanticCompleteRequest
from src.rag.service.candidate_grouping_service import annotate_candidate_groups
from src.rag.service.conflict_classification_service import classify_candidate_group


def _payload():
    return SemanticCompleteRequest(
        scenic_id="4",
        node={"source_node_id": "947", "name": "test"},
        max_web_results=1,
        use_web_search=False,
        use_web_extractor=False,
    )


def _claim(claim_id, value, source_id="doc-1"):
    return CandidateClaim(
        claim_id=claim_id,
        claim_type="property",
        subject_node_id="947",
        predicate="始建年代",
        object_value=value,
        source_id=source_id,
        quote=value,
        evidence_status="supported",
    )


def test_same_source_same_value_is_not_conflict():
    payload = _payload()
    claims = [_claim("c1", "明代"), _claim("c2", "明代")]
    info = classify_candidate_group(claims, payload)
    assert info["conflict_class"] == "same_value"
    conflicts = [ClaimConflict(conflict_type="conflicting", claim_id="c1", predicate="始建年代", existing_value="明代", candidate_value="明代")]
    rows = annotate_candidate_groups(payload, claims, conflicts)
    assert rows[0]["conflict_class"] == "same_value"
    assert rows[0]["gap_status"] != "conflicted"


def test_different_values_remain_conflict():
    payload = _payload()
    claims = [_claim("c1", "明代"), _claim("c2", "清代")]
    conflicts = [ClaimConflict(conflict_type="conflicting", claim_id="c1", predicate="始建年代", existing_value="明代", candidate_value="清代")]
    rows = annotate_candidate_groups(payload, claims, conflicts)
    assert rows[0]["conflict_class"] == "conflicting"
    assert rows[0]["gap_status"] == "conflicted"
