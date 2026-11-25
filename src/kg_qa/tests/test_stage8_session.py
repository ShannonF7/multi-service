from src.kg_qa.schemas import InputAnalysis, QARequest
from src.kg_qa.planner import build_query_plan
from src.kg_qa.session import ConversationSessionStore
from src.kg_qa.trace import TraceCollector


def test_session_compresses_old_turns_into_summary():
    store = ConversationSessionStore(max_turns=2, keep_turns=1)
    context = store.open("s1")
    store.append(context, "第一问", "第一答")
    store.append(context, "第二问", "第二答")
    pending = store.open("s1")
    assert pending.needs_compression
    compact = store.compress(pending)
    assert compact.compressed
    assert "第一问" in compact.summary
    assert len(compact.turns) == 1


def test_trace_emits_started_event_before_long_callback():
    events = []
    trace = TraceCollector()
    trace.run("CONTEXT_COMPRESSION", lambda: "ok", on_event=events.append)
    assert [item.status for item in events] == ["STARTED", "COMPLETED"]


def test_image_question_creates_image_search_step():
    request = QARequest(domain_id="a", message="展示行勉楼的图片")
    analysis = InputAnalysis(
        query_modality="TEXT", target_modality="TEXT", modality_intent="TEXT_TO_TEXT",
        knowledge_intents=["IMAGE"], mentions=["行勉楼"], has_text=True,
    )
    plan = build_query_plan(request, analysis, [])
    assert any(step.type == "IMAGE_SEARCH" for step in plan.steps)


def test_hierarchy_question_creates_bounded_neighborhood_step():
    request = QARequest(domain_id="a", message="行勉楼包含哪些楼座？", current_node_id="n1")
    analysis = InputAnalysis(
        query_modality="TEXT", target_modality="TEXT", modality_intent="TEXT_TO_TEXT",
        knowledge_intents=["HIERARCHY"], mentions=["行勉楼"], has_text=True,
    )
    plan = build_query_plan(request, analysis, [])
    assert any(step.type == "GRAPH_NEIGHBORHOOD" for step in plan.steps)


def test_context_keeps_last_linked_entity_for_pronoun_followup():
    from src.kg_qa.input_analyzer import analyze_input
    from src.kg_qa.service import _apply_context_mentions

    store = ConversationSessionStore(max_turns=4)
    context = store.open("pronoun-session")
    store.append(context, "行勉楼在哪里？", "行勉楼位于东部教学区。", entities=[{"name": "行勉楼"}])
    current = store.open("pronoun-session")
    analysis = analyze_input(QARequest(domain_id="4", message="他在哪里？"))
    assert analysis.mentions == []
    resolved = _apply_context_mentions(analysis, "他在哪里？", current)
    assert resolved.mentions == ["行勉楼"]


def test_graph_profile_evidence_contains_coordinates_and_assets():
    from src.kg_qa.evidence import fuse_evidence

    result = {
        "domain_id": "4",
        "node_id": "963",
        "root": {
            "node_id": "963", "name": "行勉楼", "lng": 112.723844, "lat": 37.750420,
            "assets": [{"asset_id": "1663", "image_url": "/zhangbi_feedback/wiki_custom/kg-qa/assets/298/"}],
        },
        "facts": [], "relations": [],
    }
    evidence = fuse_evidence("4", [result], [], limit=4)
    profile = next(item for item in evidence if item.evidence_id.endswith(":profile"))
    assert "112.723844" in profile.content
    assert profile.image_url.endswith("/298/")
    assert profile.metadata["assets"][0]["asset_id"] == "1663"
