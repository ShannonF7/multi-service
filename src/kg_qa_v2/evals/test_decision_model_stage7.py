from src.kg_qa_v2.core.decision_model import LocalDecisionClient, TransformersDecisionClient, build_decision_model_from_env
from src.kg_qa_v2.models import ContextSnapshot
from src.kg_qa_v2.core.understanding import QuestionUnderstandingService


def test_local_decision_client_uses_openai_compatible_json_transport():
    calls = []

    def transport(body, base_url, api_key, timeout):
        calls.append((body, base_url, api_key, timeout))
        return {"choices": [{"message": {"content": '{"intent":"LOCATION","confidence":0.9,"needs_current_location":false}'}}]}

    client = LocalDecisionClient(
        base_url="http://127.0.0.1:8000/v1",
        model="Qwen3.5-4B",
        api_key="local-test",
        transport=transport,
    )
    result = client.decide("router", {"question": "校长办公室在哪里？"})
    assert result["intent"] == "LOCATION"
    assert set(result) == {"intent", "context_dependency", "confidence"}
    assert result["context_dependency"] == "NONE"
    assert calls[0][0]["response_format"] == {"type": "json_object"}
    assert calls[0][0]["model"] == "Qwen3.5-4B"


def test_local_decision_client_disables_retrieval_flags_for_casual_chat():
    calls = []

    def transport(body, *_):
        calls.append(body)
        return {
            "choices": [{
                "message": {
                    "content": '{"intent":"CASUAL","route":"CASUAL","needs_text":true,"needs_image":true,"context_dependent":true}'
                }
            }]
        }

    client = LocalDecisionClient(
        base_url="http://127.0.0.1:8000/v1",
        model="Qwen3.5-4B",
        transport=transport,
    )
    result = client.decide("router", {"question": "你好"})
    assert result == {"intent": "CASUAL", "context_dependency": "NONE", "confidence": 0.0}


def test_local_decision_model_is_used_by_understanding_without_changing_slot_ownership():
    def transport(body, *_):
        return {"choices": [{"message": {"content": '{"intent":"LOCATION","confidence":0.93}'}}]}

    decision = LocalDecisionClient(
        base_url="http://127.0.0.1:8000/v1",
        model="Qwen3.5-4B",
        transport=transport,
    )
    result = QuestionUnderstandingService(
        decision_model=decision,
        slot_client=lambda **_: {"targets": ["校长办公室"], "attribute": "location"},
    ).understand("校长办公室在哪里？", context=ContextSnapshot(domain_id="tyut"))
    assert result.route.source == "local"
    assert result.query_spec.intent == "LOCATION"
    assert result.query_spec.targets == ["校长办公室"]


def test_env_can_select_local_decision_model(monkeypatch):
    monkeypatch.setenv("DECISION_MODEL_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_DECISION_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LOCAL_DECISION_MODEL", "Qwen3.5-9B")
    client = build_decision_model_from_env()
    assert isinstance(client, LocalDecisionClient)
    assert client.model == "Qwen3.5-9B"
    assert client.timeout == 60.0


def test_transformers_decision_client_supports_injected_generator_without_loading_weights():
    prompts = []

    def generator(prompt):
        prompts.append(prompt)
        return "```json\n{\"intent\":\"CASUAL\",\"confidence\":0.88}\n```"

    client = TransformersDecisionClient(
        model_path="/models/not-loaded-in-test",
        generator=generator,
    )
    result = client.decide("router", {"question": "你好"})
    assert result == {"intent": "CASUAL", "context_dependency": "NONE", "confidence": 0.88}
    assert "你好" in prompts[0]


def test_env_can_select_in_process_transformers_backend(monkeypatch):
    monkeypatch.setenv("DECISION_MODEL_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_DECISION_BACKEND", "transformers")
    monkeypatch.setenv("LOCAL_DECISION_MODEL_PATH", "/models/qwen")
    client = build_decision_model_from_env()
    assert isinstance(client, TransformersDecisionClient)
    assert client.model_path == "/models/qwen"
