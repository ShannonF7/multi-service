from types import SimpleNamespace
from unittest.mock import Mock

from src.kg_qa.answer import (
    CallableAIAnswerGenerator,
    DashScopeQAClient,
    SafeQAAnswerGenerator,
)


def _kwargs():
    return {
        "question": "问题",
        "context": SimpleNamespace(name="领域", domain_id="SC_TEST"),
        "analysis": SimpleNamespace(knowledge_intents=["FACT"]),
        "entities": [],
        "plan": SimpleNamespace(steps=[]),
        "evidence": [],
    }


def test_callable_ai_explicitly_disables_web_search():
    client = Mock(return_value={"answer": "ok", "citations": []})
    generator = CallableAIAnswerGenerator(client)
    result = generator.generate(**_kwargs())
    assert result["answer"] == "ok"
    assert client.call_args.kwargs["web_search"] is False


def test_dashscope_response_json_is_parsed():
    content = '{"answer":"ok","citations":[{"evidence_id":"e1"}]}'
    assert DashScopeQAClient._parse_json(content)["answer"] == "ok"


def test_safe_generator_keeps_fallback_warning():
    primary = Mock()
    primary.generate.side_effect = RuntimeError("provider down")
    fallback = Mock()
    fallback.generate.return_value = {"answer": "fallback", "citations": []}
    result = SafeQAAnswerGenerator(primary, fallback).generate(**_kwargs())
    assert result["answer"] == "fallback"
    assert "provider down" in result["warnings"][0]

