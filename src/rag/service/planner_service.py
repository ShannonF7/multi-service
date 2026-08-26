"""Rule-based query planner for evidence-first semantic completion.

The first version deliberately avoids LLM planning. It turns each requested
property/relation gap into a focused retrieval question, so downstream RAG can
search the domain KB and web with a narrow target instead of one broad query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.rag.schemas import SemanticCompleteRequest


@dataclass
class CompletionQuestion:
    question_id: str
    target_kind: str
    target_field: str | None
    relation_intent: str | None
    temporal_role: str | None
    query_text: str
    search_terms: list[str]
    priority: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_terms(values: list[Any]) -> list[str]:
    terms: list[str] = []
    for value in values:
        text = _clean(value)
        if text and text not in terms:
            terms.append(text)
    return terms


def _subject_prefix(payload: SemanticCompleteRequest) -> str:
    scenic_name = _clean(payload.node.scenic_name if payload.subgraph_depth != 0 else "")
    parent_name = _clean(payload.node.parent_name if payload.subgraph_depth != 0 else "")
    node_name = _clean(payload.node.name)
    parts = _dedupe_terms([scenic_name, parent_name, node_name])
    return "".join(parts) or node_name


TEMPORAL_FIELD_ROLES = [
    ("construction_time", "始建时期或建造年代"),
    ("renovation_time", "修缮、重建或扩建年代"),
    ("current_status_time", "现存建筑或现存形制所属时期"),
    ("legend_time", "相关传说发生的历史时期"),
    ("protection_time", "文物保护公布时间"),
]

TEMPORAL_ROLE_NODE_TYPES = {"building", "poi", "object"}


def _is_temporal_field(field_name: str) -> bool:
    return _clean(field_name) in {"时期", "历史时期", "年代", "时间"}


def _uses_structural_temporal_roles(payload: SemanticCompleteRequest) -> bool:
    return _clean(payload.node.node_type).lower() in TEMPORAL_ROLE_NODE_TYPES


def _generic_temporal_query(payload: SemanticCompleteRequest) -> str:
    subject = _subject_prefix(payload)
    node_type = _clean(payload.node.node_type).lower()
    if node_type == "person":
        return f"{subject}的生卒年代或主要活动时期是什么？"
    if node_type in {"region", "scenicarea"}:
        return f"{subject}的形成、沿革或重要历史时期是什么？"
    return f"{subject}的历史时期或所属年代是什么？"


def _temporal_questions(payload: SemanticCompleteRequest, field_name: str) -> list[CompletionQuestion]:
    subject = _subject_prefix(payload)
    questions: list[CompletionQuestion] = []
    for role, label in TEMPORAL_FIELD_ROLES:
        questions.append(
            CompletionQuestion(
                question_id=f"prop:{field_name}:{role}",
                target_kind="property",
                target_field=field_name,
                relation_intent=None,
                temporal_role=role,
                query_text=f"{subject}的{label}是什么？",
                search_terms=_dedupe_terms([
                    payload.node.scenic_name if payload.subgraph_depth != 0 else "",
                    payload.node.parent_name if payload.subgraph_depth != 0 else "",
                    payload.node.name,
                    field_name,
                    role,
                ]),
                priority=82,
                metadata={"planner": "rule_template_v1", "temporal_role": role},
            )
        )
    return questions


def build_property_question(payload: SemanticCompleteRequest, field_name: str) -> CompletionQuestion:
    field_name = _clean(field_name)
    if _is_temporal_field(field_name) and _uses_structural_temporal_roles(payload):
        return _temporal_questions(payload, field_name)[0]
    subject = _subject_prefix(payload)
    node_name = _clean(payload.node.name)
    scenic_name = _clean(payload.node.scenic_name if payload.subgraph_depth != 0 else "")
    parent_name = _clean(payload.node.parent_name if payload.subgraph_depth != 0 else "")

    templates = {
        "历史时期": f"{subject}的历史时期是什么？",
        "时期": f"{subject}的历史时期或所属年代是什么？",
        "年代": f"{subject}的年代或历史时期是什么？",
        "时间": f"{subject}的建造时间、历史时期或相关时间是什么？",
        "建造时间": f"{subject}的建造时间是什么？",
        "始建时间": f"{subject}的始建时间是什么？",
        "修缮": f"{subject}是否有修缮、重建或保护信息？",
        "修缮时间": f"{subject}的修缮、重建或保护时间是什么？",
        "功能": f"{subject}的主要功能、用途或作用是什么？",
        "用途": f"{subject}的用途是什么？",
        "作用": f"{subject}的作用是什么？",
        "来源": f"有哪些资料记载了{subject}？",
        "出处": f"{subject}的信息来源或出处是什么？",
        "描述": f"{subject}的基本介绍是什么？",
        "简介": f"{subject}的基本介绍是什么？",
        "现状": f"{subject}的现存状态是什么？",
        "特色": f"{subject}的特色或价值是什么？",
    }
    query_text = _generic_temporal_query(payload) if _is_temporal_field(field_name) else templates.get(field_name, f"{subject}的{field_name}是什么？")
    return CompletionQuestion(
        question_id=f"prop:{field_name}",
        target_kind="property",
        target_field=field_name,
        relation_intent=None,
        temporal_role=None,
        query_text=query_text,
        search_terms=_dedupe_terms([scenic_name, parent_name, node_name, field_name]),
        priority=80,
        metadata={"planner": "rule_template_v1"},
    )


def build_relation_question(payload: SemanticCompleteRequest, relation: str) -> CompletionQuestion:
    relation = _clean(relation)
    subject = _subject_prefix(payload)
    node_name = _clean(payload.node.name)
    scenic_name = _clean(payload.node.scenic_name if payload.subgraph_depth != 0 else "")
    parent_name = _clean(payload.node.parent_name if payload.subgraph_depth != 0 else "")

    templates = {
        "位于": f"{subject}位于哪个区域、建筑或空间？",
        "归属": f"{subject}归属于哪个区域、建筑或体系？",
        "上级区域": f"{subject}的上级区域是什么？",
        "所属景区": f"{subject}属于哪个景区或片区？",
        "包含": f"{subject}内部包含哪些景点、文物、人物或陈列对象？",
        "相邻": f"{subject}附近或相邻的景点有哪些？",
        "展示": f"{subject}展示了哪些文物、人物或内容？",
        "陈列": f"{subject}陈列了哪些文物、人物或内容？",
        "关联": f"{subject}与哪些节点、人物、事件或建筑存在关联？",
        "通往": f"{subject}通往哪些区域、建筑或节点？",
    }
    query_text = templates.get(relation, f"{subject}和其他节点之间是否存在{relation}关系？")
    return CompletionQuestion(
        question_id=f"rel:{relation}",
        target_kind="relation",
        target_field=None,
        relation_intent=relation,
        temporal_role=None,
        query_text=query_text,
        search_terms=_dedupe_terms([scenic_name, parent_name, node_name, relation]),
        priority=75,
        metadata={"planner": "rule_template_v1"},
    )


def build_message_question(payload: SemanticCompleteRequest) -> CompletionQuestion | None:
    message = _clean(payload.message)
    if not message:
        return None
    node_name = _clean(payload.node.name)
    scenic_name = _clean(payload.node.scenic_name if payload.subgraph_depth != 0 else "")
    parent_name = _clean(payload.node.parent_name if payload.subgraph_depth != 0 else "")
    return CompletionQuestion(
        question_id="user:message",
        target_kind="fact",
        target_field=None,
        relation_intent=None,
        temporal_role=None,
        query_text=message,
        search_terms=_dedupe_terms([scenic_name, parent_name, node_name, message]),
        priority=60,
        metadata={"planner": "user_message_v1"},
    )


def plan_completion_questions(payload: SemanticCompleteRequest) -> list[CompletionQuestion]:
    questions: list[CompletionQuestion] = []
    seen: set[str] = set()

    def add(question: CompletionQuestion | None) -> None:
        if not question:
            return
        key = question.question_id or question.query_text
        if key in seen:
            return
        seen.add(key)
        questions.append(question)

    for field_name in payload.target_fields or []:
        if _clean(field_name):
            if _is_temporal_field(field_name) and _uses_structural_temporal_roles(payload):
                for temporal_question in _temporal_questions(payload, field_name):
                    add(temporal_question)
            else:
                add(build_property_question(payload, field_name))

    for relation in payload.relation_intents or []:
        if _clean(relation):
            add(build_relation_question(payload, relation))

    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    if bool(metadata.get("verify_existing_facts")):
        for existing in payload.existing_properties or []:
            field_name = _clean(getattr(existing, "key", ""))
            if field_name:
                question = build_property_question(payload, field_name)
                question.metadata = {
                    **question.metadata,
                    "verification": True,
                    "existing_value": _clean(getattr(existing, "value", "")),
                }
                add(question)
        for existing in payload.existing_relations or []:
            relation_name = _clean(getattr(existing, "relation_type", ""))
            if relation_name:
                question = build_relation_question(payload, relation_name)
                question.metadata = {
                    **question.metadata,
                    "verification": True,
                    "existing_target": _clean(getattr(existing, "target_name", "")),
                }
                add(question)

    add(build_message_question(payload))

    if not questions:
        subject = _subject_prefix(payload)
        add(
            CompletionQuestion(
                question_id="node:overview",
                target_kind="fact",
                target_field=None,
                relation_intent=None,
                temporal_role=None,
                query_text=f"{subject}的基本信息是什么？",
                search_terms=_dedupe_terms([payload.node.scenic_name, payload.node.parent_name, payload.node.name]),
                priority=40,
                metadata={"planner": "fallback_overview_v1"},
            )
        )

    questions.sort(key=lambda item: item.priority, reverse=True)
    return questions
