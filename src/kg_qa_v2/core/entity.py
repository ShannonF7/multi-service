"""候选搜索与实体对齐契约。

候选搜索只回答“可能是谁”；实体对齐才回答“用户到底指谁”。
本模块不直接生成 Cypher，领域图谱通过 gateway 注入。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..models import ContextSnapshot, EntityReference
from .entity_reference import recall_terms
from .entity_role import infer_entity_role


class EntityCandidate(BaseModel):
    node_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    hierarchy_path: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    node_type: Optional[str] = None
    match_type: Optional[str] = None
    candidate_class: Literal["IDENTITY", "RELATED"] = "IDENTITY"
    entity_role: Literal["IDENTITY", "CONTENT", "UNKNOWN"] = "UNKNOWN"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    identity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    context_score: float = Field(default=0.0, ge=0.0, le=1.0)
    distance_m: Optional[float] = Field(default=None, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateSearchResult(BaseModel):
    mention: str
    domain_id: str
    candidates: List[EntityCandidate] = Field(default_factory=list)
    related_entities: List[EntityCandidate] = Field(default_factory=list)


class AlignmentDecision(BaseModel):
    mention: str
    status: Literal["LINKED", "AMBIGUOUS", "UNRESOLVED"]
    node_id: Optional[str] = None
    name: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    source: Literal["jev", "deterministic"] = "deterministic"
    candidates: List[EntityCandidate] = Field(default_factory=list)


class PublishedCandidateSearch:
    """当前领域已发布实体搜索的窄适配器。

    一个复杂提法会被拆成“完整表述、核心名称、必要的层级词”多个召回词；
    每次仍然只调用已发布图谱 gateway，结果在本层去重和重新评分。
    """

    def __init__(self, gateway: Callable[..., Any]) -> None:
        self.gateway = gateway

    def search(
        self,
        mention: str,
        *,
        context: ContextSnapshot,
        limit: int = 8,
        reference: Optional[EntityReference] = None,
    ) -> CandidateSearchResult:
        mention = str(mention or "").strip()
        if not mention:
            return CandidateSearchResult(mention="", domain_id=context.domain_id)

        terms = recall_terms(reference) if reference is not None else [mention]
        terms = [item for item in terms if item] or [mention]
        merged: dict[str, EntityCandidate] = {}
        related: dict[str, EntityCandidate] = {}
        matched_terms: dict[str, list[str]] = {}
        related_terms: dict[str, list[str]] = {}

        for term in terms:
            raw = self.gateway(
                domain_id=context.domain_id,
                mention=term,
                current_node_id=context.current_node_id,
                location_node_id=context.location_node_id,
                limit=max(1, min(int(limit), 20)),
                published_only=True,
            )
            if isinstance(raw, dict):
                values = raw.get("candidates", raw.get("items", []))
                related_values = raw.get("related_entities", raw.get("related_items", []))
            else:
                values = raw
                related_values = []

            for item in values or []:
                candidate = _normalize_candidate(
                    item, term=term, reference=reference, candidate_class="IDENTITY"
                )
                if candidate is not None:
                    _merge_candidate(merged, matched_terms, candidate, term)

            for item in related_values or []:
                candidate = _normalize_candidate(
                    item, term=term, reference=reference, candidate_class="RELATED"
                )
                if candidate is not None:
                    _merge_candidate(related, related_terms, candidate, term)

        ordered = sorted(
            merged.values(),
            key=lambda item: (item.score, len(matched_terms.get(item.node_id, []))),
            reverse=True,
        )
        related_ordered = sorted(
            related.values(),
            key=lambda item: (item.context_score, len(related_terms.get(item.node_id, []))),
            reverse=True,
        )
        for candidate in ordered:
            candidate.metadata["matched_terms"] = list(dict.fromkeys(matched_terms.get(candidate.node_id, [])))
        for candidate in related_ordered:
            candidate.metadata["matched_terms"] = list(dict.fromkeys(related_terms.get(candidate.node_id, [])))
        return CandidateSearchResult(
            mention=mention,
            domain_id=context.domain_id,
            candidates=ordered[:limit],
            related_entities=related_ordered[:limit],
        )


class EntityAlignmentService:
    """先用确定性门槛，必要时让 Jev 在有限候选中选择。"""

    def __init__(self, jev_client: Optional[Callable[..., Any]] = None) -> None:
        self.jev_client = jev_client

    def align(
        self,
        mention: str,
        candidates: List[EntityCandidate],
        *,
        context: ContextSnapshot,
        question: str = "",
    ) -> AlignmentDecision:
        ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
        if not ordered:
            return AlignmentDecision(
                mention=mention,
                status="UNRESOLVED",
                reason="当前领域已发布图谱中没有可用候选",
                candidates=[],
            )

        exact = [item for item in ordered if _same_name(mention, item.name) or any(_same_name(mention, alias) for alias in item.aliases)]
        if len(exact) == 1 and not _has_conflicting_scope(exact[0], context):
            return _linked(mention, exact[0], confidence=max(exact[0].score, 0.95), reason="名称或别名唯一匹配")

        has_semantic_candidate = any(
            item.match_type in {"semantic_evidence", "semantic_description"}
            for item in ordered
        )
        has_duplicate_name = len({_norm(item.name) for item in ordered}) < len(ordered)
        has_hierarchy_hint = _question_has_hierarchy_hint(question, ordered)
        if (
            self.jev_client is not None
            and len(ordered) > 1
            and not has_semantic_candidate
            and (not has_duplicate_name or has_hierarchy_hint)
        ):
            decision = self._jev_align(mention, ordered, context=context, question=question)
            if decision is not None:
                return decision

        # 候选召回已经限定在当前领域和已发布图谱；只有一个候选时不应再制造“多个候选”。
        # 低分候选仍保留最低门槛，避免空泛词被误绑定。
        if (
            len(ordered) == 1
            and ordered[0].score >= 0.55
            and not _has_conflicting_scope(ordered[0], context)
        ):
            return _linked(
                mention,
                ordered[0],
                confidence=ordered[0].score,
                reason="当前领域已发布图谱中只有一个可用候选",
            )

        strong_types = {"exact", "alias", "core_exact", "reverse_contains", "hierarchy"}
        if (
            len(ordered) == 1
            and ordered[0].score >= 0.75
            and ordered[0].match_type in strong_types
            and not _has_conflicting_scope(ordered[0], context)
        ):
            return _linked(mention, ordered[0], confidence=ordered[0].score, reason="核心名称或层级定语唯一匹配")
        if (
            len(ordered) > 1
            and ordered[0].score >= 0.75
            and ordered[0].match_type in strong_types
            and (ordered[0].score - ordered[1].score) >= 0.06
        ):
            return _linked(mention, ordered[0], confidence=ordered[0].score, reason="层级定语使候选分数和间隔达到安全门槛")
        if len(ordered) == 1 and ordered[0].score >= 0.92 and not _has_conflicting_scope(ordered[0], context):
            return _linked(mention, ordered[0], confidence=ordered[0].score, reason="唯一候选且置信度达到安全门槛")
        if len(ordered) > 1 and ordered[0].score >= 0.90 and (ordered[0].score - ordered[1].score) >= 0.18:
            return _linked(mention, ordered[0], confidence=ordered[0].score, reason="候选分数和间隔均达到安全门槛")
        return AlignmentDecision(
            mention=mention,
            status="AMBIGUOUS",
            confidence=ordered[0].score,
            reason="存在多个候选，当前证据不足以自动选择",
            candidates=ordered,
        )

    def _jev_align(self, mention: str, candidates: List[EntityCandidate], *, context: ContextSnapshot, question: str) -> Optional[AlignmentDecision]:
        payload = {
            "question": question,
            "mention": mention,
            "domain_id": context.domain_id,
            "current_node_id": context.current_node_id,
            "location_node_id": context.location_node_id,
            "candidates": [item.model_dump() for item in candidates],
        }
        prompt = (
            "请只在给定候选中判断用户指向的对象。禁止创造候选，禁止跨领域选择。\n"
            "如果上下文不足，返回 UNKNOWN。只能输出 JSON："
            '{"selected_node_id":"...或UNKNOWN","confidence":0.0,"reason":"..."}\n'
            f"输入：{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            if hasattr(self.jev_client, "decide"):
                raw = self.jev_client.decide("entity_alignment", payload)
            else:
                raw = self.jev_client(prompt=prompt, response_format="json")
            value = _as_dict(raw)
            selected = str(value.get("selected_node_id") or "UNKNOWN")
            confidence = float(value.get("confidence") or 0.0)
            candidate = next((item for item in candidates if item.node_id == selected), None)
            if candidate is None or confidence < 0.75:
                return None
            return _linked(mention, candidate, confidence=confidence, reason=str(value.get("reason") or "Jev 在有限候选中完成对齐"), source="jev")
        except Exception:
            return None


def _normalize_candidate(
    value: Any,
    *,
    term: str,
    reference: Optional[EntityReference],
    candidate_class: str = "IDENTITY",
) -> Optional[EntityCandidate]:
    if not isinstance(value, dict):
        return None
    node_id = str(value.get("node_id") or value.get("id") or "").strip()
    name = str(value.get("name") or value.get("node_name") or "").strip()
    if not node_id or not name:
        return None

    path = _string_list(value.get("hierarchy_path"))
    aliases = _string_list(value.get("aliases"))
    metadata = dict(value.get("metadata") or {})
    raw_properties = value.get("properties") or {}
    if not aliases and isinstance(raw_properties, dict):
        aliases = _string_list(raw_properties.get("aliases") or raw_properties.get("alias") or raw_properties.get("别名"))
    raw_score = value.get("score")
    if raw_score is None:
        raw_score = value.get("match_score")
    score = _score_value(raw_score)
    candidate_class = str(value.get("candidate_class") or candidate_class or "IDENTITY").upper()
    if candidate_class not in {"IDENTITY", "RELATED"}:
        candidate_class = "IDENTITY"

    normalized_term = _norm(term)
    normalized_name = _norm(name)
    normalized_aliases = {_norm(item) for item in aliases}
    if normalized_term == normalized_name:
        score = max(score, 1.0)
        match_type = "exact"
    elif normalized_term in normalized_aliases:
        score = max(score, 0.96)
        match_type = "alias"
    elif reference is not None and _norm(reference.core_mention) == normalized_name:
        score = max(score, 0.86)
        match_type = "core_exact"
    elif normalized_name and normalized_name in normalized_term:
        score = max(score, 0.72)
        match_type = "reverse_contains"
    elif normalized_term and normalized_term in normalized_name:
        score = max(score, 0.70)
        match_type = "contains"
    else:
        match_type = str(value.get("match_type") or "weak")

    qualifier_hits = _qualifier_hits(reference, path)
    if qualifier_hits and any(item.type in {"hierarchy", "area"} for item in (reference.qualifiers if reference else [])):
        score = min(score, min(0.98, 0.68 + 0.10 * qualifier_hits))
        match_type = "hierarchy"

    identity_score = _score_value(value.get("identity_score"))
    context_score = _score_value(value.get("context_score"))
    if candidate_class == "RELATED":
        context_score = max(context_score, score)
    else:
        identity_score = max(identity_score, score)

    return EntityCandidate(
        node_id=node_id,
        name=name,
        hierarchy_path=path,
        aliases=aliases,
        node_type=value.get("node_type"),
        match_type=match_type,
        candidate_class=candidate_class,
        entity_role=infer_entity_role(value),
        score=min(max(score, 0.0), 1.0),
        identity_score=min(max(identity_score, 0.0), 1.0),
        context_score=min(max(context_score, 0.0), 1.0),
        distance_m=_optional_float(value.get("distance_m")),
        metadata={
            **metadata,
            "raw_candidate": value,
            "matched_term": term,
            "qualifier_hits": qualifier_hits,
            "candidate_class": candidate_class,
        },
    )



def _merge_candidate(
    target: dict[str, EntityCandidate],
    matched_terms: dict[str, list[str]],
    candidate: EntityCandidate,
    term: str,
) -> None:
    key = candidate.node_id
    matched_terms.setdefault(key, []).append(term)
    previous = target.get(key)
    if previous is None or candidate.score > previous.score:
        target[key] = candidate
        return
    previous.aliases = list(dict.fromkeys([*previous.aliases, *candidate.aliases]))
    previous.hierarchy_path = list(dict.fromkeys([
        *previous.hierarchy_path, *candidate.hierarchy_path
    ]))
    previous.metadata.update(candidate.metadata)
    previous.identity_score = max(previous.identity_score, candidate.identity_score)
    previous.context_score = max(previous.context_score, candidate.context_score)

def _qualifier_hits(reference: Optional[EntityReference], path: list[str]) -> int:
    if reference is None:
        return 0
    path_norm = [_qualifier_key(item) for item in path]
    hits = 0
    for item in reference.qualifiers:
        qualifier = _qualifier_key(item.text)
        if qualifier and any(qualifier == part or qualifier in part or part in qualifier for part in path_norm):
            hits += 1
    return hits


def _qualifier_key(value: str) -> str:
    value = _norm(value)
    return re.sub(r"([一二三四五六七八九十0-9]+)楼$", r"\1层", value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return []
    return [str(item).strip() for item in values if str(item).strip() and str(item).strip().lower() != "none"]


def _score_value(value: Any) -> float:
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return score / 100.0 if score > 1.0 else score


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _linked(mention: str, candidate: EntityCandidate, *, confidence: float, reason: str, source: str = "deterministic") -> AlignmentDecision:
    return AlignmentDecision(
        mention=mention,
        status="LINKED",
        node_id=candidate.node_id,
        name=candidate.name,
        confidence=min(max(confidence, 0.0), 1.0),
        reason=reason,
        source=source,
        candidates=[candidate],
    )


def _norm(value: str) -> str:
    return re.sub(r"[\s，。！？?；;、的（）()【】\[\]]+", "", str(value or "")).casefold()


def _same_name(left: str, right: str) -> bool:
    return bool(_norm(left)) and _norm(left) == _norm(right)


def _has_conflicting_scope(candidate: EntityCandidate, context: ContextSnapshot) -> bool:
    if not context.current_node_name or not candidate.hierarchy_path:
        return False
    return context.current_node_name not in candidate.hierarchy_path and context.current_node_id is not None


def _question_has_hierarchy_hint(question: str, candidates: list[EntityCandidate]) -> bool:
    question_norm = _norm(question).replace("楼", "层")
    if not question_norm:
        return False
    for candidate in candidates:
        path_items = list(candidate.hierarchy_path)
        if path_items and _norm(path_items[-1]) == _norm(candidate.name):
            path_items = path_items[:-1]
        for item in path_items:
            value = _qualifier_key(item)
            if value and value in question_norm:
                return True
    return False


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError("Jev entity alignment client must return an object")
