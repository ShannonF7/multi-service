from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import unicodedata
from http import HTTPStatus
from typing import Any

from dashscope import Generation
from sqlalchemy import text

from src.core.config import settings
from src.rag.dependencies import ai_session_scope
from src.rag.service.claim_type_router import route_claim

logger = logging.getLogger(__name__)

ENTITY_TYPE_ALIASES = {
    "organization": "organization",
    "organisation": "organization",
    "机构": "organization",
    "组织": "organization",
    "university": "organization",
    "educationalinstitution": "organization",
    "governmentagency": "organization",
    "person": "person",
    "人物": "person",
    "location": "region",
    "city": "region",
    "place": "region",
    "region": "region",
    "地点": "region",
    "区域": "region",
    "building": "building",
    "建筑": "building",
    "event": "event",
    "事件": "event",
    "program": "program",
    "universityprogram": "program",
    "concept": "concept",
    "description": "description",
    "duration": "duration",
    "time": "time_literal",
    "date": "time_literal",
    "时间": "time_literal",
}


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalized(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def _entity_type(value: Any) -> str:
    raw = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    if "=" in raw:
        code, _label = raw.split("=", 1)
        if code.strip().startswith(("type_", "domain_")) or code.strip() == "ce5":
            raw = code.strip()
    return ENTITY_TYPE_ALIASES.get(raw, raw)


def _looks_like_non_entity_mention(mention: Any, raw_type: Any = "", mention_role: Any = "") -> bool:
    """Reject clause-like mentions before they can become graph entities.

    A sentence or process phrase may be valid evidence, but it is not a named
    entity. It must remain in the raw claim/evidence lineage instead of being
    persisted as a Node candidate.
    """
    text = unicodedata.normalize("NFKC", str(mention or "")).strip()
    role = str(mention_role or "").strip().upper()
    if not text:
        return True
    if role in {"CLAIM", "STATEMENT", "SENTENCE"}:
        return True
    if len(text) > 40 and re.search(r"[，。；：:,.!?！？]", text):
        return True
    clause_cues = ("是", "为", "之一", "储库", "表明", "说明", "导致", "形成", "演化", "过程", "阶段")
    return len(text) >= 8 and any(cue in text for cue in clause_cues)


def _surface_contains(content: Any, value: Any) -> bool:
    """只判断证据是否包含原文词面，不做翻译或语义推断。"""
    needle = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    haystack = unicodedata.normalize("NFKC", str(content or "")).casefold()
    needle = re.sub(r"\s+", " ", needle)
    haystack = re.sub(r"\s+", " ", haystack)
    return bool(needle and needle in haystack)


def _looks_like_internal_schema_key(value: Any) -> bool:
    raw = str(value or "").strip()
    return bool(raw and re.fullmatch(r"[A-Za-z][A-Za-z0-9_ -]*", raw) )


_BOOLEAN_CLAUSE_CUES = (
    "锚定", "落实", "推动", "打造", "建成", "构建", "营造", "承担", "获得",
    "努力", "矢志", "做出", "服务", "坚持", "开展", "形成", "入选", "存在",
    "达到", "突破", "培养", "建设",
)


def _looks_like_boolean_clause(predicate: str, object_text: str) -> bool:
    predicate = str(predicate or "").strip()
    object_text = str(object_text or "").strip()
    if object_text in {"是", "否"}:
        if predicate.startswith(("是否", "有无")):
            return False
        return (
            any(mark in predicate for mark in ("“", "”", '"', "‘", "’"))
            or any(cue in predicate for cue in _BOOLEAN_CLAUSE_CUES)
            or len(predicate) > 12
        )
    return (
        len(object_text) >= 4
        and (object_text == predicate or (object_text in predicate and len(predicate) > len(object_text) + 1))
        and any(cue in predicate for cue in _BOOLEAN_CLAUSE_CUES)
    )


def _tool_calls(response: Any) -> list[dict[str, Any]]:
    message = response.output.choices[0].message if response.output and response.output.choices else None
    calls = getattr(message, "tool_calls", None) if message else None
    result: list[dict[str, Any]] = []
    for call in calls or []:
        function = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
        name = function.get("name") if isinstance(function, dict) else getattr(function, "name", "")
        arguments = function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", "{}")
        try:
            payload = json.loads(arguments or "{}")
        except Exception:
            continue
        result.append({"name": str(name or ""), "arguments": payload})
    return result


ENTITY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "emit_entity_mention",
            "description": "记录证据中明确出现的实体提及。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mention_text": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "mention_role": {"type": "string"},
                    "quote": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["mention_text", "quote", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emit_event_mention",
            "description": "记录修建、迁移、毁损、任职等复杂事件提及；事件暂不直接成为正式图节点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "mention_text": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "mention_role": {"type": "string"},
                    "quote": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["mention_text", "quote", "confidence"],
            },
        },
    },
]


CLAIM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "emit_claim_batch",
            "description": "一次完整返回证据中全部原子属性与关系，不能只返回第一条。",
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_type": {"type": "string", "enum": ["property", "relation"]},
                                "subject_name": {"type": "string"},
                                "subject_type": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object_text": {"type": "string"},
                                "object_type": {"type": "string"},
                                "temporal_role": {"type": "string"},
                                "quote": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["claim_type", "subject_name", "predicate", "object_text", "quote", "confidence"],
                        },
                    },
                },
                "required": ["claims"],
            },
        },
    },
]


def materialize_evidence_units(
    *, growth_run_id: str, source_scenic_id: str, batch: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    with ai_session_scope() as db:
        for item in batch:
            content = str(item.get("content") or item.get("evidence_text") or "").strip()
            if not content:
                continue
            identity = {
                "source_scenic_id": str(source_scenic_id),
                "source_id": str(item.get("source_id") or ""),
                "chunk_id": int(item.get("id") or item.get("chunk_id") or 0),
                "chunk_hash": str(item.get("content_hash") or _hash(content)),
            }
            # EvidenceUnit belongs to one run. Reusing one mutable row across
            # runs would rewrite provenance for earlier raw claims.
            uid = _hash([growth_run_id, identity])
            metadata = {
                "asset_type": str(item.get("asset_type") or "text"),
                "asset_id": item.get("asset_id"),
                "image_url": item.get("image_url"),
                "bound_source_node_id": item.get("source_node_id"),
                "bound_source_node_name": item.get("source_node_name"),
                "consumption_id": item.get("consumption_id"),
            }
            row = db.execute(
                text(
                    """
                    insert into semantic_growth_evidence_units (
                        evidence_unit_uid, growth_run_id, consumption_id, source_scenic_id,
                        source_id, chunk_id, chunk_hash, source_type, source_title,
                        source_url, content, source_authority, metadata, updated_at
                    ) values (
                        :uid, :growth_run_id, :consumption_id, :source_scenic_id,
                        :source_id, :chunk_id, :chunk_hash, :source_type, :source_title,
                        :source_url, :content, :source_authority, cast(:metadata as jsonb), now()
                    ) on conflict (evidence_unit_uid) do update set
                        growth_run_id=excluded.growth_run_id,
                        consumption_id=excluded.consumption_id,
                        source_title=excluded.source_title,
                        source_url=excluded.source_url,
                        content=excluded.content,
                        metadata=semantic_growth_evidence_units.metadata || excluded.metadata,
                        updated_at=now()
                    returning *
                    """
                ),
                {
                    **identity,
                    "uid": uid,
                    "growth_run_id": str(growth_run_id),
                    "consumption_id": int(item["consumption_id"]) if item.get("consumption_id") else None,
                    "source_type": "image" if str(item.get("asset_type") or "") == "image" else "text",
                    "source_title": str(item.get("source_title") or item.get("title") or ""),
                    "source_url": str(item.get("source_url") or item.get("image_url") or ""),
                    "content": content,
                    "source_authority": float(item.get("source_authority") or (0.55 if str(item.get("asset_type") or "") == "image" else 0.9)),
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                },
            ).mappings().one()
            units.append(dict(row))
    return units


def _call_entities(unit: dict[str, Any]) -> list[dict[str, Any]]:
    content = str(unit.get("content") or "")[:12000]
    response = Generation.call(
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        result_format="message",
        tools=ENTITY_TOOLS,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是证据优先的知识图谱实体与事件发现器。只记录原文明确出现的实体和事件；"
                    "不要查询现有图谱，不要把文本强制套入既有节点或既有schema。每项必须带原文quote。"
                ),
            },
            {"role": "user", "content": f"证据单元：\n{content}"},
        ],
    )
    if getattr(response, "status_code", None) != HTTPStatus.OK:
        raise RuntimeError(f"entity discovery failed: {getattr(response, 'message', '')}")
    entities: list[dict[str, Any]] = []
    for call in _tool_calls(response):
        args = call["arguments"]
        mention = str(args.get("mention_text") or "").strip()
        quote = str(args.get("quote") or "").strip()
        if not mention or mention not in content or not quote or quote not in content:
            continue
        raw_type = _entity_type(args.get("entity_type") or ("event" if call["name"] == "emit_event_mention" else ""))
        mention_role = str(args.get("mention_role") or ("EVENT" if call["name"] == "emit_event_mention" else "ENTITY"))
        if _looks_like_non_entity_mention(mention, raw_type, mention_role):
            continue
        entities.append(
            {
                "mention_text": mention,
                "normalized_text": _normalized(mention),
                "raw_type": raw_type,
                "mention_role": mention_role,
                "quote": quote,
                "confidence": max(0.0, min(float(args.get("confidence") or 0.0), 1.0)),
            }
        )
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for entity in entities:
        key = (entity["normalized_text"], entity["mention_role"])
        if key not in unique or entity["confidence"] > unique[key]["confidence"]:
            unique[key] = entity
    return list(unique.values())


def _call_claims(
    unit: dict[str, Any], entities: list[dict[str, Any]], *, focus_entities: list[str] | None = None
) -> list[dict[str, Any]]:
    content = str(unit.get("content") or "")[:12000]
    entity_names = [item["mention_text"] for item in entities]
    focus = focus_entities or []
    metadata = unit.get("metadata") if isinstance(unit.get("metadata"), dict) else {}
    bound_name = str(metadata.get("bound_source_node_name") or "").strip()
    response = Generation.call(
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        result_format="message",
        tools=CLAIM_TOOLS,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是证据优先的原子知识发现器。基于原文和已发现实体，尽可能完整地输出属性与关系。"
                    "先保留原始中文谓词，不做实体合并、schema归一化、冲突判断或现有事实过滤。"
                    "每条必须有原文quote；复杂事件拆成属性、关系和temporal_role。"
                    "宾语是明确命名实体时必须输出关系，只有日期、数值、类别、描述等字面量才输出属性。"
                    "布尔值只能用于短的属性名或明确的“是否/有无”谓词；绝不能把完整动作、目标、计划或陈述句当作谓词再填“是/否”。" "不得把完整短语同时作为谓词和属性值输出；如果值只是谓词本身或其片段，应丢弃该条，除非它是名称、别名等明确字面量。"
                    "谓词只写证据中出现的原文语义，不得把英文 schema 键（如 founded_year、located_in）当作抽取结果；证据没有英文时不得输出英文谓词。"
                    "不得带年份、同年等时间词，不得用“与、和、是”为关系谓词；时间写入temporal_role。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"证据单元：\n{content}\n\n已发现实体：{json.dumps(entity_names, ensure_ascii=False)}"
                    f"\n图片已绑定主体（只在图片证据中可作为主体，不代表OCR出现）：{bound_name}"
                    f"\n需要补覆盖的实体：{json.dumps(focus, ensure_ascii=False)}"
                ),
            },
        ],
    )
    if getattr(response, "status_code", None) != HTTPStatus.OK:
        raise RuntimeError(f"claim discovery failed: {getattr(response, 'message', '')}")
    claims: list[dict[str, Any]] = []
    for call in _tool_calls(response):
        payloads = call["arguments"].get("claims") or []
        if not isinstance(payloads, list):
            continue
        for args in payloads:
            if not isinstance(args, dict):
                continue
            subject = str(args.get("subject_name") or "").strip()
            predicate = str(args.get("predicate") or "").strip()
            quote = str(args.get("quote") or "").strip()
            claim_type = str(args.get("claim_type") or "property").strip().lower()
            object_text = str(args.get("object_text") or "").strip()
            if claim_type not in {"property", "relation"}:
                continue
            if not subject or not predicate or not object_text or not quote or quote not in content:
                continue
            if subject not in content and subject not in quote and subject != bound_name:
                continue
            if object_text not in content and object_text not in quote:
                continue
            routed = route_claim(
                claim_type=claim_type,
                predicate=predicate,
                value=object_text,
                raw_text=quote,
            )
            routed_type = str(routed["claim_type"]).lower()
            # Keep action/goal/policy statements in raw evidence lineage, but
            # never send them through the ordinary graph-candidate lane.
            if routed_type == "background":
                claim_type = "background"
                semantic_role = routed["semantic_role"]
            else:
                claim_type = routed_type
                semantic_role = ""
                if _looks_like_boolean_clause(predicate, object_text):
                    continue
                if (
                    claim_type == "property"
                    and any(cue in predicate for cue in _BOOLEAN_CLAUSE_CUES)
                    and not re.search(r"[0-9]|年|月|日|亿元|万元|平方米|公里|项|人|号|次|%", object_text)
                ):
                    continue
            # 防止模型把内部英文 schema 键泄漏成“证据事实”。只有证据原文
            # 确实出现该英文词面时才允许保留；否则整条 claim 在落库前丢弃。
            if _looks_like_internal_schema_key(predicate) and not _surface_contains(content, predicate) and not _surface_contains(quote, predicate):
                continue
            claims.append(
                {
                    "subject_text": subject,
                    "subject_type": _entity_type(args.get("subject_type") or ""),
                    "claim_type": claim_type,
                    "semantic_role": semantic_role,
                    "raw_predicate": predicate,
                    "predicate_surface": predicate,
                    "object_text": object_text,
                    "object_type": _entity_type(args.get("object_type") or ""),
                    "temporal_role": str(args.get("temporal_role") or ""),
                    "quote": quote,
                    "confidence": max(0.0, min(float(args.get("confidence") or 0.0), 1.0)),
                    "extraction_pass": "COVERAGE" if focus else "CLAIM",
                }
            )
    return claims


def _discover_one(unit: dict[str, Any]) -> dict[str, Any]:
    try:
        metadata = unit.get("metadata") if isinstance(unit.get("metadata"), dict) else {}
        bound_name = str(metadata.get("bound_source_node_name") or "").strip()
        if str(unit.get("source_type") or "") == "image" and bound_name:
            # The image upload has already been bound to its graph subject.
            # OCR is evidence text, not a safe entity-naming mechanism.
            entities = [
                {
                    "mention_text": bound_name,
                    "normalized_text": _normalized(bound_name),
                    "raw_type": "",
                    "mention_role": "BOUND_IMAGE_SUBJECT",
                    "quote": str(unit.get("content") or "")[:500],
                    "confidence": 1.0,
                }
            ]
        else:
            entities = _call_entities(unit)
        claims = _call_claims(unit, entities)
        entity_types = {
            _normalized(item["mention_text"]): item.get("raw_type") or ""
            for item in entities
        }
        for claim in claims:
            claim["subject_type"] = claim.get("subject_type") or entity_types.get(_normalized(claim["subject_text"]), "")
            if claim["claim_type"] == "relation":
                claim["object_type"] = claim.get("object_type") or entity_types.get(_normalized(claim["object_text"]), "")
        if str(unit.get("source_type") or "") == "image":
            # OCR-only image processing may suggest properties of the bound
            # subject, but must not mint relation-target entities.
            claims = [
                item for item in claims
                if item["claim_type"] == "property" and item["subject_text"] == bound_name
            ]
        covered = {
            _normalized(value)
            for claim in claims
            for value in (claim["subject_text"], claim["object_text"] if claim["claim_type"] == "relation" else "")
            if value
        }
        uncovered = [item["mention_text"] for item in entities if item["mention_role"] != "EVENT" and item["normalized_text"] not in covered]
        if uncovered:
            coverage_claims = _call_claims(unit, entities, focus_entities=uncovered[:12])
            for claim in coverage_claims:
                claim["subject_type"] = claim.get("subject_type") or entity_types.get(_normalized(claim["subject_text"]), "")
                if claim["claim_type"] == "relation":
                    claim["object_type"] = claim.get("object_type") or entity_types.get(_normalized(claim["object_text"]), "")
            claims.extend(coverage_claims)
        unique_claims: dict[str, dict[str, Any]] = {}
        for claim in claims:
            key = _hash(
                [
                    _normalized(claim["subject_text"]),
                    claim["claim_type"],
                    _normalized(claim["raw_predicate"]),
                    _normalized(claim["object_text"]),
                    _normalized(claim["temporal_role"]),
                ]
            )
            if key not in unique_claims or claim["confidence"] > unique_claims[key]["confidence"]:
                unique_claims[key] = claim
        return {"unit": unit, "entities": entities, "claims": list(unique_claims.values()), "error": None}
    except Exception as exc:
        logger.warning("open discovery failed for evidence unit %s: %s", unit.get("evidence_unit_uid"), exc)
        return {"unit": unit, "entities": [], "claims": [], "error": str(exc)}


def discover_evidence_units(units: list[dict[str, Any]], *, max_concurrency: int = 4) -> list[dict[str, Any]]:
    workers = max(1, min(int(max_concurrency or 1), 8, len(units) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_discover_one, units))


def persist_raw_discovery(*, growth_run_id: str, discoveries: list[dict[str, Any]]) -> dict[str, Any]:
    entity_count = claim_count = 0
    persisted_claims: list[dict[str, Any]] = []
    with ai_session_scope() as db:
        for discovery in discoveries:
            unit = discovery["unit"]
            unit_id = int(unit["id"])
            for entity in discovery.get("entities") or []:
                uid = _hash([growth_run_id, unit["evidence_unit_uid"], entity["normalized_text"], entity["mention_role"]])
                row = db.execute(
                    text(
                        """
                        insert into semantic_growth_raw_entities (
                            raw_entity_uid, growth_run_id, evidence_unit_id, mention_text,
                            normalized_text, raw_type, mention_role, quote, confidence, updated_at
                        ) values (
                            :uid, :growth_run_id, :unit_id, :mention_text,
                            :normalized_text, :raw_type, :mention_role, :quote, :confidence, now()
                        ) on conflict (raw_entity_uid) do update set
                            quote=excluded.quote, confidence=greatest(semantic_growth_raw_entities.confidence, excluded.confidence),
                            raw_type=coalesce(nullif(excluded.raw_type, ''), semantic_growth_raw_entities.raw_type), updated_at=now()
                        returning id
                        """
                    ),
                    {"uid": uid, "growth_run_id": growth_run_id, "unit_id": unit_id, **entity},
                ).scalar_one()
                entity["id"] = int(row)
                entity_count += 1
            for claim in discovery.get("claims") or []:
                uid = _hash(
                    [growth_run_id, unit["evidence_unit_uid"], claim["subject_text"], claim["claim_type"], claim["raw_predicate"], claim["object_text"], claim["temporal_role"]]
                )
                row = db.execute(
                    text(
                        """
                        insert into semantic_growth_raw_claims (
                            raw_claim_uid, growth_run_id, evidence_unit_id, extraction_pass,
                            subject_text, subject_type, claim_type, raw_predicate, object_text,
                            object_type, temporal_role, quote, confidence, metadata, updated_at
                        ) values (
                            :uid, :growth_run_id, :unit_id, :extraction_pass,
                            :subject_text, :subject_type, :claim_type, :raw_predicate, :object_text,
                            :object_type, :temporal_role, :quote, :confidence,
                            cast(:metadata as jsonb), now()
                        ) on conflict (raw_claim_uid) do update set
                            quote=excluded.quote, confidence=greatest(semantic_growth_raw_claims.confidence, excluded.confidence),
                            updated_at=now()
                        returning id
                        """
                    ),
                    {"uid": uid, "growth_run_id": growth_run_id, "unit_id": unit_id, **claim, "metadata": json.dumps(claim.get("metadata") or {}, ensure_ascii=False)},
                ).scalar_one()
                persisted_claims.append({**claim, "id": int(row), "raw_claim_uid": uid, "evidence_unit": unit})
                claim_count += 1
    return {
        "entity_count": entity_count,
        "claim_count": claim_count,
        "claims": persisted_claims,
        "error_count": sum(1 for item in discoveries if item.get("error")),
        "errors": [item["error"] for item in discoveries if item.get("error")],
    }
