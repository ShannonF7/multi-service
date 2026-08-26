"""Evidence-first semantic completion service.

This service owns the AI/RAG side of semantic completion:
search -> extract readable evidence -> extract CandidateClaim -> verify evidence -> detect conflicts.
It does not write A-side NodeProperty/NodeRelation records.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from dashscope import Generation

from src.core.config import settings
from src.rag.schemas import (
    CandidateClaim,
    ClaimConflict,
    EvidenceChunk,
    SemanticCompleteRequest,
    SemanticCompleteResponse,
)
from src.rag.service.semantic_candidate_store import persist_semantic_candidates
from src.rag.service.domain_kb_service import search_domain_kb
from src.rag.service.evidence_store import persist_semantic_evidence_items, persist_semantic_completion_questions
from src.rag.service.planner_service import CompletionQuestion, plan_completion_questions
from src.rag.service.value_normalization_service import normalize_candidate_claims, canonical_predicate
from src.rag.service.candidate_grouping_service import annotate_candidate_groups, assign_candidate_group_keys
from src.rag.service.conflict_classification_service import classify_conflicts
from src.rag.service.entity_resolution_service import resolve_candidate_entities
from src.rag.service.risk_classification_service import apply_recommendation_and_risk
from src.rag.service.source_weighting_service import chunk_source_weight, source_weight
from src.rag.service.gap_status_service import update_semantic_gap_status
from src.rag.service.graph_discovery_service import augment_questions_with_graph_discovery

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEMANTIC_LOG_DIR = PROJECT_ROOT / "logs"
SEMANTIC_LOG_DIR.mkdir(parents=True, exist_ok=True)
SEMANTIC_LOG_FILE = SEMANTIC_LOG_DIR / "semantic_completion.log"
semantic_logger = logging.getLogger("rag.semantic_completion")
semantic_logger.setLevel(logging.INFO)
if not any(getattr(h, "baseFilename", "") == str(SEMANTIC_LOG_FILE) for h in semantic_logger.handlers):
    handler = logging.FileHandler(SEMANTIC_LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    semantic_logger.addHandler(handler)


def semantic_log(trace_id: str, stage: str, **data: Any) -> None:
    record = {"trace_id": trace_id, "stage": stage, **data}
    semantic_logger.info("[SemanticComplete] %s", json.dumps(record, ensure_ascii=False, default=str))


RELATION_RULES = {
    "位于": {"cardinality": "single", "conflict_policy": "exclusive"},
    "归属": {"cardinality": "single", "conflict_policy": "exclusive"},
    "上级区域": {"cardinality": "single", "conflict_policy": "exclusive"},
    "所属景区": {"cardinality": "single", "conflict_policy": "exclusive"},
    "包含": {"cardinality": "multi", "conflict_policy": "append"},
    "相邻": {"cardinality": "multi", "conflict_policy": "append"},
    "展示": {"cardinality": "multi", "conflict_policy": "append"},
    "关联": {"cardinality": "multi", "conflict_policy": "append"},
    "通往": {"cardinality": "multi", "conflict_policy": "append"},
    "陈列": {"cardinality": "multi", "conflict_policy": "append"},
    "呈现": {"cardinality": "multi", "conflict_policy": "append"},
}

GENERIC_TERMS = {"ai", "AI", "补全", "语义补全", "检索", "搜索", "查询", "信息", "资料", "介绍", "相关", "来源", "属性", "关系"}


def _split_query_terms(text: str) -> list[str]:
    text = str(text or "").strip().lower()
    if not text:
        return []
    raw_parts = re.split(r"[\s,，。；;、/|]+", text)
    raw_parts.extend(re.findall(r"[《〈](.*?)[》〉]", text))
    terms: list[str] = []
    generic = {x.lower() for x in GENERIC_TERMS}
    for part in raw_parts:
        cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff_-]+", "", part).strip().lower()
        if len(cleaned) >= 2 and cleaned not in generic and cleaned not in terms:
            terms.append(cleaned)
    return terms[:24]


def _normalize_url(raw_url: str) -> str:
    url = html.unescape(str(raw_url or "").strip())
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    if parsed.scheme in {"http", "https"}:
        return url
    if url.startswith("//"):
        return f"https:{url}"
    return url



def get_graph_context(payload: SemanticCompleteRequest) -> dict[str, Any]:
    graph = getattr(payload, "graph_context", None) or {}
    return graph if isinstance(graph, dict) else {}


def get_graph_context_terms(payload: SemanticCompleteRequest) -> list[str]:
    if payload.subgraph_depth == 0:
        return []
    graph = get_graph_context(payload)
    terms: list[str] = []
    for item in graph.get("search_terms") or []:
        text = str(item or "").strip()
        if text and text not in terms:
            terms.append(text)
    if terms:
        return terms[:12]
    for item in graph.get("nodes") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("name") or "").strip()
        if text and text not in terms:
            terms.append(text)
        if len(terms) >= 12:
            break
    return terms


def build_graph_context_prompt(payload: SemanticCompleteRequest) -> str:
    graph = get_graph_context(payload)
    if not graph:
        return "none"
    nodes = graph.get("nodes") or []
    relations = graph.get("relations") or []
    slim_nodes = []
    for item in nodes[:30]:
        if isinstance(item, dict):
            slim_nodes.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "node_type": item.get("node_type"),
                "depth": item.get("depth"),
                "role": item.get("role"),
            })
    slim_relations = []
    for item in relations[:60]:
        if isinstance(item, dict):
            slim_relations.append({
                "source_name": item.get("source_name"),
                "relation_type": item.get("relation_type"),
                "relation_category": item.get("relation_category"),
                "target_name": item.get("target_name"),
            })
    return json.dumps({
        "scope": graph.get("scope"),
        "depth": graph.get("depth"),
        "nodes": slim_nodes,
        "relations": slim_relations,
    }, ensure_ascii=False)



def get_source_scope(payload: SemanticCompleteRequest) -> set[str]:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    raw = metadata.get("source_scope") or metadata.get("evidence_source_scope")
    if raw is None:
        values = ["provided_evidence"]
        if payload.subgraph_depth != 0:
            values.append("domain_kb")
        if payload.use_web_search:
            values.append("web_search")
        if payload.use_web_extractor:
            values.append("web_extractor")
        return set(values)
    if isinstance(raw, str):
        raw_values = re.split(r"[\s,?;?|]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        raw_values = list(raw)
    else:
        raw_values = []
    aliases = {
        "provided": "provided_evidence",
        "local_kb": "domain_kb",
        "kb": "domain_kb",
        "web": "web_search",
        "extractor": "web_extractor",
    }
    result: set[str] = set()
    for item in raw_values:
        text_value = str(item or "").strip().lower()
        if not text_value:
            continue
        result.add(aliases.get(text_value, text_value))
    retrieval_scope = str(metadata.get("retrieval_scope") or "").strip().lower()
    if payload.subgraph_depth == 0 or retrieval_scope in {"self", "self_web", "node"}:
        result.discard("domain_kb")
        if payload.use_web_search:
            result.add("web_search")
        if payload.use_web_extractor:
            result.add("web_extractor")
    return result or {"provided_evidence"}


def source_enabled(payload: SemanticCompleteRequest, name: str) -> bool:
    return name in get_source_scope(payload)


def get_question_batch_size(payload: SemanticCompleteRequest) -> int:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    mode = completion_mode(payload)
    default_size = 8 if mode in {"deep", "web", "full", "batch"} else (5 if mode == "standard" else 3)
    try:
        # max_questions used to truncate the plan. Keep accepting it as a
        # legacy batch-size hint so old callers no longer lose later gaps.
        batch_size = int(metadata.get("question_batch_size") or metadata.get("max_questions") or default_size)
    except Exception:
        batch_size = default_size
    return max(1, min(batch_size, 20))


def batch_completion_questions(
    payload: SemanticCompleteRequest,
    questions: list[CompletionQuestion],
) -> list[list[CompletionQuestion]]:
    batch_size = get_question_batch_size(payload)
    return [questions[index:index + batch_size] for index in range(0, len(questions), batch_size)]


def plan_sources(payload: SemanticCompleteRequest) -> list[str]:
    scope = get_source_scope(payload)
    sources: list[str] = []
    if payload.evidence and "provided_evidence" in scope:
        sources.append("provided_evidence")
    if "domain_kb" in scope:
        sources.append("domain_kb")
    if payload.use_web_search and "web_search" in scope:
        sources.append("web_search")
    if payload.use_web_extractor and "web_extractor" in scope:
        sources.append("web_extractor")
    if not sources:
        sources.append("provided_evidence" if payload.evidence else "none")
    return sources


def build_query(payload: SemanticCompleteRequest) -> str:
    parts: list[str] = []
    for item in [
        payload.node.name,
        payload.node.parent_name if payload.subgraph_depth != 0 else "",
        payload.node.scenic_name if payload.subgraph_depth != 0 else "",
        payload.message,
        payload.source_note,
        " ".join(payload.target_fields or []),
        " ".join(payload.relation_intents or []),
        " ".join(get_graph_context_terms(payload)),
    ]:
        item = str(item or "").strip()
        if item and item not in parts:
            parts.append(item)
    return " ".join(parts).strip()


def web_search(query: str, limit: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use DashScope built-in web search to collect URLs/snippets."""
    if not query:
        return [], {"reason": "empty_query"}
    messages = [
        {
            "role": "user",
            "content": (
                "搜索与以下知识图谱节点补全任务直接相关的中文资料。"
                "优先官方网站、机构官网和可直接访问的原始来源，降低百科聚合页优先级。"
                "只需要返回相关网页，不要编造。检索目标："
                + query
            ),
        }
    ]
    try:
        response = Generation.call(
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            messages=messages,
            result_format="message",
            enable_search=True,
            search_options={
                "enable_source": True,
                "forced_search": True,
                "search_strategy": "max",
            },
        )
        if getattr(response, "status_code", None) != HTTPStatus.OK:
            return [], {"error": f"{getattr(response, 'code', '')}: {getattr(response, 'message', '')}"}
        search_info = getattr(response.output, "search_info", None)
        raw_results = []
        if isinstance(search_info, dict):
            raw_results = search_info.get("search_results") or []
        elif search_info is not None:
            raw_results = getattr(search_info, "search_results", None) or []
        results: list[dict[str, Any]] = []
        for item in raw_results[:limit]:
            if not isinstance(item, dict):
                continue
            url = _normalize_url(item.get("url") or item.get("source_url") or item.get("link") or "")
            title = str(item.get("title") or item.get("site_name") or item.get("site") or url).strip()
            content = str(item.get("snippet") or item.get("content") or item.get("summary") or title).strip()
            if not url and not content:
                continue
            results.append(
                {
                    "title": title,
                    "content": content,
                    "quote": content,
                    "source": item.get("site_name") or item.get("site") or "DashScope WebSearch",
                    "source_type": "web_search",
                    "source_url": url,
                    "score": float(item.get("score") or 0.6),
                }
            )
        return results, {"search_info": search_info if isinstance(search_info, dict) else str(search_info or "")[:1000]}
    except Exception as exc:
        logger.warning("web_search failed: %s", exc, exc_info=True)
        return [], {"error": str(exc)}


def build_web_query_variants(
    payload: SemanticCompleteRequest,
    question: CompletionQuestion,
) -> list[str]:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    try:
        limit = max(1, min(int(metadata.get("web_query_variants_per_question") or 1), 3))
    except Exception:
        limit = 1
    variants = [str(question.query_text or "").strip()]
    if limit > 1:
        target = str(question.target_field or question.relation_intent or "").strip()
        subject = str(payload.node.name or "").strip()
        verification_query = " ".join(
            value for value in [subject, target, "\u5b98\u65b9\u8d44\u6599 \u6570\u636e \u4e0d\u540c\u6765\u6e90"] if value
        ).strip()
        if verification_query and verification_query not in variants:
            variants.append(verification_query)
    return variants[:limit]


def extract_web_page(url: str, *, timeout: int = 8) -> dict[str, Any] | None:
    """Extract readable text from one URL.

    This is the local fallback/replacement point for Aliyun WebExtractor.
    """
    url = _normalize_url(url)
    if not url or not url.startswith(("http://", "https://")):
        return None
    headers = {
        "User-Agent": "Mozilla/5.0 AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            return None
        resp.encoding = resp.apparent_encoding or resp.encoding
        soup = BeautifulSoup(resp.text or "", "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
            tag.decompose()
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")[:200]
        text = soup.get_text("\n")
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if len(line) >= 8]
        content = "\n".join(lines)
        content = content[:6000]
        if not content:
            return None
        return {
            "title": title or url,
            "content": content,
            "quote": content[:500],
            "source": urlparse(url).netloc,
            "source_type": "web_extractor",
            "source_url": url,
            "score": 0.85,
        }
    except Exception as exc:
        logger.info("extract_web_page failed for %s: %s", url, exc)
        return None


def _normalize_match_term(term: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff_-]+", "", str(term or "").lower()).strip()


def score_source_relevance(
    source: dict[str, Any],
    *,
    required_terms: list[str],
    scope_terms: list[str],
    scenic_terms: list[str] | None = None,
    target_terms: list[str] | None = None,
    rank: int | None = None,
) -> dict[str, Any]:
    raw_haystack = " ".join(str(source.get(k) or "") for k in ("title", "content", "quote", "source", "source_url")).lower()
    normalized_haystack = _normalize_match_term(raw_haystack)
    score = 0.0
    matched: list[str] = []
    reasons: list[str] = []

    def add_matches(terms: list[str], weight: float, label: str) -> int:
        nonlocal score
        hits = 0
        for term in terms or []:
            cleaned = _normalize_match_term(term)
            if len(cleaned) < 2 or cleaned in matched:
                continue
            if cleaned in normalized_haystack:
                matched.append(cleaned)
                score += weight
                hits += 1
        if hits:
            reasons.append(f"{label}:{hits}")
        return hits

    add_matches(required_terms, 3.0, "node")
    add_matches(scenic_terms or [], 1.6, "scenic")
    add_matches(scope_terms, 1.0, "context")
    add_matches(target_terms or [], 0.35, "target")

    domain = urlparse(str(source.get("source_url") or "")).netloc.lower()
    authority_domains = ("gov.cn", "xinhuanet.com", "people.com.cn", "stdaily.com", "cntour2.com")
    if any(domain.endswith(d) for d in authority_domains):
        score += 0.8
        reasons.append("authority_domain")
    if rank is not None and rank <= 2:
        score += 0.4
        reasons.append("top_rank")

    return {
        "score": round(score, 2),
        "matched_terms": matched[:16],
        "reasons": reasons,
        "domain": domain,
    }


def source_matches_scope(source: dict[str, Any], required_terms: list[str], scope_terms: list[str]) -> bool:
    return score_source_relevance(source, required_terms=required_terms, scope_terms=scope_terms).get("score", 0) >= 1.0


def _question_terms(question: CompletionQuestion) -> list[str]:
    return _split_query_terms(" ".join([question.query_text, " ".join(question.search_terms or [])]))


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _clean_source_url(value: Any) -> str:
    url = str(value or "").strip()
    if url.startswith("domain-kb://"):
        return ""
    return url


def _tag_source_with_question(source: dict[str, Any], question: CompletionQuestion) -> dict[str, Any]:
    data = dict(source)
    data["question_id"] = question.question_id
    data["target_kind"] = question.target_kind
    data["target_field"] = question.target_field
    data["relation_intent"] = question.relation_intent
    data["query_text"] = question.query_text
    return data



def _question_local_coverage(payload: SemanticCompleteRequest, question: CompletionQuestion, sources: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    try:
        min_hits = int(metadata.get("local_coverage_min_hits") or 2)
    except Exception:
        min_hits = 2
    try:
        min_score = float(metadata.get("local_coverage_min_score") or 1.2)
    except Exception:
        min_score = 1.2
    local_sources = [item for item in sources if str(item.get("source_type") or "").startswith("domain_kb") or str(item.get("source_type") or "") in {"provided", "provided_evidence"}]
    best_score = max((float(item.get("score") or 0.0) for item in local_sources), default=0.0)
    has_locator = any(item.get("source_doc_id") or item.get("doc_id") or item.get("chunk_id") or item.get("source_url") for item in local_sources)
    subject_terms = _split_query_terms(payload.node.name or "")
    matched_subject_terms = sorted({
        term
        for term in subject_terms
        if any(
            term in " ".join([
                str(item.get("title") or ""),
                str(item.get("source") or ""),
                str(item.get("content") or ""),
                str(item.get("quote") or ""),
            ]).lower()
            for item in local_sources
        )
    })
    subject_matched = not subject_terms or bool(matched_subject_terms)
    covered = len(local_sources) >= min_hits and best_score >= min_score and has_locator and subject_matched
    return {
        "covered": covered,
        "local_hits": len(local_sources),
        "best_score": round(best_score, 3),
        "min_hits": min_hits,
        "min_score": min_score,
        "has_locator": has_locator,
        "subject_matched": subject_matched,
        "matched_subject_terms": matched_subject_terms,
    }


def collect_evidence(payload: SemanticCompleteRequest, trace_id: str = "", questions: list[CompletionQuestion] | None = None) -> tuple[list[EvidenceChunk], list[dict[str, Any]]]:
    if questions is None:
        questions = plan_completion_questions(payload)
    fallback_query = build_query(payload)
    required_terms = _split_query_terms(payload.node.name or "")
    scenic_terms = [] if payload.subgraph_depth == 0 else _split_query_terms(payload.node.scenic_name or payload.scenic_id or "")
    diagnostics: list[dict[str, Any]] = []
    graph_context = get_graph_context(payload)
    semantic_log(
        trace_id,
        "collect_start",
        query=fallback_query,
        question_count=len(questions),
        questions=[{
            "question_id": q.question_id,
            "target_kind": q.target_kind,
            "target_field": q.target_field,
            "relation_intent": q.relation_intent,
            "query_text": q.query_text,
            "priority": q.priority,
        } for q in questions],
        required_terms=required_terms,
        scenic_terms=scenic_terms,
        graph_scope=graph_context.get("scope"),
        graph_node_count=len(graph_context.get("nodes") or []),
        graph_relation_count=len(graph_context.get("relations") or []),
        use_web_search=payload.use_web_search,
        use_web_extractor=payload.use_web_extractor,
        provided_evidence_count=len(payload.evidence or []),
    )

    sources: list[dict[str, Any]] = []
    primary_question = questions[0] if questions else CompletionQuestion(
        question_id="node:overview",
        target_kind="fact",
        target_field=None,
        relation_intent=None,
        temporal_role=None,
        query_text=fallback_query,
        search_terms=[],
    )

    if source_enabled(payload, "provided_evidence"):
        for item in payload.evidence or []:
            data = item.dict() if hasattr(item, "dict") else dict(item)
            if data.get("content") or data.get("quote"):
                data.setdefault("source_type", "provided")
                data.setdefault("source", data.get("title") or "provided")
                sources.append(_tag_source_with_question(data, primary_question))
    semantic_log(trace_id, "provided_evidence", accepted=len(sources), enabled=source_enabled(payload, "provided_evidence"))

    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    try:
        domain_kb_limit = max(1, min(int(metadata.get("domain_kb_limit_per_question") or 5), 20))
    except Exception:
        domain_kb_limit = 5
    try:
        per_question_limit = max(1, min(int(metadata.get("web_limit_per_question") or payload.max_web_results or 5), 10))
    except Exception:
        per_question_limit = max(1, min(payload.max_web_results or 5, 10))
    web_policy = str(metadata.get("web_search_policy") or "local_first_backfill").strip().lower()
    force_web = web_policy in {"always", "force", "force_web"}
    for question in questions:
        question_sources: list[dict[str, Any]] = []
        target_terms = _split_query_terms(" ".join([question.target_field or "", question.relation_intent or ""]))
        scope_terms = [term for term in _question_terms(question) if term not in target_terms]

        if not source_enabled(payload, "domain_kb"):
            diagnostics.append({"stage": "domain_kb", "question_id": question.question_id, "skipped": True, "reason": "source_scope_disabled"})
            semantic_log(trace_id, "domain_kb", question_id=question.question_id, skipped=True, reason="source_scope_disabled")
        else:
            try:
                domain_kb_hits = search_domain_kb(payload.scenic_id, question.query_text, limit=domain_kb_limit)
                question_sources.extend(_tag_source_with_question(item, question) for item in domain_kb_hits)
                diagnostics.append({"stage": "domain_kb", "question_id": question.question_id, "hits": len(domain_kb_hits)})
                semantic_log(trace_id, "domain_kb", question_id=question.question_id, query=question.query_text, hits=len(domain_kb_hits), titles=[x.get("title") for x in domain_kb_hits[:5]])
            except Exception as exc:
                diagnostics.append({"stage": "domain_kb", "question_id": question.question_id, "error": str(exc)})
                semantic_log(trace_id, "domain_kb", question_id=question.question_id, error=str(exc))

        coverage = _question_local_coverage(payload, question, question_sources)
        if payload.use_web_search and source_enabled(payload, "web_search") and coverage.get("covered") and not force_web:
            diagnostics.append({"stage": "web_search", "question_id": question.question_id, "skipped": True, "reason": "local_coverage_sufficient", "web_policy": web_policy, **coverage})
            semantic_log(trace_id, "web_search_skip", question_id=question.question_id, reason="local_coverage_sufficient", web_policy=web_policy, coverage=coverage)
        elif payload.use_web_search and source_enabled(payload, "web_search"):
            web_results: list[dict[str, Any]] = []
            query_diagnostics: list[dict[str, Any]] = []
            seen_web_results: set[tuple[str, str]] = set()
            query_variants = build_web_query_variants(payload, question)
            for query_index, web_query in enumerate(query_variants, start=1):
                variant_results, meta = web_search(web_query, limit=per_question_limit)
                query_diagnostics.append({
                    "query_index": query_index,
                    "query": web_query,
                    "returned": len(variant_results),
                    **meta,
                })
                for result in variant_results:
                    result_key = (
                        str(result.get("source_url") or "").strip(),
                        str(result.get("title") or result.get("content") or "").strip(),
                    )
                    if result_key in seen_web_results:
                        continue
                    seen_web_results.add(result_key)
                    result["query_variant"] = query_index
                    web_results.append(result)
            diagnostics.append({
                "stage": "web_search",
                "question_id": question.question_id,
                "queries": query_diagnostics,
            })
            accepted = 0
            rejected = 0
            scored_results: list[dict[str, Any]] = []
            for idx, result in enumerate(web_results, start=1):
                score_info = score_source_relevance(
                    result,
                    required_terms=required_terms,
                    scenic_terms=scenic_terms,
                    scope_terms=scope_terms,
                    target_terms=target_terms,
                    rank=idx,
                )
                result["relevance"] = score_info
                keep_reason = "score_threshold" if score_info["score"] >= 1.2 else ("top_rank_fallback" if idx <= 2 and score_info["score"] >= 0.4 else "rejected_low_score")
                scored_results.append({
                    "rank": idx,
                    "title": result.get("title"),
                    "url": result.get("source_url"),
                    "score": score_info["score"],
                    "matched_terms": score_info["matched_terms"],
                    "reasons": score_info["reasons"],
                    "decision": keep_reason,
                })
                if keep_reason != "rejected_low_score":
                    result["score"] = max(float(result.get("score") or 0.0), float(score_info["score"]))
                    question_sources.append(_tag_source_with_question(result, question))
                    accepted += 1
                else:
                    rejected += 1
            semantic_log(trace_id, "web_search", question_id=question.question_id, queries=query_diagnostics, returned=len(web_results), accepted=accepted, rejected_by_score=rejected, web_policy=web_policy, limit=per_question_limit, scored_results=scored_results)

        sources.extend(question_sources)

    if payload.use_web_extractor and source_enabled(payload, "web_extractor"):
        extracted: list[dict[str, Any]] = []
        seen_url_question_keys = set()
        page_cache: dict[str, dict[str, Any] | None] = {}
        for item in list(sources):
            url = str(item.get("source_url") or "").strip()
            question_id = str(item.get("question_id") or "")
            key = (url, question_id)
            if not url or key in seen_url_question_keys:
                continue
            seen_url_question_keys.add(key)
            if url not in page_cache:
                page_cache[url] = extract_web_page(url)
            cached_page = page_cache[url]
            if not cached_page:
                continue
            page = dict(cached_page)
            question = next((q for q in questions if q.question_id == question_id), primary_question)
            target_terms = _split_query_terms(" ".join([question.target_field or "", question.relation_intent or ""]))
            scope_terms = [term for term in _question_terms(question) if term not in target_terms]
            score_info = score_source_relevance(
                page,
                required_terms=required_terms,
                scenic_terms=scenic_terms,
                scope_terms=scope_terms,
                target_terms=target_terms,
            )
            page["relevance"] = score_info
            if score_info["score"] >= 1.0:
                page["score"] = max(float(page.get("score") or 0.0), float(score_info["score"]))
                extracted.append(_tag_source_with_question(page, question))
        if extracted:
            sources = extracted + sources
        diagnostics.append({"stage": "web_extractor", "attempted": len(seen_url_question_keys), "unique_urls": len(page_cache), "extracted": len(extracted)})
        semantic_log(trace_id, "web_extractor", attempted=len(seen_url_question_keys), unique_urls=len(page_cache), extracted=len(extracted), urls=list(page_cache)[:10])

    try:
        evidence_limit_per_question = max(1, min(int(metadata.get("evidence_limit_per_question") or 8), 20))
    except Exception:
        evidence_limit_per_question = 8
    def source_identity(source: dict[str, Any]) -> str:
        source_type = str(source.get("source_type") or "unknown").lower()
        document_id = str(source.get("source_doc_id") or source.get("doc_id") or "").strip()
        if document_id:
            return f"document:{document_id}"
        domain = urlparse(str(source.get("source_url") or "")).netloc.lower()
        if domain:
            return f"domain:{domain}"
        return f"{source_type}:{str(source.get('title') or source.get('source') or '')[:120]}"

    def source_family(source: dict[str, Any]) -> str:
        source_type = str(source.get("source_type") or "").lower()
        if source_type.startswith("domain_kb"):
            return "domain_kb"
        if source_type.startswith("web"):
            return "web"
        if source_type in {"provided", "provided_evidence"}:
            return "provided"
        return "other"

    def diverse_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_question: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_question.setdefault(str(item.get("question_id") or ""), []).append(item)
        ordered: list[dict[str, Any]] = []
        family_order = ["provided", "domain_kb", "web", "other"]
        for question_items in by_question.values():
            families: dict[str, dict[str, dict[str, Any]]] = {
                family: {} for family in family_order
            }
            for item in question_items:
                family = source_family(item)
                identity = source_identity(item)
                current = families.setdefault(family, {}).get(identity)
                current_score = float(current.get("score") or 0.0) if current else -1.0
                item_score = float(item.get("score") or 0.0)
                if current is None or item_score > current_score or (
                    item_score == current_score
                    and len(str(item.get("content") or "")) > len(str(current.get("content") or ""))
                ):
                    families[family][identity] = item
            family_lists = {
                family: sorted(
                    values.values(),
                    key=lambda value: float(value.get("score") or 0.0),
                    reverse=True,
                )
                for family, values in families.items()
            }
            while any(family_lists.get(family) for family in family_order):
                for family in family_order:
                    values = family_lists.get(family) or []
                    if values:
                        item = values.pop(0)
                        item["source_identity"] = source_identity(item)
                        ordered.append(item)
        return ordered

    sources = diverse_sources(sources)
    chunks: list[EvidenceChunk] = []
    seen_keys = set()
    question_chunk_counts: dict[str, int] = {}
    for source in sources:
        source_question_id = str(source.get("question_id") or "")
        if question_chunk_counts.get(source_question_id, 0) >= evidence_limit_per_question:
            continue
        key = (
            source_question_id,
            str(source.get("source_url") or ""),
            str(source.get("content") or source.get("quote") or ""),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        content = str(source.get("content") or source.get("quote") or "").strip()
        if not content:
            continue
        score = float(source.get("score") or 0.0)
        weight_info = source_weight(source)
        metadata = source.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        metadata.update(weight_info)
        metadata["source_identity"] = str(source.get("source_identity") or source_identity(source))
        chunks.append(
            EvidenceChunk(
                source_id=f"S{len(chunks) + 1}",
                title=str(source.get("title") or source.get("source") or f"来源 {len(chunks)+1}").strip(),
                content=content[:6000],
                quote=str(source.get("quote") or content[:500]).strip(),
                source=str(source.get("source") or "").strip(),
                source_type=str(source.get("source_type") or "unknown").strip(),
                source_url=_clean_source_url(source.get("source_url")),
                source_doc_id=str(source.get("source_doc_id") or source.get("doc_id") or "").strip() or None,
                chunk_id=_int_or_none(source.get("chunk_id")),
                page_no=_int_or_none(source.get("page_no")),
                score=score,
                question_id=str(source.get("question_id") or "").strip() or None,
                target_kind=str(source.get("target_kind") or "").strip() or None,
                target_field=str(source.get("target_field") or "").strip() or None,
                relation_intent=str(source.get("relation_intent") or "").strip() or None,
                temporal_role=str(source.get("temporal_role") or "").strip() or None,
                query_text=str(source.get("query_text") or "").strip() or None,
                retrieval_score=score,
                rerank_score=score,
                source_weight=float(weight_info.get("source_weight") or 0.0),
                final_evidence_score=round(min(1.0, score * 0.7 + float(weight_info.get("source_weight") or 0.0) * 0.3), 3),
                metadata=metadata,
            )
        )
        question_chunk_counts[source_question_id] = question_chunk_counts.get(source_question_id, 0) + 1
    semantic_log(
        trace_id,
        "evidence_ready",
        source_count=len(sources),
        chunk_count=len(chunks),
        chunks=[{"source_id": c.source_id, "question_id": c.question_id, "target_field": c.target_field, "relation_intent": c.relation_intent, "title": c.title, "source_type": c.source_type, "source_url": c.source_url, "content_len": len(c.content or "")} for c in chunks],
    )
    return chunks, diagnostics


def get_domain_schema(payload: SemanticCompleteRequest) -> dict[str, Any]:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    schema = metadata.get("domain_schema") or {}
    return schema if isinstance(schema, dict) else {}


def completion_mode(payload: SemanticCompleteRequest) -> str:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    return str(metadata.get("completion_mode") or metadata.get("job_mode") or "quick").strip().lower()


def allow_open_discovery(payload: SemanticCompleteRequest) -> bool:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    if metadata.get("open_discovery") is not None:
        return bool(metadata.get("open_discovery"))
    return completion_mode(payload) in {"deep", "web", "full", "batch"}


def _unique_text(values: list[Any]) -> list[str]:
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def get_schema_node_types(payload: SemanticCompleteRequest) -> list[str]:
    schema = get_domain_schema(payload)
    allowed = schema.get("allowed_node_types") or []
    values: list[str] = []
    for item in allowed:
        text = str(item or "").strip()
        if text and text not in values:
            values.append(text)
    if values:
        return values
    return ["building", "region", "poi", "person", "object"]


def get_schema_property_fields(payload: SemanticCompleteRequest) -> list[str]:
    selected = _unique_text(list(payload.target_fields or []))
    schema = get_domain_schema(payload)
    schema_map = schema.get("schema_map") if isinstance(schema.get("schema_map"), dict) else {}
    node_type = str(payload.node.node_type or "").strip()
    fields = list(schema_map.get(node_type) or [])
    if not fields:
        fields = list(schema_map.get("base") or [])
    schema_fields = _unique_text(fields)
    if selected and not allow_open_discovery(payload):
        return selected
    return _unique_text(selected + schema_fields)


def get_schema_relation_items(payload: SemanticCompleteRequest) -> list[dict[str, Any]]:
    schema = get_domain_schema(payload)
    relation_map = schema.get("relation_intents") if isinstance(schema.get("relation_intents"), dict) else {}
    node_type = str(payload.node.node_type or "").strip()
    items = relation_map.get(node_type) or []
    if not isinstance(items, list):
        items = []
    selected = {str(x or "").strip() for x in (payload.relation_intents or []) if str(x or "").strip()}
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("code") or "").strip()
        code = str(item.get("code") or label).strip()
        if selected and not allow_open_discovery(payload) and label not in selected and code not in selected:
            continue
        result.append(item)
    if result:
        return result
    return [{"label": text, "code": text, "allowed_target_types": []} for text in (payload.relation_intents or [])]


def get_schema_relation_labels(payload: SemanticCompleteRequest) -> list[str]:
    labels: list[str] = []
    for item in get_schema_relation_items(payload):
        label = str(item.get("label") or item.get("code") or "").strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def get_schema_relation_target_types(payload: SemanticCompleteRequest) -> list[str]:
    values: list[str] = []
    for item in get_schema_relation_items(payload):
        for target in item.get("allowed_target_types") or []:
            text = str(target or "").strip()
            if text and text not in values:
                values.append(text)
    return values or get_schema_node_types(payload)


def get_schema_relation_target_types_for_predicate(payload: SemanticCompleteRequest, predicate: str) -> list[str]:
    pred = str(predicate or "").strip()
    values: list[str] = []
    for item in get_schema_relation_items(payload):
        label = str(item.get("label") or item.get("code") or "").strip()
        code = str(item.get("code") or label).strip()
        if pred and pred not in {label, code}:
            continue
        for target in item.get("allowed_target_types") or []:
            text = str(target or "").strip()
            if text and text not in values:
                values.append(text)
    return values


def normalize_relation_object_type(payload: SemanticCompleteRequest, predicate: str, object_type: str) -> str:
    proposed = str(object_type or "").strip()
    if allow_open_discovery(payload) and proposed:
        return proposed
    allowed_for_relation = get_schema_relation_target_types_for_predicate(payload, predicate)
    if allowed_for_relation:
        # Preserve evidence-supported new types for human review.
        return proposed or (allowed_for_relation[0] if len(allowed_for_relation) == 1 else "")
    if proposed:
        return proposed
    return ""


def build_domain_schema_prompt(payload: SemanticCompleteRequest) -> str:
    schema = get_domain_schema(payload)
    if not schema:
        if allow_open_discovery(payload):
            return "No domain schema was provided. In deep/full/batch mode, use explicit targets as guidance but preserve additional evidence-grounded properties, relation predicates, and related entities discovered in the sources."
        return "No domain schema was provided; in quick mode use only explicit target fields and relation intents."
    type_labels = schema.get("type_labels") if isinstance(schema.get("type_labels"), dict) else {}
    node_types = get_schema_node_types(payload)
    type_lines = [f"{code}={type_labels.get(code) or code}" for code in node_types]
    fields = get_schema_property_fields(payload)
    relation_lines = []
    for item in get_schema_relation_items(payload):
        label = str(item.get("label") or item.get("code") or "").strip()
        category = str(item.get("relation_category") or item.get("category") or "").strip()
        targets = item.get("allowed_target_types") or []
        relation_lines.append(f"{label}({category}) -> {targets}")
    return chr(10).join([
        f"Domain schema profile={schema.get('profile')}; root_node_type={schema.get('root_node_type')}",
        "Allowed node types: " + json.dumps(type_lines, ensure_ascii=False),
        "Allowed property predicates for current node: " + json.dumps(fields, ensure_ascii=False),
        "Allowed relation predicates for current node: " + json.dumps(relation_lines, ensure_ascii=False),
        "Constraint: quick mode only fills selected template gaps and keeps template predicates and target types. Deep/full/batch mode may discover additional evidence-grounded properties, relation predicates, and related entities. Schema predicates and types are preferred labels and quality hints in open discovery, not a hard allowlist. Preserve a model-inferred predicate or entity type when it is supported by a source quote. Put only facts that are not node properties or relations into background claims.",
    ])

def build_claim_tools(payload: SemanticCompleteRequest) -> list[dict[str, Any]]:
    open_discovery = allow_open_discovery(payload)
    prop_schema: dict[str, Any] = {"type": "string", "description": "property predicate/field name"}
    property_fields = get_schema_property_fields(payload)
    if property_fields and not open_discovery:
        prop_schema["enum"] = property_fields
    rel_schema: dict[str, Any] = {"type": "string", "description": "relation predicate"}
    relation_labels = get_schema_relation_labels(payload)
    if relation_labels and not open_discovery:
        rel_schema["enum"] = relation_labels
    object_type_schema: dict[str, Any] = {
        "type": "string",
        "description": "related entity type; in open discovery preserve the type inferred from evidence even when it is not in the domain template",
    }
    relation_target_types = get_schema_relation_target_types(payload)
    if relation_target_types and not open_discovery:
        object_type_schema["enum"] = relation_target_types
    return [
        {
            "type": "function",
            "function": {
                "name": "extract_property_claim",
                "description": "从 evidence_chunks 抽取候选属性事实。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "predicate": prop_schema,
                        "object_value": {"type": "string"},
                        "source_id": {"type": "string"},
                        "question_id": {"type": "string"},
                        "temporal_role": {"type": "string"},
                        "quote": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["predicate", "object_value", "source_id", "quote", "confidence"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_relation_claim",
                "description": "从 evidence_chunks 抽取候选关系事实。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "predicate": rel_schema,
                        "object_name": {"type": "string"},
                        "object_type": object_type_schema,
                        "source_id": {"type": "string"},
                        "question_id": {"type": "string"},
                        "temporal_role": {"type": "string"},
                        "quote": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["predicate", "object_name", "source_id", "quote", "confidence"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_background_claim",
                "description": "抽取背景事实、历史沿革、民俗说法等候选事实。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "predicate": {"type": "string"},
                        "object_value": {"type": "string"},
                        "source_id": {"type": "string"},
                        "question_id": {"type": "string"},
                        "temporal_role": {"type": "string"},
                        "quote": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["predicate", "object_value", "source_id", "quote", "confidence"],
                },
            },
        },
    ]


def call_claim_extractor(payload: SemanticCompleteRequest, chunks: list[EvidenceChunk], trace_id: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not chunks:
        semantic_log(trace_id, "claim_extraction_skip", reason="no_evidence_chunks")
        return [], {"reason": "no_evidence_chunks"}
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    try:
        content_char_limit = max(400, min(int(metadata.get("extractor_content_char_limit") or 1800), 4000))
    except Exception:
        content_char_limit = 1800
    compact_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        item = chunk.dict()
        item["content"] = str(item.get("content") or "")[:content_char_limit]
        item["quote"] = str(item.get("quote") or "")[:800]
        compact_chunks.append(item)
    evidence_json = json.dumps(compact_chunks, ensure_ascii=False)
    NL = chr(10)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence-first knowledge graph claim extractor. "
                "Extract CandidateClaim only from evidence_chunks. Never invent URLs. "
                "Every claim must include source_id and quote, and quote must come from the corresponding source text. "
                "A search-result title alone is not evidence. Do not use the current subject name as the answer to another property. "
                "All user-facing predicate names, relation names, and inferred entity type labels must use Simplified Chinese unless an explicit domain schema code is provided. Preserve source quotes and values exactly as written. "
                "Follow the domain schema constraints in the user message. In quick mode obey the template; in deep/full/batch mode preserve additional evidence-grounded properties, relation predicates, and entity types instead of discarding them. "
                "Treat each independent source separately: emit a separate claim for every source-supported value, preserve disagreements, and never merge conflicting values into one synthesized answer. "
                "A boolean value 是/否 is allowed only for a short attribute or an explicit 是否/有无 predicate. Never turn a complete action, goal, policy, or sentence into a predicate with 是/否; if the source says an action or plan, keep the action as a relation/event or omit it when no atomic fact can be formed." "Do not emit a clause as both predicate and object; if the object merely repeats the predicate or a phrase contained in it, omit the claim unless it is a named value such as a title or alias."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Current node: {payload.node.name} ID={payload.node.source_node_id} type={payload.node.node_type}{NL}"
                f"Graph/domain context: {'disabled for self scope' if payload.subgraph_depth == 0 else (payload.node.scenic_name or payload.scenic_id)}; parent={payload.node.parent_name or 'none'}{NL}"
                f"User request: {payload.message or ''}{NL}"
                f"Target properties: {json.dumps(get_schema_property_fields(payload), ensure_ascii=False)}{NL}"
                f"Target relations: {json.dumps(get_schema_relation_labels(payload), ensure_ascii=False)}{NL}"
                f"Temporal roles: {json.dumps([q.metadata.get('temporal_role') for q in plan_completion_questions(payload) if q.temporal_role], ensure_ascii=False)}{NL}"
                f"Domain schema constraints:{NL}{build_domain_schema_prompt(payload)}{NL}"
                f"Existing properties: {json.dumps([x.dict() for x in payload.existing_properties], ensure_ascii=False)}{NL}"
                f"Existing relations: {json.dumps([x.dict() for x in payload.existing_relations], ensure_ascii=False)}{NL}"
                f"Adopted candidate context (strong context only; not a published fact and not evidence by itself): {json.dumps(metadata.get('adopted_candidate_context') or [], ensure_ascii=False)}{NL}"
                f"User selected graph scope: {build_graph_context_prompt(payload)}{NL}"
                f"Scope rule: if graph scope is self, do not infer claims from parent/domain/neighbors; if scope is subgraph/all, use graph context only to disambiguate search and relation targets, not as evidence without a source URL.{NL}"
                f"evidence_chunks: {evidence_json}"
            ),
        },
    ]
    graph_context = get_graph_context(payload)
    semantic_log(trace_id, "claim_extraction_start", chunk_count=len(chunks), target_fields=get_schema_property_fields(payload), relation_intents=get_schema_relation_labels(payload), domain_schema_present=bool(get_domain_schema(payload)), graph_scope=graph_context.get("scope"), graph_node_count=len(graph_context.get("nodes") or []), graph_relation_count=len(graph_context.get("relations") or []))
    try:
        response = Generation.call(
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            messages=messages,
            result_format="message",
            tools=build_claim_tools(payload),
        )
        if getattr(response, "status_code", None) != HTTPStatus.OK:
            error = f"{getattr(response, 'code', '')}: {getattr(response, 'message', '')}"
            semantic_log(trace_id, "claim_extraction_error", error=error)
            return [], {"error": error}
        choice_msg = response.output.choices[0].message if response.output and response.output.choices else None
        tool_calls = getattr(choice_msg, "tool_calls", None) if choice_msg else None
        if not tool_calls:
            text = getattr(choice_msg, "content", "") if choice_msg else ""
            semantic_log(trace_id, "claim_extraction_no_tool_calls", text=str(text)[:1000])
            return [], {"warning": "no_tool_calls", "text": text}
        normalized = normalize_tool_calls(tool_calls)
        semantic_log(trace_id, "claim_extraction_done", tool_calls=len(tool_calls), normalized_calls=[{"name": c.get("name"), "arguments_len": len(str(c.get("arguments") or ""))} for c in normalized])
        return normalized, {
            "tool_calls": len(tool_calls),
            "input_chunks": len(compact_chunks),
            "input_chars": len(evidence_json),
        }
    except Exception as exc:
        logger.warning("claim extraction failed: %s", exc, exc_info=True)
        semantic_log(trace_id, "claim_extraction_exception", error=str(exc))
        return [], {"error": str(exc)}


def normalize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            normalized.append({"name": fn.get("name", ""), "arguments": fn.get("arguments", "{}")})
            continue
        fn = getattr(tc, "function", None)
        normalized.append({"name": getattr(fn, "name", ""), "arguments": getattr(fn, "arguments", "{}")})
    return normalized


def claims_from_tool_calls(
    payload: SemanticCompleteRequest,
    chunks: list[EvidenceChunk],
    calls: list[dict[str, Any]],
    source_evidence_map: dict[str, int] | None = None,
) -> list[CandidateClaim]:
    chunk_map = {chunk.source_id: chunk for chunk in chunks}
    source_evidence_map = source_evidence_map or {}
    claims: list[CandidateClaim] = []
    for call in calls:
        try:
            args = json.loads(call.get("arguments") or "{}")
        except Exception:
            continue
        name = call.get("name")
        if name == "extract_property_claim":
            claim_type = "property"
        elif name == "extract_relation_claim":
            claim_type = "relation"
        elif name == "extract_background_claim":
            claim_type = "fact"
        else:
            continue
        source_id = str(args.get("source_id") or "").strip()
        question_id = str(args.get("question_id") or "").strip()
        predicate_value = str(args.get("predicate") or "").strip()
        object_type_value = str(args.get("object_type") or "").strip()
        if claim_type == "relation":
            object_type_value = normalize_relation_object_type(payload, predicate_value, object_type_value)
        chunk = chunk_map.get(source_id)
        claim_value = str(args.get("object_name") if claim_type == "relation" else args.get("object_value") or "").strip()
        quote_value = str(args.get("quote") or "").strip()
        if not predicate_value or not claim_value or not source_id or not chunk or not quote_value:
            continue
        temporal_role = str(args.get("temporal_role") or (chunk.temporal_role if chunk else "") or "").strip()
        try:
            confidence = float(args.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        evidence_ids: list[int] = []
        if source_id in source_evidence_map:
            evidence_ids = [int(source_evidence_map[source_id])]
        question_value = question_id or (chunk.question_id if chunk else "")
        selected_props = {str(x or "").strip() for x in (payload.target_fields or []) if str(x or "").strip()}
        selected_rels = {str(x or "").strip() for x in (payload.relation_intents or []) if str(x or "").strip()}
        if claim_type == "property":
            discovery_scope = "open" if allow_open_discovery(payload) and predicate_value not in selected_props else "template"
        elif claim_type == "relation":
            discovery_scope = "open" if allow_open_discovery(payload) and predicate_value not in selected_rels else "template"
        else:
            discovery_scope = "background"
        claims.append(
            CandidateClaim(
                claim_id=f"c_{len(claims)+1:03d}",
                claim_type=claim_type,
                subject_node_id=str(payload.node.source_node_id),
                subject_name=payload.node.name,
                predicate=predicate_value,
                object_value=str(args.get("object_value") or "").strip(),
                object_name=str(args.get("object_name") or "").strip(),
                object_type=object_type_value,
                source_id=source_id,
                source_url=chunk.source_url if chunk else "",
                quote=quote_value,
                question_id=question_value or None,
                temporal_role=temporal_role or None,
                evidence_ids=evidence_ids,
                support_status="needs_more_evidence",
                recommend_score=round(max(0.0, min(confidence, 1.0)) * 0.35, 3),
                confidence=max(0.0, min(confidence, 1.0)),
                metadata={"discovery_scope": discovery_scope, "completion_mode": completion_mode(payload)},
            )
        )
    return claims


def is_low_information_evidence(chunk: EvidenceChunk) -> bool:
    def normalize_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().lower()

    content = normalize_text(chunk.content)
    title = normalize_text(chunk.title)
    quote = normalize_text(chunk.quote)
    if not content:
        return True
    return bool(title and content == title and (not quote or quote == title))


def select_extractor_chunks(
    payload: SemanticCompleteRequest,
    questions: list[CompletionQuestion],
    chunks: list[EvidenceChunk],
) -> list[EvidenceChunk]:
    metadata = payload.metadata if isinstance(payload.metadata, dict) else {}
    try:
        per_question = max(1, min(int(metadata.get("extractor_chunks_per_question") or 3), 8))
    except Exception:
        per_question = 3
    by_question: dict[str, list[EvidenceChunk]] = {}
    for chunk in chunks:
        if is_low_information_evidence(chunk):
            continue
        by_question.setdefault(str(chunk.question_id or ""), []).append(chunk)
    for question_id, items in by_question.items():
        family_order = ["provided", "domain_kb", "web", "other"]
        families: dict[str, dict[str, EvidenceChunk]] = {family: {} for family in family_order}
        for item in items:
            source_type = str(item.source_type or "").lower()
            family = (
                "domain_kb" if source_type.startswith("domain_kb")
                else "web" if source_type.startswith("web")
                else "provided" if source_type in {"provided", "provided_evidence"}
                else "other"
            )
            item_metadata = getattr(item, "metadata", {})
            item_metadata = item_metadata if isinstance(item_metadata, dict) else {}
            identity = str(item_metadata.get("source_identity") or item.source_doc_id or item.source_url or item.source_id)
            current = families[family].get(identity)
            if current is None or float(item.final_evidence_score or item.score or 0.0) > float(current.final_evidence_score or current.score or 0.0):
                families[family][identity] = item
        family_lists = {
            family: sorted(
                values.values(),
                key=lambda item: float(item.final_evidence_score or item.score or 0.0),
                reverse=True,
            )
            for family, values in families.items()
        }
        diverse_items: list[EvidenceChunk] = []
        while any(family_lists.get(family) for family in family_order):
            for family in family_order:
                values = family_lists.get(family) or []
                if values:
                    diverse_items.append(values.pop(0))
        by_question[question_id] = diverse_items

    selected: list[EvidenceChunk] = []
    for rank in range(per_question):
        for question in questions:
            items = by_question.get(question.question_id) or []
            if rank < len(items):
                selected.append(items[rank])
    if not selected:
        selected.extend((by_question.get("") or [])[:per_question])
    return selected


def extract_claims_in_question_batches(
    payload: SemanticCompleteRequest,
    questions: list[CompletionQuestion],
    chunks: list[EvidenceChunk],
    *,
    trace_id: str = "",
    source_evidence_map: dict[str, int] | None = None,
) -> tuple[list[CandidateClaim], dict[str, Any]]:
    all_claims: list[CandidateClaim] = []
    batch_diagnostics: list[dict[str, Any]] = []
    unscoped_chunks = [chunk for chunk in chunks if not chunk.question_id]
    question_batches = batch_completion_questions(payload, questions)

    for batch_index, question_batch in enumerate(question_batches, start=1):
        question_ids = {question.question_id for question in question_batch}
        batch_chunks = [chunk for chunk in chunks if chunk.question_id in question_ids]
        if batch_index == 1:
            batch_chunks.extend(unscoped_chunks)
        extractor_chunks = select_extractor_chunks(payload, question_batch, batch_chunks)
        calls, call_meta = call_claim_extractor(payload, extractor_chunks, trace_id=trace_id)
        batch_claims = claims_from_tool_calls(
            payload,
            extractor_chunks,
            calls,
            source_evidence_map=source_evidence_map,
        )
        all_claims.extend(batch_claims)
        batch_diagnostics.append({
            "batch_index": batch_index,
            "question_ids": [question.question_id for question in question_batch],
            "question_count": len(question_batch),
            "available_chunks": len(batch_chunks),
            "skipped_low_information_chunks": sum(1 for chunk in batch_chunks if is_low_information_evidence(chunk)),
            "extractor_chunks": len(extractor_chunks),
            "claim_count": len(batch_claims),
            **call_meta,
        })

    unique_claims: dict[tuple[str, ...], CandidateClaim] = {}
    for claim in all_claims:
        key = (
            str(claim.claim_type or ""),
            str(claim.predicate or "").strip().lower(),
            str(claim.object_name or claim.object_value or "").strip().lower(),
            str(claim.source_id or ""),
            str(claim.question_id or ""),
            str(claim.temporal_role or ""),
        )
        previous = unique_claims.get(key)
        if previous is None or claim.confidence > previous.confidence:
            unique_claims[key] = claim
    claims = list(unique_claims.values())
    for index, claim in enumerate(claims, start=1):
        claim.claim_id = f"c_{index:03d}"
    return claims, {
        "batch_size": get_question_batch_size(payload),
        "batch_count": len(question_batches),
        "planned_question_count": len(questions),
        "raw_claim_count": len(all_claims),
        "deduplicated_claim_count": len(claims),
        "batches": batch_diagnostics,
    }


def _claim_surface_supported(claim: CandidateClaim, chunk: EvidenceChunk) -> tuple[bool, str]:
    """Enforce lexical grounding before any candidate can be reviewable.

    Normalization/canonicalization may create internal values, but those are not
    evidence. The raw value, quote, and schema-looking predicate must be present
    in the supplied content/quote; no translation is accepted here.
    """
    evidence_text = _normalize_match_term(" ".join([chunk.content or "", chunk.quote or ""]))
    raw_value = str(claim.raw_value or claim.object_value or claim.object_name or "").strip()
    quote = str(claim.quote or "").strip()
    predicate = str((claim.metadata or {}).get("raw_predicate") or claim.predicate or "").strip()
    if not quote or _normalize_match_term(quote) not in evidence_text:
        return False, "quote_not_in_evidence"
    if raw_value and _normalize_match_term(raw_value) not in evidence_text:
        return False, "raw_value_not_in_evidence"
    if (
        predicate
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_ .-]*", predicate)
        and _normalize_match_term(predicate) not in evidence_text
    ):
        return False, "internal_schema_predicate_not_in_evidence"
    return True, ""


def _growth_claim_value_in_evidence(claim: CandidateClaim, chunk: EvidenceChunk) -> bool:
    """Growth candidates must be anchored to text in the supplied chunk."""
    values = (
        [claim.object_name, claim.raw_value, claim.normalized_value]
        if claim.claim_type == "relation"
        else [claim.normalized_value, claim.raw_value, claim.object_value, claim.object_name]
    )
    evidence_text = _normalize_match_term(
        " ".join([chunk.content or "", chunk.quote or ""])
    )
    return any(
        normalized_value and normalized_value in evidence_text
        for normalized_value in (_normalize_match_term(value) for value in values)
    )


def _growth_quote_in_evidence(claim: CandidateClaim, chunk: EvidenceChunk) -> bool:
    quote = _normalize_match_term(claim.quote or "")
    if len(quote) < 8:
        return True
    evidence_text = _normalize_match_term(
        " ".join([chunk.content or "", chunk.quote or ""])
    )
    return quote in evidence_text


def verify_claim_evidence(claim: CandidateClaim, chunks: list[EvidenceChunk]) -> CandidateClaim:
    chunk_map = {chunk.source_id: chunk for chunk in chunks}
    chunk = chunk_map.get(str(claim.source_id or ""))
    if not chunk:
        claim.evidence_status = "unsupported"
        claim.support_status = "unsupported"
        claim.status = "low_evidence"
        claim.recommend_score = round(max(0.0, min(claim.confidence, 1.0)) * 0.25, 3)
        return claim

    weight_info = chunk_source_weight(chunk)
    meta = getattr(chunk, "metadata", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    merged_meta = dict(meta)
    merged_meta.update(weight_info)
    merged_meta.update({
        "source_type": chunk.source_type,
        "source_title": chunk.title or chunk.source,
        "doc_title": chunk.title,
        "source_doc_id": chunk.source_doc_id,
        "chunk_id": chunk.chunk_id,
        "page_no": chunk.page_no,
    })
    claim.metadata = dict(claim.metadata or {})
    claim.metadata.update(merged_meta)

    haystack = " ".join([chunk.title or "", chunk.content or "", chunk.quote or "", chunk.source or ""]).lower()
    quote = (claim.quote or "").strip().lower()
    subject = (claim.subject_name or "").strip().lower()
    predicate = canonical_predicate(claim.predicate or "", temporal_role=claim.temporal_role).strip().lower()
    obj = (claim.normalized_value or claim.object_value or claim.object_name or "").strip().lower()

    subject_echo = (
        claim.claim_type == "property"
        and bool(subject)
        and obj == subject
        and predicate not in {"名称", "姓名", "name"}
    )
    if is_low_information_evidence(chunk) or subject_echo:
        claim.metadata["evidence_rejection_reason"] = "title_only_evidence" if is_low_information_evidence(chunk) else "subject_echo"
        claim.evidence_score = 0.0
        claim.evidence_status = "unsupported"
        claim.support_status = "unsupported"
        claim.status = "low_evidence"
        claim.recommend_score = round(max(0.0, min(claim.confidence, 1.0)) * 0.1, 3)
        return claim

    # Evidence-first is a hard invariant for every completion mode, not only G2.
    # A translated value or leaked English schema key must be rejected before it
    # can become a reviewable candidate.
    surface_supported, rejection_reason = _claim_surface_supported(claim, chunk)
    if not surface_supported:
        claim.metadata["evidence_rejection_reason"] = rejection_reason
        claim.evidence_score = 0.0
        claim.evidence_status = "unsupported"
        claim.support_status = "unsupported"
        claim.status = "low_evidence"
        claim.recommend_score = 0.0
        return claim

    # G2 remains evidence-only and keeps its stricter value/quote rule for
    # normalized claims as a second defense.
    completion_mode = str((getattr(claim, "metadata", {}) or {}).get("completion_mode") or "")
    if completion_mode == "growth_g2":
        if not _growth_claim_value_in_evidence(claim, chunk) or not _growth_quote_in_evidence(claim, chunk):
            claim.metadata["evidence_rejection_reason"] = "growth_value_or_quote_not_in_evidence"
            claim.evidence_score = 0.0
            claim.evidence_status = "unsupported"
            claim.support_status = "unsupported"
            claim.status = "low_evidence"
            claim.recommend_score = 0.0
            return claim

    quote_grounding = 0.0
    if quote and len(quote) >= 8 and quote in haystack:
        quote_grounding = 1.0
    elif quote and len(quote) >= 4:
        quote_grounding = 0.45

    relevance_hits = 0
    relevance_total = 0
    for value in (subject, predicate, obj):
        if value:
            relevance_total += 1
            if value in haystack:
                relevance_hits += 1
    answer_relevance = relevance_hits / relevance_total if relevance_total else 0.35

    locator_integrity = 0.0
    if chunk.source_url:
        locator_integrity = 1.0
    elif chunk.source_doc_id and chunk.chunk_id:
        locator_integrity = 1.0
    elif chunk.source_doc_id or chunk.chunk_id:
        locator_integrity = 0.65

    retrieval_raw = max(float(chunk.retrieval_score or 0.0), float(chunk.rerank_score or 0.0), float(chunk.score or 0.0))
    retrieval_quality = max(0.0, min(1.0, retrieval_raw / 5.0))

    claim.evidence_score = round(min(1.0, 0.35 * quote_grounding + 0.25 * answer_relevance + 0.20 * locator_integrity + 0.20 * retrieval_quality), 3)
    if claim.evidence_score >= 0.75:
        claim.evidence_status = "supported"
        claim.support_status = "supported"
        claim.status = "adoptable"
    elif claim.evidence_score >= 0.45:
        claim.evidence_status = "weak"
        claim.support_status = "weakly_supported"
        claim.status = "needs_review"
    else:
        claim.evidence_status = "unsupported"
        claim.support_status = "unsupported"
        claim.status = "low_evidence"

    source_authority = float(weight_info.get("source_authority_score") or 0.0)
    model_confidence = max(0.0, min(claim.confidence, 1.0))
    graph_consistency = 0.65
    components = {
        "source_authority": source_authority,
        "retrieval_relevance": retrieval_quality,
        "evidence_support": claim.evidence_score,
        "multi_source_support": 0.0,
        "graph_consistency": graph_consistency,
        "model_confidence": model_confidence,
    }
    claim.score_components = components
    claim.metadata.update({"retrieval_relevance": retrieval_quality, "score_components": components})
    claim.recommend_score = round(min(1.0,
        0.25 * source_authority
        + 0.20 * retrieval_quality
        + 0.20 * claim.evidence_score
        + 0.10 * graph_consistency
        + 0.10 * model_confidence
    ), 3)
    return claim


def _canonical_claim_key(claim: CandidateClaim) -> str:
    return canonical_predicate(claim.predicate or "", temporal_role=claim.temporal_role)


def detect_conflicts(payload: SemanticCompleteRequest, claims: list[CandidateClaim]) -> list[ClaimConflict]:
    return classify_conflicts(payload, claims)


def _legacy_claim_source_fields(claim: CandidateClaim) -> dict[str, Any]:
    meta = claim.metadata or {}
    return {
        "source_url": claim.source_url or meta.get("source_url") or "",
        "source_type": meta.get("source_type") or meta.get("retrieval_source") or "",
        "source_title": meta.get("source_title") or meta.get("doc_title") or "",
        "source_doc_id": meta.get("source_doc_id") or meta.get("doc_id"),
        "chunk_id": meta.get("chunk_id"),
        "page_no": meta.get("page_no"),
        "evidence_ids": claim.evidence_ids or [],
        "candidate_group_key": claim.candidate_group_key or meta.get("candidate_group_key"),
        "value_group_key": claim.value_group_key or meta.get("value_group_key"),
        "conflict_class": claim.conflict_class or meta.get("conflict_class"),
        "gap_status": claim.gap_status or meta.get("gap_status"),
        "recommend_score": claim.recommend_score,
        "support_status": claim.support_status,
    }


def to_legacy_payload(payload: SemanticCompleteRequest, claims: list[CandidateClaim], conflicts: list[ClaimConflict]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    conflict_ids = {c.claim_id for c in conflicts}
    props: list[dict[str, Any]] = []
    rels: list[dict[str, Any]] = []
    discovery_props: list[dict[str, Any]] = []
    discovery_rels: list[dict[str, Any]] = []
    discovery_entities: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    unverified_facts: list[dict[str, Any]] = []
    entity_seen: set[tuple[str, str, str]] = set()
    for claim in claims:
        if claim.status in {"duplicate", "low_evidence"} and claim.claim_type != "fact":
            continue
        meta = claim.metadata or {}
        discovery_scope = str(meta.get("discovery_scope") or "template")
        if claim.claim_type == "property" and claim.claim_id not in conflict_ids:
            item = {"name": meta.get("canonical_predicate") or claim.predicate, "value": claim.display_value or claim.object_value, "confidence": claim.confidence, "quote": claim.quote, "claim_id": claim.claim_id, "candidate_id": claim.candidate_id or meta.get("candidate_id"), "candidate_type": claim.candidate_type or meta.get("candidate_type"), "conflict_group": claim.conflict_group or meta.get("conflict_group"), "evidence_status": claim.evidence_status, "temporal_role": claim.temporal_role or meta.get("temporal_role")}
            item.update(_legacy_claim_source_fields(claim))
            if discovery_scope == "open":
                discovery_props.append(item)
            else:
                props.append(item)
        elif claim.claim_type == "relation" and claim.claim_id not in conflict_ids:
            relation_type = meta.get("canonical_predicate") or claim.predicate
            target_name = claim.display_value or claim.object_name
            target_type = normalize_relation_object_type(payload, relation_type, claim.object_type)
            item = {"relation_type": relation_type, "target_name": target_name, "target_type": target_type, "confidence": claim.confidence, "quote": claim.quote, "claim_id": claim.claim_id, "candidate_id": claim.candidate_id or meta.get("candidate_id"), "candidate_type": claim.candidate_type or meta.get("candidate_type"), "conflict_group": claim.conflict_group or meta.get("conflict_group"), "evidence_status": claim.evidence_status, "temporal_role": claim.temporal_role or meta.get("temporal_role")}
            item.update(_legacy_claim_source_fields(claim))
            if discovery_scope == "open":
                discovery_rels.append(item)
                entity_key = (str(target_name or "").strip(), str(target_type or "").strip(), str(relation_type or "").strip())
                if entity_key[0] and entity_key not in entity_seen:
                    entity_seen.add(entity_key)
                    ent = {"name": entity_key[0], "node_type": entity_key[1] or "unresolved", "entity_type": entity_key[1] or "unresolved", "relation_to_current": entity_key[2], "confidence": claim.confidence, "quote": claim.quote, "claim_id": claim.claim_id}
                    ent.update(_legacy_claim_source_fields(claim))
                    discovery_entities.append(ent)
            else:
                rels.append(item)
        elif claim.claim_type == "fact":
            item = {"category": claim.predicate or "补充事实", "content": claim.object_value, "confidence": claim.confidence, "quote": claim.quote, "claim_id": claim.claim_id, "evidence_status": claim.evidence_status}
            item.update(_legacy_claim_source_fields(claim))
            if claim.source_url and claim.evidence_status in {"supported", "weak"}:
                facts.append(item)
            else:
                unverified_facts.append(item)
    legacy_conflicts = [c.dict() for c in conflicts]
    template_fill = {"properties": props, "relations": rels}
    discoveries = {"properties": discovery_props, "entities": discovery_entities, "relations": discovery_rels, "facts": facts, "unverified_facts": unverified_facts, "conflicts": legacy_conflicts}
    candidates = {"properties": props + discovery_props, "relations": rels + discovery_rels, "entities": discovery_entities, "extensions": []}
    return template_fill, discoveries, candidates


async def complete_semantic_service(
    payload: SemanticCompleteRequest,
    *,
    trace_id_override: str | None = None,
    job_id: int | None = None,
) -> SemanticCompleteResponse:
    started_at = time.time()
    trace_id = str(trace_id_override or payload.metadata.get("trace_id") or payload.metadata.get("request_id") or uuid.uuid4().hex[:12])
    semantic_log(
        trace_id,
        "start",
        scenic_id=payload.scenic_id,
        node=payload.node.dict(),
        message=payload.message,
        source_note_len=len(payload.source_note or ""),
        target_fields=payload.target_fields,
        relation_intents=payload.relation_intents,
        existing_properties=len(payload.existing_properties or []),
        existing_relations=len(payload.existing_relations or []),
        graph_scope=get_graph_context(payload).get("scope"),
        graph_nodes=len(get_graph_context(payload).get("nodes") or []),
        graph_relations=len(get_graph_context(payload).get("relations") or []),
        metadata=payload.metadata,
    )
    diagnostics: list[dict[str, Any]] = []
    questions = plan_completion_questions(payload)
    try:
        questions, graph_discovery = augment_questions_with_graph_discovery(payload, questions)
    except Exception as exc:
        graph_discovery = {
            "enabled": False,
            "reason": "graph_discovery_failed_open",
            "question_count": 0,
            "error": str(exc),
        }
        semantic_log(trace_id, "graph_discovery_error", error=str(exc))
    diagnostics.append({"stage": "graph_discovery", **graph_discovery})
    planned = plan_sources(payload)
    semantic_log(
        trace_id,
        "plan",
        planned_sources=planned,
        question_count=len(questions),
        graph_discovery=graph_discovery,
    )
    # GrowthRun has no legacy completion job, but its evidence must still be
    # durable so candidates can point to EvidenceRecord rows and the A端 can
    # show the exact source, quote, and image.  job_id is nullable by schema.
    if job_id is not None or completion_mode(payload) == "growth_g2":
        try:
            question_store = persist_semantic_completion_questions(
                trace_id=trace_id,
                job_id=int(job_id),
                payload=payload,
                questions=questions,
            )
            diagnostics.append({"stage": "question_store", "saved": int(question_store.get("saved") or 0)})
        except Exception as exc:
            diagnostics.append({"stage": "question_store", "error": str(exc)})
            semantic_log(trace_id, "question_store_error", error=str(exc))
    chunks, evidence_diagnostics = collect_evidence(payload, trace_id=trace_id, questions=questions)
    diagnostics.extend(evidence_diagnostics)
    source_evidence_map: dict[str, int] = {}
    if job_id is not None:
        try:
            evidence_store = persist_semantic_evidence_items(
                trace_id=trace_id,
                job_id=int(job_id),
                payload=payload,
                chunks=chunks,
            )
            source_evidence_map = {str(k): int(v) for k, v in (evidence_store.get("source_to_evidence_id") or {}).items()}
            diagnostics.append({"stage": "evidence_store", "saved": int(evidence_store.get("saved") or 0)})
        except Exception as exc:
            diagnostics.append({"stage": "evidence_store", "error": str(exc)})
            semantic_log(trace_id, "evidence_store_error", error=str(exc))
    claims, call_meta = extract_claims_in_question_batches(
        payload,
        questions,
        chunks,
        trace_id=trace_id,
        source_evidence_map=source_evidence_map,
    )
    diagnostics.append({"stage": "claim_extraction", **call_meta})
    try:
        entity_resolution = resolve_candidate_entities(
            payload,
            claims,
            trace_id=trace_id,
            job_id=job_id,
            persist_new_candidates=False,
        )
        diagnostics.append({"stage": "entity_resolution", **entity_resolution})
    except Exception as exc:
        entity_resolution = {"error": str(exc)}
        diagnostics.append({"stage": "entity_resolution", "error": str(exc)})
        semantic_log(trace_id, "entity_resolution_error", error=str(exc))
    claims = normalize_candidate_claims(payload, claims)
    if completion_mode(payload) == "growth_g2":
        before_count = len(claims)
        claims = [
            claim for claim in claims
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_ .-]*", str((claim.metadata or {}).get("raw_predicate") or claim.predicate or "").strip())
        ]
        diagnostics.append({
            "stage": "growth_quality_guard",
            "dropped_non_chinese_predicates": before_count - len(claims),
        })
    semantic_log(trace_id, "claims_parsed", claim_count=len(claims), entity_resolution=entity_resolution, claims=[c.dict() for c in claims])
    claims = [verify_claim_evidence(claim, chunks) for claim in claims]
    strict_rejection_reasons = {
        "quote_not_in_evidence",
        "raw_value_not_in_evidence",
        "internal_schema_predicate_not_in_evidence",
    }
    before_lexical_guard = len(claims)
    claims = [
        claim for claim in claims
        if str((claim.metadata or {}).get("evidence_rejection_reason") or "") not in strict_rejection_reasons
    ]
    diagnostics.append({
        "stage": "evidence_lexical_guard",
        "dropped_unsupported_claims": before_lexical_guard - len(claims),
    })
    reviewable_entity_claims = [
        claim for claim in claims
        if claim.claim_type == "relation" and claim.evidence_status in {"supported", "weak"}
    ]
    if reviewable_entity_claims:
        try:
            entity_persistence = resolve_candidate_entities(
                payload,
                reviewable_entity_claims,
                trace_id=trace_id,
                job_id=job_id,
                persist_new_candidates=True,
            )
            diagnostics.append({"stage": "entity_candidate_store", **entity_persistence})
        except Exception as exc:
            diagnostics.append({"stage": "entity_candidate_store", "error": str(exc)})
            semantic_log(trace_id, "entity_candidate_store_error", error=str(exc))
    assign_candidate_group_keys(payload, claims)
    status_counts: dict[str, int] = {}
    for claim in claims:
        status_counts[claim.status] = status_counts.get(claim.status, 0) + 1
    semantic_log(trace_id, "claims_verified", status_counts=status_counts, claims=[c.dict() for c in claims])
    conflicts = detect_conflicts(payload, claims)
    semantic_log(trace_id, "conflicts_detected", conflict_count=len(conflicts), conflicts=[c.dict() for c in conflicts])
    candidate_groups = annotate_candidate_groups(payload, claims, conflicts)
    risk_counts = apply_recommendation_and_risk(payload, claims, candidate_groups)
    diagnostics.append({"stage": "risk_classification", **risk_counts})
    semantic_log(trace_id, "candidate_groups", group_count=len(candidate_groups), risk_counts=risk_counts, groups=candidate_groups)
    try:
        candidate_store = persist_semantic_candidates(
            payload,
            trace_id=trace_id,
            job_id=job_id,
            claims=claims,
            conflicts=conflicts,
            chunks=chunks,
        )
        for claim in claims:
            meta = claim.metadata or {}
            claim.candidate_id = meta.get("candidate_id")
            claim.candidate_uid = meta.get("candidate_uid")
            claim.candidate_type = meta.get("candidate_type")
            claim.conflict_group = meta.get("conflict_group")
        semantic_log(trace_id, "candidate_store", **candidate_store)
    except Exception as exc:
        candidate_store = {"saved": 0, "conflict_groups": 0, "error": str(exc)}
        semantic_log(trace_id, "candidate_store_error", error=str(exc))
    if job_id is not None:
        try:
            gap_store = update_semantic_gap_status(
                payload=payload,
                trace_id=trace_id,
                job_id=int(job_id),
                questions=questions,
                chunks=chunks,
                claims=claims,
                candidate_groups=candidate_groups,
            )
            diagnostics.append({"stage": "gap_status", **gap_store})
            semantic_log(trace_id, "gap_status", **gap_store)
        except Exception as exc:
            diagnostics.append({"stage": "gap_status", "error": str(exc)})
            semantic_log(trace_id, "gap_status_error", error=str(exc))
    template_fill, discoveries, candidates = to_legacy_payload(payload, claims, conflicts)
    discoveries["graph"] = graph_discovery
    elapsed_ms = round((time.time() - started_at) * 1000, 2)
    diagnostics.insert(0, {"stage": "trace", "trace_id": trace_id})
    semantic_log(
        trace_id,
        "finish",
        elapsed_ms=elapsed_ms,
        evidence_chunks=len(chunks),
        candidate_claims=len(claims),
        conflicts=len(conflicts),
        fillable_properties=len(template_fill.get("properties") or []),
        fillable_relations=len(template_fill.get("relations") or []),
    )
    return SemanticCompleteResponse(
        trace_id=trace_id,
        summary=f"语义补全完成：证据 {len(chunks)} 条，候选 claim {len(claims)} 条，冲突 {len(conflicts)} 条。",
        planned_sources=planned,
        evidence_chunks=chunks,
        candidate_claims=claims,
        conflicts=conflicts,
        template_fill=template_fill,
        discoveries=discoveries,
        candidates=candidates,
        candidate_groups=candidate_groups,
        diagnostics=diagnostics,
        candidate_store=candidate_store,
    )
