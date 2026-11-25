"""QA v2 的 SSE 流式答案出口。

本模块不负责检索。调用方先完成 Pipeline 和 Evidence，再把证据交给本模块。
事件分为可审计进度、答案 token、最终引用和结束事件，不输出模型私有思维链。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable, Iterator, Optional
from urllib import request as urllib_request

from .answering import AnswerDraft, CitationValidator, ValidatedAnswer, _build_answer_prompt, _labels_from_answer, _strip_citation_markers, fast_attribute_answer, fast_attribute_unavailable_answer, fast_casual_answer, fast_fact_answer, fast_graph_answer, fast_nearby_answer, fast_visual_answer
from src.kg_qa_shared.budget import BudgetSection, ContextBudgetManager
from ..models import QuerySpec
from .prompts import build_answer_system_prompt
from .retrieval import ClaimResult, Evidence, bind_claims_to_evidence


class LocalStreamingAnswerClient:
    """调用本地 vLLM OpenAI-compatible 流式接口。"""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        configured_provider = (provider or os.getenv("ANSWER_PROVIDER") or "local").strip().lower().replace("-", "_")
        self.provider = "openai_compatible" if configured_provider in {"api", "openai", "openai_compatible"} else configured_provider
        if self.provider == "openai_compatible":
            self.base_url = (base_url or os.getenv("ANSWER_API_BASE_URL") or os.getenv("ANSWER_BASE_URL") or "").rstrip("/")
            self.model = model or os.getenv("ANSWER_API_MODEL") or os.getenv("ANSWER_MODEL") or ""
        else:
            self.base_url = (base_url or os.getenv("LOCAL_ANSWER_BASE_URL") or os.getenv("LOCAL_DECISION_BASE_URL") or "").rstrip("/")
            self.model = model or os.getenv("LOCAL_ANSWER_MODEL") or os.getenv("LOCAL_DECISION_MODEL") or ""
        self.api_key = api_key
        try:
            configured_output_tokens = int(os.getenv("ANSWER_MAX_OUTPUT_TOKENS", "256"))
        except ValueError:
            configured_output_tokens = 768
        self.max_output_tokens = max(64, min(configured_output_tokens, 4096))
        self.timeout = float(timeout)
        self.last_runtime_meta: dict[str, Any] = {}

    def stream(self, prompt: str) -> Iterator[str]:
        provider = self.provider
        if provider == "openai_compatible":
            base_url = self.base_url
            model = self.model
            api_key = self.api_key or os.getenv("ANSWER_API_KEY") or os.getenv("LLM_API_KEY") or ""
            if not base_url or not model:
                raise RuntimeError("流式答案 API 未配置 ANSWER_API_BASE_URL/ANSWER_API_MODEL")
        elif provider == "local":
            base_url = self.base_url
            model = self.model
            api_key = self.api_key or ""
            if not base_url or not model:
                raise RuntimeError("流式答案模型未配置 LOCAL_ANSWER_BASE_URL/LOCAL_ANSWER_MODEL")
        else:
            raise RuntimeError(f"不支持的 ANSWER_PROVIDER: {provider}")
        body = json.dumps({
            "model": model,
            "temperature": 0.1,
            "stream": True,
            "max_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": build_answer_system_prompt(streaming=True)},
                {"role": "user", "content": prompt},
            ],
        }, ensure_ascii=False).encode("utf-8")
        endpoint = base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib_request.Request(endpoint, data=body, headers=headers, method="POST")
        started = time.perf_counter()
        first_chunk_at: Optional[float] = None
        finish_reason: Optional[str] = None
        usage: dict[str, Any] = {}
        with urllib_request.urlopen(req, timeout=self.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload.get("usage"), dict):
                    usage.update(payload["usage"])
                choices = payload.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    yield str(content)
        total_ms = int((time.perf_counter() - started) * 1000)
        self.last_runtime_meta = {
            "provider": self.provider,
            "model": self.model,
            "total_ms": max(0, total_ms),
        }
        if first_chunk_at is not None:
            self.last_runtime_meta["ttft_ms"] = int((first_chunk_at - started) * 1000)
        if finish_reason:
            self.last_runtime_meta["finish_reason"] = str(finish_reason)
        aliases = {
            "prompt_tokens": "input_tokens",
            "completion_tokens": "output_tokens",
            "input_tokens": "input_tokens",
            "output_tokens": "output_tokens",
            "total_tokens": "total_tokens",
        }
        for source, target in aliases.items():
            if source in usage and target not in self.last_runtime_meta:
                self.last_runtime_meta[target] = usage[source]


class QAStreamService:
    """把已完成的 QA 证据转换为 SSE 事件迭代器。"""

    def __init__(
        self,
        *,
        answer_client: Any,
        validator: Optional[CitationValidator] = None,
        budget_manager: Optional[ContextBudgetManager] = None,
    ) -> None:
        self.answer_client = answer_client
        self.validator = validator or CitationValidator()
        self.budget_manager = budget_manager or ContextBudgetManager.from_env()

    def stream(
        self,
        question: str,
        *,
        spec: QuerySpec,
        evidence: list[Evidence],
        process: Optional[Iterable[dict[str, Any]]] = None,
        emit_started: bool = True,
        started_at: Optional[float] = None,
        timings: Optional[dict[str, int]] = None,
        runtime_meta: Optional[dict[str, Any]] = None,
    ) -> Iterator[dict[str, Any]]:
        started = started_at or time.perf_counter()
        router_runtime_meta = dict(runtime_meta or {})
        if emit_started:
            yield {"event": "started", "data": {"message": "小V开始整理答案"}}
        for item in process or ():
            yield {"event": "process", "data": dict(item)}

        evidence_context, budget_info, budget_warnings = _build_budgeted_evidence_context(
            question,
            spec,
            evidence,
            self.budget_manager,
        )
        prompt = _build_stream_prompt(question, spec, evidence_context)
        yield {
            "event": "answer_started",
            "data": {"evidence_count": len(evidence), "budget": budget_info},
        }
        fast_draft = fast_casual_answer(question) if spec.route == "CASUAL" else None
        if fast_draft is None:
            fast_draft = fast_nearby_answer(spec, evidence)
        if fast_draft is None:
            fast_draft = fast_visual_answer(spec, evidence)
        if fast_draft is None:
            fast_draft = fast_graph_answer(spec, evidence)
        attribute_unavailable = None
        if fast_draft is None:
            fast_draft = fast_fact_answer(spec, evidence)
        if fast_draft is None:
            fast_draft = fast_attribute_answer(spec, evidence)
        if fast_draft is None:
            attribute_unavailable = fast_attribute_unavailable_answer(spec, evidence)
            fast_draft = attribute_unavailable
        if fast_draft is not None:
            # 快速答案完全由图谱证据模板生成，结构和引用已确定，
            # 不再调用 Jev 做重复语义校验，避免首 token 被模型阻塞。
            validated = (
                ValidatedAnswer(answer=fast_draft.answer, citations=[], warnings=[])
                if attribute_unavailable is not None
                else self.validator.validate(fast_draft, evidence, semantic=False)
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            yield {"event": "token", "data": {"text": validated.answer}}
            completion = _build_completion_payload(evidence, validated)
            completion["runtime"] = dict(router_runtime_meta)
            completion.update({
                "images": [item.model_dump() for item in evidence if item.source_type == "image"][:3],
                "elapsed_ms": elapsed_ms,
                "timings": {**(timings or {}), "answer_first_token_ms": elapsed_ms, "total_ms": elapsed_ms},
                "budget": budget_info,
            })
            completion["warnings"] = list(completion.get("warnings") or []) + budget_warnings
            yield {"event": "completed", "data": completion}
            return
        if spec.route != "CASUAL" and not evidence:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            yield {
                "event": "completed",
                "data": {
                    "answer": "当前领域的已发布图谱暂未返回足够的可核验资料，暂时不能确认这个问题。",
                    "citations": [],
                    "claims": [],
                    "evidence_bindings": [],
                    "evidence": [],
                    "images": [],
                    "warnings": ["没有可展示的检索来源，未使用模型自由生成答案"],
                    "runtime": dict(router_runtime_meta),
                    "budget": budget_info,
                    "elapsed_ms": elapsed_ms,
                    "timings": {**(timings or {}), "answer_first_token_ms": 0, "total_ms": elapsed_ms},
                },
            }
            return
        chunks: list[str] = []
        first_token_ms: Optional[int] = None
        try:
            for chunk in self.answer_client.stream(prompt):
                chunks.append(chunk)
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - started) * 1000)
                yield {"event": "token", "data": {"text": chunk}}
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            yield {"event": "error", "data": {"message": str(exc)[:500], "elapsed_ms": elapsed_ms}}
            return

        runtime_meta = dict(getattr(self.answer_client, "last_runtime_meta", {}) or {})
        if runtime_meta.get("finish_reason") == "length":
            budget_warnings.append("答案模型达到输出长度上限，答案可能被截断。")
        answer = "".join(chunks).strip()
        if spec.route == "CASUAL":
            final_answer = _strip_citation_markers(answer)
            final_citations = []
            final_claims = []
            final_warnings = []
        elif not evidence:
            final_answer = "当前领域的已发布图谱暂未返回足够的可核验资料，暂时不能确认这个问题。"
            final_citations = []
            final_claims = []
            final_warnings = ["没有可展示的检索来源，未使用模型自由生成答案"]
        else:
            draft = AnswerDraft(answer=answer, citation_labels=_labels_from_answer(answer))
            draft_claims = getattr(draft, "claims", [])
            validation_evidence = _evidence_for_claims(evidence, draft_claims) if draft_claims else evidence
            semantic_validation = os.getenv("KG_QA_STREAM_SEMANTIC_VALIDATION", "0").strip().lower() in {"1", "true", "yes"}
            validated = self.validator.validate(draft, validation_evidence, semantic=semantic_validation)
            final_answer = validated.answer
            final_citations = [item.model_dump() for item in validated.citations]
            final_claims = validated.claims
            final_warnings = validated.warnings
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        timing_data = {**(timings or {}), "answer_first_token_ms": first_token_ms or elapsed_ms, "total_ms": elapsed_ms}
        completion = _build_completion_payload(
            evidence,
            ValidatedAnswer(
                answer=final_answer,
                citations=final_citations,
                claims=final_claims,
                warnings=final_warnings,
                runtime_meta=runtime_meta,
            ),
        )
        completion["runtime"] = {
            **router_runtime_meta,
            **(completion.get("runtime") or {}),
        }
        completion.update({
            "images": [
                item.model_dump()
                for item in evidence
                if item.source_type == "image"
            ][:3],
            "budget": budget_info,
            "elapsed_ms": elapsed_ms,
            "timings": timing_data,
        })
        completion["warnings"] = list(completion.get("warnings") or []) + budget_warnings
        yield {"event": "completed", "data": completion}


def encode_sse(event: dict[str, Any]) -> str:
    """将单个事件编码为浏览器 EventSource 可消费的 SSE 帧。"""
    name = str(event.get("event") or "message")
    data = json.dumps(event.get("data") or {}, ensure_ascii=False)
    return f"event: {name}\ndata: {data}\n\n"



def _build_completion_payload(
    evidence: list[Evidence],
    validated: ValidatedAnswer,
) -> dict[str, Any]:
    """把完成阶段的答案、Claim 和 Evidence 统一成稳定的 SSE 契约。

    Claim 存在时以 ClaimEvidenceBinding 为准生成 citations；没有 Claim 时
    保留 CitationValidator 的结果，但仍按 evidence 输入顺序重建显示编号。
    evidence 列表始终保持融合后的原始顺序，保证 E{n} 与前端证据位置一致。
    """
    evidence_index = {
        item.evidence_id: index
        for index, item in enumerate(evidence, start=1)
    }
    warnings = list(validated.warnings or [])
    normalized_claims = _coerce_claim_results(validated.claims)
    bindings = []
    claims_payload = []

    if normalized_claims:
        bound = bind_claims_to_evidence(
            normalized_claims,
            evidence,
            strict=False,
        )
        warnings.extend(bound.warnings)
        bindings = [
            item.model_dump()
            for item in sorted(bound.bindings, key=lambda item: item.display_index)
        ]
        claims_payload = [item.model_dump() for item in bound.claims]

    citation_by_evidence: dict[str, dict[str, Any]] = {}
    for item in validated.citations or []:
        record = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        evidence_id = str(record.get("evidence_id") or "").strip()
        if evidence_id in evidence_index:
            citation_by_evidence.setdefault(evidence_id, record)

    if bindings:
        citations = []
        for binding in bindings:
            evidence_id = binding["evidence_id"]
            record = dict(citation_by_evidence.get(evidence_id) or {
                "evidence_id": evidence_id,
                "status": "STRUCTURAL",
                "confidence": 1.0,
                "reason": "",
            })
            record["label"] = f"E{binding['display_index']}"
            record["evidence_id"] = evidence_id
            citations.append(record)
    else:
        citations = []
        for evidence_id, record in citation_by_evidence.items():
            record = dict(record)
            record["label"] = f"E{evidence_index[evidence_id]}"
            citations.append(record)
        citations.sort(key=lambda item: int(str(item["label"])[1:]))

    return {
        "answer": validated.answer,
        "citations": citations,
        "claims": claims_payload,
        "evidence_bindings": bindings,
        "evidence": [item.model_dump() for item in evidence],
        "warnings": warnings,
        "runtime": dict(validated.runtime_meta or {}),
    }


def _coerce_claim_results(values: Any) -> list[ClaimResult]:
    """兼容旧的 dict claims，同时转换为统一 ClaimResult。"""
    result: list[ClaimResult] = []
    for index, value in enumerate(values or [], start=1):
        raw = value.model_dump() if hasattr(value, "model_dump") else dict(value or {})
        evidence_ids = raw.get("evidence_ids") or []
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        claim_text = str(
            raw.get("claim")
            or raw.get("text")
            or raw.get("value")
            or raw.get("request")
            or f"声明 {index}"
        ).strip()
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        result.append(
            ClaimResult(
                claim_id=str(raw.get("claim_id") or f"stream-claim-{index}"),
                claim=claim_text,
                claim_type=str(raw.get("claim_type") or raw.get("request") or "FACT"),
                evidence_ids=[str(item).strip() for item in evidence_ids if str(item).strip()],
                confidence=max(0.0, min(1.0, confidence)),
            )
        )
    return result

def _build_stream_prompt(question: str, spec: QuerySpec, evidence_context: str) -> str:
    separator = chr(10)
    if spec.route == "CASUAL":
        return (
            "这是闲聊，请自然回答，不需要检索、引用或证据标签。"
            "不要输出 [E1]、[E2] 等引用标记，只输出答案正文。"
            + separator
            + f"用户问题：{question}"
        )
    image_rule = ""
    if spec.intent == "IMAGE" and spec.constraints.get("image_path"):
        image_rule = (
            "这是上传图片识别问题。若证据中有图片匹配节点，必须先写“根据上传图片，"
            "最匹配的已发布图谱节点是……”；不要把图片识别结果写成“当前位置”，"
            "除非用户明确询问 GPS 当前所在位置。"
        )
    return (
        "请回答用户问题，只能使用下面的证据。先给直接结论，再补充必要的层级、属性或关系信息。"
        "答案使用自然中文，分成 1 到 3 个短段落，不要重复同一事实，不要把原始关系逐条堆叠。"
        "位置问题优先写成“对象位于……”，不要出现“实体”“检索过程”等技术词。"
        + image_rule
        + "每个事实句末尾使用证据标签 [E1]、[E2]，标签只能来自给定证据；"
        "不要输出模型私有思维链，不要输出 JSON，只输出答案正文。"
        + separator
        + f"用户问题：{question}"
        + separator
        + f"问题规格：{json.dumps(spec.model_dump(), ensure_ascii=False)}"
        + separator
        + "证据："
        + separator
        + evidence_context
    )


def _build_budgeted_evidence_context(
    question: str,
    spec: QuerySpec,
    evidence: list[Evidence],
    budget_manager: ContextBudgetManager,
) -> tuple[str, dict[str, Any], list[str]]:
    """为流式答案保留问题/规格后，把剩余输入预算分配给证据。"""
    empty_evidence_prompt = _build_stream_prompt(question, spec, "暂无可用证据。")
    prompt_overhead_tokens = budget_manager.estimate_tokens(empty_evidence_prompt)
    sections = [
        BudgetSection(
            name=f"evidence:{index}",
            text=_evidence_context_entry(index, item),
            priority=1000 - index,
        )
        for index, item in enumerate(evidence, start=1)
    ]
    result = budget_manager.pack(
        sections,
        extra_reserved_tokens=prompt_overhead_tokens,
    )
    original_tokens = sum(budget_manager.estimate_tokens(item.text) for item in sections)
    if result.text:
        context = result.text
    elif evidence:
        context = "证据因输入预算不足未注入。"
    else:
        context = "暂无可用证据。"
    warnings: list[str] = []
    if result.truncated or result.dropped:
        warnings.append("证据上下文超出输入预算，已裁剪或省略部分证据。")
    if prompt_overhead_tokens > budget_manager.input_budget_tokens:
        warnings.append("问题或问题规格本身已接近上下文上限，证据可用预算不足。")
    info = {
        "max_context_tokens": budget_manager.max_context_tokens,
        "input_budget_tokens": budget_manager.input_budget_tokens,
        "prompt_overhead_tokens": prompt_overhead_tokens,
        "evidence_budget_tokens": result.budget_tokens,
        "evidence_original_tokens": original_tokens,
        "evidence_used_tokens": result.used_tokens,
        "evidence_truncated": bool(result.truncated),
        "evidence_dropped": list(result.dropped),
        "warnings": list(warnings),
    }
    return context, info, warnings


def _evidence_context_entry(index: int, item: Evidence) -> str:
    return (
        f"[E{index}] evidence_id={item.evidence_id}; source_type={item.source_type}; "
        f"content={item.content}; source_url={item.source_url or ''}"
    )


def _evidence_for_claims(evidence: list[Evidence], claims: list[dict[str, Any]]) -> list[Evidence]:
    ids = {str(value) for claim in claims for value in (claim.get("evidence_ids") or []) if isinstance(claim, dict)}
    return [item for item in evidence if item.evidence_id in ids] if claims else evidence
