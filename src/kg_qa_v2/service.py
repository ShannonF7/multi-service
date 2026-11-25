"""QA v2 的生产编排门面。

本模块把已有窄适配器组装起来，提供同步 ask() 和 SSE 事件流 stream_ask()。
主应用通过独立 v2 路由调用本门面，旧版路由保留作为回退。
"""

from __future__ import annotations

import inspect
import os
import time
from queue import Queue
from threading import Thread
from typing import Any, Iterator, Optional

from .adapters.graph_adapters import build_published_graph_ports
from .adapters.qwen_client import build_qwen_slot_client_from_env
from .adapters.retrieval_adapters import build_domain_kb_retrievers
from .core.answering import (
    AnswerGenerator,
    CitationValidator,
    ValidatedAnswer,
    fast_attribute_answer,
    fast_attribute_unavailable_answer,
    fast_casual_answer,
    fast_fact_answer,
    fast_graph_answer,
    fast_nearby_answer,
    fast_visual_answer,
)
from .core.clarification import resolve_pending_clarification
from .core.decision_model import build_decision_model_from_env
from .core.entity import EntityAlignmentService
from .core.memory import ContextResolver
from .core.pipeline import QAV2Pipeline
from .core.retrieval import Evidence, PlanExecutor
from .core.streaming import LocalStreamingAnswerClient, QAStreamService
from .core.understanding import QuestionUnderstandingService
from .schemas import QAResponse


def _build_answer_generator_from_env() -> AnswerGenerator:
    """按答案模型配置构造同步答案生成器。

    API 模式和本地模式必须在这里完成分流；不能先把本地 endpoint/model
    传入 AnswerGenerator，再期待它在内部猜测 provider，否则 API 配置会被
    本地默认值覆盖。
    """
    provider = (os.getenv("ANSWER_PROVIDER") or "local").strip().lower()
    is_api = provider in {"api", "openai", "openai_compatible", "openai-compatible"}
    if not is_api and provider not in {"local", "vllm"}:
        raise ValueError("不支持的 ANSWER_PROVIDER；可选 local 或 api")
    if is_api:
        return AnswerGenerator(
            provider=provider,
            base_url=os.getenv("ANSWER_API_BASE_URL") or os.getenv("ANSWER_BASE_URL"),
            model=os.getenv("ANSWER_API_MODEL") or os.getenv("ANSWER_MODEL"),
            api_key=os.getenv("ANSWER_API_KEY") or os.getenv("LLM_API_KEY"),
            timeout=float(os.getenv("ANSWER_API_TIMEOUT", "120")),
        )
    return AnswerGenerator(
        provider="local",
        base_url=os.getenv("LOCAL_ANSWER_BASE_URL", os.getenv("LOCAL_DECISION_BASE_URL", "http://127.0.0.1:7020/v1")),
        model=os.getenv("LOCAL_ANSWER_MODEL", os.getenv("LOCAL_DECISION_MODEL", "Qwen2.5-7B-Instruct-AWQ")),
        timeout=float(os.getenv("LOCAL_ANSWER_TIMEOUT", os.getenv("LOCAL_DECISION_TIMEOUT", "60"))),
    )


def _build_stream_answer_client(answer_generator: AnswerGenerator) -> Any:
    """使用答案生成器的 endpoint/model，并在客户端支持时透传 provider/key。

    旧版 LocalStreamingAnswerClient 只有 local 参数，不能伪造 API 鉴权能力；
    因此 provider/api_key 仅在构造器明确声明支持时传入，保持旧 local 兼容。
    """
    kwargs: dict[str, Any] = {
        "base_url": answer_generator.base_url,
        "model": answer_generator.model,
        "timeout": answer_generator.timeout,
    }
    parameters = inspect.signature(LocalStreamingAnswerClient).parameters
    if "provider" in parameters:
        kwargs["provider"] = getattr(answer_generator, "provider", "local")
    if "api_key" in parameters:
        kwargs["api_key"] = getattr(answer_generator, "api_key", None)
    return LocalStreamingAnswerClient(**kwargs)


class QAV2Service:
    """面向接口层的同步和流式 QA v2 服务。"""

    def __init__(
        self,
        *,
        pipeline: QAV2Pipeline,
        answer_generator: AnswerGenerator,
        validator: Optional[CitationValidator] = None,
        stream_service: Optional[QAStreamService] = None,
        session_store: Optional[Any] = None,
    ) -> None:
        self.pipeline = pipeline
        self.answer_generator = answer_generator
        self.validator = validator or CitationValidator()
        self.session_store = session_store
        self.stream_service = stream_service or QAStreamService(
            answer_client=_build_stream_answer_client(answer_generator),
            validator=self.validator,
        )

    @classmethod
    def from_env(cls) -> "QAV2Service":
        """使用当前部署的本地决策模型、图谱、文本和图片服务。"""
        decision_model = build_decision_model_from_env()
        if decision_model is None and os.getenv("DECISION_MODEL_PROVIDER", "auto").strip().lower() in {"", "auto", "local"}:
            from .core.decision_model import CascadingDecisionClient, LightweightRouterClient, LocalDecisionClient
            fallback = LocalDecisionClient(
                base_url=os.getenv("LOCAL_DECISION_BASE_URL", "http://127.0.0.1:7020/v1"),
                model=os.getenv("LOCAL_DECISION_MODEL", "Qwen2.5-7B-Instruct-AWQ"),
                timeout=float(os.getenv("LOCAL_DECISION_TIMEOUT", "60")),
            )
            decision_model = CascadingDecisionClient(
                LightweightRouterClient(),
                fallback,
                threshold=float(os.getenv("LOCAL_ROUTER_CONFIDENCE_THRESHOLD", "0.82")),
            )
        understanding = QuestionUnderstandingService(
            decision_model=decision_model,
            slot_client=build_qwen_slot_client_from_env(),
        )
        candidate_search, graph_gateway = build_published_graph_ports()
        text_retriever, image_retriever = build_domain_kb_retrievers()
        session_store = None
        context_resolver = ContextResolver()
        try:
            from src.kg_qa_shared.session import PersistentConversationSessionStore
            session_store = PersistentConversationSessionStore()
            context_resolver = ContextResolver(
                session_reader=lambda **kwargs: _legacy_session_state(session_store, **kwargs),
                long_term_reader=lambda **kwargs: _legacy_long_term_memory(session_store, **kwargs),
            )
        except Exception:
            session_store = None
        pipeline = QAV2Pipeline(
            context_resolver=context_resolver,
            understanding=understanding,
            candidate_search=candidate_search,
            alignment=EntityAlignmentService(decision_model),
            executor=PlanExecutor(
                graph_retriever=graph_gateway.retrieve,
                fact_retriever=graph_gateway.retrieve_facts,
                text_retriever=text_retriever,
                image_retriever=image_retriever,
            ),
        )
        answer_generator = _build_answer_generator_from_env()
        validator = CitationValidator(decision_model)
        return cls(
            pipeline=pipeline,
            answer_generator=answer_generator,
            validator=validator,
            session_store=session_store,
        )

    def ask(
        self,
        question: str,
        *,
        domain_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        current_location: Optional[dict[str, Any]] = None,
        has_image: bool = False,
        image_path: Optional[str] = None,
        target_modality: Optional[str] = None,
        max_evidence: int = 8,
    ) -> QAResponse:
        started = time.perf_counter()
        session_context = self._open_session_context(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
        )
        memory_process = self._prepare_session(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
            context=session_context,
        )
        session_context = self._open_session_context(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
        )
        pending_resolution = resolve_pending_clarification(
            question,
            session_context.pending_clarification if session_context else None,
        )
        effective_question = (
            pending_resolution.get("original_question") or question
            if pending_resolution else question
        )
        result = self.pipeline.run(
            effective_question,
            domain_id=domain_id,
            session_id=session_id,
            user_id=user_id,
            current_location=current_location,
            has_image=has_image,
            image_path=image_path,
            target_modality=target_modality,
            max_evidence=max_evidence,
            clarification_selection=pending_resolution,
        )
        process = [*memory_process, *_process_for(result)]
        evidence = result.retrieval.evidence
        if result.status == "NEEDS_CLARIFICATION":
            answer = "；".join(result.warnings) or "请补充一下你要查询的位置。"
            clarification = _clarification_payload(result, question, domain_id)
            self._append_session(
                session_id=session_id,
                user_id=user_id,
                domain_id=domain_id,
                question=question,
                answer=answer,
            )
            self._set_pending_clarification(
                session_id=session_id,
                user_id=user_id,
                domain_id=domain_id,
                payload=clarification,
            )
            return _response(
                result,
                answer=answer,
                evidence=evidence,
                process=process,
                clarification=clarification,
                elapsed_ms=_elapsed_ms(started),
                timings={**result.timings, "total_ms": _elapsed_ms(started)},
            )

        if not evidence and result.understanding.route.route != "CASUAL" and not result.understanding.query_spec.requests:
            answer = "当前领域的已发布图谱暂未返回足够的可核验资料，暂时不能确认这个问题。"
            self._clear_pending_clarification(
                session_id=session_id,
                user_id=user_id,
                domain_id=domain_id,
            )
            self._append_session(
                session_id=session_id,
                user_id=user_id,
                domain_id=domain_id,
                question=question,
                answer=answer,
            )
            return _response(
                result,
                answer=answer,
                evidence=[],
                warnings=["没有可展示的检索来源，未使用模型自由生成答案"],
                process=process,
                elapsed_ms=_elapsed_ms(started),
                timings={**result.timings, "answer_ms": 0, "total_ms": _elapsed_ms(started)},
            )

        answer_started = time.perf_counter()
        draft = fast_casual_answer(question) if result.understanding.route.route == "CASUAL" else None
        if draft is None:
            draft = fast_nearby_answer(result.understanding.query_spec, evidence)
        if draft is None:
            draft = fast_visual_answer(result.understanding.query_spec, evidence)
        if draft is None:
            draft = fast_graph_answer(result.understanding.query_spec, evidence)
        attribute_unavailable = None
        if draft is None:
            draft = fast_fact_answer(result.understanding.query_spec, evidence)
        if draft is None:
            draft = fast_attribute_answer(result.understanding.query_spec, evidence)
        if draft is None:
            attribute_unavailable = fast_attribute_unavailable_answer(
                result.understanding.query_spec,
                evidence,
            )
            draft = attribute_unavailable
        is_fast_draft = draft is not None
        if draft is None:
            draft = self.answer_generator.generate(
                effective_question,
                spec=result.understanding.query_spec,
                evidence=evidence,
            )
        if attribute_unavailable is not None:
            validated = ValidatedAnswer(
                answer=attribute_unavailable.answer,
                citations=[],
                warnings=[],
            )
        else:
            try:
                draft_claims = getattr(draft, "claims", [])
                validation_evidence = _evidence_for_claims(evidence, draft_claims) if draft_claims else evidence
                validated = self.validator.validate(
                    draft,
                    validation_evidence,
                    semantic=not is_fast_draft,
                )
            except TypeError as exc:
                # 兼容旧版注入式校验器；生产 CitationValidator 支持 semantic 开关。
                if "semantic" not in str(exc):
                    raise
                validated = self.validator.validate(draft, evidence)
        answer_ms = int((time.perf_counter() - answer_started) * 1000)
        process.append({"stage": "answer_validation", "status": "completed"})
        self._clear_pending_clarification(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
        )
        self._append_session(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
            question=question,
            answer=validated.answer,
            entities=[
                item.name for item in result.alignments
                if item.status == "LINKED" and item.name
            ],
            active_entity=_active_entity_payload(result),
        )
        return _response(
            result,
            answer=validated.answer,
            citations=[item.model_dump() for item in validated.citations],
            evidence=_evidence_for_claims(evidence, getattr(validated, "claims", [])),
            claims=getattr(validated, "claims", []),
            images=_images_for(result.understanding.query_spec, evidence),
            warnings=[*result.warnings, *validated.warnings],
            process=process,
            elapsed_ms=_elapsed_ms(started),
            timings={**result.timings, "answer_ms": answer_ms, "total_ms": _elapsed_ms(started)},
            runtime=getattr(validated, "runtime_meta", {}) or {},
        )

    def stream_ask(
        self,
        question: str,
        *,
        domain_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        current_location: Optional[dict[str, Any]] = None,
        has_image: bool = False,
        image_path: Optional[str] = None,
        target_modality: Optional[str] = None,
        max_evidence: int = 8,
    ) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        yield {"event": "started", "data": {"message": "小V开始整理答案"}}
        yield {
            "event": "process",
            "data": {
                "stage": "session",
                "status": "started",
                "message": "正在读取会话上下文",
            },
        }
        session_context = self._open_session_context(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
        )
        memory_process = self._prepare_session(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
            context=session_context,
        )
        session_context = self._open_session_context(
            session_id=session_id,
            user_id=user_id,
            domain_id=domain_id,
        )
        pending_resolution = resolve_pending_clarification(
            question,
            session_context.pending_clarification if session_context else None,
        )
        effective_question = (
            pending_resolution.get("original_question") or question
            if pending_resolution else question
        )
        yield {
            "event": "process",
            "data": {
                "stage": "session",
                "status": "completed",
                "message": "已读取当前会话和领域上下文",
            },
        }
        for item in memory_process:
            yield {"event": "process", "data": item}

        queue: Queue[tuple[str, Any]] = Queue()
        pipeline_events: list[dict[str, Any]] = []

        def on_pipeline_event(event: dict[str, Any]) -> None:
            pipeline_events.append(event)
            queue.put(("process", event))

        def run_pipeline() -> None:
            try:
                value = self.pipeline.run(
                    effective_question,
                    domain_id=domain_id,
                    session_id=session_id,
                    user_id=user_id,
                    current_location=current_location,
                    has_image=has_image,
                    image_path=image_path,
                    target_modality=target_modality,
                    max_evidence=max_evidence,
                    clarification_selection=pending_resolution,
                    on_event=on_pipeline_event,
                )
                queue.put(("result", value))
            except Exception as exc:
                queue.put(("error", exc))

        Thread(target=run_pipeline, name="kg-qa-v2-pipeline", daemon=True).start()
        try:
            while True:
                kind, payload = queue.get()
                if kind == "process":
                    yield {"event": "process", "data": payload}
                    continue
                if kind == "error":
                    raise payload
                result = payload
                break

            process = [] if pipeline_events else _process_for(result)
            if result.status == "NEEDS_CLARIFICATION":
                answer = "；".join(result.warnings) or "请补充一下你要查询的位置。"
                clarification = _clarification_payload(result, question, domain_id)
                self._append_session(
                    session_id=session_id,
                    user_id=user_id,
                    domain_id=domain_id,
                    question=question,
                    answer=answer,
                )
                self._set_pending_clarification(
                    session_id=session_id,
                    user_id=user_id,
                    domain_id=domain_id,
                    payload=clarification,
                )
                yield {
                    "event": "completed",
                    "data": {
                        "answer": answer,
                        "citations": [],
                        "images": [],
                        "warnings": result.warnings,
                        "clarification": clarification,
                        "elapsed_ms": _elapsed_ms(started),
                        "timings": {**result.timings, "total_ms": _elapsed_ms(started)},
                    },
                }
                return
            for event in self.stream_service.stream(
                effective_question,
                spec=result.understanding.query_spec,
                evidence=result.retrieval.evidence,
                process=process,
                emit_started=False,
                started_at=started,
                timings=result.timings,
                runtime_meta=result.understanding.runtime_meta,
            ):
                if event.get("event") == "completed":
                    self._clear_pending_clarification(
                        session_id=session_id,
                        user_id=user_id,
                        domain_id=domain_id,
                    )
                    self._append_session(
                        session_id=session_id,
                        user_id=user_id,
                        domain_id=domain_id,
                        question=question,
                        answer=str((event.get("data") or {}).get("answer") or ""),
                        entities=[
                            item.name for item in result.alignments
                            if item.status == "LINKED" and item.name
                        ],
                        active_entity=_active_entity_payload(result),
                    )
                yield event
        except Exception as exc:
            yield {
                "event": "error",
                "data": {
                    "message": str(exc)[:500],
                    "elapsed_ms": _elapsed_ms(started),
                },
            }

    def _open_session_context(
        self,
        *,
        session_id: Optional[str],
        user_id: Optional[str],
        domain_id: str,
    ) -> Any:
        if not self.session_store or not session_id:
            return None
        try:
            return self.session_store.open(
                session_id,
                user_id=user_id,
                domain_id=domain_id,
            )
        except Exception:
            return None

    def _set_pending_clarification(self, *, session_id: Optional[str], user_id: Optional[str],
                                   domain_id: str, payload: dict[str, Any]) -> None:
        context = self._open_session_context(session_id=session_id, user_id=user_id, domain_id=domain_id)
        if context is not None and hasattr(self.session_store, "set_pending_clarification"):
            try:
                self.session_store.set_pending_clarification(context, payload)
            except Exception:
                pass

    def _clear_pending_clarification(self, *, session_id: Optional[str], user_id: Optional[str],
                                      domain_id: str) -> None:
        context = self._open_session_context(session_id=session_id, user_id=user_id, domain_id=domain_id)
        if context is not None and hasattr(self.session_store, "clear_pending_clarification"):
            try:
                self.session_store.clear_pending_clarification(context)
            except Exception:
                pass

    def _prepare_session(
        self,
        *,
        session_id: Optional[str],
        user_id: Optional[str],
        domain_id: str,
        context: Any = None,
    ) -> list[dict[str, Any]]:
        if not self.session_store or not session_id:
            return []
        try:
            context = context or self.session_store.open(
                session_id,
                user_id=user_id,
                domain_id=domain_id,
            )
            if not context.needs_compression:
                return []
            self.session_store.compress(context)
            return [{
                "stage": "memory_compression",
                "status": "completed",
                "message": "已整理较早的会话记录，保留近期对话和摘要",
            }]
        except Exception:
            return [{
                "stage": "memory_compression",
                "status": "failed",
                "message": "会话摘要暂时不可用，本轮继续使用当前问题",
            }]

    def _append_session(
        self,
        *,
        session_id: Optional[str],
        user_id: Optional[str],
        domain_id: str,
        question: str,
        answer: str,
        entities: Optional[list[str]] = None,
        active_entity: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.session_store or not session_id:
            return
        try:
            context = self.session_store.open(
                session_id,
                user_id=user_id,
                domain_id=domain_id,
            )
            self.session_store.append(
                context,
                question,
                answer,
                entities=entities or [],
                active_entity=active_entity,
            )
        except Exception:
            return



def _active_entity_payload(result: Any) -> dict[str, Any] | None:
    linked = next((item for item in reversed(result.alignments) if item.status == "LINKED"), None)
    if linked is None:
        return None
    candidate = linked.candidates[0] if linked.candidates else None
    context = result.context
    node_id = linked.node_id or context.current_node_id
    name = linked.name or context.current_node_name or (candidate.name if candidate else None)
    if not node_id or not name:
        return None
    return {"node_id": str(node_id), "name": str(name), "domain_id": context.domain_id, "hierarchy_path": list(candidate.hierarchy_path) if candidate else [], "node_type": candidate.node_type if candidate else None, "last_focus": result.understanding.query_spec.intent}

def _legacy_session_state(store: Any, **kwargs: Any) -> dict[str, Any]:
    context = store.open(
        kwargs.get("session_id"),
        user_id=kwargs.get("user_id"),
        domain_id=kwargs.get("domain_id"),
    )
    recent_entities = []
    active_state = {}
    for turn in reversed(context.turns or []):
        if not active_state:
            states = turn.get("entity_states") or []
            if states and isinstance(states[0], dict):
                active_state = dict(states[0])
        recent_entities.extend(
            str(value).strip()
            for value in (turn.get("entities") or [])
            if str(value).strip()
        )
    recent_entities = list(dict.fromkeys(recent_entities))[:5]
    messages = context.to_dict().get("messages") or []
    return {
        "entity_state": {
            "active_entity": active_state.get("name") or (recent_entities[0] if recent_entities else None),
            "active_node_id": active_state.get("node_id"),
            "active_domain_id": active_state.get("domain_id"),
            "active_hierarchy_path": list(active_state.get("hierarchy_path") or []),
            "active_node_type": active_state.get("node_type"),
            "last_focus": active_state.get("last_focus"),
            "recent_entities": recent_entities,
            "pending_candidates": list(
                (context.pending_clarification or {}).get("pending_candidates") or []
            ),
        },
        "session_summary": {
            "summary": context.summary or None,
            "confirmed_entities": recent_entities,
        },
        "messages": messages[-8:],
        "recent_messages": messages[-8:],
        "last_question": (
            str((context.turns or [])[-1].get("user") or "")
            if context.turns else None
        ),
        "pending_clarification": context.pending_clarification or {},
        "long_term_memory": dict(context.long_term_memory or {}),
    }

def _legacy_long_term_memory(store: Any, **kwargs: Any) -> list[dict[str, Any]]:
    context = store.open(kwargs.get("session_id"), user_id=kwargs.get("user_id"), domain_id=kwargs.get("domain_id"))
    profile = dict(getattr(context, "long_term_memory", {}) or {})
    result: list[dict[str, Any]] = []
    for key, value in profile.items():
        if key == "known_entities" and isinstance(value, list):
            for index, item in enumerate(value):
                name = str(item or "").strip()
                if name:
                    result.append({"memory_id": f"profile:known_entity:{index}", "content": f"用户曾关注：{name}", "memory_type": "profile", "domain_id": kwargs.get("domain_id"), "confidence": 1.0})
            continue
        result.append({"memory_id": f"profile:{key}", "content": f"{key}: {value}"[:500], "memory_type": "profile", "domain_id": kwargs.get("domain_id"), "confidence": 0.8})
    return result

def _clarification_payload(result: Any, question: str, domain_id: str) -> dict[str, Any]:
    unresolved = [
        item for item in result.alignments
        if item.status != "LINKED"
    ]
    candidates = []
    mention = ""
    for item in unresolved:
        mention = mention or item.mention
        for candidate in item.candidates:
            value = candidate.model_dump()
            if not any(existing.get("node_id") == value.get("node_id") for existing in candidates):
                candidates.append(value)
    return {
        "pending_mention": mention,
        "pending_candidates": candidates,
        "pending_question": question,
        "pending_domain_id": domain_id,
        "pending_query_spec": result.understanding.query_spec.model_dump(),
        "pending_route": result.understanding.route.model_dump(),
    }

def _response(
    result: Any,
    *,
    answer: str,
    evidence: list[Evidence],
    process: list[dict[str, Any]],
    elapsed_ms: int,
    citations: Optional[list[dict[str, Any]]] = None,
    images: Optional[list[dict[str, Any]]] = None,
    warnings: Optional[list[str]] = None,
    timings: Optional[dict[str, int]] = None,
    clarification: Optional[dict[str, Any]] = None,
    claims: Optional[list[dict[str, Any]]] = None,
    runtime: Optional[dict[str, Any]] = None,
) -> QAResponse:
    merged_runtime = {
        **(getattr(result.understanding, "runtime_meta", {}) or {}),
        **(runtime or {}),
    }
    return QAResponse(
        status=result.status,
        answer=answer,
        citations=citations or [],
        claims=claims or [],
        images=images or [],
        evidence=[item.model_dump() for item in evidence],
        warnings=warnings or [],
        process=process,
        elapsed_ms=elapsed_ms,
        timings=timings or {},
        runtime=merged_runtime,
        route=result.understanding.route.route,
        intent=result.understanding.query_spec.intent,
        clarification=clarification,
    )



def _evidence_for_claims(evidence: list[Evidence], claims: list[dict[str, Any]]) -> list[Evidence]:
    ids = {str(value) for claim in claims for value in (claim.get("evidence_ids") or []) if isinstance(claim, dict)}
    return [item for item in evidence if item.evidence_id in ids] if claims else evidence

def _images_for(spec: Any, evidence: list[Evidence]) -> list[dict[str, Any]]:
    if not spec.needs_image and spec.intent != "IMAGE":
        return []
    return [item.model_dump() for item in evidence if item.source_type == "image"][:3]


def _process_for(result: Any) -> list[dict[str, Any]]:
    process = [
        {"stage": "context", "status": "completed", "message": "已读取当前会话和领域上下文"},
        {"stage": "understanding", "status": "completed", "message": "已完成问题理解与模态判断"},
    ]
    if result.alignments:
        process.append({"stage": "entity_alignment", "status": "completed", "message": "已完成候选位置对齐"})
    if result.understanding.route.route == "CASUAL":
        return process
    for step in result.plan.steps:
        if step.type in {"GRAPH", "FACT", "TEXT", "IMAGE"}:
            process.append({
                "stage": step.type.lower(),
                "status": "completed" if step.type in result.retrieval.executed_steps else "failed",
                "message": step.reason,
            })
    process.append({"stage": "evidence_fusion", "status": "completed", "message": "已整理可引用来源"})
    return process


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
