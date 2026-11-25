"""在线 KG-QA 主编排服务。"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from .answer import CasualChatGenerator, QAAnswerGenerator, build_default_answer_generator, validate_generated_answer
from .domain import DomainAccessError, DomainGuard
from .entity_linker import EntityLinker
from .evidence import fuse_evidence
from .input_analyzer import analyze_input, is_casual_message, is_current_location_question
from .planner import build_query_plan
from .retriever import DefaultRetriever
from .schemas import QARequest, QAResponse, QueryPlan, TraceEvent
from .session import ConversationContext, ConversationSessionStore, build_default_session_store
from .trace import TraceCollector
from src.reasoning.executor import ReasoningExecutor
from src.reasoning.integration import build_logical_query
from src.reasoning.response import unresolved_position_answer


def _select_answer_images(evidence: list[Any], analysis: Any, message: str) -> list[dict[str, Any]]:
    """仅在图片意图下返回答案区图片，默认最多三张。"""
    if "IMAGE" not in set(analysis.knowledge_intents or []) and analysis.target_modality != "IMAGE":
        return []
    all_requested = any(token in str(message or "") for token in ("全部图片", "所有图片", "全部照片", "所有照片"))
    limit = 20 if all_requested else 3
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in evidence:
        candidates = []
        if getattr(item, "image_url", None):
            candidates.append({"image_url": item.image_url, "title": item.title})
        metadata = getattr(item, "metadata", {}) or {}
        candidates.extend(asset for asset in (metadata.get("assets") or []) if isinstance(asset, dict))
        for asset in candidates:
            image_url = str(asset.get("image_url") or "").strip()
            if not image_url or image_url in seen:
                continue
            seen.add(image_url)
            selected.append({
                "image_url": image_url,
                "title": str(asset.get("title") or getattr(item, "title", None) or "").strip() or None,
                "evidence_id": getattr(item, "evidence_id", None),
            })
            if len(selected) >= limit:
                return selected
    return selected


def _total_duration(trace: TraceCollector) -> float:
    return round(sum(
        event.duration_ms or 0.0
        for event in trace.events
        if event.status in {"COMPLETED", "FAILED"}
    ), 2)


def _context_entity_names(context: ConversationContext) -> list[str]:
    for turn in reversed(context.turns or []):
        names = [str(value).strip() for value in (turn.get("entities") or []) if str(value).strip()]
        if names:
            return list(dict.fromkeys(names))[:3]
    return []


def _candidate_label(candidate: Any) -> str:
    path=list(dict.fromkeys(str(value).strip() for value in (candidate.hierarchy_path or []) if str(value).strip()))
    if path and path[-1]==candidate.name: path=path[:-1]
    label=candidate.name or candidate.node_id
    if path: label += f"（{'·'.join(path)}）"
    return label



def _candidate_payload(candidate: Any) -> dict[str, Any]:
    if hasattr(candidate, "model_dump"):
        return candidate.model_dump()
    if hasattr(candidate, "dict"):
        return candidate.dict()
    return dict(candidate or {})


def _candidate_from_payload(value: dict[str, Any]) -> Any:
    from .schemas import EntityCandidate
    return EntityCandidate(
        node_id=str(value.get("node_id") or ""),
        name=str(value.get("name") or ""),
        node_type=value.get("node_type"),
        match_type=value.get("match_type"),
        score=value.get("score"),
        parent_name=value.get("parent_name"),
        grandparent_name=value.get("grandparent_name"),
        hierarchy_path=[
            str(item).strip() for item in (value.get("hierarchy_path") or [])
            if item is not None and str(item).strip()
        ],
        distance_m=value.get("distance_m"),
    )


def _normalize_choice_text(value: str) -> str:
    return re.sub(r"[\s，。！？?；;、的那这个请选我]", "", str(value or "").strip().replace("楼", "层"))


def _choice_index(value: str) -> int | None:
    text = str(value or "").strip()
    match = re.search(r"第\s*([0-9一二三四五六七八九十]+)\s*(?:个|项|号)?", text)
    if not match and re.fullmatch(r"[0-9一二三四五六七八九十]+", text):
        match = re.match(r"([0-9一二三四五六七八九十]+)", text)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    numbers = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if token in numbers:
        return numbers[token]
    return None


def _resolve_pending_choice(message: str, pending: dict[str, Any] | None) -> tuple[dict[str, Any], int] | None:
    if not pending:
        return None
    options = [item for item in (pending.get("candidates") or []) if isinstance(item, dict)]
    if not options:
        return None
    index = _choice_index(message)
    if index is not None and 1 <= index <= len(options):
        return options[index - 1], index - 1
    query = _normalize_choice_text(message)
    if not query:
        return None
    matches: list[tuple[dict[str, Any], int]] = []
    for option_index, option in enumerate(options):
        values = [option.get("name") or ""]
        values.extend(option.get("hierarchy_path") or [])
        normalized_values = [_normalize_choice_text(value) for value in values if str(value or "").strip()]
        if any(len(value) >= 2 and value in query for value in normalized_values):
            matches.append((option, option_index))
    return matches[0] if len(matches) == 1 else None


def _apply_context_mentions(analysis: Any, message: str, context: ConversationContext) -> Any:
    if not any(token in str(message or "") for token in ("他", "她", "它", "这个", "那个", "这里", "那里", "该校", "该楼")):
        return analysis
    names = _context_entity_names(context)
    if not names:
        return analysis
    explicit_mentions = [
        value for value in (analysis.mentions or [])
        if value and not str(value).startswith(("他", "她", "它", "这个", "那个", "这里", "那里", "该校", "该楼"))
    ]
    mentions = list(dict.fromkeys([*explicit_mentions, *names]))
    if hasattr(analysis, "model_copy"):
        return analysis.model_copy(update={"mentions": mentions})
    return analysis.copy(update={"mentions": mentions})


class QAService:
    def __init__(
        self,
        *,
        domain_guard: Optional[DomainGuard] = None,
        entity_linker: Optional[EntityLinker] = None,
        retriever: Optional[DefaultRetriever] = None,
        answer_generator: Optional[QAAnswerGenerator] = None,
        session_store: Optional[ConversationSessionStore] = None,
        reasoning_executor: Optional[ReasoningExecutor] = None,
        use_reasoning_core: bool = True,
    ) -> None:
        self.domain_guard = domain_guard or DomainGuard()
        self.entity_linker = entity_linker or EntityLinker()
        self.retriever = retriever or DefaultRetriever()
        self.answer_generator = answer_generator or build_default_answer_generator()
        self.session_store = session_store or build_default_session_store()
        self.use_reasoning_core = bool(use_reasoning_core)
        self.reasoning_executor = reasoning_executor or ReasoningExecutor(
            entity_linker=self.entity_linker,
            retriever=self.retriever,
        )

    def run(
        self,
        request: QARequest,
        on_trace: Optional[Callable[[Any], None]] = None,
        on_answer_delta: Optional[Callable[[str], None]] = None,
    ) -> QAResponse:
        trace = TraceCollector()
        warnings: list[str] = []

        def run_stage(stage: str, callback: Callable[[], Any], summary: str) -> Any:
            return trace.run(stage, callback, summary=summary, on_event=on_trace)

        session_id = request.session_id or request.conversation_id
        session_context = ConversationContext(session_id="")
        if session_id:
            session_context = run_stage(
                "SESSION_CONTEXT", lambda: self.session_store.open(
                    session_id, user_id=request.user_id, domain_id=request.domain_id
                ),
                "读取当前会话上下文",
            )
            if session_context.needs_compression:
                session_context = run_stage(
                    "CONTEXT_COMPRESSION", lambda: self.session_store.compress(session_context),
                    "压缩较早的会话内容，保留摘要和最近对话",
                )

        try:
            if is_casual_message(request.message) and not request.image_path and not request.image_ids:
                analysis = run_stage(
                    "INPUT_ANALYSIS", lambda: analyze_input(request),
                    "识别为日常聊天，不执行知识检索",
                )
                if request.user_location:
                    trace.events[-1].metadata = {"user_location": request.user_location.model_dump() if hasattr(request.user_location, "model_dump") else request.user_location.dict()}
                    if on_trace:
                        on_trace(trace.events[-1])

                def generate_chat() -> str:
                    if on_answer_delta:
                        return CasualChatGenerator().generate_stream(
                            request.message,
                            conversation_context=session_context.to_dict(),
                            on_delta=on_answer_delta,
                        )
                    return CasualChatGenerator().generate(
                        request.message,
                        conversation_context=session_context.to_dict(),
                    )

                answer = run_stage(
                    "CASUAL_CHAT", generate_chat,
                    "直接进行轻松对话，不查询图谱或领域资料",
                )
                self.session_store.append(session_context, request.message, answer)
                return QAResponse(
                    run_id=trace.run_id,
                    status="COMPLETED",
                    answer=answer,
                    total_duration_ms=_total_duration(trace),
                    analysis=analysis,
                    trace=trace.events,
                    warnings=[],
                )

            context = run_stage(
                "DOMAIN_GUARD", lambda: self.domain_guard.validate(request),
                "确认请求只访问当前领域的已发布图谱",
            )
            if request.user_location:
                def resolve_location():
                    from src.rag.service.graph_discovery_service import resolve_nearest_published_node
                    location = request.user_location
                    return resolve_nearest_published_node(
                        context.graph_domain_id,
                        latitude=location.latitude,
                        longitude=location.longitude,
                    )
                location_node = run_stage(
                    "LOCATION_CONTEXT", resolve_location,
                    "在当前领域已发布节点中定位用户附近的图谱节点",
                )
                if location_node:
                    update = {
                        "location_node_id": str(location_node.get("node_id") or "") or None,
                        "location_node_name": str(location_node.get("name") or "") or None,
                        "location_distance_m": location_node.get("distance_m"),
                    }
                    if hasattr(context, "model_copy"):
                        context = context.model_copy(update=update)
                    else:
                        context = context.copy(update=update)
            analysis = run_stage(
                "INPUT_ANALYSIS", lambda: analyze_input(
                    request, session_context.to_dict()
                ),
                "识别问题模态、意图和位置提及",
            )
            analysis = _apply_context_mentions(analysis, request.message, session_context)
            effective_request = request
            standalone_question = str(getattr(analysis, "standalone_question", "") or "").strip()
            if standalone_question:
                if hasattr(request, "model_copy"):
                    effective_request = request.model_copy(update={"message": standalone_question})
                else:
                    effective_request = request.copy(update={"message": standalone_question})
            if request.user_location:
                trace.events[-1].metadata = {"user_location": request.user_location.model_dump() if hasattr(request.user_location, "model_dump") else request.user_location.dict()}
                if on_trace:
                    on_trace(trace.events[-1])

            pending_question = None
            reasoning_target_node_id = None
            reasoning_target_name = None
            pending_choice = _resolve_pending_choice(
                request.message, session_context.pending_clarification
            )
            if pending_choice:
                pending = session_context.pending_clarification or {}
                pending_question = str(pending.get("question") or request.message)
                selected_candidate, selected_index = pending_choice
                reasoning_target_node_id = str(selected_candidate.get("node_id") or "") or None
                reasoning_target_name = str(selected_candidate.get("name") or "") or None
                previous_analysis = analyze_input(
                    request.model_copy(update={"message": str(pending.get("question") or request.message)})
                    if hasattr(request, "model_copy")
                    else request.copy(update={"message": str(pending.get("question") or request.message)}),
                    session_context.to_dict(),
                )
                selected_label = _candidate_label(_candidate_from_payload(selected_candidate))
                standalone_question = (
                    f"{pending.get('question') or request.message}；用户选择了第{selected_index + 1}项：{selected_label}"
                )
                if hasattr(analysis, "model_copy"):
                    analysis = analysis.model_copy(update={
                        "mentions": [str(pending.get("mention") or selected_candidate.get("name") or "")],
                        "knowledge_intents": previous_analysis.knowledge_intents,
                        "standalone_question": standalone_question,
                        "intent_source": "rules+clarification",
                    })
                    effective_request = request.model_copy(update={"message": standalone_question})
                else:
                    analysis = analysis.copy(update={
                        "mentions": [str(pending.get("mention") or selected_candidate.get("name") or "")],
                        "knowledge_intents": previous_analysis.knowledge_intents,
                        "standalone_question": standalone_question,
                        "intent_source": "rules+clarification",
                    })
                    effective_request = request.copy(update={"message": standalone_question})
                self.session_store.clear_pending_clarification(session_context)

            if is_current_location_question(request.message) and not (
                context.location_node_id or request.current_node_id
            ):
                answer = "我还没有收到你的实时位置。请允许浏览器获取定位后，再问我“我现在在哪里”。"
                return QAResponse(
                    run_id=trace.run_id,
                    status="COMPLETED",
                    answer=answer,
                    total_duration_ms=_total_duration(trace),
                    domain=context,
                    analysis=analysis,
                    trace=trace.events,
                    warnings=["未收到可用的实时位置"],
                )

            if self.use_reasoning_core:
                reasoning_message = pending_question or effective_request.message
                logical_query = build_logical_query(
                    reasoning_message,
                    context,
                    session_context,
                    effective_request,
                    target_node_id=reasoning_target_node_id,
                    target_name=reasoning_target_name,
                )
                self.reasoning_executor.entity_linker = self.entity_linker
                self.reasoning_executor.retriever = self.retriever

                def on_reasoning_phase(
                    stage: str,
                    status: str,
                    duration_ms: float | None,
                    summary: str,
                    metadata: dict[str, Any],
                ) -> None:
                    event = TraceEvent(
                        stage=stage,
                        status=status,
                        duration_ms=duration_ms,
                        summary=summary,
                        metadata=dict(metadata or {}),
                    )
                    trace.events.append(event)
                    if on_trace:
                        on_trace(event)

                reasoning_result = self.reasoning_executor.run(
                    logical_query,
                    effective_request,
                    context,
                    on_phase=on_reasoning_phase,
                )
                entities = reasoning_result.entities
                if reasoning_result.clarification:
                    for item in entities:
                        if item.status != "AMBIGUOUS":
                            continue
                        self.session_store.set_pending_clarification(session_context, {
                            "question": request.message,
                            "mention": item.mention,
                            "candidates": [_candidate_payload(candidate) for candidate in item.candidates],
                        })
                    location_hint = ""
                    if context.location_node_name:
                        location_hint = f"你当前定位在“{context.location_node_name}”附近。"
                    return QAResponse(
                        run_id=trace.run_id,
                        status="NEEDS_CLARIFICATION",
                        answer=location_hint + (reasoning_result.clarification or ""),
                        total_duration_ms=_total_duration(trace),
                        domain=context,
                        analysis=analysis,
                        entities=entities,
                        trace=trace.events,
                        warnings=reasoning_result.warnings or [],
                    )

                unresolved = [item for item in entities if item.status == "UNRESOLVED"]
                linked = [item for item in entities if item.status == "LINKED"]
                if unresolved and not linked:
                    names = "、".join(item.mention for item in unresolved)
                    return QAResponse(
                        run_id=trace.run_id,
                        status="COMPLETED",
                        answer=unresolved_position_answer(
                            item.mention for item in unresolved
                        ),
                        total_duration_ms=_total_duration(trace),
                        domain=context,
                        analysis=analysis,
                        entities=entities,
                        trace=trace.events,
                        warnings=reasoning_result.warnings or [f"未找到已发布位置：{names}"],
                    )

                plan = reasoning_result.legacy_plan or QueryPlan()
                raw = {
                    "graph_results": reasoning_result.graph_results,
                    "retrieval_results": reasoning_result.retrieval_results,
                    "diagnostics": reasoning_result.diagnostics,
                }
                evidence = reasoning_result.evidence
                warnings.extend(reasoning_result.warnings or [])
            else:
                from .schemas import LinkedEntity
                if pending_choice:
                    selected_candidate, _ = pending_choice
                    linked_entity = LinkedEntity(
                        mention=str((session_context.pending_clarification or {}).get("mention") or selected_candidate.get("name") or ""),
                        status="LINKED",
                        node_id=str(selected_candidate.get("node_id") or ""),
                        name=str(selected_candidate.get("name") or ""),
                        node_type=selected_candidate.get("node_type"),
                        candidates=[_candidate_from_payload(selected_candidate)],
                    )
                    entities = run_stage(
                        "ENTITY_LINKING",
                        lambda: [linked_entity],
                        "根据上一轮选择确定位置",
                    )
                else:
                    entities = run_stage(
                        "ENTITY_LINKING",
                        lambda: self.entity_linker.link(context, analysis.mentions),
                        "在已发布图谱中查找对应位置",
                    )
                ambiguous = [item for item in entities if item.status == "AMBIGUOUS"]
                if ambiguous:
                    options = []
                    for item in ambiguous:
                        labels = [
                            f"{index + 1}. {_candidate_label(candidate)}"
                            for index, candidate in enumerate(item.candidates)
                        ]
                        options.append(f"{item.mention}：" + "；".join(labels))
                        self.session_store.set_pending_clarification(session_context, {
                            "question": request.message,
                            "mention": item.mention,
                            "candidates": [_candidate_payload(candidate) for candidate in item.candidates],
                        })
                    names = "；".join(options)
                    location_hint = ""
                    if context.location_node_name:
                        location_hint = f"你当前定位在“{context.location_node_name}”附近。"
                    return QAResponse(
                        run_id=trace.run_id,
                        status="NEEDS_CLARIFICATION",
                        answer=f"{location_hint}请确认你指的是哪个位置或部门：{names}",
                        total_duration_ms=_total_duration(trace),
                        domain=context,
                        analysis=analysis,
                        entities=entities,
                        trace=trace.events,
                        warnings=["存在多个同名或相近的位置，已暂停检索"],
                    )

                unresolved = [item for item in entities if item.status == "UNRESOLVED"]
                linked = [item for item in entities if item.status == "LINKED"]
                if unresolved and not linked:
                    names = "、".join(item.mention for item in unresolved)
                    return QAResponse(
                        run_id=trace.run_id,
                        status="COMPLETED",
                        answer=unresolved_position_answer(
                            item.mention for item in unresolved
                        ),
                        total_duration_ms=_total_duration(trace),
                        domain=context,
                        analysis=analysis,
                        entities=entities,
                        trace=trace.events,
                        warnings=[f"未找到已发布位置：{names}"],
                    )

                plan = run_stage(
                    "QUERY_PLANNING", lambda: build_query_plan(
                        effective_request, analysis, entities, context=context
                    ),
                    "根据问题意图选择图谱节点、关系和文本证据查询",
                )
                raw = run_stage(
                    "RETRIEVAL", lambda: self.retriever.execute(context, effective_request, plan),
                    "查询当前领域的已发布图谱与领域资料",
                )
                trace.events[-1].metadata = {
                    "graph_result_count": len(raw.get("graph_results") or []),
                    "retrieval_result_count": len(raw.get("retrieval_results") or []),
                }
                if on_trace:
                    on_trace(trace.events[-1])
                evidence = run_stage(
                    "EVIDENCE_FUSION",
                    lambda: fuse_evidence(
                        context.graph_domain_id,
                        raw.get("graph_results") or [],
                        raw.get("retrieval_results") or [],
                        limit=request.max_evidence,
                    ),
                    "合并并保留可定位的图谱、文档和图片证据",
                )
                trace.events[-1].metadata = {"evidence_count": len(evidence)}
                if on_trace:
                    on_trace(trace.events[-1])
            def generate_grounded_answer() -> dict[str, Any]:
                kwargs = {
                    "question": effective_request.message,
                    "context": context,
                    "analysis": analysis,
                    "entities": entities,
                    "plan": plan,
                    "evidence": evidence,
                    "conversation_context": session_context.to_dict(),
                }
                if on_answer_delta:
                    stream_method = getattr(self.answer_generator, "generate_stream", None)
                    if stream_method:
                        return stream_method(on_delta=on_answer_delta, **kwargs)
                return self.answer_generator.generate(**kwargs)

            generated = run_stage(
                "ANSWER_GENERATION",
                generate_grounded_answer,
                "仅依据检索证据生成回答，不进行联网搜索",
            )
            answer, citations, validation_warnings = run_stage(
                "ANSWER_VALIDATION",
                lambda: validate_generated_answer(generated, evidence),
                "校验答案中的引用是否指向实际检索证据",
            )
            trace.events[-1].metadata = {"citation_count": len(citations)}
            if on_trace:
                on_trace(trace.events[-1])
            self.session_store.append(session_context, request.message, answer, entities=[item.name for item in entities if item.status == "LINKED" and item.name])
            warnings.extend(validation_warnings)
            warnings.extend(
                str(item.get("reason")) for item in raw.get("diagnostics") or [] if item.get("reason")
            )
            return QAResponse(
                run_id=trace.run_id,
                status="COMPLETED" if answer else "FAILED",
                answer=answer,
                total_duration_ms=_total_duration(trace),
                domain=context,
                analysis=analysis,
                entities=entities,
                plan=plan,
                graph={"results": raw.get("graph_results") or []},
                evidence=evidence,
                citations=citations,
                images=_select_answer_images(evidence, analysis, request.message),
                trace=trace.events,
                warnings=warnings,
            )
        except DomainAccessError as exc:
            return QAResponse(
                run_id=trace.run_id,
                status="FAILED",
                total_duration_ms=_total_duration(trace),
                trace=trace.events,
                warnings=[str(exc)],
            )
        except Exception as exc:
            warnings.append(str(exc))
            return QAResponse(
                run_id=trace.run_id,
                status="FAILED",
                total_duration_ms=_total_duration(trace),
                trace=trace.events,
                warnings=warnings,
            )

