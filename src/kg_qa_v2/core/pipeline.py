"""QA v2 独立主链：Context → Understanding → Alignment → Plan → Retrieval。"""

from __future__ import annotations

from typing import Any, Callable, Optional
import time

from pydantic import BaseModel, Field

from ..models import ContextSnapshot, EntityReference, ExecutionPlan, QuerySpec, UnderstandingResult
from .entity import AlignmentDecision, EntityAlignmentService, EntityCandidate, PublishedCandidateSearch
from .entity_reference import build_entity_references, is_image_deictic_reference
from .memory import ContextResolver, MemoryContext
from .plan_compiler import PlanCompiler
from .retrieval import PlanExecutor, RetrievalResult
from .understanding import QuestionUnderstandingService


class PipelineResult(BaseModel):
    status: str
    memory: MemoryContext
    context: ContextSnapshot
    understanding: UnderstandingResult
    alignments: list[AlignmentDecision] = Field(default_factory=list)
    related_entities: list[EntityCandidate] = Field(default_factory=list)
    plan: ExecutionPlan
    retrieval: RetrievalResult
    warnings: list[str] = Field(default_factory=list)
    timings: dict[str, int] = Field(default_factory=dict)


class QAV2Pipeline:
    """可注入、可测试的新问答主链，不修改现有 QAService。"""

    def __init__(
        self,
        *,
        context_resolver: Optional[ContextResolver] = None,
        understanding: Optional[QuestionUnderstandingService] = None,
        candidate_search: Optional[PublishedCandidateSearch] = None,
        alignment: Optional[EntityAlignmentService] = None,
        plan_compiler: Optional[PlanCompiler] = None,
        executor: Optional[PlanExecutor] = None,
    ) -> None:
        self.context_resolver = context_resolver or ContextResolver()
        self.understanding = understanding or QuestionUnderstandingService()
        self.candidate_search = candidate_search
        self.alignment = alignment or EntityAlignmentService()
        self.plan_compiler = plan_compiler or PlanCompiler()
        self.executor = executor or PlanExecutor()

    def run(
        self,
        question: str,
        *,
        domain_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        current_location: Optional[dict[str, Any]] = None,
        has_image: bool = False,
        target_modality: Optional[str] = None,
        image_path: Optional[str] = None,
        max_evidence: int = 8,
        clarification_selection: Optional[dict[str, Any]] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        pipeline_started = time.perf_counter()
        timings: dict[str, int] = {}
        context_started = time.perf_counter()
        _emit(on_event, "context", "started", "正在读取会话上下文")
        memory = self.context_resolver.resolve(
            question=question,
            domain_id=domain_id,
            session_id=session_id,
            user_id=user_id,
            current_location=current_location,
            has_image=has_image,
            need_long_term=bool(user_id and _needs_long_term_context(question)),
            long_term_query=question,
        )
        context = self.context_resolver.to_snapshot(memory, session_id=session_id, user_id=user_id)
        timings["context_ms"] = int((time.perf_counter() - context_started) * 1000)
        _emit(on_event, "context", "completed", "已读取当前会话和领域上下文")
        _emit(on_event, "understanding", "started", "正在理解问题和判断回答方式")
        understanding = self.understanding.understand(
            question,
            context=context,
            has_image=has_image,
            target_modality=target_modality,
            timings=timings,
        )
        spec = understanding.query_spec
        _emit(on_event, "understanding", "completed", "已完成问题理解与模态判断")
        if image_path:
            # “这张/这张图片”是输入模态指代，不是图谱中的文本节点。
            # 清除它，避免图片问题先进入实体澄清并错误结束主链。
            if spec.intent == "IMAGE":
                image_refs = [
                    ref for ref in spec.entity_refs
                    if not is_image_deictic_reference(ref)
                ]
                if len(image_refs) != len(spec.entity_refs):
                    spec = spec.model_copy(update={
                        "entity_refs": image_refs,
                        "targets": [
                            target for target in spec.targets
                            if target and not target.strip() in {"这张", "这张图片", "这张照片", "图片", "照片", "该图", "这幅图", "此图"}
                        ],
                    })
            spec = spec.model_copy(update={
                "constraints": {
                    **spec.constraints,
                    "image_path": str(image_path),
                }
            })
            understanding = understanding.model_copy(update={"query_spec": spec})
        if spec.route == "CASUAL":
            plan = self.plan_compiler.compile(spec, max_evidence=max_evidence)
            _emit(on_event, "planning", "completed", "已确认这是聊天消息，不进行资料查询")
            timings["plan_ms"] = 0
            timings["pipeline_ms"] = int((time.perf_counter() - pipeline_started) * 1000)
            return PipelineResult(
                status="COMPLETED",
                memory=memory,
                context=context,
                understanding=understanding,
                plan=plan,
                retrieval=RetrievalResult(),
                timings=timings,
            )

        alignments: list[AlignmentDecision] = []
        related_entities: list[EntityCandidate] = []
        alignment_started = time.perf_counter()
        if clarification_selection and clarification_selection.get("candidate"):
            selected = EntityCandidate.model_validate(clarification_selection["candidate"])
            selected_ref = EntityReference(
                surface=selected.name,
                core_mention=selected.name,
                reference_type="EXPLICIT",
                resolved_node_id=selected.node_id,
            )
            spec = spec.model_copy(update={
                "targets": [selected.name],
                "entity_refs": [selected_ref],
                "needs_clarification": False,
                "constraints": {**spec.constraints, "resolved_node_ids": [selected.node_id]},
            })
            understanding = understanding.model_copy(update={"query_spec": spec})
            alignments.append(AlignmentDecision(
                mention=clarification_selection.get("mention") or selected.name,
                status="LINKED",
                node_id=selected.node_id,
                name=selected.name,
                confidence=1.0,
                reason="根据上一轮候选确认继续查询",
                candidates=[selected],
            ))
        else:
            references = spec.entity_refs or build_entity_references(
                question,
                spec.targets,
                intent=spec.intent,
                context=context,
                needs_current_location=spec.needs_current_location,
            )
            searchable_refs = [
                item for item in references
                if item.reference_type not in {"CURRENT_LOCATION", "UNKNOWN"}
                and not item.resolved_node_id
            ]
            direct_refs = [
                item for item in references
                if item.resolved_node_id and item.reference_type == "CONTEXTUAL"
            ]
            for reference in direct_refs:
                alignments.append(AlignmentDecision(
                    mention=reference.core_mention or reference.surface,
                    status="LINKED",
                    node_id=reference.resolved_node_id,
                    name=reference.core_mention or reference.surface,
                    confidence=1.0,
                    reason="沿用会话中已确认的当前对象",
                ))
            if searchable_refs and self.candidate_search is not None:
                _emit(on_event, "entity_search", "started", "正在当前领域查找相关位置")
                for reference in searchable_refs:
                    mention = reference.core_mention or reference.surface
                    search_result = _search_candidates(self.candidate_search, mention, context=context, reference=reference)
                    for related in search_result.related_entities:
                        if related.node_id not in {item.node_id for item in related_entities}:
                            related_entities.append(related)
                    decision = self.alignment.align(
                        mention,
                        search_result.candidates,
                        context=context,
                        question=spec.question_rewrite or question,
                    )
                    alignments.append(decision)
                _emit(on_event, "entity_search", "completed", "已完成位置候选查找与对齐")
                unresolved = [item for item in alignments if item.status != "LINKED"]
                if unresolved:
                    clarification_spec = spec.model_copy(update={"needs_clarification": True})
                    plan = self.plan_compiler.compile(clarification_spec, max_evidence=max_evidence)
                    return PipelineResult(
                        status="NEEDS_CLARIFICATION",
                        memory=memory,
                        context=context,
                        understanding=understanding,
                        alignments=alignments,
                        related_entities=related_entities,
                        plan=plan,
                        retrieval=RetrievalResult(),
                        warnings=[item.reason for item in unresolved],
                        timings=timings,
                    )
        timings["alignment_ms"] = int((time.perf_counter() - alignment_started) * 1000)

        linked_ids = [
            item.node_id
            for item in alignments
            if item.status == "LINKED" and item.node_id
        ]
        if linked_ids:
            spec = spec.model_copy(update={
                "constraints": {
                    **spec.constraints,
                    "resolved_node_ids": linked_ids,
                    "resolved_candidates": [
                        item.candidates[0].model_dump()
                        for item in alignments
                        if item.status == "LINKED" and item.candidates
                    ],
                }
            })
            understanding = understanding.model_copy(update={"query_spec": spec})

        plan_started = time.perf_counter()
        plan = self.plan_compiler.compile(spec, max_evidence=max_evidence)
        timings["plan_ms"] = int((time.perf_counter() - plan_started) * 1000)
        _emit(on_event, "planning", "completed", "已生成本轮回答计划")
        retrieval_started = time.perf_counter()
        _emit(on_event, "retrieval", "started", "正在查询已发布图谱和领域资料")
        retrieval = self.executor.execute(
            plan,
            spec=spec,
            context=context,
            question=spec.question_rewrite or question,
        )
        timings["retrieval_ms"] = int((time.perf_counter() - retrieval_started) * 1000)
        timings["pipeline_ms"] = int((time.perf_counter() - pipeline_started) * 1000)
        _emit(on_event, "retrieval", "completed", "已完成图谱与资料查询")
        return PipelineResult(
            status="COMPLETED",
            memory=memory,
            context=context,
            understanding=understanding,
            alignments=alignments,
            related_entities=related_entities,
            plan=plan,
            retrieval=retrieval,
            warnings=retrieval.warnings,
            timings=timings,
        )


def _needs_long_term_context(question: str) -> bool:
    text = str(question or "")
    return any(marker in text for marker in ("我之前", "我刚才", "我的偏好", "我喜欢", "我习惯", "记住我", "我的资料", "我的信息", "长期记忆", "我常问", "我关注过"))

def _emit(
    callback: Optional[Callable[[dict[str, Any]], None]],
    stage: str,
    status: str,
    message: str,
) -> None:
    if callback is not None:
        callback({
            "stage": stage,
            "status": status,
            "message": message,
        })


def _search_candidates(
    searcher: PublishedCandidateSearch,
    mention: str,
    *,
    context: ContextSnapshot,
    reference: EntityReference,
):
    try:
        return searcher.search(mention, context=context, reference=reference)
    except TypeError as exc:
        # 兼容旧的注入测试替身；生产适配器走带 reference 的新接口。
        if "reference" not in str(exc):
            raise
        return searcher.search(mention, context=context)
