"""Graph/Text/Image 检索执行、证据融合与 Claim/Evidence 绑定。

检索层只返回已发布 Evidence；答案层通过 ClaimEvidenceBinder 显式声明每条
事实使用的 evidence_id，引用编号由融合后的 Evidence 顺序稳定生成。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..models import ContextSnapshot, ExecutionPlan, QuerySpec
from .skill_router import SkillRouter


class Evidence(BaseModel):
    evidence_id: str = Field(..., min_length=1)
    source_type: str
    content: str = ""
    score: float = 0.0
    node_id: Optional[str] = None
    asset_id: Optional[str] = None
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaimResult(BaseModel):
    """答案中的一条可核验事实，不允许脱离 Evidence 独立绑定。"""

    claim_id: str = Field(..., min_length=1)
    claim: str = Field(..., min_length=1)
    claim_type: str = "FACT"
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceBinding(BaseModel):
    """Claim 与 Evidence 的唯一引用绑定契约。"""

    claim_id: str = Field(..., min_length=1)
    evidence_id: str = Field(..., min_length=1)
    display_index: int = Field(..., ge=1)
    source_type: str
    role: Literal["SUPPORT", "QUALIFIER", "CONFLICT"] = "SUPPORT"


class ClaimEvidenceResult(BaseModel):
    claims: List[ClaimResult] = Field(default_factory=list)
    bindings: List[EvidenceBinding] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def bind_claims_to_evidence(
    claims: List[ClaimResult],
    evidence: List[Evidence],
    *,
    strict: bool = True,
) -> ClaimEvidenceResult:
    """把模型/代码产出的 claim 绑定到实际 Evidence，并分配稳定显示编号。

    输入：ClaimResult 列表和融合后的 Evidence 列表。输出：只引用存在于
    EvidenceSet 的绑定；未知 evidence_id 在 strict 模式直接失败，非 strict
    模式会被丢弃并写入 warning，避免前端显示失效引用。
    """
    evidence_by_id = {item.evidence_id: item for item in evidence}
    display_index = {
        item.evidence_id: index
        for index, item in enumerate(evidence, start=1)
    }
    warnings: list[str] = []
    bound_claims: list[ClaimResult] = []
    bindings: list[EvidenceBinding] = []

    for claim in claims:
        valid_ids: list[str] = []
        seen: set[str] = set()
        for evidence_id in claim.evidence_ids:
            evidence_id = str(evidence_id).strip()
            if not evidence_id or evidence_id in seen:
                continue
            seen.add(evidence_id)
            item = evidence_by_id.get(evidence_id)
            if item is None:
                message = f"claim {claim.claim_id} 引用了不存在的 evidence_id: {evidence_id}"
                if strict:
                    raise ValueError(message)
                warnings.append(message)
                continue
            valid_ids.append(evidence_id)
            bindings.append(
                EvidenceBinding(
                    claim_id=claim.claim_id,
                    evidence_id=evidence_id,
                    display_index=display_index[evidence_id],
                    source_type=item.source_type,
                )
            )
        if not valid_ids:
            warnings.append(f"claim {claim.claim_id} 没有可用的 Evidence 绑定")
        bound_claims.append(claim.model_copy(update={"evidence_ids": valid_ids}))

    return ClaimEvidenceResult(
        claims=bound_claims,
        bindings=bindings,
        warnings=warnings,
    )


class RetrievalResult(BaseModel):
    evidence: List[Evidence] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    executed_steps: List[str] = Field(default_factory=list)

    def bind_claims(
        self,
        claims: List[ClaimResult],
        *,
        strict: bool = True,
    ) -> ClaimEvidenceResult:
        """在当前融合结果上执行统一 Claim/Evidence 绑定。"""
        return bind_claims_to_evidence(claims, self.evidence, strict=strict)


class EvidenceFusion:
    """对来自不同检索器的证据做稳定去重和排序。"""

    def fuse(self, values: List[Evidence], *, max_items: int = 8) -> List[Evidence]:
        by_id: Dict[str, Evidence] = {}
        for item in values:
            previous = by_id.get(item.evidence_id)
            if previous is None or item.score > previous.score:
                by_id[item.evidence_id] = item
        priority = {"graph": 3, "text": 2, "image": 1}
        ordered = sorted(
            by_id.values(),
            key=lambda item: (
                -float(item.score),
                -priority.get(item.source_type, 0),
                item.evidence_id,
            ),
        )
        return ordered[: max(1, min(int(max_items), 20))]


class PlanExecutor:
    """执行固定计划；端口由生产环境注入现有 Graph/Text/Image 服务。"""

    def __init__(
        self,
        *,
        graph_retriever: Optional[Callable[..., Any]] = None,
        fact_retriever: Optional[Callable[..., Any]] = None,
        text_retriever: Optional[Callable[..., Any]] = None,
        image_retriever: Optional[Callable[..., Any]] = None,
        max_workers: int = 3,
    ) -> None:
        self.skill_router = SkillRouter(
            graph_retriever=graph_retriever,
            fact_retriever=fact_retriever,
            text_retriever=text_retriever,
            image_retriever=image_retriever,
            max_workers=max_workers,
        )
        self.fusion = EvidenceFusion()

    def execute(
        self,
        plan: ExecutionPlan,
        *,
        spec: QuerySpec,
        context: ContextSnapshot,
        question: str,
    ) -> RetrievalResult:
        if any(step.type == "CLARIFICATION" for step in plan.steps):
            return RetrievalResult(warnings=["计划要求先完成澄清，未执行检索"])
        results = self.skill_router.execute(
            plan,
            spec=spec,
            context=context,
            question=question,
        )
        if not results:
            return RetrievalResult(warnings=["计划中没有可执行的 Graph/Fact/Text/Image Skill"])
        evidence: List[Evidence] = []
        warnings: List[str] = []
        executed: List[str] = []
        for result in results:
            warnings.extend(result.warnings)
            if not result.ok:
                continue
            evidence.extend(Evidence.model_validate(item) for item in result.evidence)
            executed.append(result.step_type)
        return RetrievalResult(
            evidence=self.fusion.fuse(evidence, max_items=plan.max_evidence),
            warnings=warnings,
            executed_steps=executed,
        )
