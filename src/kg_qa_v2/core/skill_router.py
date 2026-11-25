"""QA v2 的确定性 Skill/Tool 契约与调度。

Task Skill 负责用户任务边界和执行结果；Graph/Text/Image Tool 只负责访问
对应数据源。两层都不重新理解问题、不解析实体、不生成自由 Cypher 或答案。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models import ContextSnapshot, ExecutionPlan, PlanStep, QuerySpec
from .skill_registry import SkillRegistry


class SkillContext(BaseModel):
    """Skill 执行时可读取的统一上下文。"""

    domain_id: str = Field(..., min_length=1)
    question: str = ""
    query_spec: QuerySpec
    context: ContextSnapshot
    step: PlanStep
    published_only: bool = True


class ToolResult(BaseModel):
    """数据源 Tool 的统一输出；Tool 不理解用户任务，也不生成答案。"""

    tool_name: str
    step_type: str
    ok: bool = True
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SkillResult(BaseModel):
    """Task Skill 的统一输出；Skill 负责任务边界，Tool 负责数据访问。"""

    skill_name: str
    task_skill_name: str = ""
    step_type: str
    tool_name: str = ""
    ok: bool = True
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalTool:
    """把既有检索端口包装为数据源 Tool。"""

    def __init__(self, *, name: str, step_type: str, port: Optional[Callable[..., Any]]) -> None:
        self.name = name
        self.step_type = step_type
        self.port = port

    def execute(self, context: SkillContext) -> ToolResult:
        if self.port is None:
            raise RuntimeError(f"未配置 {self.step_type} 检索端口")
        raw = self.port(
            domain_id=context.domain_id,
            question=context.question,
            query_spec=context.query_spec,
            step=context.step,
            context=context.context,
            published_only=context.published_only,
        )
        values = raw.get("evidence", raw) if isinstance(raw, dict) else raw
        evidence = [_dump_evidence(item) for item in (values or [])]
        warnings = list(raw.get("warnings") or []) if isinstance(raw, dict) else []
        return ToolResult(
            tool_name=self.name,
            step_type=self.step_type,
            evidence=evidence,
            warnings=[str(item) for item in warnings],
            metadata={"published_only": context.published_only},
        )


class GraphTool(RetrievalTool):
    def __init__(self, port: Optional[Callable[..., Any]]) -> None:
        super().__init__(name="GraphTool", step_type="GRAPH", port=port)


class FactTool(RetrievalTool):
    def __init__(self, port: Optional[Callable[..., Any]]) -> None:
        super().__init__(name="FactTool", step_type="FACT", port=port)


class TextTool(RetrievalTool):
    def __init__(self, port: Optional[Callable[..., Any]]) -> None:
        super().__init__(name="TextTool", step_type="TEXT", port=port)


class ImageTool(RetrievalTool):
    def __init__(self, port: Optional[Callable[..., Any]]) -> None:
        super().__init__(name="ImageTool", step_type="IMAGE", port=port)


class TaskSkill:
    """任务层适配器：不直接访问数据库，只把任务请求交给 Tool。"""

    def __init__(self, *, name: str, step_type: str, tool: RetrievalTool) -> None:
        self.name = name
        self.step_type = step_type
        self.tool = tool

    def execute(self, context: SkillContext) -> SkillResult:
        result = self.tool.execute(context)
        return SkillResult(
            skill_name=self.name,
            task_skill_name=self.name,
            step_type=self.step_type,
            tool_name=result.tool_name,
            ok=result.ok,
            evidence=result.evidence,
            warnings=result.warnings,
            metadata={"tool": result.tool_name, **result.metadata},
        )


class RegisteredTaskSkill(TaskSkill):
    """可注册的任务 Skill：根据计划步骤选择对应数据源 Tool。"""

    def __init__(
        self,
        *,
        name: str,
        step_types: tuple[str, ...],
        tools: Dict[str, RetrievalTool],
    ) -> None:
        if not step_types:
            raise ValueError(f"{name} 至少需要一个执行步骤类型")
        self.step_types = frozenset(item.upper() for item in step_types)
        self.tools = dict(tools)
        primary = self.tools.get(step_types[0].upper())
        if primary is None:
            raise ValueError(f"{name} 缺少主 Tool：{step_types[0]}")
        super().__init__(name=name, step_type=step_types[0], tool=primary)

    def execute(self, context: SkillContext) -> SkillResult:
        tool = self.tools.get(context.step.type.upper())
        if tool is None:
            raise RuntimeError(f"{self.name} 不支持执行步骤：{context.step.type}")
        result = tool.execute(context)
        return SkillResult(
            skill_name=self.name,
            task_skill_name=self.name,
            step_type=context.step.type,
            tool_name=result.tool_name,
            ok=result.ok,
            evidence=result.evidence,
            warnings=result.warnings,
            metadata={"task_skill": self.name, "tool": result.tool_name, **result.metadata},
        )


class RetrievalSkill(TaskSkill):
    """兼容旧调用的任务 Skill；实际数据访问已下沉到 Graph/Text/Image Tool。"""

    def __init__(
        self,
        *,
        name: str,
        step_type: str,
        port: Optional[Callable[..., Any]] = None,
        tool: Optional[RetrievalTool] = None,
    ) -> None:
        super().__init__(
            name=name,
            step_type=step_type,
            tool=tool or RetrievalTool(
                name=f"{step_type.title()}Tool",
                step_type=step_type,
                port=port,
            ),
        )


class GraphSkill(RetrievalSkill):
    def __init__(self, port: Optional[Callable[..., Any]]) -> None:
        super().__init__(name="GraphSkill", step_type="GRAPH", tool=GraphTool(port))


class FactSkill(RetrievalSkill):
    """事实任务 Skill；保留旧的 ``FactSkill(port)`` 构造方式。"""

    def __init__(
        self,
        port: Optional[Callable[..., Any]] = None,
        *,
        tool: Optional[RetrievalTool] = None,
    ) -> None:
        super().__init__(name="FactSkill", step_type="FACT", tool=tool or FactTool(port))


class TextSkill(RetrievalSkill):
    def __init__(self, port: Optional[Callable[..., Any]]) -> None:
        super().__init__(name="TextSkill", step_type="TEXT", tool=TextTool(port))


class ImageSkill(RetrievalSkill):
    def __init__(self, port: Optional[Callable[..., Any]]) -> None:
        super().__init__(name="ImageSkill", step_type="IMAGE", tool=ImageTool(port))


class LocationSkill(RegisteredTaskSkill):
    def __init__(self, tools: Dict[str, RetrievalTool]) -> None:
        super().__init__(name="LocationSkill", step_types=("GRAPH", "FACT"), tools=tools)


class HierarchySkill(RegisteredTaskSkill):
    def __init__(self, tools: Dict[str, RetrievalTool]) -> None:
        super().__init__(name="HierarchySkill", step_types=("GRAPH",), tools=tools)


class PathSkill(RegisteredTaskSkill):
    def __init__(self, tools: Dict[str, RetrievalTool]) -> None:
        super().__init__(name="PathSkill", step_types=("GRAPH",), tools=tools)


class CompareSkill(RegisteredTaskSkill):
    def __init__(self, tools: Dict[str, RetrievalTool]) -> None:
        super().__init__(name="CompareSkill", step_types=("GRAPH", "TEXT", "IMAGE"), tools=tools)


class ExplainSkill(RegisteredTaskSkill):
    def __init__(self, tools: Dict[str, RetrievalTool]) -> None:
        super().__init__(name="ExplainSkill", step_types=("GRAPH", "TEXT"), tools=tools)


class ImageRecognitionSkill(RegisteredTaskSkill):
    def __init__(self, tools: Dict[str, RetrievalTool]) -> None:
        super().__init__(name="ImageRecognitionSkill", step_types=("IMAGE", "GRAPH"), tools=tools)


class VisualQuerySkill(RegisteredTaskSkill):
    """实体到图片的视觉查询：读取节点详情并返回已发布关联图片。"""

    def __init__(self, tools: Dict[str, RetrievalTool]) -> None:
        super().__init__(name="VisualQuerySkill", step_types=("IMAGE", "GRAPH"), tools=tools)


class NearbySkill(RegisteredTaskSkill):
    """实体到空间邻近节点的查询：由 GraphTool 执行坐标排序。"""

    def __init__(self, tools: Dict[str, RetrievalTool]) -> None:
        super().__init__(name="NearbySkill", step_types=("GRAPH",), tools=tools)


class CasualSkill:
    """闲聊任务标记；闲聊不进入检索 Tool。"""

    name = "CasualSkill"
    step_type = "CASUAL"
    tool = None

    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(
            skill_name=self.name,
            step_type=self.step_type,
            metadata={"retrieval_skipped": True},
        )


class SkillRouter:
    """根据固定 ExecutionPlan 选择 Task Skill，并按计划要求并行执行。"""

    def __init__(
        self,
        *,
        graph_retriever: Optional[Callable[..., Any]] = None,
        fact_retriever: Optional[Callable[..., Any]] = None,
        text_retriever: Optional[Callable[..., Any]] = None,
        image_retriever: Optional[Callable[..., Any]] = None,
        max_workers: int = 3,
    ) -> None:
        self.max_workers = max(1, min(int(max_workers), 3))
        self.tools = {
            "GRAPH": GraphTool(graph_retriever),
            "FACT": FactTool(fact_retriever),
            "TEXT": TextTool(text_retriever),
            "IMAGE": ImageTool(image_retriever),
        }
        self.skills = {
            "GRAPH": TaskSkill(name="GraphSkill", step_type="GRAPH", tool=self.tools["GRAPH"]),
            "FACT": TaskSkill(name="FactSkill", step_type="FACT", tool=self.tools["FACT"]),
            "TEXT": TaskSkill(name="TextSkill", step_type="TEXT", tool=self.tools["TEXT"]),
            "IMAGE": TaskSkill(name="ImageSkill", step_type="IMAGE", tool=self.tools["IMAGE"]),
        }
        self.task_skills = {
            "LocationSkill": LocationSkill(self.tools),
            "FactSkill": FactSkill(tool=self.tools["FACT"]),
            "HierarchySkill": HierarchySkill(self.tools),
            "PathSkill": PathSkill(self.tools),
            "CompareSkill": CompareSkill(self.tools),
            "ExplainSkill": ExplainSkill(self.tools),
            "ImageRecognitionSkill": ImageRecognitionSkill(self.tools),
            "VisualQuerySkill": VisualQuerySkill(self.tools),
            "NearbySkill": NearbySkill(self.tools),
            "CasualSkill": CasualSkill(),
        }
        self.skill_registry = SkillRegistry()
        self.skill_registry.register(self.task_skills["LocationSkill"], intents=("LOCATION",), step_types=("GRAPH", "FACT"))
        self.skill_registry.register(self.task_skills["FactSkill"], intents=("DIRECT_FACT", "LIST"), step_types=("FACT",))
        self.skill_registry.register(self.task_skills["HierarchySkill"], intents=("HIERARCHY",), step_types=("GRAPH",))
        self.skill_registry.register(self.task_skills["PathSkill"], intents=("PATH",), step_types=("GRAPH",))
        self.skill_registry.register(self.task_skills["CompareSkill"], intents=("COMPARE",), step_types=("GRAPH", "TEXT", "IMAGE"))
        self.skill_registry.register(self.task_skills["ExplainSkill"], intents=("EXPLAIN",), step_types=("GRAPH", "TEXT"))
        self.skill_registry.register(self.task_skills["ImageRecognitionSkill"], intents=("IMAGE",), step_types=("IMAGE", "GRAPH"), request_types=("IMAGE_IDENTIFICATION",))
        self.skill_registry.register(self.task_skills["VisualQuerySkill"], intents=("IMAGE",), step_types=("IMAGE", "GRAPH"), request_types=("VISUAL_APPEARANCE", "IMAGE_RETRIEVAL"))
        self.skill_registry.register(self.task_skills["NearbySkill"], intents=("LIST",), step_types=("GRAPH",), request_types=("NEARBY",))
        self.skill_registry.register(self.task_skills["CasualSkill"], intents=("CASUAL",), step_types=())

    def select(self, plan: ExecutionPlan, *, spec: Optional[QuerySpec] = None) -> List[Any]:
        """选择任务 Skill；不传 spec 时保留旧数据源 Skill 选择行为。"""
        if spec is not None:
            return self.skill_registry.select_for_plan(
                intent=spec.intent,
                steps=plan.steps,
                request_types=[item.request_type for item in spec.requests],
            )
        return [
            self.skills[step.type]
            for step in plan.steps
            if step.type in self.skills
        ]

    def execute(
        self,
        plan: ExecutionPlan,
        *,
        spec: QuerySpec,
        context: ContextSnapshot,
        question: str,
    ) -> List[SkillResult]:
        selected_steps = [step for step in plan.steps if step.type in self.skills]
        if not selected_steps:
            return []

        selected_skills = self.select(plan, spec=spec)
        if len(selected_skills) != len(selected_steps):
            selected_skills = [self.skills[step.type] for step in selected_steps]

        def run(pair: tuple[PlanStep, Any]) -> SkillResult:
            step, skill = pair
            legacy_skill = self.skills[step.type]
            skill_context = SkillContext(
                domain_id=context.domain_id,
                question=question,
                query_spec=spec,
                context=context,
                step=step,
                published_only=True,
            )
            try:
                result = skill.execute(skill_context)
                # 保留旧的 skill_name 给现有调用方；新任务名称通过
                # task_skill_name 暴露，避免 P3-C 破坏旧响应契约。
                return result.model_copy(update={
                    "skill_name": legacy_skill.name,
                    "task_skill_name": getattr(skill, "name", legacy_skill.name),
                })
            except Exception as exc:
                return SkillResult(
                    skill_name=legacy_skill.name,
                    task_skill_name=getattr(skill, "name", legacy_skill.name),
                    step_type=step.type,
                    tool_name=getattr(getattr(skill, "tool", None), "name", ""),
                    ok=False,
                    warnings=[f"{step.type} 检索失败：{exc}"],
                )

        if plan.parallel and len(selected_steps) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(selected_steps))) as pool:
                futures = [pool.submit(run, pair) for pair in zip(selected_steps, selected_skills)]
                return [future.result() for future in as_completed(futures)]
        return [run(pair) for pair in zip(selected_steps, selected_skills)]


def _dump_evidence(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    raise TypeError("检索 Tool 必须返回 Evidence 字典或 Pydantic 模型")
