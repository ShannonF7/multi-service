"""问答 v2 的稳定数据契约。

本模块只描述跨步骤传递的数据，不执行检索，也不访问数据库。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .core.prompts import QUESTION_UNDERSTANDING_PROMPT_VERSION, ROUTER_PROMPT_VERSION


Intent = Literal[
    "CASUAL",
    "LOCATION",
    "DIRECT_FACT",
    "HIERARCHY",
    "PATH",
    "IMAGE",
    "COMPARE",
    "EXPLAIN",
    "LIST",
    "UNKNOWN",
]

RouteName = Literal["CASUAL", "KNOWLEDGE", "CLARIFICATION"]
ContextDependency = Literal["NONE", "SESSION", "LOCATION", "PENDING_CLARIFICATION", "IMAGE"]
PlanStepType = Literal["GRAPH", "FACT", "TEXT", "IMAGE", "CLARIFICATION"]
RequestType = Literal[
    "ATTRIBUTE",
    "LOCATION",
    "NEARBY",
    "HIERARCHY",
    "PATH",
    "IMAGE",
    "IMAGE_RETRIEVAL",
    "IMAGE_IDENTIFICATION",
    "VISUAL_APPEARANCE",
    "TEXT",
    "COMPARE",
]


class ContextSnapshot(BaseModel):
    """问题理解模块使用的最小上下文数据契约。

    输入：会话、位置和长期记忆解析后的上下文信息。
    输出：完成字段校验、供问题理解模块读取的上下文对象。
    逻辑：只定义跨步骤传递的字段，不在模型中读取会话或数据库。
    """

    domain_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    current_node_id: Optional[str] = None
    current_node_name: Optional[str] = None
    location_node_id: Optional[str] = None
    location_node_name: Optional[str] = None
    location_in_domain: Optional[bool] = None
    location_domain_name: Optional[str] = None
    distance_to_domain_m: Optional[float] = None
    domain_scope_radius_m: Optional[float] = None
    last_question: Optional[str] = None
    last_entities: List[str] = Field(default_factory=list)
    pending_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    pending_clarification: Dict[str, Any] = Field(default_factory=dict)
    recent_messages: List[Dict[str, Any]] = Field(default_factory=list)
    session_summary: Optional[str] = None
    long_term_memory: List[Dict[str, Any]] = Field(default_factory=list)


class RouteDecision(BaseModel):
    """问题路由器的决策结果。

    输入：路由判断、意图、置信度及兼容字段。
    输出：通过类型和取值范围校验的路由决策对象。
    逻辑：描述请求交由哪条处理路径；具体实体由后续步骤抽取。
    """

    route: RouteName
    intent: Intent = "UNKNOWN"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # 以下字段保留用于兼容旧接口；真实执行需求由程序编译。
    needs_graph: bool = False
    needs_text: bool = False
    needs_image: bool = False
    needs_path: bool = False
    context_dependent: bool = False
    context_dependency: ContextDependency = "NONE"
    needs_current_location: bool = False
    needs_clarification: bool = False
    source: Literal["jev", "local", "fallback"] = "fallback"


class EntityQualifier(BaseModel):
    type: str
    text: str = Field(..., min_length=1)


class EntityReference(BaseModel):
    surface: str = Field(..., min_length=1)
    core_mention: str = ""
    qualifiers: List[EntityQualifier] = Field(default_factory=list)
    reference_type: Literal["EXPLICIT", "CONTEXTUAL", "CURRENT_LOCATION", "UNKNOWN"] = "EXPLICIT"
    resolved_node_id: Optional[str] = None


class QueryRequest(BaseModel):
    """从用户问题中抽取的一项可独立执行的查询。

    输入：查询类型、目标、属性、范围和过滤条件。
    输出：通过 Pydantic 校验的结构化子查询。
    逻辑：把复合问题拆成可执行请求，并限制查询半径等字段的取值。
    """

    request_type: RequestType = "ATTRIBUTE"
    attribute: Optional[str] = None
    surface: str = ""
    target: Optional[str] = None
    radius: Optional[float] = Field(default=None, ge=0.0)
    category: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)


class ResponseRequirements(BaseModel):
    """当前答案需要返回的内容模态。

    输入：文本、图片和地图结果的布尔需求。
    输出：本轮答案的模态要求对象。
    逻辑：只声明期望的答案形式，不决定检索权限或实体身份。
    """

    text: bool = True
    images: bool = False
    map: bool = False


class QuerySpec(BaseModel):
    """一次问题理解步骤的标准输出。

    输入：意图、目标实体、子查询、约束和答案形式等理解结果。
    输出：供计划编译器及执行器消费的结构化对象。
    逻辑：集中承载问题理解结果，后续步骤不再依赖分散的临时字段。
    """

    intent: Intent = "UNKNOWN"
    route: RouteName = "KNOWLEDGE"
    targets: List[str] = Field(default_factory=list)
    entity_refs: List[EntityReference] = Field(default_factory=list)
    requests: List[QueryRequest] = Field(default_factory=list)
    attribute: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    time_range: Optional[str] = None
    comparison_targets: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    response_requirements: ResponseRequirements = Field(default_factory=ResponseRequirements)
    scope: Literal["CURRENT_DOMAIN", "CURRENT_SUBGRAPH", "CURRENT_NODE"] = "CURRENT_DOMAIN"
    needs_graph: bool = False
    needs_text: bool = False
    needs_image: bool = False
    needs_path: bool = False
    context_dependent: bool = False
    context_dependency: ContextDependency = "NONE"
    needs_current_location: bool = False
    needs_clarification: bool = False
    question_rewrite: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Literal["jev+qwen", "fallback"] = "fallback"

    @field_validator("targets", "comparison_targets", mode="before")
    @classmethod
    def normalize_names(cls, value: Any) -> List[str]:
        """清理目标名称列表并按首次出现顺序去重。

        输入：目标字段原始值，支持 None、单个字符串或可迭代值。
        输出：去除空白、空项和重复项后的字符串列表。
        逻辑：None 转为空列表，字符串转为单元素列表，再清理并保序去重。
        """
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


class PlanStep(BaseModel):
    """执行计划中的一个检索或澄清步骤。

    输入：步骤类型、执行原因、目标及参数。
    输出：供计划执行器消费的单步定义。
    逻辑：用固定步骤类型和结构化参数描述执行动作。
    """

    type: PlanStepType
    reason: str
    parallel_group: Optional[str] = None
    target: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """计划编译器生成的确定性执行计划。

    输入：步骤顺序、并行设置和检索上限。
    输出：供计划执行器按约束执行的计划对象。
    逻辑：通过字段约束限制跳数和证据数量，避免计划超过服务上限。
    """

    steps: List[PlanStep] = Field(default_factory=list)
    parallel: bool = False
    max_hops: int = Field(default=1, ge=0, le=4)
    max_evidence: int = Field(default=8, ge=1, le=20)


class UnderstandingResult(BaseModel):
    """问题理解阶段返回的完整结果。

    输入：路由决策、结构化查询、提示词版本和运行元数据。
    输出：供计划编译与运行诊断使用的问题理解对象。
    逻辑：将理解结果及其来源信息集中在一个 Pydantic 模型中。
    """

    route: RouteDecision
    query_spec: QuerySpec
    router_prompt_version: str = ROUTER_PROMPT_VERSION
    slot_prompt_version: str = QUESTION_UNDERSTANDING_PROMPT_VERSION
    runtime_meta: Dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "Intent",
    "RouteName",
    "ContextDependency",
    "PlanStepType",
    "RequestType",
    "ContextSnapshot",
    "RouteDecision",
    "EntityQualifier",
    "EntityReference",
    "QueryRequest",
    "ResponseRequirements",
    "QuerySpec",
    "PlanStep",
    "ExecutionPlan",
    "UnderstandingResult",
]
