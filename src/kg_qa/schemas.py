"""KG-QA 的输入、计划、证据、答案和轨迹契约。"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.kg_qa_shared.schemas import QARequest, UserLocation


class DomainContext(BaseModel):
    domain_id: str
    source_scenic_id: str
    graph_domain_id: str
    name: Optional[str] = None
    scope: Literal["NODE", "SUBGRAPH", "DOMAIN"] = "DOMAIN"
    current_node_id: Optional[str] = None
    location_node_id: Optional[str] = None
    location_node_name: Optional[str] = None
    location_distance_m: Optional[float] = None


class InputAnalysis(BaseModel):
    query_modality: Literal["TEXT", "IMAGE", "MULTIMODAL"]
    target_modality: Literal["TEXT", "IMAGE"]
    modality_intent: str
    knowledge_intents: List[str] = Field(default_factory=list)
    mentions: List[str] = Field(default_factory=list)
    has_text: bool = False
    has_image: bool = False
    standalone_question: Optional[str] = None
    intent_confidence: float = 1.0
    intent_source: str = "rules"


class EntityCandidate(BaseModel):
    node_id: str
    name: str
    node_type: Optional[str] = None
    match_type: Optional[str] = None
    score: Optional[float] = None
    parent_name: Optional[str] = None
    grandparent_name: Optional[str] = None
    hierarchy_path: List[str] = Field(default_factory=list)
    distance_m: Optional[float] = None


class LinkedEntity(BaseModel):
    mention: str
    status: Literal["LINKED", "AMBIGUOUS", "UNRESOLVED"]
    node_id: Optional[str] = None
    name: Optional[str] = None
    node_type: Optional[str] = None
    candidates: List[EntityCandidate] = Field(default_factory=list)


class QueryStep(BaseModel):
    type: Literal[
        "GRAPH_NODE_DETAIL",
        "GRAPH_NEIGHBORHOOD",
        "GRAPH_PATH",
        "TEXT_SEARCH",
        "IMAGE_SEARCH",
    ]
    target_node_id: Optional[str] = None
    target_node_ids: List[str] = Field(default_factory=list)
    query: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class QueryPlan(BaseModel):
    steps: List[QueryStep] = Field(default_factory=list)
    max_hops: int = 2
    max_nodes: int = 20
    max_edges: int = 40


class EvidenceItem(BaseModel):
    evidence_id: str
    kind: Literal["graph_fact", "text", "image"]
    content: str
    title: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    image_url: Optional[str] = None
    page_no: Optional[int] = None
    source_label: Optional[str] = None
    location_label: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    evidence_id: str
    label: Optional[str] = None


class TraceEvent(BaseModel):
    stage: str
    status: Literal["STARTED", "COMPLETED", "FAILED"]
    duration_ms: Optional[float] = None
    summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QAResponse(BaseModel):
    run_id: str
    status: Literal["COMPLETED", "NEEDS_CLARIFICATION", "FAILED"]
    answer: str = ""
    total_duration_ms: Optional[float] = None
    domain: Optional[DomainContext] = None
    analysis: Optional[InputAnalysis] = None
    entities: List[LinkedEntity] = Field(default_factory=list)
    plan: Optional[QueryPlan] = None
    graph: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    trace: List[TraceEvent] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

