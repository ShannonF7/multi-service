"""Pydantic schemas for RAG scenic sync."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FlexibleModel(BaseModel):
    class Config:
        extra = "allow"


class ScenicPayload(FlexibleModel):
    source_scenic_id: str
    source_scenic_pk: Optional[int] = None
    code: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodePayload(FlexibleModel):
    source_node_id: str
    parent_source_node_id: Optional[str] = None
    name: Optional[str] = None
    node_type: Optional[str] = None
    description: Optional[str] = None
    lng: Optional[float] = None
    lat: Optional[float] = None
    tags: List[Any] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_updated_at: Optional[datetime] = None
    source_url: Optional[str] = None
    sort_order: Optional[int] = None
    is_active: Optional[bool] = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationTypePayload(FlexibleModel):
    code: str
    label: Optional[str] = None
    description: Optional[str] = None
    allowed_source_types: List[str] = Field(default_factory=list)
    allowed_target_types: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationPayload(FlexibleModel):
    source_relation_id: Optional[str] = None
    source_node_id: str
    target_node_id: str
    relation_type: str
    relation_type_label: Optional[str] = None
    relation_category: Optional[str] = None
    relation_layer: Optional[str] = None
    description: Optional[str] = None
    evidence_text: Optional[str] = None
    evidence_source_id: Optional[str] = None
    confidence: Optional[float] = None
    extraction_method: Optional[str] = None
    is_verified: Optional[bool] = False
    version: Optional[int] = 1
    sort_order: Optional[int] = 0
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PropertyPayload(FlexibleModel):
    source_property_id: str
    source_node_id: str
    key: str
    raw_value: Optional[str] = None
    value: Optional[str] = None
    value_type: Optional[str] = "string"
    outer_status: Optional[str] = None
    status: Optional[str] = None
    claim_status: Optional[str] = None
    confidence: Optional[float] = None
    source_text: Optional[str] = None
    source_url: Optional[str] = None
    evidence_source_id: Optional[str] = None
    is_locked: Optional[bool] = False
    version: Optional[int] = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImageAssetPayload(FlexibleModel):
    source_asset_id: str
    url: Optional[str] = None
    file_url: Optional[str] = None
    title: Optional[str] = None
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    file_hash: Optional[str] = None
    original_filename: Optional[str] = None
    source: Optional[str] = None
    exif_lat: Optional[float] = None
    exif_lng: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImageBindingPayload(FlexibleModel):
    source_binding_id: Optional[str] = None
    source_asset_id: str
    source_node_id: str
    role: Optional[str] = None
    is_cover: Optional[bool] = False
    sort_order: Optional[int] = 0
    object_type: Optional[str] = "node"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScenicSyncPayload(FlexibleModel):
    schema_version: str = "2026-06-26.v1"
    source_system: str = "A"
    source_job_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    generated_at: Optional[datetime] = None
    submitted_by: Optional[str] = None
    scenic: ScenicPayload
    nodes: List[NodePayload] = Field(default_factory=list)
    relation_types: List[RelationTypePayload] = Field(default_factory=list)
    relations: List[RelationPayload] = Field(default_factory=list)
    properties: List[PropertyPayload] = Field(default_factory=list)
    image_assets: List[ImageAssetPayload] = Field(default_factory=list)
    image_bindings: List[ImageBindingPayload] = Field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScenicSyncResponse(FlexibleModel):
    job_id: str
    status: str
    source_scenic_id: str
    counts: Dict[str, int] = Field(default_factory=dict)
    diagnostics: List[Dict[str, Any]] = Field(default_factory=list)


class SyncJobStatusResponse(FlexibleModel):
    job_id: str
    status: str
    current_step: Optional[str] = None
    counts: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
    error_message: Optional[str] = None
    events: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evidence-first semantic completion schemas
# ---------------------------------------------------------------------------


class SemanticNodeContext(FlexibleModel):
    source_node_id: str
    name: Optional[str] = None
    node_type: Optional[str] = None
    description: Optional[str] = None
    parent_name: Optional[str] = None
    scenic_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticPropertyContext(FlexibleModel):
    key: str
    value: Optional[str] = None
    value_type: Optional[str] = None
    source_text: Optional[str] = None
    source_url: Optional[str] = None
    confidence: Optional[float] = None
    status: Optional[str] = None


class SemanticRelationContext(FlexibleModel):
    relation_type: str
    target_name: Optional[str] = None
    target_type: Optional[str] = None
    evidence_text: Optional[str] = None
    source_url: Optional[str] = None
    confidence: Optional[float] = None


class SemanticEvidenceInput(FlexibleModel):
    """补全与自增长共用的证据输入契约。

    输入：文本或图片证据及其定位信息；输出：传递给语义抽取服务的单条证据。
    图片证据额外保留页码、框坐标、标题、邻近文本和 OCR 块，不改变旧补全字段。
    """
    title: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    source_url: Optional[str] = None
    quote: Optional[str] = None
    score: Optional[float] = 0.0
    source_doc_id: Optional[str] = None
    chunk_id: Optional[int] = None
    page_no: Optional[int] = None
    asset_id: Optional[int] = None
    image: Optional[str] = None
    caption: Optional[str] = None
    nearby_text: Optional[str] = None
    bbox: Optional[Any] = None
    ocr_blocks: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticCompleteRequest(FlexibleModel):
    scenic_id: str
    node: SemanticNodeContext
    message: Optional[str] = ""
    source_note: Optional[str] = ""
    target_fields: List[str] = Field(default_factory=list)
    relation_intents: List[str] = Field(default_factory=list)
    subgraph_depth: Optional[int] = 0
    existing_properties: List[SemanticPropertyContext] = Field(default_factory=list)
    existing_relations: List[SemanticRelationContext] = Field(default_factory=list)
    graph_context: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[SemanticEvidenceInput] = Field(default_factory=list)
    max_web_results: int = 5
    use_web_search: bool = True
    use_web_extractor: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceChunk(FlexibleModel):
    source_id: str
    title: Optional[str] = None
    content: str
    quote: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    source_url: Optional[str] = None
    source_doc_id: Optional[str] = None
    chunk_id: Optional[int] = None
    page_no: Optional[int] = None
    score: float = 0.0
    question_id: Optional[str] = None
    target_kind: Optional[str] = None
    target_field: Optional[str] = None
    relation_intent: Optional[str] = None
    temporal_role: Optional[str] = None
    query_text: Optional[str] = None
    retrieval_score: Optional[float] = 0.0
    rerank_score: Optional[float] = 0.0
    source_weight: Optional[float] = 0.0
    final_evidence_score: Optional[float] = 0.0


class CandidateClaim(FlexibleModel):
    claim_id: str
    claim_type: str
    subject_node_id: str
    subject_name: Optional[str] = None
    predicate: Optional[str] = None
    object_value: Optional[str] = None
    object_name: Optional[str] = None
    object_type: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    quote: Optional[str] = None
    question_id: Optional[str] = None
    temporal_role: Optional[str] = None
    evidence_ids: List[int] = Field(default_factory=list)
    raw_value: Optional[str] = None
    normalized_value: Optional[str] = None
    display_value: Optional[str] = None
    confidence: float = 0.0
    evidence_score: float = 0.0
    evidence_status: str = "unverified"
    support_status: str = "needs_more_evidence"
    recommend_score: float = 0.0
    status: str = "pending"
    candidate_id: Optional[int] = None
    candidate_uid: Optional[str] = None
    candidate_type: Optional[str] = None
    candidate_group_key: Optional[str] = None
    value_group_key: Optional[str] = None
    conflict_group: Optional[str] = None
    conflict_class: Optional[str] = None
    gap_status: Optional[str] = None
    target_node_id: Optional[str] = None
    target_node_candidate_id: Optional[int] = None
    entity_resolution_status: Optional[str] = None
    possible_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    raw_type: Optional[str] = None
    suggested_type: Optional[str] = None
    type_confidence: float = 0.0
    risk_level: Optional[str] = None
    publication_policy: Optional[str] = None
    score_components: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaimConflict(FlexibleModel):
    conflict_type: str
    claim_id: Optional[str] = None
    predicate: Optional[str] = None
    existing_value: Optional[str] = None
    candidate_value: Optional[str] = None
    existing_target: Optional[str] = None
    candidate_target: Optional[str] = None
    reason: Optional[str] = None


class SemanticCompleteResponse(FlexibleModel):
    mode: str = "evidence_claims"
    trace_id: Optional[str] = None
    summary: str
    planned_sources: List[str] = Field(default_factory=list)
    evidence_chunks: List[EvidenceChunk] = Field(default_factory=list)
    candidate_claims: List[CandidateClaim] = Field(default_factory=list)
    conflicts: List[ClaimConflict] = Field(default_factory=list)
    template_fill: Dict[str, Any] = Field(default_factory=dict)
    discoveries: Dict[str, Any] = Field(default_factory=dict)
    candidates: Dict[str, Any] = Field(default_factory=dict)
    candidate_groups: List[Dict[str, Any]] = Field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_store: Dict[str, Any] = Field(default_factory=dict)



class SemanticCandidateStatusUpdate(FlexibleModel):
    status: str
    reviewed_by: Optional[str] = None
    review_note: Optional[str] = None


class SemanticCandidateStatusBatchUpdate(SemanticCandidateStatusUpdate):
    candidate_ids: List[int] = Field(default_factory=list)


class SemanticCandidateListResponse(FlexibleModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0


class SemanticEvidenceResponse(FlexibleModel):
    id: int
    trace_id: str
    job_id: Optional[int] = None
    scenic_id: Optional[int] = None
    node_id: Optional[int] = None
    source_type: str
    source_title: Optional[str] = None
    source_url: Optional[str] = None
    source_doc_id: Optional[str] = None
    chunk_id: Optional[int] = None
    page_no: Optional[int] = None
    quote: Optional[str] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    final_evidence_score: Optional[float] = None
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    source_weight: Optional[float] = None


class SemanticEvidenceListResponse(FlexibleModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0


class KnowledgeChunkResponse(FlexibleModel):
    chunk_id: int
    doc_id: Optional[str] = None
    doc_title: Optional[str] = None
    page_no: Optional[int] = None
    content: Optional[str] = None
    source_file: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateEvidenceResponse(FlexibleModel):
    candidate_id: int
    evidence: List[Dict[str, Any]] = Field(default_factory=list)


class SemanticCompletionJobResponse(FlexibleModel):
    id: int
    trace_id: str
    status: str
    progress: int = 0
    question_count: int = 0
    evidence_count: int = 0
    candidate_count: int = 0
    conflict_count: int = 0
    error_message: Optional[str] = None
    source_scenic_id: Optional[str] = None
    source_node_id: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    reused: bool = False
    result_data: Optional[Dict[str, Any]] = None


class SemanticCompletionJobCreateResponse(FlexibleModel):
    job: SemanticCompletionJobResponse


class DomainShellRequest(FlexibleModel):
    source_scenic_id: str
    source_scenic_pk: Optional[int] = None
    code: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DomainShellResponse(FlexibleModel):
    id: int
    source_scenic_id: str
    source_scenic_pk: Optional[int] = None
    name: Optional[str] = None
    created_or_updated: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DomainDeleteResponse(FlexibleModel):
    deleted: bool = False
    source_scenic_id: str
    scenic_id: Optional[int] = None
    counts: Dict[str, int] = Field(default_factory=dict)
    files_deleted: bool = False
    file_delete_error: Optional[str] = None


class DomainKbUploadResponse(FlexibleModel):
    source_scenic_id: str
    files: List[Dict[str, Any]] = Field(default_factory=list)
    total_files: int = 0
    total_chunks: int = 0


class DomainKbUploadTaskResponse(FlexibleModel):
    upload_id: str
    source_scenic_id: str
    filename: str
    original_filename: Optional[str] = None
    total_size: int = 0
    total_chunks: int = 0
    received_chunks: int = 0
    uploaded_bytes: int = 0
    progress: int = 0
    status: str = "UPLOADING"
    stage: str = "uploading"
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DomainKbUploadTaskListResponse(FlexibleModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0


class DomainKbDocumentListResponse(FlexibleModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0


class DomainKbSearchResponse(FlexibleModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0


class DomainKbEmbedResponse(FlexibleModel):
    source_scenic_id: str
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0


class DomainKbEmbeddingJobListResponse(FlexibleModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0


class DomainKbDeleteResponse(FlexibleModel):
    deleted: bool = False
    source_scenic_id: str
    source_id: str
    chunks_deleted: int = 0
    embeddings_deleted: int = 0
    jobs_deleted: int = 0
    files_deleted: bool = False
    file_delete_error: Optional[str] = None

