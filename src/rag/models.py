from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from src.database.base import Base


class ScenicArea(Base):
    __tablename__ = 'scenic_areas'

    id = Column(Integer, primary_key=True)
    source_scenic_id = Column(String(255), index=True, nullable=False, comment='稳定景区ID，例如 code')
    source_scenic_pk = Column(Integer, comment='A 端自增主键')
    name = Column(String(255), comment='景区名称')
    description = Column(Text, comment='景区描述')
    location = Column(String(255), comment='文本化位置描述')
    metadata = Column(JSONB, nullable=False, server_default="{}", comment='额外元数据')
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())


class SemanticNode(Base):
    __tablename__ = 'semantic_nodes'

    id = Column(Integer, primary_key=True)
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='CASCADE'), index=True, nullable=False)
    source_scenic_id = Column(String(255), index=True, nullable=False)
    source_node_id = Column(String(255), index=True, nullable=False, comment='A端节点ID')
    parent_source_node_id = Column(String(255), comment='父节点的 source_node_id')
    node_name = Column(String(255))
    node_type = Column(String(64), comment='ScenicArea/Region/Building/POI/Object/Person')
    description = Column(Text)
    lng = Column(Float)
    lat = Column(Float)
    properties = Column(JSONB, nullable=False, server_default="{}", comment='聚合的节点属性')
    tags = Column(JSONB, nullable=False, server_default="[]")
    content_hash = Column(String(128))
    sync_version = Column(String(64))
    source_updated_at = Column(TIMESTAMP)
    source_table = Column(String(128), nullable=False, default='wiki_custom_node')
    source_pk = Column(String(128))
    source_url = Column(Text)
    source_title = Column(String(255))
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())


class SemanticEdge(Base):
    __tablename__ = 'semantic_edges'

    id = Column(Integer, primary_key=True)
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='CASCADE'), index=True, nullable=False)
    source_scenic_id = Column(String(255), index=True, nullable=False)
    source_relation_id = Column(String(255), index=True)
    source_node_id = Column(String(255), nullable=False, comment='source node id')
    target_node_id = Column(String(255), nullable=False, comment='target node id')
    relation_type = Column(String(128), nullable=False)
    relation_label = Column(String(128))
    relation_layer = Column(String(32), nullable=False, comment='spatial or semantic')
    relation_category = Column(String(128))
    description = Column(Text)
    evidence_text = Column(Text)
    confidence = Column(Float)
    is_verified = Column(Boolean, default=False)
    properties = Column(JSONB, nullable=False, server_default="{}")
    sync_version = Column(String(64))
    source_table = Column(String(128), nullable=False, default='wiki_custom_noderelation')
    source_pk = Column(String(128))
    source_url = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())


class NodeAsset(Base):
    __tablename__ = 'node_assets'

    id = Column(Integer, primary_key=True)
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='CASCADE'), index=True, nullable=False)
    source_scenic_id = Column(String(255), index=True, nullable=False)
    source_asset_id = Column(String(255), index=True, nullable=False)
    source_binding_id = Column(String(255), index=True)
    source_node_id = Column(String(255), index=True, nullable=False)
    asset_type = Column(String(32), comment='image/audio/video/document')
    url = Column(Text)
    title = Column(String(255))
    caption = Column(Text)
    ocr_text = Column(Text)
    role = Column(String(64))
    is_cover = Column(Boolean, default=False)
    file_hash = Column(String(128))
    metadata = Column(JSONB, nullable=False, server_default="{}")
    content_hash = Column(String(128))
    sync_version = Column(String(64))
    source_table = Column(String(128), nullable=False, default='wiki_custom_imageasset')
    source_pk = Column(String(128))
    source_url = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())


class KnowledgeChunk(Base):
    __tablename__ = 'knowledge_chunks'

    id = Column(Integer, primary_key=True)
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='CASCADE'), index=True, nullable=False)
    source_scenic_id = Column(String(255), index=True, nullable=False)
    source_type = Column(String(64), comment='semantic_node/node_asset/scenic_overview/...')
    source_id = Column(String(255), comment='对应来源的 id，例如 node id / asset id')
    source_node_id = Column(String(255), index=True, nullable=False)
    chunk_type = Column(String(64))
    title = Column(String(255))
    content = Column(Text)
    metadata = Column(JSONB, nullable=False, server_default="{}")
    content_hash = Column(String(128))
    sync_version = Column(String(64))
    source_table = Column(String(128))
    source_pk = Column(String(128))
    source_field = Column(String(128))
    source_title = Column(String(255))
    source_url = Column(Text)
    evidence_text = Column(Text)
    source_updated_at = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class TextEmbedding(Base):
    __tablename__ = 'text_embeddings'

    id = Column(Integer, primary_key=True)
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='CASCADE'), index=True, nullable=False)
    chunk_id = Column(Integer, ForeignKey('knowledge_chunks.id', ondelete='CASCADE'), index=True, nullable=False)
    source_node_id = Column(String(255), index=True, nullable=False)
    embedding = Column(Vector(1024))
    model_name = Column(String(255))
    sync_version = Column(String(64))
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class ImageEmbedding(Base):
    __tablename__ = 'image_embeddings'

    id = Column(Integer, primary_key=True)
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='CASCADE'), index=True, nullable=False)
    asset_id = Column(Integer, ForeignKey('node_assets.id', ondelete='CASCADE'), index=True, nullable=False)
    source_node_id = Column(String(255), index=True, nullable=False)
    embedding = Column(Vector(128))
    model_name = Column(String(255))
    sync_version = Column(String(64))
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class ClipImageEmbedding(Base):
    __tablename__ = 'clip_image_embeddings'

    id = Column(Integer, primary_key=True)
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='CASCADE'), index=True, nullable=False)
    asset_id = Column(Integer, ForeignKey('node_assets.id', ondelete='CASCADE'), index=True, nullable=False)
    source_node_id = Column(String(255), index=True, nullable=False)
    embedding = Column(Vector(512))
    model_name = Column(String(255))
    embedding_version = Column(String(64))
    sync_version = Column(String(64))
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class SyncJob(Base):
    __tablename__ = 'sync_jobs'

    id = Column(Integer, primary_key=True)
    job_id = Column(String(255), unique=True, index=True, nullable=False)
    source_scenic_id = Column(String(255), index=True, nullable=False)
    sync_version = Column(String(64))
    job_type = Column(String(64))
    status = Column(String(32), nullable=False, default='PENDING', comment='PENDING/PROCESSING/SUCCESS/FAILED')
    error_message = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    started_at = Column(TIMESTAMP)
    finished_at = Column(TIMESTAMP)


class RetrievalRun(Base):
    __tablename__ = 'retrieval_runs'

    id = Column(Integer, primary_key=True)
    run_id = Column(String(128), unique=True, nullable=False)
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='SET NULL'))
    source_scenic_id = Column(String(255))
    query_text = Column(Text)
    query_image_url = Column(Text)
    intent_type = Column(String(64))
    weights = Column(JSONB, nullable=False, server_default="{}")
    answer = Column(Text)
    model_name = Column(String(255))
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class RetrievalHit(Base):
    __tablename__ = 'retrieval_hits'

    id = Column(Integer, primary_key=True)
    run_id = Column(String(128), ForeignKey('retrieval_runs.run_id', ondelete='CASCADE'), nullable=False)
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='SET NULL'))
    hit_type = Column(String(64), nullable=False)
    hit_id = Column(Integer)
    source_node_id = Column(String(255))
    source_table = Column(String(128))
    source_pk = Column(String(128))
    source_field = Column(String(128))
    score = Column(Float)
    rank = Column(Integer)
    snippet = Column(Text)
    metadata = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())


class IngestBatch(Base):
    __tablename__ = 'ingest_batches'

    id = Column(Integer, primary_key=True)
    batch_id = Column(String(128), unique=True, nullable=False)
    sync_job_id = Column(Integer, ForeignKey('sync_jobs.id', ondelete='SET NULL'))
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='CASCADE'))
    source_scenic_id = Column(String(255), nullable=False)
    batch_type = Column(String(64), nullable=False, default='new_docs')
    status = Column(String(32), nullable=False, default='PENDING')
    total_chunks = Column(Integer, default=0)
    duplicate_chunks = Column(Integer, default=0)
    low_quality_chunks = Column(Integer, default=0)
    conflict_chunks = Column(Integer, default=0)
    candidate_facts_count = Column(Integer, default=0)
    metadata = Column(JSONB, nullable=False, server_default="{}")
    error_message = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    started_at = Column(TIMESTAMP)
    finished_at = Column(TIMESTAMP)


class CandidateFact(Base):
    __tablename__ = 'candidate_facts'

    id = Column(Integer, primary_key=True)
    batch_id = Column(String(128), ForeignKey('ingest_batches.batch_id', ondelete='SET NULL'))
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='CASCADE'))
    source_scenic_id = Column(String(255))
    source_chunk_id = Column(Integer, ForeignKey('knowledge_chunks.id', ondelete='SET NULL'))
    related_node_id = Column(Integer, ForeignKey('semantic_nodes.id', ondelete='SET NULL'))
    fact_type = Column(String(64), nullable=False)
    subject_name = Column(String(255))
    subject_node_id = Column(Integer, ForeignKey('semantic_nodes.id', ondelete='SET NULL'))
    predicate = Column(String(128))
    object_value = Column(Text)
    object_node_id = Column(Integer, ForeignKey('semantic_nodes.id', ondelete='SET NULL'))
    confidence = Column(Float)
    evidence_text = Column(Text)
    llm_reasoning = Column(Text)
    fact_hash = Column(String(128))
    status = Column(String(32), nullable=False, default='PENDING')
    reviewed_by = Column(String(128))
    reviewed_at = Column(TIMESTAMP)
    review_note = Column(Text)
    accepted_node_id = Column(Integer, ForeignKey('semantic_nodes.id', ondelete='SET NULL'))
    accepted_edge_id = Column(Integer, ForeignKey('semantic_edges.id', ondelete='SET NULL'))
    metadata = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())


class ConflictCase(Base):
    __tablename__ = 'conflict_cases'

    id = Column(Integer, primary_key=True)
    case_id = Column(String(128), unique=True, nullable=False)
    batch_id = Column(String(128), ForeignKey('ingest_batches.batch_id', ondelete='SET NULL'))
    candidate_fact_id = Column(Integer, ForeignKey('candidate_facts.id', ondelete='SET NULL'))
    scenic_id = Column(Integer, ForeignKey('scenic_areas.id', ondelete='CASCADE'))
    source_scenic_id = Column(String(255))
    source_chunk_id = Column(Integer, ForeignKey('knowledge_chunks.id', ondelete='SET NULL'))
    related_node_id = Column(Integer, ForeignKey('semantic_nodes.id', ondelete='SET NULL'))
    related_edge_id = Column(Integer, ForeignKey('semantic_edges.id', ondelete='SET NULL'))
    conflict_type = Column(String(64), nullable=False)
    conflict_field = Column(String(128))
    existing_value = Column(Text)
    new_value = Column(Text)
    evidence_text = Column(Text)
    llm_analysis = Column(Text)
    confidence = Column(Float)
    conflict_hash = Column(String(128))
    status = Column(String(32), nullable=False, default='PENDING')
    reviewed_by = Column(String(128))
    reviewed_at = Column(TIMESTAMP)
    review_note = Column(Text)
    resolved_value = Column(Text)
    metadata = Column(JSONB, nullable=False, server_default="{}")
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())