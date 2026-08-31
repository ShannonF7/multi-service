"""Semantic completion router for RAG evidence-first pipeline."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

_RAG_TMP_DIR = Path(__file__).resolve().parents[2] / "data" / "tmp"
_RAG_TMP_DIR.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_RAG_TMP_DIR)

from fastapi import APIRouter, BackgroundTasks, Body, File, Form, HTTPException, Query, UploadFile

from src.rag.schemas import DomainShellRequest, DomainShellResponse, DomainDeleteResponse, DomainKbDeleteResponse, DomainKbDocumentListResponse, DomainKbEmbedResponse, DomainKbEmbeddingJobListResponse, DomainKbSearchResponse, DomainKbUploadResponse, DomainKbUploadTaskListResponse, DomainKbUploadTaskResponse, SemanticCandidateListResponse, SemanticCandidateStatusBatchUpdate, SemanticCandidateStatusUpdate, SemanticCompleteRequest, SemanticCompleteResponse, SemanticCompletionJobCreateResponse, SemanticCompletionJobResponse, KnowledgeChunkResponse, CandidateEvidenceResponse
from src.rag.service.semantic_completion_service import complete_semantic_service
from src.rag.service.semantic_candidate_store import list_semantic_candidates, list_semantic_candidate_groups, update_semantic_candidate_status, update_semantic_candidate_status_batch
from src.rag.service.domain_kb_service import delete_domain_kb_document, ingest_domain_kb_files, list_domain_kb_documents, search_domain_kb, upsert_domain_shell, delete_domain_all
from src.rag.service.embedding_job_service import enqueue_domain_kb_embedding_jobs, list_embedding_jobs, process_one_embedding_job
from src.rag.service.domain_kb_upload_task_service import assemble_upload_task, create_upload_task, get_upload_task, list_upload_tasks, process_upload_task, save_upload_chunk
from src.rag.service.completion_job_service import create_or_get_semantic_completion_job, get_latest_semantic_completion_job, get_semantic_completion_job
from src.rag.service.evidence_store import list_semantic_evidence_items, get_candidate_evidence, get_knowledge_chunk
from src.rag.service.gap_status_service import list_semantic_gap_status
from src.rag.service.semantic_node_status_service import get_node_semantic_status, list_node_candidates, list_node_conflicts, list_node_gaps
from src.rag.service.semantic_batch_job_service import attach_semantic_batch_item, create_semantic_batch_job, get_semantic_batch_job, list_semantic_batch_jobs

logger = logging.getLogger(__name__)

router = APIRouter()




def _force_fast_sync_payload(payload: SemanticCompleteRequest) -> SemanticCompleteRequest:
    data = payload.dict()
    metadata = data.setdefault("metadata", {}) or {}
    depth = int(data.get("subgraph_depth") or 0)
    retrieval_scope = "self_web" if depth == 0 else ("domain" if depth < 0 else "subgraph")
    metadata["retrieval_scope"] = retrieval_scope
    metadata.setdefault("completion_mode", "quick")
    metadata.setdefault("question_batch_size", 3)
    if retrieval_scope == "self_web":
        metadata["source_scope"] = ["provided_evidence"]
    elif "source_scope" not in metadata:
        metadata["source_scope"] = ["provided_evidence", "domain_kb"]
    metadata["source_scope"] = [item for item in metadata.get("source_scope", []) if item not in {"web", "web_search", "web_extractor"}]
    data["metadata"] = metadata
    data["use_web_search"] = False
    data["use_web_extractor"] = False
    return SemanticCompleteRequest.parse_obj(data)


@router.post("/rag/semantic/complete", response_model=SemanticCompleteResponse)
async def semantic_complete(payload: SemanticCompleteRequest):
    try:
        return await complete_semantic_service(_force_fast_sync_payload(payload))
    except Exception as exc:
        logger.error("semantic completion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))



@router.post("/rag/semantic/batch-jobs")
async def semantic_batch_job_create(payload: dict = Body(default={})):
    try:
        created_by = str(payload.get("created_by") or (payload.get("metadata") or {}).get("username") or "") or None
        return {"batch": create_semantic_batch_job(payload, created_by=created_by)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("create semantic batch job failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/semantic/batch-jobs")
async def semantic_batch_job_list(
    source_scenic_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    try:
        return list_semantic_batch_jobs(source_scenic_id=source_scenic_id, status=status, limit=limit, offset=offset)
    except Exception as exc:
        logger.error("list semantic batch jobs failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/semantic/batch-jobs/{batch_id}")
async def semantic_batch_job_get(batch_id: int):
    try:
        row = get_semantic_batch_job(batch_id)
        if not row:
            raise HTTPException(status_code=404, detail="batch job not found")
        return row
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get semantic batch job failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rag/semantic/batch-jobs/{batch_id}/items")
async def semantic_batch_job_attach_item(batch_id: int, payload: dict = Body(default={})):
    try:
        return {"batch": attach_semantic_batch_item(batch_id, payload)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("attach semantic batch item failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rag/semantic/complete/jobs", response_model=SemanticCompletionJobCreateResponse)
async def semantic_complete_create_job(payload: SemanticCompleteRequest):
    try:
        created_by = str((payload.metadata or {}).get("username") or (payload.metadata or {}).get("user_id") or "") or None
        job = create_or_get_semantic_completion_job(payload, created_by=created_by)
        return {"job": job}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("create semantic completion job failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/semantic/complete/jobs/{job_id}", response_model=SemanticCompletionJobResponse)
async def semantic_complete_job_status(job_id: int):
    try:
        row = get_semantic_completion_job(job_id)
        if not row:
            raise HTTPException(status_code=404, detail="job not found")
        return row
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get semantic completion job failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/semantic/complete/jobs/latest", response_model=SemanticCompletionJobResponse)
async def semantic_complete_latest_job(
    source_scenic_id: str = Query(...),
    source_node_id: str = Query(...),
):
    try:
        row = get_latest_semantic_completion_job(source_scenic_id=source_scenic_id, source_node_id=source_node_id)
        if not row:
            raise HTTPException(status_code=404, detail="latest job not found")
        return row
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get latest semantic completion job failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))



@router.get("/rag/semantic/nodes/{source_node_id}/status")
async def semantic_node_status(
    source_node_id: str,
    source_scenic_id: str = Query(...),
):
    try:
        return get_node_semantic_status(source_scenic_id=source_scenic_id, source_node_id=source_node_id)
    except Exception as exc:
        logger.error("get semantic node status failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/semantic/nodes/{source_node_id}/candidates")
async def semantic_node_candidates(
    source_node_id: str,
    source_scenic_id: str = Query(...),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        return list_node_candidates(source_scenic_id=source_scenic_id, source_node_id=source_node_id, status=status, limit=limit, offset=offset)
    except Exception as exc:
        logger.error("list semantic node candidates failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/semantic/nodes/{source_node_id}/conflicts")
async def semantic_node_conflicts(
    source_node_id: str,
    source_scenic_id: str = Query(...),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        return list_node_conflicts(source_scenic_id=source_scenic_id, source_node_id=source_node_id, limit=limit, offset=offset)
    except Exception as exc:
        logger.error("list semantic node conflicts failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/semantic/nodes/{source_node_id}/gaps")
async def semantic_node_gaps(
    source_node_id: str,
    source_scenic_id: str = Query(...),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        return list_node_gaps(source_scenic_id=source_scenic_id, source_node_id=source_node_id, status=status, limit=limit, offset=offset)
    except Exception as exc:
        logger.error("list semantic node gaps failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/semantic/candidates", response_model=SemanticCandidateListResponse)
async def semantic_candidates(
    source_scenic_id: str | None = Query(default=None),
    source_node_id: str | None = Query(default=None),
    trace_id: str | None = Query(default=None),
    job_id: int | None = Query(default=None),
    status: str | None = Query(default=None),
    risk_level: str | None = Query(default=None),
    publication_policy: str | None = Query(default=None),
    discovery_track: str | None = Query(default=None, description="审核业务链：TARGETED_COMPLETION/OPEN_DISCOVERY/ASSET_BINDING"),
    candidate_kind: str | None = Query(default=None, description="候选类型：PROPERTY/RELATION/NODE/ASSET_BINDING/CONFLICT"),
    review_surface: str | None = Query(default=None, description="审核页面：NODE_WORKBENCH/GROWTH_RUN/AUDIT_ONLY"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    # 中文说明：节点工作台默认只看定向补全，避免 GrowthRun 候选混入审核池。
    if source_node_id and discovery_track is None:
        discovery_track = "TARGETED_COMPLETION"
    try:
        return list_semantic_candidates(
            source_scenic_id=source_scenic_id,
            source_node_id=source_node_id,
            trace_id=trace_id,
            job_id=job_id,
            status=status,
            risk_level=risk_level,
            publication_policy=publication_policy,
            discovery_track=discovery_track,
            candidate_kind=candidate_kind,
            review_surface=review_surface,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("list semantic candidates failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/semantic/candidate-groups")
async def semantic_candidate_groups(
    source_scenic_id: str | None = Query(default=None),
    source_node_id: str | None = Query(default=None),
    trace_id: str | None = Query(default=None),
    job_id: int | None = Query(default=None),
    gap_status: str | None = Query(default=None),
    conflict_class: str | None = Query(default=None),
    discovery_track: str | None = Query(default=None, description="审核业务链：TARGETED_COMPLETION/OPEN_DISCOVERY/ASSET_BINDING"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """返回候选分组并隔离节点工作台与 GrowthRun 审核面。

    调用：A 端节点语义补全分组页面。
    """
    if source_node_id and discovery_track is None:
        discovery_track = "TARGETED_COMPLETION"
    try:
        return list_semantic_candidate_groups(
            source_scenic_id=source_scenic_id,
            source_node_id=source_node_id,
            trace_id=trace_id,
            job_id=job_id,
            gap_status=gap_status,
            conflict_class=conflict_class,
            discovery_track=discovery_track,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("list semantic candidate groups failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))




@router.get("/rag/semantic/gap-status")
async def semantic_gap_status(
    source_scenic_id: str | None = Query(default=None),
    source_node_id: str | None = Query(default=None),
    job_id: int | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        return list_semantic_gap_status(
            source_scenic_id=source_scenic_id,
            source_node_id=source_node_id,
            job_id=job_id,
            status=status,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("list semantic gap status failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rag/semantic/candidates/status-batch")
async def semantic_candidate_status_batch(payload: SemanticCandidateStatusBatchUpdate):
    try:
        return update_semantic_candidate_status_batch(
            payload.candidate_ids,
            status=payload.status,
            reviewed_by=payload.reviewed_by,
            review_note=payload.review_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("batch update semantic candidate status failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/rag/semantic/candidates/{candidate_id}/status")
async def semantic_candidate_status(candidate_id: int, payload: SemanticCandidateStatusUpdate):
    try:
        row = update_semantic_candidate_status(
            candidate_id,
            status=payload.status,
            reviewed_by=payload.reviewed_by,
            review_note=payload.review_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("update semantic candidate status failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    if row is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    return row


@router.get("/rag/semantic/evidence")
async def semantic_evidence_items(
    job_id: int | None = Query(default=None),
    trace_id: str | None = Query(default=None),
    source_scenic_id: str | None = Query(default=None),
    source_node_id: str | None = Query(default=None),
    question_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        return list_semantic_evidence_items(
            job_id=job_id,
            trace_id=trace_id,
            source_scenic_id=source_scenic_id,
            source_node_id=source_node_id,
            question_id=question_id,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("list semantic evidence failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/semantic/candidates/{candidate_id}/evidence", response_model=CandidateEvidenceResponse)
async def semantic_candidate_evidence(candidate_id: int):
    try:
        row = get_candidate_evidence(candidate_id)
    except Exception as exc:
        logger.error("get candidate evidence failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    if row is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    return row


@router.get("/rag/knowledge-chunks/{chunk_id}", response_model=KnowledgeChunkResponse)
async def rag_knowledge_chunk(chunk_id: int):
    try:
        row = get_knowledge_chunk(chunk_id)
    except Exception as exc:
        logger.error("get knowledge chunk failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    if row is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return row


@router.post("/rag/domains", response_model=DomainShellResponse)
async def upsert_rag_domain(payload: DomainShellRequest):
    try:
        return upsert_domain_shell(payload.dict())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("upsert rag domain failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/rag/domains/{source_scenic_id}", response_model=DomainDeleteResponse)
async def delete_rag_domain(source_scenic_id: str):
    try:
        return delete_domain_all(source_scenic_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("delete rag domain failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rag/domain-kb/upload/init", response_model=DomainKbUploadTaskResponse)
async def domain_kb_upload_init(
    source_scenic_id: str = Form(...),
    filename: str = Form(...),
    total_size: int = Form(default=0),
    total_chunks: int = Form(...),
    source_scenic_pk: int | None = Form(default=None),
    scenic_name: str | None = Form(default=None),
    submitted_by: str | None = Form(default=None),
):
    try:
        return create_upload_task(
            source_scenic_id=source_scenic_id,
            source_scenic_pk=source_scenic_pk,
            scenic_name=scenic_name,
            submitted_by=submitted_by,
            filename=filename,
            total_size=total_size,
            total_chunks=total_chunks,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("init domain kb upload task failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rag/domain-kb/upload/chunk", response_model=DomainKbUploadTaskResponse)
async def domain_kb_upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
):
    try:
        return save_upload_chunk(upload_id, chunk_index, await chunk.read())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("save domain kb upload chunk failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rag/domain-kb/upload/complete", response_model=DomainKbUploadTaskResponse)
async def domain_kb_upload_complete(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
):
    try:
        task = assemble_upload_task(upload_id)
        background_tasks.add_task(process_upload_task, upload_id)
        return task
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("complete domain kb upload task failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/domain-kb/upload/tasks/{upload_id}", response_model=DomainKbUploadTaskResponse)
async def domain_kb_upload_task(upload_id: str):
    try:
        return get_upload_task(upload_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("get domain kb upload task failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/domain-kb/upload/tasks", response_model=DomainKbUploadTaskListResponse)
async def domain_kb_upload_tasks(
    source_scenic_id: str = Query(...),
    limit: int = Query(default=50, ge=1, le=200),
):
    try:
        return list_upload_tasks(source_scenic_id, limit=limit)
    except Exception as exc:
        logger.error("list domain kb upload tasks failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rag/domain-kb/upload", response_model=DomainKbUploadResponse)
async def upload_domain_kb(
    source_scenic_id: str = Form(...),
    source_scenic_pk: int | None = Form(default=None),
    scenic_name: str | None = Form(default=None),
    submitted_by: str | None = Form(default=None),
    files: list[UploadFile] = File(...),
):
    """Upload domain pre-knowledge files and store them as knowledge chunks."""
    try:
        payload_files = []
        for item in files:
            payload_files.append((item.filename or "upload", await item.read()))
        if not payload_files:
            raise HTTPException(status_code=400, detail="files required")
        result = ingest_domain_kb_files(
            source_scenic_id=source_scenic_id,
            source_scenic_pk=source_scenic_pk,
            scenic_name=scenic_name,
            submitted_by=submitted_by,
            files=payload_files,
        )
        source_ids = [item.get("doc_id") for item in result.get("files", []) if item.get("chunks", 0) > 0]
        if source_ids:
            result["embedding_jobs"] = enqueue_domain_kb_embedding_jobs(source_scenic_id, source_ids)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("upload domain kb failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/domain-kb/documents", response_model=DomainKbDocumentListResponse)
async def domain_kb_documents(
    source_scenic_id: str = Query(...),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        return list_domain_kb_documents(source_scenic_id, limit=limit, offset=offset)
    except Exception as exc:
        logger.error("list domain kb documents failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))





@router.delete("/rag/domain-kb/documents/{source_id}", response_model=DomainKbDeleteResponse)
async def domain_kb_delete_document(
    source_id: str,
    source_scenic_id: str = Query(...),
):
    try:
        result = delete_domain_kb_document(source_scenic_id, source_id)
        logger.info(
            "domain kb document delete scenic=%s source_id=%s chunks=%s embeddings=%s jobs=%s files=%s deleted=%s",
            result.get("source_scenic_id"),
            result.get("source_id"),
            result.get("chunks_deleted"),
            result.get("embeddings_deleted"),
            result.get("jobs_deleted"),
            result.get("files_deleted"),
            result.get("deleted"),
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("delete domain kb document failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/rag/domain-kb/search", response_model=DomainKbSearchResponse)
async def domain_kb_search(
    source_scenic_id: str = Query(...),
    q: str = Query(...),
    limit: int = Query(default=5, ge=1, le=20),
):
    try:
        items = search_domain_kb(source_scenic_id, q, limit=limit)
        return {"items": items, "total": len(items)}
    except Exception as exc:
        logger.error("search domain kb failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))



@router.post("/rag/domain-kb/embed", response_model=DomainKbEmbedResponse)
async def domain_kb_embed(
    source_scenic_id: str = Form(...),
    source_id: str | None = Form(default=None),
    run_inline: bool = Form(default=False),
):
    try:
        if source_id:
            queued = enqueue_domain_kb_embedding_jobs(source_scenic_id, [source_id])
        else:
            docs = list_domain_kb_documents(source_scenic_id, limit=500)
            queued = enqueue_domain_kb_embedding_jobs(source_scenic_id, [x.get("source_id") for x in docs.get("items", [])])
        if run_inline:
            processed = process_one_embedding_job(worker_id="api-inline", device="")
            return {"source_scenic_id": source_scenic_id, "documents": [processed] if processed else [], "total": 1 if processed else 0, "queued": queued}
        return {"source_scenic_id": source_scenic_id, "documents": [], "total": 0, "queued": queued}
    except Exception as exc:
        logger.error("embed domain kb failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/domain-kb/embedding-jobs", response_model=DomainKbEmbeddingJobListResponse)
async def domain_kb_embedding_jobs(
    source_scenic_id: str = Query(...),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        return list_embedding_jobs(source_scenic_id, limit=limit, offset=offset)
    except Exception as exc:
        logger.error("list domain kb embedding jobs failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
