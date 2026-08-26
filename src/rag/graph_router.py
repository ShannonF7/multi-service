"""API for durable projection of A-side published graph snapshots."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Query

from src.rag.service.graph_sync_service import (
    create_or_get_graph_sync_job,
    get_graph_sync_job,
    neo4j_domain_stats,
    neo4j_health,
)
from src.rag.service.graph_discovery_service import (
    discover_published_graph_hypotheses,
    find_published_shortest_path,
    get_published_domain_overview,
    get_published_neighborhood,
    get_published_node_detail,
    search_published_entities,
)
from src.rag.service.graph_growth_service import (
    create_or_get_graph_discovery_job,
    get_graph_discovery_job,
    list_graph_discoveries,
    list_graph_discovery_jobs,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/rag/graph/sync/jobs")
async def graph_sync_job_create(payload: dict = Body(...)):
    try:
        return {"job": create_or_get_graph_sync_job(payload)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("create graph sync job failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/graph/sync/jobs/{job_id}")
async def graph_sync_job_get(job_id: int):
    try:
        job = get_graph_sync_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="graph sync job not found")
        return {"job": job}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get graph sync job failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/graph/health")
async def graph_health():
    try:
        return neo4j_health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/rag/graph/domains/{domain_id}/stats")
async def graph_domain_stats(domain_id: str):
    try:
        return neo4j_domain_stats(domain_id)
    except Exception as exc:
        logger.error("get graph domain stats failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/rag/graph/domains/{domain_id}/overview")
async def graph_domain_overview(
    domain_id: str,
    node_limit: int = Query(default=500, ge=1, le=2000),
    relation_limit: int = Query(default=3000, ge=1, le=10000),
):
    try:
        return get_published_domain_overview(
            domain_id,
            node_limit=node_limit,
            relation_limit=relation_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("get graph domain overview failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/rag/graph/domains/{domain_id}/nodes")
async def graph_node_search(
    domain_id: str,
    q: str = Query(default="", max_length=100),
    node_type: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=10000),
):
    try:
        return search_published_entities(
            domain_id,
            q,
            node_type=node_type,
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("search graph nodes failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/graph/domains/{domain_id}/nodes/{node_id}/detail")
async def graph_node_detail(
    domain_id: str,
    node_id: str,
    relation_limit: int = Query(default=2000, ge=1, le=5000),
):
    try:
        return get_published_node_detail(
            domain_id,
            node_id,
            relation_limit=relation_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("get graph node detail failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/rag/graph/domains/{domain_id}/nodes/{node_id}/neighborhood")
async def graph_node_neighborhood(
    domain_id: str,
    node_id: str,
    depth: int = Query(default=1, ge=1, le=3),
    limit: int = Query(default=80, ge=1, le=300),
    offset: int = Query(default=0, ge=0, le=10000),
    node_type: list[str] | None = Query(default=None),
    relation_type: list[str] | None = Query(default=None),
):
    try:
        return get_published_neighborhood(
            domain_id,
            node_id,
            depth=depth,
            limit=limit,
            offset=offset,
            node_types=node_type,
            relation_types=relation_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("get graph neighborhood failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/graph/domains/{domain_id}/nodes/{node_id}/discoveries")
async def graph_node_discoveries(
    domain_id: str,
    node_id: str,
    peer_limit: int = Query(default=5, ge=0, le=12),
    related_limit: int = Query(default=5, ge=0, le=12),
):
    try:
        return discover_published_graph_hypotheses(
            domain_id,
            node_id,
            peer_limit=peer_limit,
            related_limit=related_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("get graph discoveries failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/graph/domains/{domain_id}/nodes/{node_id}/paths/{target_node_id}")
async def graph_node_shortest_path(
    domain_id: str,
    node_id: str,
    target_node_id: str,
    max_depth: int = Query(default=6, ge=1, le=10),
):
    try:
        return find_published_shortest_path(
            domain_id,
            node_id,
            target_node_id,
            max_depth=max_depth,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("get graph shortest path failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/rag/graph/discovery/jobs")
async def graph_discovery_job_create(payload: dict = Body(...)):
    try:
        return {"job": create_or_get_graph_discovery_job(payload)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("create graph discovery job failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/graph/discovery/jobs")
async def graph_discovery_job_list(
    domain_identifier: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    try:
        return list_graph_discovery_jobs(
            domain_identifier=domain_identifier,
            status=status,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("list graph discovery jobs failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/graph/discovery/jobs/{job_id}")
async def graph_discovery_job_get(job_id: int):
    try:
        job = get_graph_discovery_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="graph discovery job not found")
        return {"job": job}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get graph discovery job failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/rag/graph/discoveries")
async def graph_discovery_list(
    domain_id: str | None = Query(default=None),
    source_node_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        return list_graph_discoveries(
            domain_id=domain_id,
            source_node_id=source_node_id,
            status=status,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("list graph discoveries failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
