"""Lazy BGE embedding helpers for RAG knowledge chunks."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np
from sqlalchemy import text

from src.core.config import settings
from src.rag.dependencies import ai_session_scope

logger = logging.getLogger(__name__)

BGE_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH") or getattr(settings, "EMBEDDING_MODEL_PATH", "")
EMBED_DIM = 1024
MODEL_NAME = os.path.basename(os.path.normpath(BGE_MODEL_PATH)) or "bge-large-zh-v1.5"

_model = None
_model_lock = threading.Lock()


def load_bge_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if not BGE_MODEL_PATH:
            raise RuntimeError("EMBEDDING_MODEL_PATH is not configured")
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading BGE model from %s", BGE_MODEL_PATH)
        _model = BGEM3FlagModel(BGE_MODEL_PATH, device=os.getenv("EMBEDDING_DEVICE", "cuda:0"), use_fp16=True)
        logger.info("BGE model loaded")
        return _model


def embed_texts(texts: list[str], batch_size: int = 4) -> list[list[float]]:
    model = load_bge_model()
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outputs = model.encode(batch)
        if isinstance(outputs, dict) and "dense_vecs" in outputs:
            vecs = outputs["dense_vecs"]
        elif isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], dict) and "dense_vecs" in outputs[0]:
            vecs = outputs[0]["dense_vecs"]
        else:
            vecs = outputs
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        for item in vecs:
            if item.shape[0] != EMBED_DIM:
                raise RuntimeError(f"Unexpected embedding dim {item.shape[0]}, expected {EMBED_DIM}")
            embeddings.append(item.tolist())
    return embeddings


def embed_text(text_value: str) -> list[float]:
    return embed_texts([text_value], batch_size=1)[0]


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def _json_string(value: str) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def mark_chunks_embedding_status(chunk_ids: list[int], status: str, *, error: str = "") -> None:
    if not chunk_ids:
        return
    with ai_session_scope() as db:
        db.execute(
            text(
                """
                update knowledge_chunks
                set metadata = jsonb_set(
                    jsonb_set(coalesce(metadata, '{}'::jsonb), '{embedding_status}', cast(:status_json as jsonb), true),
                    '{embedding_error}', cast(:error_json as jsonb), true
                )
                where id = any(:chunk_ids)
                """
            ),
            {"chunk_ids": chunk_ids, "status_json": _json_string(status), "error_json": _json_string(error[:500])},
        )


def embed_domain_kb_document(source_scenic_id: str, source_id: str, *, batch_size: int = 4) -> dict[str, Any]:
    with ai_session_scope() as db:
        rows = db.execute(
            text(
                """
                select id, scenic_id, source_node_id, content
                from knowledge_chunks
                where source_scenic_id=:source_scenic_id
                  and source_type='domain_kb'
                  and source_id=:source_id
                  and coalesce(content, '') <> ''
                order by id
                """
            ),
            {"source_scenic_id": source_scenic_id, "source_id": source_id},
        ).mappings().all()
    chunk_ids = [int(row["id"]) for row in rows]
    if not rows:
        return {"embedded": 0, "status": "skipped", "reason": "no_chunks"}

    try:
        vectors = embed_texts([str(row["content"] or "") for row in rows], batch_size=batch_size)
        with ai_session_scope() as db:
            db.execute(text("delete from text_embeddings where chunk_id = any(:chunk_ids)"), {"chunk_ids": chunk_ids})
            for row, vector in zip(rows, vectors):
                db.execute(
                    text(
                        """
                        insert into text_embeddings (scenic_id, chunk_id, source_node_id, embedding, model_name, sync_version, created_at)
                        values (:scenic_id, :chunk_id, :source_node_id, cast(:embedding as vector), :model_name, 'domain-kb-v1', now())
                        """
                    ),
                    {
                        "scenic_id": int(row["scenic_id"]),
                        "chunk_id": int(row["id"]),
                        "source_node_id": row["source_node_id"] or "__domain__",
                        "embedding": to_pgvector(vector),
                        "model_name": MODEL_NAME,
                    },
                )
            db.execute(
                text(
                    """
                    update knowledge_chunks
                    set metadata = jsonb_set(
                        jsonb_set(
                            jsonb_set(coalesce(metadata, '{}'::jsonb), '{embedding_status}', cast(:status_json as jsonb), true),
                            '{embedding_model}', cast(:model_json as jsonb), true
                        ),
                        '{embedding_error}', cast(:error_json as jsonb), true
                    )
                    where id = any(:chunk_ids)
                    """
                ),
                {"chunk_ids": chunk_ids, "status_json": _json_string("done"), "model_json": _json_string(MODEL_NAME), "error_json": _json_string("")},
            )
        return {"embedded": len(vectors), "status": "done", "model_name": MODEL_NAME}
    except Exception as exc:
        logger.error("domain kb embedding failed: %s", exc, exc_info=True)
        mark_chunks_embedding_status(chunk_ids, "failed", error=str(exc))
        return {"embedded": 0, "status": "failed", "error": str(exc)}


def search_text_embedding_chunks(source_scenic_id: str, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
    if not str(query or "").strip():
        return []
    vector = to_pgvector(embed_text(query))
    with ai_session_scope() as db:
        rows = db.execute(
            text(
                """
                select kc.id, kc.source_id, kc.source_title, kc.title, kc.content, kc.source_url,
                       kc.evidence_text, kc.metadata,
                       (te.embedding <-> cast(:embedding as vector)) as distance
                from text_embeddings te
                join knowledge_chunks kc on kc.id = te.chunk_id
                where kc.source_scenic_id=:source_scenic_id
                  and kc.source_type='domain_kb'
                order by te.embedding <-> cast(:embedding as vector)
                limit :limit
                """
            ),
            {"source_scenic_id": source_scenic_id, "embedding": vector, "limit": limit},
        ).mappings().all()
    results: list[dict[str, Any]] = []
    for row in rows:
        content = str(row.get("content") or "")
        distance = float(row.get("distance") or 0.0)
        results.append(
            {
                "title": row.get("title") or row.get("source_title") or "前置知识库",
                "content": content,
                "quote": row.get("evidence_text") or content[:500],
                "source": row.get("source_title") or "前置知识库",
                "source_type": "domain_kb_vector",
                "source_url": None,
                "source_doc_id": row.get("source_id"),
                "chunk_id": row.get("id"),
                "page_no": int((row.get("metadata") or {}).get("chunk_index") or 0) or None,
                "score": max(0.0, 3.0 - distance),
                "distance": distance,
                "metadata": row.get("metadata") or {},
            }
        )
    return results



def embed_pending_domain_kb_documents(source_scenic_id: str, *, limit_docs: int = 20) -> dict[str, Any]:
    """Embed pending/failed domain-kb documents for one source scenic id.

    This is safe to run as a FastAPI BackgroundTask. It embeds document by
    document so one bad file does not block the rest.
    """
    with ai_session_scope() as db:
        rows = db.execute(
            text(
                """
                select source_id
                from knowledge_chunks
                where source_scenic_id=:source_scenic_id
                  and source_type='domain_kb'
                  and coalesce(metadata->>'embedding_status', 'pending') in ('pending', 'failed')
                group by source_id
                order by min(created_at) asc
                limit :limit_docs
                """
            ),
            {"source_scenic_id": source_scenic_id, "limit_docs": limit_docs},
        ).fetchall()
    results = []
    for row in rows:
        source_id = row[0]
        results.append({"source_id": source_id, **embed_domain_kb_document(source_scenic_id, source_id)})
    return {"source_scenic_id": source_scenic_id, "documents": results, "total": len(results)}
