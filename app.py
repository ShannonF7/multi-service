import os
import json
import time
import atexit
import logging
import asyncio
import signal
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
import psycopg2
import psycopg2.pool
import psycopg2.extras
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv

load_dotenv()

PG_CONFIG = {
    "host": "localhost",
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

BGE_MODEL_PATH = "/home/zhangbi/Zhangbi_Traveler/LLM_model/Model_api/checkpoints/bge-large-zh-v1.5/"
EMBED_DIM = 1024

# Thread pool and DB pool sizes
THREAD_WORKERS = 8
DB_MINCONN = 3
DB_MAXCONN = 20

# Search params
KW_MAX_RESULTS = 50
KW_DISTANCE_THRESHOLD = 1.5
GLOBAL_MAX_RESULTS = 100
DEFAULT_TOPK = 10

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pg_hybrid_search")

# ----------------------
# Initialize FastAPI
# ----------------------
app = FastAPI(title="Hybrid RAG Search (pgvector + BGE)")

# ----------------------
# DB connection pool
# ----------------------
pg_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None

def init_pg_pool():
    global pg_pool
    if pg_pool is None:
        pg_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=DB_MINCONN,
            maxconn=DB_MAXCONN,
            **PG_CONFIG
        )
        logger.info("Initialized Postgres connection pool.")

# ----------------------
# ThreadPool for blocking tasks
# ----------------------
executor = ThreadPoolExecutor(max_workers=THREAD_WORKERS)

# ----------------------
# Load BGE model (single global instance)
# ----------------------
_model: Optional[BGEM3FlagModel] = None

def load_bge_model():
    global _model
    if _model is not None:
        return
    logger.info("Loading BGE model from %s", BGE_MODEL_PATH)
    # prefer GPU if available; the FlagEmbedding wrapper handles internals
    _model = BGEM3FlagModel(BGE_MODEL_PATH, device="cuda", use_fp16=True)
    logger.info("BGE model loaded.")

def close_bge_model():
    global _model
    if _model is None:
        return
    try:
        # FlagEmbedding may provide stop_pool / close
        if hasattr(_model, "stop_pool"):
            _model.stop_pool()
        if hasattr(_model, "close"):
            _model.close()
    except Exception as e:
        logger.warning("Error closing BGE model pool: %s", e)
    finally:
        _model = None
        logger.info("BGE model closed.")

atexit.register(close_bge_model)

# ----------------------
# Utility: embed and convert for pgvector
# ----------------------
def embed_texts(texts: List[str], batch_size: int = 4) -> List[List[float]]:
    """
    Embed texts using the loaded BGE model.
    Returns list of float lists (normalized).
    """
    load_bge_model()
    global _model
    if _model is None:
        raise RuntimeError("BGE model not loaded")

    embeddings: List[List[float]] = []
    # model.encode returns dict with "dense_vecs" (numpy array)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        outputs = _model.encode(batch)  # may return dict
        # outputs["dense_vecs"] may be numpy array shape (N, D)
        vecs = None
        if isinstance(outputs, dict) and "dense_vecs" in outputs:
            vecs = outputs["dense_vecs"]
        elif isinstance(outputs, (list, tuple)):
            # occasional wrapper; try first element
            first = outputs[0]
            if isinstance(first, dict) and "dense_vecs" in first:
                vecs = first["dense_vecs"]
            else:
                # last fallback: treat outputs as array-like
                vecs = np.array(outputs)
        else:
            vecs = np.array(outputs)

        vecs = np.asarray(vecs, dtype=np.float32)
        # l2 normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        for v in vecs:
            embeddings.append(v.tolist())
    return embeddings

def embed_text(text: str) -> List[float]:
    return embed_texts([text], batch_size=1)[0]

def to_pgvector(vec: List[float]) -> str:
    """Convert python list to pgvector literal string like '[0.012345, ...]'"""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

# ----------------------
# DB search functions
# ----------------------
def run_keyword_search_pg_blocking(query_embedding: List[float], n_results: int, conn) -> List[Dict[str, Any]]:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    results: List[Dict[str, Any]] = []
    try:
        emb_str = to_pgvector(query_embedding)
        limit = min(n_results, KW_MAX_RESULTS)
        sql = """
            SELECT id, keyword, refs, (embedding <-> %s::vector) AS distance
            FROM keywords
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """
        cur.execute(sql, (emb_str, emb_str, limit))
        kw_rows = cur.fetchall()

        doc_ids = set()
        keyword_matches = []
        for row in kw_rows:
            distance = row.get("distance")
            if distance is None:
                continue
            if float(distance) > KW_DISTANCE_THRESHOLD:
                continue
            refs = row.get("refs") or []
            if isinstance(refs, str):
                try:
                    refs = json.loads(refs)
                except Exception:
                    refs = []
            for ref in refs:
                if isinstance(ref, dict) and "doc_id" in ref:
                    doc_ids.add(ref["doc_id"])
                    keyword_matches.append({
                        "doc_id": ref["doc_id"],
                        "distance": float(distance),
                        "similarity": 1.0 / (1.0 + float(distance))
                    })

        if not doc_ids:
            return []

        cur.execute("SELECT id, content, metadata FROM texts WHERE id = ANY(%s)", (list(doc_ids),))
        doc_rows = cur.fetchall()
        doc_map = {r["id"]: r for r in doc_rows}
        for match in keyword_matches:
            row = doc_map.get(match["doc_id"])
            if not row:
                continue
            content = (row.get("content") or "").strip()
            if (not content or len(content) < 10 or content in {"整理内容如下：", "暂无描述", "待补充", "敬请期待"}):
                continue
            metadata = row.get("metadata") or {}
            position = metadata.get("position", "")
            results.append({
                "id": match["doc_id"],
                "content": content,
                "position": position,
                "metadata": metadata,
                "similarity": match["similarity"] * 1.15,
                "distance": match["distance"],
                "match_type": "keyword"
            })
        return results
    finally:
        cur.close()

def enhanced_doc_search_pg_blocking(query: str, n_results: int, conn) -> List[Dict[str, Any]]:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        query_text = f"query: {query.strip()}" if not query.startswith("query:") else query
        # embed
        q_emb = embed_text(query_text)
        emb_str = to_pgvector(q_emb)
        limit = min(n_results * 2, GLOBAL_MAX_RESULTS)
        sql = """
            SELECT id, content, metadata, (embedding <-> %s::vector) AS distance
            FROM texts
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """
        cur.execute(sql, (emb_str, emb_str, limit))
        rows = cur.fetchall()
        results = []
        seen_contents = set()
        for row in rows:
            if len(results) >= n_results:
                break
            content = (row.get("content") or "").strip()
            distance = row.get("distance")
            if distance is None:
                continue
            metadata = row.get("metadata") or {}
            if (not content or content in seen_contents or content in {"整理内容如下：", "暂无描述", "待补充", "敬请期待"} or len(content) < 10):
                continue
            seen_contents.add(content)
            results.append({
                "id": row.get("id"),
                "content": content,
                "position": metadata.get("position", ""),
                "metadata": metadata,
                "similarity": 1.0 / (1.0 + float(distance)),
                "distance": float(distance),
                "match_type": "global"
            })
        return results
    finally:
        cur.close()

# ----------------------
# Async wrapper that uses executor
# ----------------------
async def run_keyword_search_pg(query_embedding: List[float], n_results: int) -> List[Dict[str, Any]]:
    conn = pg_pool.getconn()
    try:
        return await asyncio.get_event_loop().run_in_executor(executor, run_keyword_search_pg_blocking, query_embedding, n_results, conn)
    finally:
        pg_pool.putconn(conn)

async def enhanced_doc_search_pg(query: str, n_results: int) -> List[Dict[str, Any]]:
    conn = pg_pool.getconn()
    try:
        return await asyncio.get_event_loop().run_in_executor(executor, enhanced_doc_search_pg_blocking, query, n_results, conn)
    finally:
        pg_pool.putconn(conn)

# ----------------------
# Hybrid search endpoint logic
# ----------------------
class SearchResponseItem(BaseModel):
    id: str
    content: str
    position: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = {}
    similarity: float
    distance: float
    match_type: str

# return ids
class SearchResponse(BaseModel):
    results: List[SearchResponseItem]
    source_ids: List[str]   

@app.on_event("startup")
async def on_startup():
    init_pg_pool()
    load_bge_model()
    logger.info("App startup complete.")

@app.on_event("shutdown")
async def on_shutdown():
    # close pools gracefully
    if pg_pool:
        try:
            pg_pool.closeall()
        except Exception as e:
            logger.warning("Error closing PG pool: %s", e)
    # shutdown executor
    try:
        executor.shutdown(wait=True)
    except Exception as e:
        logger.warning("Error shutting down thread pool: %s", e)
    # close model
    close_bge_model()
    logger.info("App shutdown complete.")

@app.get("/health")
async def health():
    return {"status": "ok", "pid": os.getpid(), "mem_percent": psutil.virtual_memory().percent}

@app.get("/search", response_model=List[SearchResponseItem])
async def search_api(q: str = Query(..., min_length=1), topk: int = Query(10, ge=1, le=50)):
    """
    Hybrid search API:
    1. Embed query using BGE
    2. Run keyword search and global vector search in parallel
    3. Merge & deduplicate results, prioritizing keyword matches
    4. Return top-k results
    """
    start = time.time()
    # resource check
    memp = psutil.virtual_memory().percent
    if memp > 95:
        logger.warning("High memory usage: %s%%", memp)

    # 1. embedding (sync call) -> run in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    try:
        query_embedding = await loop.run_in_executor(executor, embed_text, q)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise HTTPException(status_code=500, detail="Embedding failed")

    # 2. parallel searches
    kw_task = asyncio.create_task(run_keyword_search_pg(query_embedding, min(15, KW_MAX_RESULTS)))
    global_task = asyncio.create_task(enhanced_doc_search_pg(q, int(topk * 1.5)))

    try:
        keyword_results, global_results = await asyncio.wait_for(asyncio.gather(kw_task, global_task), timeout=30.0)
    except asyncio.TimeoutError:
        logger.error("Search tasks timed out")
        raise HTTPException(status_code=504, detail="Search timeout")
    except Exception as e:
        logger.error("Search tasks error: %s", e)
        raise HTTPException(status_code=500, detail="Search internal error")

    # 3. Merge & dedupe preserving your original priority: keyword first
    combined = []
    seen_ids = set()
    seen_contents = set()
    for r in keyword_results:
        if len(combined) >= topk:
            break
        chash = hash(r["content"].strip().lower()) if r.get("content") else None
        if r["id"] in seen_ids or (chash is not None and chash in seen_contents):
            continue
        combined.append(r)
        seen_ids.add(r["id"])
        if chash is not None:
            seen_contents.add(chash)

    for r in global_results:
        if len(combined) >= topk:
            break
        chash = hash(r["content"].strip().lower()) if r.get("content") else None
        if r["id"] in seen_ids or (chash is not None and chash in seen_contents):
            continue
        combined.append(r)
        seen_ids.add(r["id"])
        if chash is not None:
            seen_contents.add(chash)

    combined.sort(key=lambda x: -x["similarity"])
    final = combined[:topk]
    elapsed = time.time() - start
    logger.info("Search q=%s topk=%d results=%d time=%.3fs", q, topk, len(final), elapsed)
    return final

# Admin endpoint: create vector indexes (careful: long running)
@app.post("/admin/reindex")
async def reindex_vectors():
    """
    Create vector indexes (HNSW). Run after bulk-loading data.
    Note: requires appropriate privileges. Long-running for large datasets.
    """
    conn = pg_pool.getconn()
    try:
        cur = conn.cursor()
        # adjust operators to desired distance type (vector_l2_ops / vector_cosine_ops)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_text_embedding ON texts USING hnsw (embedding vector_l2_ops)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kw_embedding ON keywords USING hnsw (embedding vector_l2_ops)")
        conn.commit()
        return {"status": "ok", "msg": "indexes created"}
    except Exception as e:
        logger.error("Reindex failed: %s", e)
        conn.rollback()
        raise HTTPException(status_code=500, detail="Reindex failed")
    finally:
        pg_pool.putconn(conn)


if __name__ == "__main__":
    import uvicorn
    init_pg_pool()
    load_bge_model()
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1)