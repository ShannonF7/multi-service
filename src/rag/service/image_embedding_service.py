"""前置知识库图片 Qwen-VL-Embedding 持久化服务。

本模块从 node_assets 读取图片路径，调用本机 7012 Qwen 服务得到 2048 维
向量，写入 domain_kb_image_embeddings；不负责 OCR，也不负责图片检索排序。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import tempfile

from sqlalchemy import text
from src.rag.dependencies import ai_session_scope

QWEN_IMAGE_EMBEDDING_URL = os.getenv("QWEN_IMAGE_EMBEDDING_URL", "http://127.0.0.1:7012").rstrip("/")
MODEL_NAME = "qwen3-vl-embedding-2b"
VECTOR_DIM = 2048
IMAGE_CACHE_ROOT = Path(
    os.getenv(
        "QWEN_VL_EMBEDDING_CACHE_ROOT",
        "/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/pgvector_optimized/data/domain_kb",
    )
).resolve()


def _resolve_image_url(value: Any) -> str:
    """将 A 端保存的本机媒体 URL 转为 B 端可访问地址。

    输入：node_assets.url/source_url 中的本地路径或 URL。
    输出：可供 B 端下载的 URL；非本机 URL 原样保留。
    该函数只做地址转换，不发起网络请求。
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost"}:
        base = os.getenv("RAG_IMAGE_MEDIA_BASE_URL", "https://ai.smartoptiks.cn").rstrip("/")
        return base + (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
    return raw


def _download_image_for_embedding(value: Any) -> tuple[str, bool]:
    """准备模型可读取的图片路径，并标记是否需要清理临时文件。

    输入：本地图片路径或 URL。输出为 (path, temporary)；临时文件由调用方
    在 embedding 请求完成后删除，避免批处理在磁盘留下无法追踪的文件。
    """
    raw = str(value or "").strip()
    local = Path(raw).expanduser()
    if local.is_file():
        return str(local.resolve()), False
    url = _resolve_image_url(raw)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"image source is not a readable file or URL: {raw}")
    IMAGE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = Path(parsed.path).suffix[:10] or ".img"
    handle = tempfile.NamedTemporaryFile(
        prefix="growth-image-",
        suffix=suffix,
        dir=str(IMAGE_CACHE_ROOT),
        delete=False,
    )
    temp_path = handle.name
    try:
        request = Request(url, headers={"User-Agent": "semantic-growth-image-embedding/1.0"})
        with urlopen(request, timeout=float(os.getenv("QWEN_VL_IMAGE_DOWNLOAD_TIMEOUT", "30"))) as response:
            handle.write(response.read())
    except Exception:
        handle.close()
        Path(temp_path).unlink(missing_ok=True)
        raise
    handle.close()
    return temp_path, True


def _pgvector(values: list[float]) -> str:
    return "[" + ",".join(f"{float(item):.8f}" for item in values) + "]"


def ensure_domain_image_embedding_schema() -> None:
    """按需创建图片向量表和领域索引，保证服务可幂等启动。"""
    with ai_session_scope() as db:
        db.execute(text(f"""
            create table if not exists domain_kb_image_embeddings (
                id bigserial primary key,
                scenic_id integer not null,
                source_scenic_id varchar(255) not null,
                asset_id integer not null references node_assets(id) on delete cascade,
                source_node_id varchar(255) not null,
                embedding vector({VECTOR_DIM}) not null,
                model_name varchar(255) not null,
                sync_version varchar(64) not null,
                created_at timestamptz not null default now(),
                unique(asset_id, model_name)
            )
        """))
        db.execute(text("create index if not exists idx_domain_kb_image_embeddings_scope on domain_kb_image_embeddings(source_scenic_id)"))


def _embed_paths(paths: list[str]) -> dict[str, Any]:
    """调用 7012 Qwen 服务，把一批本地图片路径转换为向量。"""
    body = json.dumps({"image_paths": paths}, ensure_ascii=False).encode("utf-8")
    request = Request(QWEN_IMAGE_EMBEDDING_URL + "/embed-images", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=float(os.getenv("QWEN_IMAGE_EMBEDDING_TIMEOUT", "180"))) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def _embed_texts(texts: list[str]) -> dict[str, Any]:
    """调用 7012 Qwen 服务，把文本查询转换为同空间的 2048 维向量。"""
    body = json.dumps({"texts": texts}, ensure_ascii=False).encode("utf-8")
    request = Request(QWEN_IMAGE_EMBEDDING_URL + "/embed-texts", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=float(os.getenv("QWEN_IMAGE_EMBEDDING_TIMEOUT", "180"))) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def embed_domain_kb_images(source_scenic_id: str, source_id: str, *, batch_size: int = 8) -> dict[str, Any]:
    """处理一个上传文档关联的全部图片 asset，并幂等写入图片向量。"""
    ensure_domain_image_embedding_schema()
    with ai_session_scope() as db:
        rows = db.execute(text("""
            select na.id, na.scenic_id, na.source_node_id, na.url
            from node_assets na
            where na.source_scenic_id=:sid and na.source_asset_id like :prefix and na.asset_type='image'
              and coalesce(na.url, '') <> ''
              and not exists (select 1 from domain_kb_image_embeddings de where de.asset_id=na.id and de.model_name=:model_name)
            order by na.id
        """), {"sid": source_scenic_id, "prefix": source_id + ':%', "model_name": MODEL_NAME}).mappings().all()
    if not rows:
        return {"status": "skipped", "embedded": 0, "model_name": MODEL_NAME}
    embedded = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        payload = _embed_paths([str(row["url"]) for row in batch])
        vectors = payload.get("vectors") or []
        if len(vectors) != len(batch):
            raise RuntimeError("Qwen image embedding count mismatch")
        with ai_session_scope() as db:
            for row, vector in zip(batch, vectors):
                if len(vector) != VECTOR_DIM:
                    raise RuntimeError(f"unexpected Qwen image vector dimension: {len(vector)}")
                db.execute(text("""
                    insert into domain_kb_image_embeddings
                        (scenic_id, source_scenic_id, asset_id, source_node_id, embedding, model_name, sync_version)
                    values (:scenic_id, :sid, :asset_id, :source_node_id, cast(:embedding as vector), :model_name, 'domain-kb-image-v1')
                    on conflict (asset_id, model_name) do nothing
                """), {"scenic_id": int(row["scenic_id"]), "sid": source_scenic_id, "asset_id": int(row["id"]), "source_node_id": row["source_node_id"] or '__domain__', "embedding": _pgvector(vector), "model_name": MODEL_NAME})
                embedded += 1
    return {"status": "done", "embedded": embedded, "model_name": MODEL_NAME, "dimension": VECTOR_DIM}


def search_domain_kb_image_embeddings(source_scenic_id: str, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """用 Qwen 文本向量召回图片 asset，并返回可展示的图片证据。

    输入为领域编号和用户文本查询；先调用 7012 `/embed-texts` 得到同空间
    的 2048 维查询向量，再从 `domain_kb_image_embeddings` 做 pgvector 近邻
    查询。输出保留 asset、图片路径、OCR/标题和距离，供 `search_domain_kb`
    与前端证据展示使用；本函数只读数据库，不修改生产数据。
    """
    if not str(query or "").strip():
        return []
    payload = _embed_texts([str(query)])
    vectors = payload.get("vectors") or []
    if len(vectors) != 1 or len(vectors[0]) != VECTOR_DIM:
        raise RuntimeError("Qwen text embedding dimension mismatch")
    vector = vectors[0]
    return _search_domain_kb_image_vectors(source_scenic_id, vector, limit=limit)


def search_domain_kb_image_by_path(source_scenic_id: str, image_path: str, *, limit: int = 5) -> list[dict[str, Any]]:
    """用上传图片本身作为查询，召回同领域中最相近的图片证据。

    输入图片路径必须由 API 临时保存到 7012 服务允许的目录；输出格式与文本
    查询的图片证据一致。调用方负责在请求结束后删除临时文件，本函数不写库。
    """
    if not str(image_path or "").strip():
        return []
    payload = _embed_paths([str(image_path)])
    vectors = payload.get("vectors") or []
    if len(vectors) != 1 or len(vectors[0]) != VECTOR_DIM:
        raise RuntimeError("Qwen image query embedding dimension mismatch")
    return _search_domain_kb_image_vectors(source_scenic_id, vectors[0], limit=limit)


def _search_domain_kb_image_vectors(source_scenic_id: str, vector: list[float], *, limit: int) -> list[dict[str, Any]]:
    """按已生成的图片向量执行只读 pgvector 查询，统一组装证据字段。"""
    with ai_session_scope() as db:
        rows = db.execute(text("""
            select de.asset_id, de.source_node_id, na.url, na.title, na.caption, na.ocr_text,
                   na.metadata, (de.embedding <-> cast(:embedding as vector)) as distance
            from domain_kb_image_embeddings de
            join node_assets na on na.id=de.asset_id
            where de.source_scenic_id=:sid and de.model_name=:model_name
            order by de.embedding <-> cast(:embedding as vector)
            limit :limit
        """), {"sid": source_scenic_id, "model_name": MODEL_NAME, "embedding": _pgvector(vector), "limit": int(limit)}).mappings().all()
    results = []
    for row in rows:
        distance = float(row.get("distance") or 0.0)
        results.append({
            "title": row.get("title") or "图片证据",
            "content": row.get("ocr_text") or row.get("caption") or row.get("title") or "图片证据",
            "quote": row.get("ocr_text") or row.get("caption") or "图片证据",
            "source": row.get("title") or "前置知识库图片",
            "source_type": "domain_kb_image_vector",
            "source_url": row.get("url"),
            "source_doc_id": None,
            "asset_id": int(row["asset_id"]),
            "score": max(0.0, 1.0 - distance / 2.0),
            "image_score": max(0.0, 1.0 - distance / 2.0),
            "distance": distance,
            "metadata": row.get("metadata") or {},
        })
    return results


def embed_scenic_images(
    source_scenic_id: str,
    *,
    batch_size: int = 8,
    max_images: int = 2000,
) -> dict[str, Any]:
    """按景区批量补齐 A 端同步图片的 Qwen 向量。

    输入：景区来源 ID、每批图片数和本次最大图片数。
    输出：处理数量、失败数量、模型版本和失败摘要。函数只写入
    domain_kb_image_embeddings，不修改节点、候选或证据状态；重复调用会跳过
    已存在同模型向量的资产。
    """
    ensure_domain_image_embedding_schema()
    with ai_session_scope() as db:
        rows = db.execute(text("""
            select na.id, na.scenic_id, na.source_scenic_id, na.source_node_id,
                   coalesce(na.url, na.source_url) as image_url
            from node_assets na
            where na.source_scenic_id=:sid
              and na.asset_type='image'
              and coalesce(na.url, na.source_url, '') <> ''
              and not exists (
                  select 1 from domain_kb_image_embeddings de
                  where de.asset_id=na.id and de.model_name=:model_name
              )
            order by na.id
            limit :max_images
        """), {
            "sid": str(source_scenic_id),
            "model_name": MODEL_NAME,
            "max_images": max(0, min(int(max_images), 2000)),
        }).mappings().all()
    if not rows:
        return {"status": "skipped", "embedded": 0, "failed": 0, "model_name": MODEL_NAME}
    size = max(1, min(int(batch_size), 32))
    embedded = 0
    failures: list[dict[str, Any]] = []
    for start in range(0, len(rows), size):
        batch = rows[start : start + size]
        prepared: list[tuple[dict[str, Any], str, bool]] = []
        for row in batch:
            try:
                path, temporary = _download_image_for_embedding(row["image_url"])
                prepared.append((dict(row), path, temporary))
            except Exception as exc:
                failures.append({"asset_id": int(row["id"]), "error": str(exc)[:300]})
        try:
            if not prepared:
                continue
            payload = _embed_paths([item[1] for item in prepared])
            vectors = payload.get("vectors") or []
            if len(vectors) != len(prepared):
                raise RuntimeError("Qwen image embedding count mismatch")
            with ai_session_scope() as db:
                for (row, _path, _temporary), vector in zip(prepared, vectors):
                    if len(vector) != VECTOR_DIM:
                        raise RuntimeError(f"unexpected Qwen image vector dimension: {len(vector)}")
                    db.execute(text("""
                        insert into domain_kb_image_embeddings
                            (scenic_id, source_scenic_id, asset_id, source_node_id,
                             embedding, model_name, sync_version)
                        values (:scenic_id, :sid, :asset_id, :source_node_id,
                                cast(:embedding as vector), :model_name,
                                'scenic-image-batch-v1')
                        on conflict (asset_id, model_name) do nothing
                    """), {
                        "scenic_id": int(row["scenic_id"]),
                        "sid": str(source_scenic_id),
                        "asset_id": int(row["id"]),
                        "source_node_id": row.get("source_node_id") or "__domain__",
                        "embedding": _pgvector(vector),
                        "model_name": MODEL_NAME,
                    })
                    embedded += 1
        except Exception as exc:
            failures.extend({"asset_id": int(row["id"]), "error": str(exc)[:300]} for row, _path, _temporary in prepared)
        finally:
            for _row, path, temporary in prepared:
                if temporary:
                    Path(path).unlink(missing_ok=True)
    return {
        "status": "done" if embedded or not failures else "failed",
        "embedded": embedded,
        "failed": len(failures),
        "failures": failures[:20],
        "model_name": MODEL_NAME,
        "dimension": VECTOR_DIM,
    }
