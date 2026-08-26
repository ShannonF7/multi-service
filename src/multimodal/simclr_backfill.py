"""Backfill E0 SimCLR image embeddings for synced node assets.

Default behavior is conservative: dry-run is enabled unless --write is passed.

Examples:
    PYTHONPATH=. python -m src.multimodal.simclr_backfill --source-scenic-id 4 --limit 5
    PYTHONPATH=. python -m src.multimodal.simclr_backfill --dataset data/multimodal_eval/scenic_4_e0_pilot_v1.jsonl --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from PIL import Image
from sqlalchemy import bindparam, text

from src.rag.dependencies import get_ai_engine
from src.multimodal.encoders import create_image_encoder

MODEL_NAME = "legacy_simclr_128"
EMBEDDING_DIM = 128
DEFAULT_MEDIA_BASE_URL = os.getenv("A_MEDIA_BASE_URL", "http://ai.smartoptiks.cn")
CACHE_DIR = Path(os.getenv("MULTIMODAL_IMAGE_CACHE", "/tmp/zhangbi_multimodal_images"))


def to_pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(item):.8f}" for item in vector) + "]"


def resolve_asset_url(url: str, media_base_url: str) -> str:
    value = (url or "").strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return urljoin(media_base_url.rstrip("/") + "/", value.lstrip("/"))
    return value


def fetch_image_to_cache(url: str, media_base_url: str, timeout: int = 20) -> Path:
    resolved = resolve_asset_url(url, media_base_url)
    if not resolved:
        raise ValueError("empty asset url")
    if os.path.exists(resolved):
        return Path(resolved)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(resolved.split("?", 1)[0]).suffix or ".img"
    cache_name = hashlib.sha256(resolved.encode("utf-8")).hexdigest() + suffix
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    response = requests.get(resolved, timeout=timeout)
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return cache_path


def load_dataset_asset_ids(dataset: str | None) -> list[int]:
    if not dataset:
        return []
    ids: list[int] = []
    with Path(dataset).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("usable", True) and row.get("asset_id") is not None:
                value = int(row["asset_id"])
                if value not in ids:
                    ids.append(value)
    return ids


def load_candidate_assets(
    connection,
    source_scenic_id: str | None,
    limit: int,
    dataset_asset_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"model_name": MODEL_NAME, "limit": int(limit)}
    filters = ["coalesce(a.asset_type, 'image') = 'image'", "coalesce(a.url, '') <> ''"]
    if source_scenic_id:
        filters.append("a.source_scenic_id = :source_scenic_id")
        params["source_scenic_id"] = str(source_scenic_id)
    stmt = text(
        f"""
        select a.id, a.scenic_id, a.source_scenic_id, a.source_node_id,
               a.source_asset_id, a.url, a.file_hash
        from node_assets a
        where {' and '.join(filters)}
          and not exists (
              select 1 from image_embeddings e
              where e.asset_id = a.id and e.model_name = :model_name
          )
        order by a.is_cover desc, a.id asc
        limit :limit
        """
    )
    if dataset_asset_ids:
        filters.append("a.id in :dataset_asset_ids")
        params["dataset_asset_ids"] = dataset_asset_ids
        stmt = text(
            f"""
            select a.id, a.scenic_id, a.source_scenic_id, a.source_node_id,
                   a.source_asset_id, a.url, a.file_hash
            from node_assets a
            where {' and '.join(filters)}
              and not exists (
                  select 1 from image_embeddings e
                  where e.asset_id = a.id and e.model_name = :model_name
              )
            order by a.is_cover desc, a.id asc
            limit :limit
            """
        ).bindparams(bindparam("dataset_asset_ids", expanding=True))
    rows = connection.execute(stmt, params).mappings().all()
    return [dict(row) for row in rows]


def insert_embedding(connection, asset: dict[str, Any], vector: list[float], sync_version: str) -> None:
    if len(vector) != EMBEDDING_DIM:
        raise RuntimeError(f"embedding dim={len(vector)}, expected {EMBEDDING_DIM}")
    connection.execute(
        text(
            """
            insert into image_embeddings (
                scenic_id, asset_id, source_node_id, embedding, model_name,
                sync_version, created_at
            ) values (
                :scenic_id, :asset_id, :source_node_id,
                cast(:embedding as vector), :model_name, :sync_version, now()
            )
            """
        ),
        {
            "scenic_id": int(asset["scenic_id"]),
            "asset_id": int(asset["id"]),
            "source_node_id": str(asset["source_node_id"]),
            "embedding": to_pgvector_literal(vector),
            "model_name": MODEL_NAME,
            "sync_version": sync_version,
        },
    )


def backfill(
    source_scenic_id: str | None,
    limit: int,
    write: bool,
    media_base_url: str,
    dataset: str | None = None,
) -> dict[str, Any]:
    dataset_asset_ids = load_dataset_asset_ids(dataset)
    effective_limit = len(dataset_asset_ids) if dataset_asset_ids else limit
    summary: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "source_scenic_id": source_scenic_id,
        "dataset": dataset,
        "dataset_assets": len(dataset_asset_ids),
        "limit": effective_limit,
        "write": write,
        "processed": 0,
        "inserted": 0,
        "failed": 0,
        "failures": [],
    }
    with get_ai_engine().begin() as connection:
        assets = load_candidate_assets(connection, source_scenic_id, effective_limit, dataset_asset_ids)
        summary["candidate_assets"] = len(assets)
        if not write:
            summary["sample_asset_ids"] = [item["id"] for item in assets[:10]]
            return summary

        encoder = create_image_encoder(MODEL_NAME)
        for asset in assets:
            summary["processed"] += 1
            try:
                image_path = fetch_image_to_cache(str(asset.get("url") or ""), media_base_url)
                with Image.open(image_path).convert("RGB") as image:
                    vector = encoder.encode_image(image)
                insert_embedding(connection, asset, vector, sync_version="multimodal-e0")
                summary["inserted"] += 1
            except Exception as exc:
                summary["failed"] += 1
                summary["failures"].append({"asset_id": asset.get("id"), "error": str(exc)[:300]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill legacy SimCLR image embeddings.")
    parser.add_argument("--source-scenic-id", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--media-base-url", default=DEFAULT_MEDIA_BASE_URL)
    args = parser.parse_args()
    result = backfill(args.source_scenic_id, max(1, args.limit), args.write, args.media_base_url, args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
