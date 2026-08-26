"""E0 SimCLR image-image and image-node search.

Examples:
    PYTHONPATH=. python -m src.multimodal.simclr_search --asset-id 1158 --top-k 10
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import get_ai_engine
from src.multimodal.simclr_backfill import MODEL_NAME, to_pgvector_literal


def get_asset_embedding(connection, asset_id: int) -> list[float]:
    row = connection.execute(
        text(
            """
            select embedding
            from image_embeddings
            where asset_id = :asset_id and model_name = :model_name
            order by id desc
            limit 1
            """
        ),
        {"asset_id": int(asset_id), "model_name": MODEL_NAME},
    ).fetchone()
    if not row:
        raise LookupError(f"no {MODEL_NAME} embedding for asset_id={asset_id}")
    value = row[0]
    if isinstance(value, str):
        return [float(item) for item in value.strip("[]").split(",") if item]
    return [float(item) for item in value]


def search_by_vector(
    connection,
    vector: list[float],
    source_scenic_id: str | None,
    top_k: int,
    exclude_asset_id: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "embedding": to_pgvector_literal(vector),
        "model_name": MODEL_NAME,
        "top_k": int(top_k),
    }
    filters = []
    if source_scenic_id:
        filters.append("a.source_scenic_id = :source_scenic_id")
        params["source_scenic_id"] = str(source_scenic_id)
    if exclude_asset_id is not None:
        filters.append("a.id <> :exclude_asset_id")
        params["exclude_asset_id"] = int(exclude_asset_id)
    where = " and " + " and ".join(filters) if filters else ""

    image_rows = connection.execute(
        text(
            f"""
            select a.id as asset_id, a.source_node_id, a.source_asset_id,
                   a.role, a.is_cover, a.url,
                   (e.embedding <-> cast(:embedding as vector)) as distance
            from image_embeddings e
            join node_assets a on a.id = e.asset_id
            where e.model_name = :model_name
              {where}
            order by e.embedding <-> cast(:embedding as vector) asc
            limit :top_k
            """
        ),
        params,
    ).mappings().all()

    node_rows = connection.execute(
        text(
            f"""
            select a.source_node_id,
                   min(e.embedding <-> cast(:embedding as vector)) as best_distance,
                   count(*) as matched_images
            from image_embeddings e
            join node_assets a on a.id = e.asset_id
            where e.model_name = :model_name
              {where}
            group by a.source_node_id
            order by min(e.embedding <-> cast(:embedding as vector)) asc
            limit :top_k
            """
        ),
        params,
    ).mappings().all()

    return {
        "model_name": MODEL_NAME,
        "image_results": [dict(row) for row in image_rows],
        "node_results": [dict(row) for row in node_rows],
    }


def search_by_asset(asset_id: int, source_scenic_id: str | None, top_k: int) -> dict[str, Any]:
    with get_ai_engine().connect() as connection:
        vector = get_asset_embedding(connection, asset_id)
        return search_by_vector(
            connection,
            vector,
            source_scenic_id=source_scenic_id,
            top_k=top_k,
            exclude_asset_id=asset_id,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search E0 SimCLR image embeddings.")
    parser.add_argument("--asset-id", type=int, required=True)
    parser.add_argument("--source-scenic-id", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    result = search_by_asset(args.asset_id, args.source_scenic_id, max(1, args.top_k))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
