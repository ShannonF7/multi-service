"""Read-only audit for synced node image assets and embeddings.

Usage:
    PYTHONPATH=. python -m src.multimodal.asset_audit --source-scenic-id 4 \
      --output data/multimodal_eval/scenic_4_asset_audit.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.rag.dependencies import get_ai_engine


def _scalar(connection, sql: str, params: dict[str, Any]) -> int:
    row = connection.execute(text(sql), params).fetchone()
    return int(row[0] or 0) if row else 0


def audit_assets(source_scenic_id: str | None = None, min_images_per_node: int = 2) -> dict[str, Any]:
    asset_where = ""
    joined_asset_where = ""
    params: dict[str, Any] = {}
    if source_scenic_id:
        asset_where = " where source_scenic_id = :source_scenic_id"
        joined_asset_where = " where a.source_scenic_id = :source_scenic_id"
        params["source_scenic_id"] = str(source_scenic_id)

    with get_ai_engine().connect() as connection:
        node_assets = _scalar(connection, f"select count(*) from node_assets{asset_where}", params)
        image_embeddings = _scalar(
            connection,
            """
            select count(*)
            from image_embeddings e
            join node_assets a on a.id = e.asset_id
            """ + joined_asset_where,
            params,
        )
        clip_embeddings = _scalar(
            connection,
            """
            select count(*)
            from clip_image_embeddings e
            join node_assets a on a.id = e.asset_id
            """ + joined_asset_where,
            params,
        )
        cover_assets = _scalar(connection, f"select count(*) from node_assets{asset_where + (' and' if asset_where else ' where')} is_cover = true", params)
        missing_url = _scalar(connection, f"select count(*) from node_assets{asset_where + (' and' if asset_where else ' where')} coalesce(url, '') = ''", params)
        by_role = connection.execute(
            text(
                f"""
                select coalesce(role, '') as role, count(*) as total
                from node_assets
                {asset_where}
                group by coalesce(role, '')
                order by total desc
                limit 20
                """
            ),
            params,
        ).mappings().all()
        node_rows = connection.execute(
            text(
                f"""
                select a.source_node_id, coalesce(n.node_name, '') as node_name,
                       coalesce(n.node_type, 'Other') as node_type,
                       coalesce(n.parent_source_node_id, '') as parent_node_id,
                       count(*) as image_count,
                       count(e.id) as simclr_count
                from node_assets a
                left join semantic_nodes n on n.source_scenic_id = a.source_scenic_id
                  and n.source_node_id = a.source_node_id
                left join image_embeddings e on e.asset_id = a.id and e.model_name = 'legacy_simclr_128'
                {asset_where.replace('where', 'where a.') if asset_where else ''}
                group by a.source_node_id, n.node_name, n.node_type, n.parent_source_node_id
                order by count(*) desc, a.source_node_id asc
                """
            ),
            params,
        ).mappings().all()

    image_count_distribution = Counter()
    type_distribution = Counter()
    eligible_nodes = 0
    eligible_images = 0
    node_image_counts = []
    for row in node_rows:
        image_count = int(row["image_count"] or 0)
        node_type = str(row["node_type"] or "Other")
        image_count_distribution[str(image_count)] += 1
        type_distribution[node_type] += image_count
        if image_count >= min_images_per_node:
            eligible_nodes += 1
            eligible_images += image_count
        node_image_counts.append(dict(row))

    return {
        "source_scenic_id": source_scenic_id,
        "node_assets": node_assets,
        "image_embeddings_128": image_embeddings,
        "clip_image_embeddings_512": clip_embeddings,
        "missing_simclr_embeddings": max(node_assets - image_embeddings, 0),
        "missing_clip_embeddings": max(node_assets - clip_embeddings, 0),
        "cover_assets": cover_assets,
        "missing_url": missing_url,
        "eligible_nodes_for_e0": eligible_nodes,
        "eligible_images_for_e0": eligible_images,
        "min_images_per_node": min_images_per_node,
        "image_count_distribution": dict(sorted(image_count_distribution.items(), key=lambda item: int(item[0]))),
        "node_type_image_distribution": dict(type_distribution),
        "by_role": [dict(row) for row in by_role],
        "node_image_counts": node_image_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit synced node image assets.")
    parser.add_argument("--source-scenic-id", default=None)
    parser.add_argument("--min-images-per-node", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = audit_assets(args.source_scenic_id, args.min_images_per_node)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
