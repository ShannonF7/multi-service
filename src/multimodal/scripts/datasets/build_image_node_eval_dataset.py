"""Build versioned E0 image-node evaluation dataset from synced node_assets."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from sqlalchemy import text

from src.rag.dependencies import get_ai_engine
from src.multimodal.schemas import ImageNodeEvalItem


def load_assets(source_scenic_id: str) -> list[dict]:
    with get_ai_engine().connect() as connection:
        rows = connection.execute(
            text(
                """
                select a.id as asset_id, a.source_node_id as node_id, a.url as image_url,
                       a.role as asset_role, a.is_cover, a.file_hash,
                       n.node_name, n.node_type, n.parent_source_node_id
                from node_assets a
                join semantic_nodes n on n.source_scenic_id = a.source_scenic_id
                  and n.source_node_id = a.source_node_id
                where a.source_scenic_id = :sid
                  and coalesce(a.asset_type, 'image') = 'image'
                  and coalesce(a.url, '') <> ''
                  and coalesce(n.node_name, '') <> ''
                order by a.source_node_id, a.is_cover desc, a.id asc
                """
            ),
            {"sid": str(source_scenic_id)},
        ).mappings().all()
    return [dict(row) for row in rows]


def load_duplicate_map(path: str | None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not path:
        return mapping
    p = Path(path)
    if not p.exists():
        return mapping
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            group_id = str(row.get("duplicate_group_id") or row.get("group_id") or "")
            for member in row.get("members") or []:
                if member.get("asset_id") is not None and group_id:
                    mapping[int(member["asset_id"])] = group_id
    return mapping


def split_node_assets(assets: list[dict], query_ratio: float, seed: int, duplicate_map: dict[int, str]) -> dict[int, str]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for asset in assets:
        group_id = duplicate_map.get(int(asset["asset_id"]), f"asset:{asset['asset_id']}")
        groups[group_id].append(asset)

    group_items = list(groups.items())
    rng.shuffle(group_items)
    query_group_count = max(1, round(len(group_items) * query_ratio))
    if len(group_items) - query_group_count < 1:
        query_group_count = max(1, len(group_items) - 1)
    query_groups = {group_id for group_id, _ in group_items[:query_group_count]}
    roles = {}
    for group_id, members in group_items:
        role = "query" if group_id in query_groups else "gallery"
        for member in members:
            roles[int(member["asset_id"])] = role
    return roles


def build_dataset(
    source_scenic_id: str,
    dataset_version: str,
    min_images_per_node: int,
    query_ratio: float,
    seed: int,
    output: str,
    duplicates: str | None,
    max_nodes: int | None,
    default_visual_label_status: str,
) -> dict:
    duplicate_map = load_duplicate_map(duplicates)
    assets = load_assets(source_scenic_id)
    by_node: dict[str, list[dict]] = defaultdict(list)
    for asset in assets:
        by_node[str(asset["node_id"])].append(asset)

    eligible = [(node_id, items) for node_id, items in by_node.items() if len(items) >= min_images_per_node]
    eligible.sort(key=lambda item: (-len(item[1]), item[0]))
    if max_nodes:
        eligible = eligible[:max_nodes]

    rows = []
    excluded = []
    for node_id, items in eligible:
        roles = split_node_assets(items, query_ratio, seed + int(node_id), duplicate_map)
        if "query" not in set(roles.values()) or "gallery" not in set(roles.values()):
            for item in items:
                excluded.append({"asset_id": item["asset_id"], "node_id": node_id, "reason": "split_failed"})
            continue
        for item in items:
            asset_id = int(item["asset_id"])
            rows.append(
                ImageNodeEvalItem(
                    dataset_version=dataset_version,
                    source_scenic_id=str(source_scenic_id),
                    asset_id=asset_id,
                    node_id=str(item["node_id"]),
                    node_name=str(item.get("node_name") or ""),
                    node_type=str(item.get("node_type") or "Other"),
                    parent_node_id=str(item.get("parent_source_node_id")) if item.get("parent_source_node_id") else None,
                    image_url=str(item.get("image_url") or ""),
                    visual_label_status=default_visual_label_status,  # weak visual label; manually refine only abnormal pilot rows.
                    duplicate_group_id=duplicate_map.get(asset_id),
                    role=roles[asset_id],
                    metadata={"asset_role": item.get("asset_role"), "is_cover": bool(item.get("is_cover"))},
                ).to_dict()
            )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    summary = {
        "dataset_version": dataset_version,
        "source_scenic_id": source_scenic_id,
        "total_source_assets": len(assets),
        "eligible_nodes": len(eligible),
        "items": len(rows),
        "query_items": sum(1 for row in rows if row["role"] == "query"),
        "gallery_items": sum(1 for row in rows if row["role"] == "gallery"),
        "excluded": len(excluded),
        "min_images_per_node": min_images_per_node,
        "query_ratio": query_ratio,
        "seed": seed,
        "default_visual_label_status": default_visual_label_status,
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if excluded:
        with out.with_name(out.stem + "_excluded.jsonl").open("w", encoding="utf-8") as f:
            for row in excluded:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E0 image-node eval dataset.")
    parser.add_argument("--source-scenic-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--min-images-per-node", type=int, default=2)
    parser.add_argument("--query-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duplicates", default=None)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--default-visual-label-status", default="direct")
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.source_scenic_id, args.dataset_version, args.min_images_per_node, args.query_ratio, args.seed, args.output, args.duplicates, args.max_nodes, args.default_visual_label_status), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
