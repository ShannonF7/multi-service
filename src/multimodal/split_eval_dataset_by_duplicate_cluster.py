"""Split image-node eval queries into validation/frozen sets by duplicate cluster.

Gallery rows are copied into both split files as the fixed retrieval index.
Only query rows are split. This matches the task: identify the existing node
for a new image query against a known gallery.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def cluster_id(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("duplicate_cluster_id") or row.get("duplicate_group_id") or f"asset:{row.get('asset_id')}")


def split_dataset(dataset: str, validation_output: str, frozen_output: str, summary_output: str, seed: int, validation_ratio: float) -> dict[str, Any]:
    rows = [row for row in read_jsonl(dataset) if row.get("usable", True) is not False]
    query_rows = [row for row in rows if row.get("role") == "query"]
    gallery_rows = [row for row in rows if row.get("role") == "gallery"]

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        clusters[cluster_id(row)].append(row)

    by_type: dict[str, list[str]] = defaultdict(list)
    for cid, items in clusters.items():
        node_types = Counter(str(row.get("node_type") or "Other") for row in items)
        by_type[node_types.most_common(1)[0][0]].append(cid)

    rng = random.Random(seed)
    validation_clusters: set[str] = set()
    frozen_clusters: set[str] = set()
    for node_type, cids in sorted(by_type.items()):
        rng.shuffle(cids)
        count = round(len(cids) * validation_ratio)
        if len(cids) > 1:
            count = min(max(1, count), len(cids) - 1)
        else:
            count = 1
        validation_clusters.update(cids[:count])
        frozen_clusters.update(cids[count:])

    # If a rare type had only one cluster, move it to frozen when validation would otherwise consume everything.
    if not frozen_clusters and len(validation_clusters) > 1:
        moved = sorted(validation_clusters)[-1]
        validation_clusters.remove(moved)
        frozen_clusters.add(moved)

    def build_split(name: str, split_clusters: set[str]) -> list[dict[str, Any]]:
        out_rows = []
        for row in gallery_rows:
            item = dict(row)
            item["eval_split"] = "gallery_index"
            out_rows.append(item)
        for row in query_rows:
            if cluster_id(row) not in split_clusters:
                continue
            item = dict(row)
            item["eval_split"] = name
            out_rows.append(item)
        return out_rows

    validation_rows = build_split("validation", validation_clusters)
    frozen_rows = build_split("frozen_test", frozen_clusters)
    write_jsonl(validation_output, validation_rows)
    write_jsonl(frozen_output, frozen_rows)

    validation_query = [row for row in validation_rows if row.get("role") == "query"]
    frozen_query = [row for row in frozen_rows if row.get("role") == "query"]
    validation_query_clusters = {cluster_id(row) for row in validation_query}
    frozen_query_clusters = {cluster_id(row) for row in frozen_query}
    summary = {
        "source_dataset": dataset,
        "validation_dataset": validation_output,
        "frozen_test_dataset": frozen_output,
        "gallery_shared": True,
        "gallery_items": len(gallery_rows),
        "query_total": len(query_rows),
        "validation_queries": len(validation_query),
        "frozen_test_queries": len(frozen_query),
        "validation_query_nodes": len({str(row.get("node_id")) for row in validation_query}),
        "frozen_test_query_nodes": len({str(row.get("node_id")) for row in frozen_query}),
        "validation_query_clusters": len(validation_query_clusters),
        "frozen_test_query_clusters": len(frozen_query_clusters),
        "query_cluster_intersection": sorted(validation_query_clusters.intersection(frozen_query_clusters)),
        "node_type_validation_queries": dict(Counter(str(row.get("node_type") or "Other") for row in validation_query)),
        "node_type_frozen_test_queries": dict(Counter(str(row.get("node_type") or "Other") for row in frozen_query)),
        "seed": seed,
        "validation_ratio": validation_ratio,
        "note": "Only query clusters are split. Gallery rows are the shared retrieval index and are copied into both split files.",
    }
    Path(summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split eval dataset by duplicate cluster.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--frozen-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--validation-ratio", type=float, default=0.5)
    args = parser.parse_args()
    print(
        json.dumps(
            split_dataset(
                args.dataset,
                args.validation_output,
                args.frozen_output,
                args.summary_output,
                args.seed,
                args.validation_ratio,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
