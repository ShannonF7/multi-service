"""Filter an image-node eval dataset using duplicate audit outputs.

This script only writes a new JSONL dataset. It does not modify business tables.
The default policy is conservative: exclude every asset involved in a
cross-node exact duplicate pair, then exclude nodes that no longer have both
query and gallery rows.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def load_exact_duplicate_asset_ids(duplicate_audit_dir: str | Path) -> set[int]:
    path = Path(duplicate_audit_dir) / "cross_node_exact_duplicates.jsonl"
    asset_ids: set[int] = set()
    if not path.exists():
        return asset_ids
    for row in read_jsonl(path):
        if row.get("asset_id") is not None:
            asset_ids.add(int(row["asset_id"]))
        if row.get("duplicate_asset_id") is not None:
            asset_ids.add(int(row["duplicate_asset_id"]))
    return asset_ids


def filter_dataset(dataset: str, duplicate_audit_dir: str, output: str, dataset_version: str | None) -> dict[str, Any]:
    rows = read_jsonl(dataset)
    exact_assets = load_exact_duplicate_asset_ids(duplicate_audit_dir)

    filtered: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if dataset_version:
            item["dataset_version"] = dataset_version
        metadata = dict(item.get("metadata") or {})
        if int(item.get("asset_id") or -1) in exact_assets:
            item["usable"] = False
            item["visual_label_status"] = "invalid"
            metadata["exclusion_reason"] = "cross_node_exact_duplicate_auto_excluded"
        item["metadata"] = metadata
        filtered.append(item)

    usable_by_node: dict[str, set[str]] = defaultdict(set)
    for row in filtered:
        if row.get("usable", True) is False:
            continue
        usable_by_node[str(row.get("node_id") or "")].add(str(row.get("role") or ""))

    invalid_nodes = {node_id for node_id, roles in usable_by_node.items() if not {"query", "gallery"}.issubset(roles)}
    for item in filtered:
        node_id = str(item.get("node_id") or "")
        if node_id in invalid_nodes and item.get("usable", True) is not False:
            metadata = dict(item.get("metadata") or {})
            item["usable"] = False
            item["visual_label_status"] = "invalid"
            metadata["exclusion_reason"] = "node_without_query_gallery_after_duplicate_filter"
            item["metadata"] = metadata

    write_jsonl(output, filtered)

    usable = [row for row in filtered if row.get("usable", True) is not False]
    summary = {
        "source_dataset": dataset,
        "output": output,
        "dataset_version": dataset_version,
        "items": len(filtered),
        "usable": len(usable),
        "excluded": len(filtered) - len(usable),
        "cross_node_exact_duplicate_assets": len(exact_assets),
        "nodes_without_query_gallery_after_filter": len(invalid_nodes),
        "query_items": sum(1 for row in usable if row.get("role") == "query"),
        "gallery_items": sum(1 for row in usable if row.get("role") == "gallery"),
        "usable_nodes": len({str(row.get("node_id")) for row in usable}),
    }
    Path(output).with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter eval dataset by duplicate audit.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--duplicate-audit-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-version", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            filter_dataset(args.dataset, args.duplicate_audit_dir, args.output, args.dataset_version),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
