"""Validate E0 image-node JSONL dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

VALID_VISUAL_STATUS = {"direct", "contextual", "shared", "uncertain", "invalid"}

def is_usable_eval_row(row: dict) -> bool:
    return row.get("usable", True) is not False

def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_dataset(dataset: str, eval_output_dir: str | None = None) -> dict:
    rows = read_jsonl(dataset)
    usable_rows = [row for row in rows if is_usable_eval_row(row)]
    issues = {
        "missing_asset": 0,
        "missing_node": 0,
        "self_leakage": 0,
        "duplicate_leakage": 0,
        "query_without_positive_gallery": 0,
        "invalid_role": 0,
        "invalid_visual_label_status": 0,
        "cross_node_exact_duplicate": 0,
        "closed_set_node_rank_null": 0,
    }
    by_asset = defaultdict(list)
    by_node = defaultdict(list)
    duplicate_roles = defaultdict(set)
    for row in usable_rows:
        if not row.get("asset_id"):
            issues["missing_asset"] += 1
        if not row.get("node_id"):
            issues["missing_node"] += 1
        if row.get("role") not in {"query", "gallery"}:
            issues["invalid_role"] += 1
        if str(row.get("visual_label_status") or "direct") not in VALID_VISUAL_STATUS:
            issues["invalid_visual_label_status"] += 1
        by_asset[int(row.get("asset_id") or -1)].append(row)
        by_node[str(row.get("node_id") or "")].append(row)
        if row.get("duplicate_group_id"):
            duplicate_roles[str(row["duplicate_group_id"])].add(str(row.get("role")))

    for items in by_asset.values():
        roles = {item.get("role") for item in items}
        if len(roles) > 1:
            issues["self_leakage"] += 1

    for roles in duplicate_roles.values():
        if "query" in roles and "gallery" in roles:
            issues["duplicate_leakage"] += 1

    for row in usable_rows:
        if row.get("role") != "query":
            continue
        same_node_gallery = [item for item in by_node[str(row.get("node_id"))] if item.get("role") == "gallery"]
        if not same_node_gallery:
            issues["query_without_positive_gallery"] += 1

    if eval_output_dir:
        per_query = Path(eval_output_dir) / "per_query_results.jsonl"
        if per_query.exists():
            for row in read_jsonl(str(per_query)):
                for key, value in row.items():
                    if key.startswith("node_rank_") and value is None:
                        issues["closed_set_node_rank_null"] += 1
                        break
        dup_paths = [Path(eval_output_dir) / "cross_node_exact_duplicates.jsonl", Path(eval_output_dir) / "duplicate_audit" / "cross_node_exact_duplicates.jsonl"]
        for dup_path in dup_paths:
            if dup_path.exists():
                issues["cross_node_exact_duplicate"] += sum(1 for line in dup_path.open("r", encoding="utf-8") if line.strip())

    result = {"dataset": dataset, "items": len(rows), "usable_items": len(usable_rows), "passed": all(value == 0 for value in issues.values()), **issues}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate E0 JSONL dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval-output-dir", default=None)
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset, args.eval_output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

