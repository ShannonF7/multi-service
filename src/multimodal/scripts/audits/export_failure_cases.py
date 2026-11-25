"""Export typed failure cases from an E0-A evaluation output directory."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compact(row: dict) -> dict:
    return {
        "query_asset_id": row.get("query_asset_id"),
        "query_node_id": row.get("query_node_id"),
        "query_node_type": row.get("query_node_type"),
        "visual_label_status": row.get("visual_label_status"),
        "failure_types": row.get("failure_types") or [],
        "image_rank": row.get("image_rank"),
        "node_rank_max": row.get("node_rank_max"),
        "node_rank_centroid": row.get("node_rank_centroid"),
        "node_rank_top3_mean": row.get("node_rank_top3_mean"),
        "node_rank_hybrid": row.get("node_rank_hybrid"),
        "tie_status_max": row.get("tie_status_max"),
        "tie_status_centroid": row.get("tie_status_centroid"),
        "tie_status_top3_mean": row.get("tie_status_top3_mean"),
        "tie_status_hybrid": row.get("tie_status_hybrid"),
        "top_images": (row.get("image_hits") or [])[:5],
        "top_nodes_max": ((row.get("node_rankings") or {}).get("max") or [])[:5],
        "top_nodes_centroid": ((row.get("node_rankings") or {}).get("centroid") or [])[:5],
        "top_nodes_top3_mean": ((row.get("node_rankings") or {}).get("top3_mean") or [])[:5],
        "top_nodes_hybrid": ((row.get("node_rankings") or {}).get("hybrid") or [])[:5],
    }


def export_failures(input_dir: str, output_dir: str, limit: int) -> dict:
    source = Path(input_dir) / "failure_cases.jsonl"
    rows = read_jsonl(source)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for failure_type in row.get("failure_types") or ["unknown_failure"]:
            by_type[str(failure_type)].append(compact(row))

    for failure_type, items in by_type.items():
        with (out / f"{failure_type}.jsonl").open("w", encoding="utf-8") as f:
            for item in items[:limit]:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    all_compact = [compact(row) for row in rows[:limit]]
    with (out / "failure_cases_compact.jsonl").open("w", encoding="utf-8") as f:
        for item in all_compact:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    summary = {"input_dir": input_dir, "output_dir": output_dir, "total_failures": len(rows), "types": {key: len(value) for key, value in by_type.items()}}
    (out / "failure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export typed E0-A failure cases.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(export_failures(args.input_dir, args.output_dir, args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
