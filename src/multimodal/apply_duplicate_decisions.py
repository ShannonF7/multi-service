"""Apply cross-node duplicate review decisions to build Pilot v3.

This script only writes a new JSONL evaluation dataset. It never updates DB rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VALID_DECISIONS = {"left_correct", "right_correct", "shared", "same_entity", "uncertain", ""}


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def apply_decisions(dataset: str, decisions_path: str, output: str, dataset_version: str) -> dict:
    rows = read_jsonl(dataset)
    decisions = read_jsonl(decisions_path)
    by_asset = {int(row["asset_id"]): dict(row) for row in rows}
    audit_log = []
    invalid = []

    for decision in decisions:
        left = int(decision.get("left_asset_id") or 0)
        right = int(decision.get("right_asset_id") or 0)
        action = str(decision.get("decision") or "").strip()
        reason = str(decision.get("reason") or "").strip()
        canonical = str(decision.get("canonical_node_id") or "").strip()
        if action not in VALID_DECISIONS:
            invalid.append({"decision": decision, "error": "invalid_decision"})
            continue
        if not action:
            invalid.append({"decision": decision, "error": "missing_decision"})
            continue

        affected = []
        if action == "left_correct":
            if right in by_asset:
                by_asset[right]["usable"] = False
                by_asset[right]["exclusion_reason"] = "cross_node_duplicate_right_excluded"
                by_asset[right]["visual_label_status"] = "invalid"
                affected.append(right)
        elif action == "right_correct":
            if left in by_asset:
                by_asset[left]["usable"] = False
                by_asset[left]["exclusion_reason"] = "cross_node_duplicate_left_excluded"
                by_asset[left]["visual_label_status"] = "invalid"
                affected.append(left)
        elif action == "shared":
            for asset_id in (left, right):
                if asset_id in by_asset:
                    by_asset[asset_id]["usable"] = False
                    by_asset[asset_id]["exclusion_reason"] = "shared_visual_asset_excluded_from_single_label"
                    by_asset[asset_id]["visual_label_status"] = "shared"
                    affected.append(asset_id)
        elif action == "same_entity":
            if not canonical:
                invalid.append({"decision": decision, "error": "canonical_node_id_required"})
                continue
            for asset_id in (left, right):
                if asset_id in by_asset:
                    by_asset[asset_id].setdefault("metadata", {})
                    by_asset[asset_id]["metadata"]["original_node_id"] = by_asset[asset_id].get("node_id")
                    by_asset[asset_id]["metadata"]["same_entity_decision"] = True
                    by_asset[asset_id]["node_id"] = canonical
                    by_asset[asset_id]["visual_label_status"] = "direct"
                    affected.append(asset_id)
        elif action == "uncertain":
            for asset_id in (left, right):
                if asset_id in by_asset:
                    by_asset[asset_id]["usable"] = False
                    by_asset[asset_id]["exclusion_reason"] = "cross_node_duplicate_uncertain"
                    by_asset[asset_id]["visual_label_status"] = "uncertain"
                    affected.append(asset_id)
        audit_log.append({"decision": decision, "affected_asset_ids": affected, "reason": reason})

    output_rows = []
    for row in rows:
        item = by_asset[int(row["asset_id"])]
        item["dataset_version"] = dataset_version
        output_rows.append(item)
    write_jsonl(output, output_rows)
    out = Path(output)
    out.with_suffix(".decision_audit.json").write_text(json.dumps({"audit_log": audit_log, "invalid": invalid}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "source_dataset": dataset,
        "output": output,
        "dataset_version": dataset_version,
        "items": len(output_rows),
        "usable": sum(1 for row in output_rows if row.get("usable", True)),
        "excluded": sum(1 for row in output_rows if not row.get("usable", True)),
        "invalid_decisions": len(invalid),
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply duplicate decisions to build Pilot v3.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-version", default="scenic_4_e0_pilot_v3")
    args = parser.parse_args()
    print(json.dumps(apply_decisions(args.dataset, args.decisions, args.output, args.dataset_version), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
