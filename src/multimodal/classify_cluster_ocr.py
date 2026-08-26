"""Classify OCR evidence for high-near duplicate image clusters."""

from __future__ import annotations

import argparse
import json
import re
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


def norm(text: str) -> str:
    return re.sub(r"[\s,，。.:：;；/\\\-_()（）\[\]【】\"'“”‘’]+", "", str(text or "")).lower()


def split_path(path: str) -> list[str]:
    return [x.strip() for x in re.split(r"\s*/\s*", str(path or "")) if x.strip()]


def aliases_for_name(name: str) -> set[str]:
    raw = str(name or "").strip()
    aliases = {raw}
    m = re.match(r"^(\d{2,4})(.+)$", raw)
    if m:
        number, tail = m.groups()
        aliases.update({number, f"{number}室", f"{number}房间", tail})
    if "/" in raw:
        aliases.update(x.strip() for x in raw.split("/") if x.strip())
    return {a for a in aliases if len(norm(a)) >= 2}


def build_node_profiles(profile_path: str | Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if not profile_path or not Path(profile_path).exists():
        return profiles
    for row in read_jsonl(profile_path):
        node_id = str(row.get("node_id") or "")
        if not node_id:
            continue
        item = profiles.setdefault(
            node_id,
            {
                "node_id": node_id,
                "node_name": str(row.get("node_name") or ""),
                "node_type": str(row.get("node_type") or ""),
                "node_type_label": str(row.get("node_type_label") or ""),
                "parent_node_id": str(row.get("parent_node_id") or ""),
                "hierarchy_path": str(row.get("hierarchy_path") or ""),
                "texts": {},
            },
        )
        variant = str(row.get("profile_variant") or "")
        if variant:
            item["texts"][variant] = str(row.get("text") or "")
    return profiles


def node_terms(node: dict[str, Any]) -> dict[str, set[str]]:
    name = str(node.get("node_name") or "")
    path_parts = split_path(str(node.get("hierarchy_path") or ""))
    parent_terms = set(path_parts[:-1])
    return {
        "exact": aliases_for_name(name),
        "parent": {x for x in parent_terms if len(norm(x)) >= 2},
    }


def match_terms(ocr_norm: str, terms: set[str]) -> list[str]:
    hits = []
    for term in terms:
        term_norm = norm(term)
        if term_norm and term_norm in ocr_norm:
            hits.append(term)
    return sorted(hits, key=lambda x: (-len(norm(x)), x))


def classify_asset(
    asset: dict[str, Any],
    ocr_row: dict[str, Any] | None,
    cluster_nodes: dict[str, dict[str, Any]],
    min_max_score: float,
) -> dict[str, Any]:
    node_id = str(asset.get("node_id") or "")
    ocr_text = str((ocr_row or {}).get("ocr_text") or "")
    ocr_norm = norm(ocr_text)
    if not ocr_row or ocr_row.get("ocr_extract_status") != "ok":
        return {"ocr_status": "unresolved", "ocr_reason": "ocr_extract_failed", "matched_node_ids": [], "matched_terms": {}}
    if not ocr_norm:
        return {"ocr_status": "no_effective_text", "ocr_reason": "empty_ocr_text", "matched_node_ids": [], "matched_terms": {}}
    if float(ocr_row.get("max_score") or 0.0) < min_max_score:
        return {"ocr_status": "low_confidence", "ocr_reason": "max_score_below_threshold", "matched_node_ids": [], "matched_terms": {}}

    matched: dict[str, list[str]] = {}
    parent_hits: list[str] = []
    for other_id, node in cluster_nodes.items():
        terms = node_terms(node)
        exact_hits = match_terms(ocr_norm, terms["exact"])
        if exact_hits:
            matched[other_id] = exact_hits
        if other_id == node_id:
            parent_hits = match_terms(ocr_norm, terms["parent"])

    matched_node_ids = sorted(matched)
    if node_id in matched_node_ids and len(matched_node_ids) == 1:
        exact_name = norm(str(cluster_nodes.get(node_id, {}).get("node_name") or ""))
        status = "exact_node_match" if exact_name and exact_name in ocr_norm else "alias_match"
        return {"ocr_status": status, "ocr_reason": "ocr_matches_current_node", "matched_node_ids": matched_node_ids, "matched_terms": matched}
    if node_id in matched_node_ids and len(matched_node_ids) > 1:
        return {"ocr_status": "multi_node_match", "ocr_reason": "ocr_matches_current_and_other_nodes", "matched_node_ids": matched_node_ids, "matched_terms": matched}
    if matched_node_ids:
        return {"ocr_status": "conflicting_node_match", "ocr_reason": "ocr_matches_other_cluster_nodes", "matched_node_ids": matched_node_ids, "matched_terms": matched}
    if parent_hits:
        return {"ocr_status": "parent_only_match", "ocr_reason": "ocr_matches_parent_path_only", "matched_node_ids": [], "matched_terms": {"parent": parent_hits}}
    return {"ocr_status": "no_effective_text", "ocr_reason": "no_node_or_parent_terms_matched", "matched_node_ids": [], "matched_terms": {}}


def cluster_decision(statuses: list[str]) -> str:
    counts = Counter(statuses)
    useful = len(statuses) - counts["no_effective_text"] - counts["low_confidence"] - counts["unresolved"]
    if not statuses or useful <= 0:
        return "unresolved"
    if counts["conflicting_node_match"]:
        return "possible_wrong_binding"
    if counts["multi_node_match"]:
        return "possible_shared_image"
    if counts["parent_only_match"] and useful == counts["parent_only_match"]:
        return "unresolved"
    if counts["exact_node_match"] + counts["alias_match"] == useful and useful == len(statuses):
        return "auto_hard_negative"
    if counts["exact_node_match"] + counts["alias_match"] > 0:
        return "mixed"
    return "unresolved"


def classify(args: argparse.Namespace) -> dict[str, Any]:
    clusters = read_jsonl(args.clusters)
    ocr_by_asset = {int(row["asset_id"]): row for row in read_jsonl(args.ocr_jsonl) if row.get("asset_id") is not None}
    profiles = build_node_profiles(args.node_profiles)

    cluster_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    status_counter: Counter[str] = Counter()
    decision_counter: Counter[str] = Counter()
    for cluster in clusters:
        cluster_nodes: dict[str, dict[str, Any]] = {}
        for asset in cluster.get("assets") or []:
            node_id = str(asset.get("node_id") or "")
            node = dict(profiles.get(node_id) or {})
            node.setdefault("node_id", node_id)
            node.setdefault("node_name", str(asset.get("node_name") or ""))
            node.setdefault("node_type", str(asset.get("node_type") or ""))
            node.setdefault("parent_node_id", str(asset.get("parent_node_id") or ""))
            cluster_nodes[node_id] = node

        enriched_assets = []
        statuses = []
        for asset in cluster.get("assets") or []:
            asset_id = int(asset["asset_id"])
            cls = classify_asset(asset, ocr_by_asset.get(asset_id), cluster_nodes, args.min_max_score)
            status_counter[cls["ocr_status"]] += 1
            statuses.append(cls["ocr_status"])
            enriched = dict(asset)
            enriched.update(cls)
            enriched["ocr_text"] = str((ocr_by_asset.get(asset_id) or {}).get("ocr_text") or "")
            enriched["ocr_max_score"] = float((ocr_by_asset.get(asset_id) or {}).get("max_score") or 0.0)
            enriched_assets.append(enriched)
            asset_rows.append({**enriched, "cluster_id": cluster.get("cluster_id")})

        decision = cluster_decision(statuses)
        decision_counter[decision] += 1
        cluster_rows.append(
            {
                **cluster,
                "assets": enriched_assets,
                "ocr_status_counts": dict(Counter(statuses)),
                "cluster_ocr_decision": decision,
            }
        )

    write_jsonl(args.output_clusters, cluster_rows)
    write_jsonl(args.output_assets, asset_rows)
    summary = {
        "clusters": str(args.clusters),
        "ocr_jsonl": str(args.ocr_jsonl),
        "output_clusters": str(args.output_clusters),
        "output_assets": str(args.output_assets),
        "cluster_count": len(cluster_rows),
        "asset_count": len(asset_rows),
        "ocr_status_counts": dict(status_counter),
        "cluster_ocr_decision_counts": dict(decision_counter),
    }
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify OCR statuses for near duplicate clusters.")
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--ocr-jsonl", required=True)
    parser.add_argument("--node-profiles", default="")
    parser.add_argument("--output-clusters", required=True)
    parser.add_argument("--output-assets", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--min-max-score", type=float, default=0.55)
    args = parser.parse_args()
    print(json.dumps(classify(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
