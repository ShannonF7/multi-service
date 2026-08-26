"""Build expanded_v2 image-node eval dataset with near-duplicate cluster isolation.

This script writes evaluation artifacts only. It does not modify production
image/node bindings.
"""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


class DSU:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
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


def resolve_url(value: str, base_url: str) -> str:
    value = str(value or "")
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return urljoin(base_url.rstrip("/") + "/", value.lstrip("/"))
    return value


def pair_assets(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["asset_id"]), int(row["duplicate_asset_id"])


def classify_cross_near(row: dict[str, Any], high_phash_threshold: int) -> str:
    phash = row.get("phash_distance")
    if phash is None:
        return "unknown_near"
    if int(phash) <= high_phash_threshold:
        return "cross_node_high_near_review_excluded"
    return "hard_negative_retained"


def build_review_page(rows: list[dict[str, Any]], output_html: str, base_url: str) -> None:
    cards = []
    template = []
    for index, row in enumerate(rows, start=1):
        left = row.get("left") or {}
        right = row.get("right") or {}
        key = f"{row.get('asset_id')}__{row.get('duplicate_asset_id')}"
        template.append(
            {
                "pair_key": key,
                "left_asset_id": row.get("asset_id"),
                "right_asset_id": row.get("duplicate_asset_id"),
                "left_node_id": row.get("node_id"),
                "right_node_id": row.get("duplicate_node_id"),
                "decision": "",
                "category": "",
                "reason": "",
                "canonical_node_id": "",
            }
        )
        labels = [
            ("same_visual_asset_variant", "A 同一视觉资产变体"),
            ("same_scene_burst", "B 同一节点/同场景连续拍摄"),
            ("shared_or_wrong_binding", "C 跨节点共享或错误绑定"),
            ("hard_negative", "D 不同节点但视觉相似，保留 hard negative"),
            ("uncertain", "无法判断，暂不进标准测试"),
        ]
        radio_html = "".join(
            f'<label><input type="radio" name="cat_{html.escape(key)}" value="{html.escape(value)}"> {html.escape(text)}</label>'
            for value, text in labels
        )
        cards.append(
            f"""
<section class="pair" data-pair-key="{html.escape(key)}">
  <div class="pair-head">
    <h2>#{index} pair {html.escape(key)}</h2>
    <div class="meta">phash_distance={html.escape(str(row.get('phash_distance')))} · simclr_distance={html.escape(str(row.get('simclr_distance')))}</div>
  </div>
  <div class="grid">
    <div class="panel">
      <img src="{html.escape(resolve_url(str(left.get('image_url') or ''), base_url))}">
      <h3>{html.escape(str(left.get('node_name') or ''))}</h3>
      <p>node {html.escape(str(row.get('node_id')))} · {html.escape(str(left.get('node_type') or ''))}</p>
      <p>asset {html.escape(str(row.get('asset_id')))}</p>
    </div>
    <div class="panel">
      <img src="{html.escape(resolve_url(str(right.get('image_url') or ''), base_url))}">
      <h3>{html.escape(str(right.get('node_name') or ''))}</h3>
      <p>node {html.escape(str(row.get('duplicate_node_id')))} · {html.escape(str(right.get('node_type') or ''))}</p>
      <p>asset {html.escape(str(row.get('duplicate_asset_id')))}</p>
    </div>
  </div>
  <div class="decision">
    {radio_html}
    <label>canonical_node_id <input type="text" class="canonical"></label>
    <label>reason <textarea class="reason"></textarea></label>
  </div>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Expanded v2 High Near Duplicate Review</title>
<style>
body{{margin:0;font-family:Arial,"Microsoft YaHei",sans-serif;background:#f6f7f9;color:#172033}}
header{{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #d8dee8;padding:14px 22px;display:flex;justify-content:space-between;gap:16px;align-items:center}}
h1{{margin:0;font-size:18px}}button{{border:1px solid #2563eb;background:#2563eb;color:white;border-radius:6px;padding:8px 12px;cursor:pointer}}
main{{padding:20px;max-width:1280px;margin:0 auto}}.pair{{background:#fff;border:1px solid #d8dee8;border-radius:8px;margin-bottom:16px;padding:14px}}
.pair-head{{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px}}h2{{margin:0;font-size:16px}}.meta{{color:#667085;font-size:13px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.panel{{border:1px solid #e6e9ef;border-radius:8px;padding:10px;background:#fbfcfe}}
.panel img{{width:100%;height:300px;object-fit:contain;background:#111827;border-radius:6px}}.panel h3{{margin:8px 0 4px;font-size:15px}}.panel p{{margin:4px 0;color:#475467}}
.decision{{margin-top:10px;display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:8px}}.decision label{{display:block;font-size:14px}}
.decision input[type=text],.decision textarea{{width:100%;box-sizing:border-box;border:1px solid #cfd6e4;border-radius:6px;padding:7px;margin-top:4px}}textarea{{min-height:54px}}
#exportBox{{width:100%;height:180px;margin-top:14px;font-family:Consolas,monospace}}@media(max-width:860px){{.grid,.decision{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div><h1>Expanded v2 高相似跨节点图片审核</h1><div class="meta">共 {len(rows)} 组。当前已从单标签评测中保守排除，人工审核后可恢复 hard negative 或 shared/multilabel。</div></div><button onclick="exportDecisions()">导出 decisions JSONL</button></header>
<main>{''.join(cards)}<textarea id="exportBox" placeholder="点击导出后复制 JSONL"></textarea></main>
<script>
const PAIRS={json.dumps(template, ensure_ascii=False)};
function exportDecisions(){{
 const lines=[];
 for(const pair of PAIRS){{
  const section=document.querySelector(`[data-pair-key="${{pair.pair_key}}"]`);
  const checked=section.querySelector(`input[name="cat_${{pair.pair_key}}"]:checked`);
  lines.push(JSON.stringify({{...pair,category:checked?checked.value:"",decision:checked?checked.value:"",canonical_node_id:section.querySelector('.canonical').value.trim(),reason:section.querySelector('.reason').value.trim()}}));
 }}
 document.getElementById('exportBox').value=lines.join('\\n')+'\\n';
}}
</script></body></html>"""
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")


def build_v2(
    dataset: str,
    duplicate_audit_dir: str,
    output: str,
    output_dir: str,
    dataset_version: str,
    seed: int,
    query_ratio: float,
    high_phash_threshold: int,
    media_base_url: str,
) -> dict[str, Any]:
    rows = [dict(row) for row in read_jsonl(dataset)]
    usable_source = [row for row in rows if row.get("usable", True) is not False]
    by_asset = {int(row["asset_id"]): row for row in rows}

    audit = Path(duplicate_audit_dir)
    same_exact = read_jsonl(audit / "same_node_exact_duplicates.jsonl")
    cross_exact = read_jsonl(audit / "cross_node_exact_duplicates.jsonl")
    cross_near = read_jsonl(audit / "cross_node_near_duplicates.jsonl")
    all_pairs = read_jsonl(audit / "all_duplicate_candidates.jsonl")

    dsu = DSU()
    for asset_id in by_asset:
        dsu.find(asset_id)

    # A: exact visual variants are always one cluster.
    for row in same_exact + cross_exact:
        left, right = pair_assets(row)
        dsu.union(left, right)

    # B: same-node high-near images are one cluster; keep cluster isolated across query/gallery.
    for row in all_pairs:
        if str(row.get("node_id")) != str(row.get("duplicate_node_id")):
            continue
        phash = row.get("phash_distance")
        if phash is not None and int(phash) <= high_phash_threshold:
            left, right = pair_assets(row)
            dsu.union(left, right)

    cluster_members: dict[str, list[int]] = defaultdict(list)
    for asset_id in by_asset:
        cluster_members[f"dc_{dsu.find(asset_id)}"].append(asset_id)

    cross_high_rows = [row for row in cross_near if classify_cross_near(row, high_phash_threshold).startswith("cross_node_high")]
    hard_negative_rows = [row for row in cross_near if classify_cross_near(row, high_phash_threshold) == "hard_negative_retained"]
    excluded_assets: set[int] = set()
    for row in cross_exact + cross_high_rows:
        left, right = pair_assets(row)
        excluded_assets.add(left)
        excluded_assets.add(right)

    # Keep only one representative from same exact cluster to avoid trivial duplicates.
    representative_by_cluster = {cluster_id: min(asset_ids) for cluster_id, asset_ids in cluster_members.items()}
    for cluster_id, asset_ids in cluster_members.items():
        nodes = {str(by_asset[aid].get("node_id")) for aid in asset_ids if aid in by_asset}
        if len(nodes) == 1 and len(asset_ids) > 1:
            for asset_id in asset_ids:
                if asset_id != representative_by_cluster[cluster_id]:
                    excluded_assets.add(asset_id)

    # Assign query/gallery by node-level duplicate clusters, not by individual assets.
    rng = random.Random(seed)
    node_clusters: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for asset_id, row in by_asset.items():
        if row.get("usable", True) is False or asset_id in excluded_assets:
            continue
        cluster_id = f"dc_{dsu.find(asset_id)}"
        node_clusters[str(row.get("node_id"))][cluster_id].append(asset_id)

    role_by_asset: dict[int, str] = {}
    invalid_nodes: set[str] = set()
    for node_id, clusters in node_clusters.items():
        cluster_ids = sorted(clusters)
        if len(cluster_ids) < 2:
            invalid_nodes.add(node_id)
            continue
        rng.shuffle(cluster_ids)
        query_count = max(1, round(len(cluster_ids) * query_ratio))
        if len(cluster_ids) - query_count < 1:
            query_count = len(cluster_ids) - 1
        query_clusters = set(cluster_ids[:query_count])
        for cluster_id, asset_ids in clusters.items():
            role = "query" if cluster_id in query_clusters else "gallery"
            for asset_id in asset_ids:
                role_by_asset[asset_id] = role

    output_rows = []
    for row in rows:
        item = dict(row)
        item["dataset_version"] = dataset_version
        asset_id = int(item["asset_id"])
        cluster_id = f"dc_{dsu.find(asset_id)}"
        metadata = dict(item.get("metadata") or {})
        metadata["duplicate_cluster_id"] = cluster_id
        metadata["duplicate_cluster_size"] = len(cluster_members.get(cluster_id, []))
        metadata["expanded_v2_policy"] = {
            "high_phash_threshold": high_phash_threshold,
            "cross_node_high_near": "excluded_pending_manual_review",
            "cross_node_medium_near": "retained_as_hard_negative",
            "same_node_duplicate_cluster": "cluster_isolated_and_representative_kept",
        }
        item["duplicate_group_id"] = cluster_id if len(cluster_members.get(cluster_id, [])) > 1 else item.get("duplicate_group_id")
        if asset_id in excluded_assets:
            item["usable"] = False
            item["visual_label_status"] = "invalid"
            metadata["exclusion_reason"] = "expanded_v2_duplicate_or_high_near_excluded"
        elif str(item.get("node_id")) in invalid_nodes:
            item["usable"] = False
            item["visual_label_status"] = "invalid"
            metadata["exclusion_reason"] = "node_without_two_independent_clusters_after_v2_filter"
        else:
            item["usable"] = True
            item["visual_label_status"] = "contextual"
            item["role"] = role_by_asset.get(asset_id, item.get("role"))
        item["metadata"] = metadata
        output_rows.append(item)

    write_jsonl(output, output_rows)

    usable = [row for row in output_rows if row.get("usable", True) is not False]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "cross_node_high_near_review.jsonl", cross_high_rows)
    write_jsonl(out_dir / "hard_negative_retained_pairs.jsonl", hard_negative_rows)
    build_review_page(cross_high_rows[:300], str(out_dir / "cross_node_high_near_review.html"), media_base_url)

    raw_distribution = Counter(str(len(items)) for items in defaultdict(list, {}).values())
    per_node_counts = Counter()
    for row in usable:
        per_node_counts[str(row["node_id"])] += 1
    image_count_distribution = Counter(str(count) for count in per_node_counts.values())
    clusters_with_roles: dict[str, set[str]] = defaultdict(set)
    for row in usable:
        cid = str((row.get("metadata") or {}).get("duplicate_cluster_id") or row.get("asset_id"))
        clusters_with_roles[cid].add(str(row.get("role")))
    cross_partition_clusters = sum(1 for roles in clusters_with_roles.values() if {"query", "gallery"}.issubset(roles))

    summary = {
        "dataset_version": dataset_version,
        "source_dataset": dataset,
        "output": output,
        "raw_items": len(rows),
        "raw_usable_items": len(usable_source),
        "raw_nodes": len({str(row.get("node_id")) for row in usable_source}),
        "clean_items": len(usable),
        "clean_nodes": len({str(row.get("node_id")) for row in usable}),
        "query_items": sum(1 for row in usable if row.get("role") == "query"),
        "gallery_items": sum(1 for row in usable if row.get("role") == "gallery"),
        "image_count_distribution": dict(sorted(image_count_distribution.items(), key=lambda item: int(item[0]))),
        "exact_duplicate_pairs": len(same_exact) + len(cross_exact),
        "same_node_exact_pairs": len(same_exact),
        "cross_node_exact_pairs": len(cross_exact),
        "near_duplicate_pairs": len(cross_near),
        "cross_node_high_near_pairs_review_excluded": len(cross_high_rows),
        "hard_negative_retained_pairs": len(hard_negative_rows),
        "duplicate_clusters": sum(1 for items in cluster_members.values() if len(items) > 1),
        "cross_partition_duplicate_clusters": cross_partition_clusters,
        "manual_review_required": len(cross_high_rows),
        "excluded_assets": len(excluded_assets),
        "invalid_nodes_after_filter": len(invalid_nodes),
        "rules": {
            "A_same_visual_asset_variant": "same_url/same_sha exact variants are duplicate_cluster; one representative kept for same-node exact clusters; cross-node exact excluded",
            "B_same_node_same_scene_burst": f"same-node phash <= {high_phash_threshold} clustered and not split across query/gallery",
            "C_cross_node_shared_or_wrong_binding": f"cross-node phash <= {high_phash_threshold} excluded from single-label eval pending manual review",
            "D_visual_similar_different_nodes": f"cross-node phash > {high_phash_threshold} and <= 5 retained as hard negative",
        },
    }
    Path(output).with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Expanded v2 Dataset Report",
        "",
        "## Summary",
        "",
        f"- raw nodes/images: {summary['raw_nodes']} / {summary['raw_usable_items']}",
        f"- cleaned effective nodes/images: {summary['clean_nodes']} / {summary['clean_items']}",
        f"- query/gallery: {summary['query_items']} / {summary['gallery_items']}",
        f"- exact duplicate pairs: {summary['exact_duplicate_pairs']}",
        f"- near duplicate pairs: {summary['near_duplicate_pairs']}",
        f"- duplicate clusters: {summary['duplicate_clusters']}",
        f"- cross-partition duplicate clusters: {summary['cross_partition_duplicate_clusters']}",
        f"- manual review required: {summary['manual_review_required']}",
        "",
        "## Policy",
        "",
        "- A same visual asset variants: cluster and isolate; same-node exact keeps one representative.",
        f"- B same-node high-near burst: phash <= {high_phash_threshold} cluster-isolated.",
        f"- C cross-node high-near/shared/wrong binding: phash <= {high_phash_threshold} excluded pending review.",
        f"- D visual-similar different nodes: phash 3-5 retained as hard negative.",
        "",
        "## Artifacts",
        "",
        f"- dataset: `{output}`",
        f"- high-near review JSONL: `{out_dir / 'cross_node_high_near_review.jsonl'}`",
        f"- high-near review HTML: `{out_dir / 'cross_node_high_near_review.html'}`",
        f"- hard negatives: `{out_dir / 'hard_negative_retained_pairs.jsonl'}`",
    ]
    (out_dir / "EXPANDED_V2_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expanded_v2 eval dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--duplicate-audit-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-version", default="scenic_4_image_node_expanded_v2")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--query-ratio", type=float, default=0.3)
    parser.add_argument("--high-phash-threshold", type=int, default=2)
    parser.add_argument("--media-base-url", default="http://ai.smartoptiks.cn")
    args = parser.parse_args()
    print(
        json.dumps(
            build_v2(
                args.dataset,
                args.duplicate_audit_dir,
                args.output,
                args.output_dir,
                args.dataset_version,
                args.seed,
                args.query_ratio,
                args.high_phash_threshold,
                args.media_base_url,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
