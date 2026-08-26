"""Evaluate E1 original CLIP baseline on the curated image-node dataset.

This script is intentionally file-cache based: it does not write CLIP vectors to
image_embeddings, so the experiment does not depend on pgvector column dims.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import bindparam, text

from src.rag.dependencies import get_ai_engine
from src.multimodal.clip_feature_extractor import get_clip_extractor
from src.multimodal.metrics import RankingCase, topk_accuracy
from src.multimodal.node_aggregation import aggregate_centroid_scores, aggregate_node_scores, normalize, dot
from src.multimodal.simclr_backfill import DEFAULT_MEDIA_BASE_URL, fetch_image_to_cache

MODEL_NAME = "openai_clip_vit_b32"
STAGE = "E1"
SETTING = "cross_domain_zero_shot"


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def rank_of(expected: set[str], ranked: list[str]) -> int | None:
    for index, item in enumerate(ranked, start=1):
        if item in expected:
            return index
    return None


def reciprocal_rank_from_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def tie_aware_hit(expected: set[str], rankings: list[dict[str, Any]], k: int) -> bool:
    for row in rankings:
        if int(row.get("rank") or 999999) > k:
            continue
        tie_group = {str(item) for item in row.get("tie_group") or [row.get("node_id")]}
        if expected.intersection(tie_group):
            return True
    return False


def node_metrics(records: list[dict[str, Any]], aggregation: str, top_ks: list[int]) -> dict[str, float]:
    total = len(records)
    if not total:
        return {**{f"strict_top{k}": 0.0 for k in top_ks}, **{f"tie_aware_top{k}": 0.0 for k in top_ks}, "mrr": 0.0, "tie_case_rate": 0.0, "rank_null": 0.0}
    metrics: dict[str, float] = {}
    for k in top_ks:
        metrics[f"strict_top{k}"] = sum(1 for row in records if row.get(f"node_rank_{aggregation}") and int(row[f"node_rank_{aggregation}"]) <= k) / total
        metrics[f"tie_aware_top{k}"] = sum(1 for row in records if tie_aware_hit({str(row["query_node_id"])}, row["node_rankings"].get(aggregation, []), k)) / total
    metrics["mrr"] = sum(reciprocal_rank_from_rank(row.get(f"node_rank_{aggregation}")) for row in records) / total
    metrics["tie_case_rate"] = sum(1 for row in records if row.get(f"tie_status_{aggregation}") == "tied") / total
    metrics["rank_null"] = sum(1 for row in records if row.get(f"node_rank_{aggregation}") is None) / total
    return metrics


def load_assets(connection, asset_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not asset_ids:
        return {}
    stmt = text(
        """
        select id, scenic_id, source_scenic_id, source_node_id, source_asset_id,
               role, is_cover, url, file_hash
        from node_assets
        where id in :asset_ids
        """
    ).bindparams(bindparam("asset_ids", expanding=True))
    rows = connection.execute(stmt, {"asset_ids": asset_ids}).mappings().all()
    return {int(row["id"]): dict(row) for row in rows}


def load_or_create_vectors(rows: list[dict[str, Any]], assets: dict[int, dict[str, Any]], output_dir: Path, media_base_url: str, force: bool) -> tuple[dict[int, list[float]], list[dict[str, Any]]]:
    cache_path = output_dir / "clip_image_vectors.jsonl"
    vectors: dict[int, list[float]] = {}
    failures: list[dict[str, Any]] = []
    if cache_path.exists() and not force:
        for row in read_jsonl(str(cache_path)):
            vectors[int(row["asset_id"])] = [float(item) for item in row["vector"]]

    needed = []
    for row in rows:
        asset_id = int(row["asset_id"])
        if asset_id not in vectors:
            needed.append(asset_id)
    if needed:
        extractor = get_clip_extractor()
        cache_rows = [{"asset_id": aid, "vector": vec} for aid, vec in sorted(vectors.items())]
        for asset_id in needed:
            asset = assets.get(asset_id)
            if not asset:
                failures.append({"asset_id": asset_id, "error": "asset_not_found"})
                continue
            try:
                image_path = fetch_image_to_cache(str(asset.get("url") or ""), media_base_url)
                vector = normalize(extractor.extract(str(image_path)))
                vectors[asset_id] = vector
                cache_rows.append({"asset_id": asset_id, "vector": vector})
            except Exception as exc:
                failures.append({"asset_id": asset_id, "error": str(exc)[:500]})
        write_jsonl(cache_path, cache_rows)
    return vectors, failures


def ranking_metrics(cases: list[RankingCase], top_ks: list[int]) -> dict[str, float]:
    return topk_accuracy(cases, top_ks)


def evaluate(dataset: str, output_dir: str, visual_statuses: set[str], aggregations: list[str], top_ks: list[int], retrieval_top_k: int, media_base_url: str, force_recompute: bool) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = [row for row in read_jsonl(dataset) if row.get("usable", True)]
    rows = [row for row in all_rows if str(row.get("visual_label_status") or "direct") in visual_statuses]
    queries = [row for row in rows if row.get("role") == "query"]
    gallery = [row for row in rows if row.get("role") == "gallery"]
    asset_ids = sorted({int(row["asset_id"]) for row in rows})

    with get_ai_engine().connect() as connection:
        assets = load_assets(connection, asset_ids)
    vectors, vector_failures = load_or_create_vectors(rows, assets, out, media_base_url, force_recompute)

    gallery = [row for row in gallery if int(row["asset_id"]) in vectors]
    queries = [row for row in queries if int(row["asset_id"]) in vectors]
    gallery_by_node: dict[str, list[int]] = defaultdict(list)
    for row in gallery:
        gallery_by_node[str(row["node_id"])].append(int(row["asset_id"]))
    node_vectors = {node_id: [vectors[asset_id] for asset_id in ids if asset_id in vectors] for node_id, ids in gallery_by_node.items()}

    image_cases: list[RankingCase] = []
    retrieval_image_cases: list[RankingCase] = []
    per_query: list[dict[str, Any]] = []
    for query in queries:
        q_asset = int(query["asset_id"])
        expected_node = str(query["node_id"])
        query_vector = vectors[q_asset]
        hits = []
        for item in gallery:
            asset_id = int(item["asset_id"])
            sim = dot(query_vector, vectors[asset_id])
            asset = assets.get(asset_id, {})
            hits.append({
                "asset_id": asset_id,
                "node_id": str(item["node_id"]),
                "source_node_id": str(item["node_id"]),
                "source_asset_id": asset.get("source_asset_id"),
                "image_url": asset.get("url"),
                "similarity": float(sim),
                "distance": float(1.0 - sim),
            })
        hits.sort(key=lambda item: (-float(item["similarity"]), int(item["asset_id"])))
        for index, hit in enumerate(hits, start=1):
            hit["rank"] = index

        ranked_assets = [str(hit["asset_id"]) for hit in hits]
        expected_assets = {str(row["asset_id"]) for row in gallery if str(row["node_id"]) == expected_node}
        image_cases.append(RankingCase(str(q_asset), expected_assets, ranked_assets))
        retrieval_image_cases.append(RankingCase(str(q_asset), expected_assets, ranked_assets[:retrieval_top_k]))

        record: dict[str, Any] = {
            "evaluation_stage": STAGE,
            "model": MODEL_NAME,
            "evaluation_setting": SETTING,
            "query_asset_id": q_asset,
            "query_node_id": expected_node,
            "query_node_type": query.get("node_type") or "Other",
            "visual_label_status": query.get("visual_label_status") or "direct",
            "image_rank": rank_of(expected_assets, ranked_assets),
            "image_hits": hits,
            "node_rankings": {},
        }
        for agg in aggregations:
            if agg == "centroid":
                node_rankings = aggregate_centroid_scores(query_vector, node_vectors)
            else:
                node_rankings = aggregate_node_scores(hits, agg)
            ranked_nodes = [str(item["node_id"]) for item in node_rankings]
            rank = rank_of({expected_node}, ranked_nodes)
            record["node_rankings"][agg] = node_rankings
            record[f"node_rank_{agg}"] = rank
            expected_rows = [item for item in node_rankings if str(item.get("node_id")) == expected_node]
            record[f"tie_status_{agg}"] = expected_rows[0].get("tie_status") if expected_rows else "missing"
            record[f"tie_group_{agg}"] = expected_rows[0].get("tie_group") if expected_rows else []
        per_query.append(record)

    closed_image_metrics = ranking_metrics(image_cases, top_ks)
    retrieval_image_metrics = ranking_metrics(retrieval_image_cases, [k for k in top_ks if k <= retrieval_top_k])
    closed_node_metrics = {agg: node_metrics(per_query, agg, top_ks) for agg in aggregations}

    retrieval_records = []
    for row in per_query:
        subset = dict(row)
        top_hits = row["image_hits"][:retrieval_top_k]
        subset["node_rankings"] = {}
        for agg in aggregations:
            if agg == "centroid":
                continue
            node_rankings = aggregate_node_scores(top_hits, agg)
            ranked_nodes = [str(item["node_id"]) for item in node_rankings]
            subset["node_rankings"][agg] = node_rankings
            subset[f"node_rank_{agg}"] = rank_of({str(row["query_node_id"])}, ranked_nodes)
            expected_rows = [item for item in node_rankings if str(item.get("node_id")) == str(row["query_node_id"])]
            subset[f"tie_status_{agg}"] = expected_rows[0].get("tie_status") if expected_rows else "missing"
        retrieval_records.append(subset)
    retrieval_node_metrics = {agg: node_metrics(retrieval_records, agg, [k for k in top_ks if k <= retrieval_top_k]) for agg in aggregations if agg != "centroid"}

    metrics = {
        "model": MODEL_NAME,
        "evaluation_stage": STAGE,
        "evaluation_setting": SETTING,
        "dataset": dataset,
        "visual_statuses": sorted(visual_statuses),
        "query_count": len(per_query),
        "gallery_count": len(gallery),
        "vector_failures": vector_failures,
        "closed_set": {"image_retrieval": closed_image_metrics, "node_matching": closed_node_metrics},
        "retrieval_then_rank": {"retrieval_top_k": retrieval_top_k, "image_retrieval": retrieval_image_metrics, "node_matching": retrieval_node_metrics},
    }
    (out / "closed_set_metrics.json").write_text(json.dumps(metrics["closed_set"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "retrieval_metrics.json").write_text(json.dumps(metrics["retrieval_then_rank"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out / "per_query_results.jsonl", per_query)

    failures = []
    for row in per_query:
        failure_types = []
        if row.get("image_rank") != 1:
            failure_types.append("image_top1_failure")
        if not row.get("image_rank") or int(row["image_rank"]) > 5:
            failure_types.append("image_top5_failure")
        for agg in aggregations:
            if row.get(f"node_rank_{agg}") != 1:
                failure_types.append(f"node_{agg}_top1_failure")
            if row.get(f"tie_status_{agg}") == "tied":
                failure_types.append("tie_case")
        if failure_types:
            item = dict(row)
            item["failure_types"] = sorted(set(failure_types))
            failures.append(item)
    write_jsonl(out / "failure_cases.jsonl", failures)

    lines = [
        "# E1 Original CLIP Cross-Domain Evaluation",
        "",
        f"- dataset: `{dataset}`",
        f"- model: `{MODEL_NAME}`",
        f"- setting: `{SETTING}`",
        f"- queries: {len(per_query)}",
        f"- gallery: {len(gallery)}",
        f"- vector_failures: {len(vector_failures)}",
        "",
        "## Closed-Set Image Retrieval",
    ]
    for key, value in closed_image_metrics.items():
        lines.append(f"- {key}: {value:.4f}")
    lines.append("")
    lines.append("## Closed-Set Node Matching")
    for agg, values in closed_node_metrics.items():
        lines.append(f"- aggregation: {agg}")
        for key, value in values.items():
            lines.append(f"  - {key}: {value:.4f}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"output_dir": str(out), "query_count": len(per_query), "gallery_count": len(gallery), "closed_set": metrics["closed_set"], "retrieval_then_rank": metrics["retrieval_then_rank"], "vector_failures": len(vector_failures)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E1 original CLIP baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visual-statuses", default="direct")
    parser.add_argument("--aggregations", default="max,centroid,top3_mean,hybrid")
    parser.add_argument("--top-k", default="1,3,5,10")
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--media-base-url", default=DEFAULT_MEDIA_BASE_URL)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        dataset=args.dataset,
        output_dir=args.output_dir,
        visual_statuses={item.strip() for item in args.visual_statuses.split(",") if item.strip()},
        aggregations=[item.strip() for item in args.aggregations.split(",") if item.strip()],
        top_ks=[int(item.strip()) for item in args.top_k.split(",") if item.strip()],
        retrieval_top_k=args.retrieval_top_k,
        media_base_url=args.media_base_url,
        force_recompute=args.force_recompute,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
