"""Evaluate score-level fusion for two image encoders on an image-node dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.multimodal.metrics import RankingCase, topk_accuracy
from src.multimodal.node_aggregation import aggregate_node_scores, dot, normalize


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


def load_vectors(path: str | Path) -> dict[int, list[float]]:
    vectors: dict[int, list[float]] = {}
    for row in read_jsonl(path):
        vectors[int(row["asset_id"])] = normalize([float(x) for x in row["vector"]])
    return vectors


def rank_of(expected: set[str], ranked: list[str]) -> int | None:
    for index, item in enumerate(ranked, start=1):
        if item in expected:
            return index
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def metrics_from_ranks(ranks: list[int | None], top_ks: list[int]) -> dict[str, float]:
    total = len(ranks)
    if not total:
        return {**{f"recall@{k}": 0.0 for k in top_ks}, "mrr": 0.0}
    result = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / total for k in top_ks}
    result["mrr"] = sum(reciprocal_rank(r) for r in ranks) / total
    return result


def node_metrics(records: list[dict[str, Any]], aggregation: str, top_ks: list[int]) -> dict[str, float]:
    ranks = [row.get(f"node_rank_{aggregation}") for row in records]
    result = {}
    base = metrics_from_ranks(ranks, top_ks)
    for key, value in base.items():
        if key.startswith("recall@"):
            result["strict_top" + key.split("@", 1)[1]] = value
        else:
            result[key] = value
    result["rank_null"] = sum(1 for rank in ranks if rank is None) / len(ranks) if ranks else 0.0
    return result


def evaluate(
    dataset: str,
    left_vectors_path: str,
    right_vectors_path: str,
    output_dir: str,
    left_key: str,
    right_key: str,
    alphas: list[float],
    top_ks: list[int],
    aggregations: list[str],
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(dataset) if row.get("usable", True) is not False]
    queries = [row for row in rows if row.get("role") == "query"]
    gallery = [row for row in rows if row.get("role") == "gallery"]
    left_vectors = load_vectors(left_vectors_path)
    right_vectors = load_vectors(right_vectors_path)

    common_assets = set(left_vectors).intersection(right_vectors)
    queries = [row for row in queries if int(row["asset_id"]) in common_assets]
    gallery = [row for row in gallery if int(row["asset_id"]) in common_assets]

    result: dict[str, Any] = {
        "dataset": dataset,
        "left_model": left_key,
        "right_model": right_key,
        "query_count": len(queries),
        "gallery_count": len(gallery),
        "common_asset_count": len(common_assets),
        "alphas": {},
    }
    all_alpha_cases: list[dict[str, Any]] = []

    for alpha in alphas:
        image_cases: list[RankingCase] = []
        image_ranks: list[int | None] = []
        per_query: list[dict[str, Any]] = []
        for query in queries:
            q_asset = int(query["asset_id"])
            expected_node = str(query["node_id"])
            hits = []
            for item in gallery:
                asset_id = int(item["asset_id"])
                left_score = dot(left_vectors[q_asset], left_vectors[asset_id])
                right_score = dot(right_vectors[q_asset], right_vectors[asset_id])
                score = alpha * left_score + (1.0 - alpha) * right_score
                hits.append(
                    {
                        "asset_id": asset_id,
                        "node_id": str(item["node_id"]),
                        "similarity": float(score),
                        "left_similarity": float(left_score),
                        "right_similarity": float(right_score),
                    }
                )
            hits.sort(key=lambda item: (-float(item["similarity"]), int(item["asset_id"])))
            for index, hit in enumerate(hits, start=1):
                hit["rank"] = index
            ranked_assets = [str(hit["asset_id"]) for hit in hits]
            expected_assets = {str(row["asset_id"]) for row in gallery if str(row["node_id"]) == expected_node}
            image_cases.append(RankingCase(str(q_asset), expected_assets, ranked_assets))
            image_rank = rank_of(expected_assets, ranked_assets)
            image_ranks.append(image_rank)

            record: dict[str, Any] = {
                "alpha_left": alpha,
                "query_asset_id": q_asset,
                "query_node_id": expected_node,
                "image_rank": image_rank,
                "image_hits": hits[:50],
                "node_rankings": {},
            }
            for aggregation in aggregations:
                node_rankings = aggregate_node_scores(hits, aggregation)
                ranked_nodes = [str(item["node_id"]) for item in node_rankings]
                record["node_rankings"][aggregation] = node_rankings[:50]
                record[f"node_rank_{aggregation}"] = rank_of({expected_node}, ranked_nodes)
            per_query.append(record)

        alpha_key = f"{alpha:.2f}"
        result["alphas"][alpha_key] = {
            "image_retrieval": topk_accuracy(image_cases, top_ks),
            "image_mrr": metrics_from_ranks(image_ranks, top_ks)["mrr"],
            "node_matching": {aggregation: node_metrics(per_query, aggregation, top_ks) for aggregation in aggregations},
        }
        all_alpha_cases.extend(per_query)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out / "per_query_results.jsonl", all_alpha_cases)

    lines = [
        f"# {left_key} + {right_key} Image Fusion",
        "",
        f"- dataset: `{dataset}`",
        f"- query_count: {result['query_count']}",
        f"- gallery_count: {result['gallery_count']}",
        "",
        "| alpha_left | Image R@1 | R@5 | R@10 | MRR | Node hybrid Top1 | Top5 | Top10 | MRR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha_key, metrics in result["alphas"].items():
        image = metrics["image_retrieval"]
        node = metrics["node_matching"].get("hybrid") or metrics["node_matching"][aggregations[0]]
        lines.append(
            f"| {alpha_key} | {image.get('recall@1', 0):.4f} | {image.get('recall@5', 0):.4f} | "
            f"{image.get('recall@10', 0):.4f} | {metrics.get('image_mrr', 0):.4f} | "
            f"{node.get('strict_top1', 0):.4f} | {node.get('strict_top5', 0):.4f} | "
            f"{node.get('strict_top10', 0):.4f} | {node.get('mrr', 0):.4f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate score fusion between two image encoders.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--left-vectors", required=True)
    parser.add_argument("--right-vectors", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--left-key", default="left")
    parser.add_argument("--right-key", default="right")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--top-k", default="1,3,5,10")
    parser.add_argument("--aggregations", default="max,top3_mean,hybrid")
    args = parser.parse_args()
    result = evaluate(
        args.dataset,
        args.left_vectors,
        args.right_vectors,
        args.output_dir,
        args.left_key,
        args.right_key,
        [float(x) for x in args.alphas.split(",") if x.strip()],
        [int(x) for x in args.top_k.split(",") if x.strip()],
        [x.strip() for x in args.aggregations.split(",") if x.strip()],
    )
    print(json.dumps({"output_dir": args.output_dir, "alphas": result["alphas"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
