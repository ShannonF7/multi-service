"""Evaluate E0-A SimCLR cross-domain baseline.

The closed-set evaluation ranks every gallery image and every gallery node for
all query images. It is separate from retrieval-then-rank simulation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from sqlalchemy import bindparam, text

from src.rag.dependencies import get_ai_engine
from src.multimodal.metrics import RankingCase, topk_accuracy
from src.multimodal.node_aggregation import aggregate_centroid_scores, aggregate_node_scores, distance_to_similarity
from src.multimodal.simclr_backfill import MODEL_NAME, to_pgvector_literal
from src.multimodal.simclr_search import get_asset_embedding

CANONICAL_MODEL = "simclr_source_scenic_v1_128"


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rank_of(expected: set[str], ranked: list[str]) -> int | None:
    for index, item in enumerate(ranked, start=1):
        if item in expected:
            return index
    return None


def tie_aware_hit(expected: set[str], rankings: list[dict], k: int) -> bool:
    for row in rankings:
        if int(row.get("rank") or 999999) > k:
            continue
        tie_group = {str(item) for item in row.get("tie_group") or [row.get("node_id")]}
        if expected.intersection(tie_group):
            return True
    return False


def reciprocal_rank_from_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def load_embeddings(connection, asset_ids: list[int]) -> dict[int, list[float]]:
    if not asset_ids:
        return {}
    stmt = text(
        """
        select asset_id, embedding
        from image_embeddings
        where model_name = :model_name and asset_id in :asset_ids
        """
    ).bindparams(bindparam("asset_ids", expanding=True))
    rows = connection.execute(stmt, {"model_name": MODEL_NAME, "asset_ids": asset_ids}).fetchall()
    out = {}
    for asset_id, embedding in rows:
        if isinstance(embedding, str):
            vector = [float(item) for item in embedding.strip("[]").split(",") if item]
        else:
            vector = [float(item) for item in embedding]
        out[int(asset_id)] = vector
    return out


def search_gallery(connection, query_vector: list[float], gallery_asset_ids: list[int]) -> list[dict]:
    stmt = text(
        """
        select a.id as asset_id, a.source_node_id as node_id, a.url as image_url,
               (e.embedding <-> cast(:embedding as vector)) as distance
        from image_embeddings e
        join node_assets a on a.id = e.asset_id
        where e.model_name = :model_name
          and a.id in :gallery_asset_ids
        order by e.embedding <-> cast(:embedding as vector) asc, a.id asc
        """
    ).bindparams(bindparam("gallery_asset_ids", expanding=True))
    rows = connection.execute(
        stmt,
        {
            "embedding": to_pgvector_literal(query_vector),
            "model_name": MODEL_NAME,
            "gallery_asset_ids": gallery_asset_ids,
        },
    ).mappings().all()
    hits = []
    for index, row in enumerate(rows, start=1):
        item = dict(row)
        item["rank"] = index
        item["similarity"] = distance_to_similarity(float(item["distance"]))
        hits.append(item)
    return hits


def ranking_metrics(cases: list[RankingCase], top_ks: list[int]) -> dict[str, float]:
    return topk_accuracy(cases, top_ks)


def node_metrics(records: list[dict], aggregation: str, top_ks: list[int]) -> dict[str, float]:
    total = len(records)
    if not total:
        return {**{f"strict_top{k}": 0.0 for k in top_ks}, **{f"tie_aware_top{k}": 0.0 for k in top_ks}, "mrr": 0.0, "tie_case_rate": 0.0, "rank_null": 0.0}
    metrics = {}
    for k in top_ks:
        metrics[f"strict_top{k}"] = sum(1 for row in records if row.get(f"node_rank_{aggregation}") and int(row[f"node_rank_{aggregation}"]) <= k) / total
        metrics[f"tie_aware_top{k}"] = sum(1 for row in records if tie_aware_hit({str(row["query_node_id"])}, row["node_rankings"].get(aggregation, []), k)) / total
    metrics["mrr"] = sum(reciprocal_rank_from_rank(row.get(f"node_rank_{aggregation}")) for row in records) / total
    metrics["tie_case_rate"] = sum(1 for row in records if row.get(f"tie_status_{aggregation}") == "tied") / total
    metrics["rank_null"] = sum(1 for row in records if row.get(f"node_rank_{aggregation}") is None) / total
    return metrics


def evaluate(dataset: str, aggregations: list[str], top_ks: list[int], output_dir: str, visual_statuses: set[str], retrieval_top_k: int) -> dict:
    all_rows = [row for row in read_jsonl(dataset) if row.get("usable", True)]
    rows = [row for row in all_rows if str(row.get("visual_label_status") or "direct") in visual_statuses]
    queries = [row for row in rows if row.get("role") == "query"]
    gallery = [row for row in rows if row.get("role") == "gallery"]
    gallery_asset_ids = [int(row["asset_id"]) for row in gallery]
    gallery_by_node: dict[str, list[int]] = defaultdict(list)
    for row in gallery:
        gallery_by_node[str(row["node_id"])].append(int(row["asset_id"]))

    image_cases: list[RankingCase] = []
    retrieval_image_cases: list[RankingCase] = []
    per_query: list[dict] = []
    missing_embedding: list[dict] = []

    with get_ai_engine().connect() as connection:
        gallery_vectors = load_embeddings(connection, gallery_asset_ids)
        node_vectors = {node_id: [gallery_vectors[asset_id] for asset_id in ids if asset_id in gallery_vectors] for node_id, ids in gallery_by_node.items()}
        for query in queries:
            q_asset = int(query["asset_id"])
            expected_node = str(query["node_id"])
            try:
                query_vector = get_asset_embedding(connection, q_asset)
            except Exception as exc:
                missing_embedding.append({"asset_id": q_asset, "error": str(exc)[:300]})
                continue

            hits = search_gallery(connection, query_vector, gallery_asset_ids)
            ranked_assets = [str(hit["asset_id"]) for hit in hits]
            expected_assets = {str(row["asset_id"]) for row in gallery if str(row["node_id"]) == expected_node}
            image_cases.append(RankingCase(str(q_asset), expected_assets, ranked_assets))
            retrieval_image_cases.append(RankingCase(str(q_asset), expected_assets, ranked_assets[:retrieval_top_k]))

            record = {
                "evaluation_stage": "E0-A",
                "model": CANONICAL_MODEL,
                "evaluation_setting": "cross_domain_zero_shot",
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

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": CANONICAL_MODEL,
        "evaluation_stage": "E0-A",
        "evaluation_setting": "cross_domain_zero_shot",
        "dataset": dataset,
        "visual_statuses": sorted(visual_statuses),
        "query_count": len(per_query),
        "gallery_count": len(gallery),
        "missing_embedding": missing_embedding,
        "closed_set": {"image_retrieval": closed_image_metrics, "node_matching": closed_node_metrics},
        "retrieval_then_rank": {"retrieval_top_k": retrieval_top_k, "image_retrieval": retrieval_image_metrics, "node_matching": retrieval_node_metrics},
    }
    (out / "closed_set_metrics.json").write_text(json.dumps(metrics["closed_set"], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "retrieval_metrics.json").write_text(json.dumps(metrics["retrieval_then_rank"], ensure_ascii=False, indent=2), encoding="utf-8")
    tie_metrics = {agg: {"tie_case_rate": values.get("tie_case_rate", 0.0), "rank_null": values.get("rank_null", 0.0)} for agg, values in closed_node_metrics.items()}
    (out / "tie_metrics.json").write_text(json.dumps(tie_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "per_query_results.jsonl").open("w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with (out / "failure_cases.jsonl").open("w", encoding="utf-8") as f:
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
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    lines = [
        "# E0-A SimCLR Cross-Domain Evaluation",
        "",
        f"- dataset: `{dataset}`",
        f"- model: `{CANONICAL_MODEL}`",
        "- setting: `cross_domain_zero_shot`",
        f"- queries: {len(per_query)}",
        f"- gallery: {len(gallery)}",
        f"- missing_embedding: {len(missing_embedding)}",
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
    return {"output_dir": str(out), "query_count": len(per_query), "gallery_count": len(gallery), "closed_set": metrics["closed_set"], "retrieval_then_rank": metrics["retrieval_then_rank"], "missing_embedding": len(missing_embedding)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E0-A SimCLR baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=CANONICAL_MODEL)
    parser.add_argument("--evaluation-mode", default="closed-set")
    parser.add_argument("--aggregations", default="max,centroid,top3_mean,hybrid")
    parser.add_argument("--top-k", default="1,3,5,10")
    parser.add_argument("--visual-statuses", default="direct")
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    aggregations = [item.strip() for item in args.aggregations.split(",") if item.strip()]
    top_ks = [int(item.strip()) for item in args.top_k.split(",") if item.strip()]
    visual_statuses = {item.strip() for item in args.visual_statuses.split(",") if item.strip()}
    print(json.dumps(evaluate(args.dataset, aggregations, top_ks, args.output_dir, visual_statuses, args.retrieval_top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
