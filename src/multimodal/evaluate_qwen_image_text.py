"""Evaluate Qwen3-VL-Embedding image-text retrieval on node text profiles."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import bindparam, text

from src.multimodal.node_aggregation import dot, normalize
from src.multimodal.simclr_backfill import DEFAULT_MEDIA_BASE_URL, fetch_image_to_cache
from src.rag.dependencies import get_ai_engine

SETTING = "cross_domain_zero_shot_image_text"


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def rank_of(expected: set[str], ranked: list[str]) -> int | None:
    for index, item in enumerate(ranked, start=1):
        if item in expected:
            return index
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


class QwenEmbeddingEncoder:
    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_path, device=self.device, local_files_only=True, trust_remote_code=True)

    def encode_image(self, image_path: str) -> list[float]:
        vector = self.model.encode([{"image": image_path}], convert_to_numpy=True, normalize_embeddings=True)[0]
        return [float(item) for item in vector.tolist()]

    def encode_texts(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            vectors.extend([[float(item) for item in row.tolist()] for row in encoded])
        return vectors


def load_assets(connection, asset_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not asset_ids:
        return {}
    stmt = text(
        """
        select id, scenic_id, source_scenic_id, source_node_id, source_asset_id, role, is_cover, url, file_hash
        from node_assets where id in :asset_ids
        """
    ).bindparams(bindparam("asset_ids", expanding=True))
    return {int(row["id"]): dict(row) for row in connection.execute(stmt, {"asset_ids": asset_ids}).mappings().all()}


def load_or_create_image_vectors(
    rows: list[dict[str, Any]],
    assets: dict[int, dict[str, Any]],
    output_dir: Path,
    encoder: QwenEmbeddingEncoder,
    media_base_url: str,
    force: bool,
) -> tuple[dict[int, list[float]], list[dict[str, Any]], float]:
    path = output_dir / "image_vectors.jsonl"
    vectors: dict[int, list[float]] = {}
    failures: list[dict[str, Any]] = []
    if path.exists() and not force:
        for row in read_jsonl(str(path)):
            vectors[int(row["asset_id"])] = [float(item) for item in row["vector"]]
    cache_rows = [{"asset_id": aid, "vector": vec} for aid, vec in sorted(vectors.items())]
    elapsed = 0.0
    for asset_id in sorted({int(row["asset_id"]) for row in rows}):
        if asset_id in vectors:
            continue
        asset = assets.get(asset_id)
        if not asset:
            failures.append({"asset_id": asset_id, "error": "asset_not_found"})
            continue
        try:
            image_path = fetch_image_to_cache(str(asset.get("url") or ""), media_base_url)
            started = time.time()
            vector = normalize(encoder.encode_image(str(image_path)))
            elapsed += time.time() - started
            vectors[asset_id] = vector
            cache_rows.append({"asset_id": asset_id, "vector": vector})
        except Exception as exc:
            failures.append({"asset_id": asset_id, "error": str(exc)[:500]})
    write_jsonl(path, cache_rows)
    return vectors, failures, elapsed


def load_or_create_text_vectors(
    profiles: list[dict[str, Any]],
    output_dir: Path,
    encoder: QwenEmbeddingEncoder,
    force: bool,
    batch_size: int,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], float]:
    path = output_dir / "text_vectors.jsonl"
    vectors: dict[str, list[float]] = {}
    failures: list[dict[str, Any]] = []
    if path.exists() and not force:
        for row in read_jsonl(str(path)):
            vectors[str(row["profile_id"])] = [float(item) for item in row["vector"]]
    missing = [profile for profile in profiles if profile["profile_id"] not in vectors]
    elapsed = 0.0
    if missing:
        try:
            started = time.time()
            encoded = encoder.encode_texts([profile["text"] for profile in missing], batch_size=batch_size)
            elapsed += time.time() - started
            for profile, vector in zip(missing, encoded):
                vectors[profile["profile_id"]] = normalize(vector)
        except Exception as exc:
            failures.append({"error": str(exc)[:500], "count": len(missing)})
    write_jsonl(path, [{"profile_id": pid, "vector": vec} for pid, vec in sorted(vectors.items())])
    return vectors, failures, elapsed


def load_node_id_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    text_value = Path(path).read_text(encoding="utf-8").strip()
    if not text_value:
        return set()
    if text_value.startswith("["):
        return {str(item) for item in json.loads(text_value)}
    ids: set[str] = set()
    for line in text_value.splitlines():
        value = line.strip()
        if not value:
            continue
        if value.startswith("{"):
            row = json.loads(value)
            ids.add(str(row.get("node_id") or row.get("id")))
        else:
            ids.add(str(value))
    return ids


def metrics_from_ranks(records: list[dict[str, Any]], rank_key: str, top_ks: list[int]) -> dict[str, float]:
    total = len(records)
    if not total:
        return {**{f"recall@{k}": 0.0 for k in top_ks}, "mrr": 0.0}
    return {
        **{
            f"recall@{k}": sum(1 for row in records if row.get(rank_key) and int(row[rank_key]) <= k) / total
            for k in top_ks
        },
        "mrr": sum(reciprocal_rank(row.get(rank_key)) for row in records) / total,
    }


def evaluate(
    dataset: str,
    profiles_path: str,
    output_dir: str,
    model_key: str,
    model_path: str,
    stage: str,
    variants: set[str],
    top_ks: list[int],
    media_base_url: str,
    force: bool,
    text_query_node_ids: set[str] | None,
    text_batch_size: int,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = [row for row in read_jsonl(dataset) if row.get("usable", True)]
    queries = [row for row in ds if row.get("role") == "query"]
    gallery = [row for row in ds if row.get("role") == "gallery"]
    profiles = []
    for profile in read_jsonl(profiles_path):
        if profile.get("profile_variant") not in variants:
            continue
        item = dict(profile)
        item["profile_id"] = f"{profile['node_id']}::{profile['profile_variant']}"
        profiles.append(item)

    with get_ai_engine().connect() as connection:
        assets = load_assets(connection, sorted({int(row["asset_id"]) for row in ds}))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model_load_started = time.time()
    encoder = QwenEmbeddingEncoder(model_path)
    model_load_seconds = time.time() - model_load_started
    image_vectors, image_failures, image_encode_seconds = load_or_create_image_vectors(
        ds, assets, out, encoder, media_base_url, force
    )
    text_vectors, text_failures, text_encode_seconds = load_or_create_text_vectors(
        profiles, out, encoder, force, text_batch_size
    )
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None

    profiles_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        if profile["profile_id"] in text_vectors:
            profiles_by_variant[profile["profile_variant"]].append(profile)
    gallery_by_node: dict[str, list[int]] = defaultdict(list)
    for row in gallery:
        asset_id = int(row["asset_id"])
        if asset_id in image_vectors:
            gallery_by_node[str(row["node_id"])].append(asset_id)

    results: dict[str, Any] = {
        "model": model_key,
        "model_path": model_path,
        "stage": stage,
        "setting": SETTING,
        "dataset": dataset,
        "profiles": profiles_path,
        "query_count": len(queries),
        "gallery_count": len(gallery),
        "profile_count": len(profiles),
        "text_query_node_count": None if text_query_node_ids is None else len(text_query_node_ids),
        "model_load_seconds": model_load_seconds,
        "image_encode_seconds": image_encode_seconds,
        "text_encode_seconds": text_encode_seconds,
        "peak_memory_mb": peak_memory_mb,
        "image_failures": image_failures,
        "text_failures": text_failures,
        "variants": {},
    }
    per_query: list[dict[str, Any]] = []
    for variant, variant_profiles in sorted(profiles_by_variant.items()):
        image_to_text: list[dict[str, Any]] = []
        for query in queries:
            asset_id = int(query["asset_id"])
            expected_node = str(query["node_id"])
            if asset_id not in image_vectors:
                continue
            ranked = []
            for profile in variant_profiles:
                sim = dot(image_vectors[asset_id], text_vectors[profile["profile_id"]])
                ranked.append(
                    {
                        "profile_id": profile["profile_id"],
                        "node_id": str(profile["node_id"]),
                        "text": profile["text"],
                        "similarity": float(sim),
                    }
                )
            ranked.sort(key=lambda item: (-item["similarity"], item["node_id"], item["profile_id"]))
            ranked_nodes: list[str] = []
            seen: set[str] = set()
            for item in ranked:
                if item["node_id"] not in seen:
                    seen.add(item["node_id"])
                    ranked_nodes.append(item["node_id"])
            record = {
                "direction": "image_to_node_text",
                "variant": variant,
                "query_asset_id": asset_id,
                "expected_node_id": expected_node,
                "rank": rank_of({expected_node}, ranked_nodes),
                "top_texts": ranked[:10],
            }
            image_to_text.append(record)
            per_query.append(record)

        text_to_image: list[dict[str, Any]] = []
        for profile in variant_profiles:
            if text_query_node_ids is not None and str(profile.get("node_id")) not in text_query_node_ids:
                continue
            expected_assets = {str(asset_id) for asset_id in gallery_by_node.get(str(profile["node_id"]), [])}
            if not expected_assets:
                continue
            ranked = []
            for gallery_row in gallery:
                asset_id = int(gallery_row["asset_id"])
                if asset_id not in image_vectors:
                    continue
                sim = dot(text_vectors[profile["profile_id"]], image_vectors[asset_id])
                ranked.append({"asset_id": str(asset_id), "node_id": str(gallery_row["node_id"]), "similarity": float(sim)})
            ranked.sort(key=lambda item: (-item["similarity"], int(item["asset_id"])))
            ranked_assets = [item["asset_id"] for item in ranked]
            record = {
                "direction": "node_text_to_image",
                "variant": variant,
                "profile_id": profile["profile_id"],
                "node_id": str(profile["node_id"]),
                "text": profile["text"],
                "expected_assets": sorted(expected_assets),
                "rank": rank_of(expected_assets, ranked_assets),
                "top_images": ranked[:10],
            }
            text_to_image.append(record)
            per_query.append(record)

        results["variants"][variant] = {
            "image_to_node_text": metrics_from_ranks(image_to_text, "rank", top_ks),
            "node_text_to_image": metrics_from_ranks(text_to_image, "rank", top_ks),
            "image_to_node_text_count": len(image_to_text),
            "node_text_to_image_count": len(text_to_image),
        }

    (out / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out / "per_query_results.jsonl", per_query)
    lines = [
        f"# {stage} {model_key} Image-Text Evaluation",
        "",
        f"- dataset: `{dataset}`",
        f"- profiles: `{profiles_path}`",
        f"- image_failures: {len(image_failures)}",
        f"- text_failures: {len(text_failures)}",
        f"- model_load_seconds: {model_load_seconds:.2f}",
        f"- image_encode_seconds: {image_encode_seconds:.2f}",
        f"- text_encode_seconds: {text_encode_seconds:.2f}",
        f"- peak_memory_mb: {peak_memory_mb if peak_memory_mb is not None else 'n/a'}",
        "",
        "| Variant | Image->Text R@1 | R@5 | R@10 | MRR | Text->Image R@1 | R@5 | R@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, value in results["variants"].items():
        image_text = value["image_to_node_text"]
        text_image = value["node_text_to_image"]
        lines.append(
            f"| {variant} | {image_text.get('recall@1', 0):.4f} | {image_text.get('recall@5', 0):.4f} | "
            f"{image_text.get('recall@10', 0):.4f} | {image_text.get('mrr', 0):.4f} | "
            f"{text_image.get('recall@1', 0):.4f} | {text_image.get('recall@5', 0):.4f} | "
            f"{text_image.get('recall@10', 0):.4f} | {text_image.get('mrr', 0):.4f} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL-Embedding image-text retrieval.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--variants", default="T1,T2,T3,T4")
    parser.add_argument("--top-k", default="1,5,10")
    parser.add_argument("--media-base-url", default=DEFAULT_MEDIA_BASE_URL)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--text-query-node-ids", default=None)
    parser.add_argument("--text-batch-size", type=int, default=16)
    args = parser.parse_args()
    result = evaluate(
        dataset=args.dataset,
        profiles_path=args.profiles,
        output_dir=args.output_dir,
        model_key=args.model_key,
        model_path=args.model_path,
        stage=args.stage,
        variants={item.strip() for item in args.variants.split(",") if item.strip()},
        top_ks=[int(item.strip()) for item in args.top_k.split(",") if item.strip()],
        media_base_url=args.media_base_url,
        force=args.force_recompute,
        text_query_node_ids=load_node_id_filter(args.text_query_node_ids),
        text_batch_size=args.text_batch_size,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "variants": result["variants"],
                "image_failures": len(result["image_failures"]),
                "text_failures": len(result["text_failures"]),
                "image_encode_seconds": result["image_encode_seconds"],
                "text_encode_seconds": result["text_encode_seconds"],
                "peak_memory_mb": result["peak_memory_mb"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
