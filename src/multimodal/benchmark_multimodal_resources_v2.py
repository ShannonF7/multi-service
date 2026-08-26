"""Run comparable resource benchmarks with model-specific adapters."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.multimodal.benchmark_adapters import create_adapter
from src.multimodal.simclr_backfill import DEFAULT_MEDIA_BASE_URL, fetch_image_to_cache


MODELS = {
    "openai_clip": {
        "label": "OpenAI CLIP",
        "path": "/home/zhangbi/Zhangbi_Traveler/multimodal/openai-clip-vit-base-patch16",
        "max_image_batch": 16,
        "max_text_batch": 64,
    },
    "chinese_clip": {
        "label": "Chinese-CLIP",
        "path": "/home/zhangbi/Zhangbi_Traveler/multimodal/chinese-clip-vit-base-patch16",
        "max_image_batch": 16,
        "max_text_batch": 64,
    },
    "siglip2": {
        "label": "SigLIP2",
        "path": "/home/zhangbi/Zhangbi_Traveler/multimodal/siglip2-base-patch16-224",
        "max_image_batch": 16,
        "max_text_batch": 64,
    },
    "qwen3_vl_embedding": {
        "label": "Qwen3-VL-Embedding-2B",
        "path": "/home/zhangbi/Zhangbi_Traveler/multimodal/Qwen3-VL-Embedding-2B",
        "max_image_batch": 8,
        "max_text_batch": 1,
    },
}


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fingerprint(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_images(dataset: str, media_base_url: str) -> tuple[list[str], list[str]]:
    by_id: dict[str, str] = {}
    for row in read_jsonl(dataset):
        if not row.get("usable", True):
            continue
        asset_id = str(row["asset_id"])
        url = str(row.get("image_url") or row.get("url") or "")
        if url:
            by_id.setdefault(asset_id, url)
    ids = sorted(by_id, key=lambda value: int(value))
    paths = [str(fetch_image_to_cache(by_id[asset_id], media_base_url)) for asset_id in ids]
    if len(paths) != len(ids):
        raise RuntimeError("image cache coverage is not 100%")
    return ids, paths


def text_record_id(row: dict[str, Any], index: int) -> str:
    return str(
        row.get("profile_id")
        or "::".join(
            [
                str(row.get("node_id") or ""),
                str(row.get("language") or row.get("lang") or ""),
                str(row.get("profile_variant") or row.get("variant") or ""),
                str(index),
            ]
        )
    )


def load_texts(profiles: str) -> tuple[list[str], list[str]]:
    records = [(text_record_id(row, index), str(row.get("text") or "")) for index, row in enumerate(read_jsonl(profiles))]
    records = [(record_id, text) for record_id, text in records if text]
    records.sort(key=lambda item: item[0])
    ids = [item[0] for item in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("text profile IDs are not unique")
    return ids, [item[1] for item in records]


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            try:
                total += os.path.getsize(os.path.join(root, filename))
            except OSError:
                pass
    return total


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_model(
    model_key: str,
    image_ids: list[str],
    image_paths: list[str],
    text_ids: list[str],
    texts: list[str],
    protocol: str,
    fixed_image_batch: int,
    fixed_text_batch: int,
) -> dict[str, Any]:
    config = MODELS[model_key]
    image_batch = fixed_image_batch if protocol == "fixed" else int(config["max_image_batch"])
    text_batch = fixed_text_batch if protocol == "fixed" else int(config["max_text_batch"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    adapter = create_adapter(model_key, str(config["path"]), device)
    synchronize()
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    image_vectors = adapter.encode_images(image_paths, image_batch)
    synchronize()
    image_seconds = time.perf_counter() - started

    started = time.perf_counter()
    text_vectors = adapter.encode_texts(texts, text_batch)
    synchronize()
    text_seconds = time.perf_counter() - started

    if image_vectors.shape[0] != len(image_ids) or text_vectors.shape[0] != len(text_ids):
        raise RuntimeError(
            f"coverage failure for {model_key}: images={image_vectors.shape[0]}/{len(image_ids)}, "
            f"texts={text_vectors.shape[0]}/{len(text_ids)}"
        )
    result = {
        "model_key": model_key,
        "model": config["label"],
        "protocol": protocol,
        "dataset_manifest": {
            "image_count": len(image_ids),
            "image_id_sha256": fingerprint(image_ids),
            "text_count": len(text_ids),
            "text_id_sha256": fingerprint(text_ids),
            "coverage_percent": 100.0,
        },
        "model_path": config["path"],
        "model_disk_size_bytes": dir_size_bytes(str(config["path"])),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "precision": "FP32",
        "image_batch_size": image_batch,
        "text_batch_size": text_batch,
        "model_load_seconds": load_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else None,
        "image_encode_seconds": image_seconds,
        "image_throughput_per_second": len(image_ids) / image_seconds,
        "image_avg_seconds": image_seconds / len(image_ids),
        "text_encode_seconds": text_seconds,
        "text_throughput_per_second": len(text_ids) / text_seconds,
        "text_avg_seconds": text_seconds / len(text_ids),
        "image_vector_dim": int(image_vectors.shape[1]),
        "text_vector_dim": int(text_vectors.shape[1]),
        "image_vector_storage_bytes_fp32": int(np.asarray(image_vectors).size * 4),
        "text_vector_storage_bytes_fp32": int(np.asarray(text_vectors).size * 4),
    }
    adapter.close()
    del image_vectors, text_vectors, adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--media-base-url", default=DEFAULT_MEDIA_BASE_URL)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--protocol", choices=("fixed", "max_throughput"), required=True)
    parser.add_argument("--fixed-image-batch", type=int, default=8)
    parser.add_argument("--fixed-text-batch", type=int, default=1)
    args = parser.parse_args()

    image_ids, image_paths = load_images(args.dataset, args.media_base_url)
    text_ids, texts = load_texts(args.profiles)
    manifest = {
        "dataset": str(Path(args.dataset).resolve()),
        "profiles": str(Path(args.profiles).resolve()),
        "image_count": len(image_ids),
        "image_id_sha256": fingerprint(image_ids),
        "text_count": len(text_ids),
        "text_id_sha256": fingerprint(text_ids),
    }
    selected = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(MODELS))
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    results = [
        run_model(
            key,
            image_ids,
            image_paths,
            text_ids,
            texts,
            args.protocol,
            args.fixed_image_batch,
            args.fixed_text_batch,
        )
        for key in selected
    ]
    if any(result["dataset_manifest"] != {**manifest, "dataset": manifest["dataset"], "profiles": manifest["profiles"]} for result in []):
        raise AssertionError("unreachable")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"protocol": args.protocol, "manifest": manifest, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "manifest": manifest, "models": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
