"""Unified benchmark with isolated Qwen image and text model instances."""
from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import torch

from src.multimodal import benchmark_multimodal_resources_v2 as benchmark
from src.multimodal.benchmark_adapters_native_v3 import create_native_adapter
ORIGINAL_RUN_MODEL = benchmark.run_model


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def _reset_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


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
    if model_key != "qwen3_vl_embedding":
        return ORIGINAL_RUN_MODEL(
            model_key, image_ids, image_paths, text_ids, texts, protocol,
            fixed_image_batch, fixed_text_batch
        )

    config = benchmark.MODELS[model_key]
    image_batch = fixed_image_batch if protocol == "fixed" else int(config["max_image_batch"])
    text_batch = fixed_text_batch if protocol == "fixed" else int(config["max_text_batch"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _reset_cuda()
    started = time.perf_counter()
    image_adapter = create_native_adapter(model_key, str(config["path"]), device)
    _sync()
    image_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    image_vectors = image_adapter.encode_images(image_paths, image_batch)
    _sync()
    image_seconds = time.perf_counter() - started
    image_peak_mb = _peak_mb()
    image_adapter.close()
    del image_adapter

    _reset_cuda()
    started = time.perf_counter()
    text_adapter = create_native_adapter(model_key, str(config["path"]), device)
    _sync()
    text_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    text_vectors = text_adapter.encode_texts(texts, text_batch)
    _sync()
    text_seconds = time.perf_counter() - started
    text_peak_mb = _peak_mb()

    if image_vectors.shape[0] != len(image_ids) or text_vectors.shape[0] != len(text_ids):
        raise RuntimeError(
            f"coverage failure: images={image_vectors.shape[0]}/{len(image_ids)}, "
            f"texts={text_vectors.shape[0]}/{len(text_ids)}"
        )
    result = {
        "model_key": model_key,
        "model": config["label"],
        "protocol": protocol,
        "dataset_manifest": {
            "image_count": len(image_ids),
            "image_id_sha256": benchmark.fingerprint(image_ids),
            "text_count": len(text_ids),
            "text_id_sha256": benchmark.fingerprint(text_ids),
            "coverage_percent": 100.0,
        },
        "model_path": config["path"],
        "model_disk_size_bytes": benchmark.dir_size_bytes(str(config["path"])),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "precision": "FP32",
        "image_batch_size": image_batch,
        "text_batch_size": text_batch,
        "model_load_seconds": image_load_seconds,
        "text_model_reload_seconds": text_load_seconds,
        "peak_memory_mb": max(value for value in (image_peak_mb, text_peak_mb) if value is not None),
        "image_peak_memory_mb": image_peak_mb,
        "text_peak_memory_mb": text_peak_mb,
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
        "modality_instances_isolated": True,
    }
    text_adapter.close()
    del image_vectors, text_vectors, text_adapter
    _reset_cuda()
    return result


benchmark.create_adapter = create_native_adapter
benchmark.run_model = run_model


if __name__ == "__main__":
    benchmark.main()
