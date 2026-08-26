"""Native Qwen adapter with Transformers 5 compatibility shims."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import transformers.utils.generic as transformers_generic

from src.multimodal.benchmark_adapters import ADAPTERS, EmbeddingAdapter


def _identity_decorator(function):
    return function


class Qwen3VLEmbeddingNativeAdapter(EmbeddingAdapter):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        if not hasattr(transformers_generic, "check_model_inputs"):
            transformers_generic.check_model_inputs = _identity_decorator
        module_name = "qwen3_vl_embedding_native"
        module_path = Path(model_path) / "scripts" / "qwen3_vl_embedding.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load Qwen native embedder: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.embedder = module.Qwen3VLEmbedder(model_path)
        self.model = self.embedder.model

    def _encode(self, records: list[dict[str, str]], batch_size: int) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(records), batch_size):
            output = self.embedder.process(records[start : start + batch_size], normalize=True)
            vectors.append(output.detach().cpu().float().numpy())
        return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)

    def encode_images(self, image_paths: list[str], batch_size: int) -> np.ndarray:
        return self._encode([{"image": path} for path in image_paths], batch_size)

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode([{"text": text} for text in texts], batch_size)


def create_native_adapter(model_key: str, model_path: str, device: str) -> EmbeddingAdapter:
    if model_key == "qwen3_vl_embedding":
        return Qwen3VLEmbeddingNativeAdapter(model_path, device)
    try:
        adapter_class = ADAPTERS[model_key]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark model: {model_key}") from exc
    return adapter_class(model_path, device)
