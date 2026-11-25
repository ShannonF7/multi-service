"""Model-specific adapters for a shared multimodal benchmark protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


def _as_tensor(value: Any, names: tuple[str, ...]) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    for name in names:
        candidate = getattr(value, name, None)
        if torch.is_tensor(candidate):
            return candidate
    raise RuntimeError(f"model output has none of {names}")


def _normalize(value: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.clip(denominator, 1e-12, None)


class EmbeddingAdapter(ABC):
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def encode_images(self, image_paths: list[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        for name in ("model", "processor", "tokenizer"):
            if hasattr(self, name):
                delattr(self, name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class HFClipAdapter(EmbeddingAdapter):
    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True, use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        ).to(device)
        self.model.eval()

    def encode_images(self, image_paths: list[str], batch_size: int) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            images = [Image.open(path).convert("RGB") for path in image_paths[start : start + batch_size]]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model.get_image_features(**inputs)
            tensor = _as_tensor(output, ("image_embeds", "pooler_output"))
            vectors.append(tensor.detach().cpu().float().numpy())
        return _normalize(np.concatenate(vectors, axis=0))

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(
                texts[start : start + batch_size], padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = self.model.get_text_features(**inputs)
            tensor = _as_tensor(output, ("text_embeds", "pooler_output"))
            vectors.append(tensor.detach().cpu().float().numpy())
        return _normalize(np.concatenate(vectors, axis=0))


class OpenAIClipAdapter(HFClipAdapter):
    pass


class ChineseClipAdapter(HFClipAdapter):
    pass


class SigLIP2Adapter(HFClipAdapter):
    pass


class Qwen3VLEmbeddingAdapter(EmbeddingAdapter):
    """Adapter for Qwen's multimodal SentenceTransformer module.

    The model expects structured multimodal records for both modalities. Bare
    strings use the generic SentenceTransformer path and fail in its processor.
    """

    def __init__(self, model_path: str, device: str):
        super().__init__(model_path, device)
        self.model = SentenceTransformer(
            model_path, device=device, local_files_only=True, trust_remote_code=True
        )

    def _encode(self, records: list[dict[str, str]], batch_size: int) -> np.ndarray:
        output = self.model.encode(
            records,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(output, dtype=np.float32)

    def encode_images(self, image_paths: list[str], batch_size: int) -> np.ndarray:
        return self._encode([{"image": path} for path in image_paths], batch_size)

    def encode_texts(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode([{"text": text} for text in texts], batch_size)


ADAPTERS = {
    "openai_clip": OpenAIClipAdapter,
    "chinese_clip": ChineseClipAdapter,
    "siglip2": SigLIP2Adapter,
    "qwen3_vl_embedding": Qwen3VLEmbeddingAdapter,
}


def create_adapter(model_key: str, model_path: str, device: str) -> EmbeddingAdapter:
    try:
        adapter_class = ADAPTERS[model_key]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark model: {model_key}") from exc
    return adapter_class(model_path, device)
