"""Lazy encoder adapters used by multimodal experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from .model_registry import get_model_spec


class ImageEncoder(ABC):
    model_key: str
    vector_dim: int

    @abstractmethod
    def encode_image(self, image_input: Any) -> List[float]:
        raise NotImplementedError


class TextEncoder(ABC):
    model_key: str
    vector_dim: int

    @abstractmethod
    def encode_text(self, text: str) -> List[float]:
        raise NotImplementedError


class LegacySimCLREncoder(ImageEncoder):
    """Adapter for the existing src.cv.feature_extractor SimCLR model."""

    model_key = "legacy_simclr_128"
    vector_dim = 128

    def __init__(self, model_path: str | None = None):
        from src.cv.feature_extractor import get_feature_extractor

        self._extractor = get_feature_extractor(model_path=model_path)

    def encode_image(self, image_input: Any) -> List[float]:
        vector = self._extractor.extract(image_input)
        if len(vector) != self.vector_dim:
            raise RuntimeError(
                f"{self.model_key} returned dim={len(vector)}, expected {self.vector_dim}"
            )
        return [float(item) for item in vector]


def create_image_encoder(model_key: str, **kwargs: Any) -> ImageEncoder:
    spec = get_model_spec(model_key)
    if model_key == "legacy_simclr_128":
        return LegacySimCLREncoder(**kwargs)
    raise NotImplementedError(
        f"{model_key} is registered for {spec.stage}, but its encoder is not implemented yet"
    )

