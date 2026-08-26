"""Model registry for multimodal experiments.

The registry records experiment intent and expected vector dimensions. Heavy
model dependencies are loaded lazily by encoders.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable


@dataclass(frozen=True)
class ModelSpec:
    key: str
    stage: str
    family: str
    vector_dim: int | None
    supports_image: bool
    supports_text: bool
    description: str
    status: str = "planned"
    metadata: dict[str, Any] | None = None


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "simclr_source_scenic_v1_128": ModelSpec(
        key="simclr_source_scenic_v1_128",
        stage="E0-A",
        family="simclr",
        vector_dim=128,
        supports_image=True,
        supports_text=False,
        description=(
            "Existing SimCLR ResNet50 image encoder trained on a source scenic "
            "domain and evaluated zero-shot on the target scenic domain."
        ),
        status="available",
        metadata={
            "training_domain": "source_scenic",
            "evaluation_domain": "scenic_4",
            "target_domain_seen": False,
            "evaluation_setting": "cross_domain_zero_shot",
            "legacy_alias": "legacy_simclr_128",
        },
    ),
    "legacy_simclr_128": ModelSpec(
        key="legacy_simclr_128",
        stage="E0-A",
        family="simclr",
        vector_dim=128,
        supports_image=True,
        supports_text=False,
        description="Backward-compatible alias for simclr_source_scenic_v1_128.",
        status="available",
        metadata={"alias_for": "simclr_source_scenic_v1_128"},
    ),
    "openai_clip_vit_b32": ModelSpec(
        key="openai_clip_vit_b32",
        stage="E1",
        family="clip",
        vector_dim=512,
        supports_image=True,
        supports_text=True,
        description="Original CLIP baseline for image-image, text-image, image-text retrieval.",
        status="planned",
    ),
    "chinese_clip": ModelSpec(
        key="chinese_clip",
        stage="E2",
        family="chinese_clip",
        vector_dim=None,
        supports_image=True,
        supports_text=True,
        description="Chinese adapted CLIP baseline for Chinese cultural-tourism queries.",
        status="planned",
    ),
    "qwen_vl_embedding": ModelSpec(
        key="qwen_vl_embedding",
        stage="E3",
        family="qwen_vl",
        vector_dim=None,
        supports_image=True,
        supports_text=True,
        description="Stronger multimodal embedding model; exact checkpoint and dim are runtime config.",
        status="planned",
    ),
    "fusion_context_v1": ModelSpec(
        key="fusion_context_v1",
        stage="E4",
        family="fusion",
        vector_dim=None,
        supports_image=True,
        supports_text=True,
        description="Fusion scorer: visual, OCR, caption, node type, parent/spatial context.",
        status="planned",
    ),
}


def get_model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_key]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"unknown model_key={model_key!r}; known: {known}") from exc


def canonical_model_key(model_key: str) -> str:
    spec = get_model_spec(model_key)
    metadata = spec.metadata or {}
    return str(metadata.get("alias_for") or spec.key)


def list_model_specs() -> Iterable[ModelSpec]:
    return MODEL_REGISTRY.values()
