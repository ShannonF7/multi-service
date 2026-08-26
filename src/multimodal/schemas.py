"""Shared schemas for multimodal E0 evaluation files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

VisualLabelStatus = Literal["direct", "contextual", "shared", "uncertain", "invalid"]


@dataclass
class ImageNodeEvalItem:
    dataset_version: str
    source_scenic_id: str
    asset_id: int
    node_id: str
    node_name: str
    node_type: str
    parent_node_id: str | None
    image_url: str
    label_source: str = "existing_image_binding"
    label_status: str = "confirmed"
    visual_label_status: VisualLabelStatus = "direct"
    duplicate_group_id: str | None = None
    role: str = "gallery"
    usable: bool = True
    exclusion_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageSearchResult:
    asset_id: int
    node_id: str
    distance: float
    similarity: float
    rank: int
    image_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NodeSearchResult:
    node_id: str
    score: float
    rank: int
    matched_images: int
    aggregation: str
    tie_status: str = "unique"
    tie_group: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    dataset: str
    query_count: int
    gallery_count: int
    image_retrieval: dict[str, float]
    node_matching: dict[str, dict[str, float]]
    by_node_type: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
