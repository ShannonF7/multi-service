"""Aggregate image search hits into node rankings."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def distance_to_similarity(distance: float) -> float:
    """Convert cosine distance to cosine similarity."""
    return 1.0 - float(distance)


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(item) * float(item) for item in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(item) / norm for item in vector]


def dot(left: list[float], right: list[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def mean_vector(vectors: Iterable[list[float]]) -> list[float]:
    values = list(vectors)
    if not values:
        return []
    dim = len(values[0])
    sums = [0.0] * dim
    for vector in values:
        for index, value in enumerate(vector):
            sums[index] += float(value)
    return [value / len(values) for value in sums]


def aggregate_node_scores(image_hits: list[dict], method: str = "max") -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for hit in image_hits:
        node_id = str(hit.get("node_id") or hit.get("source_node_id") or "")
        if not node_id:
            continue
        if hit.get("similarity") is not None:
            similarity = float(hit["similarity"])
        else:
            similarity = distance_to_similarity(float(hit.get("distance") or 0.0))
        grouped[node_id].append(similarity)

    rows = []
    for node_id, scores in grouped.items():
        ordered = sorted(scores, reverse=True)
        if method == "max":
            score = ordered[0]
        elif method == "mean":
            score = sum(ordered) / len(ordered)
        elif method == "top3_mean":
            top = ordered[:3]
            score = sum(top) / len(top)
        elif method == "hybrid":
            top = ordered[:3]
            top3_mean = sum(top) / len(top)
            score = 0.7 * ordered[0] + 0.3 * top3_mean
        else:
            raise ValueError(f"unknown aggregation method: {method}")
        rows.append({"node_id": node_id, "score": score, "matched_images": len(scores), "aggregation": method})
    return rank_with_ties(rows)


def aggregate_centroid_scores(query_vector: list[float], node_vectors: dict[str, list[list[float]]]) -> list[dict]:
    query = normalize(query_vector)
    rows = []
    for node_id, vectors in node_vectors.items():
        centroid = normalize(mean_vector([normalize(vector) for vector in vectors]))
        score = dot(query, centroid) if centroid else -1.0
        rows.append({"node_id": str(node_id), "score": score, "matched_images": len(vectors), "aggregation": "centroid"})
    return rank_with_ties(rows)


def rank_with_ties(rows: list[dict], tolerance: float = 1e-9) -> list[dict]:
    rows = sorted(rows, key=lambda item: (-float(item["score"]), str(item["node_id"])))
    index = 0
    while index < len(rows):
        score = float(rows[index]["score"])
        end = index + 1
        while end < len(rows) and abs(float(rows[end]["score"]) - score) <= tolerance:
            end += 1
        tie_group = [str(item["node_id"]) for item in rows[index:end]]
        rank = index + 1
        tie_status = "tied" if len(tie_group) > 1 else "unique"
        for item in rows[index:end]:
            item["rank"] = rank
            item["tie_status"] = tie_status
            item["tie_group"] = tie_group
        index = end
    return rows
