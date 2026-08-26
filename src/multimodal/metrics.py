"""Evaluation metrics for image retrieval and image-node matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class RankingCase:
    query_id: str
    expected_ids: set[str]
    ranked_ids: list[str]


def recall_at_k(cases: Iterable[RankingCase], k: int) -> float:
    total = 0
    hits = 0
    for case in cases:
        total += 1
        top_k = set(case.ranked_ids[:k])
        hits += int(bool(case.expected_ids.intersection(top_k)))
    return hits / total if total else 0.0


def reciprocal_rank(case: RankingCase) -> float:
    for index, item_id in enumerate(case.ranked_ids, start=1):
        if item_id in case.expected_ids:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(cases: Iterable[RankingCase]) -> float:
    values = [reciprocal_rank(case) for case in cases]
    return sum(values) / len(values) if values else 0.0


def topk_accuracy(cases: Iterable[RankingCase], ks: Sequence[int]) -> dict[str, float]:
    case_list = list(cases)
    metrics = {f"recall@{k}": recall_at_k(case_list, k) for k in ks}
    metrics["mrr"] = mean_reciprocal_rank(case_list)
    return metrics

