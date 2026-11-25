"""待确认对象的恢复解析。

这里不重新做全领域召回，只在上一轮保存的候选集内解析用户补充。
"""
from __future__ import annotations

import re
from typing import Any, Optional


_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def resolve_pending_clarification(
    question: str,
    pending: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not pending:
        return None
    candidates = list(pending.get("pending_candidates") or pending.get("candidates") or [])
    if not candidates:
        return None
    text = str(question or "").strip()
    selected_index = _selected_index(text, len(candidates))
    if selected_index is not None:
        return {
            "matched": True,
            "candidate": dict(candidates[selected_index]),
            "original_question": str(pending.get("pending_question") or ""),
            "mention": str(pending.get("pending_mention") or ""),
        }

    normalized = _compact(text)
    if not normalized:
        return None
    matched = []
    for index, candidate in enumerate(candidates):
        labels = [
            str(candidate.get("name") or ""),
            *[str(item) for item in (candidate.get("aliases") or [])],
            *[str(item) for item in (candidate.get("hierarchy_path") or [])],
        ]
        if any(_qualifier_matches(normalized, _compact(label)) for label in labels if label):
            matched.append((index, candidate))
    if len(matched) == 1:
        return {
            "matched": True,
            "candidate": dict(matched[0][1]),
            "original_question": str(pending.get("pending_question") or ""),
            "mention": str(pending.get("pending_mention") or ""),
        }
    return None


def _selected_index(text: str, size: int) -> Optional[int]:
    match = re.search(r"第?\s*(\d+)\s*个", text)
    if match:
        index = int(match.group(1)) - 1
        return index if 0 <= index < size else None
    match = re.search(r"第?\s*([一二三四五六七八九十两]+)\s*[个项种]", text)
    if match:
        index = _cn_number(match.group(1)) - 1
        return index if 0 <= index < size else None
    return None


def _qualifier_matches(question: str, label: str) -> bool:
    if not label:
        return False
    question = question.replace("楼", "层")
    label = label.replace("楼", "层")
    if label in question:
        return True
    # “二楼的”“B座的”等补充限定常带“的”，按层级标签匹配即可。
    for suffix in ("的", "那栋", "那个", "这一层", "那一层"):
        if question.replace(suffix, "") == label:
            return True
    return False


def _cn_number(value: str) -> int:
    if value == "十":
        return 10
    if "十" in value:
        left, _, right = value.partition("十")
        return (_CN_NUM.get(left, 1) * 10 if left else 10) + (_CN_NUM.get(right, 0) if right else 0)
    return _CN_NUM.get(value, 0)


def _compact(value: str) -> str:
    return re.sub(r"[\s，。！？?；;、]+", "", str(value or "")).casefold()
