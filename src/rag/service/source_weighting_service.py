"""Source provenance and authority scoring for semantic completion evidence."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

AUTHORITY_SCORES = {
    "official": 0.95,
    "government": 0.92,
    "museum_or_institution": 0.88,
    "academic": 0.85,
    "encyclopedia": 0.72,
    "news": 0.62,
    "travel_platform": 0.55,
    "self_media": 0.35,
    "unknown": 0.45,
}

# Keep provenance classification data in UTF-8 source instead of opaque
# literals. Explicit metadata still takes precedence below.
SOURCE_KEYWORDS = {
    "government": (
        "政府", "国务院", "教育部", "文化和旅游部", "文物局", "自然资源部",
        "住建部", "交通运输部", "卫生健康委", "人民政府", "gov.cn",
    ),
    "official": (
        "官网", "官方网站", "官方", "管理委员会", "景区管理", "博物馆",
        "纪念馆", "研究院", "研究所", "museum", "official",
    ),
    "academic": (
        "大学", "学院", "高校", "学术", "论文", "期刊", "cnki", ".edu",
    ),
    "news": (
        "新华社", "人民日报", "中国新闻网", "中新网", "新闻", "xinhuanet",
        "people.com.cn", "chinanews",
    ),
}


def _metadata(source: dict[str, Any] | None) -> dict[str, Any]:
    meta = (source or {}).get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def provenance_type(source: dict[str, Any] | None) -> str:
    source = source or {}
    source_type = str(source.get("source_type") or "").lower()
    if source_type.startswith("domain_kb"):
        return "local_kb"
    if source_type in {"provided", "provided_evidence"}:
        return "provided"
    if source_type.startswith("web") or source.get("source_url"):
        return "web"
    if source_type.startswith("graph"):
        return "graph"
    return "unknown"


def retrieval_method(source: dict[str, Any] | None) -> str:
    source = source or {}
    source_type = str(source.get("source_type") or "").lower()
    if source_type.endswith("_vector") or "vector" in source_type:
        return "vector"
    if source_type.endswith("_keyword") or "keyword" in source_type:
        return "keyword"
    if source_type == "web_extractor":
        return "web_extractor"
    if source_type.startswith("web"):
        return "web_search"
    if source_type in {"provided", "provided_evidence"}:
        return "provided"
    if source_type.startswith("graph"):
        return "graph_lookup"
    return source_type or "unknown"


def infer_authority_class(source: dict[str, Any] | None) -> str:
    source = source or {}
    meta = _metadata(source)
    explicit = str(source.get("authority_class") or meta.get("authority_class") or "").strip().lower()
    if explicit:
        return explicit
    url = str(source.get("source_url") or meta.get("original_url") or "").strip().lower()
    host = urlparse(url).netloc.lower()
    title = " ".join(str(source.get(k) or "") for k in ("title", "source")).lower()
    text = f"{host} {title}"
    if host.endswith(".gov.cn") or ".gov." in host or any(
        key in text for key in SOURCE_KEYWORDS["government"]
    ):
        return "government"
    if any(key in text for key in SOURCE_KEYWORDS["official"]):
        return "official"
    if any(key in text for key in SOURCE_KEYWORDS["academic"]):
        return "academic"
    if any(key in text for key in SOURCE_KEYWORDS["news"]):
        return "news"
    if any(key in host for key in ("baike.baidu", "wikipedia", "wiki")):
        return "encyclopedia"
    if any(key in host for key in ("ctrip", "mafengwo", "trip", "dianping")):
        return "travel_platform"
    if provenance_type(source) in {"local_kb", "provided"}:
        if meta.get("verified") is True:
            return "official"
        return "museum_or_institution"
    return "unknown"


def authority_score(authority_class: str | None) -> float:
    return float(AUTHORITY_SCORES.get(str(authority_class or "unknown").lower(), AUTHORITY_SCORES["unknown"]))


def source_weight(source: dict[str, Any] | None) -> dict[str, Any]:
    source = source or {}
    ptype = provenance_type(source)
    method = retrieval_method(source)
    aclass = infer_authority_class(source)
    base = authority_score(aclass)
    method_bonus = {
        "provided": 0.08,
        "vector": 0.04,
        "keyword": 0.0,
        "web_extractor": 0.04,
        "web_search": 0.0,
        "graph_lookup": -0.05,
    }.get(method, 0.0)
    provenance_bonus = {
        "provided": 0.06,
        "local_kb": 0.05,
        "web": 0.0,
        "graph": -0.08,
    }.get(ptype, 0.0)
    score = max(0.0, min(1.0, base + method_bonus + provenance_bonus))
    return {
        "provenance_type": ptype,
        "retrieval_method": method,
        "authority_class": aclass,
        "source_authority_score": round(base, 3),
        "source_weight": round(score, 3),
    }


def chunk_source_weight(chunk: Any) -> dict[str, Any]:
    meta = getattr(chunk, "metadata", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    source = {
        "source_type": getattr(chunk, "source_type", None),
        "source_url": getattr(chunk, "source_url", None),
        "title": getattr(chunk, "title", None),
        "source": getattr(chunk, "source", None),
        "metadata": meta,
    }
    return source_weight(source)
