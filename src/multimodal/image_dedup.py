"""Detect exact and near duplicate node images for E0 dataset building.

The first version is file based and writes JSONL. It does not modify DB rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urljoin

import requests
from PIL import Image
from sqlalchemy import text

from src.rag.dependencies import get_ai_engine

DEFAULT_MEDIA_BASE_URL = os.getenv("A_MEDIA_BASE_URL", "http://ai.smartoptiks.cn")
CACHE_DIR = Path(os.getenv("MULTIMODAL_IMAGE_CACHE", "/tmp/zhangbi_multimodal_images"))


def resolve_url(url: str, base_url: str) -> str:
    value = (url or "").strip()
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return urljoin(base_url.rstrip("/") + "/", value.lstrip("/"))
    return value


def fetch_to_cache(url: str, base_url: str, timeout: int = 20) -> Path:
    resolved = resolve_url(url, base_url)
    if not resolved:
        raise ValueError("empty url")
    if os.path.exists(resolved):
        return Path(resolved)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(resolved.split("?", 1)[0]).suffix or ".img"
    path = CACHE_DIR / (hashlib.sha256(resolved.encode("utf-8")).hexdigest() + suffix)
    if path.exists() and path.stat().st_size > 0:
        return path
    response = requests.get(resolved, timeout=timeout)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(path: Path, hash_size: int = 8) -> str:
    with Image.open(path).convert("L") as image:
        image = image.resize((hash_size, hash_size))
        values = list(image.getdata())
    avg = sum(values) / len(values)
    bits = ["1" if value >= avg else "0" for value in values]
    return "".join(bits)


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))


def load_assets(source_scenic_id: str, limit: int | None) -> list[dict]:
    params = {"sid": str(source_scenic_id)}
    limit_sql = ""
    if limit:
        limit_sql = " limit :limit"
        params["limit"] = int(limit)
    with get_ai_engine().connect() as connection:
        rows = connection.execute(
            text(
                """
                select a.id as asset_id, a.source_asset_id, a.source_node_id, a.url,
                       a.file_hash, a.role, n.node_name, n.node_type,
                       n.parent_source_node_id
                from node_assets a
                left join semantic_nodes n on n.source_node_id = a.source_node_id
                  and n.source_scenic_id = a.source_scenic_id
                where a.source_scenic_id = :sid
                  and coalesce(a.asset_type, 'image') = 'image'
                  and coalesce(a.url, '') <> ''
                order by a.id asc
                """ + limit_sql
            ),
            params,
        ).mappings().all()
    return [dict(row) for row in rows]


def dedup(source_scenic_id: str, output: str, limit: int | None, base_url: str) -> dict:
    assets = load_assets(source_scenic_id, limit)
    exact_groups: dict[str, list[dict]] = {}
    perceptual: list[dict] = []
    failures: list[dict] = []
    for asset in assets:
        try:
            path = fetch_to_cache(str(asset.get("url") or ""), base_url)
            file_sha = str(asset.get("file_hash") or "") or sha256_file(path)
            ahash = average_hash(path)
            item = {**asset, "sha256": file_sha, "ahash": ahash}
            exact_groups.setdefault(file_sha, []).append(item)
            perceptual.append(item)
        except Exception as exc:
            failures.append({"asset_id": asset.get("asset_id"), "error": str(exc)[:300]})

    rows = []
    for group_id, members in exact_groups.items():
        if len(members) > 1:
            rows.append({"group_type": "exact", "duplicate_group_id": "sha256:" + group_id, "distance": 0, "members": members})

    for i, left in enumerate(perceptual):
        members = [left]
        for right in perceptual[i + 1:]:
            if left.get("sha256") == right.get("sha256"):
                continue
            distance = hamming(str(left.get("ahash") or ""), str(right.get("ahash") or ""))
            if distance <= 5:
                members.append(right)
        if len(members) > 1:
            rows.append({"group_type": "near", "duplicate_group_id": "ahash:" + str(left["asset_id"]), "distance_threshold": 5, "members": members})

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary = {"source_scenic_id": source_scenic_id, "assets": len(assets), "duplicate_groups": len(rows), "failures": len(failures), "failure_items": failures[:20]}
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect duplicate images for multimodal E0.")
    parser.add_argument("--source-scenic-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--media-base-url", default=DEFAULT_MEDIA_BASE_URL)
    args = parser.parse_args()
    print(json.dumps(dedup(args.source_scenic_id, args.output, args.limit, args.media_base_url), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
