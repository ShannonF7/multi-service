"""Audit cross-node duplicate or near-duplicate images.

This script is read-only. It writes JSONL files for manual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin

import requests
from PIL import Image
from sqlalchemy import text

from src.rag.dependencies import get_ai_engine
from src.multimodal.simclr_search import get_asset_embedding

DEFAULT_MEDIA_BASE_URL = os.getenv("A_MEDIA_BASE_URL", "http://ai.smartoptiks.cn")
CACHE_DIR = Path(os.getenv("MULTIMODAL_IMAGE_CACHE", "/tmp/zhangbi_multimodal_images"))


def read_jsonl(path: str | None) -> list[dict]:
    if not path:
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
    return "".join("1" if value >= avg else "0" for value in values)


def hamming(left: str, right: str) -> int:
    return sum(1 for a, b in zip(left, right) if a != b) + abs(len(left) - len(right))


def l2_distance(left: list[float], right: list[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) ** 0.5


def is_usable_eval_row(row: dict) -> bool:
    return row.get("usable", True) is not False


def load_assets_from_dataset(dataset: str | None) -> list[dict]:
    rows = read_jsonl(dataset)
    if rows:
        return [row for row in rows if is_usable_eval_row(row)]
    return []


def load_assets_from_db(source_scenic_id: str, limit: int | None) -> list[dict]:
    params = {"sid": str(source_scenic_id)}
    limit_sql = ""
    if limit:
        limit_sql = " limit :limit"
        params["limit"] = int(limit)
    with get_ai_engine().connect() as connection:
        rows = connection.execute(
            text(
                """
                select a.id as asset_id, a.source_node_id as node_id, a.url as image_url,
                       n.node_name, n.node_type, n.parent_source_node_id as parent_node_id
                from node_assets a
                left join semantic_nodes n on n.source_scenic_id = a.source_scenic_id
                  and n.source_node_id = a.source_node_id
                where a.source_scenic_id = :sid
                  and coalesce(a.asset_type, 'image') = 'image'
                  and coalesce(a.url, '') <> ''
                order by a.id asc
                """ + limit_sql
            ),
            params,
        ).mappings().all()
    return [dict(row) for row in rows]


def enrich_assets(assets: list[dict], media_base_url: str, include_simclr: bool) -> list[dict]:
    enriched = []
    with get_ai_engine().connect() as connection:
        for asset in assets:
            item = dict(asset)
            item["asset_id"] = int(item["asset_id"])
            item["node_id"] = str(item.get("node_id") or item.get("source_node_id") or "")
            try:
                path = fetch_to_cache(str(item.get("image_url") or item.get("url") or ""), media_base_url)
                item["sha256"] = sha256_file(path)
                item["ahash"] = average_hash(path)
            except Exception as exc:
                item["load_error"] = str(exc)[:300]
            if include_simclr:
                try:
                    item["simclr_embedding"] = get_asset_embedding(connection, item["asset_id"])
                except Exception:
                    item["simclr_embedding"] = None
            enriched.append(item)
    return enriched


def suggested_action(left: dict, right: dict, same_url: bool, same_sha: bool, phash_distance: int | None, simclr_distance: float | None) -> str:
    if left.get("node_id") == right.get("node_id"):
        return "same_node_duplicate_keep_one"
    if same_url or same_sha:
        return "cross_node_exact_duplicate_manual_review"
    if simclr_distance is not None and simclr_distance <= 1e-6:
        return "cross_node_simclr_identical_manual_review"
    if phash_distance is not None and phash_distance <= 5:
        return "cross_node_near_duplicate_manual_review"
    return "no_action"


def audit(source_scenic_id: str | None, dataset: str | None, output_dir: str, media_base_url: str, limit: int | None, simclr_threshold: float) -> dict:
    assets = load_assets_from_dataset(dataset)
    if not assets and source_scenic_id:
        assets = load_assets_from_db(source_scenic_id, limit)
    if limit and assets:
        assets = assets[:limit]
    enriched = enrich_assets(assets, media_base_url, include_simclr=True)

    same_node_exact = []
    cross_node_exact = []
    cross_node_near = []
    shared_visual = []
    all_pairs = []
    for i, left in enumerate(enriched):
        for right in enriched[i + 1:]:
            if left.get("load_error") or right.get("load_error"):
                continue
            same_url = str(left.get("image_url") or "") == str(right.get("image_url") or "")
            same_sha = bool(left.get("sha256") and left.get("sha256") == right.get("sha256"))
            phash_distance = hamming(str(left.get("ahash") or ""), str(right.get("ahash") or "")) if left.get("ahash") and right.get("ahash") else None
            simclr_distance = None
            if left.get("simclr_embedding") and right.get("simclr_embedding"):
                simclr_distance = l2_distance(left["simclr_embedding"], right["simclr_embedding"])
            action = suggested_action(left, right, same_url, same_sha, phash_distance, simclr_distance)
            if action == "no_action" and (simclr_distance is None or simclr_distance > simclr_threshold):
                continue
            row = {
                "asset_id": left.get("asset_id"),
                "node_id": left.get("node_id"),
                "duplicate_asset_id": right.get("asset_id"),
                "duplicate_node_id": right.get("node_id"),
                "same_url": same_url,
                "same_sha256": same_sha,
                "phash_distance": phash_distance,
                "simclr_distance": simclr_distance,
                "suggested_action": action if action != "no_action" else "simclr_near_duplicate_manual_review",
                "left": {k: left.get(k) for k in ["asset_id", "node_id", "node_name", "node_type", "parent_node_id", "image_url"]},
                "right": {k: right.get(k) for k in ["asset_id", "node_id", "node_name", "node_type", "parent_node_id", "image_url"]},
            }
            all_pairs.append(row)
            same_node = str(left.get("node_id")) == str(right.get("node_id"))
            if same_node and (same_url or same_sha):
                same_node_exact.append(row)
            elif (not same_node) and (same_url or same_sha):
                cross_node_exact.append(row)
            elif (not same_node) and phash_distance is not None and phash_distance <= 5:
                cross_node_near.append(row)
            elif (not same_node) and simclr_distance is not None and simclr_distance <= simclr_threshold:
                shared_visual.append(row)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "same_node_exact_duplicates.jsonl": same_node_exact,
        "cross_node_exact_duplicates.jsonl": cross_node_exact,
        "cross_node_near_duplicates.jsonl": cross_node_near,
        "shared_visual_assets.jsonl": shared_visual,
        "all_duplicate_candidates.jsonl": all_pairs,
    }
    for name, rows in outputs.items():
        with (out / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    summary = {"assets": len(enriched), "pairs": len(all_pairs), **{name: len(rows) for name, rows in outputs.items()}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cross-node duplicate images.")
    parser.add_argument("--source-scenic-id", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--media-base-url", default=DEFAULT_MEDIA_BASE_URL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--simclr-threshold", type=float, default=1e-6)
    args = parser.parse_args()
    print(json.dumps(audit(args.source_scenic_id, args.dataset, args.output_dir, args.media_base_url, args.limit, args.simclr_threshold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
