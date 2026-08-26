"""Persist the exact expanded_v3 image assets and write a checksum manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from src.multimodal.simclr_backfill import DEFAULT_MEDIA_BASE_URL, fetch_image_to_cache


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--media-base-url", default=DEFAULT_MEDIA_BASE_URL)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_asset = {}
    for row in read_jsonl(dataset):
        if not row.get("usable", True):
            continue
        asset_id = int(row["asset_id"])
        by_asset.setdefault(asset_id, row)

    manifest_rows = []
    failures = []
    for asset_id, row in sorted(by_asset.items()):
        url = str(row.get("image_url") or row.get("url") or "")
        try:
            source = fetch_image_to_cache(url, args.media_base_url)
            suffix = source.suffix.lower() or ".img"
            target = output_dir / f"asset_{asset_id}{suffix}"
            if not target.exists() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
            manifest_rows.append({
                "asset_id": asset_id,
                "node_id": str(row.get("node_id") or ""),
                "roles": sorted({str(item.get("role") or "") for item in read_jsonl(dataset) if int(item.get("asset_id", -1)) == asset_id}),
                "source_url": url,
                "local_path": str(target.relative_to(dataset.parent)),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            })
        except Exception as exc:
            failures.append({"asset_id": asset_id, "url": url, "error": str(exc)})

    manifest = Path(args.manifest)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": str(dataset),
        "expected_assets": len(by_asset),
        "persisted_assets": len(manifest_rows),
        "failures": failures,
        "total_bytes": sum(row["size_bytes"] for row in manifest_rows),
    }
    manifest.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures or len(manifest_rows) != len(by_asset):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
