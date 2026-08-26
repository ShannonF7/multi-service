"""Run PaddleOCR PP-OCRv5 on near-duplicate review clusters.

This script is intentionally independent from the main llama_factory runtime.
Run it inside the isolated ``paddle_ocr`` conda environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def resolve_url(value: str, base_url: str) -> str:
    value = str(value or "")
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return urljoin(base_url.rstrip("/") + "/", value.lstrip("/"))
    return value


def image_cache_path(cache_dir: str | Path, url: str, asset_id: int) -> Path:
    suffix = Path(url.split("?", 1)[0]).suffix or ".img"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / f"{asset_id}_{digest}{suffix}"


def download_image(url: str, cache_dir: str | Path, asset_id: int, timeout: int) -> Path:
    path = image_cache_path(cache_dir, url, asset_id)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=path.suffix, dir=str(path.parent))
    os.close(tmp_fd)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = resp.read()
        Path(tmp_name).write_bytes(data)
        Path(tmp_name).replace(path)
        return path
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def unique_assets(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_asset: dict[int, dict[str, Any]] = {}
    for cluster in clusters:
        for asset in cluster.get("assets") or []:
            asset_id = int(asset["asset_id"])
            row = dict(asset)
            row["cluster_ids"] = sorted(set(row.get("cluster_ids") or []) | {str(cluster.get("cluster_id") or "")})
            by_asset[asset_id] = row
    return [by_asset[k] for k in sorted(by_asset)]


def to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return [to_list(x) for x in value]
    if isinstance(value, tuple):
        return [to_list(x) for x in value]
    return value


def parse_ocr_result(result: Any, min_score: float) -> dict[str, Any]:
    page = result[0] if isinstance(result, list) and result else result
    if not isinstance(page, dict):
        return {"ocr_text": "", "ocr_items": [], "raw_result_type": type(page).__name__}

    texts = list(page.get("rec_texts") or [])
    scores = list(page.get("rec_scores") or [])
    raw_boxes = page.get("rec_boxes")
    if raw_boxes is None:
        raw_boxes = page.get("rec_polys")
    boxes = to_list(raw_boxes if raw_boxes is not None else [])
    items: list[dict[str, Any]] = []
    kept_texts: list[str] = []
    for index, text in enumerate(texts):
        text = str(text or "").strip()
        score = float(scores[index]) if index < len(scores) and scores[index] is not None else 0.0
        box = boxes[index] if isinstance(boxes, list) and index < len(boxes) else None
        item = {"text": text, "score": score, "box": box}
        items.append(item)
        if text and score >= min_score:
            kept_texts.append(text)
    return {
        "ocr_text": "\n".join(kept_texts),
        "ocr_items": items,
        "text_count": len([x for x in texts if str(x or "").strip()]),
        "kept_text_count": len(kept_texts),
        "max_score": max((float(x or 0) for x in scores), default=0.0),
        "avg_score": sum(float(x or 0) for x in scores) / len(scores) if scores else 0.0,
    }


def build_ocr(args: argparse.Namespace) -> dict[str, Any]:
    from paddleocr import PaddleOCR

    clusters = read_jsonl(args.clusters)
    assets = unique_assets(clusters)
    existing: dict[int, dict[str, Any]] = {}
    output_path = Path(args.output_jsonl)
    if args.resume and output_path.exists():
        for row in read_jsonl(output_path):
            if row.get("asset_id") is not None and row.get("ocr_extract_status") == "ok":
                existing[int(row["asset_id"])] = row

    ocr = PaddleOCR(
        text_detection_model_name=args.det_model,
        text_recognition_model_name=args.rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    rows: list[dict[str, Any]] = []
    for index, asset in enumerate(assets, start=1):
        asset_id = int(asset["asset_id"])
        if asset_id in existing:
            rows.append(existing[asset_id])
            continue
        image_url = resolve_url(str(asset.get("image_url") or ""), args.media_base_url)
        started = time.time()
        row = {
            "asset_id": asset_id,
            "node_id": str(asset.get("node_id") or ""),
            "node_name": str(asset.get("node_name") or ""),
            "node_type": str(asset.get("node_type") or ""),
            "parent_node_id": str(asset.get("parent_node_id") or ""),
            "cluster_ids": asset.get("cluster_ids") or [],
            "image_url": str(asset.get("image_url") or ""),
            "resolved_image_url": image_url,
            "ocr_model": f"{args.det_model}+{args.rec_model}",
        }
        try:
            image_path = download_image(image_url, args.cache_dir, asset_id, args.download_timeout)
            parsed = parse_ocr_result(ocr.predict(str(image_path)), args.min_score)
            row.update(parsed)
            row["ocr_extract_status"] = "ok"
        except Exception as exc:
            row["ocr_extract_status"] = "error"
            row["error"] = repr(exc)
        row["elapsed_sec"] = round(time.time() - started, 3)
        rows.append(row)
        if index % args.flush_every == 0:
            write_jsonl(output_path, rows)
            print(json.dumps({"processed": index, "total": len(assets), "output": str(output_path)}, ensure_ascii=False))

    write_jsonl(output_path, rows)
    ok_count = sum(1 for row in rows if row.get("ocr_extract_status") == "ok")
    text_count = sum(1 for row in rows if row.get("ocr_text"))
    summary = {
        "clusters": str(args.clusters),
        "output_jsonl": str(output_path),
        "asset_count": len(rows),
        "ok_count": ok_count,
        "error_count": len(rows) - ok_count,
        "with_effective_text": text_count,
        "det_model": args.det_model,
        "rec_model": args.rec_model,
        "min_score": args.min_score,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaddleOCR PP-OCRv5 on cluster assets.")
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--cache-dir", default="/tmp/paddle_ocr_image_cache")
    parser.add_argument("--media-base-url", default="http://ai.smartoptiks.cn")
    parser.add_argument("--det-model", default="PP-OCRv5_server_det")
    parser.add_argument("--rec-model", default="PP-OCRv5_server_rec")
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--download-timeout", type=int, default=30)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_ocr(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
