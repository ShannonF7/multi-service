"""Build a small manual OCR quality sample from an image-node eval dataset."""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import defaultdict
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


def build_sample(dataset: str, output_jsonl: str, output_html: str, sample_size: int, seed: int, base_url: str) -> dict:
    rows = [row for row in read_jsonl(dataset) if row.get("usable", True) is not False]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("node_type") or "Other")].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    per_type = max(1, sample_size // max(1, len(by_type)))
    for node_type, items in sorted(by_type.items()):
        rng.shuffle(items)
        selected.extend(items[:per_type])
    if len(selected) < sample_size:
        remaining = [row for row in rows if int(row["asset_id"]) not in {int(x["asset_id"]) for x in selected}]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])
    selected = selected[:sample_size]

    sample_rows = []
    cards = []
    for index, row in enumerate(selected, start=1):
        item = {
            "sample_id": f"ocr_{index:03d}",
            "asset_id": int(row["asset_id"]),
            "node_id": str(row.get("node_id") or ""),
            "node_name": str(row.get("node_name") or ""),
            "node_type": str(row.get("node_type") or ""),
            "parent_node_id": row.get("parent_node_id"),
            "image_url": str(row.get("image_url") or ""),
            "resolved_image_url": resolve_url(str(row.get("image_url") or ""), base_url),
            "manual_has_text": "",
            "manual_text": "",
            "manual_quality": "",
            "notes": "",
        }
        sample_rows.append(item)
        cards.append(
            f"""
<section class="card" data-sample-id="{html.escape(item['sample_id'])}">
  <div class="head">
    <h2>#{index} {html.escape(item['node_name'])}</h2>
    <span>{html.escape(item['node_type'])} · node {html.escape(item['node_id'])} · asset {item['asset_id']}</span>
  </div>
  <img src="{html.escape(item['resolved_image_url'])}" alt="{html.escape(item['node_name'])}">
  <div class="fields">
    <label>是否有可读文字
      <select class="has-text">
        <option value=""></option>
        <option value="yes">yes</option>
        <option value="no">no</option>
        <option value="uncertain">uncertain</option>
      </select>
    </label>
    <label>质量
      <select class="quality">
        <option value=""></option>
        <option value="clear">clear</option>
        <option value="partial">partial</option>
        <option value="blurred">blurred</option>
        <option value="tiny">tiny</option>
        <option value="none">none</option>
      </select>
    </label>
    <label>人工文字<textarea class="text"></textarea></label>
    <label>备注<textarea class="notes"></textarea></label>
  </div>
</section>"""
        )

    write_jsonl(output_jsonl, sample_rows)
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>OCR Quality Sample Review</title>
<style>
body {{ margin:0; font-family: Arial, "Microsoft YaHei", sans-serif; background:#f5f6f8; color:#172033; }}
header {{ position:sticky; top:0; z-index:2; background:white; border-bottom:1px solid #d8dee8; padding:14px 20px; display:flex; justify-content:space-between; gap:12px; align-items:center; }}
h1 {{ margin:0; font-size:18px; }}
button {{ border:1px solid #2563eb; background:#2563eb; color:white; border-radius:6px; padding:8px 12px; cursor:pointer; }}
main {{ max-width:1280px; margin:0 auto; padding:18px; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.card {{ background:white; border:1px solid #d8dee8; border-radius:8px; padding:12px; }}
.head {{ display:flex; justify-content:space-between; gap:8px; align-items:flex-start; margin-bottom:8px; }}
h2 {{ margin:0; font-size:15px; }}
.head span {{ color:#667085; font-size:12px; }}
img {{ width:100%; height:280px; object-fit:contain; background:#111827; border-radius:6px; }}
.fields {{ margin-top:10px; display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
label {{ display:block; font-size:13px; color:#344054; }}
select, textarea {{ width:100%; box-sizing:border-box; margin-top:4px; border:1px solid #cfd6e4; border-radius:6px; padding:7px; }}
textarea {{ min-height:56px; }}
#exportBox {{ grid-column:1/-1; width:100%; height:180px; box-sizing:border-box; font-family:Consolas, monospace; }}
@media (max-width: 900px) {{ main {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <div><h1>OCR 小样本质量审核</h1><div>共 {len(sample_rows)} 张。只用于评测标注，不修改业务数据。</div></div>
  <button onclick="exportLabels()">导出 OCR labels JSONL</button>
</header>
<main>
{''.join(cards)}
<textarea id="exportBox" placeholder="点击导出后复制 JSONL"></textarea>
</main>
<script>
const SAMPLES = {json.dumps(sample_rows, ensure_ascii=False)};
function exportLabels() {{
  const lines = [];
  for (const sample of SAMPLES) {{
    const card = document.querySelector(`[data-sample-id="${{sample.sample_id}}"]`);
    lines.push(JSON.stringify({{
      ...sample,
      manual_has_text: card.querySelector('.has-text').value,
      manual_quality: card.querySelector('.quality').value,
      manual_text: card.querySelector('.text').value.trim(),
      notes: card.querySelector('.notes').value.trim()
    }}));
  }}
  document.getElementById('exportBox').value = lines.join('\\n') + '\\n';
}}
</script>
</body>
</html>"""
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    Path(output_html).write_text(page, encoding="utf-8")
    return {"sample_size": len(sample_rows), "output_jsonl": output_jsonl, "output_html": output_html}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual OCR quality sample.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--media-base-url", default="http://ai.smartoptiks.cn")
    args = parser.parse_args()
    print(
        json.dumps(
            build_sample(
                args.dataset,
                args.output_jsonl,
                args.output_html,
                args.sample_size,
                args.seed,
                args.media_base_url,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
