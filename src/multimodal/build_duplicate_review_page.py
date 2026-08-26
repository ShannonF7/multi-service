"""Build an HTML review page for cross-node exact duplicate image pairs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import urljoin

DEFAULT_BASE_URL = "http://ai.smartoptiks.cn"
DECISIONS = ["left_correct", "right_correct", "shared", "same_entity", "uncertain"]


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def image_url(value: str, base_url: str) -> str:
    value = str(value or "")
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if value.startswith("/"):
        return urljoin(base_url.rstrip("/") + "/", value.lstrip("/"))
    return value


def pair_key(row: dict) -> str:
    return f"{row.get('asset_id')}__{row.get('duplicate_asset_id')}"


def build_review(input_path: str, output_html: str, decisions_template: str, base_url: str) -> dict:
    rows = read_jsonl(input_path)
    decisions = []
    cards = []
    for index, row in enumerate(rows, start=1):
        left = row.get("left") or {}
        right = row.get("right") or {}
        key = pair_key(row)
        decisions.append({
            "pair_key": key,
            "left_asset_id": row.get("asset_id"),
            "right_asset_id": row.get("duplicate_asset_id"),
            "left_node_id": row.get("node_id"),
            "right_node_id": row.get("duplicate_node_id"),
            "decision": "",
            "reason": "",
            "canonical_node_id": "",
        })
        left_url = image_url(left.get("image_url") or "", base_url)
        right_url = image_url(right.get("image_url") or "", base_url)
        radios = []
        labels = {
            "left_correct": "左边正确，排除右侧绑定",
            "right_correct": "右边正确，排除左侧绑定",
            "shared": "两边都可用，标记 shared，不进单标签评测",
            "same_entity": "两个节点是同一实体，使用 canonical_node_id",
            "uncertain": "无法判断，两张均排除",
        }
        for decision in DECISIONS:
            radios.append(
                f'<label><input type="radio" name="decision_{html.escape(key)}" value="{decision}"> {html.escape(labels[decision])}</label>'
            )
        cards.append(f'''
<section class="pair" data-pair-key="{html.escape(key)}">
  <div class="pair-head">
    <h2>#{index} pair {html.escape(key)}</h2>
    <div class="meta">same_sha256={html.escape(str(row.get('same_sha256')))} · phash_distance={html.escape(str(row.get('phash_distance')))} · simclr_distance={html.escape(str(row.get('simclr_distance')))}</div>
  </div>
  <div class="grid">
    <div class="panel">
      <img src="{html.escape(left_url)}" alt="left asset {html.escape(str(row.get('asset_id')))}">
      <h3>{html.escape(str(left.get('node_name') or ''))}</h3>
      <p>node {html.escape(str(row.get('node_id')))} · {html.escape(str(left.get('node_type') or ''))}</p>
      <p>asset {html.escape(str(row.get('asset_id')))}</p>
      <code>{html.escape(str(left.get('image_url') or ''))}</code>
    </div>
    <div class="panel">
      <img src="{html.escape(right_url)}" alt="right asset {html.escape(str(row.get('duplicate_asset_id')))}">
      <h3>{html.escape(str(right.get('node_name') or ''))}</h3>
      <p>node {html.escape(str(row.get('duplicate_node_id')))} · {html.escape(str(right.get('node_type') or ''))}</p>
      <p>asset {html.escape(str(row.get('duplicate_asset_id')))}</p>
      <code>{html.escape(str(right.get('image_url') or ''))}</code>
    </div>
  </div>
  <div class="decision">
    {''.join(radios)}
    <label>canonical_node_id <input type="text" class="canonical" placeholder="same_entity 时填写"></label>
    <label>reason <textarea class="reason" placeholder="说明判断依据"></textarea></label>
  </div>
</section>
''')
    page = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>E0-A Cross Node Duplicate Review</title>
<style>
body {{ margin:0; font-family: Arial, "Microsoft YaHei", sans-serif; background:#f6f7f9; color:#172033; }}
header {{ position:sticky; top:0; z-index:10; background:#ffffff; border-bottom:1px solid #dfe3ea; padding:14px 22px; display:flex; justify-content:space-between; gap:16px; align-items:center; }}
header h1 {{ margin:0; font-size:18px; }}
button {{ border:1px solid #1f6feb; background:#1f6feb; color:white; border-radius:6px; padding:8px 12px; cursor:pointer; }}
main {{ padding:20px; max-width:1280px; margin:0 auto; }}
.pair {{ background:#fff; border:1px solid #dfe3ea; border-radius:8px; margin-bottom:18px; padding:16px; }}
.pair-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:12px; }}
.pair h2 {{ margin:0; font-size:16px; }}
.meta {{ color:#667085; font-size:13px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.panel {{ border:1px solid #e6e9ef; border-radius:8px; padding:12px; background:#fbfcfe; }}
.panel img {{ width:100%; height:320px; object-fit:contain; background:#111827; border-radius:6px; }}
.panel h3 {{ margin:10px 0 4px; font-size:16px; }}
.panel p {{ margin:4px 0; color:#475467; }}
.panel code {{ display:block; white-space:pre-wrap; word-break:break-all; color:#667085; font-size:12px; margin-top:8px; }}
.decision {{ margin-top:12px; display:grid; grid-template-columns:repeat(2, minmax(260px, 1fr)); gap:10px; align-items:start; }}
.decision label {{ display:block; font-size:14px; }}
.decision input[type=text], .decision textarea {{ width:100%; box-sizing:border-box; border:1px solid #cfd6e4; border-radius:6px; padding:8px; margin-top:4px; }}
.decision textarea {{ min-height:58px; }}
#exportBox {{ width:100%; height:180px; margin-top:14px; font-family:Consolas, monospace; }}
@media (max-width: 860px) {{ .grid, .decision {{ grid-template-columns:1fr; }} .panel img {{ height:240px; }} }}
</style>
</head>
<body>
<header>
  <div><h1>E0-A 跨节点完全重复图片审核</h1><div class="meta">共 {len(rows)} 组。只修改评测数据集，不修改业务数据库。</div></div>
  <button onclick="exportDecisions()">导出 decisions JSONL</button>
</header>
<main>
{''.join(cards)}
<textarea id="exportBox" placeholder="点击导出后，这里会出现 JSONL。保存为 cross_node_duplicate_decisions.jsonl"></textarea>
</main>
<script>
const PAIRS = {json.dumps(decisions, ensure_ascii=False)};
function exportDecisions() {{
  const lines = [];
  for (const pair of PAIRS) {{
    const section = document.querySelector(`[data-pair-key="${{pair.pair_key}}"]`);
    const checked = section.querySelector(`input[name="decision_${{pair.pair_key}}"]:checked`);
    const decision = checked ? checked.value : "";
    const reason = section.querySelector('.reason').value.trim();
    const canonical = section.querySelector('.canonical').value.trim();
    lines.push(JSON.stringify({{...pair, decision, reason, canonical_node_id: canonical}}));
  }}
  document.getElementById('exportBox').value = lines.join('\\n') + '\\n';
}}
</script>
</body>
</html>'''
    out = Path(output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    dt = Path(decisions_template)
    dt.parent.mkdir(parents=True, exist_ok=True)
    with dt.open("w", encoding="utf-8") as f:
        for row in decisions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"pairs": len(rows), "output_html": str(out), "decisions_template": str(dt)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build duplicate review HTML page.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--decisions-template", required=True)
    parser.add_argument("--media-base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()
    print(json.dumps(build_review(args.input, args.output_html, args.decisions_template, args.media_base_url), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

