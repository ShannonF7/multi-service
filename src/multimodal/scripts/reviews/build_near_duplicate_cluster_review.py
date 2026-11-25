"""Build cluster-level review page for cross-node high-near duplicate pairs."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


class DSU:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


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


def asset_from_side(side: dict[str, Any], fallback_asset_id: int, fallback_node_id: str) -> dict[str, Any]:
    return {
        "asset_id": int(side.get("asset_id") or fallback_asset_id),
        "node_id": str(side.get("node_id") or fallback_node_id),
        "node_name": str(side.get("node_name") or ""),
        "node_type": str(side.get("node_type") or ""),
        "parent_node_id": str(side.get("parent_node_id") or ""),
        "image_url": str(side.get("image_url") or ""),
    }


def build(input_pairs: str, output_jsonl: str, output_html: str, base_url: str, max_clusters_in_html: int) -> dict[str, Any]:
    pairs = read_jsonl(input_pairs)
    dsu = DSU()
    assets: dict[int, dict[str, Any]] = {}
    pair_rows = []
    for row in pairs:
        left_id = int(row["asset_id"])
        right_id = int(row["duplicate_asset_id"])
        dsu.union(left_id, right_id)
        assets[left_id] = asset_from_side(row.get("left") or {}, left_id, str(row.get("node_id") or ""))
        assets[right_id] = asset_from_side(row.get("right") or {}, right_id, str(row.get("duplicate_node_id") or ""))
        pair_rows.append(
            {
                "left_asset_id": left_id,
                "right_asset_id": right_id,
                "phash_distance": row.get("phash_distance"),
                "simclr_distance": row.get("simclr_distance"),
            }
        )

    cluster_assets: dict[int, list[int]] = defaultdict(list)
    for asset_id in assets:
        cluster_assets[dsu.find(asset_id)].append(asset_id)

    clusters = []
    for index, (root, asset_ids) in enumerate(sorted(cluster_assets.items(), key=lambda item: (-len(item[1]), item[0])), start=1):
        asset_set = set(asset_ids)
        cluster_pairs = [p for p in pair_rows if p["left_asset_id"] in asset_set and p["right_asset_id"] in asset_set]
        node_ids = sorted({assets[asset_id]["node_id"] for asset_id in asset_ids})
        clusters.append(
            {
                "cluster_id": f"near_cluster_{index:04d}",
                "root_asset_id": root,
                "asset_count": len(asset_ids),
                "node_count": len(node_ids),
                "node_ids": node_ids,
                "assets": [assets[asset_id] for asset_id in sorted(asset_ids)],
                "pair_count": len(cluster_pairs),
                "min_phash_distance": min((int(p["phash_distance"]) for p in cluster_pairs if p.get("phash_distance") is not None), default=None),
                "max_phash_distance": max((int(p["phash_distance"]) for p in cluster_pairs if p.get("phash_distance") is not None), default=None),
                "review_category": "",
                "decision": "",
                "canonical_node_id": "",
                "reason": "",
            }
        )

    write_jsonl(output_jsonl, clusters)

    labels = [
        ("wrong_binding", "错误绑定：修正节点后可进入 v3"),
        ("shared_image", "共享图片：排除出单标签评测，可保留共享资产"),
        ("same_entity_alias", "同一实体/别名节点：先处理合并或别名关系"),
        ("hard_negative", "不同节点但视觉相似：作为 hard negative 保留"),
        ("uncertain", "无法确定：排除出正式评测"),
    ]
    cards = []
    for cluster in clusters[:max_clusters_in_html]:
        options = "".join(
            f'<label><input type="radio" name="cat_{html.escape(cluster["cluster_id"])}" value="{html.escape(value)}"> {html.escape(text)}</label>'
            for value, text in labels
        )
        images = []
        for asset in cluster["assets"]:
            images.append(
                f"""
<div class="asset">
  <img src="{html.escape(resolve_url(asset['image_url'], base_url))}" onclick="openLightbox(this.src)" title="点击放大">
  <h3>{html.escape(asset['node_name'])}</h3>
  <p>node {html.escape(asset['node_id'])} · {html.escape(asset['node_type'])}</p>
  <p>parent {html.escape(asset['parent_node_id'])} · asset {asset['asset_id']}</p>
</div>"""
            )
        cards.append(
            f"""
<section class="cluster" data-cluster-id="{html.escape(cluster['cluster_id'])}">
  <div class="cluster-head">
    <h2>{html.escape(cluster['cluster_id'])}</h2>
    <div class="meta">assets={cluster['asset_count']} · nodes={cluster['node_count']} · pairs={cluster['pair_count']} · phash={cluster['min_phash_distance']}..{cluster['max_phash_distance']}</div>
  </div>
  <div class="assets">{''.join(images)}</div>
  <div class="decision">
    {options}
    <label>canonical_node_id <input type="text" class="canonical"></label>
    <label>reason <textarea class="reason"></textarea></label>
  </div>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Near Duplicate Cluster Review</title>
<style>
body{{margin:0;font-family:Arial,"Microsoft YaHei",sans-serif;background:#f6f7f9;color:#172033}}
header{{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #d8dee8;padding:14px 22px;display:flex;justify-content:space-between;gap:16px;align-items:center}}
h1{{margin:0;font-size:18px}}button{{border:1px solid #2563eb;background:#2563eb;color:white;border-radius:6px;padding:8px 12px;cursor:pointer}}
main{{padding:20px;max-width:1440px;margin:0 auto}}.cluster{{background:#fff;border:1px solid #d8dee8;border-radius:8px;margin-bottom:16px;padding:14px}}
.cluster-head{{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px}}h2{{margin:0;font-size:16px}}.meta{{color:#667085;font-size:13px}}
.assets{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}.asset{{border:1px solid #e6e9ef;border-radius:8px;padding:10px;background:#fbfcfe}}
.asset img{{width:100%;height:190px;object-fit:contain;background:#111827;border-radius:6px;cursor:zoom-in}}.asset h3{{margin:8px 0 4px;font-size:14px}}.asset p{{margin:3px 0;color:#475467;font-size:12px}}.lightbox{{position:fixed;inset:0;background:rgba(7,12,24,.88);z-index:1000;display:none;align-items:center;justify-content:center;padding:28px}}.lightbox.open{{display:flex}}.lightbox img{{max-width:96vw;max-height:92vh;object-fit:contain;transform:scale(var(--zoom,1));transition:transform .08s linear;cursor:grab}}.lightbox-tools{{position:fixed;top:16px;right:18px;display:flex;gap:8px}}.lightbox-tools button{{background:#fff;color:#172033;border-color:#d8dee8}}
.decision{{margin-top:10px;display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:8px}}.decision label{{display:block;font-size:14px}}
.decision input[type=text],.decision textarea{{width:100%;box-sizing:border-box;border:1px solid #cfd6e4;border-radius:6px;padding:7px;margin-top:4px}}textarea{{min-height:54px}}
#exportBox{{width:100%;height:180px;margin-top:14px;font-family:Consolas,monospace}}@media(max-width:860px){{.decision{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div><h1>跨节点高相似图片簇审核</h1><div class="meta">共 {len(clusters)} 个连通簇，来自 {len(pairs)} 个 pair。页面不展示模型预测，避免审核偏置。</div></div><button onclick="exportDecisions()">导出 cluster decisions JSONL</button></header>
<div id="lightbox" class="lightbox" onclick="closeLightbox(event)"><div class="lightbox-tools"><button onclick="zoomLightbox(0.2,event)">放大</button><button onclick="zoomLightbox(-0.2,event)">缩小</button><button onclick="resetLightbox(event)">还原</button><button onclick="forceCloseLightbox(event)">关闭</button></div><img id="lightboxImg" alt="preview" ondblclick="resetLightbox(event)"></div><main>{''.join(cards)}<textarea id="exportBox" placeholder="点击导出后复制 JSONL"></textarea></main>
<script>
const CLUSTERS={json.dumps(clusters[:max_clusters_in_html], ensure_ascii=False)};
let lightboxZoom=1;
function openLightbox(src){{
 const box=document.getElementById('lightbox');
 const img=document.getElementById('lightboxImg');
 img.src=src;
 lightboxZoom=1;
 img.style.setProperty('--zoom', lightboxZoom);
 box.classList.add('open');
 document.body.style.overflow='hidden';
}}
function forceCloseLightbox(event){{ event.stopPropagation(); document.getElementById('lightbox').classList.remove('open'); document.body.style.overflow=''; }}
function closeLightbox(event){{ if(event.target.id==='lightbox'){{ document.getElementById('lightbox').classList.remove('open'); document.body.style.overflow=''; }} }}
function zoomLightbox(delta,event){{ event.stopPropagation(); lightboxZoom=Math.max(0.4,Math.min(5,lightboxZoom+delta)); document.getElementById('lightboxImg').style.setProperty('--zoom', lightboxZoom); }}
function resetLightbox(event){{ event.stopPropagation(); lightboxZoom=1; document.getElementById('lightboxImg').style.setProperty('--zoom', lightboxZoom); }}
document.addEventListener('keydown', event=>{{ if(event.key==='Escape'){{ document.getElementById('lightbox').classList.remove('open'); document.body.style.overflow=''; }} }});
document.addEventListener('wheel', event=>{{
 const box=document.getElementById('lightbox');
 if(!box.classList.contains('open')) return;
 event.preventDefault();
 zoomLightbox(event.deltaY<0?0.15:-0.15,event);
}}, {{passive:false}});
function exportDecisions(){{
 const lines=[];
 for(const cluster of CLUSTERS){{
  const section=document.querySelector(`[data-cluster-id="${{cluster.cluster_id}}"]`);
  const checked=section.querySelector(`input[name="cat_${{cluster.cluster_id}}"]:checked`);
  lines.push(JSON.stringify({{
   cluster_id:cluster.cluster_id,
   root_asset_id:cluster.root_asset_id,
   asset_count:cluster.asset_count,
   node_count:cluster.node_count,
   node_ids:cluster.node_ids,
   decision:checked?checked.value:"",
   review_category:checked?checked.value:"",
   canonical_node_id:section.querySelector('.canonical').value.trim(),
   reason:section.querySelector('.reason').value.trim()
  }}));
 }}
 document.getElementById('exportBox').value=lines.join('\\n')+'\\n';
}}
</script></body></html>"""
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    Path(output_html).write_text(page, encoding="utf-8")
    summary = {
        "input_pairs": input_pairs,
        "pair_count": len(pairs),
        "cluster_count": len(clusters),
        "asset_count": len(assets),
        "max_assets_per_cluster": max((c["asset_count"] for c in clusters), default=0),
        "max_nodes_per_cluster": max((c["node_count"] for c in clusters), default=0),
        "output_jsonl": output_jsonl,
        "output_html": output_html,
        "html_cluster_count": min(len(clusters), max_clusters_in_html),
    }
    Path(output_jsonl).with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build near duplicate cluster review artifacts.")
    parser.add_argument("--input-pairs", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--media-base-url", default="http://ai.smartoptiks.cn")
    parser.add_argument("--max-clusters-in-html", type=int, default=10000)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.input_pairs, args.output_jsonl, args.output_html, args.media_base_url, args.max_clusters_in_html),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
