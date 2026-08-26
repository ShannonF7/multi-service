from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_url(value: str, base_url: str) -> str:
    value = str(value or '')
    if value.startswith('http://') or value.startswith('https://'):
        return value
    if value.startswith('/'):
        return urljoin(base_url.rstrip('/') + '/', value.lstrip('/'))
    return value


def box_to_rect(box: Any) -> dict[str, float] | None:
    if not box:
        return None
    try:
        if len(box) == 4 and all(not isinstance(x, list) for x in box):
            x1, y1, x2, y2 = [float(x) for x in box]
            return {'x': min(x1, x2), 'y': min(y1, y2), 'w': abs(x2 - x1), 'h': abs(y2 - y1)}
        points = box
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return {'x': min(xs), 'y': min(ys), 'w': max(xs) - min(xs), 'h': max(ys) - min(ys)}
    except Exception:
        return None


def ocr_items_for(asset: dict[str, Any], ocr_by_asset: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    row = ocr_by_asset.get(int(asset.get('asset_id') or 0)) or {}
    items = []
    for item in row.get('ocr_items') or []:
        text = str(item.get('text') or '').strip()
        rect = box_to_rect(item.get('box'))
        if not text or rect is None:
            continue
        items.append({'text': text, 'score': float(item.get('score') or 0.0), 'box': rect})
    return items


def matched_terms_text(value: Any) -> str:
    if not isinstance(value, dict):
        return ''
    parts = []
    for node_id, terms in sorted(value.items()):
        if isinstance(terms, list):
            parts.append(str(node_id) + ': ' + '、'.join(str(x) for x in terms))
    return '；'.join(parts)


def compact_cluster(cluster: dict[str, Any], ocr_by_asset: dict[int, dict[str, Any]], base_url: str) -> dict[str, Any]:
    nodes = []
    seen_nodes = set()
    for asset in cluster.get('assets') or []:
        node_id = str(asset.get('node_id') or '')
        if node_id and node_id not in seen_nodes:
            seen_nodes.add(node_id)
            nodes.append({
                'node_id': node_id,
                'node_name': str(asset.get('node_name') or ''),
                'node_type': str(asset.get('node_type') or ''),
                'parent_node_id': str(asset.get('parent_node_id') or ''),
            })
    assets = []
    for asset in cluster.get('assets') or []:
        ocr_row = ocr_by_asset.get(int(asset.get('asset_id') or 0)) or {}
        assets.append({
            'asset_id': int(asset.get('asset_id') or 0),
            'node_id': str(asset.get('node_id') or ''),
            'node_name': str(asset.get('node_name') or ''),
            'node_type': str(asset.get('node_type') or ''),
            'parent_node_id': str(asset.get('parent_node_id') or ''),
            'image_url': resolve_url(str(asset.get('image_url') or ''), base_url),
            'ocr_status': str(asset.get('ocr_status') or ''),
            'ocr_reason': str(asset.get('ocr_reason') or ''),
            'ocr_text': str(asset.get('ocr_text') or ''),
            'ocr_max_score': float(asset.get('ocr_max_score') or 0.0),
            'matched_node_ids': asset.get('matched_node_ids') or [],
            'matched_terms': asset.get('matched_terms') or {},
            'matched_terms_text': matched_terms_text(asset.get('matched_terms')),
            'ocr_items': ocr_items_for(asset, ocr_by_asset),
            'ocr_extract_status': str(ocr_row.get('ocr_extract_status') or ''),
        })
    return {
        'cluster_id': str(cluster.get('cluster_id') or ''),
        'asset_count': int(cluster.get('asset_count') or len(assets)),
        'node_count': int(cluster.get('node_count') or len(nodes)),
        'pair_count': int(cluster.get('pair_count') or 0),
        'cluster_ocr_decision': str(cluster.get('cluster_ocr_decision') or ''),
        'ocr_status_counts': cluster.get('ocr_status_counts') or {},
        'nodes': nodes,
        'assets': assets,
    }


def render_page(clusters: list[dict[str, Any]]) -> str:
    data = json.dumps(clusters, ensure_ascii=False)
    decision_counts: dict[str, int] = {}
    for cluster in clusters:
        key = str(cluster.get('cluster_ocr_decision') or 'unclassified')
        decision_counts[key] = decision_counts.get(key, 0) + 1
    summary = ' · '.join(k + '=' + str(v) for k, v in sorted(decision_counts.items()))
    return '''<!doctype html>
<html lang='zh-CN'>
<head>
<meta charset='utf-8'>
<title>OCR Cluster Review</title>
<style>
:root{--bg:#f4f6f8;--panel:#ffffff;--line:#d9e0ea;--text:#142033;--muted:#64748b;--blue:#2563eb;--green:#15803d;--orange:#b45309;--red:#b91c1c;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,'Microsoft YaHei',sans-serif}header{position:sticky;top:0;z-index:20;background:var(--panel);border-bottom:1px solid var(--line);padding:14px 20px;display:flex;gap:16px;align-items:center;justify-content:space-between}h1{font-size:18px;margin:0}.sub{font-size:13px;color:var(--muted);margin-top:4px}.toolbar{display:flex;gap:8px;align-items:center}.toolbar select,.toolbar input{border:1px solid var(--line);border-radius:6px;padding:7px 9px;background:#fff}button{border:1px solid var(--blue);background:var(--blue);color:#fff;border-radius:6px;padding:8px 12px;cursor:pointer}main{max-width:1600px;margin:0 auto;padding:18px}.cluster{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin-bottom:16px;overflow:hidden}.cluster-head{display:flex;justify-content:space-between;gap:12px;padding:12px 14px;border-bottom:1px solid var(--line);background:#fbfcfe}.cluster h2{margin:0;font-size:16px}.badge{display:inline-flex;align-items:center;border:1px solid var(--line);background:#f8fafc;border-radius:999px;padding:3px 8px;font-size:12px;color:#334155;margin-left:6px}.badge.auto_hard_negative{border-color:#bbf7d0;background:#f0fdf4;color:var(--green)}.badge.possible_shared_image,.badge.mixed{border-color:#fed7aa;background:#fff7ed;color:var(--orange)}.badge.unresolved{border-color:#fecaca;background:#fef2f2;color:var(--red)}.nodes{font-size:12px;color:var(--muted);margin-top:6px}.assets{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px;padding:12px}.asset{border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}.asset-head{padding:10px 10px 8px;border-bottom:1px solid var(--line)}.asset-title{font-weight:700;font-size:14px}.asset-meta{font-size:12px;color:var(--muted);margin-top:4px}.image-wrap{position:relative;background:#111827;height:260px;display:flex;align-items:center;justify-content:center;overflow:hidden}.image-wrap img{max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-in}.ocr-layer{position:absolute;left:0;top:0;pointer-events:none}.ocr-box{position:absolute;border:2px solid rgba(37,99,235,.85);background:rgba(37,99,235,.08);border-radius:2px}.ocr-box.low{border-color:rgba(180,83,9,.9);background:rgba(251,146,60,.10)}.ocr-panel{padding:10px;display:grid;gap:8px}.kv{display:grid;grid-template-columns:92px minmax(0,1fr);gap:8px;font-size:12px}.k{color:var(--muted)}.v{min-width:0;word-break:break-all}.ocr-text{max-height:96px;overflow:auto;white-space:pre-wrap;background:#f8fafc;border:1px solid var(--line);border-radius:6px;padding:8px;font-size:12px;line-height:1.45}.term{font-weight:700;color:#0f172a}.decision{border-top:1px solid var(--line);padding:10px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.decision label{font-size:12px}.reason{grid-column:1/-1;width:100%;min-height:44px;border:1px solid var(--line);border-radius:6px;padding:7px}.export{width:100%;height:160px;margin-top:12px;border:1px solid var(--line);border-radius:6px;padding:8px;font-family:Consolas,monospace}.hidden{display:none}.lightbox{position:fixed;inset:0;background:rgba(2,6,23,.9);z-index:100;display:none;align-items:center;justify-content:center;padding:24px}.lightbox.open{display:flex}.lightbox img{max-width:96vw;max-height:92vh;object-fit:contain}.close{position:fixed;right:18px;top:18px;background:#fff;color:#111;border-color:#fff}@media(max-width:900px){.assets{grid-template-columns:1fr}.decision{grid-template-columns:1fr}header{align-items:flex-start;flex-direction:column}}
</style>
</head>
<body>
<header>
 <div><h1>OCR增强跨节点高相似图片簇审核</h1><div class='sub'>''' + html.escape(summary) + '''。重点判断：OCR是否支持当前节点、是否冲突、是否共享。</div></div>
 <div class='toolbar'><input id='searchBox' placeholder='搜索节点/OCR文字'><select id='decisionFilter'><option value=''>全部簇</option></select><button onclick='exportDecisions()'>导出审核JSONL</button></div>
</header>
<main id='app'></main>
<textarea id='exportBox' class='export' placeholder='导出后复制 JSONL'></textarea>
<div id='lightbox' class='lightbox' onclick='closeLightbox(event)'><button class='close' onclick='forceCloseLightbox(event)'>关闭</button><img id='lightboxImg'></div>
<script id='clustersData' type='application/json'>''' + data.replace(chr(60) + chr(47), chr(60) + chr(92) + chr(47)) + '''</script>
<script>
const CLUSTERS = JSON.parse(document.getElementById('clustersData').textContent);
const app = document.getElementById('app');
const decisionFilter = document.getElementById('decisionFilter');
const searchBox = document.getElementById('searchBox');
const decisions = [...new Set(CLUSTERS.map(c => c.cluster_ocr_decision || 'unclassified'))].sort();
for (const d of decisions) { const o=document.createElement('option'); o.value=d; o.textContent=d; decisionFilter.appendChild(o); }
function esc(v){return String(v ?? '').replace(/[&<>]/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[s]));}
function render(){
 const q=searchBox.value.trim().toLowerCase(); const f=decisionFilter.value; app.innerHTML='';
 for(const c of CLUSTERS){
  const hay=JSON.stringify(c).toLowerCase(); if(f && c.cluster_ocr_decision!==f) continue; if(q && !hay.includes(q)) continue;
  const sec=document.createElement('section'); sec.className='cluster'; sec.dataset.clusterId=c.cluster_id;
  const nodeText=c.nodes.map(n=>${n.node_name} ()).join(' / ');
  sec.innerHTML=<div class='cluster-head'><div><h2> <span class='badge '></span></h2><div class='nodes'>同簇节点：</div></div><div class='sub'>assets= · nodes= · pairs= · </div></div><div class='assets'></div><div class='decision'><label><input type='radio' name='cat_' value='hard_negative'> OCR支持各自节点，保留hard negative</label><label><input type='radio' name='cat_' value='wrong_binding'> OCR冲突，疑似错误绑定</label><label><input type='radio' name='cat_' value='shared_image'> 共享图片/多节点可用</label><label><input type='radio' name='cat_' value='same_entity_alias'> 同一实体/别名</label><label><input type='radio' name='cat_' value='uncertain'> 无法判断</label><textarea class='reason' placeholder='审核理由'></textarea></div>;
  const grid=sec.querySelector('.assets');
  for(const a of c.assets){
   const div=document.createElement('div'); div.className='asset';
   div.innerHTML=<div class='asset-head'><div class='asset-title'></div><div class='asset-meta'>当前节点 node  ·  · parent  · asset </div></div><div class='image-wrap'><img src='' onclick='openLightbox(this.src)' onload='drawBoxes(this)'><div class='ocr-layer'></div></div><div class='ocr-panel'><div class='kv'><div class='k'>匹配方式</div><div class='v term'></div></div><div class='kv'><div class='k'>置信度</div><div class='v'></div></div><div class='kv'><div class='k'>命中节点</div><div class='v'></div></div><div class='kv'><div class='k'>命中字段</div><div class='v'></div></div><div class='kv'><div class='k'>同簇其他节点</div><div class='v'></div></div><div class='ocr-text'></div></div>;
   div._ocrItems=a.ocr_items || []; grid.appendChild(div);
  }
  app.appendChild(sec);
 }
}
function drawBoxes(img){
 const wrap=img.closest('.image-wrap'); const layer=wrap.querySelector('.ocr-layer'); const asset=img.closest('.asset'); const items=asset._ocrItems || [];
 const rect=img.getBoundingClientRect(); const wrapRect=wrap.getBoundingClientRect();
 layer.style.left=(rect.left-wrapRect.left)+'px'; layer.style.top=(rect.top-wrapRect.top)+'px'; layer.style.width=rect.width+'px'; layer.style.height=rect.height+'px'; layer.innerHTML='';
 const sx=rect.width/(img.naturalWidth||1); const sy=rect.height/(img.naturalHeight||1);
 for(const item of items){ const b=item.box||{}; const el=document.createElement('div'); el.className='ocr-box'+((item.score||0)<0.75?' low':''); el.title=${item.text} ; el.style.left=(b.x*sx)+'px'; el.style.top=(b.y*sy)+'px'; el.style.width=Math.max(4,b.w*sx)+'px'; el.style.height=Math.max(4,b.h*sy)+'px'; layer.appendChild(el); }
}
function openLightbox(src){document.getElementById('lightboxImg').src=src;document.getElementById('lightbox').classList.add('open');}
function closeLightbox(e){if(e.target.id==='lightbox') forceCloseLightbox(e)}
function forceCloseLightbox(e){e.stopPropagation();document.getElementById('lightbox').classList.remove('open');}
function exportDecisions(){
 const lines=[]; for(const c of CLUSTERS){ const sec=document.querySelector([data-cluster-id='']); if(!sec) continue; const checked=sec.querySelector(input[name='cat_']:checked); lines.push(JSON.stringify({cluster_id:c.cluster_id, cluster_ocr_decision:c.cluster_ocr_decision, asset_count:c.asset_count, node_count:c.node_count, decision:checked?checked.value:'', reason:sec.querySelector('.reason').value.trim()})); }
 document.getElementById('exportBox').value=lines.join('\n')+'\n';
}
searchBox.addEventListener('input', render); decisionFilter.addEventListener('change', render); render();
</script>
</body>
</html>'''


def build(args: argparse.Namespace) -> dict[str, Any]:
    clusters = read_jsonl(args.clusters)
    ocr_by_asset = {int(row.get('asset_id') or 0): row for row in read_jsonl(args.ocr_jsonl)}
    compact = [compact_cluster(row, ocr_by_asset, args.media_base_url) for row in clusters[: args.max_clusters]]
    out = Path(args.output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(compact), encoding='utf-8')
    summary = {'output_html': str(out), 'cluster_count': len(compact), 'asset_count': sum(len(c.get('assets') or []) for c in compact)}
    out.with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Build OCR enhanced near duplicate cluster review page.')
    parser.add_argument('--clusters', required=True)
    parser.add_argument('--ocr-jsonl', required=True)
    parser.add_argument('--output-html', required=True)
    parser.add_argument('--media-base-url', default='http://ai.smartoptiks.cn')
    parser.add_argument('--max-clusters', type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
