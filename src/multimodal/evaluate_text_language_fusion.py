"""Evaluate Chinese/English score fusion for image-text retrieval."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Any
from src.multimodal.node_aggregation import dot

def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows=[]
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def load_vecs(path: str, key: str) -> dict[str, list[float]]:
    out={}
    for row in read_jsonl(path):
        out[str(row[key])] = [float(x) for x in row['vector']]
    return out

def rank_of(expected: set[str], ranked: list[str]) -> int|None:
    for i,x in enumerate(ranked, start=1):
        if x in expected: return i
    return None

def rr(rank): return 1.0/rank if rank else 0.0

def load_node_id_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    text = p.read_text(encoding='utf-8').strip()
    if not text:
        return set()
    if text.startswith('['):
        return {str(x) for x in json.loads(text)}
    ids = set()
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        if value.startswith('{'):
            row = json.loads(value)
            ids.add(str(row.get('node_id') or row.get('id')))
        else:
            ids.add(str(value))
    return ids

def metrics(records, top_ks):
    total=len(records)
    if not total: return {**{f'recall@{k}':0.0 for k in top_ks}, 'mrr':0.0}
    return {**{f'recall@{k}':sum(1 for r in records if r.get('rank') and r['rank']<=k)/total for k in top_ks}, 'mrr':sum(rr(r.get('rank')) for r in records)/total}

def evaluate(dataset, zh_profiles_path, en_profiles_path, zh_dir, en_dir, output_dir, model_key, alphas, variants, top_ks, text_query_node_ids=None):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    ds=[r for r in read_jsonl(dataset) if r.get('usable', True)]
    queries=[r for r in ds if r.get('role')=='query']
    gallery=[r for r in ds if r.get('role')=='gallery']
    zh_profiles=[dict(p, profile_id=f"{p['node_id']}::{p['profile_variant']}") for p in read_jsonl(zh_profiles_path) if p.get('profile_variant') in variants]
    en_profiles=[dict(p, profile_id=f"{p['node_id']}::{p['profile_variant']}") for p in read_jsonl(en_profiles_path) if p.get('profile_variant') in variants]
    zh_by_id={p['profile_id']:p for p in zh_profiles}; en_by_id={p['profile_id']:p for p in en_profiles}
    profile_ids=sorted(set(zh_by_id).intersection(en_by_id))
    zh_img=load_vecs(str(Path(zh_dir)/'image_vectors.jsonl'), 'asset_id')
    en_img=load_vecs(str(Path(en_dir)/'image_vectors.jsonl'), 'asset_id')
    zh_txt=load_vecs(str(Path(zh_dir)/'text_vectors.jsonl'), 'profile_id')
    en_txt=load_vecs(str(Path(en_dir)/'text_vectors.jsonl'), 'profile_id')
    gallery_by_node=defaultdict(list)
    for g in gallery:
        aid=str(g['asset_id'])
        if aid in zh_img and aid in en_img: gallery_by_node[str(g['node_id'])].append(aid)
    results={'model':model_key,'dataset':dataset,'zh_dir':zh_dir,'en_dir':en_dir,'alphas':alphas,'text_query_node_count': None if text_query_node_ids is None else len(text_query_node_ids),'variants':{},'note':'score = alpha * Chinese score + (1-alpha) * English score'}
    per=[]
    for variant in sorted(variants):
        ids=[pid for pid in profile_ids if pid.endswith(f'::{variant}') and pid in zh_txt and pid in en_txt]
        results['variants'][variant]={}
        for alpha in alphas:
            i2t=[]; t2i=[]
            for q in queries:
                aid=str(q['asset_id']); expected=str(q['node_id'])
                if aid not in zh_img or aid not in en_img: continue
                ranked=[]
                for pid in ids:
                    p=zh_by_id[pid]
                    s=alpha*dot(zh_img[aid], zh_txt[pid]) + (1-alpha)*dot(en_img[aid], en_txt[pid])
                    ranked.append({'profile_id':pid,'node_id':str(p['node_id']),'score':float(s),'text_zh':p.get('text'),'text_en':en_by_id[pid].get('text')})
                ranked.sort(key=lambda x:(-x['score'], x['node_id'], x['profile_id']))
                ranked_nodes=[]; seen=set()
                for x in ranked:
                    if x['node_id'] not in seen:
                        seen.add(x['node_id']); ranked_nodes.append(x['node_id'])
                rec={'direction':'image_to_node_text_fusion','variant':variant,'alpha_zh':alpha,'query_asset_id':aid,'expected_node_id':expected,'rank':rank_of({expected}, ranked_nodes),'top_texts':ranked[:10]}
                i2t.append(rec); per.append(rec)
            for pid in ids:
                p=zh_by_id[pid]
                if text_query_node_ids is not None and str(p.get('node_id')) not in text_query_node_ids:
                    continue
                expected_assets=set(gallery_by_node.get(str(p['node_id']), []))
                if not expected_assets: continue
                ranked=[]
                for g in gallery:
                    aid=str(g['asset_id'])
                    if aid not in zh_img or aid not in en_img: continue
                    s=alpha*dot(zh_txt[pid], zh_img[aid]) + (1-alpha)*dot(en_txt[pid], en_img[aid])
                    ranked.append({'asset_id':aid,'node_id':str(g['node_id']),'score':float(s)})
                ranked.sort(key=lambda x:(-x['score'], int(x['asset_id'])))
                ranked_assets=[x['asset_id'] for x in ranked]
                rec={'direction':'node_text_to_image_fusion','variant':variant,'alpha_zh':alpha,'profile_id':pid,'node_id':str(p['node_id']),'rank':rank_of(expected_assets, ranked_assets),'top_images':ranked[:10]}
                t2i.append(rec); per.append(rec)
            results['variants'][variant][str(alpha)]={'image_to_node_text':metrics(i2t, top_ks),'node_text_to_image':metrics(t2i, top_ks),'image_to_node_text_count':len(i2t),'node_text_to_image_count':len(t2i)}
    (out/'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out/'per_query_results.jsonl').open('w', encoding='utf-8') as f:
        for r in per: f.write(json.dumps(r, ensure_ascii=False, default=str)+'\n')
    lines=[f'# {model_key} Chinese-English Score Fusion','',results['note'],'','| Variant | alpha_zh | I2T R@1 | R@5 | R@10 | MRR | T2I R@1 | R@5 | R@10 | MRR |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v, by_alpha in results['variants'].items():
        for a, m in by_alpha.items():
            x=m['image_to_node_text']; y=m['node_text_to_image']
            lines.append(f"| {v} | {float(a):.2f} | {x.get('recall@1',0):.4f} | {x.get('recall@5',0):.4f} | {x.get('recall@10',0):.4f} | {x.get('mrr',0):.4f} | {y.get('recall@1',0):.4f} | {y.get('recall@5',0):.4f} | {y.get('recall@10',0):.4f} | {y.get('mrr',0):.4f} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'output_dir':output_dir,'variants':results['variants']}, ensure_ascii=False, indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', default='data/multimodal_eval/scenic_4_e0_pilot_v3.jsonl')
    p.add_argument('--zh-profiles', default='data/multimodal_eval/pilot_v3_node_text_profiles.jsonl')
    p.add_argument('--en-profiles', default='data/multimodal_eval/pilot_v3_node_text_profiles_en.jsonl')
    p.add_argument('--zh-dir', required=True); p.add_argument('--en-dir', required=True); p.add_argument('--output-dir', required=True); p.add_argument('--model-key', required=True)
    p.add_argument('--alphas', default='0,0.25,0.5,0.75,1'); p.add_argument('--variants', default='T1,T2,T3,T4'); p.add_argument('--top-k', default='1,5,10'); p.add_argument('--text-query-node-ids', default=None)
    a=p.parse_args(); evaluate(a.dataset,a.zh_profiles,a.en_profiles,a.zh_dir,a.en_dir,a.output_dir,a.model_key,[float(x) for x in a.alphas.split(',') if x.strip()],{x.strip() for x in a.variants.split(',') if x.strip()},[int(x) for x in a.top_k.split(',') if x.strip()],load_node_id_filter(a.text_query_node_ids))
if __name__ == '__main__': main()
