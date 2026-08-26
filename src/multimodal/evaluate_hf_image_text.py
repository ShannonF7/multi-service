"""Evaluate image-text retrieval for local HuggingFace multimodal models."""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch
from PIL import Image
from sqlalchemy import bindparam, text
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from src.rag.dependencies import get_ai_engine
from src.multimodal.metrics import RankingCase, topk_accuracy
from src.multimodal.node_aggregation import dot, normalize
from src.multimodal.simclr_backfill import DEFAULT_MEDIA_BASE_URL, fetch_image_to_cache

SETTING='cross_domain_zero_shot_image_text'

def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows=[]
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, default=str)+'\n')

def rank_of(expected: set[str], ranked: list[str]) -> int | None:
    for i,x in enumerate(ranked, start=1):
        if x in expected: return i
    return None

def rr(rank: int | None) -> float:
    return 1.0/rank if rank else 0.0

class LocalHFMultiModalEncoder:
    def __init__(self, model_path: str, device: str|None=None):
        self.model_path=model_path
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor=AutoImageProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, use_fast=True)
        self.model=AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        self.model.to(self.device); self.model.eval()
    def encode_image(self, image_path: str) -> list[float]:
        img=Image.open(image_path).convert('RGB')
        inputs=self.image_processor(images=img, return_tensors='pt')
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            if not hasattr(self.model,'get_image_features'):
                raise RuntimeError(f'{self.model_path} has no get_image_features')
            features=self.model.get_image_features(**inputs)
        features=features/features.norm(dim=-1, keepdim=True)
        return [float(x) for x in features.detach().cpu().numpy().flatten().tolist()]
    def encode_texts(self, texts: list[str], batch_size: int=32) -> list[list[float]]:
        out=[]
        if not hasattr(self.model,'get_text_features'):
            raise RuntimeError(f'{self.model_path} has no get_text_features')
        for start in range(0, len(texts), batch_size):
            batch=texts[start:start+batch_size]
            inputs=self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            with torch.no_grad():
                features=self.model.get_text_features(**inputs)
            features=features/features.norm(dim=-1, keepdim=True)
            for vec in features.detach().cpu().numpy().tolist():
                out.append([float(x) for x in vec])
        return out

def load_assets(conn, asset_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not asset_ids: return {}
    stmt=text('''
        select id, scenic_id, source_scenic_id, source_node_id, source_asset_id, role, is_cover, url, file_hash
        from node_assets where id in :asset_ids
    ''').bindparams(bindparam('asset_ids', expanding=True))
    return {int(r['id']):dict(r) for r in conn.execute(stmt, {'asset_ids':asset_ids}).mappings().all()}

def load_or_create_image_vectors(rows, assets, out_dir: Path, encoder, media_base_url: str, force: bool):
    path=out_dir/'image_vectors.jsonl'; vectors={}; failures=[]
    if path.exists() and not force:
        for r in read_jsonl(str(path)): vectors[int(r['asset_id'])]=[float(x) for x in r['vector']]
    cache_rows=[{'asset_id':aid,'vector':vec} for aid,vec in sorted(vectors.items())]
    for aid in sorted({int(r['asset_id']) for r in rows}):
        if aid in vectors: continue
        asset=assets.get(aid)
        if not asset:
            failures.append({'asset_id':aid,'error':'asset_not_found'}); continue
        try:
            img=fetch_image_to_cache(str(asset.get('url') or ''), media_base_url)
            vec=normalize(encoder.encode_image(str(img)))
            vectors[aid]=vec; cache_rows.append({'asset_id':aid,'vector':vec})
        except Exception as e:
            failures.append({'asset_id':aid,'error':str(e)[:500]})
    write_jsonl(path, cache_rows)
    return vectors, failures

def load_or_create_text_vectors(profiles, out_dir: Path, encoder, force: bool):
    path=out_dir/'text_vectors.jsonl'; vectors={}; failures=[]
    if path.exists() and not force:
        for r in read_jsonl(str(path)): vectors[str(r['profile_id'])]=[float(x) for x in r['vector']]
    missing=[p for p in profiles if p['profile_id'] not in vectors]
    if missing:
        texts=[p['text'] for p in missing]
        try:
            encoded=encoder.encode_texts(texts)
            for p,vec in zip(missing, encoded): vectors[p['profile_id']]=normalize(vec)
        except Exception as e:
            failures.append({'error':str(e)[:500], 'count':len(missing)})
    write_jsonl(path, [{'profile_id':pid,'vector':vec} for pid,vec in sorted(vectors.items())])
    return vectors, failures

def load_node_id_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'text query node id file not found: {path}')
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

def metrics_from_ranks(records: list[dict[str, Any]], rank_key: str, top_ks: list[int]) -> dict[str,float]:
    total=len(records)
    if not total: return {**{f'recall@{k}':0.0 for k in top_ks}, 'mrr':0.0}
    out={f'recall@{k}':sum(1 for r in records if r.get(rank_key) and int(r[rank_key]) <= k)/total for k in top_ks}
    out['mrr']=sum(rr(r.get(rank_key)) for r in records)/total
    return out

def evaluate(dataset, profiles_path, output_dir, model_key, model_path, stage, variants, top_ks, media_base_url, force, text_query_node_ids=None):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    ds=[r for r in read_jsonl(dataset) if r.get('usable', True)]
    queries=[r for r in ds if r.get('role')=='query']
    gallery=[r for r in ds if r.get('role')=='gallery']
    profiles_raw=read_jsonl(profiles_path)
    if variants:
        profiles_raw=[p for p in profiles_raw if p.get('profile_variant') in variants]
    profiles=[]
    for p in profiles_raw:
        item=dict(p); item['profile_id']=f"{p['node_id']}::{p['profile_variant']}"; profiles.append(item)
    with get_ai_engine().connect() as conn:
        assets=load_assets(conn, sorted({int(r['asset_id']) for r in ds}))
    t0=time.time(); encoder=LocalHFMultiModalEncoder(model_path)
    image_vectors, image_failures=load_or_create_image_vectors(ds, assets, out, encoder, media_base_url, force)
    text_vectors, text_failures=load_or_create_text_vectors(profiles, out, encoder, force)
    encode_seconds=time.time()-t0
    profiles_by_variant=defaultdict(list)
    for p in profiles:
        if p['profile_id'] in text_vectors: profiles_by_variant[p['profile_variant']].append(p)
    gallery_by_node=defaultdict(list)
    for r in gallery:
        aid=int(r['asset_id'])
        if aid in image_vectors: gallery_by_node[str(r['node_id'])].append(aid)
    results={'model':model_key,'model_path':model_path,'stage':stage,'setting':SETTING,'dataset':dataset,'profiles':profiles_path,'query_count':len(queries),'gallery_count':len(gallery),'profile_count':len(profiles),'text_query_node_count': None if text_query_node_ids is None else len(text_query_node_ids),'encode_seconds':encode_seconds,'image_failures':image_failures,'text_failures':text_failures,'variants':{}}
    per_query=[]
    for variant, profs in sorted(profiles_by_variant.items()):
        image_to_text=[]
        for q in queries:
            aid=int(q['asset_id']); expected=str(q['node_id'])
            if aid not in image_vectors: continue
            ranked=[]
            for p in profs:
                sim=dot(image_vectors[aid], text_vectors[p['profile_id']])
                ranked.append({'profile_id':p['profile_id'],'node_id':str(p['node_id']),'text':p['text'],'similarity':float(sim)})
            ranked.sort(key=lambda x:(-x['similarity'], x['node_id'], x['profile_id']))
            ranked_nodes=[]; seen=set()
            for x in ranked:
                if x['node_id'] not in seen:
                    seen.add(x['node_id']); ranked_nodes.append(x['node_id'])
            rec={'direction':'image_to_node_text','variant':variant,'query_asset_id':aid,'expected_node_id':expected,'rank':rank_of({expected}, ranked_nodes),'top_texts':ranked[:10]}
            image_to_text.append(rec); per_query.append(rec)
        text_to_image=[]
        for p in profs:
            if text_query_node_ids is not None and str(p.get('node_id')) not in text_query_node_ids:
                continue
            expected_assets={str(a) for a in gallery_by_node.get(str(p['node_id']), [])}
            if not expected_assets: continue
            ranked=[]
            for g in gallery:
                aid=int(g['asset_id'])
                if aid not in image_vectors: continue
                sim=dot(text_vectors[p['profile_id']], image_vectors[aid])
                ranked.append({'asset_id':str(aid),'node_id':str(g['node_id']),'similarity':float(sim),'image_url':assets.get(aid,{}).get('url')})
            ranked.sort(key=lambda x:(-x['similarity'], int(x['asset_id'])))
            ranked_assets=[x['asset_id'] for x in ranked]
            rec={'direction':'node_text_to_image','variant':variant,'profile_id':p['profile_id'],'node_id':str(p['node_id']),'text':p['text'],'expected_assets':sorted(expected_assets),'rank':rank_of(expected_assets, ranked_assets),'top_images':ranked[:10]}
            text_to_image.append(rec); per_query.append(rec)
        results['variants'][variant]={'image_to_node_text':metrics_from_ranks(image_to_text,'rank',top_ks),'node_text_to_image':metrics_from_ranks(text_to_image,'rank',top_ks),'image_to_node_text_count':len(image_to_text),'node_text_to_image_count':len(text_to_image)}
    (out/'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    write_jsonl(out/'per_query_results.jsonl', per_query)
    lines=[f'# {stage} {model_key} Image-Text Evaluation','',f'- dataset: `{dataset}`',f'- profiles: `{profiles_path}`',f'- image_failures: {len(image_failures)}',f'- text_failures: {len(text_failures)}',f'- encode_seconds: {encode_seconds:.2f}','','| Variant | Image->Text R@1 | R@5 | R@10 | MRR | Text->Image R@1 | R@5 | R@10 | MRR |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v,m in results['variants'].items():
        a=m['image_to_node_text']; b=m['node_text_to_image']
        lines.append(f"| {v} | {a.get('recall@1',0):.4f} | {a.get('recall@5',0):.4f} | {a.get('recall@10',0):.4f} | {a.get('mrr',0):.4f} | {b.get('recall@1',0):.4f} | {b.get('recall@5',0):.4f} | {b.get('recall@10',0):.4f} | {b.get('mrr',0):.4f} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', default='data/multimodal_eval/scenic_4_e0_pilot_v3.jsonl')
    p.add_argument('--profiles', default='data/multimodal_eval/pilot_v3_node_text_profiles.jsonl')
    p.add_argument('--output-dir', required=True); p.add_argument('--model-key', required=True); p.add_argument('--model-path', required=True); p.add_argument('--stage', required=True)
    p.add_argument('--variants', default='T1,T2,T3,T4'); p.add_argument('--top-k', default='1,5,10'); p.add_argument('--media-base-url', default=DEFAULT_MEDIA_BASE_URL); p.add_argument('--force-recompute', action='store_true'); p.add_argument('--text-query-node-ids', default=None)
    a=p.parse_args(); res=evaluate(a.dataset,a.profiles,a.output_dir,a.model_key,a.model_path,a.stage,{x.strip() for x in a.variants.split(',') if x.strip()},[int(x) for x in a.top_k.split(',') if x.strip()],a.media_base_url,a.force_recompute,load_node_id_filter(a.text_query_node_ids))
    print(json.dumps({'output_dir':a.output_dir,'variants':res['variants'],'image_failures':len(res['image_failures']),'text_failures':len(res['text_failures']),'encode_seconds':res['encode_seconds']}, ensure_ascii=False, indent=2))
if __name__ == '__main__': main()
