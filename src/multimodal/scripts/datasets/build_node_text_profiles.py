"""Build node text profiles for multimodal pilot v3 using full domain hierarchy."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Any

TYPE_LABELS = {
    'Region': '区域', 'POI': '兴趣点', 'Building': '建筑', 'Floor': '楼层',
    'Room': '房间', 'Facility': '设施', 'Gate': '出入口', 'Road': '道路',
    'Landscape': '景观', 'College': '学院区域', 'Department': '学院区域',
}
TYPE_PURPOSES = {
    'Region': '空间组织、导览和区域定位', 'POI': '参观、识别和导览', 'Building': '教学、科研、办公或公共服务',
    'Floor': '楼层空间组织和位置索引', 'Room': '教学、科研、办公或管理服务', 'Facility': '配套服务和空间使用',
    'College': '教学、科研和办公', 'Department': '教学、科研和办公',
}

def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows=[]
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    out=Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, default=str)+'\n')

def clean(value: Any, max_len: int=600) -> str:
    return ' '.join(str(value or '').strip().split())[:max_len]

def type_label(raw: str, name: str='') -> str:
    raw=clean(raw)
    if name.endswith('学院'):
        return '学院区域'
    if raw.startswith('type_') or raw.startswith('domain_'):
        return '节点'
    return TYPE_LABELS.get(raw, raw or '节点')

def type_purpose(raw: str, label: str) -> str:
    if label == '学院区域': return '教学、科研和办公'
    return TYPE_PURPOSES.get(raw, '导览、识别和知识组织')

def load_tree(path: str) -> dict[str, dict[str, Any]]:
    if not path or not Path(path).exists(): return {}
    out={}
    for row in read_jsonl(path):
        node_id=str(row.get('id') or row.get('node_id') or '')
        if not node_id: continue
        out[node_id]={
            'node_id': node_id,
            'node_name': clean(row.get('name') or row.get('node_name')),
            'node_type': clean(row.get('node_type') or row.get('type')),
            'parent_node_id': str(row.get('parent_id') or row.get('parent_node_id') or ''),
            'description': clean(row.get('description'), 1000),
            'scenic_name': clean(row.get('scenic_name')),
        }
    return out

def hierarchy_ids(node_id: str, nodes: dict[str, dict[str, Any]]) -> list[str]:
    seen=set(); ids=[]; current=node_id
    while current and current not in seen:
        seen.add(current)
        node=nodes.get(current)
        if not node: break
        ids.append(current)
        current=str(node.get('parent_node_id') or '')
    return list(reversed(ids))

def hierarchy_names(node_id: str, nodes: dict[str, dict[str, Any]]) -> list[str]:
    return [clean(nodes[i].get('node_name')) for i in hierarchy_ids(node_id, nodes) if clean(nodes[i].get('node_name'))]

def location_sentence(name: str, hierarchy: list[str]) -> str:
    parents=[x for x in hierarchy if x and x != name]
    if not parents: return name
    return f'{name}位于' + '的'.join(parents)

def description_sentence(name: str, label: str, hierarchy: list[str], raw_desc: str, raw_type: str) -> str:
    if raw_desc: return raw_desc
    loc=location_sentence(name, hierarchy)
    purpose=type_purpose(raw_type, label)
    return f'{loc}，属于{label}，主要用于{purpose}'

def build_profiles(dataset: str, tree: str, output: str) -> dict[str, Any]:
    ds=[r for r in read_jsonl(dataset) if r.get('usable', True)]
    full=load_tree(tree)
    node_ids=sorted({str(r.get('node_id') or '') for r in ds if r.get('node_id')}, key=lambda x: int(x) if x.isdigit() else x)
    roles=defaultdict(set); counts=defaultdict(int); sample={}
    for row in ds:
        nid=str(row.get('node_id') or '')
        if not nid: continue
        roles[nid].add(str(row.get('role') or '')); counts[nid]+=1; sample.setdefault(nid,row)
    rows=[]
    for nid in node_ids:
        base=full.get(nid, {})
        fallback=sample.get(nid, {})
        name=clean(base.get('node_name') or fallback.get('node_name')) or f'node {nid}'
        raw_type=clean(base.get('node_type') or fallback.get('node_type'))
        label=type_label(raw_type, name)
        h=hierarchy_names(nid, full) or [name]
        loc=location_sentence(name, h)
        desc=description_sentence(name, label, h, clean(base.get('description'), 1000), raw_type)
        texts={
            'T1': name,
            'T2': f'{name}，{label}',
            'T3': loc,
            'T4': desc,
        }
        for variant,text in texts.items():
            rows.append({'dataset_version': fallback.get('dataset_version') or 'scenic_4_e0_pilot_v3','source_scenic_id': str(fallback.get('source_scenic_id') or '4'),'node_id':nid,'node_name':name,'node_type':raw_type,'node_type_label':label,'parent_node_id':str(base.get('parent_node_id') or fallback.get('parent_node_id') or ''),'hierarchy_path':' / '.join(h),'profile_variant':variant,'text':clean(text,1200),'has_description':bool(clean(base.get('description'))),'asset_count':counts.get(nid,0),'roles':sorted(r for r in roles[nid] if r)})
    write_jsonl(output, rows)
    return {'dataset':dataset,'tree':tree,'output':output,'nodes':len(node_ids),'profiles':len(rows),'with_description':sum(1 for x in rows if x.get('has_description')),'variants':sorted({x['profile_variant'] for x in rows})}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', default='data/multimodal_eval/scenic_4_e0_pilot_v3.jsonl')
    p.add_argument('--tree', default='data/multimodal_eval/scenic_4_nodes_full.jsonl')
    p.add_argument('--output', default='data/multimodal_eval/pilot_v3_node_text_profiles.jsonl')
    a=p.parse_args(); print(json.dumps(build_profiles(a.dataset,a.tree,a.output), ensure_ascii=False, indent=2))
if __name__ == '__main__': main()
