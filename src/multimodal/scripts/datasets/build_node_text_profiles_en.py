"""Build English node text profiles from Pilot v3 Chinese profiles."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any

TYPE_EN = {
    '区域': 'area', '兴趣点': 'point of interest', '建筑': 'building', '楼层': 'floor',
    '房间': 'room', '设施': 'facility', '出入口': 'entrance', '道路': 'road',
    '景观': 'landscape', '学院区域': 'college area', '节点': 'node',
    'Region': 'area', 'POI': 'point of interest', 'Building': 'building',
}
NAME_MAP = {
    '太原理工大学（明向校区）': 'Taiyuan University of Technology Mingxiang Campus',
    '太原理工大学明向校区': 'Taiyuan University of Technology Mingxiang Campus',
    '明向校区': 'Mingxiang Campus',
    '东区': 'East Area', '西区': 'West Area', '南区': 'South Area', '北区': 'North Area',
    '东部教学区': 'Eastern Teaching Area', '西部教学区': 'Western Teaching Area',
    '教学区': 'Teaching Area', '生活区': 'Residential Area', '公寓区': 'Dormitory Area',
    '信息学院': 'College of Information', '计算机与软件学院': 'College of Computer Science and Software Engineering', '物理学院': 'College of Physics', '数学学院': 'College of Mathematics',
    '化学学院': 'College of Chemistry', '机械学院': 'College of Mechanical Engineering',
    '电气学院': 'College of Electrical Engineering', '软件学院': 'College of Software', '计算机与软件': 'Computer Science and Software Engineering',
    '行知楼': 'Xingzhi Building', '图书馆': 'Library', '体育馆': 'Gymnasium', '游泳馆': 'Natatorium',
    '一层': 'first floor', '二层': 'second floor', '三层': 'third floor', '四层': 'fourth floor', '五层': 'fifth floor',
    '六层': 'sixth floor', '七层': 'seventh floor', '八层': 'eighth floor',
    '东门': 'East Gate', '西门': 'West Gate', '南门': 'South Gate', '北门': 'North Gate',
    '教学档案室': 'teaching archives room', '副院长室': 'vice dean office', '办公室': 'office',
    '会议室': 'meeting room', '实验室': 'laboratory', '教室': 'classroom', '报告厅': 'lecture hall',
}
PURPOSE_EN = {
    'college area': 'teaching, research and office work', 'area': 'spatial organization, navigation and positioning',
    'building': 'teaching, research, office work or public services', 'floor': 'floor-level spatial organization and positioning',
    'room': 'teaching, research, office work or administrative services', 'facility': 'supporting services and spatial use',
    'point of interest': 'visiting, recognition and navigation', 'node': 'navigation, recognition and knowledge organization',
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

def clean(v: Any) -> str:
    return ' '.join(str(v or '').strip().split())

def translate_name(name: str) -> str:
    name=clean(name)
    if name in NAME_MAP: return NAME_MAP[name]
    # common numbered building patterns
    m=re.match(r'(.+?)(\d+)号楼$', name)
    if m:
        prefix=NAME_MAP.get(m.group(1), m.group(1))
        return f'{prefix} Building {m.group(2)}'
    m=re.match(r'(.+?)(\d+)层$', name)
    if m:
        prefix=NAME_MAP.get(m.group(1), m.group(1))
        return f'{prefix} floor {m.group(2)}'
    if name.endswith('学院'):
        stem=name[:-2]
        stem_en=NAME_MAP.get(stem, stem)
        return f'College of {stem_en}' if stem_en == stem else stem_en
    return name

def translate_path(path: str) -> list[str]:
    return [translate_name(x.strip()) for x in clean(path).split('/') if x.strip()]

def type_en(label: str, raw_type: str) -> str:
    return TYPE_EN.get(clean(label), TYPE_EN.get(clean(raw_type), clean(label) or clean(raw_type) or 'node'))

def location_sentence(name_en: str, path_en: list[str]) -> str:
    parents=[x for x in path_en if x and x != name_en]
    if not parents: return name_en
    return f'{name_en} is located in ' + ', '.join(reversed(parents))

def description_sentence(name_en: str, label_en: str, path_en: list[str]) -> str:
    loc=location_sentence(name_en, path_en)
    purpose=PURPOSE_EN.get(label_en, 'navigation, recognition and knowledge organization')
    return f'{loc}. It is a {label_en} mainly used for {purpose}.'

def build(input_path: str, output_path: str) -> dict[str, Any]:
    src=read_jsonl(input_path)
    rows=[]
    for row in src:
        name_en=translate_name(row.get('node_name') or '')
        label_en=type_en(row.get('node_type_label') or '', row.get('node_type') or '')
        path_en=translate_path(row.get('hierarchy_path') or row.get('node_name') or '') or [name_en]
        variant=row.get('profile_variant')
        if variant == 'T1': text=name_en
        elif variant == 'T2': text=f'{name_en}, {label_en}'
        elif variant == 'T3': text=location_sentence(name_en, path_en)
        elif variant == 'T4': text=description_sentence(name_en, label_en, path_en)
        else: text=name_en
        out=dict(row)
        out['language']='en'
        out['source_language']='zh'
        out['node_name_en']=name_en
        out['node_type_label_en']=label_en
        out['hierarchy_path_en']=' / '.join(path_en)
        out['text_zh']=row.get('text')
        out['text']=text
        rows.append(out)
    write_jsonl(output_path, rows)
    return {'input':input_path,'output':output_path,'profiles':len(rows),'nodes':len({r['node_id'] for r in rows}),'variants':sorted({r['profile_variant'] for r in rows})}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input', default='data/multimodal_eval/pilot_v3_node_text_profiles.jsonl')
    p.add_argument('--output', default='data/multimodal_eval/pilot_v3_node_text_profiles_en.jsonl')
    a=p.parse_args(); print(json.dumps(build(a.input,a.output), ensure_ascii=False, indent=2))
if __name__ == '__main__': main()
