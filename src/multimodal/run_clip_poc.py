"""
PoC: add embedding_clip (vector(512)), backfill CLIP embeddings for seed images,
and run CLIP-based retrieval on test_dataset queries to produce a comparative report.

Usage: python -m src.multimodal.run_clip_poc

Note: Requires `open_clip_torch` or `transformers` + `torch` installed.
"""
import os
import json
from sqlalchemy import create_engine, text
from src.multimodal.clip_feature_extractor import get_clip_extractor
from src.multimodal.image_retrieval_pipeline import DBConfig
from PIL import Image

DATASET_ROOT = os.path.join(os.path.dirname(__file__), 'test_dataset')
REPORT_PATH = os.path.join(DATASET_ROOT, 'eval_report_clip.json')


def ensure_column(engine):
    # Add embedding_clip column if not exists
    with engine.begin() as conn:
        # Check
        exists = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='attraction_images' AND column_name='embedding_clip'"))
        if not exists.fetchone():
            print('Adding column attraction_images.embedding_clip vector(512)')
            conn.execute(text('ALTER TABLE public.attraction_images ADD COLUMN embedding_clip vector(512)'))


def backfill_clip(engine, extractor):
    # For rows with upload_by = 'multimodal_dataset_eval', update embedding_clip by file_path
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT id, file_path FROM attraction_images WHERE upload_by = :u"), {"u": 'multimodal_dataset_eval'}).fetchall()
        print(f'Found {len(rows)} seed rows to update')
        for r in rows:
            fid = int(r.id)
            fp = str(r.file_path)
            if not os.path.isabs(fp):
                # some file_path may be relative; try to resolve inside dataset
                candidate = os.path.join(DATASET_ROOT, fp)
                if os.path.exists(candidate):
                    fp_use = candidate
                else:
                    fp_use = fp
            else:
                fp_use = fp

            try:
                vec = extractor.extract(fp_use)
            except Exception as e:
                print('Failed extract for', fp_use, e)
                continue

            emb_literal = '[' + ','.join(f'{x:.8f}' for x in vec) + ']'
            conn.execute(text('UPDATE attraction_images SET embedding_clip = CAST(:emb AS vector) WHERE id = :id'), {"emb": emb_literal, "id": fid})


def search_clip(engine, query_vec, top_k=5):
    emb_literal = '[' + ','.join(f'{x:.8f}' for x in query_vec) + ']'
    sql = '''
    SELECT ai.id, ai.attraction_id, a.name as attraction_name, ai.file_path, ai.upload_by, (ai.embedding_clip <-> CAST(:embedding AS vector)) as distance
    FROM attraction_images ai
    JOIN attractions a ON a.id = ai.attraction_id
    WHERE ai.embedding_clip IS NOT NULL
    ORDER BY ai.embedding_clip <-> CAST(:embedding AS vector) ASC
    LIMIT :k
    '''
    rows = engine.execute(text(sql), {"embedding": emb_literal, "k": top_k}).fetchall()
    return rows


def evaluate(engine, extractor):
    report = {"details": [], "metrics": {"normal": {"total": 0, "top1_hit": 0, "top3_hit": 0}, "hard": {"total": 0, "top1_hit": 0, "top3_hit": 0}, "negative": {"total": 0, "top1_is_negative_like": 0}}}
    splits = ['normal', 'hard', 'negative']
    for split in splits:
        folder = os.path.join(DATASET_ROOT, 'query', split)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            fp = os.path.join(folder, f)
            try:
                qvec = extractor.extract(fp)
            except Exception as e:
                print('extract fail', fp, e)
                continue

            rows = search_clip(engine, qvec, top_k=5)
            ranked = []
            for r in rows:
                ranked.append({"image_id": int(r.id), "attraction_id": int(r.attraction_id), "attraction_name": str(r.attraction_name), "distance": float(r.distance), "file_path": str(r.file_path)})

            top1_label = None
            if ranked:
                # map attraction_name like '[MM_EVAL] label'
                name = ranked[0]['attraction_name']
                if name.startswith('[MM_EVAL] '):
                    top1_label = name.replace('[MM_EVAL] ', '', 1)
                else:
                    top1_label = 'external'

            expected = None
            if 'great_wall' in f or 'temple' in f:
                expected = 'architecture'
            elif 'landscape' in f or 'mountain' in f:
                expected = 'nature'

            hit_top1 = expected is not None and top1_label == expected
            hit_top3 = expected is not None and any((r.get('attraction_name','').startswith('[MM_EVAL] '+expected)) for r in ranked[:3])

            if split in ('normal','hard'):
                report['metrics'][split]['total'] += 1
                report['metrics'][split]['top1_hit'] += int(hit_top1)
                report['metrics'][split]['top3_hit'] += int(hit_top3)
            else:
                report['metrics'][split]['total'] += 1
                if top1_label in ('animal','vehicle'):
                    report['metrics'][split]['top1_is_negative_like'] += 1

            report['details'].append({"split": split, "query_image": fp, "expected_label": expected, "top1_label": top1_label, "hit_top1": hit_top1, "hit_top3": hit_top3, "ranked": ranked})

    return report


def main():
    cfg = DBConfig()
    engine = create_engine(cfg.sqlalchemy_url, future=True)
    extractor = get_clip_extractor()

    ensure_column(engine)
    backfill_clip(engine, extractor)
    report = evaluate(engine, extractor)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('CLIP eval written to', REPORT_PATH)


if __name__ == '__main__':
    main()
