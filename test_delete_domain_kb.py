from sqlalchemy import text
from src.rag.dependencies import ai_session_scope
from src.rag.service.domain_kb_service import ingest_domain_kb_files, delete_domain_kb_document, UPLOAD_ROOT
from src.rag.service.embedding_job_service import apply_embedding_job_schema, enqueue_domain_kb_embedding_jobs

sid = 'codex_delete_test'
content = b'Codex delete test document. This is only for deletion verification.'
filename = 'delete-test.txt'

with ai_session_scope() as db:
    apply_embedding_job_schema(db)
    db.execute(text("DELETE FROM text_embeddings WHERE chunk_id IN (SELECT id FROM knowledge_chunks WHERE source_scenic_id=:sid)"), {'sid': sid})
    db.execute(text("DELETE FROM domain_kb_embedding_jobs WHERE source_scenic_id=:sid"), {'sid': sid})
    db.execute(text("DELETE FROM knowledge_chunks WHERE source_scenic_id=:sid"), {'sid': sid})
    db.execute(text("DELETE FROM scenic_areas WHERE source_scenic_id=:sid"), {'sid': sid})

result = ingest_domain_kb_files(source_scenic_id=sid, files=[(filename, content)])
source_id = result['files'][0]['doc_id']
enqueue_domain_kb_embedding_jobs(sid, [source_id])

with ai_session_scope() as db:
    row = db.execute(text("SELECT id, scenic_id, source_node_id FROM knowledge_chunks WHERE source_scenic_id=:sid AND source_id=:source_id LIMIT 1"), {'sid': sid, 'source_id': source_id}).mappings().one()
    db.execute(text("INSERT INTO text_embeddings (scenic_id, chunk_id, source_node_id, embedding, model_name, sync_version, created_at) VALUES (:scenic_id, :chunk_id, :source_node_id, :embedding, 'codex-test', 'codex-test', now())"), {
        'scenic_id': row['scenic_id'],
        'chunk_id': row['id'],
        'source_node_id': row['source_node_id'],
        'embedding': '[' + ','.join(['0'] * 1024) + ']'
    })

before_dir = UPLOAD_ROOT / sid / source_id
print('before_dir_exists=', before_dir.exists())
deleted = delete_domain_kb_document(sid, source_id)
print('delete_result=', deleted)

with ai_session_scope() as db:
    chunks = db.execute(text("SELECT count(*) FROM knowledge_chunks WHERE source_scenic_id=:sid AND source_id=:source_id"), {'sid': sid, 'source_id': source_id}).scalar_one()
    jobs = db.execute(text("SELECT count(*) FROM domain_kb_embedding_jobs WHERE source_scenic_id=:sid AND source_id=:source_id"), {'sid': sid, 'source_id': source_id}).scalar_one()
    embeddings = db.execute(text("SELECT count(*) FROM text_embeddings te JOIN knowledge_chunks kc ON te.chunk_id=kc.id WHERE kc.source_scenic_id=:sid AND kc.source_id=:source_id"), {'sid': sid, 'source_id': source_id}).scalar_one()
    db.execute(text("DELETE FROM scenic_areas WHERE source_scenic_id=:sid"), {'sid': sid})
print('after_counts=', {'chunks': chunks, 'jobs': jobs, 'embeddings': embeddings, 'dir_exists': before_dir.exists()})
