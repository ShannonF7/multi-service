from sqlalchemy import text
from src.rag.dependencies import ai_session_scope
from src.rag.service.domain_kb_service import list_domain_kb_documents, search_domain_kb
from src.rag.service.embedding_job_service import list_embedding_jobs
sid = "codex_worker_test"
print("docs", list_domain_kb_documents(sid))
print("jobs", list_embedding_jobs(sid))
items = search_domain_kb(sid, "二郎庙 建筑 历史 位置", limit=3)
print("search", len(items), items[0].get("source_type") if items else None)
with ai_session_scope() as db:
    cnt = db.execute(text("select count(*) from text_embeddings where chunk_id in (select id from knowledge_chunks where source_scenic_id=:sid)"), {"sid": sid}).scalar()
    print("embedding_count", cnt)
