from sqlalchemy import text
from src.rag.dependencies import ai_session_scope
from src.rag.service.domain_kb_service import upsert_domain_shell, delete_domain_all
sid = 'codex_domain_lifecycle_test'
# cleanup first
try:
    print('cleanup_before=', delete_domain_all(sid))
except Exception as exc:
    print('cleanup_before_error=', exc)
created = upsert_domain_shell({
    'source_scenic_id': sid,
    'source_scenic_pk': 987654,
    'code': sid,
    'name': 'Codex Domain Lifecycle Test',
    'description': 'temporary test domain',
    'location': '0,0',
    'metadata': {'test': True},
})
print('created=', created)
with ai_session_scope() as db:
    row = db.execute(text('select id, source_scenic_id, source_scenic_pk, name from scenic_areas where source_scenic_id=:sid'), {'sid': sid}).mappings().first()
    print('db_row=', dict(row or {}))
deleted = delete_domain_all(sid)
print('deleted=', deleted)
with ai_session_scope() as db:
    count = db.execute(text('select count(*) from scenic_areas where source_scenic_id=:sid'), {'sid': sid}).scalar()
    print('remaining=', count)