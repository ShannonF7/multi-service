import os
from fastapi.testclient import TestClient
from sqlalchemy import text
from app import app
from src.rag.dependencies import ai_session_scope
sid = 'codex_http_domain_test'
client = TestClient(app)
headers = {'X-API-KEY': os.getenv('API_KEY', '')}
client.delete(f'/rag/domains/{sid}', headers=headers)
resp = client.post('/rag/domains', json={'source_scenic_id': sid, 'source_scenic_pk': 123456, 'code': sid, 'name': 'Codex HTTP Domain Test'}, headers=headers)
print('post_status=', resp.status_code, resp.json())
with ai_session_scope() as db:
    count = db.execute(text('select count(*) from scenic_areas where source_scenic_id=:sid'), {'sid': sid}).scalar()
    print('after_post_count=', count)
resp2 = client.delete(f'/rag/domains/{sid}', headers=headers)
print('delete_status=', resp2.status_code, resp2.json())
with ai_session_scope() as db:
    count = db.execute(text('select count(*) from scenic_areas where source_scenic_id=:sid'), {'sid': sid}).scalar()
    print('after_delete_count=', count)