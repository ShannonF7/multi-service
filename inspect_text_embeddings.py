from sqlalchemy import text
from src.rag.dependencies import ai_session_scope
with ai_session_scope() as db:
    rows = db.execute(text("select column_name, data_type from information_schema.columns where table_name='text_embeddings' order by ordinal_position")).fetchall()
    for row in rows:
        print(row)
