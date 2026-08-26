from sqlalchemy import text
from src.rag.dependencies import ai_session_scope
with ai_session_scope() as db:
    rows = db.execute(text("""
        select table_name, column_name
        from information_schema.columns
        where table_schema='public'
          and column_name in ('scenic_id','source_scenic_id')
        order by table_name, column_name
    """)).fetchall()
    for row in rows:
        print(row)