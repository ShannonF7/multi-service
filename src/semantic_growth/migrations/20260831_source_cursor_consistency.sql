-- P1：为历史 evidence consumption 补齐 SourceCursor 审计记录。
-- 该脚本幂等执行，只新增/重算游标元数据，不修改证据正文、候选或正式图谱。
insert into semantic_growth_source_cursors
    (source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version)
select distinct source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version
from semantic_growth_evidence_consumptions
where consumer_version = 'growth-open-v2'
on conflict (source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version)
do nothing;

-- 游标只有在同一 chunk 的所有 target scope 都 PROCESSED 时才推进。
update semantic_growth_source_cursors c
set expected_scope_count = s.expected_scope_count,
    processed_scope_count = s.processed_scope_count,
    cursor_state = case when s.expected_scope_count > 0
                              and s.expected_scope_count = s.processed_scope_count
                         then 'ADVANCED' else 'OPEN' end,
    advanced_at = case when s.expected_scope_count > 0
                              and s.expected_scope_count = s.processed_scope_count
                        then coalesce(c.advanced_at, now()) else null end,
    updated_at = now()
from (
    select source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version,
           count(*) as expected_scope_count,
           count(*) filter (where state = 'PROCESSED') as processed_scope_count
    from semantic_growth_evidence_consumptions
    where consumer_version = 'growth-open-v2'
    group by source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version
) s
where c.source_scenic_id = s.source_scenic_id
  and c.source_id = s.source_id
  and c.chunk_id = s.chunk_id
  and c.chunk_hash = s.chunk_hash
  and c.consumer_version = s.consumer_version;
