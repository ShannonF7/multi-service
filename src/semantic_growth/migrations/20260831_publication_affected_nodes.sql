-- P5：发布记录显式保存受影响正式节点，供后续 affected scope 重算使用。
alter table semantic_growth_publication_records
    add column if not exists affected_node_ids jsonb not null default '[]'::jsonb;
