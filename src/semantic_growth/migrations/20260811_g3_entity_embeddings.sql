create table if not exists semantic_growth_entity_embeddings (
    id bigserial primary key,
    source_scenic_id varchar(128) not null,
    source_node_id varchar(255) not null,
    label_text text not null,
    content_hash varchar(64) not null,
    embedding vector(1024) not null,
    model_name varchar(255) not null default 'bge',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique(source_scenic_id, source_node_id)
);
create index if not exists semantic_growth_entity_embeddings_scope_idx
    on semantic_growth_entity_embeddings(source_scenic_id, source_node_id);
