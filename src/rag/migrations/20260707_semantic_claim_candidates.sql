-- Semantic completion candidate pool.
-- Safe to rerun: only additive changes.

create table if not exists semantic_claim_candidates (
    id bigserial primary key,
    candidate_uid varchar(128) unique not null,
    trace_id varchar(64),
    run_id varchar(64),
    scenic_id bigint references scenic_areas(id) on delete cascade,
    source_scenic_id varchar(255) not null,
    source_node_id varchar(255) not null,
    subject_name varchar(255),
    subject_type varchar(64),
    graph_scope varchar(32),
    retrieval_source varchar(64) default 'web',

    claim_id varchar(128),
    claim_type varchar(32) not null,
    candidate_type varchar(64) not null,
    predicate varchar(128),
    object_value text,
    object_name varchar(255),
    object_type varchar(64),
    target_source_node_id varchar(255),

    source_id varchar(128),
    source_title varchar(512),
    source_url text,
    quote text,
    confidence double precision default 0,
    evidence_score double precision default 0,
    evidence_status varchar(32),

    status varchar(32) not null default 'PENDING',
    conflict_key varchar(512),
    conflict_group varchar(128),
    raw_payload jsonb default '{}'::jsonb,
    metadata jsonb default '{}'::jsonb,
    reviewed_by varchar(128),
    reviewed_at timestamptz,
    review_note text,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create index if not exists idx_semantic_claim_candidates_node
    on semantic_claim_candidates (source_scenic_id, source_node_id, claim_type, predicate);
create index if not exists idx_semantic_claim_candidates_trace
    on semantic_claim_candidates (trace_id);
create index if not exists idx_semantic_claim_candidates_status
    on semantic_claim_candidates (status, created_at desc);
create index if not exists idx_semantic_claim_candidates_conflict
    on semantic_claim_candidates (conflict_group, status);
create index if not exists idx_semantic_claim_candidates_metadata_gin
    on semantic_claim_candidates using gin (metadata);

create table if not exists semantic_conflict_groups (
    id bigserial primary key,
    conflict_group varchar(128) unique not null,
    conflict_key varchar(512) not null,
    trace_id varchar(64),
    scenic_id bigint references scenic_areas(id) on delete cascade,
    source_scenic_id varchar(255) not null,
    source_node_id varchar(255) not null,
    claim_type varchar(32) not null,
    predicate varchar(128),
    conflict_type varchar(64) not null,
    candidate_count integer default 0,
    distinct_value_count integer default 0,
    status varchar(32) not null default 'PENDING',
    summary text,
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create index if not exists idx_semantic_conflict_groups_node
    on semantic_conflict_groups (source_scenic_id, source_node_id, claim_type, predicate);
create index if not exists idx_semantic_conflict_groups_status
    on semantic_conflict_groups (status, created_at desc);
