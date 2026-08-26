-- Candidate grouping, conflict classification, and gap status.
-- Safe to rerun: additive changes only.

alter table if exists semantic_claim_candidates
    add column if not exists candidate_group_key varchar(128);
alter table if exists semantic_claim_candidates
    add column if not exists value_group_key varchar(128);
alter table if exists semantic_claim_candidates
    add column if not exists conflict_class varchar(32);
alter table if exists semantic_claim_candidates
    add column if not exists gap_status varchar(32);

alter table if exists semantic_claim_candidates
    add column if not exists source_authority_score double precision;
alter table if exists semantic_claim_candidates
    add column if not exists source_weight double precision;
alter table if exists semantic_claim_candidates
    add column if not exists provenance_type varchar(32);
alter table if exists semantic_claim_candidates
    add column if not exists retrieval_method varchar(32);
alter table if exists semantic_claim_candidates
    add column if not exists authority_class varchar(64);
alter table if exists semantic_claim_candidates
    add column if not exists target_node_id varchar(255);
alter table if exists semantic_claim_candidates
    add column if not exists target_node_candidate_id bigint;
alter table if exists semantic_claim_candidates
    add column if not exists entity_resolution_status varchar(32);
alter table if exists semantic_claim_candidates
    add column if not exists possible_nodes jsonb default '[]'::jsonb;

create index if not exists idx_semantic_claim_candidates_group_key
    on semantic_claim_candidates (candidate_group_key, status);
create index if not exists idx_semantic_claim_candidates_gap_status
    on semantic_claim_candidates (gap_status, created_at desc);

create table if not exists semantic_candidate_groups (
    id bigserial primary key,
    candidate_group_key varchar(128) unique not null,
    trace_id varchar(64),
    job_id bigint references semantic_completion_jobs(id) on delete set null,
    scenic_id bigint references scenic_areas(id) on delete cascade,
    source_scenic_id varchar(255) not null,
    source_node_id varchar(255) not null,
    question_id varchar(128),
    claim_type varchar(32) not null,
    predicate varchar(128),
    temporal_role varchar(64),
    conflict_class varchar(32) not null default 'insufficient',
    gap_status varchar(32) not null default 'needs_review',
    candidate_count integer default 0,
    distinct_value_count integer default 0,
    source_count integer default 0,
    best_candidate_uid varchar(128),
    recommend_score double precision default 0,
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create index if not exists idx_semantic_candidate_groups_node
    on semantic_candidate_groups (source_scenic_id, source_node_id, gap_status);
create index if not exists idx_semantic_candidate_groups_job
    on semantic_candidate_groups (job_id, updated_at desc);
create index if not exists idx_semantic_candidate_groups_class
    on semantic_candidate_groups (conflict_class, gap_status, updated_at desc);


create table if not exists semantic_gap_status (
    id bigserial primary key,
    source_scenic_id varchar(255) not null,
    source_node_id varchar(255) not null,
    target_kind varchar(32) not null,
    target_field varchar(128),
    relation_intent varchar(128),
    temporal_role varchar(64),
    status varchar(32) not null default 'pending_review',
    job_id bigint references semantic_completion_jobs(id) on delete set null,
    trace_id varchar(64),
    question_id varchar(128),
    evidence_count integer default 0,
    candidate_count integer default 0,
    conflict_class varchar(32),
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),
    unique(source_scenic_id, source_node_id, target_kind, target_field, relation_intent, temporal_role)
);

create index if not exists idx_semantic_gap_status_node
    on semantic_gap_status (source_scenic_id, source_node_id, status);

create table if not exists semantic_node_candidates (
    id bigserial primary key,
    candidate_uid varchar(128) unique not null,
    trace_id varchar(64),
    job_id bigint references semantic_completion_jobs(id) on delete set null,
    source_scenic_id varchar(255) not null,
    name varchar(512) not null,
    normalized_name varchar(512),
    node_type varchar(128),
    parent_node_id varchar(255),
    parent_hint text,
    evidence_ids jsonb default '[]'::jsonb,
    confidence double precision default 0,
    source_count integer default 0,
    entity_resolution_status varchar(32) default 'NEW_ENTITY',
    status varchar(32) default 'PENDING',
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create index if not exists idx_semantic_node_candidates_domain
    on semantic_node_candidates (source_scenic_id, status, updated_at desc);
