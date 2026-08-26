-- Semantic completion async jobs and evidence storage.
-- Safe to rerun: additive changes only.

create table if not exists semantic_completion_jobs (
    id bigserial primary key,
    trace_id varchar(64) unique not null,

    scenic_id bigint not null references scenic_areas(id) on delete cascade,
    node_id bigint not null references semantic_nodes(id) on delete cascade,
    source_scenic_id varchar(255) not null,
    source_node_id varchar(255) not null,

    job_key varchar(128) not null,
    status varchar(32) not null,
    progress integer default 0,
    current_stage varchar(64),

    request_payload jsonb not null,

    question_count integer default 0,
    evidence_count integer default 0,
    candidate_count integer default 0,
    conflict_count integer default 0,

    error_message text,
    attempt_count integer default 0,
    max_attempts integer default 3,
    locked_by varchar(128),
    locked_at timestamptz,
    heartbeat_at timestamptz,
    lease_expires_at timestamptz,
    next_retry_at timestamptz,
    cancel_requested boolean default false,
    worker_version varchar(64),
    pipeline_version varchar(64),
    last_error_code varchar(64),
    last_error_message text,

    created_by varchar(128),
    created_at timestamptz default now(),
    started_at timestamptz,
    finished_at timestamptz,
    updated_at timestamptz default now()
);

alter table if exists semantic_completion_jobs
    add column if not exists current_stage varchar(64);
alter table if exists semantic_completion_jobs
    add column if not exists attempt_count integer default 0;
alter table if exists semantic_completion_jobs
    add column if not exists max_attempts integer default 3;
alter table if exists semantic_completion_jobs
    add column if not exists locked_by varchar(128);
alter table if exists semantic_completion_jobs
    add column if not exists locked_at timestamptz;
alter table if exists semantic_completion_jobs
    add column if not exists heartbeat_at timestamptz;
alter table if exists semantic_completion_jobs
    add column if not exists lease_expires_at timestamptz;
alter table if exists semantic_completion_jobs
    add column if not exists next_retry_at timestamptz;
alter table if exists semantic_completion_jobs
    add column if not exists cancel_requested boolean default false;
alter table if exists semantic_completion_jobs
    add column if not exists worker_version varchar(64);
alter table if exists semantic_completion_jobs
    add column if not exists pipeline_version varchar(64);
alter table if exists semantic_completion_jobs
    add column if not exists last_error_code varchar(64);
alter table if exists semantic_completion_jobs
    add column if not exists last_error_message text;

create index if not exists idx_semantic_completion_jobs_node
    on semantic_completion_jobs (source_scenic_id, source_node_id, created_at desc);
create index if not exists idx_semantic_completion_jobs_status
    on semantic_completion_jobs (status, created_at desc);
create index if not exists idx_semantic_completion_jobs_pick
    on semantic_completion_jobs (status, next_retry_at, created_at);
create index if not exists idx_semantic_completion_jobs_lease
    on semantic_completion_jobs (status, lease_expires_at);
create index if not exists idx_semantic_completion_jobs_job_key
    on semantic_completion_jobs (source_scenic_id, source_node_id, job_key, status);

create table if not exists semantic_completion_questions (
    id bigserial primary key,
    job_id bigint references semantic_completion_jobs(id) on delete cascade,
    trace_id varchar(64) not null,
    source_scenic_id varchar(255) not null,
    source_node_id varchar(255) not null,
    question_id varchar(128) not null,
    target_kind varchar(32) not null,
    target_field varchar(128),
    relation_intent varchar(128),
    temporal_role varchar(64),
    query_text text not null,
    search_terms jsonb default '[]'::jsonb,
    priority integer default 50,
    status varchar(32) default 'planned',
    evidence_count integer default 0,
    candidate_count integer default 0,
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),
    unique(job_id, question_id)
);

create index if not exists idx_semantic_completion_questions_job
    on semantic_completion_questions (job_id, priority desc, id);
create index if not exists idx_semantic_completion_questions_node
    on semantic_completion_questions (source_scenic_id, source_node_id, question_id);

create table if not exists semantic_evidence_items (
    id bigserial primary key,

    trace_id varchar(64) not null,
    job_id bigint references semantic_completion_jobs(id) on delete set null,

    scenic_id bigint not null references scenic_areas(id) on delete cascade,
    node_id bigint not null references semantic_nodes(id) on delete cascade,
    source_scenic_id varchar(255) not null,
    source_node_id varchar(255) not null,

    question_id varchar(128) not null,

    target_kind varchar(32) not null,
    target_field varchar(128),
    relation_intent varchar(128),
    temporal_role varchar(64),

    query_text text not null,

    source_type varchar(64) not null,
    source_title text,
    source_url text,
    source_doc_id varchar(128),
    chunk_id bigint,
    page_no integer,

    quote text,
    content text,

    retrieval_score double precision,
    rerank_score double precision,
    source_weight double precision,
    provenance_type varchar(32),
    retrieval_method varchar(32),
    authority_class varchar(64),
    source_authority_score double precision,
    final_evidence_score double precision,

    metadata jsonb default '{}'::jsonb,

    created_at timestamptz default now()
);

create index if not exists idx_semantic_evidence_items_job
    on semantic_evidence_items (job_id, created_at desc);
create index if not exists idx_semantic_evidence_items_trace
    on semantic_evidence_items (trace_id);
create index if not exists idx_semantic_evidence_items_node
    on semantic_evidence_items (source_scenic_id, source_node_id, question_id);

alter table if exists semantic_evidence_items
    add column if not exists page_no integer;
alter table if exists semantic_evidence_items
    add column if not exists provenance_type varchar(32);
alter table if exists semantic_evidence_items
    add column if not exists retrieval_method varchar(32);
alter table if exists semantic_evidence_items
    add column if not exists authority_class varchar(64);
alter table if exists semantic_evidence_items
    add column if not exists source_authority_score double precision;

alter table if exists semantic_claim_candidates
    add column if not exists job_id bigint references semantic_completion_jobs(id) on delete set null;
alter table if exists semantic_claim_candidates
    add column if not exists question_id varchar(128);
alter table if exists semantic_claim_candidates
    add column if not exists evidence_ids jsonb default '[]'::jsonb;
alter table if exists semantic_claim_candidates
    add column if not exists recommend_score double precision;
alter table if exists semantic_claim_candidates
    add column if not exists support_status varchar(32);

create index if not exists idx_semantic_claim_candidates_job
    on semantic_claim_candidates (job_id, created_at desc);
create index if not exists idx_semantic_claim_candidates_question
    on semantic_claim_candidates (question_id);
