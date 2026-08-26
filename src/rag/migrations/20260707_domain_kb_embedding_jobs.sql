-- Domain KB embedding job queue.
-- Safe to rerun: additive/idempotent.

create table if not exists domain_kb_embedding_jobs (
    id bigserial primary key,
    job_id uuid unique not null,
    source_scenic_id varchar(255) not null,
    source_id varchar(255) not null,
    status varchar(32) not null default 'PENDING',
    priority integer not null default 100,
    attempts integer not null default 0,
    max_attempts integer not null default 3,
    worker_id varchar(128),
    device varchar(64),
    total_chunks integer not null default 0,
    embedded_chunks integer not null default 0,
    model_name varchar(255),
    error_message text,
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz default now(),
    updated_at timestamptz default now(),
    started_at timestamptz,
    finished_at timestamptz
);

create unique index if not exists uq_domain_kb_embedding_jobs_source
    on domain_kb_embedding_jobs (source_scenic_id, source_id);
create index if not exists idx_domain_kb_embedding_jobs_status
    on domain_kb_embedding_jobs (status, priority, created_at);
create index if not exists idx_domain_kb_embedding_jobs_source_scenic
    on domain_kb_embedding_jobs (source_scenic_id, created_at desc);
