select pg_advisory_xact_lock(hashtext('rag_graph_sync_schema_v1'));

create table if not exists semantic_graph_sync_jobs (
    id bigserial primary key,
    event_key varchar(160) not null unique,
    domain_id varchar(64) not null,
    payload jsonb not null default '{}'::jsonb,
    status varchar(20) not null default 'PENDING',
    attempt_count integer not null default 0,
    max_attempts integer not null default 5,
    next_retry_at timestamptz null,
    locked_by varchar(128) null,
    locked_at timestamptz null,
    lease_expires_at timestamptz null,
    error_message text null,
    result jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    started_at timestamptz null,
    finished_at timestamptz null
);

create index if not exists semantic_graph_sync_jobs_pick_idx
on semantic_graph_sync_jobs(status, next_retry_at, created_at);

create index if not exists semantic_graph_sync_jobs_domain_idx
on semantic_graph_sync_jobs(domain_id, created_at desc);
