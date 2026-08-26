create table if not exists semantic_graph_discovery_jobs (
    id bigserial primary key,
    event_key varchar(160) not null unique,
    domain_identifier varchar(128) not null,
    domain_id varchar(64) null,
    source_graph_sync_job_id bigint null,
    algorithm varchar(64) not null default 'gds_node_similarity_v1',
    payload jsonb not null default '{}'::jsonb,
    status varchar(32) not null default 'PENDING',
    attempt_count integer not null default 0,
    max_attempts integer not null default 3,
    next_retry_at timestamptz null,
    locked_by varchar(128) null,
    locked_at timestamptz null,
    lease_expires_at timestamptz null,
    discovery_count integer not null default 0,
    validation_job_count integer not null default 0,
    error_message text null,
    result jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    started_at timestamptz null,
    finished_at timestamptz null
);

create table if not exists semantic_graph_discoveries (
    id bigserial primary key,
    discovery_key varchar(160) not null unique,
    first_job_id bigint not null,
    last_job_id bigint not null,
    domain_id varchar(64) not null,
    domain_code varchar(128) null,
    discovery_type varchar(48) not null default 'potential_relation',
    algorithm varchar(64) not null,
    source_node_id varchar(128) not null,
    source_name text null,
    source_type varchar(128) null,
    target_node_id varchar(128) not null,
    target_name text null,
    target_type varchar(128) null,
    relation_hint varchar(128) null,
    score double precision not null default 0,
    common_neighbor_count integer not null default 0,
    support jsonb not null default '{}'::jsonb,
    validation_question text not null,
    status varchar(32) not null default 'PENDING_EVIDENCE',
    evidence_job_id bigint null,
    last_error text null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now()
);

create index if not exists semantic_graph_discovery_jobs_pick_idx
    on semantic_graph_discovery_jobs(status, next_retry_at, created_at);

create index if not exists semantic_graph_discoveries_node_idx
    on semantic_graph_discoveries(domain_id, source_node_id, status, updated_at desc);

create index if not exists semantic_graph_discoveries_validation_idx
    on semantic_graph_discoveries(evidence_job_id) where evidence_job_id is not null;
