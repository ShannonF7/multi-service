-- RAG one-click scenic sync support.
-- Safe to rerun: only additive changes.

alter table sync_jobs add column if not exists source_system varchar(64) default 'A';
alter table sync_jobs add column if not exists source_job_id varchar(255);
alter table sync_jobs add column if not exists idempotency_key varchar(255);
alter table sync_jobs add column if not exists payload_hash varchar(128);
alter table sync_jobs add column if not exists current_step varchar(128);
alter table sync_jobs add column if not exists counts jsonb default '{}'::jsonb;
alter table sync_jobs add column if not exists diagnostics jsonb default '[]'::jsonb;
alter table sync_jobs add column if not exists metadata jsonb default '{}'::jsonb;
alter table sync_jobs add column if not exists submitted_by varchar(128);

create unique index if not exists uniq_sync_jobs_idempotency_key
    on sync_jobs (idempotency_key)
    where idempotency_key is not null;
create index if not exists idx_sync_jobs_source_system_scenic
    on sync_jobs (source_system, source_scenic_id);
create index if not exists idx_sync_jobs_status_created
    on sync_jobs (status, created_at desc);

create table if not exists node_property_claims (
    id bigserial primary key,
    scenic_id bigint not null references scenic_areas(id) on delete cascade,
    source_scenic_id varchar(255) not null,
    source_property_id varchar(255) not null,
    source_node_id varchar(255) not null,
    property_key varchar(128) not null,
    raw_value text,
    value text,
    value_type varchar(32),
    outer_status varchar(32),
    claim_status varchar(32),
    confidence double precision,
    source_text text,
    source_url text,
    evidence_source_id varchar(255),
    is_locked boolean default false,
    version integer default 1,
    sync_version varchar(64),
    source_table varchar(128) default 'wiki_custom_nodeproperty',
    source_pk varchar(128),
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create unique index if not exists uniq_node_property_claim_source
    on node_property_claims (scenic_id, source_property_id);
create index if not exists idx_node_property_claims_node
    on node_property_claims (scenic_id, source_node_id);
create index if not exists idx_node_property_claims_key
    on node_property_claims (scenic_id, property_key);
create index if not exists idx_node_property_claims_status
    on node_property_claims (scenic_id, claim_status);
create index if not exists idx_node_property_claims_metadata_gin
    on node_property_claims using gin (metadata);

create table if not exists sync_job_events (
    id bigserial primary key,
    job_id varchar(255) not null,
    event_type varchar(64) not null,
    step varchar(128),
    level varchar(32) default 'info',
    message text,
    payload jsonb default '{}'::jsonb,
    created_at timestamptz default now()
);

create index if not exists idx_sync_job_events_job_created
    on sync_job_events (job_id, created_at);
create index if not exists idx_sync_job_events_job_step
    on sync_job_events (job_id, step);
