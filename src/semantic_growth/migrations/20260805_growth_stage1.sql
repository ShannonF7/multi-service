create table if not exists semantic_growth_runs (
    id bigserial primary key,
    growth_run_id varchar(64) not null unique,
    thread_id varchar(128) not null unique,
    domain_id varchar(128) not null,
    scenic_id varchar(128),
    seed_node_ids jsonb not null default '[]'::jsonb,
    status varchar(32) not null default 'PENDING',
    iteration integer not null default 0,
    max_iterations integer not null default 1,
    budget integer not null default 1,
    created_by varchar(128),
    stop_reason text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    finished_at timestamptz
);
create index if not exists semantic_growth_runs_status_idx on semantic_growth_runs(status, created_at desc);

create table if not exists semantic_growth_opportunities (
    id bigserial primary key,
    opportunity_id varchar(64) not null unique,
    growth_run_id varchar(64) not null references semantic_growth_runs(growth_run_id) on delete cascade,
    node_id varchar(255),
    opportunity_type varchar(64) not null,
    target_property varchar(128),
    target_relation varchar(128),
    priority double precision not null default 1,
    reason text,
    status varchar(32) not null default 'PENDING',
    dedupe_key varchar(160) not null unique,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
create index if not exists semantic_growth_opportunities_run_idx on semantic_growth_opportunities(growth_run_id, status, priority desc);

create table if not exists semantic_growth_step_records (
    id bigserial primary key,
    growth_run_id varchar(64) not null references semantic_growth_runs(growth_run_id) on delete cascade,
    opportunity_id varchar(64),
    step_name varchar(64) not null,
    status varchar(32) not null,
    input_ref jsonb not null default '{}'::jsonb,
    output_ref jsonb not null default '{}'::jsonb,
    model_name varchar(128),
    tool_name varchar(128),
    error text,
    started_at timestamptz not null default now(),
    finished_at timestamptz
);
create index if not exists semantic_growth_steps_run_idx on semantic_growth_step_records(growth_run_id, started_at);

create table if not exists semantic_growth_candidate_links (
    id bigserial primary key,
    growth_run_id varchar(64) not null references semantic_growth_runs(growth_run_id) on delete cascade,
    opportunity_id varchar(64) not null references semantic_growth_opportunities(opportunity_id) on delete cascade,
    candidate_id bigint not null references semantic_claim_candidates(id) on delete cascade,
    iteration integer not null default 0,
    created_at timestamptz not null default now(),
    unique(growth_run_id, opportunity_id, candidate_id)
);
create index if not exists semantic_growth_candidate_links_run_idx on semantic_growth_candidate_links(growth_run_id, iteration);

alter table semantic_growth_runs
    add column if not exists status_reason_code varchar(64);
alter table semantic_growth_runs
    add column if not exists failed_opportunity_count integer not null default 0;
alter table semantic_growth_runs
    add column if not exists warning_codes jsonb not null default '[]'::jsonb;



create table if not exists semantic_growth_evidence_consumptions (
    id bigserial primary key,
    growth_run_id varchar(64) not null,
    source_scenic_id varchar(128) not null,
    source_id varchar(128) not null,
    chunk_id bigint not null,
    chunk_hash varchar(128) not null,
    consumer_version varchar(64) not null,
    target_scope varchar(128) not null,
    state varchar(32) not null default 'DISCOVERED',
    lease_owner varchar(128),
    lease_expires_at timestamptz,
    attempt_count integer not null default 0,
    result varchar(32),
    error text,
    discovered_at timestamptz not null default now(),
    claimed_at timestamptz,
    processed_at timestamptz,
    updated_at timestamptz not null default now(),
    unique(source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version, target_scope)
);
create index if not exists semantic_growth_evidence_pick_idx
    on semantic_growth_evidence_consumptions(source_scenic_id, state, lease_expires_at, updated_at);

create table if not exists semantic_growth_source_cursors (
    id bigserial primary key,
    source_scenic_id varchar(128) not null,
    source_id varchar(128) not null,
    chunk_id bigint not null,
    chunk_hash varchar(128) not null,
    consumer_version varchar(64) not null,
    expected_scope_count integer not null default 0,
    processed_scope_count integer not null default 0,
    cursor_state varchar(32) not null default 'OPEN',
    advanced_at timestamptz,
    updated_at timestamptz not null default now(),
    unique(source_scenic_id, source_id, chunk_id, chunk_hash, consumer_version)
);

create table if not exists semantic_growth_evidence_mentions (
    id bigserial primary key,
    consumption_id bigint not null references semantic_growth_evidence_consumptions(id) on delete cascade,
    source_scenic_id varchar(128) not null,
    chunk_id bigint not null,
    node_id varchar(128) not null,
    node_name varchar(255) not null,
    node_type varchar(128),
    mention_text varchar(255) not null,
    match_method varchar(32) not null,
    match_score double precision not null default 1,
    created_at timestamptz not null default now(),
    unique(consumption_id, node_id, mention_text)
);
create index if not exists semantic_growth_mentions_chunk_idx
    on semantic_growth_evidence_mentions(source_scenic_id, chunk_id, node_id);


create table if not exists semantic_growth_candidate_dependencies (
    id bigserial primary key,
    growth_run_id varchar(64) not null references semantic_growth_runs(growth_run_id) on delete cascade,
    downstream_candidate_id bigint not null references semantic_claim_candidates(id) on delete cascade,
    upstream_candidate_id bigint references semantic_claim_candidates(id) on delete cascade,
    upstream_node_candidate_id bigint references semantic_node_candidates(id) on delete cascade,
    dependency_type varchar(64) not null,
    state varchar(32) not null default 'PENDING',
    reason text,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (upstream_candidate_id is not null or upstream_node_candidate_id is not null),
    unique(growth_run_id, downstream_candidate_id, upstream_candidate_id, upstream_node_candidate_id, dependency_type)
);
create index if not exists semantic_growth_candidate_dependencies_downstream_idx
    on semantic_growth_candidate_dependencies(growth_run_id, downstream_candidate_id, state);
create index if not exists semantic_growth_candidate_dependencies_upstream_idx
    on semantic_growth_candidate_dependencies(upstream_candidate_id, upstream_node_candidate_id, state);

create table if not exists semantic_growth_publication_records (
    id bigserial primary key,
    growth_run_id varchar(64) not null references semantic_growth_runs(growth_run_id) on delete cascade,
    publication_batch_id varchar(128),
    status varchar(32) not null default 'PENDING',
    candidate_ids jsonb not null default '[]'::jsonb,
    affected_scope jsonb not null default '[]'::jsonb,
    published_candidate_count integer not null default 0,
    warning text,
    error text,
    attempt_count integer not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    published_at timestamptz,
    unique(growth_run_id)
);
create index if not exists semantic_growth_publication_status_idx
    on semantic_growth_publication_records(status, updated_at desc);
