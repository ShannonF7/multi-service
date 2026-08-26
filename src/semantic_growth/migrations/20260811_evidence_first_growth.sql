create table if not exists semantic_growth_evidence_units (
    id bigserial primary key,
    evidence_unit_uid varchar(96) not null unique,
    growth_run_id varchar(64) not null references semantic_growth_runs(growth_run_id) on delete cascade,
    consumption_id bigint references semantic_growth_evidence_consumptions(id) on delete set null,
    source_scenic_id varchar(128) not null,
    source_id varchar(256) not null,
    chunk_id bigint not null,
    chunk_hash varchar(128) not null,
    source_type varchar(64) not null,
    source_title text,
    source_url text,
    content text not null,
    source_authority double precision not null default 0.5,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique(growth_run_id, source_scenic_id, source_id, chunk_id, chunk_hash)
);
create index if not exists semantic_growth_evidence_units_run_idx
    on semantic_growth_evidence_units(growth_run_id, id);

create table if not exists semantic_growth_raw_entities (
    id bigserial primary key,
    raw_entity_uid varchar(96) not null unique,
    growth_run_id varchar(64) not null references semantic_growth_runs(growth_run_id) on delete cascade,
    evidence_unit_id bigint not null references semantic_growth_evidence_units(id) on delete cascade,
    mention_text varchar(512) not null,
    normalized_text varchar(512) not null,
    raw_type varchar(128),
    mention_role varchar(64),
    quote text not null,
    confidence double precision not null default 0,
    resolution_status varchar(32) not null default 'UNRESOLVED',
    resolved_node_id varchar(255),
    node_candidate_id bigint references semantic_node_candidates(id) on delete set null,
    possible_nodes jsonb not null default '[]'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
create index if not exists semantic_growth_raw_entities_run_idx
    on semantic_growth_raw_entities(growth_run_id, evidence_unit_id, resolution_status);

create table if not exists semantic_growth_raw_claims (
    id bigserial primary key,
    raw_claim_uid varchar(96) not null unique,
    growth_run_id varchar(64) not null references semantic_growth_runs(growth_run_id) on delete cascade,
    evidence_unit_id bigint not null references semantic_growth_evidence_units(id) on delete cascade,
    extraction_pass varchar(32) not null,
    subject_text varchar(512) not null,
    subject_type varchar(128),
    claim_type varchar(32) not null,
    raw_predicate varchar(255) not null,
    object_text text not null,
    object_type varchar(128),
    temporal_role varchar(128),
    quote text not null,
    confidence double precision not null default 0,
    status varchar(32) not null default 'RAW',
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
create index if not exists semantic_growth_raw_claims_run_idx
    on semantic_growth_raw_claims(growth_run_id, evidence_unit_id, status);

create table if not exists semantic_growth_candidate_evidence_bindings (
    id bigserial primary key,
    growth_run_id varchar(64) not null references semantic_growth_runs(growth_run_id) on delete cascade,
    candidate_id bigint not null references semantic_claim_candidates(id) on delete cascade,
    evidence_unit_id bigint not null references semantic_growth_evidence_units(id) on delete cascade,
    raw_claim_id bigint references semantic_growth_raw_claims(id) on delete set null,
    source_independence_key varchar(256) not null,
    support_role varchar(32) not null default 'SUPPORTS',
    evidence_score double precision not null default 0,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique(candidate_id, evidence_unit_id, raw_claim_id)
);
create index if not exists semantic_growth_candidate_evidence_idx
    on semantic_growth_candidate_evidence_bindings(candidate_id, support_role);

create table if not exists semantic_growth_fact_evidence_bindings (
    id bigserial primary key,
    binding_uid varchar(96) not null unique,
    growth_run_id varchar(64) not null references semantic_growth_runs(growth_run_id) on delete cascade,
    source_scenic_id varchar(128) not null,
    fact_kind varchar(32) not null,
    source_node_id varchar(255) not null,
    predicate varchar(255) not null,
    normalized_value text,
    target_node_id varchar(255),
    temporal_role varchar(128),
    evidence_unit_id bigint not null references semantic_growth_evidence_units(id) on delete cascade,
    raw_claim_id bigint references semantic_growth_raw_claims(id) on delete set null,
    source_independence_key varchar(256) not null,
    evidence_score double precision not null default 0,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);
create index if not exists semantic_growth_fact_evidence_idx
    on semantic_growth_fact_evidence_bindings(source_scenic_id, source_node_id, predicate);

alter table semantic_claim_candidates
    add column if not exists update_operation varchar(32);
alter table semantic_claim_candidates
    add column if not exists aggregation_key varchar(128);
alter table semantic_claim_candidates
    add column if not exists independent_source_count integer not null default 0;
create index if not exists semantic_claim_candidates_delta_idx
    on semantic_claim_candidates(source_scenic_id, update_operation, status);
