alter table if exists semantic_claim_candidates
    add column if not exists canonical_claim_key varchar(128);
alter table if exists semantic_claim_candidates
    add column if not exists conflict_scope_key varchar(128);
alter table if exists semantic_claim_candidates
    add column if not exists trust_version varchar(64);
alter table if exists semantic_claim_candidates
    add column if not exists trust_components jsonb not null default '{}'::jsonb;
alter table if exists semantic_claim_candidates
    add column if not exists final_trust_score double precision;

create index if not exists idx_semantic_claim_candidates_canonical_key
    on semantic_claim_candidates (source_scenic_id, canonical_claim_key);
create index if not exists idx_semantic_claim_candidates_conflict_scope
    on semantic_claim_candidates (source_scenic_id, conflict_scope_key, status);

