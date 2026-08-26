-- Entity resolution, explainable ranking, and publication risk policy.
-- Safe to rerun: additive changes only.

alter table if exists semantic_node_candidates
    add column if not exists entity_group_key varchar(128);
alter table if exists semantic_node_candidates
    add column if not exists raw_type varchar(128);
alter table if exists semantic_node_candidates
    add column if not exists suggested_type varchar(128);
alter table if exists semantic_node_candidates
    add column if not exists type_confidence double precision default 0;

create index if not exists idx_semantic_node_candidates_group
    on semantic_node_candidates (source_scenic_id, entity_group_key, status);

alter table if exists semantic_claim_candidates
    add column if not exists risk_level varchar(16);
alter table if exists semantic_claim_candidates
    add column if not exists publication_policy varchar(32);
alter table if exists semantic_claim_candidates
    add column if not exists score_components jsonb default '{}'::jsonb;
alter table if exists semantic_claim_candidates
    add column if not exists raw_type varchar(128);
alter table if exists semantic_claim_candidates
    add column if not exists suggested_type varchar(128);
alter table if exists semantic_claim_candidates
    add column if not exists type_confidence double precision default 0;

alter table if exists semantic_candidate_groups
    add column if not exists risk_level varchar(16);
alter table if exists semantic_candidate_groups
    add column if not exists publication_policy varchar(32);
alter table if exists semantic_candidate_groups
    add column if not exists score_components jsonb default '{}'::jsonb;

create index if not exists idx_semantic_claim_candidates_risk
    on semantic_claim_candidates (source_scenic_id, risk_level, status, updated_at desc);
create index if not exists idx_semantic_candidate_groups_risk
    on semantic_candidate_groups (source_scenic_id, risk_level, gap_status, updated_at desc);
