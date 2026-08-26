# Semantic Growth Implementation Status

Updated: 2026-08-09

## Current stage

G6 publication-resume foundation is now connected on the A端 and verified by static/runtime checks. GrowthRun creation is asynchronous: the API returns STARTING immediately and the detail page polls lightweight status JSON until the first trajectory/candidates are available. The live path is now:

evidence consumption -> entity mention alignment -> open-ended candidate extraction -> mixed normalization (including bounded BGE recall) -> conflict quality validation -> candidate dependency persistence -> A端图谱式审核.

Existing BGE text vectors are used for recall only; exact/alias graph matching, reranking context, human review, and publication remain authoritative. The existing A-side writeback creates formal properties/relations, enqueues the existing Neo4j graph-sync outbox, syncs B candidate status to PUBLISHED, and only then resumes the LangGraph interrupt. The A端 growth detail now surfaces writeback and GraphSyncOutbox status, error, and retry counts. Durable GrowthRun publication records and affected-scope reconciliation remain the next hardening item.

## G0 completed

- Removed the previous test GrowthRun records, their opportunities, step records, candidate links, and linked test candidates.
- Removed orphaned LangGraph checkpoints whose thread_id started with growth- and had no GrowthRun record.
- Unified accepted-context lookup from ACCEPTED to ADOPTED.
- Disabled the legacy template-gap completion adapter as a GrowthRun entry path.
- A run without a real G1 evidence batch now ends as NO_CHANGE with an explicit reason instead of PAUSED or COMPLETED.
- Added status_reason_code, failed_opportunity_count, and warning_codes to GrowthRun.
- Added explicit /resume-paused; generic /resume now rejects PAUSED runs.
- Added INVALIDATED as a review terminal status and accepted it in complete-round.
- Added aggregation rules for all-worker failure and partial worker failure.

## G1 completed

- Added durable EvidenceConsumption records with chunk-level claiming, lease expiry, retry state, and idempotency keys.
- Added SourceCursor records with per-chunk expected/processed target-scope counts.
- Added evidence mention persistence with source chunk, matched node, method, and score.
- Growth graph executes load_scope -> load_evidence_batch -> extract_mentions -> align_nodes -> extract_candidates -> normalize_candidates -> validate_conflicts -> persist_dependencies -> aggregate_results.
- Existing node scope falls back to semantic_nodes when Neo4j discovery is unavailable.
- Exact-name alignment only auto-aligns unique published names; duplicate names are deferred for graph-context/vector alignment.
- SourceCursor advances only after all required target scopes for a chunk reach terminal consumption.

## G2 completed foundation

- Added an evidence-scoped open-ended extraction payload with no fixed target fields or relation intents.
- Reused the existing Function Calling extraction, normalization, risk scoring, and conflict grouping service only after passing current evidence chunks as provided_evidence.
- Added candidate links and evidence-based GrowthOpportunity records for the shared review stream.
- Added a published-property filter that marks exact normalized matches as DUPLICATE with PUBLISHED_FACT_EXISTS.
- Candidate extraction failures mark claimed evidence as RETRYABLE; successful runs enter WAITING_REVIEW.

## G3 completed foundation

- Added deterministic predicate/value normalization metadata (g3_normalization) without overwriting review status.
- Reused semantic_nodes names and alias properties for exact/alias entity recall; ambiguous and unmatched targets remain REVIEW_REQUIRED.
- Connected the existing BGE/text_embeddings search through bounded vector evidence recall (maximum six unique target queries per batch).
- Every vector result is stored as evidence recall metadata; vector similarity cannot auto-merge nodes or publish candidates.
- Growth summaries expose normalization updates, vector query/recall counts, and normalization warnings.

## G4 conflict quality foundation

- Added a persistence-level conflict validator after G3.
- A conflict group whose normalized values collapse to one value is marked same_value, removed from candidate conflict display, and moved from CONFLICT to PENDING only when it is still unreviewed.
- ADOPTED, REJECTED, and INVALIDATED candidates are never overwritten by this cleanup.
- Meaningful disagreements, scope mismatches, and entity ambiguities remain reviewable conflicts.

## G5 dependency-aware review foundation

- Added semantic_growth_candidate_dependencies with explicit dependency types: CANDIDATE_CHAIN and RELATION_TO_NEW_ENTITY.
- Semantic upstream state flow is explicit:
  - PENDING upstream -> downstream BLOCKED_BY_DEPENDENCY.
  - ADOPTED/PUBLISHED upstream -> downstream returns to PENDING and becomes reviewable.
  - REJECTED/INVALIDATED upstream -> downstream becomes INVALIDATED.
  - review_terminal = ADOPTED, REJECTED, INVALIDATED; BLOCKED_BY_DEPENDENCY is not terminal.
- Affected source nodes and their parent scope are stored in GrowthRun metadata and returned by the detail API; A端 displays dependency and affected-node counts.
- The A端 growth detail page now shows dependency-aware badges and does not allow “完成本轮审核” while a candidate is still BLOCKED_BY_DEPENDENCY.
- RELATION_TO_NEW_ENTITY now blocks its relation candidate while the new-entity candidate is pending; after the new entity is ADOPTED the relation returns to PENDING, and rejection invalidates the dependent relation.

## G6 publication-resume foundation

- A端“完成本轮审核” now publishes only ADOPTED semantic candidates through the existing publish_semantic_candidates writeback service.
- The existing publication service remains the single formal write path: it writes NodeProperty/NodeRelation records, is idempotent through SemanticWritebackLog, and enqueues the existing Neo4j GraphSyncOutbox.
- After publication, A端 synchronizes B-side candidate statuses to PUBLISHED; a sync warning leaves the batch retryable instead of hiding the warning.
- B端 complete-round accepts PUBLISHED as a post-publication terminal state, then resumes the LangGraph interrupt. Pending/blocked candidates are still rejected.
- Added semantic_growth_publication_records: each completed round records publication batch id, candidate ids, affected scope, status, warning/error, and retry attempts; the record is returned by the GrowthRun detail API.
- A端 submits this publication receipt together with complete-round, so the B端 GrowthRun is auditable even when A-side Neo4j synchronization is still retrying.
- A端 GraphSyncOutbox now reports GRAPH_SYNC_PENDING, PUBLISHED, or GRAPH_SYNC_FAILED back to B through the publication-sync endpoint.
- Evidence source cursors now advance only when every derived target scope for a chunk is in a terminal consumption state; RETRYABLE scopes keep the cursor OPEN.
- If no candidate was adopted, the round resumes without creating an empty publication batch.

## Verification

- Remote source files compile successfully.
- Semantic growth tests: 14 passed.
- Authenticated real-data G5 smoke reached WAITING_REVIEW with 10 temporary candidates, ran G3 normalization with 4 vector-recall queries, persisted 4 RELATION_TO_NEW_ENTITY dependencies, and returned one affected source-node scope.
- A second authenticated smoke exposed one new-entity candidate in the detail API; its relation was BLOCKED_BY_DEPENDENCY, then returned to PENDING after the node candidate was ADOPTED. The full smoke dataset was removed afterward.
- The G5 smoke run, candidates, node candidates, dependency rows, links, opportunities, evidence mentions/consumptions, and source cursor were removed after verification. Growth tables are currently empty; uploaded knowledge chunks and ordinary semantic candidates were not modified.
- A端 Django system check and the growth-detail template compilation both pass after the G5 UI changes.

## Next stage

G6 hardening: reconcile the persisted publication affected scope with GraphSyncOutbox completion, then advance the EvidenceConsumption source cursor only after publication and all target scopes are terminal. The legacy template-gap adapter remains disabled.
