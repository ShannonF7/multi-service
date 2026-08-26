# AI Semantic Completion and Graph Growth Status

Updated: 2026-07-21

## 1. Core boundaries

- A-side PostgreSQL models are the formal source of truth.
- B-side semantic tables store questions, evidence, candidates, conflicts, risks, and jobs.
- Neo4j stores only the projection of formally published nodes, properties, and relations.
- Neo4j/GDS output is a hypothesis and is not evidence.
- A hypothesis must go through evidence retrieval and candidate validation before publication.
- Graph growth never writes formal facts directly in the current version.

## 2. Completed work

### 2.1 Completion workflow

- Rule-based question planning for each property, relation, and temporal role.
- Per-question domain KB and Web retrieval instead of one broad query.
- Real evidence records with source type, document ID, chunk ID, page, quote, and content.
- Candidate extraction from collected evidence.
- Entity resolution against formal nodes, aliases, ambiguous matches, and new entity candidates.
- Raw, normalized, and display values.
- Candidate grouping by node, predicate, relation, and temporal role.
- Conflict classes including same value, compatible, multi-value, conflicting, scope mismatch,
  entity ambiguity, weak evidence, and unsupported.
- Explainable recommendation score, risk level, and publication policy fields.
- Persistent completion Jobs, questions, evidence, candidate groups, conflicts, and gap status.
- Node-bound candidate queries so refresh or node switching does not lose results.

### 2.2 A-side review and formal publication

- Candidate review is bound to scenic ID, source node ID, job ID, and trace ID.
- Accepted candidates can be published into Node, NodeProperty, and NodeRelation.
- Formal publication retains evidence IDs, provenance, confidence, and writeback logs.
- Publication creates a transactional GraphSyncOutbox event.
- Celery sends the event to B and retries failures.

### 2.3 Dedicated Neo4j projection

- Dedicated container: travel-rag-neo4j.
- Dedicated local ports, password, project name, and data volumes.
- Domain isolation by domain_id.
- Numeric domain IDs and SC_ scenic codes are both accepted by graph queries.
- PublishedEntity, PublishedFact, KnowledgeDomain, PARENT_OF, HAS_FACT, and
  PUBLISHED_RELATION projections are available.
- Full and incremental projection are durable and idempotent.
- Neighborhood, shortest path, graph pattern, and related entity queries are available.

### 2.4 Controlled graph growth

- Durable semantic_graph_discovery_jobs queue.
- Durable semantic_graph_discoveries records.
- GDS node similarity uses the recommended GDS 2.13 aggregation projection.
- Only same-type nodes without an existing direct published relation are considered.
- Each discovery stores score, common neighbors, question, status, and evidence Job ID.
- GraphSync completion automatically enqueues graph discovery for changed nodes.
- Default similarity cutoff is 0.75.
- Default maximum is five validation tasks per graph sync event.
- Event keys and discovery keys prevent duplicate jobs and duplicate hypotheses.
- Lease recovery, retry backoff, and terminal failure state are implemented.
- Evidence validation reuses the existing asynchronous semantic completion pipeline.
- A validated result enters the normal node candidate pool; it is not directly published.

## 3. Completion modes

### 3.1 Quick completion

- Subject: one current node.
- Execution: synchronous HTTP request.
- Goal: fill selected template gaps only.
- Planning budget: normally up to three questions.
- Web search and Web extraction: disabled.
- Open discovery of extra properties, relations, and entities: disabled.
- Neo4j graph discovery: disabled.
- Current-node scope does not use the domain KB.
- The backend still accepts a subgraph scope for compatibility; if A sends it, domain KB
  evidence can participate.
- Results are persisted even though the response is synchronous.

### 3.2 Deep completion

- Subject: one current node per Job.
- Execution: durable asynchronous semantic_completion_job.
- Goal: fill template gaps and discover evidence-grounded extra properties, relations, and entities.
- Template question budget: normally up to eight questions.
- In subgraph/domain scope, up to three additional Neo4j discovery questions can be added.
- Retrieval runs separately for each question.
- Evidence is aggregated and candidate extraction normally uses one LLM tool-call phase per node Job,
  rather than one LLM call for every retrieval question.
- Current-node scope excludes domain KB and domain graph context and uses provided evidence plus Web.
- Subgraph/domain scope can use domain KB, Web, and published Neo4j hypotheses.
- Local evidence is tried first; Web backfills uncovered questions.
- Results are persisted and restored after page refresh.

### 3.3 Batch completion

- Subject: multiple selected node IDs.
- Execution: one parent Batch Job plus one independent deep completion Job per node.
- Each node has its own questions, evidence, candidates, conflicts, failures, and retries.
- It is not one giant prompt and not one LLM call for all nodes.
- Each node normally performs per-question retrieval followed by one candidate extraction phase.
- Current-node retrieval scope for each batch item excludes the domain KB.
- Subgraph/domain retrieval scope enables the same local KB, Web, and graph discovery behavior as deep mode.
- The parent Batch Job aggregates total, completed, failed, running, candidate, and conflict counts.
- The current deployment has one semantic worker, so node Jobs are independent but mostly processed
  sequentially. True parallel batch execution requires multiple controlled workers or separate queues.

## 4. What automatic growth currently uses

Current automatic growth does not directly mine the text of all accepted evidence.

The implemented flow is:

1. A candidate is reviewed and formally published on A.
2. The formal node/property/relation and its evidence references are projected to Neo4j.
3. GDS analyzes the structure of the published graph.
4. Similar nodes or missing direct links become potential-relation hypotheses.
5. Each hypothesis becomes a new verification question.
6. The verification Job searches the allowed domain KB and Web sources again.
7. Only newly retrieved supporting evidence can produce a candidate.
8. The candidate returns to the existing review and publication workflow.

Therefore, accepted evidence currently contributes in two indirect ways:

- It supports the formal facts and relations that form the trusted graph seed.
- Its evidence IDs and provenance remain attached to the published graph projection.

However, the graph-growth validator does not yet use evidence IDs to directly load accepted evidence
content, adjacent chunks, or all documents from the same trusted source. If the same content is present in
the domain KB or Web, it may be retrieved again, but there is not yet a dedicated accepted-evidence
retriever.

## 5. Current limitations

- Graph discovery starts after future formal graph sync events; historical domains were not bulk backfilled.
- Graph growth currently uses GDS node similarity and common-neighbor support, not calibrated link prediction.
- Graph discovery tasks are not yet shown in a dedicated A-side growth task center.
- Resulting semantic candidates are visible through the existing node candidate pool.
- Graph validation candidates are not automatically published.
- LOW/MEDIUM/HIGH risk is calculated, but unattended formal publication and rollback policy are not complete.
- The semantic worker imports heavy model dependencies and can take several minutes after restart before
  it begins polling graph and semantic queues.
- One worker currently handles graph sync, graph discovery, and semantic completion sequentially.
- Accepted evidence content is not yet a first-class retrieval source for graph growth.
- Image evidence, OCR, image embeddings, and image-to-fact extraction are not connected yet.

## 6. Next implementation stage

### P0: trusted evidence growth and runtime separation

1. Implement trusted_evidence_retriever.py.
2. Resolve evidence IDs on published facts and relations back to semantic evidence and knowledge chunks.
3. Expand from accepted evidence to adjacent chunks and the same trusted document.
4. Store seed_evidence_ids and retrieval lineage on every graph discovery.
5. Retrieval order: accepted evidence, adjacent trusted chunks, domain KB, then Web backfill.
6. Split graph_sync_worker / graph_growth_worker from the heavy semantic model worker.
7. Add systemd services with automatic startup, lease recovery, and health checks.
8. Add a controlled historical backfill command with domain selection, dry-run, and task budgets.

### P1: A-side growth observability

1. Add A proxy APIs for graph discovery Jobs and discoveries.
2. Add a growth task center showing graph signal, verification status, evidence Job, and candidate result.
3. Show why a discovery was proposed without presenting graph similarity as evidence.
4. Add retry, cancel, suppress, and never-suggest-this-pair controls.

### P2: risk-controlled automatic publication

1. Define a strict LOW-risk allowlist by property/relation impact.
2. Require trusted source policy, sufficient evidence score, no conflict, and graph consistency.
3. Add publication kill switch, daily quota, audit log, and rollback/retraction support.
4. LOW risk may auto-publish; MEDIUM remains batch review; HIGH requires manual review.

### P3: stronger graph algorithms and evaluation

1. Calibrate node similarity by domain and node type.
2. Add link prediction, community detection, and domain-specific graph features.
3. Build an offline evaluation set from historical human review decisions.
4. Measure precision, evidence success rate, duplicate rate, conflict rate, and reviewer acceptance rate.

### P4: multimodal and GraphRAG/KAG

1. Add OCR, caption, image embedding, and image evidence records.
2. Bind images to published facts and entities with provenance.
3. Add text-to-image, image-to-image, and image-to-text retrieval.
4. Add GraphRAG community summaries for cross-document questions.
5. Add schema-aware KAG/OpenSPG-style multi-hop reasoning only after the formal graph and evidence
   policies are stable.

## 7. Immediate acceptance criteria for the next stage

- A graph discovery exposes the exact accepted evidence IDs that seeded it.
- Validation can read accepted evidence content without requiring a Web search.
- Web is used only when trusted local evidence is insufficient.
- Graph workers start quickly without loading embedding or generation models.
- Historical backfill can be limited to one domain and a fixed daily validation budget.
- No graph algorithm output can bypass evidence verification or publication policy.
