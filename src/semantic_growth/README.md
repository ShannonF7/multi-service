# 自增长模块说明

本目录只负责证据驱动的 GrowthRun，不负责 A 端节点语义补全。补全仍由
`src/rag/service/semantic_completion_service.py` 独立处理。

## 主流程

`service.py` 创建并启动 LangGraph；`graph.py` / `b_graph.py` 执行状态图；
`discovery_orchestrator.py` 编排证据单元、开放抽取、归一化和 KG Delta；
`kg_delta_service.py` 将规范化声明分为 `EXISTS`、`ADD`、`MINT_ADD`、
`CONFLICT` 等操作；`repository.py` 负责 GrowthRun 和审核数据持久化。

## 文件与主要函数

| 文件 | 主要职责 | 关键函数（输入 → 输出） |
| --- | --- | --- |
| `service.py` | 启停 GrowthRun、恢复 checkpoint | `start_growth_run(payload)` → 任务状态；`growth_run_state(id)` → 详情 |
| `graph.py` | LangGraph 自增长工作图 | `load_scope(state)` → 已发布节点；`load_evidence_batch(state)` → 证据批次；`open_discovery_batch(state)` → 本批候选；`aggregate_results(state)` → 汇总状态 |
| `b_graph.py` | B 端兼容工作图入口 | 与 `graph.py` 使用同一证据和 KG Delta 服务 |
| `evidence.py` | 证据消费、游标、失败重试 | `claim_evidence_batch(...)` → 可消费证据；`finalize_open_discovery_batch(...)` → 消费结果和游标状态 |
| `open_discovery_service.py` | EvidenceUnit 开放式抽取 | `materialize_evidence_units(...)` → 证据单元；`discover_evidence_units(...)` → 原始实体/声明 |
| `candidate_aggregation_service.py` | 实体、谓词、值规范化和声明聚合 | `resolve_canonicalize_and_aggregate(...)` → Canonical Claim 聚合结果 |
| `kg_delta_service.py` | 与正式图谱比较并持久化差量 | `classify_kg_deltas(...)` → Delta 列表；`persist_kg_deltas(...)` → 候选/事实绑定 |
| `repository.py` | GrowthRun、机会、步骤和发布记录 | `create_run(payload)` → 任务记录；`get_run_detail(id)` → 详情数据 |
| `dependencies.py` | 候选依赖和阻塞状态 | `persist_candidate_dependencies(...)` → 依赖记录；`refresh_dependency_states(...)` → 状态更新 |
| `audit_service.py` | G2.5 只读谱系审计 | `audit_growth_run(id)` → 计数、比率、问题和样本，不修改任何状态 |
| `api.py` / `b_growth_api.py` | B 端 HTTP 接口 | `/rag/growth-runs/{id}/audit` → 调用 `audit_growth_run` |

## 调用边界

- `graph.py` 和 `b_graph.py` 只能调用自增长服务，不应重新实现补全抽取。
- `semantic_completion_adapter.py` 仅用于兼容旧节点补全机会；开放式增长主路径使用
  `open_discovery_service.py`，不能把固定属性缺口当作增长入口。
- `audit_service.py` 是只读服务，不得在其中写入候选、修改消费状态或恢复任务。
- BGE、Reranker 和图上下文属于后续 G3 实体统一阶段；G2.5 审计不触发模型推理。
