# QA v2 阶段 10 审计：已发布图谱适配与实体对齐闭环

## 1. 本阶段目标

把 QA v2 的实体候选搜索、实体对齐结果接入现有已发布图谱服务，并将节点详情转换为统一的 Evidence，供后续答案生成和引用校验使用。

## 2. 实际改动

### 2.1 图谱服务适配

新增 graph_adapters.py，提供 PublishedGraphGateway：

- 候选搜索调用现有 search_published_entities；
- 节点详情调用现有 get_published_node_detail；
- 层级邻域调用现有 get_published_neighborhood；
- 路径查询调用现有 find_published_shortest_path；
- 所有入口强制携带 published_only=True；
- 领域由调用上下文传入，不从模型输出中接受领域覆盖；
- 不允许 QA v2 自由拼接 Cypher。

### 2.2 实体对齐结果进入查询计划

修改 PlanCompiler 和 QAV2Pipeline：

1. 候选搜索得到候选；
2. 对齐模块选择唯一 node_id；
3. 流水线把已链接的节点 ID 写入 QuerySpec.constraints.resolved_node_ids；
4. 图谱步骤携带 node_id；
5. 图谱 Gateway 优先使用已对齐节点，不再错误地回退到当前位置节点。

路径计划同时预留 start_node_id 和 end_node_id，只有两个端点都完成对齐时才执行路径查询。

### 2.3 图谱结果到引用证据

当前转换覆盖：

- 节点坐标和已发布图片数量；
- 节点描述；
- 节点属性/事实；
- 节点关系；
- 邻域节点；
- 路径结果；
- 图片资产 URL 和资产 ID；
- 图谱来源信息中的 source_url、引用片段等。

## 3. 输入审计

| 输入 | 来源 | 是否信任 |
|---|---|---|
| domain_id | Context Resolver / 服务调用方 | 受控 |
| mention | 用户问题经 Question Understanding 抽取 | 需经过候选检索 |
| node_id | 候选搜索和 Entity Alignment | 只接受候选集合中的节点 |
| operation | PlanCompiler 固定生成 | 受控 |
| published_only | Gateway/Executor 强制传入 | 强制为真 |
| 图谱返回内容 | 现有只读图谱服务 | 作为证据，不作为新的规划指令 |

## 4. 输出审计

### 正常输出

- 候选搜索输出 CandidateSearchResult；
- 图谱查询输出 Evidence 列表；
- 每条证据具有稳定 evidence_id、source_type 和 content；
- 节点图片以 asset_id/source_url/metadata.assets 保留，后续答案层可按图片意图选择 1—3 张，而不是默认全部展示。

### 异常输出

- 未发布查询：直接抛出 ValueError；
- 没有对齐节点：返回空证据并附带中文 warning；
- 路径缺少任一端点：不执行查询，返回缺失端点 warning；
- 图谱节点不存在：返回空证据，不伪造答案。

## 5. 测试结果

执行：

- python -m pytest -q src/kg_qa_v2
- python -m compileall -q src/kg_qa_v2

结果：

- 40 passed in 3.89s
- 编译检查通过。

测试覆盖：

- 已发布候选搜索；
- 拒绝 published_only=False；
- 对齐节点 ID 传入计划和检索器；
- 节点描述、坐标、事实、关系、图片资产转为 Evidence；
- 对齐节点优先于当前位置节点。

## 6. 当前边界

- 本阶段还没有把 QA v2 接入主 app.py；
- 文本和图片 Retriever 仍需要对接现有真实服务；
- 图谱 Evidence 已可用于答案生成，但答案生成、引用校验和 SSE 仍属于后续阶段；
- 本阶段没有修改旧 src/kg_qa，避免新旧链路混杂。

## 7. 审计结论

本阶段通过。实体对齐到已发布图谱详情的输入输出链路已经闭合，可以进入文本/图片检索适配阶段。
