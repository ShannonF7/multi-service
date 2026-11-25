# QA v2 阶段 11 审计：文本与图片检索适配

## 1. 本阶段目标

把现有领域知识库文本检索和图片向量检索接到 QA v2 的 TEXT/IMAGE 执行端口，并统一输出为 Evidence。

## 2. 实际接入的真实能力

### 文本

生产默认调用：

- src.rag.service.domain_kb_service.search_domain_kb
- retrieval_mode=hybrid
- rerank_mode=separate
- include_images=False

因此文本步骤实际可复用现有：

- pgvector 文本向量召回；
- BM25 风格关键词召回；
- 混合分数融合；
- 独立文本重排服务；
- 当前领域过滤。

文本适配器不在 QA v2 内重新写 SQL，也不把图片结果混入文本 Evidence。

### 图片

生产默认调用：

- 文本找图：search_domain_kb_image_embeddings；
- 图片找图：search_domain_kb_image_by_path。

图片向量由现有 7012 Qwen3-VL-Embedding 服务提供，图片结果保留：

- asset_id；
- source_url；
- 图片标题/描述/OCR 内容；
- 相似度分数；
- 原始图片检索元数据。

如果用户上传图片的路径不存在，适配器返回中文 warning，不调用检索服务。

## 3. 输入审计

| 输入 | 处理 |
|---|---|
| domain_id | 直接传入现有领域知识库服务，限定检索领域 |
| question | 优先使用已重写问题，没有问题时回退到 QuerySpec.targets |
| limit | 在适配器内限制为 1—20；图片默认最多 3 条 |
| image_path | 仅在路径真实存在时进入图片查询 |
| published_only | 由 PlanExecutor 统一传入；文本/图片服务保持只读，不改变图谱发布状态 |

## 4. 输出审计

文本和图片都统一为 Evidence：

- evidence_id：领域 + 文档/切片或资产稳定标识；
- source_type：分别为 text/image；
- content：用于答案生成和引用匹配；
- score：归一化到 0—1；
- source_url：原服务提供时保留；
- asset_id：图片证据保留；
- metadata：保留原始检索元数据，便于审计和前端展示。

生成层后续只消费 Evidence，不直接消费旧检索服务的原始字典。

## 5. 服务健康检查

本阶段测试前检查通过：

- 7012：Qwen3-VL-Embedding 健康；
- 7010：Qwen3-VL-Reranker 健康；
- 7020：Qwen2.5-7B-Instruct-AWQ 本地决策模型健康。

## 6. 测试结果

执行：

- python -m pytest -q src/kg_qa_v2
- python -m compileall -q src/kg_qa_v2

结果：

- 43 passed；
- 编译检查通过。

测试覆盖：

- 文本适配器传递真实混合检索参数；
- 文本结果转换为可引用 Evidence；
- 图片文本查询转换为图片 Evidence；
- 图片资产 ID 和 URL 保留；
- 不存在的用户上传图片被拒绝；
- 不直接写数据库、不调用未注入的假服务。

## 7. 当前边界

- 还未接入答案生成与引用校验；
- 还未接入 SSE 事件流；
- 主 app.py 和 A 端仍未改动；
- 图片路径需要上层 API 保存到允许目录后写入 QuerySpec.constraints.image_path；
- 现有文本检索返回的 source_url 由旧服务决定；若源资料没有 URL，Evidence 只能展示文档/切片信息。

## 8. 审计结论

本阶段通过。QA v2 已拥有真实的 Graph、Text、Image 三类检索端口，可以进入 Evidence Fusion、答案生成、引用校验和 SSE 阶段。
