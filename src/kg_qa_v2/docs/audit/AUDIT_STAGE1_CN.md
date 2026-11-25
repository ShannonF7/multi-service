# QA v2 第一阶段中文代码审计

## 实现范围

本阶段只新增 `qa_v2_stage`，没有修改现有 `src/kg_qa`、API 路由、数据库、图谱查询或前端。

## 输入

- `question`：用户原始问题。
- `ContextSnapshot`：由上游 Context Resolver 提供的当前领域、会话摘要、最近实体、当前位置和待确认候选。
- `has_image`、`target_modality`：当前请求的模态信息。
- 可选 `router_client`：Jev JSON 路由客户端。
- 可选 `slot_client`：Qwen Structured Output 客户端。

## 输出

- `RouteDecision`：只描述应该进入闲聊、知识问答还是澄清路径。
- `QuerySpec`：问题理解的唯一结构化结果。
- `ExecutionPlan`：由代码编译的 Graph/Text/Image/Clarification 步骤。

## 逻辑审计

1. Router 不接收也不决定 `domain_id`、权限和节点 ID，避免模型越权扩大检索范围。
2. Candidate Search 和 Entity Alignment 尚未在本阶段实现；本阶段的 `targets` 只是自然语言名称，不是已确认节点。
3. PlanCompiler 不让模型生成 Cypher，只根据 `QuerySpec` 选择固定操作。
4. Graph、Text、Image 步骤使用统一计划结构；超过一路检索时标记 `parallel=true`，具体并发执行留给后续 Executor。
5. Jev 或 Qwen 不可用、返回非法 JSON 或校验失败时，系统降级到规则结果，不会阻塞主链。
6. `QuerySpec.source` 能记录模型链路或降级链路，方便 SSE 和审计。

## 已验证行为

- 位置问题生成 Graph node detail 计划。
- 图片问题生成 Image 检索计划。
- 路径问题提取起点、终点并固定最大 4 跳。
- 代词问题只读取传入的最近实体和摘要，不读取全部历史。
- Jev/Qwen 客户端可以替换为测试桩，结构化输出会经过 Pydantic 校验。

## 尚未实现

- 真实 Jev/Qwen HTTP SDK 适配和配置项。
- 当前领域已发布图谱候选召回。
- 实体对齐和歧义确认。
- Graph/Text/Image 实际并发执行。
- Evidence Fusion、Citation Validation、SSE 主链替换。

## 风险

规则降级只用于保证链路可用，不能替代正式的模型分类和实体对齐评估。进入主链前必须增加 Jev/Qwen 的契约测试、错误率统计和领域级 Gold Case。
