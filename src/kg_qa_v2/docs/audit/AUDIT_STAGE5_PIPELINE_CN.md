# QA v2 第五阶段中文代码审计：独立主链

## 实现范围

本阶段把前四阶段组合为 `QAV2Pipeline`，仍然只存在于新的 `src/kg_qa_v2` 目录，没有替换现有 API 或 `QAService`。

## 主链输入输出

- 输入：问题、领域、session、user、位置、图片标记和目标模态。
- 输出：上下文、QuerySpec、候选对齐结果、ExecutionPlan、检索证据、警告和状态。
- 状态：`COMPLETED` 或 `NEEDS_CLARIFICATION`。

## 逻辑审计

1. Context Resolver 先执行，后续模块只消费裁剪后的 ContextSnapshot。
2. CASUAL 在实体搜索和检索之前结束，不会误访问图谱。
3. 有候选搜索时，所有目标先完成实体对齐；任何 `AMBIGUOUS` 或 `UNRESOLVED` 都停止检索并返回澄清计划。
4. 只有实体对齐完成后才编译并执行 Graph/Text/Image 计划。
5. Pipeline 不生成答案，不修改会话，不写数据库；这些职责留给后续集成层。

## 已验证行为

- 同名财务部会在检索前停止并要求澄清。
- 唯一高置信度候选可以进入图谱检索。
- 闲聊不会触发候选搜索或检索端口。

## 尚未实现

- 接入真实 published entity gateway。
- 接入真实 Jev/Qwen 客户端配置。
- 接入当前 API 的 SSE trace、answer_delta 和最终结果。
- 接入答案生成、Citation Validation 和同步/异步 Memory Update。
