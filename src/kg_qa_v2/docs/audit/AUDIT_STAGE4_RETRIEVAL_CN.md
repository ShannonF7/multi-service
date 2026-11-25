# QA v2 第四阶段中文代码审计：三路检索与证据融合

## 实现范围

本阶段只新增计划执行器和证据融合器，不修改现有图谱服务、文本检索、图片检索或答案生成。

## 输入输出

- 输入：`ExecutionPlan`、`QuerySpec`、`ContextSnapshot`、问题文本。
- 端口输入：领域、问题、QuerySpec、计划步骤、上下文和 `published_only=true`。
- 输出：统一 `Evidence`、执行步骤和警告。

## 逻辑审计

1. Graph、Text、Image 由独立端口注入，执行器不关心具体 SDK 或数据库。
2. 当计划标记 `parallel=true` 且包含多个检索步骤时使用线程池并行执行。
3. 单路检索失败不会丢弃其他路的有效证据，而是写入 warnings。
4. 澄清计划不会调用任何检索端口。
5. 每个端口调用都携带 `domain_id` 和 `published_only=true`，避免越过当前领域和发布状态约束。
6. 证据按 `evidence_id` 去重，重复证据保留分数更高的版本，再按分数和来源类型排序。

## 已验证行为

- Graph/Text/Image 可以并行执行并统一返回。
- 重复 evidence_id 会被去重。
- 单路失败不影响其他证据。
- 未完成澄清时不会触发检索。

## 尚未实现

- 将现有 `graph_discovery_service`、领域混合检索和图片检索封装成生产端口。
- 证据冲突检测和句子级蕴含校验。
- 与答案生成和 SSE 事件流连接。
