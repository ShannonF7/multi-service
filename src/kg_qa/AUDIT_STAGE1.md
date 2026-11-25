# 第一阶段中文代码审计

## 输入

- `QARequest` 强制要求 `domain_id`。
- 支持文本、图片、文本加图片三种输入形态。
- `scope` 只允许 `NODE`、`SUBGRAPH`、`DOMAIN`。
- `image_path` 是 B 端内部适配字段，不应直接暴露为浏览器任意路径。

## 输出

- `DomainContext` 统一承载领域标识、图谱领域标识、知识库领域标识和当前范围。
- `InputAnalysis` 输出模态意图、知识意图和候选 mention。
- 所有契约使用结构化 Pydantic 模型，便于后续 API 和测试复用。

## 逻辑

- `DomainGuard` 必须在检索前执行。
- NODE/SUBGRAPH 范围缺少 `current_node_id` 时拒绝执行。
- 空文本且无图片时拒绝执行。
- 默认只使用确定性规则识别意图，保证低延迟和可重复。

## 安全与边界

- 所有后续检索器必须使用 `DomainContext.source_scenic_id` 和 `graph_domain_id`，不能重新读取客户端领域参数。
- `allowed_domains` 当前是可注入的权限校验点；真实用户权限仍由 A 端会话或带领域范围的 token 提供。
- 当前 mention 抽取是基线规则，不应被当作最终实体识别结果。

## 已知限制

- 文本加图片已建立输入契约，联合检索将在后续阶段接入。
- 领域名称暂未从图谱查询回填。
- 尚未修改 `app.py` 注册路由，避免超出“只新增 kg_qa 文件夹”的范围。
