# 第二、三阶段中文代码审计

## 模块职责

- `entity_linker.py`：调用已有 PublishedEntity 搜索，只负责链接结果判定，不写数据库。
- `planner.py`：根据规则生成有限查询动作，不接受模型生成的任意 Cypher。
- `retriever.py`：把查询计划转换成已有图谱、领域知识库和图片检索调用。
- `evidence.py`：把不同来源转换成统一 `EvidenceItem`，按 evidence_id 去重。
- `answer.py`：负责新 AI 适配、默认安全降级和引用校验，不参与候选事实发布。
- `service.py`：按固定阶段串联上述模块，统一捕获失败并返回 QA 运行轨迹。
- `api.py`：只定义 `/kg-qa/query`，没有修改主应用注册表。

## 输入输出审计

- 所有检索都从 `DomainContext` 读取 `source_scenic_id` 或 `graph_domain_id`，不会把客户端领域参数直接传给后端检索器。
- 实体链接结果明确区分 `LINKED`、`AMBIGUOUS`、`UNRESOLVED`。
- 出现歧义时服务返回 `NEEDS_CLARIFICATION`，不会继续生成答案。
- 最终响应包含答案、实体、计划、图谱结果、证据、引用、阶段轨迹和警告。

## 新 AI 审计

- 新 AI 通过 `CallableAIAnswerGenerator` 注入，未绑定旧语义补全服务。
- 每次调用都显式传递 `web_search=False`。
- 提示词明确要求只使用领域证据、禁止联网和禁止编造。
- AI 返回的引用必须存在于本次证据集合，否则会被删除并产生警告。
- 未配置新 AI 时使用无编造的证据拼接降级，不会静默调用旧 Web Search AI。

## 查询和边界审计

- Planner 只允许 `GRAPH_NODE_DETAIL`、`GRAPH_NEIGHBORHOOD`、`GRAPH_PATH`、`TEXT_SEARCH`、`IMAGE_SEARCH`。
- 图谱深度上限为 2，节点上限为 20，边上限由图谱查询结果侧继续受限。
- 图片查询沿用已有统一检索服务；当前仍需要 B 端受控临时文件路径，不能接受浏览器任意本地路径。
- 文本加图片已经支持在分析层同时识别，检索层会分别执行文本和图片步骤；共享跨模态联合理解仍是后续增强项。

## 失败处理审计

- 每个阶段记录 STARTED 和 COMPLETED/FAILED 事件。
- 领域校验、检索、生成等异常均返回 `FAILED` 和警告，不暴露堆栈。
- 证据为空时默认答案明确说明资料不足。
- 证据存在但 AI 未返回有效引用时保留答案，同时增加可见警告，便于后续收紧策略。

## 未完成事项

- `app.py` 尚未注册 `kg_qa.api.router`，这是遵守本轮“只新增文件夹”的结果。
- 尚未绑定具体的新 AI SDK 或 endpoint；部署侧需要注入符合 callable 契约的客户端。
- 尚未建立 QA 持久化表和 SSE；本轮先完成可测试核心，避免在未确定产品契约前修改现有数据库和应用入口。
