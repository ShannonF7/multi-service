# QA V2 Stage A-D 中文审计

## Stage A：PublishedEntity 角色

新增 `entity_role` 语义边界：

- `IDENTITY`：可作为用户指代对象的领域实体。
- `CONTENT`：文章、指南、资讯、材料等内容节点。
- `UNKNOWN`：暂时无法可靠判断的旧节点。

旧图谱没有该字段时，读取层按节点类型、父级名称、匹配类型、名称模式、坐标和层级父节点兼容推断。未来同步投影时会把推断结果写入 `PublishedEntity.entity_role`。

## Stage B：实体召回边界

图谱搜索结果现在先分类，再交给实体解析：

- `identity_items` 只进入身份候选。
- `related_items` 只作为相关上下文，不触发身份澄清。

因此“山西张壁古堡”可以成为身份候选，“张壁古堡交通指南”只能成为内容候选。

## Stage C：Router Fast Path

固定问候、笑话、故事、聊天等高确定性请求不再调用 Router 模型；明确的位置、属性、层级、路径问题也可以走确定性规则。闲聊答案对固定请求使用本地模板，不再复用 7B 答案模型。

## Stage D：Router 级联

增加可选的 `CascadingDecisionClient`：

```text
LOCAL_ROUTER_BASE_URL / LOCAL_ROUTER_MODEL
        ↓ 置信度足够
直接返回
        ↓ 低置信度或异常
LOCAL_DECISION_BASE_URL / LOCAL_DECISION_MODEL
```

7B fallback 不再是 Router 的唯一默认策略。Router 请求同时限制 `temperature=0` 和 `max_tokens`，并记录 `router_total_ms`、`router_generation_ms`。当前标准非流式 HTTP 无法真实测量服务端 queue/TTFT；这两个字段只有决策客户端提供对应测量时才写入，未伪造为 0。

## Dry-run 结果

domain `3` 的只读角色审计结果：

- 总节点：266
- `IDENTITY`：192
- `CONTENT`：74
- `UNKNOWN`：0

“张壁古堡交通指南”等节点已被识别为 `CONTENT`，不会进入身份候选。该 dry-run 没有修改 Neo4j。

## 测试与真实验收

- QA v2 全量测试：107 passed。
- Python compileall：通过。
- “张壁古堡在哪里？”：不再返回文章候选澄清，直接进入位置回答。
- “讲个笑话吧”：`router_total_ms=0`、`slot_ms=0`，答案生成约 1ms，总耗时约 93ms，未调用 7B 答案模型。

## 仍需单独处理的事项

1. 当前角色回填仍是代码兼容推断和未来同步写入，尚未执行批量数据库写入。
2. 对复杂闲聊（如“今天好累啊”），如果没有配置轻量 Router，仍会走现有 7B fallback；配置 `LOCAL_ROUTER_*` 后才启用级联。
3. 若要拆出真实 `router_queue_ms` 和 `router_ttft_ms`，需要让本地决策请求使用 SSE，并在 vLLM 返回流中采集首个 token 时间。
