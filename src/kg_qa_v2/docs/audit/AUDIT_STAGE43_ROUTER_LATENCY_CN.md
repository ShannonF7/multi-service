# KG-QA v2 第 43 阶段审计：Router Fast Path 与耗时拆分

## 1. 本阶段目标

本阶段只处理 Router 层的两个问题：

1. 生产环境的 `CascadingDecisionClient` 能否正确使用确定性 Fast Path，避免“你好”“讲个笑话吧”“东门附近有什么”等明确问题先调用本地 7B。
2. 本地决策客户端能否记录请求次数、HTTP 请求耗时、响应解析耗时和生成/首 token 相关耗时，并限制 Router 输出长度。

没有修改实体召回、图谱查询、答案生成、图片重排和会话记忆的业务语义。

## 2. 输入与输出审计

### 输入

- `JevRouter.route()` 接收当前问题、是否有上传图片、目标模态和 `ContextSnapshot`。
- 本地模型客户端仍接收原有 `task="router"` 与结构化 payload。
- 不新增用户可控的数据库查询参数，也不允许模型生成 Cypher。

### 输出

- Fast Path 仍输出原有 `RouteDecision`，例如闲聊为 `route=CASUAL`，附近问题为 `intent=LIST`。
- 模型路径仍输出标准化 Router JSON，不改变已有意图枚举。
- `timings` 新增的数值字段：
  - `router_http_ms`
  - `router_parse_ms`
  - `router_attempt_count`
  - `router_model_calls`
  - 已有的 `router_ttft_ms`、`router_generation_ms`、`router_total_ms` 保持兼容。

## 3. 逻辑审计

### 3.1 Fast Path

`CascadingDecisionClient` 显式声明 `supports_fast_path=True`。`JevRouter` 在调用级联客户端前，先判断明显闲聊和短问题规则；命中后直接使用 `fallback_route()`，不会调用 primary 或 fallback 模型。

这不是针对某个实体写死答案，而是复用已有的意图规则，将执行路径提前确定。复杂、较长或规则无法确认的问题仍进入决策模型。

### 3.2 级联计数

- `LightweightRouterClient` 标记为 `model_backed=False`，不会被计入模型调用次数。
- 本地 HTTP/Transformers/托管决策客户端默认按模型调用计数。
- 级联走 fallback 时，`router_model_calls` 反映实际模型调用数量，不把纯规则 primary 虚计为一次模型调用。

### 3.3 输出上限

本地 Router 默认 `max_tokens=64`，环境变量仍可覆盖，但最终限制在 32 至 128 之间。Router 只需要输出有限 JSON，不应使用答案生成级别的输出长度。

### 3.4 耗时边界

- `router_http_ms`：从发起本地 transport/HTTP 请求到响应返回的耗时，流式请求包含等待首 token 和后续响应读取。
- `router_parse_ms`：解析模型返回 JSON 的耗时。
- `router_ttft_ms`：仅在 Router SSE 收到首个可解析数据块时记录。
- `router_generation_ms`：优先使用流式读取阶段的近似生成耗时；非流式时表示本次请求整体耗时，不能解释为服务端纯 decode 时间。
- `router_queue_ms`：本阶段不伪造。真实服务端排队时间应从 vLLM `/metrics` 读取，不能由客户端往返时间推断。

## 4. 测试审计

新增 `test_stage43_router_latency.py`，覆盖：

1. 生产级联对“讲个笑话吧”不调用模型。
2. “东门附近有什么？”直接进入 `LIST` Fast Path。
3. 本地 Router 默认限制 `max_tokens=64`，并记录请求次数、HTTP 和解析耗时。
4. 轻量 primary + 模型 fallback 时只统计真实模型调用。

测试结果：

```text
8 passed in 0.15s
156 passed, 1 warning in 4.36s
```

另已执行相关文件 `py_compile` 和 `git diff --check`，无本阶段新增错误。

## 5. 风险与未完成项

- 当前 Router 的规则词表仍是有限的确定性入口；它只负责高置信度短问题，不能替代完整 Question Understanding。
- 级联的 `last_meta` 保存了 primary/fallback 来源和 fallback 原因，当前不进入用户答案正文，避免把内部诊断信息混入回答。若前端需要展示，应在后续统一的可审计 trace 字段中透传，而不是拼接到答案。
- 要获得真实的 `queue/prefill/decode` 拆分，仍需按请求标识关联 vLLM `/metrics`，本阶段没有把客户端总耗时冒充成 queue 时间。
