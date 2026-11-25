# KG-QA v2 第 44 阶段审计：Router 路径元数据透传

## 1. 本阶段目标

把 Router 的运行路径补进同步响应和 SSE 完成事件，解决“看到耗时但不知道到底走了哪条链路”的审计盲区。

本阶段不改变意图判断、实体解析、图谱查询或答案文本，只增加运行元数据。

## 2. 新增输出

`QAResponse.runtime` 和 SSE `completed.data.runtime` 现在可以包含：

- `router_path`：`fast_path`、`decision_model` 或 `fallback`；
- `router_source`：当前 RouteDecision 使用的来源；
- `router_fallback_reason`：级联 primary 未采用时的原因，例如低置信度或异常。

已有答案模型运行信息仍保留，并按答案模型信息覆盖同名字段，避免答案端元数据丢失。

## 3. 处理逻辑审计

1. 每次 Router 请求开始时清理上一轮的 `last_timing` 和 `last_meta`，避免跨请求污染。
2. Fast Path 在调用 `finish()` 时写入 `router_path=fast_path`，不会访问本地 7B。
3. 正常调用决策模型时写入 `router_path=decision_model`。
4. Router 异常降级时写入 `router_path=fallback`。
5. 级联客户端保留自身的 primary/fallback 元数据，由 Router 统一合并。
6. SSE 不再直接引用不存在的 pipeline `result`；由 `QAV2Service.stream_ask()` 将理解阶段元数据作为 `runtime_meta` 显式传给 `QAStreamService`。

## 4. 输入输出边界

- 元数据只来自程序状态和决策客户端，不来自模型自由文本。
- 不向前端暴露 API key、请求头、Prompt 内容或异常堆栈。
- `runtime` 不会被拼接进答案正文，也不会影响引用校验。
- 快速答案、无证据完成、流式模型答案三条 SSE 分支都保留 Router 元数据。

## 5. 测试结果

新增测试：

- Fast Path 能输出 `router_path=fast_path`；
- 模型路径能把 fallback 原因传到 `UnderstandingResult.runtime_meta`；
- 同步/SSE 相关旧测试保持兼容。

结果：

```text
20 passed in 0.19s
158 passed, 1 warning in 11.56s
```

另已执行相关 Python 编译检查和测试回归。唯一 warning 为现有 Pydantic class-based config 弃用提示，不是本阶段引入。

## 6. 后续可继续项

当前 `router_queue_ms` 仍不会由客户端伪造。若要看到 vLLM 服务端真实 queue/prefill/decode，需要另行接入 vLLM `/metrics` 的只读采样或诊断端点，并与请求 trace 做边界说明。
