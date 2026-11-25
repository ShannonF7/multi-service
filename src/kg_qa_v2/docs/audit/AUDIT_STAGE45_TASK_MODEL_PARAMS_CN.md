# KG-QA v2 第 45 阶段审计：按任务划分模型请求参数

## 1. 本阶段结论

本项目现在把“模型服务”与“任务参数”分开处理：同一个本地 OpenAI-compatible vLLM 服务可以承载多个结构化任务，但每次请求根据任务设置自己的输出上限；答案生成则由独立的答案模型配置决定。

## 2. 任务参数

| 任务 | 默认 temperature | 默认 max_tokens | 目的 |
|---|---:|---:|---|
| Router | 0 | 64 | 稳定分类，只返回有限 JSON |
| Entity Alignment | 0 | 96 | 在候选中选择节点，不生成长解释 |
| Citation Validation | 0 | 128 | 输出校验状态、置信度和简短原因 |
| Answer Generation | 0.1 | 由 `ANSWER_MAX_OUTPUT_TOKENS` 控制，当前 `.env` 为 768 | 组织自然语言答案和引用标签 |

Router、实体对齐和引用校验的 `max_tokens` 都有环境变量入口，但分别限制为 32–128、32–192、32–256，避免配置错误导致结构化任务生成过长。

## 3. 当前实际模型链路

已核对项目 `.env` 和运行代码：

```text
ANSWER_PROVIDER=api
ANSWER_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ANSWER_API_MODEL=qwen-plus
```

因此当前答案同步生成和 SSE 流式生成均使用 DashScope OpenAI-compatible API 的 `qwen-plus`，不是本地 7020 的 Qwen2.5-7B。API key 仍通过环境变量读取，审计报告不记录密钥值。

本地 7020 运行的是：

```text
Qwen2.5-7B-Instruct-AWQ
```

当前主要作为本地决策模型/Router fallback；它不是当前主答案模型。

## 4. `max_model_len=2048` 是否合理

当前 7020 启动参数为：

```text
--max-model-len 2048
```

模型 `config.json` 的 `max_position_embeddings=32768`，所以 2048 不是模型原生上限，而是当前 vLLM 服务人为设定的服务窗口。

结论分两种情况：

1. 只用于 Router/结构化决策：基本合理。Router 的请求输出已限制为 64，通常输入也较短，2048 能控制显存并降低排队风险。
2. 用同一 7020 承担复杂答案生成：偏小，不建议。答案需要系统提示、问题规格、证据和输出预留，2048 容易让输入预算、最大输出和 vLLM 总窗口互相挤压。

`max_model_len` 是 vLLM 服务级参数，不能像 `max_tokens` 一样按请求自由切换。若以后答案改回本地，建议单独部署答案 vLLM 实例，使用更大的服务窗口；不要为了答案直接把当前 Router 实例盲目改大。

## 5. 测试

新增任务参数测试，覆盖默认值、temperature、JSON response format 和环境变量上下限。

```text
13 passed in 0.16s
160 passed, 1 warning in 9.39s
```

warning 为现有 Pydantic class-based config 弃用提示，不是本阶段引入。

## 6. 未伪造的部分

本阶段没有把 vLLM 的 `max_model_len` 假装成任务级参数，也没有把 API 答案模型误报为本地模型。若下一步需要本地答案模型，应先单独确定服务窗口、上下文预算和答案流式参数，再进行切换测试。
