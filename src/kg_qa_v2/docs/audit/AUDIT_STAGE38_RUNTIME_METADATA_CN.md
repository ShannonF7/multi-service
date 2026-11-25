# QA V2 Stage 38：答案模型配置、提示词注册与运行时元数据审计

## 1. 本阶段结论

本阶段已完成并验证：

1. 同步答案生成主链不再无条件固定使用本地模型，而是根据项目根目录 .env 的 ANSWER_PROVIDER 选择模型提供方。
2. 当前生产配置已切换为 OpenAI-compatible API：
   - provider：api
   - base URL：DashScope OpenAI-compatible endpoint
   - model：qwen-plus
   - API key：复用既有 LLM_API_KEY；本报告不记录密钥值
3. Router、槽位抽取、同步答案、流式答案的提示词边界已经集中到提示词注册模块，调用方不再各自维护同一份系统提示词。
4. 同步和流式答案均记录运行时元数据；流式答案能记录首 token 时间、完成原因、输入/输出/总 token 和总耗时。
5. 当流式模型以 finish_reason=length 结束时，SSE 结果增加“答案模型达到输出长度上限，答案可能被截断”的可见警告。
6. 上下文预算修复完成：答案输出上限不会再把小窗口的证据预算完全吃掉；问题和 QuerySpec 先占用预算，证据至少保留一个最小输入预算，并在超限时按预算压缩。

## 2. 实际输入、输出和副作用

### 2.1 答案生成输入

答案生成只接收：

- 用户问题 question；
- 已完成的 QuerySpec；
- 已完成检索并统一成 Evidence 的证据；
- 上下文预算管理器给出的证据片段。

答案模型不负责重新选择领域、重新解析实体、重新生成 Cypher 或重新规划检索。

### 2.2 答案生成输出

同步答案的 QAResponse 新增 runtime 字段，可能包含：

- provider；
- model；
- ttft_ms（流式时）；
- total_ms；
- finish_reason；
- input_tokens；
- output_tokens；
- total_tokens。

如果上游没有返回 usage 或 finish_reason，则只返回能够确定的字段，不伪造 token 数。

流式 SSE 的 completion 事件同时携带 runtime；如果答案达到长度上限，会增加 warning。答案仍然保留已生成内容，不静默伪装成完整答案。

### 2.3 副作用

- 只读取环境配置，不在日志、响应或审计报告中输出 API key。
- 生产答案模型改为 API 后，实际调用会产生供应商 API 费用；本阶段只验证配置解析、代码路径、测试和服务健康，没有主动发起真实答案 API 调用。
- 服务重启会重新加载 .env 和 Python 模块；不会修改图谱、向量库或用户会话数据。

## 3. 提示词注册状态

当前统一注册以下提示词构建器：

- build_router_system_prompt：意图和路由判断；
- build_understanding_system_prompt：问题结构与实体引用抽取；
- build_slot_system_prompt：槽位抽取；
- build_answer_system_prompt：同步和流式答案生成。

答案生成仍保留用户问题、QuerySpec、Evidence 的结构化输入约束；答案模型只负责组织已有证据，不获得数据库访问权限。

## 4. 上下文预算审计

配置项：

- ANSWER_MAX_OUTPUT_TOKENS=768；
- KG_QA_CONTEXT_RESERVED_OUTPUT_TOKENS=768。

代码默认值仍保守为 256，便于没有生产配置的测试和本地运行。

预算顺序：

1. 估算 question 和 QuerySpec；
2. 预留答案输出空间；
3. 至少保留证据输入底线；
4. 证据超限时保留高优先级内容，并通过预算标记压缩文本；
5. 不使用无界拼接，也不把全部会话历史直接塞给答案模型。

本阶段修复了一个回归：当模型输出上限大于小测试窗口时，旧实现会把证据预算压缩为 1 token，导致“已按上下文预算压缩”标记无法出现。现实现会先扣除问题和 QuerySpec，再计算额外输出预留，并为证据保留底线。

## 5. 测试与上线状态

### 自动化测试

- 答案和上下文预算定向测试：5 passed；
- QA V2 全量测试：137 passed，1 warning；
- 唯一 warning 为依赖中的 Pydantic v2 class-based config 弃用提示，不是本阶段代码失败。

### 生产服务

- systemd 服务：fastapi_travel.service；
- 重启后状态：active/running；
- 进程 PID：965646；
- 健康检查：GET http://127.0.0.1:7001/docs 返回 HTTP 200。

## 6. 未完成项和边界

以下事项没有在本阶段伪装成已完成：

1. 长期记忆目前仍是项目已有的按需画像/会话存储路径，不等同于完整的 pgvector archival memory。
2. 还没有完整 LangGraph checkpoint/interrupt 级别的持久化状态机。
3. 还没有把 Python SkillRegistry 进一步标准化为独立 SKILL.md 目录协议。
4. 请求级 fan-out、统一 trace、成本统计和供应商 usage 账单聚合仍需后续补齐。
5. 需要继续建设真实 Gold Set，对 Router、实体对齐、图片 Top-K 重排和引用支持率做阈值验收。
6. 本阶段没有将本地 7B 从项目所有模型角色中删除；它仍可能承担 Router/槽位/回退等角色，变化重点是复杂答案主路径改为可配置 API。

## 7. 审计意见

本阶段改动与目标一致，输入输出边界清晰，生产配置已生效，测试闭合，服务已重新启动并通过健康检查。下一阶段应优先用真实问题集验证：

- API 答案首 token 和总耗时；
- 引用标签与 Evidence ID 的对应关系；
- 达到长度上限时前端是否正确展示 warning；
- 图片问题是否经过图片向量 Top-K 检索和重排，而不是复用当前位置；
- 会话压缩是否保留最近实体、待确认对象和用户当前问题。
