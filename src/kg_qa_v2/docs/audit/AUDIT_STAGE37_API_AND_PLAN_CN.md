# QA v2 阶段审计：API 答案主路径与改造方案对照

审计时间：2026-10-12（Asia/Shanghai）

## 一、本阶段已修复

### 1. API 强答案模型已经接入同步主路径

之前 QAV2Service.from_env() 虽然 AnswerGenerator 支持 API provider，但构造时始终传入本地答案模型地址和模型名，导致同步 ask() 仍然固定使用本地 7B。

本阶段改为统一通过：

- src/kg_qa_v2/service.py::_build_answer_generator_from_env
- ANSWER_PROVIDER
- ANSWER_API_BASE_URL
- ANSWER_API_MODEL
- ANSWER_API_KEY 或 LLM_API_KEY

构造答案生成器。

现在：

- ANSWER_PROVIDER=api：同步复杂答案走 OpenAI-compatible API；
- ANSWER_PROVIDER=local 或未配置：走本地 vLLM；
- provider 写错时直接报配置错误，不再静默降级。

### 2. SSE 答案客户端与同步客户端使用同一套配置

LocalStreamingAnswerClient 现在支持显式继承：

- provider
- base_url
- model
- api_key

API 流式请求同时支持 ANSWER_API_KEY 和 LLM_API_KEY，避免同步与 SSE 使用不同配置。

### 3. 没有改变快速答案策略

LOCATION、简单 FACT、明确闲聊仍可由确定性代码直接回答，不会为了“使用 API”而额外调用大模型。API 强模型主要负责：

- EXPLAIN
- COMPARE
- 复杂多请求
- 证据融合后的自然语言组织

这是有意保留的低延迟路径。

## 二、当前方案落地情况

| 方案项 | 当前状态 | 说明 |
|---|---|---|
| Router 只做意图/上下文判断 | 已完成 | execution policy 由代码编译 |
| Question Understanding | 已完成但有降级 | 有 Qwen 槽位客户端；无 key 时使用确定性 fallback |
| EntityReference | 已完成 | 已进入 QuerySpec 和实体对齐 |
| Identity/Content 边界 | 已完成基础实现 | 通过 entity_role 和候选过滤保护身份召回 |
| Router + Task Skill + Tool | 已完成基础实现 | SkillRegistry 已存在；Graph/Text/Image 是工具边界 |
| Request 拆分与多路执行 | 部分完成 | requests 契约已支持；尚未将多个 request 全部做成 LangGraph Send fan-out |
| ClaimResult/Evidence Binding | 已完成 | 引用 ID 会经过实际 Evidence 校验 |
| Citation Validator | 已完成基础校验 | 简单事实优先确定性绑定；复杂答案可选 DecisionModel 校验 |
| SSE | 已完成 | started/process/token/completed 契约已存在 |
| Token Budget Manager | 已完成基础版本 | 使用预算分配和裁剪，不再只按字符截断 |
| 结构化会话摘要 | 已完成确定性版本 | 目前不是摘要模型生成的高质量语义摘要 |
| 长期记忆 | 部分完成 | 已接入按需读取的用户画像/会话存储；尚未接入 pgvector archival semantic retrieval |
| Prompt 单一来源 | 部分完成 | Router/Understanding 已集中到 prompts.py；答案和槽位客户端仍有少量重复 system 文案 |
| SKILL.md Open Standard 目录 | 未完成 | 当前使用 Python SkillRegistry，没有引入 skills/*/SKILL.md |
| LangGraph State Machine | 未完成 | 当前仍是确定性 Pipeline + pending clarification 持久化 |
| Observability | 部分完成 | timings 已有；trace_id、完整 token usage、cost、tool-call trace 尚未形成统一模块 |
| Benchmark 校准阈值 | 未完成 | 有阈值配置和测试，但还没有正式 Gold Set 指标闭环 |

## 三、输入、输出和副作用审计

### API 答案生成

输入：

- 当前问题
- QuerySpec
- 已融合 Evidence
- 预算裁剪后的证据上下文

不允许：

- API 答案模型重新查数据库
- API 答案模型重新选择 domain
- API 答案模型自由生成 Cypher
- API 答案模型接收未裁剪的全部历史

输出：

- 非流式：JSON answer/citation_labels
- 流式：token 事件，结束时输出 answer/citations/evidence/evidence_bindings/timings

副作用：

- API provider 会产生外部模型调用和费用；
- API key 只从进程环境变量读取，不写入 Prompt、日志和响应；
- 没有 API 配置时不会自动伪造 API 结果。

### 快速答案

LOCATION、简单属性和固定闲聊可能不调用答案模型，直接由图谱证据模板生成。这不是漏接 API，而是为了：

- 降低首 token 延迟；
- 避免简单事实被答案模型改写；
- 保证引用绑定可确定复现。

## 四、当前部署配置结论

当前部署 .env 中已确认并写入：

- LLM_API_KEY：已配置，值未输出；
- ANSWER_PROVIDER=api；
- ANSWER_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1；
- ANSWER_API_MODEL=qwen-plus。

已通过不发起模型请求的方式验证：QAV2 答案生成器实际解析为 openai_compatible、qwen-plus，且能读取 key。

注意：如果当前 app 进程是在修改 .env 之前启动的，需要重启应用进程后新配置才会进入模块级 service 实例。本轮没有直接重启生产服务，也没有使用生产 key 发起真实计费请求。

## 五、本阶段测试

修改前基线：

- QA v2：135 passed, 1 warning

修改后定向测试：

- API/同步服务/SSE：17 passed；配置解析验证通过

需要完成部署配置后再做：

1. 非流式复杂问题实际 API 请求；
2. SSE API 首 token 和结束事件；
3. API 返回截断时的 finish_reason 和 warning；
4. API 失败时是否按产品要求回退本地模型。

## 六、仍需排期的结构性工作

优先级建议：

1. 给 API/本地答案生成增加 finish_reason、usage、TTFT、provider/model 的统一运行记录；
2. 把 slot/answer 的重复 system 文案全部收敛到 Prompt Registry；
3. 把多个 requests 真正编译为并行 Skill 执行，再统一 reducer；
4. 将 pending clarification 改为可恢复 State Machine；当前手工 resume 已能工作，但还不是 LangGraph checkpointer；
5. 建立实体对齐 Gold Set，校准自动选择和澄清阈值；
6. 再决定是否引入 LangGraph，不建议在上述契约稳定前整体替换现有 Pipeline。
