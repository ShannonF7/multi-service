# QA v2 阶段 13 审计：SSE 流式答案出口

## 1. 本阶段目标

提供可被 A 端消费的 SSE 事件迭代器，实现：

- 提问后立即返回开始事件；
- 返回可审计的处理进度；
- 按 token 返回答案；
- 最后返回引用、warning 和总耗时；
- 出错时返回 error 事件，不伪造答案。

## 2. 事件设计

### started

表示小V已接受本次问题。

### process

表示已经完成的公开处理阶段，例如：

- 问题理解；
- 已发布图谱检索；
- 文本检索；
- 图片检索；
- 证据整理。

这里展示的是可审计步骤，不展示模型私有思维链。

### answer_started

返回当前证据数量，表示开始生成答案。

### token

返回答案增量文本，前端可以立即追加到答案区域。

### completed

返回：

- 最终经过引用校验的 answer；
- citations；
- warnings；
- elapsed_ms 总耗时。

### error

返回异常信息和已消耗时间，不返回未经验证的假答案。

## 3. 输入审计

- 问题和 QuerySpec 由上游传入；
- Evidence 是答案唯一事实来源；
- 本模块不访问图谱、数据库或模型之外的工具；
- 流式模型只生成文本和引用标签；
- 最终文本仍经过 CitationValidator。

## 4. 输出审计

SSE 帧格式：

- event 行为事件名称；
- data 行为 JSON；
- 每帧以空行结束；
- encode_sse 可直接交给浏览器 EventSource 或 FastAPI StreamingResponse。

## 5. 真实验证

使用 7020 Qwen2.5-7B-Instruct-AWQ 的 OpenAI-compatible stream=true 接口验证：

- 收到 7 个流式片段；
- 合并文本为“流式连接成功[E1]”；
- 证明 vLLM 流式端点可用。

## 6. 测试结果

执行：

- python -m pytest -q src/kg_qa_v2
- python -m compileall -q src/kg_qa_v2

结果：

- 49 passed；
- 编译检查通过。

测试覆盖：

- 事件顺序；
- 进度事件；
- token 增量；
- 最终引用和总耗时；
- 模型异常事件；
- 浏览器可消费的 SSE 编码。

## 7. 当前边界

- 还没有把 SSE 路由注册到主 app.py；
- 还没有把真实 QAV2Pipeline 的每个阶段事件统一注入；
- 仍需主接口层提供当前用户、领域、session、位置和图片临时文件；
- A 端还需把 process/token/completed 映射到小V对话 UI。

## 8. 审计结论

本阶段通过。QA v2 已具备独立的真实流式答案出口，可以进入最后的编排集成阶段。
