# QA v2 阶段 12 审计：答案生成与引用校验

## 1. 本阶段目标

让答案模型只组织已有 Evidence，并保证输出中的引用可以追溯到本次检索结果；不允许答案模型重新选领域、重新找节点或重新执行查询。

## 2. 实际流程

1. Graph/Text/Image 检索输出 Evidence；
2. 为本次 Evidence 生成临时展示标签 E1、E2……；
3. 把问题规格和证据内容传给答案模型；
4. 答案模型输出 answer 和 citation_labels；
5. CitationValidator 删除不存在的引用；
6. 没有引用但存在证据时，补充最相关的 E1；
7. 配置 DecisionModel 时，对带引用的事实句调用 citation_validation；
8. Jev/本地决策模型返回 NOT_ENTAILED 的引用会被移除。

展示层后续可以把 E1 映射到真实 evidence_id、source_url、asset_id，不把内部 evidence_id 原文直接暴露给用户。

## 3. 输入审计

| 输入 | 处理 |
|---|---|
| 用户问题 | 只用于答案表达，不作为新的检索指令 |
| QuerySpec | 作为已完成的问题理解结果，模型不允许覆盖 |
| Evidence | 唯一事实来源，包含证据 ID、内容、来源 URL |
| citation_labels | 只接受 E1 到当前证据数量范围 |
| DecisionModel | 只接收事实句和对应证据，不接收数据库权限或查询能力 |

## 4. 输出审计

AnswerDraft：

- answer：自然语言答案；
- citation_labels：模型声明采用的证据标签。

ValidatedAnswer：

- answer：删除非法引用后的答案；
- citations：label、真实 evidence_id、校验状态、置信度和原因；
- warnings：引用缺失或非法引用等可审计信息。

## 5. 真实模型验证

使用本地 7020 Qwen2.5-7B-Instruct-AWQ，输入：

- 问题：校长办公室在哪里？
- Evidence：校长办公室位于行政楼B座二层。

模型实际返回：

- 答案：校长办公室位于行政楼B座二层。
- 引用标签：E1。

这证明答案生成端点不是测试替身。

## 6. 测试结果

执行：

- python -m pytest -q src/kg_qa_v2
- python -m compileall -q src/kg_qa_v2

结果：

- 46 passed；
- 编译检查通过。

测试覆盖：

- 答案模型只接收当前 Evidence；
- OpenAI-compatible 输出解析；
- 非法引用清理；
- 无引用时补充引用；
- DecisionModel 对事实句进行支持性判断；
- NOT_ENTAILED 引用移除。

## 7. 当前边界

- 当前是同步答案生成，还没有 SSE 事件层；
- 答案生成尚未接入主 app.py；
- A 端尚未接入新的 ValidatedAnswer 数据结构；
- 图片 Evidence 已保留 asset_id/source_url，但答案层尚未按用户意图自动返回图片卡片；
- 引用标签目前是 QA v2 内部的 E1/E2，最终 UI 需要映射成可悬浮、可打开的来源卡片。

## 8. 审计结论

本阶段通过。答案只消费真实检索证据，并已完成本地 Qwen 真实生成验证。下一阶段进入 SSE 事件流和 QA v2 编排出口。
