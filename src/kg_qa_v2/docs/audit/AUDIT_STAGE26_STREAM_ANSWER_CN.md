# Stage 26 SSE 引用、快速答案与组织化输出审计

## 问题定位

1. 图谱检索已经返回 Evidence，但 B 端流式 completed 事件没有透传 evidence；A 端按 evidence_id 过滤 citations 时找不到对应证据，因此显示引用来源 0。
2. 简单位置问题虽然可以使用确定性图谱答案，但之后仍调用 CitationValidator 的 Jev 语义校验，首 token 被模型调用阻塞。
3. 原答案组织规则不足，容易把图谱关系原文直接堆叠。

## 修改

- streaming.py：completed 事件统一透传 evidence。
- answering.py：CitationValidator 增加 semantic 开关；确定性图谱答案关闭重复语义校验。
- service.py：同步答案对快速图谱答案关闭语义校验，同时兼容旧版注入式 Validator。
- answering.py：位置答案优先组织为“名称位于层级路径”，坐标作为独立段落，引用分别绑定。
- answering.py、streaming.py：答案提示词要求先结论、自然分段、避免重复和原始关系堆叠。
- A 端 widget：当 entities 为空时从 evidence.metadata.root 生成相关位置展示，避免已拿到图谱证据却显示没有识别位置。

## 输入输出审计

- 输入：QuerySpec、已发布图谱 Evidence、答案模型或快速答案分支。
- 快速位置输出：自然中文位置结论，句末保留 E 标签。
- SSE completed 输出：answer、citations、evidence、images、timings。
- A 端使用 evidence_id 建立引用卡片，引用不再因缺证据对象被过滤。

## 安全边界

- 不改变领域过滤和 published_only 限制。
- 不让答案模型重新检索或生成自由 Cypher。
- 快速答案只适用于已经完成节点对齐且有图谱位置证据的问题。
- 普通复杂问题仍走流式答案模型和引用校验。

## 测试

- QA v2 全量测试：94 passed。
- 新增 Stage 26 测试覆盖：快速答案跳过语义校验、SSE completed 透传 evidence、首 token 计时。
- A 端模板 JavaScript：node --check 通过。
- 真实 B 端 SSE：首 token 约 1.3 秒，completed evidence 5 条，citations 正常返回。

## 当前性能结论

本次位置问题的检索约 1.3 秒，其中槽位抽取约 1.15 秒；不再额外等待约 15 秒的答案/引用模型校验。复杂问题的答案模型生成耗时仍需单独优化。