# QA v2 阶段 14 审计：生产编排门面

## 1. 本阶段目标

在新目录内把已经验证的各层组装成一个接口层可调用的服务：

- QAV2Service.ask：同步问答；
- QAV2Service.stream_ask：SSE 事件流；
- QAV2Service.from_env：按当前部署构造真实端口。

## 2. from_env 的真实组装

from_env 会构造：

1. DecisionModel：
   - 当前环境配置 local 时使用 7020 Qwen2.5-7B-Instruct-AWQ；
   - 若环境配置 hosted_jev，则使用已有 Jev 适配器。
2. QuestionUnderstandingService：
   - Router 使用 DecisionModel；
   - 槽位抽取使用已有 Qwen API，未配置密钥时保守降级。
3. EntityAlignmentService：
   - 使用同一个 DecisionModel 在候选集合内对齐；
   - 禁止创造候选。
4. PublishedGraphGateway：
   - 候选搜索和图谱证据检索都复用现有已发布图谱服务。
5. DomainKBTextRetriever：
   - 复用现有领域混合文本检索。
6. DomainKBImageRetriever：
   - 复用现有 Qwen-VL 图片向量检索。
7. AnswerGenerator/CitationValidator：
   - 由本地 Qwen 生成答案；
   - 可由 DecisionModel 做引用支持性校验。
8. LocalStreamingAnswerClient：
   - 复用 7020 的 OpenAI-compatible stream 接口。

## 3. 同步输入输出审计

输入包括：

- question；
- domain_id；
- session_id/user_id；
- current_location；
- has_image/image_path；
- target_modality。

输出 QAResponse 包含：

- status；
- answer；
- citations；
- images；
- evidence；
- warnings；
- process；
- elapsed_ms；
- route/intent。

其中 evidence 和 citations 可用于 A 端来源悬浮层，images 限制最多 3 条，避免把全部资产塞进答案。

## 4. SSE 输入输出审计

stream_ask 顺序：

1. 立即发送 started；
2. 执行 QA v2 主链；
3. 发送公开 process 阶段；
4. 发送 token；
5. completed 发送最终答案、引用、图片、warning 和总耗时；
6. 异常时发送 error。

不发送模型私有思维链，只发送可审计的阶段状态。

## 5. 位置与图片处理

- image_path 会写入 QuerySpec.constraints.image_path；
- 图片检索适配器仅在文件存在时调用 7012 相关检索；
- 图片答案最多返回 3 条 Evidence；
- current_location 仍通过 ContextResolver 进入问题理解和实体对齐。

## 6. 测试结果

执行：

- python -m pytest -q src/kg_qa_v2
- python -m compileall -q src/kg_qa_v2

结果：

- 51 passed；
- 编译检查通过。

测试覆盖：

- 同步答案输出；
- SSE 首事件立即发送；
- token 顺序；
- 总耗时；
- 引用与 Evidence 映射；
- Graph/Text/Image 端口组装契约；
- 图片路径安全边界。

## 7. 当前边界

- 主 app.py 尚未修改；
- 尚未添加 FastAPI 路由；
- 尚未接 A 端页面；
- session 的持久化写回和上下文摘要仍由已有 memory 存储层后续接入；
- pipeline 当前 process 是阶段完成后的审计摘要，尚未对每一个内部函数提供实时 callback。

## 8. 审计结论

本阶段通过。QA v2 的独立生产编排层已经完成，下一步才是把它安全接入主 app.py，并用真实领域、真实 session 和真实前端做联调。
