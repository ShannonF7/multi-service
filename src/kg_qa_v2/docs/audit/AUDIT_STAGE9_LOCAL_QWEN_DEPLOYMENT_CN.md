# QA v2 阶段 9：本地 Qwen 决策模型部署与接入审计

## 一、变更范围

本阶段只修改和新增 `src/kg_qa_v2` 内的 QA v2 文件，以及服务器上的独立本地推理环境；未修改主 `app.py`、旧问答链、向量化服务和重排服务。

模型目录已从：

```text
/home/zhangbi/Zhangbi_Traveler/LLM_model/Model_api/checkpoints/qa_qwen2_5_7b_awq
```

迁移到：

```text
/home/zhangbi/Zhangbi_Traveler/multimodal/qa_qwen2_5_7b_awq
```

## 二、模型与服务

- 模型：Qwen2.5-7B-Instruct-AWQ
- vLLM：0.6.4.post1
- Torch：2.5.1+cu124
- Transformers：4.46.3
- 服务地址：`http://127.0.0.1:7020/v1`
- 服务模型名：`Qwen2.5-7B-Instruct-AWQ`
- 上下文上限：2048
- JSON 约束后端：lm-format-enforcer
- 使用 GPU：GPU 1
- 服务只负责决策模型调用，不访问图谱数据库，不替代 Graph/Text/Image Retriever。

## 三、输入与输出审计

### 1. Router

输入包括：

- 当前问题
- 是否带图像
- 目标模态
- 当前领域
- 当前节点和定位节点
- 最近实体、待确认候选和会话摘要

输出限制为：

- intent
- route
- confidence
- needs_graph
- needs_text
- needs_image
- needs_path
- context_dependent
- needs_current_location
- needs_clarification

Router 不能创造实体，也不能直接执行检索。

### 2. Entity Alignment

输入包括问题、提及名称、候选列表和上下文位置。

输出只能选择候选列表中的 `selected_node_id`，或者返回 `UNKNOWN`；模型不能创建候选，不能跨领域选取对象。

### 3. Citation Validation

输入为答案句和证据集合。

输出为：

- ENTAILED
- PARTIAL
- NOT_ENTAILED
- confidence
- reason

答案模型和引用校验模型均不直接访问数据库。

### 4. 闲聊保护

当 route 为 `CASUAL` 时，归一化层强制关闭：

- 图谱检索
- 文本检索
- 图片检索
- 路径检索
- 当前位置读取
- 澄清流程
- 上下文依赖

这样即使模型误返回 `needs_text=true`，后续主链也不会进入检索。

## 四、测试结果

- QA v2 单元测试：37 passed
- Python compileall：通过
- vLLM `/v1/models`：返回正确模型
- 真实 Router：
  - “校长办公室在哪里？” → LOCATION / KNOWLEDGE
  - “你好呀” → CASUAL
- 真实实体对齐：
  - 当前在“行政楼·B座·二层”时，“财务部”选择二层候选
- 真实引用校验：
  - 证据与答案句一致时返回 ENTAILED
- 真实 `QuestionUnderstandingService.from_env()`：
  - 闲聊输出 route=CASUAL
  - needs_graph=false
  - needs_text=false

## 五、已知限制

1. 当前只完成了本地决策模型，即 Router、Entity Alignment、Citation Validation。
2. 槽位抽取仍沿用已有 Qwen Structured Slot Client；没有 API 配置时使用规则降级。
3. vLLM 使用本机回环地址，主应用必须运行在同一台服务器上，或者由部署侧增加安全代理。
4. GPU 1 目前还有其他服务，因此显存参数固定为 0.80、上下文 2048；后续并发压力测试前不能随意增大上下文。
5. 本阶段还没有把 QA v2 主链接入主 `app.py`，也没有改 A 端页面。

## 六、审计结论

本阶段本地 Qwen 已从“模型文件存在”推进到“真实 vLLM 服务可用、决策客户端可调用、QA v2 理解入口可调用、闲聊不触发检索”。输入边界、输出结构、失败降级和服务隔离均已验证，可以进入下一阶段的 QA v2 主链检索编排与 SSE 输出接入。
