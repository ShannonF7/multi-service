# 阶段 7 决策模型接口审计

## 1. 输入

`DecisionModel.decide(task, payload)` 接收已经由业务代码整理过的任务名和 JSON 对象。路由任务包含问题、领域、当前位置、最近实体和待确认候选；实体对齐任务只接收有限候选；引用校验任务预留答案句与证据输入。

## 2. 输出

本地路由输出被规范为 `intent / route / confidence / needs_*`，并限制在既有意图枚举内。实体对齐仍只消费 `selected_node_id / confidence / reason`，不会接受模型新造的节点。

## 3. 逻辑边界

决策模型不访问图谱、不执行 Cypher、不做候选召回、不生成最终答案。`LocalDecisionClient` 只负责调用本地 OpenAI-compatible 服务和解析 JSON；`JevRouter`、实体对齐服务继续负责业务门槛与安全降级。

## 4. 降级与风险

未配置本地服务时不会自动发起网络请求；`auto` 模式先找本地配置，再尝试托管 Jev，最终由现有规则回退。模型输出非法 JSON、意图越界或请求异常时，路由回退确定性规则。当前仅完成本地适配器和接口验证，没有下载或实际加载 Qwen 权重。

## 5. 测试结果

阶段 7 覆盖：OpenAI-compatible 请求体、路由输出规范化、接入 `QuestionUnderstandingService`、环境变量选择本地模型，以及进程内 Transformers 客户端的懒加载测试。测试命令及结果记录在交付说明中。

实际结果：项目 Python 3.10 环境运行 `pytest -q src/kg_qa_v2/tests`，26 passed；随后运行 `python -m compileall -q src/kg_qa_v2` 通过。另验证了本地服务异常回退、候选外 node_id 拒绝和旧 `src.kg_qa.service` 导入。

## 6. 当前未完成项

服务器当前未安装 vLLM，也没有在本阶段下载 Qwen 权重，因此尚未进行真实 GPU 推理压测。这个文件夹已经具备本地服务接入能力，但“本地模型在线”仍需要部署侧提供兼容接口。
