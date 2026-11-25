# QA v2 第六阶段中文代码审计：真实 Jev/Qwen 客户端

## 实现范围

本阶段在新目录中加入 TypeSafe 官方 Python SDK 的 Jev 适配器和 DashScope Qwen Structured Output 客户端，同时修正 intent 的唯一归属。

## 输入输出

### Jev

- 输入：问题、模态、当前领域、当前位置节点、最近实体、待确认候选和摘要。
- 请求：通过官方 `typesafe-sdk` 的 `TypeSafeClient.system_one`，一次发送一个 state 和多个命名 typed questions。
- 输出：归一化的 intent、路由标记、置信度和 usage。

### Qwen

- 输入：Jev 已经确定的 intent、有限上下文和问题。
- 输出：只允许 targets、attribute、start、end、time_range、comparison_targets、constraints、scope、question_rewrite。

## 逻辑审计

1. Jev 是唯一的 intent 决策者。
2. Qwen 返回的 intent、route、检索开关会被忽略，不能覆盖 Jev 结果。
3. Jev 客户端一次请求完成多个 typed questions，不为每个开关单独调用接口。
4. Jev 只能返回预定义 intent，未知值降级为 UNKNOWN。
5. Noul 概率当前使用 0.70 作为开发默认阈值；正式上线前必须用 Gold Set 校准。
6. Qwen 使用 DashScope `json_object` 模式，并仍通过 Pydantic/程序字段白名单校验。
7. API Key 只从服务端环境变量读取，不写入代码、Prompt、日志或测试输出。

## 配置

```text
JEV_API_KEY
# 官方 SDK 也支持 TYPESAFE_API_KEY；项目适配器优先读取 JEV_API_KEY
JEV_BASE_URL=https://api.typesafe.ai
JEV_MODEL=jev-latest
JEV_TIMEOUT=8
DASHSCOPE_API_KEY 或 LLM_API_KEY
KG_QA_QWEN_SLOT_MODEL=qwen-flash
```

官方 SDK 依赖声明位于本目录 `requirements.txt`，不会被现有主链自动安装。

## 尚未实现

- 真实密钥下的线上 smoke test。
- Gold Set 阈值校准。
- 生产环境重试、熔断和限流策略。
